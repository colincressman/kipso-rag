"""
Personal AI Server — Launcher GUI
===================================
Run with:  python launcher/app.py
           python -m launcher.app

Layout
------
┌─────────────────────────────────────────────────────────────┐
│  Personal AI Server                              ● Running   │
├──────────────┬──────────────────────────────────────────────┤
│  [▶ Start]   │  STATUS                                       │
│  [■ Stop]    │  Server:    ● Running  (localhost:8000)       │
│  [⊕ Browser] │  Ollama:    ● Connected                      │
│              │  Satellite: ✓ Online  (10.0.0.5:8100)        │
│  ─────────── │  DB:        24,341 chunks · 39 docs           │
│  Models      │  VRAM:      6.2 GB / 8.0 GB                  │
│  Retrieval   │                                               │
│  Satellite   │  CONSOLE ──────────────────────────────────── │
│  Database    │  INFO: Server started on 0.0.0.0:8000        │
│  Advanced    │  ...                                          │
└──────────────┴──────────────────────────────────────────────┘
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import messagebox, ttk

# ── Optional sv-ttk theme ────────────────────────────────────────────────────
try:
    import sv_ttk  # type: ignore
    _HAS_SV_TTK = True
except ImportError:
    _HAS_SV_TTK = False

# ── Optional yaml ─────────────────────────────────────────────────────────────
try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ── Project root ──────────────────────────────────────────────────────────────
# In a frozen (Nuitka/PyInstaller) build, sys.executable is the rag.exe itself
# and the project files are beside it — so PROJECT_ROOT = exe's directory.
# In development, PROJECT_ROOT is two levels up from launcher/app.py.
def _resolve_project_root() -> Path:
    from utils.frozen import is_frozen  # noqa: PLC0415
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent

PROJECT_ROOT = _resolve_project_root()
VENV_PYTHON  = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
MAIN_PY      = PROJECT_ROOT / "main.py"
CONFIGS_DIR  = PROJECT_ROOT / "configs"

SERVER_HOST  = "0.0.0.0"
SERVER_PORT  = 8000
STATUS_URL   = f"http://localhost:{SERVER_PORT}/api/status"
POLL_INTERVAL_MS  = 5_000   # server status poll every 5 s
OLLAMA_POLL_MS    = 3_000   # direct Ollama VRAM poll every 3 s
LOG_MAX_LINES     = 2_000   # trim console after this many lines

# ── Colours (work in both light and dark mode) ────────────────────────────────
COL_GREEN   = "#22c55e"
COL_RED     = "#ef4444"
COL_YELLOW  = "#eab308"
COL_BLUE    = "#3b82f6"
COL_GRAY    = "#6b7280"
COL_BG_DARK = "#1e1e2e"
COL_FG      = "#cdd6f4"

TAG_INFO    = "info"
TAG_WARN    = "warn"
TAG_ERROR   = "err"
TAG_DEBUG   = "dbg"


class LauncherApp(tk.Tk):
    # ── Construction ──────────────────────────────────────────────────────────
    def __init__(self) -> None:
        # Enable per-monitor DPI awareness before the window is realised so
        # Tkinter picks up the correct scale factor on HiDPI / 4K displays.
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
            except Exception:
                pass
        super().__init__()
        # Sync Tkinter's scaling to the actual monitor DPI so fonts and widgets
        # are the right physical size on HiDPI displays.
        if sys.platform == "win32":
            try:
                import ctypes
                dpi = ctypes.windll.user32.GetDpiForWindow(self.winfo_id())
                if dpi and dpi != 96:
                    self.tk.call("tk", "scaling", dpi / 72.0)
            except Exception:
                pass
        self.title("Personal AI Server")
        self.geometry("960x620")
        self.minsize(780, 480)
        self._apply_theme()

        # Set window icon (taskbar + title bar)
        try:
            from utils.frozen import get_bundle_dir  # noqa: PLC0415
            _icon = get_bundle_dir() / "installer" / "assets" / "icon.ico"
            if _icon.exists():
                self.iconbitmap(str(_icon))
        except Exception:
            pass

        # State
        self._proc:   Optional[subprocess.Popen] = None  # type: ignore[type-arg]
        self._log_q:  queue.Queue[str] = queue.Queue()
        self._stop_reader = threading.Event()
        self._ollama_direct_ok: bool = False  # True when direct /api/ps poll succeeds

        # Build UI
        self._build_titlebar()
        self._build_main()
        self._build_status_frame()
        self._build_console()
        self._build_config_tabs()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._update_server_buttons()
        self._schedule_status_poll()
        self._schedule_ollama_poll()
        self._drain_log_queue()

        # Auto-start server if configured
        if self._read_yaml_key("runtime.yaml", "launcher.autostart_server") is True:
            self.after(500, self._start_server)

    # ── Theme ─────────────────────────────────────────────────────────────────
    def _apply_theme(self) -> None:
        if _HAS_SV_TTK:
            sv_ttk.set_theme("dark")
        else:
            style = ttk.Style(self)
            try:
                style.theme_use("clam")
            except Exception:
                pass

    # ── Title bar ─────────────────────────────────────────────────────────────
    def _build_titlebar(self) -> None:
        bar = ttk.Frame(self, padding=(12, 6))
        bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(bar, text="Personal AI Server", font=("Segoe UI", 13, "bold")).pack(
            side=tk.LEFT
        )

        self._status_dot = tk.Label(bar, text="● Stopped", fg=COL_RED, bg=self._bg(),
                                    font=("Segoe UI", 10))
        self._status_dot.pack(side=tk.RIGHT, padx=6)

    # ── Main two-column layout ────────────────────────────────────────────────
    def _build_main(self) -> None:
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Left sidebar (fixed width)
        self._sidebar = ttk.Frame(outer, width=190, padding=(8, 4))
        self._sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        self._sidebar.pack_propagate(False)

        # Right content area
        self._content = ttk.Frame(outer)
        self._content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_sidebar()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self) -> None:
        sb = self._sidebar

        # Server control buttons
        self._btn_start = ttk.Button(sb, text="▶  Start", command=self._start_server)
        self._btn_start.pack(fill=tk.X, pady=(4, 2))

        self._btn_stop = ttk.Button(sb, text="■  Stop", command=self._stop_server)
        self._btn_stop.pack(fill=tk.X, pady=2)

        self._btn_browser = ttk.Button(sb, text="⊕  Open UI", command=self._open_browser)
        self._btn_browser.pack(fill=tk.X, pady=2)

        ttk.Separator(sb, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        ttk.Label(sb, text="DOCUMENTS", font=("Segoe UI", 8), foreground=COL_GRAY).pack(
            anchor=tk.W
        )
        self._btn_open_raw = ttk.Button(sb, text="📂  Open Raw Folder", command=self._open_raw_folder)
        self._btn_open_raw.pack(fill=tk.X, pady=2)

        self._btn_ingest = ttk.Button(sb, text="⬆  Ingest New Files", command=self._ingest_raw)
        self._btn_ingest.pack(fill=tk.X, pady=2)

        ttk.Separator(sb, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(sb, text="SETTINGS", font=("Segoe UI", 8), foreground=COL_GRAY).pack(
            anchor=tk.W
        )

        # Nav buttons that switch the right-side config tab
        self._nav_btns: list[ttk.Button] = []
        nav_items = [
            ("Models",     "models"),
            ("Retrieval",  "retrieval"),
            ("Satellite",  "satellite"),
            ("Database",   "database"),
            ("Advanced",   "advanced"),
        ]
        for label, tab_id in nav_items:
            btn = ttk.Button(
                sb, text=label,
                command=lambda t=tab_id: self._switch_tab(t)
            )
            btn.pack(fill=tk.X, pady=1)
            self._nav_btns.append(btn)

    # ── Right side: status + console stacked ──────────────────────────────────
    def _build_status_frame(self) -> None:
        frm = ttk.LabelFrame(self._content, text=" STATUS ", padding=(12, 8))
        frm.pack(fill=tk.X, pady=(0, 6))

        # Status grid: 3 rows × 2 cols
        grid = ttk.Frame(frm)
        grid.pack(fill=tk.X)
        grid.columnconfigure(1, weight=1)
        grid.columnconfigure(3, weight=1)

        def _row(parent, r, col_off, label_text):
            ttk.Label(parent, text=label_text, foreground=COL_GRAY,
                      font=("Segoe UI", 9)).grid(row=r, column=col_off, sticky=tk.W, padx=(0, 4))
            var = tk.StringVar(value="—")
            lbl = tk.Label(parent, textvariable=var, font=("Segoe UI", 9),
                           bg=self._bg(), anchor=tk.W)
            lbl.grid(row=r, column=col_off + 1, sticky=tk.W, padx=(0, 20))
            return var, lbl

        self._sv_server, self._lbl_server       = _row(grid, 0, 0, "Server:")
        self._sv_ollama, self._lbl_ollama        = _row(grid, 1, 0, "Ollama:")
        self._sv_satellite, self._lbl_satellite  = _row(grid, 2, 0, "Satellite:")
        self._sv_db, _                           = _row(grid, 0, 2, "DB:")
        self._sv_vram, _                         = _row(grid, 1, 2, "VRAM:")
        self._sv_model, _                        = _row(grid, 2, 2, "Model:")

    def _build_console(self) -> None:
        # Shared stacking container — both the console and the config notebook
        # live at the same grid cell; use .tkraise() to show one or the other.
        self._stack = ttk.Frame(self._content)
        self._stack.pack(fill=tk.BOTH, expand=True)
        self._stack.rowconfigure(0, weight=1)
        self._stack.columnconfigure(0, weight=1)

        frm = ttk.LabelFrame(self._stack, text=" CONSOLE ", padding=(4, 4))
        frm.grid(row=0, column=0, sticky="nsew")
        self._console_frm = frm

        # Text widget with scrollbar
        self._console = tk.Text(
            frm, wrap=tk.WORD, state=tk.DISABLED,
            font=("Consolas", 9), relief=tk.FLAT,
            bg="#0f0f17", fg=COL_FG, insertbackground=COL_FG,
            selectbackground="#3b3b5c",
        )
        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=self._console.yview)
        self._console.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._console.pack(fill=tk.BOTH, expand=True)

        # Color tags
        self._console.tag_configure(TAG_INFO,  foreground="#a6e3a1")
        self._console.tag_configure(TAG_WARN,  foreground="#f9e2af")
        self._console.tag_configure(TAG_ERROR, foreground="#f38ba8")
        self._console.tag_configure(TAG_DEBUG, foreground="#89b4fa")

    # ── Config tabs (hidden notebook, shown via sidebar nav) ──────────────────
    def _build_config_tabs(self) -> None:
        # Notebook lives in the same stacking container as the console frame.
        # _stack is created in _build_console() which is called first.
        self._tab_notebook = ttk.Notebook(self._stack)
        self._tab_notebook.grid(row=0, column=0, sticky="nsew")
        self._tab_frames: dict[str, ttk.Frame] = {}

        tabs = {
            "models":    self._build_tab_models,
            "retrieval": self._build_tab_retrieval,
            "satellite": self._build_tab_satellite,
            "database":  self._build_tab_database,
            "advanced":  self._build_tab_advanced,
        }
        for tab_id, builder in tabs.items():
            frame = ttk.Frame(self._tab_notebook, padding=12)
            self._tab_notebook.add(frame, text=tab_id.capitalize())
            self._tab_frames[tab_id] = frame
            builder(frame)

        # Start with console on top
        self._console_frm.tkraise()

    def _switch_tab(self, tab_id: str) -> None:
        """Show the config notebook with the requested tab selected."""
        idx = list(self._tab_frames.keys()).index(tab_id)
        self._tab_notebook.select(idx)
        self._tab_notebook.tkraise()

    def _show_console(self) -> None:
        """Switch back to console view."""
        self._console_frm.tkraise()

    # ── Tab: Models ───────────────────────────────────────────────────────────
    def _build_tab_models(self, parent: ttk.Frame) -> None:
        self._model_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("LLM model",     "llm.yaml", "llm.model"),
            ("LLM base_url",  "llm.yaml", "llm.base_url"),
            ("LLM timeout",   "llm.yaml", "llm.timeout_seconds"),
            ("Temperature",   "llm.yaml", "llm.temperature"),
            ("Embed model",   "runtime.yaml", "embedding.model_name"),
            ("Embed backend", "runtime.yaml", "embedding.backend"),
            ("Embed dim",     "runtime.yaml", "embedding.dimension"),
            ("Ollama URL",    "runtime.yaml", "embedding.ollama_base_url"),
        ]
        self._build_yaml_form(parent, fields, self._model_vars, "models")

    # ── Tab: Retrieval ────────────────────────────────────────────────────────
    def _build_tab_retrieval(self, parent: ttk.Frame) -> None:
        self._retrieval_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("Top-K",          "runtime.yaml", "retrieval.top_k"),
            ("Rerank pool",    "runtime.yaml", "retrieval.rerank_candidate_k"),
            ("Vector weight",  "runtime.yaml", "retrieval.alpha_vector"),
            ("Lexical weight", "runtime.yaml", "retrieval.alpha_lexical"),
            ("Internet fallback", "runtime.yaml", "retrieval.internet_fallback_enabled"),
            ("CrossEncoder model", "runtime.yaml", "retrieval.cross_encoder_model"),
            ("HyDE enabled",   "runtime.yaml", "hyde.enabled"),
            ("HyDE model",     "runtime.yaml", "hyde.model"),
            ("HyDE base_url",  "runtime.yaml", "hyde.base_url"),
        ]
        self._build_yaml_form(parent, fields, self._retrieval_vars, "retrieval")

    # ── Tab: Satellite ────────────────────────────────────────────────────────
    def _build_tab_satellite(self, parent: ttk.Frame) -> None:
        self._satellite_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("Inference URL",     "runtime.yaml", "inference_service.url"),
            ("NLI model",         "runtime.yaml", "inference_service.nli_model"),
            ("CrossEncoder model","runtime.yaml", "inference_service.ce_model"),
            ("HyDE base_url",     "runtime.yaml", "hyde.base_url"),
            ("HyDE model",        "runtime.yaml", "hyde.model"),
            ("HyDE timeout (s)",  "runtime.yaml", "hyde.timeout_seconds"),
            ("HyDE temperature",  "runtime.yaml", "hyde.temperature"),
        ]
        self._build_yaml_form(parent, fields, self._satellite_vars, "satellite")

    # ── Tab: Database ─────────────────────────────────────────────────────────
    def _build_tab_database(self, parent: ttk.Frame) -> None:
        self._database_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("DB DSN", "runtime.yaml", "paths.db_dsn"),
        ]
        self._build_yaml_form(parent, fields, self._database_vars, "database")

    # ── Tab: Advanced ─────────────────────────────────────────────────────────
    def _build_tab_advanced(self, parent: ttk.Frame) -> None:
        self._advanced_vars: dict[str, tk.StringVar] = {}
        fields = [
            ("Auto-start server",  "runtime.yaml", "launcher.autostart_server"),
            ("Min coverage",      "llm.yaml", "decision.min_coverage_score"),
            ("Med confidence",    "llm.yaml", "decision.medium_confidence_score"),
            ("High confidence",   "llm.yaml", "decision.high_confidence_score"),
            ("Max chunks",        "llm.yaml", "prompt.max_chunks"),
            ("Max chars/chunk",   "llm.yaml", "prompt.max_chars_per_chunk"),
            ("Two-stage alpha",   "runtime.yaml", "two_stage.alpha"),
            ("Chunk max tokens",  "runtime.yaml", "chunking.max_tokens"),
        ]
        self._build_yaml_form(parent, fields, self._advanced_vars, "advanced")

    # ── Generic YAML form builder ─────────────────────────────────────────────
    def _build_yaml_form(
        self,
        parent: ttk.Frame,
        fields: list,
        var_store: dict,
        tab_id: str,
    ) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0, bg=self._bg())
        vsb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        inner = ttk.Frame(canvas)
        win = canvas.create_window((0, 0), window=inner, anchor=tk.NW)

        def _on_resize(event):
            canvas.itemconfigure(win, width=event.width)
        canvas.bind("<Configure>", _on_resize)

        def _on_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_frame)

        inner.columnconfigure(1, weight=1)

        for row, (label, config_file, yaml_path) in enumerate(fields):
            val = self._read_yaml_key(config_file, yaml_path)
            var = tk.StringVar(value=str(val) if val is not None else "")
            var_store[yaml_path] = var

            ttk.Label(inner, text=label, font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3
            )
            entry = ttk.Entry(inner, textvariable=var, font=("Consolas", 9))
            entry.grid(row=row, column=1, sticky=tk.EW, pady=3)
            # Store metadata on var for save
            var._config_file = config_file   # type: ignore[attr-defined]
            var._yaml_path   = yaml_path     # type: ignore[attr-defined]

        # Save / Back buttons
        btn_row = len(fields)
        bf = ttk.Frame(inner)
        bf.grid(row=btn_row, column=0, columnspan=2, sticky=tk.W, pady=(12, 0))
        ttk.Button(bf, text="💾  Save", command=lambda: self._save_tab(var_store)).pack(
            side=tk.LEFT, padx=(0, 6)
        )
        ttk.Button(bf, text="← Console", command=self._show_console).pack(side=tk.LEFT)

    # ── YAML helpers ──────────────────────────────────────────────────────────
    def _read_yaml_key(self, config_file: str, dotted_key: str):
        if not _HAS_YAML:
            return ""
        path = CONFIGS_DIR / config_file
        if not path.exists():
            return ""
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            for part in dotted_key.split("."):
                data = data[part]
            return data
        except Exception:
            return ""

    def _write_yaml_key(self, config_file: str, dotted_key: str, value: str) -> None:
        if not _HAS_YAML:
            return
        path = CONFIGS_DIR / config_file
        data = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        keys = dotted_key.split(".")
        node = data
        for part in keys[:-1]:
            node = node.setdefault(part, {})
        # Validate and coerce to correct type
        final_key = keys[-1]
        existing = node.get(final_key, "")
        try:
            if isinstance(existing, bool) or value.lower() in ("true", "false"):
                node[final_key] = value.lower() == "true"
            elif isinstance(existing, int):
                coerced = int(value)
                if coerced < 0:
                    raise ValueError(f"'{dotted_key}' must be a non-negative integer, got: {value!r}")
                node[final_key] = coerced
            elif isinstance(existing, float):
                node[final_key] = float(value)
            else:
                node[final_key] = value
        except (ValueError, AttributeError) as exc:
            raise ValueError(str(exc)) from exc
        path.write_text(yaml.dump(data, allow_unicode=True, default_flow_style=False),
                        encoding="utf-8")

    def _save_tab(self, var_store: dict) -> None:
        errors = []
        for yaml_path, var in var_store.items():
            cfg_file = getattr(var, "_config_file", None)
            if cfg_file:
                raw = var.get()
                try:
                    self._write_yaml_key(cfg_file, yaml_path, raw)
                except ValueError as exc:
                    errors.append(f"• {yaml_path}: {exc}")
                except Exception as exc:
                    errors.append(f"• {yaml_path}: unexpected error — {exc}")
        if errors:
            messagebox.showerror(
                "Validation Error",
                "The following fields have invalid values and were NOT saved:\n\n"
                + "\n".join(errors),
            )
        else:
            messagebox.showinfo("Saved", "Configuration saved.\nRestart the server to apply changes.")

    # ── Server lifecycle ──────────────────────────────────────────────────────
    def _start_server(self) -> None:
        if self._proc and self._proc.poll() is None:
            self._log("[launcher] Server is already running.")
            return

        # When running as a frozen EXE (Nuitka standalone or PyInstaller),
        # sys.executable IS the rag.exe itself.  Pass "serve" so it starts
        # the server without opening the GUI again.
        # In development, use the venv python + main.py.
        from utils.frozen import is_frozen, get_main_exe  # noqa: PLC0415
        frozen = is_frozen()
        if frozen:
            cmd = [str(get_main_exe()), "serve",
                   "--host", SERVER_HOST, "--port", str(SERVER_PORT)]
        else:
            python = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
            cmd = [python, str(MAIN_PY), "serve",
                   "--host", SERVER_HOST, "--port", str(SERVER_PORT)]

        self._log(f"[launcher] Starting server: {' '.join(cmd)}")
        self._stop_reader.clear()
        # On Windows the frozen GUI exe may have invalid console handles;
        # stdin=DEVNULL prevents the pipe setup from touching STD_INPUT_HANDLE,
        # and CREATE_NO_WINDOW keeps the server process hidden.
        _extra: dict = {}
        if sys.platform == "win32":
            _extra["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
                **_extra,
            )
        except Exception as exc:
            self._log(f"[launcher] ERROR: Failed to start server: {exc}", TAG_ERROR)
            return

        # Daemon thread to read stdout → queue
        t = threading.Thread(target=self._read_proc_output, daemon=True)
        t.start()

        self._update_server_buttons()

    def _stop_server(self) -> None:
        if not self._proc or self._proc.poll() is not None:
            self._log("[launcher] Server is not running.")
            return

        # Warn the user if ingest is currently in progress.
        if self._btn_ingest.cget("text") == "⬆  Ingesting…":
            if not messagebox.askyesno(
                "Ingest in Progress",
                "An ingest job is currently running.\n\n"
                "Stopping the server now will abort it — any partially-processed "
                "files will need to be re-ingested.\n\n"
                "Stop the server anyway?",
                default=messagebox.NO,
            ):
                return

        self._log("[launcher] Sending SIGTERM to server…")
        self._stop_reader.set()
        try:
            self._proc.terminate()
        except Exception:
            pass

        # Wait up to 8s for graceful shutdown
        def _wait():
            try:
                self._proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._log("[launcher] Server killed (did not exit in time).", TAG_WARN)
            self.after(0, self._update_server_buttons)
            self.after(0, lambda: self._log("[launcher] Server stopped."))

        threading.Thread(target=_wait, daemon=True).start()
        self._update_server_buttons()

    def _read_proc_output(self) -> None:
        """Read server stdout line-by-line and push into the log queue."""
        assert self._proc is not None
        try:
            for line in self._proc.stdout:  # type: ignore[union-attr]
                if self._stop_reader.is_set():
                    break
                self._log_q.put(line.rstrip())
        except Exception:
            pass
        finally:
            self.after(0, self._update_server_buttons)

    # ── Status polling ────────────────────────────────────────────────────────
    def _schedule_status_poll(self) -> None:
        self._poll_status()
        self.after(POLL_INTERVAL_MS, self._schedule_status_poll)

    def _poll_status(self) -> None:
        threading.Thread(target=self._fetch_status, daemon=True).start()

    # ── Direct Ollama VRAM poll (runs independently of the RAG server) ────────
    def _schedule_ollama_poll(self) -> None:
        threading.Thread(target=self._fetch_ollama_direct, daemon=True).start()
        self.after(OLLAMA_POLL_MS, self._schedule_ollama_poll)

    def _fetch_ollama_direct(self) -> None:
        """Poll Ollama /api/ps directly so VRAM + Model update even when the
        RAG server is stopped or the 5 s server poll hasn't fired yet."""
        ollama_url = (
            self._read_yaml_key("runtime.yaml", "llm.base_url")
            or "http://localhost:11434"
        )
        ps_url  = ollama_url.rstrip("/") + "/api/ps"
        gpu_url = ollama_url.rstrip("/") + "/api/show"   # not used for VRAM
        try:
            with urllib.request.urlopen(ps_url, timeout=3) as resp:
                data = json.loads(resp.read().decode())
            models = data.get("models") or []
            total_vram_mb = sum(
                (m.get("size_vram") or m.get("size") or 0) // (1024 * 1024)
                for m in models
            )
            names = ", ".join(m["name"] for m in models) if models else None
            self._ollama_direct_ok = True
            self.after(0, lambda: self._apply_ollama_direct(total_vram_mb, names))
        except Exception:
            self._ollama_direct_ok = False  # Ollama unreachable — let _status_unreachable clear the fields

    def _apply_ollama_direct(self, used_mb: int, model_name: str | None) -> None:
        """Update VRAM and Model fields from direct Ollama data."""
        if model_name is not None:
            self._sv_model.set(model_name or "—")
        # Update VRAM used — keep total from last server poll if available
        # We store the last known total separately so we can use it here
        total = getattr(self, "_last_total_vram_mb", 0)
        if total and used_mb:
            pct = used_mb / total * 100
            self._sv_vram.set(f"{used_mb/1024:.1f} GB / {total/1024:.1f} GB  ({pct:.0f}%)")
        elif used_mb:
            self._sv_vram.set(f"{used_mb/1024:.1f} GB used")
        # Don’t overwrite with “—” when Ollama returns 0 (model may be loading)

    def _fetch_status(self) -> None:
        try:
            with urllib.request.urlopen(STATUS_URL, timeout=4) as resp:
                data = json.loads(resp.read().decode())
            self.after(0, lambda d=data: self._apply_status(d))
        except Exception:
            self.after(0, self._status_unreachable)

    def _apply_status(self, data: dict) -> None:
        # Server dot
        self._status_dot.config(text="● Running", fg=COL_GREEN)
        self._sv_server.set(f"● Running  (localhost:{SERVER_PORT})")
        self._lbl_server.config(fg=COL_GREEN)

        # Ollama
        models = data.get("ollama_models", [])
        if models:
            names = ", ".join(m["name"] for m in models)
            self._sv_ollama.set(f"● {names}")
            self._lbl_ollama.config(fg=COL_GREEN)
        else:
            self._sv_ollama.set("○ No models loaded")
            self._lbl_ollama.config(fg=COL_YELLOW)

        # DB
        db = data.get("db", {})
        chunks = db.get("chunks", 0)
        docs   = db.get("documents", 0)
        self._sv_db.set(f"{chunks:,} chunks · {docs} docs")

        # VRAM
        total = data.get("total_vram_mb", 0)
        used  = data.get("used_vram_mb", 0)
        if total:
            self._last_total_vram_mb = total  # cache for direct Ollama poll
            pct = used / total * 100
            self._sv_vram.set(f"{used/1024:.1f} GB / {total/1024:.1f} GB  ({pct:.0f}%)")
        else:
            self._sv_vram.set("—")

        # Model (first loaded)
        if models:
            self._sv_model.set(models[0]["name"])
        else:
            self._sv_model.set("—")

        # Satellite — check hyde base_url
        hyde_url = self._read_yaml_key("runtime.yaml", "hyde.base_url")
        if hyde_url and hyde_url != "http://localhost:11434":
            self._sv_satellite.set(f"✓ Configured  ({hyde_url})")
            self._lbl_satellite.config(fg=COL_GREEN)
        else:
            self._sv_satellite.set("— (local)")
            self._lbl_satellite.config(fg=COL_GRAY)

        self._update_server_buttons()

    def _status_unreachable(self) -> None:
        if self._proc and self._proc.poll() is None:
            # Process running but not responding yet
            self._status_dot.config(text="● Starting…", fg=COL_YELLOW)
            self._sv_server.set("● Starting…")
            self._lbl_server.config(fg=COL_YELLOW)
        else:
            self._status_dot.config(text="● Stopped", fg=COL_RED)
            self._sv_server.set("○ Stopped")
            self._lbl_server.config(fg=COL_RED)
        self._sv_ollama.set("—")
        self._sv_db.set("—")
        # Only clear VRAM/Model if the direct Ollama poll is also failing;
        # if Ollama is reachable, the 3-s direct poll owns those two fields.
        if not self._ollama_direct_ok:
            self._sv_vram.set("—")
            self._sv_model.set("—")

    # ── Button state ──────────────────────────────────────────────────────────
    def _update_server_buttons(self) -> None:
        running = self._proc is not None and self._proc.poll() is None
        self._btn_start.config(state=tk.DISABLED if running else tk.NORMAL)
        self._btn_stop.config(state=tk.NORMAL if running else tk.DISABLED)
        self._btn_browser.config(state=tk.NORMAL if running else tk.DISABLED)
        self._btn_ingest.config(state=tk.NORMAL if running else tk.DISABLED)

    def _open_raw_folder(self) -> None:
        import os as _os
        raw_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        _os.startfile(str(raw_dir))

    def _ingest_raw(self) -> None:
        """Ask the server to ingest all new files in data/raw/."""
        import threading, urllib.request, urllib.error, json as _json

        self._btn_ingest.config(state=tk.DISABLED, text="⬆  Ingesting…")
        self._log("Requesting ingest of new files in data/raw/…")

        def _do():
            try:
                url = f"http://localhost:{SERVER_PORT}/api/ingest-raw"
                req = urllib.request.Request(url, method="POST",
                                             headers={"Content-Length": "0"})
                with urllib.request.urlopen(req, data=b"", timeout=10) as resp:
                    result = _json.loads(resp.read())
                status = result.get("status", "")
                if status == "nothing_to_do":
                    self.after(0, lambda: self._log("Ingest: all files already ingested."))
                elif status == "already_running":
                    self.after(0, lambda: self._log("Ingest: already running, check status."))
                    self._poll_ingest_status()
                    return
                elif status == "queued":
                    count = result.get("count", 0)
                    pos   = result.get("queue_position", 1)
                    self.after(0, lambda: self._log(f"Ingest queued — {count} file(s) added (position {pos})."))
                    self._poll_ingest_status()
                    return
                elif status == "started":
                    count = result.get("count", 0)
                    self.after(0, lambda: self._log(f"Ingest started — {count} file(s) queued."))
                    self._poll_ingest_status()
                    return
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="ignore")
                self.after(0, lambda: self._log(f"Ingest error: {exc.code} {body}"))
            except Exception as exc:
                self.after(0, lambda: self._log(f"Ingest error: {exc}"))
            self.after(0, lambda: self._btn_ingest.config(state=tk.NORMAL, text="⬆  Ingest New Files"))

        threading.Thread(target=_do, daemon=True).start()

    def _poll_ingest_status(self) -> None:
        """Poll /api/ingest-raw/status every 3s and update console until done."""
        import urllib.request, json as _json

        try:
            url = f"http://localhost:{SERVER_PORT}/api/ingest-raw/status"
            with urllib.request.urlopen(url, timeout=5) as resp:
                state = _json.loads(resp.read())
            current = state.get("current_file")
            done    = state.get("done_files", [])
            failed  = state.get("failed_files", [])
            running = state.get("running", False)
            if current:
                self._log(f"Ingesting: {current} ({len(done)} done, {len(failed)} failed)")
            if running:
                self.after(3000, self._poll_ingest_status)
                return
            # Finished
            msg = f"Ingest complete — {len(done)} done, {len(failed)} failed."
            if failed:
                msg += f" Failed: {', '.join(failed)}"
            self._log(msg)
        except Exception as exc:
            self._log(f"Ingest status poll error: {exc}")
        self._btn_ingest.config(state=tk.NORMAL, text="⬆  Ingest New Files")

    def _open_browser(self) -> None:
        webbrowser.open(f"http://localhost:{SERVER_PORT}")

    # ── Console log ───────────────────────────────────────────────────────────
    def _drain_log_queue(self) -> None:
        """Pull lines from the queue and write to the Text widget."""
        try:
            while True:
                line = self._log_q.get_nowait()
                self._write_console(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_log_queue)

    def _log(self, message: str, tag: str = TAG_INFO) -> None:
        """Write a launcher message directly (not from subprocess)."""
        self._log_q.put(f"[{_tstamp()}] {message}")

    def _write_console(self, line: str) -> None:
        tag = _classify_line(line)
        self._console.config(state=tk.NORMAL)

        # Trim old lines
        count = int(self._console.index(tk.END).split(".")[0])
        if count > LOG_MAX_LINES:
            self._console.delete("1.0", f"{count - LOG_MAX_LINES}.0")

        self._console.insert(tk.END, line + "\n", tag)
        self._console.see(tk.END)
        self._console.config(state=tk.DISABLED)

    # ── Close ─────────────────────────────────────────────────────────────────
    def _on_close(self) -> None:
        if self._proc and self._proc.poll() is None:
            if not messagebox.askyesno(
                "Server Running",
                "The server is still running.\n\nStop it and exit?",
            ):
                return
            self._stop_reader.set()
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
        self.destroy()

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _bg(self) -> str:
        """Return a background colour that works with the active theme."""
        try:
            return self.cget("bg")
        except Exception:
            return COL_BG_DARK


# ── Utility functions ─────────────────────────────────────────────────────────

def _tstamp() -> str:
    return time.strftime("%H:%M:%S")


def _classify_line(line: str) -> str:
    ll = line.lower()
    if "error" in ll or "exception" in ll or "traceback" in ll or "critical" in ll:
        return TAG_ERROR
    if "warning" in ll or "warn" in ll:
        return TAG_WARN
    if "debug" in ll:
        return TAG_DEBUG
    return TAG_INFO


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # Ensure required directories exist before the server starts or UI renders.
    try:
        from utils.dirs import ensure_runtime_dirs  # noqa: PLC0415
        ensure_runtime_dirs()
    except Exception:
        pass
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
