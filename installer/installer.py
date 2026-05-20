"""
Personal AI Server — GUI Installer
====================================
A standalone Tkinter wizard that sets up everything needed to run the
Personal AI Server on a fresh Windows machine, or to configure a satellite
inference node on a second machine.

Steps
-----
  1. Welcome  (choose: Main Server or Satellite Node)
  2. Check prerequisites
  3. Database  (main server only)
  4. Models  (Ollama models for this machine + HuggingFace models for satellite)
  5. Satellites  (configure URLs for up to 2 remote inference nodes)
  6. Finish  (create shortcuts, write configs)

Build as a standalone EXE with Nuitka:
    nuitka --standalone --onefile --windows-icon-from-ico=installer/assets/icon.ico
           --output-dir=dist_nuitka installer/installer.py

The installer is stdlib-only (no third-party deps at runtime) so it can be
distributed without a venv.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import urllib.request
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

# ── Path helpers ──────────────────────────────────────────────────────────────
def _installer_bundle_root() -> Path:
    """Project root in dev; extraction root in Nuitka onefile build.

    In development installer.py lives at  <project>/installer/installer.py
    so parent.parent = project root.
    In a Nuitka --onefile build the module's __file__ is placed at the
    extraction-dir root, so parent = extraction root (no extra level).
    """
    p = Path(__file__).resolve().parent
    if globals().get("__compiled__"):
        return p  # onefile extraction root — data files are relative to here
    return p.parent  # dev: installer/ sub-dir → step up to project root


_BUNDLE_ROOT = _installer_bundle_root()
_ICON_PATH   = _BUNDLE_ROOT / "installer" / "assets" / "icon.ico"


def _real_installer_dir() -> Path:
    """Returns the directory containing the actual rag_installer.exe.

    In Nuitka --onefile, sys.executable points to the temp extraction dir
    but sys.argv[0] holds the real .exe path as invoked by the OS.
    In dev, falls back to the project root.
    """
    if globals().get("__compiled__"):
        # sys.argv[0] = real path of rag_installer.exe, e.g. dist_nuitka\rag_installer.exe
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parents[1]


def _default_install_dir() -> str:
    """In a frozen installer, rag.exe lives beside the installer in a 'rag' sub-folder.
    In dev, default to ~/PersonalAI as a conventional install location.
    """
    if globals().get("__compiled__"):
        # installer is at dist_nuitka\rag_installer.exe
        # rag.exe is at   dist_nuitka\rag\rag.exe
        return str(_real_installer_dir() / "rag")
    return str(Path.home() / "PersonalAI")

# ── Constants ─────────────────────────────────────────────────────────────────
TITLE         = "Personal AI Server — Installer"
WIN_W, WIN_H  = 700, 700

PG_VERSION    = "16"
PG_INSTALLER_URL = (
    "https://get.enterprisedb.com/postgresql/"
    f"postgresql-{PG_VERSION}.8-1-windows-x64.exe"
)
OLLAMA_URL    = "https://ollama.com/download/OllamaSetup.exe"

DEFAULT_PG_DSN   = "postgresql://postgres:postgres@localhost/rag"
DEFAULT_PG_ADMIN = "postgresql://postgres:postgres@localhost/postgres"
SCHEMA_SQL       = _BUNDLE_ROOT / "db" / "schema.sql"
RUNTIME_YAML     = _BUNDLE_ROOT / "configs" / "runtime.yaml"

# Main-server Ollama models
DEFAULT_EMBED_MODEL = "qwen3-embedding:latest"
DEFAULT_LLM_MODEL   = "qwen3.5:9b"

# Satellite Ollama model (HyDE — lives on inference node)
DEFAULT_HYDE_MODEL  = "qwen3.5:0.8b"

# ── Per-model VRAM estimates (MB) ─────────────────────────────────────────────
# Approximate loaded VRAM based on common quantisations.  Used for display
# hints only — not enforced.  Values are rounded-up typical usage figures.
_VRAM_ESTIMATES_MB: dict[str, int] = {
    # Qwen3 family
    "qwen3:0.6b": 900,    "qwen3:0.6b-q4_K_M": 700,
    "qwen3:1.7b": 2_000,  "qwen3:1.7b-q4_K_M": 1_400,
    "qwen3:4b":   3_500,  "qwen3:4b-q4_K_M":   2_800,
    "qwen3:8b":   7_000,  "qwen3:8b-q4_K_M":   5_600,
    "qwen3:14b": 12_000,  "qwen3:14b-q4_K_M":  9_500,
    "qwen3:32b": 24_000,  "qwen3:32b-q4_K_M": 20_000,
    # Qwen3.5 family
    "qwen3.5:0.8b": 1_000, "qwen3.5:9b": 8_000, "qwen3.5:14b": 12_000,
    # Qwen3 embedding
    "qwen3-embedding:latest": 3_500,
    "qwen3-embedding:0.6b":   1_000,
    "qwen3-embedding:4b":     3_500,
    # LLaMA 3 family
    "llama3:8b": 6_000, "llama3:8b-instruct-q4_0": 5_000,
    "llama3.1:8b": 6_000, "llama3.2:3b": 3_000,
    # Mistral / Mixtral
    "mistral:7b": 5_500, "mistral:7b-instruct": 5_500,
    # Phi family
    "phi3:mini": 2_500, "phi3:medium": 9_000,
    # Gemma
    "gemma:2b": 2_000, "gemma:7b": 6_000,
    # nomic embed
    "nomic-embed-text:latest": 600,
    # custom model aliases (update if you rename)
    "rag-llm": 8_000,
    "hyde-model": 1_000,
    # HuggingFace models (session 79)
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0": 1_400,
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0": 350,
    "cross-encoder/ms-marco-MiniLM-L-6-v2": 90,
    "cross-encoder/ms-marco-TinyBERT-L-2-v2": 17,
}


def _vram_note(model_name: str) -> str:
    """Return a human-readable VRAM estimate string, or '' if unknown."""
    mb = _VRAM_ESTIMATES_MB.get(model_name.strip())
    if mb is None:
        return ""
    return f"~{mb / 1024:.1f} GB VRAM"

# HuggingFace models the inference service downloads automatically on first start
INFERENCE_HF_MODELS = [
    "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",  # intent classifier
    "cross-encoder/ms-marco-MiniLM-L-6-v2",          # reranker
]

# ── Colour palette ────────────────────────────────────────────────────────────
C_BG      = "#1e1e2e"
C_PANEL   = "#181825"
C_FG      = "#cdd6f4"
C_GREEN   = "#a6e3a1"
C_RED     = "#f38ba8"
C_YELLOW  = "#f9e2af"
C_BLUE    = "#89b4fa"
C_GRAY    = "#6c7086"
C_BUTTON  = "#313244"
C_ACCENT  = "#cba6f7"

IS_WINDOWS = sys.platform == "win32"


# ── Wizard steps (pages) ─────────────────────────────────────────────────────

class InstallerApp(tk.Tk):
    STEPS = ["Welcome", "Prerequisites", "Database", "Satellites", "Models", "Finish"]

    def __init__(self) -> None:
        super().__init__()
        self.title(TITLE)
        self.geometry(f"{WIN_W}x{WIN_H}")
        self.resizable(False, False)
        self.configure(bg=C_BG)

        self._step  = 0
        self._log_q: queue.Queue[tuple[str, str]] = queue.Queue()

        # Shared state collected across steps
        self.install_dir  = tk.StringVar(value=_default_install_dir())
        # Role: "main" or "satellite"
        self.node_role = tk.StringVar(value="main")
        self.pg_dsn       = tk.StringVar(value=DEFAULT_PG_DSN)
        self.pg_password  = tk.StringVar(value="postgres")
        self.embed_model  = tk.StringVar(value=DEFAULT_EMBED_MODEL)
        self.llm_model    = tk.StringVar(value=DEFAULT_LLM_MODEL)
        self.hyde_model   = tk.StringVar(value=DEFAULT_HYDE_MODEL)
        # Satellite node URLs (up to 2)
        self.satellite1_url  = tk.StringVar(value="http://10.0.0.5:8100")
        self.satellite1_hyde = tk.StringVar(value="http://10.0.0.5:11434")
        self.satellite2_url  = tk.StringVar(value="")
        self.satellite2_hyde = tk.StringVar(value="")
        # Model placement: "localhost", "satellite1", or "satellite2" — one per model group
        self.llm_location       = tk.StringVar(value="localhost")
        self.embed_location     = tk.StringVar(value="localhost")
        self.hyde_location      = tk.StringVar(value="localhost")
        self.inference_location = tk.StringVar(value="localhost")
        # HuggingFace model selection (written to runtime.yaml on Finish)
        self.nli_model = tk.StringVar(value="MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
        self.ce_model  = tk.StringVar(value="cross-encoder/ms-marco-MiniLM-L-6-v2")
        # VRAM info detected from satellites (populated by _test_satellite)
        self.sat1_vram: str = ""
        self.sat2_vram: str = ""

        # Set window icon (taskbar + title bar)
        try:
            if _ICON_PATH.exists():
                self.iconbitmap(str(_ICON_PATH))
        except Exception:
            pass

        self._build_header()
        self._build_step_indicator()
        self._content_frame = tk.Frame(self, bg=C_BG)
        self._content_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=8)
        self._build_footer()

        self._pages: list[Callable[[], None]] = [
            self._page_welcome,
            self._page_prereqs,
            self._page_database,
            self._page_satellites,
            self._page_models,
            self._page_finish,
        ]
        self._show_step(0)
        self._drain_log()

    # ── Chrome ────────────────────────────────────────────────────────────────
    def _build_header(self) -> None:
        hdr = tk.Frame(self, bg=C_PANEL, height=56)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="Personal AI Server", font=("Segoe UI", 15, "bold"),
                 bg=C_PANEL, fg=C_ACCENT).pack(side=tk.LEFT, padx=20, pady=10)
        tk.Label(hdr, text="Setup Wizard", font=("Segoe UI", 10),
                 bg=C_PANEL, fg=C_GRAY).pack(side=tk.LEFT, pady=14)

    def _build_step_indicator(self) -> None:
        frm = tk.Frame(self, bg=C_BG)
        frm.pack(fill=tk.X, padx=24, pady=(10, 0))
        self._step_labels: list[tk.Label] = []
        for i, name in enumerate(self.STEPS):
            lbl = tk.Label(frm, text=f"{i+1}. {name}", font=("Segoe UI", 9),
                           bg=C_BG, fg=C_GRAY)
            lbl.pack(side=tk.LEFT, padx=6)
            self._step_labels.append(lbl)
            if i < len(self.STEPS) - 1:
                tk.Label(frm, text="›", font=("Segoe UI", 9),
                         bg=C_BG, fg=C_GRAY).pack(side=tk.LEFT)

    def _build_footer(self) -> None:
        frm = tk.Frame(self, bg=C_PANEL, height=48)
        frm.pack(side=tk.BOTTOM, fill=tk.X)
        frm.pack_propagate(False)

        self._btn_next = tk.Button(
            frm, text="Next ›", font=("Segoe UI", 10), bg=C_ACCENT, fg=C_BG,
            relief=tk.FLAT, padx=16, command=self._next_step
        )
        self._btn_next.pack(side=tk.RIGHT, padx=16, pady=10)

        self._btn_back = tk.Button(
            frm, text="‹ Back", font=("Segoe UI", 10), bg=C_BUTTON, fg=C_FG,
            relief=tk.FLAT, padx=16, command=self._prev_step
        )
        self._btn_back.pack(side=tk.RIGHT, padx=4, pady=10)

        self._btn_cancel = tk.Button(
            frm, text="Cancel", font=("Segoe UI", 10), bg=C_BUTTON, fg=C_FG,
            relief=tk.FLAT, padx=12, command=self.destroy
        )
        self._btn_cancel.pack(side=tk.LEFT, padx=16, pady=10)

    # ── Navigation ────────────────────────────────────────────────────────────
    def _show_step(self, step: int) -> None:
        for w in self._content_frame.winfo_children():
            w.destroy()
        self._step = step
        self._update_indicators()
        self._pages[step]()
        self._btn_back.config(state=tk.NORMAL if step > 0 else tk.DISABLED)
        if step == len(self.STEPS) - 1:
            self._btn_next.config(text="Finish", command=self.destroy)
        else:
            self._btn_next.config(text="Next ›", command=self._next_step)

    def _next_step(self) -> None:
        step = self._step + 1
        # Satellite role skips Database (index 2)
        if self.node_role.get() == "satellite" and step == 2:
            step = 3

        # When leaving the Database step (index 2), do a quick connectivity
        # test so the user is warned before trying to apply the schema later.
        if self._step == 2 and self.node_role.get() != "satellite":
            dsn = self.pg_dsn.get().strip()
            try:
                import psycopg  # noqa: PLC0415
                with psycopg.connect(dsn, connect_timeout=3):
                    pass  # connection succeeded
            except Exception as exc:
                proceed = messagebox.askyesno(
                    "Database Not Reachable",
                    f"Could not connect to the database:\n\n{exc}\n\n"
                    "You may not be able to apply the schema later.\n\n"
                    "Continue anyway?",
                )
                if not proceed:
                    return

        if step < len(self.STEPS):
            self._show_step(step)

    def _prev_step(self) -> None:
        step = self._step - 1
        # Satellite role skips Database (index 2)
        if self.node_role.get() == "satellite" and step == 2:
            step = 1
        if step >= 0:
            self._show_step(step)

    def _update_indicators(self) -> None:
        for i, lbl in enumerate(self._step_labels):
            if i < self._step:
                lbl.config(fg=C_GREEN)
            elif i == self._step:
                lbl.config(fg=C_ACCENT, font=("Segoe UI", 9, "bold"))
            else:
                lbl.config(fg=C_GRAY, font=("Segoe UI", 9))

    # ── Page 1: Welcome ───────────────────────────────────────────────────────
    def _page_welcome(self) -> None:
        cf = self._content_frame
        tk.Label(cf, text="Welcome!", font=("Segoe UI", 16, "bold"),
                 bg=C_BG, fg=C_FG).pack(anchor=tk.W, pady=(8, 4))

        # Role selector
        role_frm = tk.Frame(cf, bg=C_BG)
        role_frm.pack(anchor=tk.W, pady=(0, 10))
        tk.Label(role_frm, text="What are you setting up?",
                 font=("Segoe UI", 10, "bold"), bg=C_BG, fg=C_FG).grid(
            row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 6))

        tk.Radiobutton(
            role_frm, text="Main Server  (this machine — runs the RAG server, DB, and web UI)",
            variable=self.node_role, value="main",
            font=("Segoe UI", 10), bg=C_BG, fg=C_FG,
            selectcolor=C_PANEL, activebackground=C_BG, activeforeground=C_FG,
        ).grid(row=1, column=0, sticky=tk.W, pady=2)

        tk.Radiobutton(
            role_frm, text="Satellite Node  (a second machine — runs HuggingFace models + HyDE LLM)",
            variable=self.node_role, value="satellite",
            font=("Segoe UI", 10), bg=C_BG, fg=C_FG,
            selectcolor=C_PANEL, activebackground=C_BG, activeforeground=C_FG,
        ).grid(row=2, column=0, sticky=tk.W, pady=2)

        ttk.Separator(cf).pack(fill=tk.X, pady=8)

        tk.Label(cf, text=(
            "Main Server installs:  PostgreSQL 16 · Ollama · embedding model · LLM\n"
            "Satellite Node installs:  Ollama · HyDE model · HuggingFace inference service\n\n"
            "You need an internet connection and ~10 GB free disk space.\n"
            "Install time: ~5–15 minutes depending on connection speed."
        ), font=("Segoe UI", 10), bg=C_BG, fg=C_FG, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(0, 12)
        )

        ttk.Separator(cf).pack(fill=tk.X, pady=6)
        dir_frm = tk.Frame(cf, bg=C_BG)
        dir_frm.pack(fill=tk.X)
        tk.Label(dir_frm, text="Install location:", font=("Segoe UI", 9),
                 bg=C_BG, fg=C_GRAY).pack(side=tk.LEFT)
        tk.Entry(dir_frm, textvariable=self.install_dir, font=("Consolas", 9),
                 width=44, bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=6)
        tk.Button(dir_frm, text="Browse…", font=("Segoe UI", 9),
                  bg=C_BUTTON, fg=C_FG, relief=tk.FLAT,
                  command=self._browse_dir).pack(side=tk.LEFT)

    def _browse_dir(self) -> None:
        d = filedialog.askdirectory(title="Choose install location")
        if d:
            self.install_dir.set(d)

    # ── Page 2: Prerequisites ─────────────────────────────────────────────────
    def _page_prereqs(self) -> None:
        cf = self._content_frame
        tk.Label(cf, text="Prerequisites", font=("Segoe UI", 14, "bold"),
                 bg=C_BG, fg=C_FG).pack(anchor=tk.W, pady=(4, 8))

        self._prereq_frame = tk.Frame(cf, bg=C_BG)
        self._prereq_frame.pack(fill=tk.X)

        checks = [
            ("Python 3.11+",     self._check_python),
            ("Git",              self._check_git),
            ("Ollama",           self._check_ollama),
            ("PostgreSQL 16",    self._check_postgres),
            ("pgvector",         self._check_pgvector),
        ]

        self._prereq_rows: dict[str, tuple[tk.Label, tk.Label, tk.Button]] = {}
        for i, (name, _) in enumerate(checks):
            row = tk.Frame(self._prereq_frame, bg=C_BG)
            row.pack(fill=tk.X, pady=2)
            icon  = tk.Label(row, text="○", font=("Segoe UI", 12), width=2,
                             bg=C_BG, fg=C_GRAY)
            icon.pack(side=tk.LEFT)
            lbl   = tk.Label(row, text=name, font=("Segoe UI", 10), width=20,
                             bg=C_BG, fg=C_FG, anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            status = tk.Label(row, text="Pending…", font=("Segoe UI", 9),
                              bg=C_BG, fg=C_GRAY)
            status.pack(side=tk.LEFT, padx=8)
            fix_btn = tk.Button(row, text="Install", font=("Segoe UI", 9),
                                bg=C_BUTTON, fg=C_FG, relief=tk.FLAT,
                                state=tk.DISABLED)
            fix_btn.pack(side=tk.RIGHT, padx=4)
            self._prereq_rows[name] = (icon, status, fix_btn)

        tk.Button(cf, text="▶  Run Checks", font=("Segoe UI", 10),
                  bg=C_ACCENT, fg=C_BG, relief=tk.FLAT, padx=12,
                  command=lambda: threading.Thread(
                      target=self._run_prereq_checks, args=(checks,), daemon=True
                  ).start()
                  ).pack(anchor=tk.W, pady=12)

        # Mini log
        self._prereq_log = tk.Text(cf, height=5, font=("Consolas", 8),
                                   bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                                   state=tk.DISABLED)
        self._prereq_log.pack(fill=tk.X)

    def _run_prereq_checks(self, checks: list) -> None:
        for name, fn in checks:
            ok, msg = fn()
            self.after(0, lambda n=name, o=ok, m=msg: self._set_prereq(n, o, m))

    def _set_prereq(self, name: str, ok: bool, msg: str) -> None:
        icon, status, fix_btn = self._prereq_rows[name]
        icon.config(text="●" if ok else "✗",
                    fg=C_GREEN if ok else C_RED)
        status.config(text=msg, fg=C_GREEN if ok else C_RED)
        if ok:
            fix_btn.config(state=tk.DISABLED, text="Already installed")
        else:
            fix_btn.config(state=tk.NORMAL,
                           command=lambda n=name: self._auto_install(n))
        self._prereq_log.config(state=tk.NORMAL)
        self._prereq_log.insert(tk.END, f"{'✓' if ok else '✗'} {name}: {msg}\n")
        self._prereq_log.see(tk.END)
        self._prereq_log.config(state=tk.DISABLED)

    def _auto_install(self, name: str) -> None:
        if name == "Ollama":
            threading.Thread(target=self._install_ollama, daemon=True).start()
        elif name in ("PostgreSQL 16", "pgvector"):
            threading.Thread(target=self._install_postgres, daemon=True).start()
        else:
            messagebox.showinfo("Manual Install Required",
                                f"Please install {name} manually and re-run checks.")

    # ── Prereq checks ─────────────────────────────────────────────────────────
    def _check_python(self) -> tuple[bool, str]:
        v = sys.version_info
        ok = v >= (3, 11)
        return ok, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else " (need 3.11+)")

    def _check_git(self) -> tuple[bool, str]:
        r = shutil.which("git")
        if r:
            try:
                out = subprocess.check_output(["git", "--version"], text=True).strip()
                return True, out.split()[-1]
            except Exception:
                pass
        return False, "Not found — install from git-scm.com"

    def _check_ollama(self) -> tuple[bool, str]:
        r = shutil.which("ollama")
        if r:
            try:
                with urllib.request.urlopen("http://localhost:11434/api/version", timeout=3) as resp:
                    d = __import__("json").loads(resp.read())
                    return True, f"v{d.get('version', '?')} (running)"
            except Exception:
                return True, "installed (not running)"
        return False, "Not found"

    def _check_postgres(self) -> tuple[bool, str]:
        r = shutil.which("psql")
        if not r and IS_WINDOWS:
            for base in [r"C:\Program Files\PostgreSQL"]:
                candidates = list(Path(base).glob(f"{PG_VERSION}*/bin/psql.exe")) if Path(base).exists() else []
                if candidates:
                    r = str(candidates[0])
                    break
        if r:
            try:
                out = subprocess.check_output([r, "--version"], text=True).strip()
                return True, out
            except Exception:
                return True, "installed"
        return False, "Not found"

    def _check_pgvector(self) -> tuple[bool, str]:
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row
            with psycopg.connect(DEFAULT_PG_ADMIN, row_factory=dict_row, connect_timeout=4) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM pg_available_extensions WHERE name='vector'"
                ).fetchone()
                if row and row["n"]:
                    return True, "Available"
                return False, "Extension not installed"
        except Exception as exc:
            return False, f"Cannot connect to PostgreSQL: {exc}"

    # ── Auto-installers ───────────────────────────────────────────────────────
    def _install_ollama(self) -> None:
        if IS_WINDOWS:
            self._emit("Downloading Ollama installer…")
            tmp = Path(tempfile.mkdtemp()) / "OllamaSetup.exe"
            try:
                urllib.request.urlretrieve(OLLAMA_URL, tmp,
                    reporthook=lambda b, bs, total: self._emit(
                        f"  Ollama: {min(b*bs, total)//1024} / {total//1024} KB"
                    ))
                self._emit("Running Ollama installer (silent)…")
                subprocess.run([str(tmp), "/S"], check=True)
                self._emit("Ollama installed.", C_GREEN)
            except Exception as exc:
                self._emit(f"Ollama install failed: {exc}", C_RED)
        else:
            self._emit("Running Ollama install script (requires curl)…")
            try:
                proc = subprocess.Popen(
                    ["bash", "-c", "curl -fsSL https://ollama.com/install.sh | sh"],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                assert proc.stdout
                for line in proc.stdout:
                    line = line.rstrip()
                    if line:
                        self._emit(f"  {line}")
                proc.wait()
                if proc.returncode == 0:
                    self._emit("Ollama installed.", C_GREEN)
                else:
                    self._emit(f"Install script failed (exit {proc.returncode})", C_RED)
            except FileNotFoundError:
                self._emit("curl not found. Install curl then re-run, or visit https://ollama.com", C_RED)
            except Exception as exc:
                self._emit(f"Ollama install failed: {exc}", C_RED)

    def _install_postgres(self) -> None:
        self._emit("Downloading PostgreSQL installer (~200 MB)…")
        tmp = Path(tempfile.mkdtemp()) / "pg_installer.exe"
        try:
            urllib.request.urlretrieve(PG_INSTALLER_URL, tmp,
                reporthook=lambda b, bs, total: self._emit(
                    f"  PostgreSQL: {min(b*bs, total)//1_048_576:.0f} / {total//1_048_576:.0f} MB"
                ))
            pw = self.pg_password.get() or "postgres"
            self._emit("Running PostgreSQL installer (silent)…")
            subprocess.run([
                str(tmp), "--mode", "unattended",
                "--superpassword", pw,
                "--serverport", "5432",
            ], check=True)
            self._emit("PostgreSQL installed.", C_GREEN)
        except Exception as exc:
            self._emit(f"PostgreSQL install failed: {exc}", C_RED)

    # ── Page 3: Database ──────────────────────────────────────────────────────
    def _page_database(self) -> None:
        cf = self._content_frame
        tk.Label(cf, text="Database Setup", font=("Segoe UI", 14, "bold"),
                 bg=C_BG, fg=C_FG).pack(anchor=tk.W, pady=(4, 8))

        # DSN field
        frm = tk.Frame(cf, bg=C_BG)
        frm.pack(fill=tk.X, pady=4)
        tk.Label(frm, text="PostgreSQL DSN:", font=("Segoe UI", 9),
                 bg=C_BG, fg=C_GRAY).pack(side=tk.LEFT)
        tk.Entry(frm, textvariable=self.pg_dsn, font=("Consolas", 9),
                 width=48, bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=6)

        frm2 = tk.Frame(cf, bg=C_BG)
        frm2.pack(fill=tk.X, pady=4)
        tk.Label(frm2, text="Postgres password:", font=("Segoe UI", 9),
                 bg=C_BG, fg=C_GRAY).pack(side=tk.LEFT)
        tk.Entry(frm2, textvariable=self.pg_password, font=("Consolas", 9),
                 show="*", width=20, bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                 relief=tk.FLAT).pack(side=tk.LEFT, padx=6)

        btn_frm = tk.Frame(cf, bg=C_BG)
        btn_frm.pack(anchor=tk.W, pady=8)
        tk.Button(btn_frm, text="▶  Create DB + Apply Schema", font=("Segoe UI", 10),
                  bg=C_ACCENT, fg=C_BG, relief=tk.FLAT, padx=12,
                  command=self._run_db_setup).pack(side=tk.LEFT, padx=(0, 8))
        tk.Button(btn_frm, text="Test Connection", font=("Segoe UI", 10),
                  bg=C_BUTTON, fg=C_FG, relief=tk.FLAT, padx=12,
                  command=self._test_db).pack(side=tk.LEFT)

        self._db_log = tk.Text(cf, height=10, font=("Consolas", 8),
                               bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                               state=tk.DISABLED)
        self._db_log.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _test_db(self) -> None:
        threading.Thread(target=self._do_test_db, daemon=True).start()

    def _do_test_db(self) -> None:
        self._emit_to(self._db_log, "Testing connection…")
        try:
            import psycopg  # type: ignore
            with psycopg.connect(self.pg_dsn.get(), connect_timeout=5):
                self._emit_to(self._db_log, "✓ Connected!", C_GREEN)
        except Exception as exc:
            self._emit_to(self._db_log, f"✗ {exc}", C_RED)

    def _run_db_setup(self) -> None:
        threading.Thread(target=self._do_db_setup, daemon=True).start()

    def _do_db_setup(self) -> None:
        self._emit_to(self._db_log, "Setting up database…")
        try:
            import psycopg  # type: ignore
            from psycopg.rows import dict_row

            # 1. Create 'rag' database if missing
            admin_dsn = DEFAULT_PG_ADMIN
            self._emit_to(self._db_log, "Creating 'rag' database (if needed)…")
            with psycopg.connect(admin_dsn, autocommit=True) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM pg_database WHERE datname='rag'"
                ).fetchone()
                if not exists:
                    conn.execute("CREATE DATABASE rag")
                    self._emit_to(self._db_log, "  Database 'rag' created.", C_GREEN)
                else:
                    self._emit_to(self._db_log, "  Database 'rag' already exists.", C_GREEN)

                # 2. Enable pgvector
                rag_dsn = admin_dsn.rsplit("/", 1)[0] + "/rag"
                with psycopg.connect(
                    rag_dsn, autocommit=True
                ) as rag_conn:
                    rag_conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                    self._emit_to(self._db_log, "  pgvector extension enabled.", C_GREEN)

            # 3. Apply schema
            if SCHEMA_SQL.exists():
                self._emit_to(self._db_log, "Applying schema (IF NOT EXISTS — safe to re-run)…")
                schema = SCHEMA_SQL.read_text(encoding="utf-8")
                with psycopg.connect(self.pg_dsn.get(), autocommit=True) as conn:
                    conn.execute(schema)
                self._emit_to(self._db_log, "  Schema applied.", C_GREEN)
            else:
                self._emit_to(self._db_log, f"  Schema file not found at {SCHEMA_SQL}.", C_YELLOW)

            self._emit_to(self._db_log, "Database setup complete. ✓", C_GREEN)
        except Exception as exc:
            self._emit_to(self._db_log, f"Error: {exc}", C_RED)

    # ── Page 5: Models ────────────────────────────────────────────────────────
    def _page_models(self) -> None:
        cf = self._content_frame
        role = self.node_role.get()

        tk.Label(cf, text="AI Models", font=("Segoe UI", 14, "bold"),
                 bg=C_BG, fg=C_FG).pack(anchor=tk.W, pady=(4, 4))

        if role == "satellite":
            # Satellite: fixed layout — always HyDE + HF models
            tk.Label(cf, text=(
                "This satellite runs the HyDE Ollama model and the HuggingFace\n"
                "inference service (DeBERTa + CrossEncoder)."
            ), font=("Segoe UI", 10), bg=C_BG, fg=C_FG, justify=tk.LEFT).pack(
                anchor=tk.W, pady=(0, 8))
            frm = tk.Frame(cf, bg=C_BG)
            frm.pack(fill=tk.X, pady=3)
            tk.Label(frm, text="HyDE model:", font=("Segoe UI", 9), width=20,
                     bg=C_BG, fg=C_GRAY, anchor=tk.W).pack(side=tk.LEFT)
            tk.Entry(frm, textvariable=self.hyde_model, font=("Consolas", 9),
                     width=34, bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                     relief=tk.FLAT).pack(side=tk.LEFT, padx=6)
            tk.Label(cf, text="HuggingFace models (auto-download on first request):",
                     font=("Segoe UI", 9), bg=C_BG, fg=C_GRAY).pack(anchor=tk.W, pady=(8, 2))
            for m in INFERENCE_HF_MODELS:
                tk.Label(cf, text=f"  • {m}", font=("Consolas", 9),
                         bg=C_BG, fg=C_FG).pack(anchor=tk.W)
            tk.Button(cf, text="▶  Pull HyDE Model", font=("Segoe UI", 10),
                      bg=C_ACCENT, fg=C_BG, relief=tk.FLAT, padx=12,
                      command=self._pull_models).pack(anchor=tk.W, pady=10)
        else:
            # Main server: flexible placement — decide per model where it runs
            tk.Label(cf, text=(
                "Choose where each model runs. Models set to 'This machine' will be\n"
                "pulled now. Satellite-assigned models are pulled on that machine."
            ), font=("Segoe UI", 10), bg=C_BG, fg=C_FG, justify=tk.LEFT).pack(
                anchor=tk.W, pady=(0, 4))

            # ── VRAM info banner ──────────────────────────────────────────────
            def _refresh_vram_display() -> None:
                """Re-query satellite /health in background; updates stored VRAM."""
                def _do() -> None:
                    for url_var, attr in [
                        (self.satellite1_url, "sat1_vram"),
                        (self.satellite2_url, "sat2_vram"),
                    ]:
                        url = url_var.get().strip().rstrip("/")
                        if not url or "192.168.x.x" in url:
                            continue
                        try:
                            with urllib.request.urlopen(f"{url}/health", timeout=5) as r:
                                d = json.loads(r.read())
                                gpu  = d.get("gpu_name", "CPU")
                                vram = d.get("vram_free_mb", "?")
                                vt   = d.get("vram_total_mb")
                                s = f"{gpu} — {vram}/{vt} MB" if vt else f"{gpu} — {vram} MB"
                                setattr(self, attr, s)
                        except Exception:
                            pass
                    # Rebuild the Models page to reflect new VRAM data
                    self.after(0, lambda: self._show_step(self._step))
                threading.Thread(target=_do, daemon=True).start()

            vram_frm = tk.Frame(cf, bg=C_BG)
            vram_frm.pack(fill=tk.X, pady=(0, 6))
            s1_label = (f"Satellite 1: {self.sat1_vram}" if self.sat1_vram
                        else "Satellite 1: not yet queried")
            s2_label = (f"Satellite 2: {self.sat2_vram}" if self.sat2_vram
                        else "Satellite 2: not yet queried")
            vram_col = C_GREEN if self.sat1_vram else C_GRAY
            tk.Label(vram_frm, text=f"VRAM detected — {s1_label}",
                     font=("Segoe UI", 9), bg=C_BG, fg=vram_col).pack(side=tk.LEFT)
            if self.satellite2_url.get().strip() and "192.168.x.x" not in self.satellite2_url.get():
                tk.Label(vram_frm, text=f"  |  {s2_label}",
                         font=("Segoe UI", 9), bg=C_BG, fg=vram_col).pack(side=tk.LEFT)
            tk.Button(vram_frm, text="↻ Refresh VRAM", font=("Segoe UI", 8),
                      bg=C_BUTTON, fg=C_FG, relief=tk.FLAT, padx=8,
                      command=_refresh_vram_display).pack(side=tk.LEFT, padx=(12, 0))

            # Build location options from whatever was entered on Satellites page
            loc_opts: list[tuple[str, str]] = [("localhost", "This machine")]
            s1 = self.satellite1_url.get().strip()
            if s1 and "192.168.x.x" not in s1:
                host = s1.split("//")[-1].split(":")[0]
                vram_note = f"  [{self.sat1_vram}]" if self.sat1_vram else ""
                loc_opts.append(("satellite1", f"Satellite 1  ({host}){vram_note}"))
            s2 = self.satellite2_url.get().strip()
            if s2 and "192.168.x.x" not in s2:
                host = s2.split("//")[-1].split(":")[0]
                vram_note = f"  [{self.sat2_vram}]" if self.sat2_vram else ""
                loc_opts.append(("satellite2", f"Satellite 2  ({host}){vram_note}"))

            # ── Scrollable area for the 4 model sections ───────────────────────────
            scroll_outer = tk.Frame(cf, bg=C_BG, height=1)
            scroll_outer.pack(fill=tk.BOTH, expand=True)
            scroll_outer.pack_propagate(False)

            _canvas = tk.Canvas(scroll_outer, bg=C_BG, highlightthickness=0, bd=0)
            _vsb    = tk.Scrollbar(scroll_outer, orient=tk.VERTICAL,
                                   command=_canvas.yview)
            _canvas.configure(yscrollcommand=_vsb.set)
            _vsb.pack(side=tk.RIGHT, fill=tk.Y)
            _canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            inner = tk.Frame(_canvas, bg=C_BG)
            _iwin = _canvas.create_window((0, 0), window=inner, anchor=tk.NW)

            def _on_inner_resize(e: tk.Event) -> None:
                _canvas.configure(scrollregion=_canvas.bbox("all"))
            def _on_canvas_resize(e: tk.Event) -> None:
                _canvas.itemconfig(_iwin, width=e.width)
            inner.bind("<Configure>", _on_inner_resize)
            _canvas.bind("<Configure>", _on_canvas_resize)

            def _on_mousewheel(e: tk.Event) -> None:
                _canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
            _canvas.bind_all("<MouseWheel>", _on_mousewheel)

            def _lf(title: str) -> tk.LabelFrame:
                f = tk.LabelFrame(inner, text=f" {title} ", bg=C_BG, fg=C_ACCENT,
                                  font=("Segoe UI", 9, "bold"), relief=tk.GROOVE)
                f.pack(fill=tk.X, pady=(0, 6))
                return f

            def _model_row(parent: tk.Widget, label: str, var: tk.StringVar,
                           note: str = "") -> None:
                rf = tk.Frame(parent, bg=C_BG)
                rf.pack(fill=tk.X, padx=10, pady=3)
                tk.Label(rf, text=label, font=("Segoe UI", 9), width=16,
                         bg=C_BG, fg=C_GRAY, anchor=tk.W).pack(side=tk.LEFT)
                tk.Entry(rf, textvariable=var, font=("Consolas", 9), width=30,
                         bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                         relief=tk.FLAT).pack(side=tk.LEFT, padx=6)
                if note:
                    tk.Label(rf, text=note, font=("Segoe UI", 8),
                             bg=C_BG, fg=C_GRAY).pack(side=tk.LEFT)

            def _loc_row(parent: tk.Widget, loc_var: tk.StringVar) -> None:
                rf = tk.Frame(parent, bg=C_BG)
                rf.pack(fill=tk.X, padx=10, pady=(4, 2))
                tk.Label(rf, text="Runs on:", font=("Segoe UI", 9),
                         bg=C_BG, fg=C_GRAY, anchor=tk.W).pack(anchor=tk.W)
                for val, display in loc_opts:
                    tk.Radiobutton(rf, text=display, variable=loc_var, value=val,
                                   font=("Segoe UI", 9), bg=C_BG, fg=C_FG,
                                   selectcolor=C_PANEL, activebackground=C_BG,
                                   activeforeground=C_FG, anchor=tk.W).pack(
                        anchor=tk.W, padx=(16, 0))

            # LLM (Ollama) — now freely placeable
            lf1 = _lf("LLM  (answer generation)")
            _loc_row(lf1, self.llm_location)
            _model_row(lf1, "Model name:", self.llm_model,
                       _vram_note(self.llm_model.get()))

            # Embedding (Ollama) — now freely placeable
            lf2 = _lf("Embedding Model  (vector search)")
            _loc_row(lf2, self.embed_location)
            _model_row(lf2, "Model name:", self.embed_model,
                       _vram_note(self.embed_model.get()))

            # HyDE (Ollama) — user chooses
            lf3 = _lf("HyDE Model  (hypothesis generation for retrieval)")
            _loc_row(lf3, self.hyde_location)
            _model_row(lf3, "Model name:", self.hyde_model,
                       _vram_note(self.hyde_model.get()))

            # Inference service (HuggingFace) — user chooses
            lf4 = _lf("Inference Service  (DeBERTa + CrossEncoder)")
            _loc_row(lf4, self.inference_location)

            # NLI model selector
            _NLI_MODELS = [
                "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
                "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
            ]
            _CE_MODELS = [
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "cross-encoder/ms-marco-TinyBERT-L-2-v2",
            ]

            def _hf_row(parent: tk.Widget, label: str, var: tk.StringVar,
                        choices: list[str]) -> None:
                rf = tk.Frame(parent, bg=C_BG)
                rf.pack(fill=tk.X, padx=10, pady=3)
                tk.Label(rf, text=label, font=("Segoe UI", 9), width=18,
                         bg=C_BG, fg=C_GRAY, anchor=tk.W).pack(side=tk.LEFT)
                om = tk.OptionMenu(rf, var, *choices)
                om.config(font=("Consolas", 9), bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                          activebackground=C_BUTTON, highlightthickness=0, width=40)
                om["menu"].config(bg=C_PANEL, fg=C_FG, font=("Consolas", 9))
                om.pack(side=tk.LEFT, padx=6)
                hint_var = tk.StringVar()
                hint_lbl = tk.Label(rf, textvariable=hint_var, font=("Segoe UI", 8),
                                    bg=C_BG, fg=C_GRAY)
                hint_lbl.pack(side=tk.LEFT)

                def _update_hint(*_: object) -> None:
                    hint_var.set(_vram_note(var.get()))
                var.trace_add("write", _update_hint)
                _update_hint()

            _hf_row(lf4, "Intent classifier:", self.nli_model, _NLI_MODELS)
            _hf_row(lf4, "Reranker (CE):", self.ce_model, _CE_MODELS)
            tk.Label(lf4, bg=C_BG, height=1).pack()  # bottom padding

            tk.Button(inner, text="▶  Pull Ollama Models Assigned to This Machine",
                      font=("Segoe UI", 10), bg=C_ACCENT, fg=C_BG,
                      relief=tk.FLAT, padx=12,
                      command=self._pull_models).pack(anchor=tk.W, pady=(4, 8))

        self._model_log = tk.Text(cf, height=5, font=("Consolas", 8),
                                  bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                                  state=tk.DISABLED)
        self._model_log.pack(fill=tk.X)

    def _pull_models(self) -> None:
        threading.Thread(target=self._do_pull_models, daemon=True).start()

    def _do_pull_models(self) -> None:
        role = self.node_role.get()
        if role == "satellite":
            models = [self.hyde_model.get()]
        else:
            # Only pull models that are assigned to this machine
            models = []
            if self.llm_location.get() == "localhost":
                models.append(self.llm_model.get())
            if self.embed_location.get() == "localhost":
                models.append(self.embed_model.get())
            if self.hyde_location.get() == "localhost":
                models.append(self.hyde_model.get())

        for model in models:
            if not model.strip():
                continue
            # Skip pull if model already exists locally.
            if self._model_exists(model):
                self._emit_to(self._model_log, f"✓ {model} already downloaded — skipping.", C_GREEN)
                continue
            self._emit_to(self._model_log, f"Pulling {model}…")
            pulled = self._pull_via_api(model)
            if not pulled:
                # Fallback: subprocess
                self._emit_to(self._model_log, f"  (HTTP pull unavailable, using CLI)")
                try:
                    proc = subprocess.Popen(
                        ["ollama", "pull", model],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, bufsize=1,
                    )
                    assert proc.stdout
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line:
                            self._emit_to(self._model_log, f"  {line}")
                    proc.wait()
                    if proc.returncode == 0:
                        self._emit_to(self._model_log, f"✓ {model} ready.", C_GREEN)
                    else:
                        self._emit_to(self._model_log,
                                      f"✗ Pull failed (exit {proc.returncode})", C_RED)
                except FileNotFoundError:
                    self._emit_to(self._model_log,
                                  "Ollama not found in PATH. Is it installed and running?", C_RED)
                except Exception as exc:
                    self._emit_to(self._model_log, f"Error: {exc}", C_RED)

    def _model_exists(self, model: str) -> bool:
        """Return True if *model* is already present in the local Ollama library."""
        import json as _json
        try:
            with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as resp:
                d = _json.loads(resp.read())
                names = [m.get("name", "") for m in (d.get("models") or [])]
                return any(n == model or n.startswith(model + ":") for n in names)
        except Exception:
            return False

    def _pull_via_api(self, model: str) -> bool:
        """Stream a model pull via Ollama /api/pull.  Returns True on success."""
        import json as _json
        url = "http://localhost:11434/api/pull"
        payload = _json.dumps({"model": model, "stream": True}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                last_status = ""
                last_pct = -1
                for raw in resp:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except Exception:
                        continue
                    status = str(obj.get("status", ""))
                    completed = obj.get("completed")
                    total = obj.get("total")
                    if status == "success":
                        self._emit_to(self._model_log, f"✓ {model} ready.", C_GREEN)
                        return True
                    if total and completed is not None:
                        pct = int(completed / total * 100)
                        bar_len = 20
                        filled = int(pct / 100 * bar_len)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        line_text = (
                            f"  [{bar}] {pct:3d}%  "
                            f"{int(completed) // 1_048_576} / {int(total) // 1_048_576} MB"
                        )
                        if pct != last_pct:
                            self._emit_to(self._model_log, line_text)
                            last_pct = pct
                    elif status and status != last_status:
                        self._emit_to(self._model_log, f"  {status}")
                        last_status = status
            return True
        except urllib.error.URLError:
            return False  # Ollama not running / unreachable
        except Exception as exc:
            self._emit_to(self._model_log, f"  HTTP pull error: {exc}", C_YELLOW)
            return False

    # ── Page 4: Satellites ────────────────────────────────────────────────────
    def _page_satellites(self) -> None:
        cf = self._content_frame
        role = self.node_role.get()

        tk.Label(cf, text="Satellite Nodes", font=("Segoe UI", 14, "bold"),
                 bg=C_BG, fg=C_FG).pack(anchor=tk.W, pady=(4, 4))

        if role == "satellite":
            tk.Label(cf, text=(
                "This machine IS a satellite node. Nothing extra to configure here.\n\n"
                "To start the inference service on this machine:\n"
                "  uvicorn inference_service.server:app --host 0.0.0.0 --port 8100\n\n"
                "Make sure the main server's runtime.yaml points to this IP."
            ), font=("Segoe UI", 10), bg=C_BG, fg=C_FG, justify=tk.LEFT).pack(anchor=tk.W)
            return

        tk.Label(cf, text=(
            "Satellite nodes run HuggingFace models (DeBERTa intent classifier +\n"
            "CrossEncoder reranker) and the HyDE Ollama LLM offloaded from this PC.\n\n"
            "Configure up to 2 satellites below. Leave blank to run models locally."
        ), font=("Segoe UI", 10), bg=C_BG, fg=C_FG, justify=tk.LEFT).pack(
            anchor=tk.W, pady=(0, 10)
        )

        def _satellite_block(parent, label, url_var, hyde_var):
            lf = tk.LabelFrame(parent, text=f" {label} ", bg=C_BG, fg=C_ACCENT,
                               font=("Segoe UI", 9, "bold"), relief=tk.GROOVE)
            lf.pack(fill=tk.X, pady=4)
            for row_label, var, placeholder in [
                ("Inference URL (port 8100):", url_var,  "http://192.168.x.x:8100"),
                ("Ollama/HyDE URL (port 11434):", hyde_var, "http://192.168.x.x:11434"),
            ]:
                rf = tk.Frame(lf, bg=C_BG)
                rf.pack(fill=tk.X, padx=8, pady=3)
                tk.Label(rf, text=row_label, font=("Segoe UI", 9), width=28,
                         bg=C_BG, fg=C_GRAY, anchor=tk.W).pack(side=tk.LEFT)
                e = tk.Entry(rf, textvariable=var, font=("Consolas", 9),
                             width=30, bg=C_PANEL, fg=C_FG, insertbackground=C_FG,
                             relief=tk.FLAT)
                e.pack(side=tk.LEFT, padx=4)
                if not var.get():
                    e.insert(0, placeholder)
                    e.config(fg=C_GRAY)
                    def _clear(ev, entry=e, v=var, ph=placeholder):
                        if entry.get() == ph:
                            entry.delete(0, tk.END)
                            entry.config(fg=C_FG)
                    def _restore(ev, entry=e, v=var, ph=placeholder):
                        if not entry.get():
                            entry.insert(0, ph)
                            entry.config(fg=C_GRAY)
                    e.bind("<FocusIn>",  _clear)
                    e.bind("<FocusOut>", _restore)
                # Test button
                tk.Button(rf, text="Test", font=("Segoe UI", 8),
                          bg=C_BUTTON, fg=C_FG, relief=tk.FLAT,
                          command=lambda v=var: threading.Thread(
                              target=self._test_satellite, args=(v.get(),), daemon=True
                          ).start()).pack(side=tk.LEFT, padx=4)

        _satellite_block(cf, "Satellite 1", self.satellite1_url, self.satellite1_hyde)
        _satellite_block(cf, "Satellite 2 (optional)", self.satellite2_url, self.satellite2_hyde)

        self._sat_log = tk.Text(cf, height=4, font=("Consolas", 8),
                                bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                                state=tk.DISABLED)
        self._sat_log.pack(fill=tk.X, pady=(8, 0))

    def _test_satellite(self, url: str) -> None:
        log = getattr(self, "_sat_log", None)
        if not log:
            return
        clean = url.strip().rstrip("/")
        if not clean or "192.168.x.x" in clean:
            self._emit_to(log, "Enter a real IP address first.", C_YELLOW)
            return

        def _store_summary(summary: str) -> None:
            s1 = self.satellite1_url.get().strip().rstrip("/")
            s2 = self.satellite2_url.get().strip().rstrip("/")
            if clean == s1:
                self.sat1_vram = summary
            elif clean == s2:
                self.sat2_vram = summary

        # Try the RAG inference server /health endpoint first
        self._emit_to(log, f"Testing {clean}/health …")
        try:
            with urllib.request.urlopen(f"{clean}/health", timeout=5) as resp:
                data = json.loads(resp.read())
                gpu  = data.get("gpu_name", "CPU")
                vram = data.get("vram_free_mb", "?")
                vram_total = data.get("vram_total_mb")
                summary = f"{gpu}"
                if vram_total:
                    summary += f" — {vram} / {vram_total} MB VRAM free"
                elif vram != "?":
                    summary += f" — {vram} MB VRAM free"
                _store_summary(summary)
                self._emit_to(log, f"  ✓ Online (inference server) — {summary}", C_GREEN)
                return
        except urllib.error.HTTPError as e:
            if e.code != 404:
                self._emit_to(log, f"  ✗ {e}", C_RED)
                return
            # 404 → probably Ollama directly; fall through to Ollama check
        except Exception as exc:
            self._emit_to(log, f"  ✗ {exc}", C_RED)
            return

        # Fall back: check if this is an Ollama instance (GET /api/tags)
        self._emit_to(log, f"  /health not found — checking for Ollama at {clean}/api/tags …")
        try:
            with urllib.request.urlopen(f"{clean}/api/tags", timeout=5) as resp:
                data = json.loads(resp.read())
                models = data.get("models", [])
                model_names = ", ".join(m["name"] for m in models[:3]) if models else "no models pulled"
                summary = f"Ollama ({model_names})"
                _store_summary(summary)
                self._emit_to(log, f"  ✓ Ollama is running — {model_names}", C_GREEN)
                self._emit_to(log,
                    "  Note: point this URL at the inference server (port 8100) for VRAM info.",
                    C_YELLOW)
        except Exception as exc:
            self._emit_to(log, f"  ✗ Not an inference server or Ollama: {exc}", C_RED)

    # ── Page 6: Finish ────────────────────────────────────────────────────────
    def _page_finish(self) -> None:
        cf = self._content_frame
        role = self.node_role.get()

        tk.Label(cf, text="All Done!", font=("Segoe UI", 16, "bold"),
                 bg=C_BG, fg=C_GREEN).pack(anchor=tk.W, pady=(8, 4))

        if role == "main":
            msg = (
                "Personal AI Server is ready to use.\n\n"
                "Next steps:\n"
                "  • Launch rag.exe  (or double-click the Desktop shortcut)\n"
                "  • Click ▶ Start to start the AI server\n"
                "  • Click ⊕ Open UI to access the web interface\n"
                "  • Upload documents via the Library tab\n"
            )
        else:
            msg = (
                "Satellite node is configured.\n\n"
                "Next steps:\n"
                "  • Run:  uvicorn inference_service.server:app --host 0.0.0.0 --port 8100\n"
                "  • The main server will connect to this machine automatically\n"
                "  • HuggingFace models download on first request (~1-2 GB)\n"
            )

        tk.Label(cf, text=msg, font=("Segoe UI", 10), bg=C_BG, fg=C_FG,
                 justify=tk.LEFT).pack(anchor=tk.W)

        btn_frm = tk.Frame(cf, bg=C_BG)
        btn_frm.pack(anchor=tk.W, pady=10)

        # Row 1
        row1 = tk.Frame(btn_frm, bg=C_BG)
        row1.pack(anchor=tk.W, pady=(0, 6))

        tk.Button(row1, text="Write configs/runtime.yaml",
                  font=("Segoe UI", 10), bg=C_BUTTON, fg=C_FG,
                  relief=tk.FLAT, padx=12,
                  command=self._write_runtime_yaml).pack(side=tk.LEFT, padx=(0, 8))

        if role == "main":
            tk.Button(row1, text="Create Desktop Shortcut",
                      font=("Segoe UI", 10), bg=C_BUTTON, fg=C_FG,
                      relief=tk.FLAT, padx=12,
                      command=self._create_shortcut).pack(side=tk.LEFT, padx=(0, 8))

        # Row 2
        row2 = tk.Frame(btn_frm, bg=C_BG)
        row2.pack(anchor=tk.W)

        if role == "main":
            tk.Button(row2, text="Launch Server Now",
                      font=("Segoe UI", 10), bg=C_ACCENT, fg=C_BG,
                      relief=tk.FLAT, padx=12,
                      command=self._launch_server).pack(side=tk.LEFT, padx=(0, 8))

        tk.Button(row2, text="Run at Startup",
                  font=("Segoe UI", 10), bg=C_BUTTON, fg=C_FG,
                  relief=tk.FLAT, padx=12,
                  command=self._install_service).pack(side=tk.LEFT)

        self._finish_log = tk.Text(cf, height=6, font=("Consolas", 8),
                                   bg=C_PANEL, fg=C_FG, relief=tk.FLAT,
                                   state=tk.DISABLED)
        self._finish_log.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _write_runtime_yaml(self) -> None:
        threading.Thread(target=self._do_write_runtime_yaml, daemon=True).start()

    def _do_write_runtime_yaml(self) -> None:
        log = self._finish_log
        # Always write to the install directory — that's where rag.exe will look.
        runtime_yaml = Path(self.install_dir.get()) / "configs" / "runtime.yaml"
        # Use the install-dir copy as template if it exists, otherwise fall back
        # to the bundle template (e.g. when running from dev or fresh install).
        src_yaml = runtime_yaml if runtime_yaml.exists() else RUNTIME_YAML
        try:
            import yaml  # type: ignore
            if src_yaml.exists():
                data = yaml.safe_load(src_yaml.read_text(encoding="utf-8")) or {}
            else:
                data = {}  # no template — build from scratch

            # Helper: resolve a URL var based on placement key
            def _url(loc: str, sat1_var: tk.StringVar, sat2_var: tk.StringVar,
                     local: str) -> str:
                if loc == "localhost":
                    return local
                if loc == "satellite2":
                    return sat2_var.get().strip()
                return sat1_var.get().strip()  # satellite1 default

            # LLM Ollama base URL
            llm_url = _url(
                self.llm_location.get(),
                self.satellite1_hyde, self.satellite2_hyde,
                "http://localhost:11434",
            )
            if llm_url and "192.168.x.x" not in llm_url:
                data.setdefault("llm", {})["base_url"] = llm_url
                self._emit_to(log, f"  llm.base_url → {llm_url}")

            # Embedding Ollama base URL
            embed_url = _url(
                self.embed_location.get(),
                self.satellite1_hyde, self.satellite2_hyde,
                "http://localhost:11434",
            )
            if embed_url and "192.168.x.x" not in embed_url:
                data.setdefault("embedding", {})["ollama_base_url"] = embed_url
                self._emit_to(log, f"  embedding.ollama_base_url → {embed_url}")

            # HyDE Ollama base URL
            hyde_url = _url(
                self.hyde_location.get(),
                self.satellite1_hyde, self.satellite2_hyde,
                "http://localhost:11434",
            )
            if hyde_url and "192.168.x.x" not in hyde_url:
                data.setdefault("hyde", {})["base_url"] = hyde_url
                self._emit_to(log, f"  hyde.base_url → {hyde_url}")

            # Inference service URL (DeBERTa + CrossEncoder)
            inf_url = _url(
                self.inference_location.get(),
                self.satellite1_url, self.satellite2_url,
                "http://localhost:8100",
            )
            if inf_url and "192.168.x.x" not in inf_url:
                data.setdefault("inference_service", {})["url"] = inf_url
                self._emit_to(log, f"  inference_service.url → {inf_url}")

            # DB DSN
            dsn = self.pg_dsn.get().strip()
            if dsn:
                data.setdefault("paths", {})["db_dsn"] = dsn
                self._emit_to(log, f"  paths.db_dsn → {dsn}")

            # HuggingFace model selection (session 79)
            nli = self.nli_model.get().strip()
            ce  = self.ce_model.get().strip()
            if nli:
                data.setdefault("inference_service", {})["nli_model"] = nli
                self._emit_to(log, f"  inference_service.nli_model → {nli}")
            if ce:
                data.setdefault("inference_service", {})["ce_model"] = ce
                self._emit_to(log, f"  inference_service.ce_model → {ce}")

            runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
            runtime_yaml.write_text(
                yaml.dump(data, allow_unicode=True, default_flow_style=False),
                encoding="utf-8",
            )
            self._emit_to(log, "✓ configs/runtime.yaml saved.", C_GREEN)
        except ImportError:
            self._emit_to(log, "PyYAML not available — cannot write YAML. Edit runtime.yaml manually.", C_YELLOW)
        except Exception as exc:
            self._emit_to(log, f"Error writing runtime.yaml: {exc}", C_RED)

    def _install_service(self) -> None:
        threading.Thread(target=self._do_install_service, daemon=True).start()

    def _do_install_service(self) -> None:
        log = self._finish_log
        role = self.node_role.get()
        install_dir_path = Path(self.install_dir.get())
        python_exe = sys.executable

        if IS_WINDOWS:
            # Add a shortcut to the user's Startup folder — runs at login with
            # no elevation and no Task Scheduler needed.
            if role == "satellite":
                execute  = python_exe
                argument = "-m uvicorn inference_service.server:app --host 0.0.0.0 --port 8100"
                lnk_name = "RAG Inference Service.lnk"
            else:
                execute  = str(install_dir_path / "rag.exe")
                argument = "serve"
                lnk_name = "Personal AI Server.lnk"
            try:
                startup = Path(os.environ.get("APPDATA", "")) / \
                    "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
                startup.mkdir(parents=True, exist_ok=True)
                lnk_path = startup / lnk_name
                ps_cmd = (
                    f'$s=(New-Object -COM WScript.Shell).CreateShortcut("{lnk_path}");'
                    f'$s.TargetPath="{execute}";'
                    f'$s.Arguments="{argument}";'
                    f'$s.WorkingDirectory="{install_dir_path}";'
                    f'$s.WindowStyle=7;'   # 7 = minimised
                    f'$s.Save()'
                )
                subprocess.run(["powershell", "-Command", ps_cmd], check=True)
                self._emit_to(log, f"\u2713 Startup shortcut created: {lnk_path}", C_GREEN)
            except Exception as exc:
                self._emit_to(log, f"Startup shortcut failed: {exc}", C_RED)
        else:
            # systemd user service (no sudo needed)
            if role == "satellite":
                description = "RAG Inference Service"
                exec_start = f"{python_exe} -m uvicorn inference_service.server:app --host 0.0.0.0 --port 8100"
                service_name = "rag-inference"
            else:
                description = "Personal AI Server"
                exec_start = f"{install_dir_path / 'rag.exe'} serve"
                service_name = "rag-server"

            unit = (
                "[Unit]\n"
                f"Description={description}\n"
                "After=network.target\n\n"
                "[Service]\n"
                "Type=simple\n"
                f"WorkingDirectory={install_dir_path}\n"
                f"ExecStart={exec_start}\n"
                "Restart=on-failure\n"
                "RestartSec=5\n\n"
                "[Install]\n"
                "WantedBy=default.target\n"
            )
            systemd_dir = Path.home() / ".config" / "systemd" / "user"
            try:
                systemd_dir.mkdir(parents=True, exist_ok=True)
                unit_path = systemd_dir / f"{service_name}.service"
                unit_path.write_text(unit, encoding="utf-8")
                subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
                subprocess.run(["systemctl", "--user", "enable", service_name], check=True)
                self._emit_to(log, f"\u2713 systemd user service '{service_name}' enabled.", C_GREEN)
                self._emit_to(log, f"  Start now: systemctl --user start {service_name}")
                self._emit_to(log, f"  View logs: journalctl --user -u {service_name} -f")
            except Exception as exc:
                self._emit_to(log, f"systemd install failed: {exc}", C_RED)
                self._emit_to(log, f"  Unit file written to: {systemd_dir / (service_name + '.service')}")
                self._emit_to(log, "  Run: systemctl --user daemon-reload && systemctl --user enable " + service_name)

    def _create_shortcut(self) -> None:
        threading.Thread(target=self._do_create_shortcut, daemon=True).start()

    def _do_create_shortcut(self) -> None:
        if not IS_WINDOWS:
            # Create a .desktop launcher on Linux/macOS
            try:
                install_dir_path = Path(self.install_dir.get())
                rag_exe = install_dir_path / "rag.exe"
                desktop_dir = Path.home() / "Desktop"
                desktop_dir.mkdir(exist_ok=True)
                shortcut = desktop_dir / "Personal AI Server.desktop"
                shortcut.write_text(
                    "[Desktop Entry]\n"
                    "Type=Application\n"
                    "Name=Personal AI Server\n"
                    f"Exec={rag_exe} ui\n"
                    f"Path={install_dir_path}\n"
                    "Terminal=false\n"
                    "Categories=Utility;\n",
                    encoding="utf-8",
                )
                shortcut.chmod(0o755)
                self._emit_to(self._finish_log, f"\u2713 .desktop file created: {shortcut}", C_GREEN)
            except Exception as exc:
                self._emit_to(self._finish_log, f"Could not create shortcut: {exc}", C_RED)
            return
        try:
            import winreg as _reg  # noqa: PLC0415
            # Resolve the real Desktop path from the Windows registry (handles
            # OneDrive-redirected Desktops like C:\Users\<user>\OneDrive\Desktop)
            try:
                with _reg.OpenKey(
                    _reg.HKEY_CURRENT_USER,
                    r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
                ) as key:
                    desktop = Path(_reg.QueryValueEx(key, "Desktop")[0])
            except Exception:
                desktop = Path(os.path.expanduser("~")) / "Desktop"
            launcher_exe = Path(self.install_dir.get()) / "rag.exe"
            target = str(launcher_exe) if launcher_exe.exists() else sys.executable
            shortcut_path = desktop / "Personal AI Server.lnk"

            # Use PowerShell to create the .lnk shortcut (no COM dependency)
            ps_cmd = (
                f'$s=(New-Object -COM WScript.Shell).CreateShortcut("{shortcut_path}");'
                f'$s.TargetPath="{target}";'
                f'$s.WorkingDirectory="{self.install_dir.get()}";'
                f'$s.Description="Personal AI Server Launcher";'
                f'$s.Save()'
            )
            subprocess.run(["powershell", "-Command", ps_cmd], check=True)
            self._emit_to(self._finish_log, f"✓ Shortcut created: {shortcut_path}", C_GREEN)
        except Exception as exc:
            self._emit_to(self._finish_log, f"Could not create shortcut: {exc}", C_RED)

    def _launch_server(self) -> None:
        launcher = Path(self.install_dir.get()) / "rag.exe"
        target = str(launcher) if launcher.exists() else None
        if target:
            subprocess.Popen([target])
            self._emit_to(self._finish_log, "Launcher started.", C_GREEN)
        else:
            self._emit_to(self._finish_log, "Launcher EXE not found — run rag.exe manually.", C_YELLOW)

    # ── Log helpers ───────────────────────────────────────────────────────────
    def _emit(self, msg: str, colour: str = C_FG) -> None:
        self._log_q.put((msg, colour))

    def _emit_to(self, widget: tk.Text, msg: str, colour: str = C_FG) -> None:
        def _write():
            widget.config(state=tk.NORMAL)
            tag = f"c_{id(colour)}"
            widget.tag_configure(tag, foreground=colour)
            widget.insert(tk.END, msg + "\n", tag)
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        self.after(0, _write)

    def _drain_log(self) -> None:
        try:
            while True:
                msg, colour = self._log_q.get_nowait()
                # Route to whichever log is currently visible
                for attr in ("_prereq_log", "_db_log", "_model_log", "_finish_log"):
                    w = getattr(self, attr, None)
                    if w and w.winfo_exists() and w.winfo_ismapped():
                        self._emit_to(w, msg, colour)
                        break
        except queue.Empty:
            pass
        self.after(200, self._drain_log)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
