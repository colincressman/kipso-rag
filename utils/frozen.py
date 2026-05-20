"""Runtime path helpers — works both in development and in a PyInstaller bundle.

In a PyInstaller --onedir build:
  - sys.frozen = True, sys._MEIPASS = dist/rag/_internal/
  - sys.executable = dist/rag/rag.exe
  - Code modules live in sys._MEIPASS (read-only, gets extracted to temp on --onefile)
  - User data (db, configs, data/) lives BESIDE the exe

In development:
  - Both helpers return the project root (parent of utils/)
"""
from __future__ import annotations

import sys
from pathlib import Path


def is_frozen() -> bool:
    """True when running inside a PyInstaller or Nuitka bundle."""
    # PyInstaller sets sys.frozen + sys._MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return True
    # Nuitka standalone sets __compiled__ (a module-level builtin) but not _MEIPASS
    if globals().get("__compiled__"):
        return True
    return False


def _is_nuitka() -> bool:
    return globals().get("__compiled__", False) and not hasattr(sys, "_MEIPASS")


def get_install_dir() -> Path:
    """Directory where writable user data lives (beside the exe, or project root in dev)."""
    if is_frozen():
        return Path(sys.executable).parent
    return Path(__file__).resolve().parents[1]


def get_main_exe() -> Path:
    """Return the path to the runnable entry-point executable.

    Nuitka standalone: sys.executable is the bundled python.exe interpreter,
    NOT the compiled rag.exe — use sys.argv[0] instead.
    PyInstaller / development: sys.executable is already correct.
    """
    if _is_nuitka():
        return Path(sys.argv[0]).resolve()
    return Path(sys.executable)


def get_bundle_dir() -> Path:
    """Directory where bundled read-only resources live.

    PyInstaller: sys._MEIPASS (extracted temp dir).
    Nuitka standalone: exe directory (files are beside the exe, not in a temp dir).
    Development: project root.
    """
    if _is_nuitka():
        return Path(sys.executable).parent
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]
