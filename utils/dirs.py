"""
utils/dirs.py — Authoritative list of required runtime directories.

Call ``ensure_runtime_dirs(root)`` at startup from server, launcher, and the
migration script to guarantee the directory tree exists before any component
tries to write to it.  This prevents silent failures when the executable is
moved to a new location.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

# Relative paths (from project/install root) that must exist at runtime.
_REQUIRED_DIRS: list[str] = [
    "data/raw",
    "data/chunks",
    "data/db",
    "data/diagnostics",
    "data/extracted",
    "data/index",
    "data/markdown",
    "data/metadata",
    "data/qa",
    "data/structured",
    "data/context.json",  # parent dir only — handled specially below
    "db",
    "logs",
]

# Paths that are files, not directories — we ensure their *parent* exists.
_PARENT_ONLY: frozenset[str] = frozenset({
    "data/context.json",
})


def ensure_runtime_dirs(root: Union[str, Path, None] = None) -> None:
    """Create all required runtime directories under *root*.

    Parameters
    ----------
    root:
        Project/install root directory.  Defaults to the directory two levels
        above this file (i.e. the project root when running from source).
    """
    if root is None:
        root = Path(__file__).resolve().parents[1]
    root = Path(root)
    for rel in _REQUIRED_DIRS:
        target = root / rel
        if rel in _PARENT_ONLY:
            target.parent.mkdir(parents=True, exist_ok=True)
        else:
            target.mkdir(parents=True, exist_ok=True)
