"""Flag library persistence helpers.

Manages the persistent flag library (branch config presets) and saved projects.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

_DEFAULT_FLAG_LIBRARY_PATH = "data/flag_library/default_flags.json"


def load_flag_library(path: str = _DEFAULT_FLAG_LIBRARY_PATH) -> Dict[str, Any]:
    """Load the persistent flag library.  Returns empty dict if not found."""
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {"version": 1, "branches": []}


def save_flag(
    branch_config: Dict[str, Any],
    path: str = _DEFAULT_FLAG_LIBRARY_PATH,
) -> None:
    """Append or update a branch config in the flag library."""
    lib = load_flag_library(path)
    branches = lib.get("branches", [])
    # Replace existing entry with same name, or append
    name = branch_config.get("name", "")
    idx = next((i for i, b in enumerate(branches) if b.get("name") == name), None)
    if idx is not None:
        branches[idx] = branch_config
    else:
        branches.append(branch_config)
    lib["branches"] = branches
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(lib, indent=2, ensure_ascii=False), encoding="utf-8")


def list_projects(projects_dir: str = "data/flag_library/projects") -> List[Dict[str, str]]:
    """Return saved project slugs + names."""
    p = Path(projects_dir)
    if not p.exists():
        return []
    results = []
    for f in sorted(p.glob("*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            results.append({"slug": data.get("slug", f.stem), "name": data.get("name", f.stem)})
        except Exception:
            pass
    return results
