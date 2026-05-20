"""Extraction service facade — Large Document Branched Extraction.

This is the single import surface for the server layer.  All internals
(extraction/batch.py, extraction/branch_runner.py, etc.) are accessed only
through this module.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def run_project(
    project_config: Dict[str, Any],
    *,
    db_dsn: Optional[str] = None,
    emit: Optional[Callable[[str], None]] = None,
) -> Any:
    """Run a full branched extraction project.

    Parameters
    ----------
    project_config : dict matching ProjectConfig schema.
    db_dsn        : override the PostgreSQL DSN.
    emit           : optional progress callback.

    Returns a ProjectRunResult.
    """
    from extraction.branch_config import ProjectConfig
    from extraction.project_runner import run_project as _run
    project = ProjectConfig.from_dict(project_config)
    return _run(project, db_dsn=db_dsn, emit=emit)


def suggest_branches(
    llm_fn: Callable,
    *,
    document_path: Optional[str] = None,
    db_dsn: Optional[str] = None,
    collection_id: Optional[str] = None,
    doc_type: str = "",
    title: str = "",
    max_chunks: int = 30,
) -> List[Dict[str, Any]]:
    """Auto-suggest extraction branches from a document sample."""
    from extraction.suggest import suggest_branches as _suggest
    branches = _suggest(
        llm_fn,
        document_path=document_path,
        db_dsn=db_dsn,
        collection_id=collection_id,
        doc_type=doc_type,
        title=title,
        max_chunks=max_chunks,
    )
    return [b.to_dict() for b in branches]


def get_flag_library(path: Optional[str] = None) -> Dict[str, Any]:
    """Load the persistent flag library."""
    from extraction.project_runner import load_flag_library
    kwargs = {"path": path} if path else {}
    return load_flag_library(**kwargs)


def save_flag(branch_config: Dict[str, Any], path: Optional[str] = None) -> None:
    """Append a branch config to the flag library."""
    from extraction.project_runner import save_flag as _save
    kwargs = {"path": path} if path else {}
    _save(branch_config, **kwargs)


def list_projects(projects_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """Return saved project slugs + names."""
    from extraction.project_runner import list_projects as _list
    kwargs = {"projects_dir": projects_dir} if projects_dir else {}
    return _list(**kwargs)


def get_project(slug: str, projects_dir: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load a saved project config by slug.  Returns None if not found."""
    from extraction.branch_config import ProjectConfig
    from utils.config import load_yaml_config
    if not projects_dir:
        cfg = load_yaml_config("configs/extraction.yaml") or {}
        projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    try:
        project = ProjectConfig.load(slug, projects_dir)
        return project.to_dict() if hasattr(project, "to_dict") else project.__dict__
    except FileNotFoundError:
        return None


def save_project(project_config: Dict[str, Any], projects_dir: Optional[str] = None) -> None:
    """Save/update a project config to disk."""
    from extraction.branch_config import ProjectConfig
    from utils.config import load_yaml_config
    if not projects_dir:
        cfg = load_yaml_config("configs/extraction.yaml") or {}
        projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    project = ProjectConfig.from_dict(project_config)
    project.save(projects_dir)
