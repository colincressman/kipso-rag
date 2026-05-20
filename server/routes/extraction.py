"""Extraction API routes (large-doc ingest spec, section 11)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class ExtractionRunRequest(BaseModel):
    project_slug: Optional[str] = None
    project: Optional[Dict[str, Any]] = None
    db_dsn: Optional[str] = None
    verbose: Optional[bool] = None


class SuggestBranchesRequest(BaseModel):
    document_path: Optional[str] = None
    collection_id: Optional[str] = None
    db_dsn: Optional[str] = None
    doc_type: str = ""
    title: str = ""
    max_chunks: int = 30


class FlagSaveRequest(BaseModel):
    branch: Dict[str, Any]


class ProjectSaveRequest(BaseModel):
    project: Dict[str, Any]


# ── Helper: resolve projects_dir from config ──────────────────────────────────

def _extraction_cfg() -> Dict[str, Any]:
    from utils.config import load_yaml_config  # noqa: PLC0415
    return load_yaml_config("configs/extraction.yaml") or {}


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/extraction/run")
async def api_extraction_run(req: ExtractionRunRequest) -> StreamingResponse:
    import asyncio
    import threading
    from extraction.branch_config import ProjectConfig
    from extraction.project_runner import run_project
    from utils.runtime_defaults import DEFAULT_DB_DSN

    def _load_project() -> ProjectConfig:
        if req.project:
            return ProjectConfig.from_dict(req.project)  # type: ignore[attr-defined]
        if req.project_slug:
            cfg = _extraction_cfg()
            projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
            return ProjectConfig.load(req.project_slug, projects_dir)
        raise ValueError("Provide project_slug or inline project config")

    async def _stream():
        messages: List[str] = []
        done = asyncio.Event()
        error_holder: List[str] = []
        result_holder: List[Any] = []

        def emit(msg: str) -> None:
            messages.append(msg)

        def _worker():
            try:
                project = _load_project()
                result  = run_project(
                    project,
                    db_dsn=req.db_dsn or DEFAULT_DB_DSN,
                    emit=emit,
                    verbose=req.verbose,
                )
                result_holder.append(result)
            except Exception as exc:
                error_holder.append(str(exc))
            finally:
                done.set()

        threading.Thread(target=_worker, daemon=True).start()

        sent = 0
        while not done.is_set():
            await asyncio.sleep(0.25)
            while sent < len(messages):
                yield f"data: {json.dumps({'type': 'progress', 'message': messages[sent]})}\n\n"
                sent += 1

        while sent < len(messages):
            yield f"data: {json.dumps({'type': 'progress', 'message': messages[sent]})}\n\n"
            sent += 1

        if error_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': error_holder[0]})}\n\n"
        elif result_holder:
            r = result_holder[0]
            payload = {
                "type":            "result",
                "project_slug":    r.project_slug,
                "report_path":     r.report_path,
                "elapsed_seconds": r.elapsed_seconds,
                "checkpoint_path": r.checkpoint_path,
                "branches": [
                    {"name": br.branch_name, "status": br.status, "items": len(br.items)}
                    for br in r.branch_results
                ],
                "post_passes": [
                    {
                        "pass_name":       pp.pass_name,
                        "status":          pp.status,
                        "response_text":   pp.response_text,
                        "error":           pp.error,
                        "elapsed_seconds": pp.elapsed_seconds,
                    }
                    for pp in r.post_pass_results
                ],
                "report_markdown": r.report_markdown,
            }
            yield f"data: {json.dumps(payload)}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.post("/api/extraction/suggest-branches")
async def api_suggest_branches(req: SuggestBranchesRequest) -> JSONResponse:
    from extraction.suggest import suggest_branches
    from extraction.project_runner import _make_llm_fn
    from utils.runtime_defaults import DEFAULT_DB_DSN

    llm_fn = _make_llm_fn()
    branches = suggest_branches(
        llm_fn,
        document_path=req.document_path,
        db_dsn=req.db_dsn or DEFAULT_DB_DSN,
        collection_id=req.collection_id,
        doc_type=req.doc_type,
        title=req.title,
        max_chunks=req.max_chunks,
    )
    return JSONResponse([b.to_dict() for b in branches])


@router.get("/api/extraction/flag-library")
async def api_get_flag_library() -> JSONResponse:
    from extraction.project_runner import load_flag_library
    cfg  = _extraction_cfg()
    path = cfg.get("flag_library", {}).get("path", "data/flag_library/default_flags.json")
    return JSONResponse(load_flag_library(path))


@router.post("/api/extraction/flag-library/add", status_code=200)
async def api_add_flag(req: FlagSaveRequest) -> JSONResponse:
    from extraction.project_runner import save_flag
    cfg  = _extraction_cfg()
    path = cfg.get("flag_library", {}).get("path", "data/flag_library/default_flags.json")
    save_flag(req.branch, path=path)
    return JSONResponse({"ok": True})


@router.get("/api/extraction/projects")
async def api_list_projects() -> JSONResponse:
    from extraction.project_runner import list_projects
    cfg = _extraction_cfg()
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    return JSONResponse(list_projects(projects_dir))


@router.get("/api/extraction/projects/{slug}")
async def api_get_project(slug: str) -> JSONResponse:
    from extraction.branch_config import ProjectConfig
    cfg = _extraction_cfg()
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    try:
        project = ProjectConfig.load(slug, projects_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    return JSONResponse(project.to_dict() if hasattr(project, "to_dict") else project.__dict__)


@router.post("/api/extraction/projects/{slug}", status_code=200)
async def api_save_project(slug: str, req: ProjectSaveRequest) -> JSONResponse:
    from extraction.branch_config import ProjectConfig
    cfg = _extraction_cfg()
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    raw = req.project
    raw["slug"] = slug
    try:
        project = ProjectConfig.from_dict(raw)  # type: ignore[attr-defined]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    project.save(projects_dir)
    return JSONResponse({"ok": True, "slug": project.slug})


@router.delete("/api/extraction/projects/{slug}", status_code=200)
async def api_delete_project(slug: str) -> JSONResponse:
    cfg = _extraction_cfg()
    projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
    path = Path(projects_dir) / f"{slug}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Project '{slug}' not found")
    path.unlink()
    return JSONResponse({"ok": True, "slug": slug})


@router.get("/api/extraction/report/{slug}")
async def api_get_report(slug: str) -> PlainTextResponse:
    import glob
    cfg  = _extraction_cfg()
    base = cfg.get("report_output_path", "data/extraction_reports")
    files = sorted(glob.glob(str(Path(base) / f"{slug}_*.md")))
    if not files:
        raise HTTPException(status_code=404, detail=f"No report found for project '{slug}'")
    return PlainTextResponse(Path(files[-1]).read_text(encoding="utf-8"))


@router.get("/api/extraction/reports")
async def api_list_reports() -> List[Dict[str, Any]]:
    import glob
    import os
    cfg  = _extraction_cfg()
    base = cfg.get("report_output_path", "data/extraction_reports")
    files = sorted(glob.glob(str(Path(base) / "*.md")), reverse=True)
    results = []
    for f in files:
        p    = Path(f)
        stat = p.stat()
        results.append({"filename": p.name, "size_bytes": stat.st_size, "modified": stat.st_mtime})
    return results


@router.get("/api/extraction/report-file/{filename}")
async def api_get_report_file(filename: str) -> PlainTextResponse:
    if not re.match(r'^[\w\-. ]+\.md$', filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    cfg  = _extraction_cfg()
    base = cfg.get("report_output_path", "data/extraction_reports")
    p    = Path(base) / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found")
    return PlainTextResponse(p.read_text(encoding="utf-8"))
