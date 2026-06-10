"""Extraction API routes (large-doc ingest spec, section 11)."""
from __future__ import annotations

import json
import re
import glob
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel

router = APIRouter()
_RUNS_DIR = Path("data/extraction_runs")
_RUNS_LOCK = threading.Lock()
_RUN_THREADS: Dict[str, threading.Thread] = {}


# ── Pydantic models ───────────────────────────────────────────────────────────

class ExtractionRunRequest(BaseModel):
    project_slug: Optional[str] = None
    project: Optional[Dict[str, Any]] = None
    db_dsn: Optional[str] = None
    verbose: Optional[bool] = None


class ExtractionRerunSecondPassesRequest(BaseModel):
    project_slug: Optional[str] = None
    project: Optional[Dict[str, Any]] = None
    checkpoint_path: Optional[str] = None


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


def _latest_checkpoint_for_slug(slug: str) -> Optional[Path]:
    pattern = Path("data/extraction_checkpoints") / f"{slug}_*.jsonl"
    files = sorted(glob.glob(str(pattern)))
    if not files:
        return None
    return Path(files[-1])


def _run_meta_path(run_id: str) -> Path:
    return _RUNS_DIR / f"{run_id}.json"


def _run_events_path(run_id: str) -> Path:
    return _RUNS_DIR / f"{run_id}.jsonl"


def _read_run_meta(run_id: str) -> Dict[str, Any]:
    path = _run_meta_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"Run '{run_id}' not found")
    last_exc: Optional[Exception] = None
    for _ in range(5):
        try:
            raw = path.read_text(encoding="utf-8")
            if not raw.strip():
                time.sleep(0.02)
                continue
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            last_exc = exc
            time.sleep(0.02)
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"Run '{run_id}' not found")


def _write_run_meta(run_id: str, meta: Dict[str, Any]) -> None:
    _RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = _run_meta_path(run_id)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    tmp_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def _update_run_meta(run_id: str, **updates: Any) -> Dict[str, Any]:
    with _RUNS_LOCK:
        meta = _read_run_meta(run_id)
        meta.update(updates)
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_meta(run_id, meta)
        return meta


def _append_run_event(run_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    with _RUNS_LOCK:
        meta = _read_run_meta(run_id)
        seq = int(meta.get("last_seq", 0)) + 1
        event = {"seq": seq, **payload}
        _RUNS_DIR.mkdir(parents=True, exist_ok=True)
        with _run_events_path(run_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        meta["last_seq"] = seq
        meta["last_event_type"] = payload.get("type", "")
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_meta(run_id, meta)
        return event


def _load_run_events(run_id: str) -> List[Dict[str, Any]]:
    path = _run_events_path(run_id)
    if not path.exists():
        return []
    events: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


def _delete_run_files(run_id: str) -> None:
    _run_meta_path(run_id).unlink(missing_ok=True)
    _run_events_path(run_id).unlink(missing_ok=True)


def _classify_progress_message(message: str) -> Dict[str, Any]:
    text = str(message or "").strip()
    lowered = text.lower()
    phase = "running"
    stage = "progress"
    percent_hint: Optional[int] = None

    if lowered.startswith("starting extraction project"):
        stage = "initializing"
        percent_hint = 2
    elif lowered.startswith("collection:"):
        stage = "collection_ready"
        percent_hint = 4
    elif "ingesting" in lowered:
        phase = "ingestion"
        stage = "ingesting"
        percent_hint = 8
    elif lowered.startswith("building corpus index"):
        stage = "building_corpus"
        percent_hint = 18
    elif lowered.startswith("warming llm"):
        stage = "warming_llm"
        percent_hint = 24
    elif lowered.startswith("branch "):
        stage = "branch_start"
        percent_hint = 30
    elif "retrieving chunks" in lowered:
        stage = "retrieval"
        percent_hint = 36
    elif "scan batch" in lowered or lowered.startswith("  → scan:"):
        stage = "scan"
        percent_hint = 46
    elif "synthesis" in lowered:
        stage = "synthesis"
        percent_hint = 60
    elif "saved" in lowered and "checkpoint" in lowered:
        stage = "checkpointing"
        percent_hint = 68
    elif "second pass" in lowered:
        phase = "post"
        stage = "second_pass"
        percent_hint = 78
    elif "building report" in lowered:
        phase = "report"
        stage = "building_report"
        percent_hint = 88
    elif "report saved to:" in lowered:
        phase = "report"
        stage = "report_saved"
        percent_hint = 96
    elif "extraction complete" in lowered or "report-only rerun complete" in lowered:
        phase = "done"
        stage = "completed"
        percent_hint = 100
    elif lowered.startswith("  ✗") or lowered.startswith("error"):
        phase = "error"
        stage = "error"
    elif lowered.startswith("  ✓") or lowered.startswith("  ok"):
        stage = "completed_step"

    return {
        "phase": phase,
        "stage": stage,
        "message": text,
        "percent_hint": percent_hint,
    }


def reconcile_extraction_runs_on_startup() -> int:
    """Mark stale queued/running runs as interrupted after a server restart."""
    if not _RUNS_DIR.exists():
        return 0
    recovered = 0
    for path in sorted(_RUNS_DIR.glob("*.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        status = str(meta.get("status") or "")
        if status not in {"queued", "running", "cancel_requested"}:
            continue
        run_id = str(meta.get("run_id") or path.stem)
        meta["status"] = "error"
        meta["error"] = "Server restarted during extraction run"
        meta["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write_run_meta(run_id, meta)
        _append_run_event(
            run_id,
            {
                "type": "error",
                "message": "Server restarted during extraction run",
                "phase": "error",
                "stage": "interrupted",
                "percent_hint": None,
            },
        )
        recovered += 1
    return recovered


def _active_run_for_project(project_slug: str) -> Optional[Dict[str, Any]]:
    if not _RUNS_DIR.exists():
        return None
    metas: List[Dict[str, Any]] = []
    for path in _RUNS_DIR.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("project_slug") == project_slug and meta.get("status") in {"queued", "running"}:
            metas.append(meta)
    if not metas:
        return None
    metas.sort(key=lambda m: m.get("started_at", ""), reverse=True)
    return metas[0]


def _latest_run_for_project(project_slug: str) -> Optional[Dict[str, Any]]:
    if not _RUNS_DIR.exists():
        return None
    metas: List[Dict[str, Any]] = []
    for path in _RUNS_DIR.glob("*.json"):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("project_slug") == project_slug:
            metas.append(meta)
    if not metas:
        return None
    metas.sort(key=lambda m: m.get("started_at", ""), reverse=True)
    return metas[0]


def _result_payload(r: Any) -> Dict[str, Any]:
    return {
        "type": "result",
        "project_slug": r.project_slug,
        "report_path": r.report_path,
        "appendix_path": getattr(r, "appendix_path", None),
        "elapsed_seconds": r.elapsed_seconds,
        "checkpoint_path": r.checkpoint_path,
        "error": getattr(r, "error", None),
        "branches": [
            {"name": br.branch_name, "status": br.status, "items": len(br.items)}
            for br in r.branch_results
        ],
        "second_passes": [
            {
                "pass_name": sp.pass_name,
                "status": sp.status,
                "response_text": sp.response_text,
                "artifact_type": sp.artifact_type,
                "error": sp.error,
                "elapsed_seconds": sp.elapsed_seconds,
            }
            for sp in r.second_pass_results
        ],
        "report_markdown": r.report_markdown,
    }


def _new_run_id(prefix: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{ts}_{uuid4().hex[:8]}"


def _create_run_meta(*, run_id: str, project_slug: str, project_name: str, kind: str) -> Dict[str, Any]:
    meta = {
        "run_id": run_id,
        "project_slug": project_slug,
        "project_name": project_name,
        "kind": kind,
        "status": "queued",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "last_seq": 0,
        "checkpoint_path": None,
        "report_path": None,
        "error": None,
        "cancel_requested": False,
        "cancel_requested_at": None,
    }
    _write_run_meta(run_id, meta)
    return meta


def _is_cancel_requested(run_id: str) -> bool:
    try:
        meta = _read_run_meta(run_id)
    except FileNotFoundError:
        return False
    return bool(meta.get("cancel_requested"))


def _start_background_run(
    *,
    run_id: str,
    project: Any,
    worker_fn: Any,
    kind: str,
) -> None:
    from extraction.cancel import ExtractionCancelled

    def emit(msg: str) -> None:
        payload = {"type": "progress", **_classify_progress_message(msg)}
        _append_run_event(run_id, payload)

    def cancel_check() -> bool:
        return _is_cancel_requested(run_id)

    def _worker() -> None:
        try:
            if cancel_check():
                _update_run_meta(run_id, status="canceled")
                _append_run_event(
                    run_id,
                    {
                        "type": "canceled",
                        "message": "Run aborted by user.",
                        "phase": "error",
                        "stage": "canceled",
                        "percent_hint": None,
                    },
                )
                return
            _update_run_meta(run_id, status="running")
            result = worker_fn(project, emit, cancel_check)
            result_error = str(getattr(result, "error", "") or "").strip()
            if result_error:
                _update_run_meta(run_id, status="error", error=result_error)
                _append_run_event(
                    run_id,
                    {
                        "type": "error",
                        "message": result_error,
                        "phase": "error",
                        "stage": "result_error",
                        "percent_hint": None,
                    },
                )
                return
            _update_run_meta(
                run_id,
                status="done",
                checkpoint_path=result.checkpoint_path,
                report_path=result.report_path,
            )
            _append_run_event(run_id, _result_payload(result))
        except ExtractionCancelled as exc:
            _update_run_meta(run_id, status="canceled", error=str(exc))
            _append_run_event(
                run_id,
                {
                    "type": "canceled",
                    "message": str(exc),
                    "phase": "error",
                    "stage": "canceled",
                    "percent_hint": None,
                },
            )
        except Exception as exc:
            _update_run_meta(run_id, status="error", error=str(exc))
            _append_run_event(
                run_id,
                {
                    "type": "error",
                    "message": str(exc),
                    "phase": "error",
                    "stage": "error",
                    "percent_hint": None,
                },
            )
        finally:
            _RUN_THREADS.pop(run_id, None)

    thread = threading.Thread(target=_worker, daemon=True, name=f"extract-run-{run_id}")
    _RUN_THREADS[run_id] = thread
    thread.start()


async def _stream_existing_run(run_id: str, cursor: int = 0) -> StreamingResponse:
    async def _stream():
        sent = max(0, int(cursor or 0))
        while True:
            events = _load_run_events(run_id)
            for event in events:
                seq = int(event.get("seq", 0))
                if seq <= sent:
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                sent = seq
            meta = _read_run_meta(run_id)
            if meta.get("status") in {"done", "error", "canceled"}:
                break
            await __import__("asyncio").sleep(0.25)
        yield "data: [DONE]\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/api/extraction/run")
async def api_extraction_run(req: ExtractionRunRequest) -> StreamingResponse:
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

    project = _load_project()
    run_id = _new_run_id("extract")
    _create_run_meta(run_id=run_id, project_slug=project.slug, project_name=project.name, kind="full_run")
    _append_run_event(run_id, {"type": "run_started", "run_id": run_id, "kind": "full_run", "project_slug": project.slug})
    _start_background_run(
        run_id=run_id,
        project=project,
        kind="full_run",
        worker_fn=lambda project, emit, cancel_check: run_project(
            project,
            db_dsn=req.db_dsn or DEFAULT_DB_DSN,
            emit=emit,
            verbose=req.verbose,
            cancel_check=cancel_check,
        ),
    )
    return await _stream_existing_run(run_id, 0)


@router.post("/api/extraction/run-start")
async def api_extraction_run_start(req: ExtractionRunRequest) -> JSONResponse:
    from extraction.branch_config import ProjectConfig
    from extraction.project_runner import run_project
    from utils.runtime_defaults import DEFAULT_DB_DSN

    if req.project:
        project = ProjectConfig.from_dict(req.project)  # type: ignore[attr-defined]
    elif req.project_slug:
        cfg = _extraction_cfg()
        projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
        project = ProjectConfig.load(req.project_slug, projects_dir)
    else:
        raise HTTPException(status_code=400, detail="Provide project_slug or inline project config")

    run_id = _new_run_id("extract")
    _create_run_meta(run_id=run_id, project_slug=project.slug, project_name=project.name, kind="full_run")
    _append_run_event(run_id, {"type": "run_started", "run_id": run_id, "kind": "full_run", "project_slug": project.slug})
    _start_background_run(
        run_id=run_id,
        project=project,
        kind="full_run",
        worker_fn=lambda project, emit, cancel_check: run_project(
            project,
            db_dsn=req.db_dsn or DEFAULT_DB_DSN,
            emit=emit,
            verbose=req.verbose,
            cancel_check=cancel_check,
        ),
    )
    return JSONResponse({"run_id": run_id, "project_slug": project.slug, "status": "queued"})


@router.post("/api/extraction/rerun-second-passes")
async def api_rerun_second_passes(req: ExtractionRerunSecondPassesRequest) -> StreamingResponse:
    from extraction.branch_config import ProjectConfig
    from extraction.project_runner import rerun_reports_from_checkpoint

    def _load_project() -> ProjectConfig:
        if req.project:
            return ProjectConfig.from_dict(req.project)  # type: ignore[attr-defined]
        if req.project_slug:
            cfg = _extraction_cfg()
            projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
            return ProjectConfig.load(req.project_slug, projects_dir)
        raise ValueError("Provide project_slug or inline project config")

    project = _load_project()
    checkpoint_path = req.checkpoint_path or str(_latest_checkpoint_for_slug(project.slug) or "")
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="No checkpoint available for this project.")
    run_id = _new_run_id("rerun")
    _create_run_meta(run_id=run_id, project_slug=project.slug, project_name=project.name, kind="second_pass_rerun")
    _append_run_event(run_id, {"type": "run_started", "run_id": run_id, "kind": "second_pass_rerun", "project_slug": project.slug})
    _start_background_run(
        run_id=run_id,
        project=project,
        kind="second_pass_rerun",
        worker_fn=lambda project, emit, cancel_check: rerun_reports_from_checkpoint(
            project,
            checkpoint_path=checkpoint_path,
            emit=emit,
            cancel_check=cancel_check,
        ),
    )
    return await _stream_existing_run(run_id, 0)


@router.post("/api/extraction/rerun-second-passes-start")
async def api_rerun_second_passes_start(req: ExtractionRerunSecondPassesRequest) -> JSONResponse:
    from extraction.branch_config import ProjectConfig
    from extraction.project_runner import rerun_reports_from_checkpoint

    if req.project:
        project = ProjectConfig.from_dict(req.project)  # type: ignore[attr-defined]
    elif req.project_slug:
        cfg = _extraction_cfg()
        projects_dir = cfg.get("flag_library", {}).get("projects_dir", "data/flag_library/projects")
        project = ProjectConfig.load(req.project_slug, projects_dir)
    else:
        raise HTTPException(status_code=400, detail="Provide project_slug or inline project config")

    checkpoint_path = req.checkpoint_path or str(_latest_checkpoint_for_slug(project.slug) or "")
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="No checkpoint available for this project.")

    run_id = _new_run_id("rerun")
    _create_run_meta(run_id=run_id, project_slug=project.slug, project_name=project.name, kind="second_pass_rerun")
    _append_run_event(run_id, {"type": "run_started", "run_id": run_id, "kind": "second_pass_rerun", "project_slug": project.slug})
    _start_background_run(
        run_id=run_id,
        project=project,
        kind="second_pass_rerun",
        worker_fn=lambda project, emit, cancel_check: rerun_reports_from_checkpoint(
            project,
            checkpoint_path=checkpoint_path,
            emit=emit,
            cancel_check=cancel_check,
        ),
    )
    return JSONResponse({"run_id": run_id, "project_slug": project.slug, "status": "queued"})


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


@router.get("/api/extraction/runs/active")
async def api_get_active_run(project_slug: str) -> JSONResponse:
    active = _active_run_for_project(project_slug)
    return JSONResponse(active or {"project_slug": project_slug, "run_id": None, "status": "idle"})


@router.get("/api/extraction/runs/latest")
async def api_get_latest_run(project_slug: str) -> JSONResponse:
    latest = _latest_run_for_project(project_slug)
    return JSONResponse(latest or {"project_slug": project_slug, "run_id": None, "status": "idle"})


@router.get("/api/extraction/runs/{run_id}")
async def api_get_run(run_id: str) -> JSONResponse:
    try:
        return JSONResponse(_read_run_meta(run_id))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@router.delete("/api/extraction/runs/{run_id}", status_code=204)
async def api_delete_run(run_id: str, force: bool = False) -> None:
    try:
        meta = _read_run_meta(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    status = str(meta.get("status") or "")
    if status in {"queued", "running", "cancel_requested"} and not force:
        raise HTTPException(status_code=409, detail="Cannot delete an active run")

    with _RUNS_LOCK:
        _RUN_THREADS.pop(run_id, None)
        _delete_run_files(run_id)


@router.post("/api/extraction/runs/{run_id}/abort")
async def api_abort_run(run_id: str) -> JSONResponse:
    try:
        meta = _read_run_meta(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    status = str(meta.get("status") or "")
    if status in {"done", "error", "canceled"}:
        return JSONResponse({"ok": True, "run_id": run_id, "status": status, "already_finished": True})
    if bool(meta.get("cancel_requested")) or status == "cancel_requested":
        return JSONResponse({"ok": True, "run_id": run_id, "status": "cancel_requested", "already_requested": True})

    updated = _update_run_meta(
        run_id,
        status="cancel_requested",
        cancel_requested=True,
        cancel_requested_at=datetime.now(timezone.utc).isoformat(),
    )
    _append_run_event(
        run_id,
        {
            "type": "abort_requested",
            "message": "Abort requested by user.",
            "phase": "error",
            "stage": "abort_requested",
            "percent_hint": None,
        },
    )
    return JSONResponse({"ok": True, "run_id": run_id, "status": updated.get("status", "cancel_requested")})


@router.get("/api/extraction/runs/{run_id}/events")
async def api_stream_run_events(run_id: str, cursor: int = 0) -> StreamingResponse:
    try:
        _read_run_meta(run_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return await _stream_existing_run(run_id, cursor)


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
    payload = project.to_dict() if hasattr(project, "to_dict") else project.__dict__
    latest_checkpoint = _latest_checkpoint_for_slug(slug)
    payload["latest_checkpoint_path"] = str(latest_checkpoint) if latest_checkpoint else None
    return JSONResponse(payload)


@router.get("/api/extraction/projects/{slug}/latest-checkpoint")
async def api_get_latest_checkpoint(slug: str) -> JSONResponse:
    latest_checkpoint = _latest_checkpoint_for_slug(slug)
    return JSONResponse({
        "slug": slug,
        "checkpoint_path": str(latest_checkpoint) if latest_checkpoint else None,
    })


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
        if p.name.endswith("_evidence_appendix.md"):
            continue
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
