"""Library endpoints: collections, documents, file download, upload, bulk ingest."""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from db.client import (
    assign_to_collection,
    create_collection,
    delete_collection,
    delete_document,
    get_collection,
    list_collections,
    list_documents,
    list_unassigned_documents,
    unassign_from_collection,
)
from server.shared import (
    DEFAULT_DB_DSN,
    PROJECT_ROOT,
    delete_persisted_ingest_job,
    _gpu_state,
    _gpu_busy_response,
    _gpu_is_tight,
    _ingest_jobs,
    persist_ingest_job,
    _raw_ingest_state,
)

router = APIRouter()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_view(job_id: str, job: Dict[str, Any], queue_position: Optional[int]) -> Dict[str, Any]:
    status = str(job.get("status") or "")
    return {
        "job_id": job_id,
        "filename": job.get("filename"),
        "collection_id": job.get("collection_id"),
        "status": status,
        "error": job.get("error"),
        "dest_path": job.get("dest_path"),
        "created_at": job.get("created_at"),
        "queued_at": job.get("queued_at"),
        "run_started_at": job.get("run_started_at"),
        "queue_position": queue_position,
        "is_active": status == "running",
        "is_waiting": status == "queued",
        "gpu_holder": _gpu_state.get("holder"),
        "stage": job.get("stage"),
        "stage_detail": job.get("stage_detail"),
    }


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreateCollectionBody(BaseModel):
    collection_id: str
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None


class AssignCollectionBody(BaseModel):
    collection_id: Optional[str] = None  # None = unassign


class CreateUploadJobBody(BaseModel):
    filename: str
    collection_id: Optional[str] = None
    source_type: Optional[str] = None


# ── Collection endpoints ──────────────────────────────────────────────────────

@router.get("/api/collections")
def get_collections() -> List[Dict[str, Any]]:
    try:
        return list_collections(DEFAULT_DB_DSN)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/collections/{collection_id:path}")
def get_collection_detail(collection_id: str) -> Dict[str, Any]:
    try:
        info = get_collection(DEFAULT_DB_DSN, collection_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    if info is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    return info


@router.post("/api/collections", status_code=201)
def create_collection_endpoint(req: CreateCollectionBody) -> Dict[str, Any]:
    try:
        create_collection(
            DEFAULT_DB_DSN, req.collection_id, req.name,
            description=req.description, parent_id=req.parent_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"collection_id": req.collection_id, "name": req.name}


@router.delete("/api/collections/{collection_id:path}", status_code=204)
def delete_collection_endpoint(collection_id: str) -> None:
    try:
        delete_collection(DEFAULT_DB_DSN, collection_id, clear_chunks=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Document endpoints ────────────────────────────────────────────────────────

@router.get("/api/documents")
def get_documents(
    collection_id: Optional[str] = None,
    unassigned: bool = False,
) -> List[Dict[str, Any]]:
    try:
        if unassigned:
            return list_unassigned_documents(DEFAULT_DB_DSN)
        return list_documents(DEFAULT_DB_DSN, collection_id=collection_id or None)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.put("/api/documents/{doc_id}/collection", status_code=204)
def set_document_collection(doc_id: str, body: AssignCollectionBody) -> None:
    try:
        if body.collection_id:
            assign_to_collection(DEFAULT_DB_DSN, body.collection_id, doc_ids=[doc_id])
        else:
            unassign_from_collection(DEFAULT_DB_DSN, [doc_id])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.delete("/api/documents/{doc_id}", status_code=204)
def delete_document_endpoint(doc_id: str) -> None:
    try:
        delete_document(DEFAULT_DB_DSN, doc_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/documents/{doc_id}/file")
def download_document_file(doc_id: str) -> FileResponse:
    from db.client import _connect, init_db
    init_db(DEFAULT_DB_DSN)
    conn = _connect(DEFAULT_DB_DSN)
    try:
        row = conn.execute(
            "SELECT filename, source_path FROM documents WHERE doc_id = %s", (doc_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Document not found")
    source_path = Path(row["source_path"])
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Source file not found on disk")
    return FileResponse(
        path=str(source_path),
        filename=row["filename"],
        media_type="application/octet-stream",
    )


# ── Upload (single file) ──────────────────────────────────────────────────────

@router.post("/api/upload-init")
def create_upload_job(req: CreateUploadJobBody) -> Dict[str, Any]:
    safe_name = Path(req.filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Filename is required")
    job_id = str(uuid.uuid4())
    dest_path = PROJECT_ROOT / "data" / "raw" / safe_name
    _ingest_jobs[job_id] = {
        "status": "uploading",
        "filename": safe_name,
        "collection_id": req.collection_id or None,
        "source_type": req.source_type or None,
        "dest_path": str(dest_path),
        "created_at": _utc_now_iso(),
        "stage": "uploading",
        "stage_detail": "Waiting for file transfer",
    }
    persist_ingest_job(job_id)
    return {"job_id": job_id, "filename": safe_name, "status": "uploading"}

@router.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),
    job_id: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a file to data/raw/ and kick off ingest in a background thread."""
    import shutil

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any path traversal
    dest = raw_dir / safe_name
    resolved_job_id = job_id or str(uuid.uuid4())
    existing_job = _ingest_jobs.get(resolved_job_id)
    if existing_job is None:
        _ingest_jobs[resolved_job_id] = {
            "status": "uploading",
            "filename": safe_name,
            "collection_id": collection_id or None,
            "source_type": source_type or None,
            "dest_path": str(dest),
            "created_at": _utc_now_iso(),
            "stage": "uploading",
            "stage_detail": "Receiving file upload",
        }
    else:
        existing_job["status"] = "uploading"
        existing_job["filename"] = safe_name
        existing_job["collection_id"] = collection_id or existing_job.get("collection_id")
        existing_job["source_type"] = source_type or existing_job.get("source_type")
        existing_job["dest_path"] = str(dest)
        existing_job.setdefault("created_at", _utc_now_iso())
        existing_job["stage"] = "uploading"
        existing_job["stage_detail"] = "Receiving file upload"
    persist_ingest_job(resolved_job_id)
    try:
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
        _ingest_jobs[resolved_job_id]["status"] = "queued"
        _ingest_jobs[resolved_job_id]["queued_at"] = _utc_now_iso()
        _ingest_jobs[resolved_job_id]["stage"] = "queued"
        _ingest_jobs[resolved_job_id]["stage_detail"] = "Waiting for ingest worker"
        persist_ingest_job(resolved_job_id)
    except Exception as exc:
        _ingest_jobs[resolved_job_id]["status"] = "error"
        _ingest_jobs[resolved_job_id]["error"] = str(exc)
        _ingest_jobs[resolved_job_id]["stage"] = "error"
        _ingest_jobs[resolved_job_id]["stage_detail"] = "Upload failed"
        persist_ingest_job(resolved_job_id)
        raise
    finally:
        await file.close()

    def _run_ingest() -> None:
        _ingest_jobs[resolved_job_id]["status"] = "running"
        _ingest_jobs[resolved_job_id]["run_started_at"] = _utc_now_iso()
        _ingest_jobs[resolved_job_id]["stage"] = "waiting_for_gpu"
        _ingest_jobs[resolved_job_id]["stage_detail"] = "Waiting for GPU slot"
        persist_ingest_job(resolved_job_id)
        _gpu_state["lock"].acquire()
        _gpu_state["holder"] = f"ingesting {safe_name}"
        try:
            from pipeline.ingest_v3 import ingest_file as _ingest_file

            def _progress(stage: str, detail: str) -> None:
                job = _ingest_jobs.get(resolved_job_id)
                if not job:
                    return
                job["status"] = "running"
                job["stage"] = stage
                job["stage_detail"] = detail
                persist_ingest_job(resolved_job_id)

            result = _ingest_file(
                str(dest),
                db_dsn=DEFAULT_DB_DSN,
                collection_id=collection_id or None,
                source_type=source_type or None,
                progress_cb=_progress,
            )
            _ingest_jobs[resolved_job_id]["status"] = "done"
            _ingest_jobs[resolved_job_id]["result"] = str(result)
            _ingest_jobs[resolved_job_id]["completed_at"] = _utc_now_iso()
            _ingest_jobs[resolved_job_id]["stage"] = "completed"
            _ingest_jobs[resolved_job_id]["stage_detail"] = "Ingest complete"
            persist_ingest_job(resolved_job_id)
        except Exception as exc:
            _ingest_jobs[resolved_job_id]["status"] = "error"
            _ingest_jobs[resolved_job_id]["error"] = str(exc)
            _ingest_jobs[resolved_job_id]["completed_at"] = _utc_now_iso()
            _ingest_jobs[resolved_job_id]["stage"] = "error"
            _ingest_jobs[resolved_job_id]["stage_detail"] = "Ingest failed"
            persist_ingest_job(resolved_job_id)
        finally:
            _gpu_state["holder"] = None
            _gpu_state["lock"].release()

    threading.Thread(target=_run_ingest, daemon=True).start()
    return {"job_id": resolved_job_id, "filename": safe_name, "status": "queued"}


@router.get("/api/upload/{job_id}")
def upload_status(job_id: str) -> Dict[str, Any]:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    status = str(job.get("status") or "")
    queue_position: Optional[int] = None
    if status == "running":
        queue_position = 0
    elif status == "queued":
        queued_jobs = sorted(
            (
                (jid, j)
                for jid, j in _ingest_jobs.items()
                if str(j.get("status") or "") == "queued"
            ),
            key=lambda item: str(item[1].get("queued_at") or item[1].get("created_at") or ""),
        )
        for idx, (jid, _) in enumerate(queued_jobs, start=1):
            if jid == job_id:
                queue_position = idx
                break
    return _job_view(job_id, job, queue_position)


@router.delete("/api/upload/{job_id}", status_code=204)
def delete_upload_job(job_id: str) -> None:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    status = str(job.get("status") or "")
    if status in {"uploading", "queued", "running"}:
        raise HTTPException(status_code=409, detail="Cannot remove an active upload job")
    delete_persisted_ingest_job(job_id)


@router.get("/api/upload-active")
def list_active_uploads() -> List[Dict[str, Any]]:
    active_statuses = {"uploading", "queued", "running"}
    jobs: List[tuple[str, Dict[str, Any]]] = []
    for job_id, job in _ingest_jobs.items():
        status = str(job.get("status") or "")
        if status in active_statuses:
            jobs.append((job_id, job))

    def _job_sort_key(item: tuple[str, Dict[str, Any]]) -> tuple[int, str]:
        _, job = item
        status = str(job.get("status") or "")
        if status == "running":
            rank = 0
            ts = str(job.get("run_started_at") or job.get("queued_at") or job.get("created_at") or "")
        elif status == "queued":
            rank = 1
            ts = str(job.get("queued_at") or job.get("created_at") or "")
        else:  # uploading
            rank = 2
            ts = str(job.get("created_at") or "")
        return rank, ts

    jobs.sort(key=_job_sort_key)
    queued_counter = 0
    out: List[Dict[str, Any]] = []
    for job_id, job in jobs:
        status = str(job.get("status") or "")
        queue_position: Optional[int] = None
        if status == "running":
            queue_position = 0
        elif status == "queued":
            queued_counter += 1
            queue_position = queued_counter
        out.append(_job_view(job_id, job, queue_position))
    return out


# ── Bulk ingest from data/raw/ ────────────────────────────────────────────────

@router.post("/api/ingest-raw")
def ingest_raw_start() -> Dict[str, Any]:
    """Scan data/raw/ for new files and ingest them all."""
    import psycopg

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    supported = {".pdf", ".docx", ".txt", ".md"}

    try:
        with psycopg.connect(DEFAULT_DB_DSN) as conn:
            rows = conn.execute("SELECT source_path FROM documents").fetchall()
            already_ingested = {r[0] for r in rows}
    except Exception:
        already_ingested = set()

    already_queued = set(
        _raw_ingest_state["queued_files"]
        + _raw_ingest_state["done_files"]
        + _raw_ingest_state["failed_files"]
        + ([_raw_ingest_state["current_file"]] if _raw_ingest_state["current_file"] else [])
    )

    new_files = [
        p for p in sorted(raw_dir.iterdir())
        if p.is_file()
        and p.suffix.lower() in supported
        and str(p.resolve()) not in already_ingested
        and p.name not in already_queued
    ]

    if not new_files:
        if _raw_ingest_state["running"]:
            return {
                "status":       "already_running",
                "current_file": _raw_ingest_state["current_file"],
                "queued":       len(_raw_ingest_state["queued_files"]),
            }
        return {"status": "nothing_to_do", "message": "All files in data/raw/ are already ingested."}

    if _raw_ingest_state["running"]:
        queue_pos_before = len(_raw_ingest_state["queued_files"])
        _raw_ingest_state["queued_files"].extend(p.name for p in new_files)
        return {
            "status":         "queued",
            "files":          [p.name for p in new_files],
            "count":          len(new_files),
            "queue_position": queue_pos_before + 1,
        }

    _raw_ingest_state.update({
        "running":      True,
        "queued_files": [p.name for p in new_files],
        "done_files":   [],
        "failed_files": [],
        "current_file": None,
        "error":        None,
    })

    def _worker() -> None:
        _gpu_state["lock"].acquire()
        _gpu_state["holder"] = "bulk ingest from data/raw"
        try:
            from pipeline.ingest_v3 import ingest_file as _iv3  # noqa: PLC0415
            raw_dir_w = PROJECT_ROOT / "data" / "raw"
            while _raw_ingest_state["queued_files"]:
                fname = _raw_ingest_state["queued_files"][0]
                _raw_ingest_state["current_file"] = fname
                file_path = raw_dir_w / fname
                try:
                    _iv3(str(file_path), db_dsn=DEFAULT_DB_DSN)
                    _raw_ingest_state["done_files"].append(fname)
                except Exception as exc:
                    _raw_ingest_state["failed_files"].append(fname)
                    _raw_ingest_state["error"] = f"{fname}: {exc}"
                finally:
                    if _raw_ingest_state["queued_files"] and _raw_ingest_state["queued_files"][0] == fname:
                        _raw_ingest_state["queued_files"].pop(0)
        finally:
            _raw_ingest_state["running"] = False
            _raw_ingest_state["current_file"] = None
            _gpu_state["holder"] = None
            _gpu_state["lock"].release()

    threading.Thread(target=_worker, daemon=True).start()
    return {
        "status": "started",
        "files":  [p.name for p in new_files],
        "count":  len(new_files),
    }


@router.get("/api/ingest-raw/status")
def ingest_raw_status() -> Dict[str, Any]:
    return dict(_raw_ingest_state)
