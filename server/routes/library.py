"""Library endpoints: collections, documents, file download, upload, bulk ingest."""
from __future__ import annotations

import threading
import uuid
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
    _gpu_state,
    _gpu_busy_response,
    _gpu_is_tight,
    _ingest_jobs,
    _raw_ingest_state,
)

router = APIRouter()


# ── Pydantic models ───────────────────────────────────────────────────────────

class CreateCollectionBody(BaseModel):
    collection_id: str
    name: str
    description: Optional[str] = None
    parent_id: Optional[str] = None


class AssignCollectionBody(BaseModel):
    collection_id: Optional[str] = None  # None = unassign


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

@router.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    collection_id: Optional[str] = Form(None),
    source_type: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """Upload a file to data/raw/ and kick off ingest in a background thread."""
    import shutil

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename).name  # strip any path traversal
    dest = raw_dir / safe_name
    try:
        with dest.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    job_id = str(uuid.uuid4())
    _ingest_jobs[job_id] = {"status": "queued", "filename": safe_name}

    def _run_ingest() -> None:
        _ingest_jobs[job_id]["status"] = "running"
        _gpu_state["lock"].acquire()
        _gpu_state["holder"] = f"ingesting {safe_name}"
        try:
            from pipeline.ingest_v3 import ingest_file as _ingest_file
            result = _ingest_file(
                str(dest),
                db_dsn=DEFAULT_DB_DSN,
                collection_id=collection_id or None,
                source_type=source_type or None,
            )
            _ingest_jobs[job_id]["status"] = "done"
            _ingest_jobs[job_id]["result"] = str(result)
        except Exception as exc:
            _ingest_jobs[job_id]["status"] = "error"
            _ingest_jobs[job_id]["error"] = str(exc)
        finally:
            _gpu_state["holder"] = None
            _gpu_state["lock"].release()

    threading.Thread(target=_run_ingest, daemon=True).start()
    return {"job_id": job_id, "filename": safe_name, "status": "queued"}


@router.get("/api/upload/{job_id}")
def upload_status(job_id: str) -> Dict[str, Any]:
    job = _ingest_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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
