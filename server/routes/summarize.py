"""Book Summarizer API endpoints."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.client import (
    get_docs_without_summary,
    get_page_range_summary_text,
    get_summary_text,
    init_db,
    list_documents,
    list_page_range_summaries,
)
from db.jobs import cancel_job, enqueue_job, get_job, list_jobs
from server.job_worker import is_worker_paused, pause_worker, resume_worker
from server.shared import DEFAULT_DB_DSN

router = APIRouter(prefix="/api/summarize", tags=["summarize"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class RunSummaryBody(BaseModel):
    doc_id: str


class RunRangeSummaryBody(BaseModel):
    doc_id: str
    page_start: int
    page_end: int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/documents")
async def list_summarize_documents() -> List[Dict[str, Any]]:
    """Return all documents with a 'has_summary' boolean flag."""
    init_db(DEFAULT_DB_DSN)
    all_docs = list_documents(DEFAULT_DB_DSN)
    no_summary_ids = {d["doc_id"] for d in get_docs_without_summary(DEFAULT_DB_DSN)}
    result = []
    for d in all_docs:
        filename = d.get("filename", "")
        filename_stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        raw_title = d.get("document_title") or d.get("title") or ""
        # If raw_title looks like a SHA-256 hash, prefer the filename stem
        if not raw_title or re.fullmatch(r"[0-9a-f]{40,}", raw_title.strip()):
            display_title = filename_stem or raw_title or d["doc_id"]
        else:
            display_title = raw_title
        result.append({
            "doc_id": d["doc_id"],
            "title": display_title,
            "filename": filename,
            "collection_id": d.get("collection_id", ""),
            "source_type": d.get("source_type", ""),
            "has_summary": d["doc_id"] not in no_summary_ids,
        })
    return result


@router.post("/run")
async def run_summarize(body: RunSummaryBody) -> Dict[str, Any]:
    """Enqueue a summarization job for a document.  Returns the job_id."""
    init_db(DEFAULT_DB_DSN)
    job_id = enqueue_job(DEFAULT_DB_DSN, "summarize", {"doc_id": body.doc_id})
    return {"job_id": job_id, "status": "pending"}


@router.get("/jobs")
async def get_jobs(status: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Return recent summarize jobs."""
    init_db(DEFAULT_DB_DSN)
    all_jobs = list_jobs(DEFAULT_DB_DSN, status=status, limit=limit)
    return [j for j in all_jobs if j.get("job_type") in ("summarize", "summarize_page_range")]


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    """Return status of a single job."""
    init_db(DEFAULT_DB_DSN)
    job = get_job(DEFAULT_DB_DSN, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.delete("/jobs/{job_id}")
async def cancel_job_endpoint(job_id: str) -> Dict[str, Any]:
    """Cancel a pending or running job."""
    init_db(DEFAULT_DB_DSN)
    ok = cancel_job(DEFAULT_DB_DSN, job_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Job not found or already finished")
    return {"job_id": job_id, "status": "cancelled"}


@router.get("/worker/status")
async def get_worker_status() -> Dict[str, Any]:
    return {"paused": is_worker_paused()}


@router.post("/worker/pause")
async def worker_pause() -> Dict[str, Any]:
    pause_worker()
    return {"paused": True}


@router.post("/worker/resume")
async def worker_resume() -> Dict[str, Any]:
    resume_worker()
    return {"paused": False}


@router.get("/text/{doc_id}")
async def get_doc_summary_text(doc_id: str) -> Dict[str, Any]:
    """Return the stored summary text for a document."""
    init_db(DEFAULT_DB_DSN)
    text = get_summary_text(DEFAULT_DB_DSN, doc_id)
    if text is None:
        raise HTTPException(status_code=404, detail="No summary found for this document")
    return {"doc_id": doc_id, "text": text}


# ── Page-range endpoints ──────────────────────────────────────────────────────

@router.post("/run_range")
async def run_range_summarize(body: RunRangeSummaryBody) -> Dict[str, Any]:
    """Enqueue a page-range summarization job."""
    if body.page_start < 0 or body.page_end < body.page_start:
        raise HTTPException(status_code=422, detail="page_end must be >= page_start >= 0")
    init_db(DEFAULT_DB_DSN)
    job_id = enqueue_job(DEFAULT_DB_DSN, "summarize_page_range", {
        "doc_id": body.doc_id,
        "page_start": body.page_start,
        "page_end": body.page_end,
    })
    return {"job_id": job_id, "status": "pending"}


@router.get("/ranges")
async def list_all_ranges() -> List[Dict[str, Any]]:
    """Return all page-range summaries across all documents."""
    init_db(DEFAULT_DB_DSN)
    return list_page_range_summaries(DEFAULT_DB_DSN)


@router.get("/ranges/{doc_id}")
async def list_doc_ranges(doc_id: str) -> List[Dict[str, Any]]:
    """Return all page-range summaries stored for a document."""
    init_db(DEFAULT_DB_DSN)
    return list_page_range_summaries(DEFAULT_DB_DSN, doc_id)


@router.get("/range_text/{doc_id}/{page_start}/{page_end}")
async def get_range_text(doc_id: str, page_start: int, page_end: int) -> Dict[str, Any]:
    """Return summary text for a specific page range."""
    init_db(DEFAULT_DB_DSN)
    text = get_page_range_summary_text(DEFAULT_DB_DSN, doc_id, page_start, page_end)
    if text is None:
        raise HTTPException(status_code=404, detail="No summary found for this range")
    return {"doc_id": doc_id, "page_start": page_start, "page_end": page_end, "text": text}
