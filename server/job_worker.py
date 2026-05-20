"""Background job worker — polls the 'jobs' table every 5 seconds.

Dispatchers by job_type:
  'summarize'  — calls generate_doc_summary() from llm/summarize.py
  'ingest'     — (reserved; not yet implemented)

The worker thread is started by server/app.py during the lifespan startup.
It is a daemon thread so it dies automatically when the server exits.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Any, Dict

logger = logging.getLogger(__name__)

_POLL_INTERVAL = 5  # seconds between polls when queue is empty
_paused = False
_pause_lock = threading.Lock()


def pause_worker() -> None:
    global _paused
    with _pause_lock:
        _paused = True
    logger.info("Job worker paused.")


def resume_worker() -> None:
    global _paused
    with _pause_lock:
        _paused = False
    logger.info("Job worker resumed.")


def is_worker_paused() -> bool:
    with _pause_lock:
        return _paused


def _dispatch(job_type: str, params: Dict[str, Any], db_dsn: str) -> None:
    """Run the work for a claimed job.  Raises on failure."""
    if job_type == "summarize":
        from llm.summarize import generate_doc_summary  # noqa: PLC0415
        from db.client import list_documents  # noqa: PLC0415
        doc_id: str = params["doc_id"]
        # Resolve document metadata so the stored summary chunk has a proper title
        docs = list_documents(db_dsn)
        doc_meta = next((d for d in docs if d["doc_id"] == doc_id), {})
        filename = doc_meta.get("filename", "")
        filename_stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        import re  # noqa: PLC0415
        raw_title = doc_meta.get("title") or doc_meta.get("document_title") or ""
        if not raw_title or re.fullmatch(r"[0-9a-f]{40,}", raw_title.strip()):
            title = filename_stem or doc_id
        else:
            title = raw_title
        generate_doc_summary(
            doc_id,
            db_dsn=db_dsn,
            document_title=title,
            collection_id=doc_meta.get("collection_id") or "",
            source_name=doc_meta.get("source_name") or title,
            document_path=doc_meta.get("source_path") or "",
            source_type=doc_meta.get("source_type") or "pdf_book",
        )
    elif job_type == "summarize_page_range":
        from llm.summarize import generate_page_range_summary  # noqa: PLC0415
        from db.client import list_documents  # noqa: PLC0415
        doc_id = params["doc_id"]
        page_start = int(params["page_start"])
        page_end = int(params["page_end"])
        docs = list_documents(db_dsn)
        doc_meta = next((d for d in docs if d["doc_id"] == doc_id), {})
        filename = doc_meta.get("filename", "")
        filename_stem = filename.rsplit(".", 1)[0] if "." in filename else filename
        import re  # noqa: PLC0415
        raw_title = doc_meta.get("title") or doc_meta.get("document_title") or ""
        if not raw_title or re.fullmatch(r"[0-9a-f]{40,}", raw_title.strip()):
            title = filename_stem or doc_id
        else:
            title = raw_title
        generate_page_range_summary(
            doc_id,
            page_start,
            page_end,
            db_dsn=db_dsn,
            document_title=title,
            collection_id=doc_meta.get("collection_id") or "",
            source_name=doc_meta.get("source_name") or title,
            document_path=doc_meta.get("source_path") or "",
            source_type=doc_meta.get("source_type") or "pdf_book",
        )
    elif job_type == "ingest":
        # Placeholder — future PDF ingest via job queue
        raise NotImplementedError(f"ingest jobs not yet implemented")
    else:
        raise ValueError(f"Unknown job_type: {job_type!r}")


def _worker_loop(db_dsn: str) -> None:
    from db.jobs import claim_next_job, complete_job, fail_job  # noqa: PLC0415

    logger.info("Job worker started (polling every %ds).", _POLL_INTERVAL)
    while True:
        try:
            if is_worker_paused():
                time.sleep(_POLL_INTERVAL)
                continue
            job = claim_next_job(db_dsn)
            if job is None:
                time.sleep(_POLL_INTERVAL)
                continue

            job_id: str = job["job_id"]
            job_type: str = job["job_type"]
            params: Dict[str, Any] = job.get("params") or {}
            logger.info("Running job %s (type=%s)", job_id, job_type)

            try:
                _dispatch(job_type, params, db_dsn)
                complete_job(db_dsn, job_id)
                logger.info("Job %s completed.", job_id)
            except Exception as exc:
                error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                logger.error("Job %s failed: %s", job_id, error_msg)
                fail_job(db_dsn, job_id, error_msg, retry=True)
                # Brief back-off before claiming next job
                time.sleep(2)

        except Exception:
            logger.exception("Unexpected error in job worker loop; retrying in %ds.", _POLL_INTERVAL)
            time.sleep(_POLL_INTERVAL)


def start_job_worker(db_dsn: str) -> threading.Thread:
    """Start the background worker thread.  Returns the thread object."""
    t = threading.Thread(target=_worker_loop, args=(db_dsn,), daemon=True, name="job-worker")
    t.start()
    return t
