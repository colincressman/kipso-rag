"""New server entry point — refactored from server/server.py.

Start with:
    python main.py serve
or directly:
    .venv\\Scripts\\uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload

Then open http://localhost:8000
"""
from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# shared must be imported first so PROJECT_ROOT / sys.path are set before anything else
from server.shared import (
    PROJECT_ROOT,
    STATIC_DIR,
    DEFAULT_LLM_BASE_URL,
    _conv_retention_days,
    _gpu_state,
    _raw_ingest_state,
)
from db.client import archive_stale_conversations


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Warm up services at startup."""
    import asyncio
    import threading as _threading

    # Ensure all required runtime directories exist.
    try:
        from utils.dirs import ensure_runtime_dirs
        ensure_runtime_dirs(PROJECT_ROOT)
    except Exception as exc:
        logging.getLogger(__name__).warning("ensure_runtime_dirs failed (non-fatal): %s", exc)

    # Warm up Ollama embedding model.
    try:
        from pipeline.embed.embedder import create_embedder
        from utils.runtime_defaults import DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME

        def _embed_warmup():
            emb = create_embedder(backend=DEFAULT_EMBED_BACKEND, model_name=DEFAULT_EMBED_MODEL_NAME)
            resp = emb._post_json("/api/embed", {
                "model":      DEFAULT_EMBED_MODEL_NAME,
                "input":      ["warmup"],
                "keep_alive": 300,
            })
            try:
                embeddings = resp.get("embeddings") if isinstance(resp, dict) else None
                if isinstance(embeddings, list) and embeddings:
                    from utils.embed_meta import detect_and_save as _meta_save
                    _meta_save(DEFAULT_EMBED_MODEL_NAME, len(embeddings[0]))
            except Exception:
                pass

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _embed_warmup)
        logging.getLogger(__name__).info("Embedding model warmed up.")
    except Exception as exc:
        logging.getLogger(__name__).warning("Embedding warmup failed (non-fatal): %s", exc)

    # Pre-load BM25 index in background.
    from utils.runtime_defaults import DEFAULT_DB_DSN

    def _bm25_warmup():
        try:
            from retrieval.query import warm_bm25_index
            warm_bm25_index(DEFAULT_DB_DSN)
        except Exception as exc:
            logging.getLogger(__name__).warning("BM25 warmup failed (non-fatal): %s", exc)

    _threading.Thread(target=_bm25_warmup, daemon=True, name="bm25-warmup").start()

    # Start background job worker.
    try:
        from server.job_worker import start_job_worker
        start_job_worker(DEFAULT_DB_DSN)
        logging.getLogger(__name__).info("Background job worker started.")
    except Exception as exc:
        logging.getLogger(__name__).warning("Job worker failed to start (non-fatal): %s", exc)

    # Prune stale conversations.
    try:
        _pruned = archive_stale_conversations(DEFAULT_DB_DSN, days=_conv_retention_days)
        if _pruned:
            logging.getLogger(__name__).info(
                "Pruned %d stale conversation(s) (retention=%d days).", _pruned, _conv_retention_days
            )
    except Exception as exc:
        logging.getLogger(__name__).warning("Conversation pruning failed (non-fatal): %s", exc)

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    import time as _time
    _log = logging.getLogger(__name__)
    _log.info("Server shutdown initiated.")

    if _raw_ingest_state.get("running"):
        _deadline = _time.monotonic() + 5.0
        while _raw_ingest_state.get("running") and _time.monotonic() < _deadline:
            _time.sleep(0.25)
        if _raw_ingest_state.get("running"):
            _log.warning(
                "Ingest worker still running at shutdown (current file: %s); "
                "allowing daemon thread to exit with process.",
                _raw_ingest_state.get("current_file"),
            )

    _log.info("Shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Personal AI", version="1.0.0", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional API-key gate — set RAG_API_KEY env var to require X-API-Key header.
_RAG_API_KEY: str = os.environ.get("RAG_API_KEY", "").strip()


@app.middleware("http")
async def _api_key_gate(request: Request, call_next):
    if _RAG_API_KEY and request.url.path.startswith("/api/"):
        if request.headers.get("X-API-Key", "") != _RAG_API_KEY:
            return JSONResponse({"detail": "Unauthorized — set X-API-Key header"}, status_code=401)
    return await call_next(request)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Static pages ──────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/library", include_in_schema=False)
def serve_library() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "library.html"))


@app.get("/extraction", include_in_schema=False)
async def extraction_ui() -> FileResponse:
    p = STATIC_DIR / "extraction.html"
    if not p.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="extraction.html not found")
    return FileResponse(str(p))


@app.get("/summarize", include_in_schema=False)
async def summarize_ui() -> FileResponse:
    p = STATIC_DIR / "summarize.html"
    if not p.exists():
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="summarize.html not found")
    return FileResponse(str(p))


# ── Register routers ──────────────────────────────────────────────────────────

from server.routes import admin, chat, conversations, extraction, library, summarize

app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(conversations.router)
app.include_router(extraction.router)
app.include_router(library.router)
app.include_router(summarize.router)


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Personal AI — RAG server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run("server.app:app", host=args.host, port=args.port, reload=args.reload)
