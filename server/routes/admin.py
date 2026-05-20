"""Admin / system routes: health, status, GPU, inference-service, feedback, context."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from server.shared import (
    DEFAULT_DB_DSN,
    CONTEXT_PATH,
    FEEDBACK_DIR,
    _gpu_state,
    _GPU_LOCK_VRAM_THRESHOLD_MB,
    _ollama_free_vram_mb,
    _gpu_is_tight,
    _load_personal_context,
    FeedbackRequest,
)

router = APIRouter()


@router.get("/api/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "db": DEFAULT_DB_DSN}


@router.get("/api/status")
def api_status() -> Dict[str, Any]:
    """Return server + VRAM status for the launcher GUI stats panel."""
    from utils.vram_manager import get_manager as _get_vram   # noqa: PLC0415
    from utils.runtime_defaults import DEFAULT_OLLAMA_BASE_URL  # noqa: PLC0415

    vram = _get_vram().get_status(ollama_base_url=DEFAULT_OLLAMA_BASE_URL)

    db_stats: Dict[str, Any] = {}
    try:
        import psycopg as _psycopg
        from psycopg.rows import dict_row as _dict_row
        with _psycopg.connect(DEFAULT_DB_DSN, row_factory=_dict_row) as conn:
            row  = conn.execute("SELECT COUNT(*) AS chunks FROM chunks").fetchone()
            row2 = conn.execute("SELECT COUNT(*) AS docs FROM documents").fetchone()
            db_stats["chunks"]    = row["chunks"]    if row  else 0
            db_stats["documents"] = row2["docs"]     if row2 else 0
    except Exception:
        pass

    return {"status": "ok", **vram, "db": db_stats}


@router.get("/api/gpu-status")
def gpu_status() -> Dict[str, Any]:
    """Return GPU lock state and VRAM headroom — used by the launcher."""
    lock = _gpu_state["lock"]
    return {
        "gpu_lock_held":    lock._value == 0,        # type: ignore[attr-defined]
        "gpu_lock_holder":  _gpu_state["holder"],
        "free_vram_mb":     _ollama_free_vram_mb(),
        "vram_threshold_mb": _GPU_LOCK_VRAM_THRESHOLD_MB,
        "queries_blocked":  not lock._value and _gpu_is_tight(),  # type: ignore[attr-defined]
    }


# ── Inference service ─────────────────────────────────────────────────────────

@router.get("/api/inference-service/status")
async def api_inference_service_status() -> Any:
    from utils.service_discovery import get_cached_status, probe_now, get_inference_url  # noqa: PLC0415
    cached = get_cached_status()
    url = cached["url"] or ""
    if not url:
        url = get_inference_url()
    if url:
        caps = probe_now(url)
        connected = bool(caps)
    else:
        caps = {}
        connected = False
    return {
        "url":              url,
        "connected":        connected,
        "capabilities":     caps,
        "cache_expires_in": cached.get("cache_expires_in", 0),
    }


@router.post("/api/inference-service/configure")
async def api_inference_service_configure(body: Dict[str, Any]) -> Any:
    from utils.service_discovery import configure  # noqa: PLC0415
    url  = str(body.get("url", "")).strip()
    caps = configure(url)
    return {"url": url, "connected": bool(caps), "capabilities": caps}


@router.post("/api/inference-service/discover")
async def api_inference_service_discover() -> Any:
    import asyncio
    from utils.service_discovery import scan_now, probe_now  # noqa: PLC0415
    loop = asyncio.get_event_loop()
    found = await loop.run_in_executor(None, scan_now)
    if found:
        caps = probe_now(found)
        return {"url": found, "connected": True, "capabilities": caps}
    return     {"url": "",    "connected": False, "capabilities": {}}


# ── Feedback ──────────────────────────────────────────────────────────────────

@router.post("/api/feedback", status_code=204)
def submit_feedback(req: FeedbackRequest) -> None:
    """Record a thumbs-up / thumbs-down rating for a previous answer."""
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "trace_id":       req.trace_id,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "rating":         req.rating,
            "query":          req.query,
            "answer_summary": (req.answer or "")[:200],
            "comment":        req.comment,
        }
        with open(FEEDBACK_DIR / "corrections.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Personal context ──────────────────────────────────────────────────────────

@router.get("/api/context")
def get_context() -> Dict[str, Any]:
    return _load_personal_context()


@router.put("/api/context", status_code=204)
def save_context(body: Dict[str, Any]) -> None:
    try:
        CONTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONTEXT_PATH.write_text(
            json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Welcome chips ─────────────────────────────────────────────────────────────

_DEFAULT_CHIPS = [
    {"label": "Supervised vs unsupervised learning",
     "prompt": "What's the difference between supervised and unsupervised learning?"},
    {"label": "Explain reinforcement learning",
     "prompt": "Explain reinforcement learning in simple terms."},
    {"label": "Attention in transformers",
     "prompt": "What is the attention mechanism in transformers?"},
    {"label": "How does backpropagation work?",
     "prompt": "How does backpropagation work?"},
]


@router.get("/api/welcome-chips")
def get_welcome_chips() -> Dict[str, Any]:
    """Return the list of welcome chip {label, prompt} pairs from runtime.yaml."""
    try:
        from utils.config import load_runtime_config  # noqa: PLC0415
        cfg = load_runtime_config()
        chips = cfg.get("welcome_chips") or []
        if chips and isinstance(chips, list):
            return {"chips": chips}
    except Exception:
        pass
    return {"chips": _DEFAULT_CHIPS}
