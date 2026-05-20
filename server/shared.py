"""Shared state, constants, and helpers used across all server route modules.

This module is imported by every route module.  It runs first and sets up
``sys.path`` so that project-root imports work regardless of how the server
was started or packaged.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading as _threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── ensure project root is on the path ──────────────────────────────────────
import sys as _sys
if getattr(_sys, "frozen", False):
    from utils.frozen import get_install_dir as _get_install_dir, get_bundle_dir as _get_bundle_dir
    PROJECT_ROOT = _get_install_dir()
    _BUNDLE_ROOT = _get_bundle_dir()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    _BUNDLE_ROOT = PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.generation import (
    ollama_chat as _raw_ollama_chat,
    ollama_stream as _raw_ollama_stream,
)
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)

# Conversation retention — read once at startup; falls back to 90 days.
try:
    from utils.config import load_runtime_config as _load_runtime_config
    _conv_retention_days: int = int(
        (_load_runtime_config().get("conversations") or {}).get("retention_days", 90)
    )
except Exception:
    _conv_retention_days = 90

# ── Paths ─────────────────────────────────────────────────────────────────────
STATIC_DIR   = _BUNDLE_ROOT / "server" / "static"
CONTEXT_PATH = PROJECT_ROOT / "data" / "context.json"
FEEDBACK_DIR = PROJECT_ROOT / "data" / "feedback"

# ── GPU lock ──────────────────────────────────────────────────────────────────
# A single semaphore serialises all GPU-heavy jobs (ingest, large-doc extraction).
# Short query/chat calls do NOT acquire this lock.
#
# We store mutable state in a dict so that workers running in other modules can
# update ``holder`` without needing the ``global`` keyword (which only works
# within a single module).

_GPU_LOCK_VRAM_THRESHOLD_MB = int(os.environ.get("INFERENCE_VRAM_KEEP_MB", "1500"))

_gpu_state: Dict[str, Any] = {
    "lock":   _threading.Semaphore(1),
    "holder": None,   # human-readable label set by whichever worker holds the lock
}


def _ollama_free_vram_mb() -> int:
    """Return estimated free VRAM (MB) by polling Ollama /api/ps."""
    import json as _j, urllib.request as _ur
    try:
        ollama_url = DEFAULT_LLM_BASE_URL.rstrip("/")
        with _ur.urlopen(
            _ur.Request(f"{ollama_url}/api/ps", method="GET"), timeout=3
        ) as resp:
            data = _j.loads(resp.read())
        ollama_used = sum(m.get("size_vram", 0) for m in data.get("models", []))
    except Exception:
        ollama_used = 0
    try:
        import torch
        if torch.cuda.is_available():
            props   = torch.cuda.get_device_properties(0)
            total   = props.total_memory
            hf_used = torch.cuda.memory_allocated(0)
            return (total - ollama_used - hf_used) // (1024 * 1024)
    except Exception:
        pass
    return 99999  # assume plenty of room so we don't false-positive block


def _gpu_is_tight() -> bool:
    return _ollama_free_vram_mb() < _GPU_LOCK_VRAM_THRESHOLD_MB


def _gpu_busy_response() -> JSONResponse:
    holder = _gpu_state["holder"] or "a background job"
    return JSONResponse(
        status_code=503,
        content={"error": f"Server busy — {holder} is running. Please try again shortly."},
    )


# ── In-memory job / ingest state ──────────────────────────────────────────────
# Shared between upload and ingest-raw endpoints in routes/library.py.

_ingest_jobs: Dict[str, Dict[str, Any]] = {}

_raw_ingest_state: Dict[str, Any] = {
    "running":      False,
    "queued_files": [],
    "done_files":   [],
    "failed_files": [],
    "current_file": None,
    "error":        None,
}

# ── Personal context ──────────────────────────────────────────────────────────

def _load_personal_context() -> Dict[str, Any]:
    """Load the personal context document from disk. Returns {} if absent."""
    try:
        if CONTEXT_PATH.exists():
            return json.loads(CONTEXT_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _personal_context_prefix() -> str:
    """Format the personal context as a brief system-prompt prefix."""
    ctx = _load_personal_context()
    if not ctx:
        return ""
    parts: List[str] = []
    if ctx.get("name"):
        parts.append(f"The user's name is {ctx['name']}.")
    if ctx.get("role"):
        parts.append(f"They work as: {ctx['role']}.")
    if ctx.get("current_projects"):
        parts.append(f"Current projects: {ctx['current_projects']}.")
    if ctx.get("technologies"):
        parts.append(f"Technologies they use: {ctx['technologies']}.")
    if ctx.get("notes"):
        parts.append(f"Additional context: {ctx['notes']}.")
    if not parts:
        return ""
    return "[User context: " + " ".join(parts) + "]\n\n"


def _personal_system_prompt_override() -> str | None:
    """Return custom system prompt from context, or None if not set."""
    ctx = _load_personal_context()
    val = ctx.get("system_prompt", "")
    return str(val).strip() if val and str(val).strip() else None


def _with_context(base_prompt: str) -> str:
    """Prepend personal context prefix to a system prompt.

    If the user has set a custom system_prompt override, that replaces the
    base_prompt entirely (the user-context prefix is still prepended).
    """
    custom = _personal_system_prompt_override()
    effective_base = custom if custom else base_prompt
    prefix = _personal_context_prefix()
    if not prefix:
        return effective_base
    if effective_base.startswith(prefix):
        return effective_base
    return prefix + effective_base


# ── LLM wrappers — personal context always injected ──────────────────────────

def ollama_chat(*, model: str, system_prompt: str, user_prompt: str, **kwargs):
    """Drop-in wrapper that always prepends personal context to the system prompt."""
    return _raw_ollama_chat(
        model=model,
        system_prompt=_with_context(system_prompt),
        user_prompt=user_prompt,
        **kwargs,
    )


def ollama_stream(*, model: str, system_prompt: str, user_prompt: str, **kwargs):
    """Drop-in wrapper that always prepends personal context to the system prompt."""
    return _raw_ollama_stream(
        model=model,
        system_prompt=_with_context(system_prompt),
        user_prompt=user_prompt,
        **kwargs,
    )


# ── Routing trace log ─────────────────────────────────────────────────────────

def _log_routing_trace(
    trace_id: str,
    query: str,
    intent: str,
    mode: str,
    chunk_ids: List[str],
    answer: str,
    latency_seconds: float,
    conversation_id: Optional[str],
) -> None:
    """Append one routing trace entry to data/feedback/routing_trace.jsonl."""
    try:
        FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
        entry = {
            "trace_id":       trace_id,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
            "query":          query,
            "intent":         intent,
            "mode":           mode,
            "chunk_ids":      chunk_ids,
            "answer_summary": answer[:200],
            "latency_seconds": latency_seconds,
            "conversation_id": conversation_id,
        }
        with open(FEEDBACK_DIR / "routing_trace.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ── Chat prompts / web constants ──────────────────────────────────────────────

CHAT_SYSTEM_PROMPT = (
    "You are a knowledgeable, thoughtful personal AI assistant. "
    "Answer questions clearly and concisely. "
    "When you don't know something, say so honestly rather than guessing."
)

CHAT_WEB_SYSTEM_PROMPT = (
    "You are a knowledgeable, thoughtful personal AI assistant with access to "
    "live web search results. When web results are provided, use them to give "
    "accurate, up-to-date answers and cite the sources you relied on. "
    "If the web results don't answer the question, say so and answer from "
    "general knowledge. Be concise."
)

_INTERNET_TIMEOUT     = 8.0
_INTERNET_MAX_RESULTS = 4

# ── Query resolution helpers ──────────────────────────────────────────────────
# Detects implicit follow-ups ("what about the CTO?") and rewrites them by
# injecting the most recently mentioned entity from conversation history.

_NOT_ENTITY = frozenset({
    "The", "This", "That", "These", "Those", "When", "Where", "What", "Which",
    "While", "With", "From", "Into", "About", "More", "Some", "Also", "Then",
    "Than", "Just", "Even", "Other", "Like", "Well", "Based", "Given",
    "According", "However", "Therefore", "Thus", "Since", "After", "Before",
    "During", "Between", "Through", "Around", "Under", "Over", "Against",
    "Within", "Without", "Both", "Either", "Yes", "No", "Not", "But", "And",
    "Or", "Nor", "So", "Yet", "For", "Nor", "At", "By", "On", "In", "Of",
    "To", "Its", "Has", "Have", "Had", "Was", "Were", "Are", "Is", "Be",
    "He", "She", "They", "We", "You", "I", "It",
})

_ENTITY_RE = re.compile(r"\b([A-Z][A-Za-z]{1,}(?:\s+[A-Z][A-Za-z]{1,})*)\b")


def _extract_last_entity(history: List[Dict[str, str]]) -> Optional[str]:
    """Return the most recently mentioned proper noun from the last 3 messages."""
    for msg in reversed((history or [])[-3:]):
        text = msg.get("content", "")
        candidates = [
            m for m in _ENTITY_RE.findall(text)
            if m.split()[0] not in _NOT_ENTITY
            and len(m) > 2
        ]
        if candidates:
            return candidates[-1]
    return None


def _resolve_query(query: str, history: Optional[List[Dict[str, str]]]) -> str:
    """Rewrite implicit follow-up queries by injecting entity context from history."""
    if not history:
        return query

    q = query.strip()
    if len(q.split()) > 12:
        return query

    try:
        from retrieval.router import classify_intent  # noqa: PLC0415
        intent, _ = classify_intent(q)
        if intent != "implicit_followup":
            return query
    except Exception:
        return query

    entity = _extract_last_entity(history)
    if not entity:
        return query

    return f"Regarding {entity}: {q}"


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    force_rag: bool = False
    force_web: bool = False
    collection_id: Optional[str] = None
    doc_ids: List[str] = []
    history: List[Message] = []
    top_k: int = 5
    stream: bool = False
    conversation_id: Optional[str] = None
    # deprecated — kept for backward compat, ignored by unified pipeline
    mode: str = "chat"
    web_search: bool = False
    prior_intents: List[str] = []
    clarification_pending: bool = False


class ChatResponse(BaseModel):
    answer: str
    mode: str
    citations: List[Dict[str, Any]] = []
    routing: Optional[Dict[str, Any]] = None
    internet_used: bool = False
    web_sources: List[Dict[str, str]] = []
    elapsed_seconds: float = 0.0
    error: Optional[str] = None
    trace_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: int            # 1 = positive, -1 = negative
    query: Optional[str] = None
    answer: Optional[str] = None
    comment: Optional[str] = None
