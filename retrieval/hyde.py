"""
HyDE — Hypothetical Document Embeddings.

Instead of embedding the user's *question*, we ask a small LLM to draft a
short hypothetical passage that *would* answer the question and embed that
instead.  Because answer-like text sits closer in embedding space to real
answer passages than a bare question does, vector recall improves — especially
for declarative technical corpora (textbooks, papers, etc.).

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
Relevance Labels"  https://arxiv.org/abs/2212.10496

Usage::

    passage, trace = generate_hyde_query(
        "What is the vanishing gradient problem?",
        model="qwen2.5:3b-instruct",
        base_url="http://localhost:11434",
    )
    # embed `passage` instead of the original query
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "You are a technical writing assistant. "
    "Write a concise passage (2-4 sentences) from a reference document or textbook "
    "that directly answers the question below. "
    "Write only the passage itself — no preamble, no labels, no quotation marks."
)

_SEARCH_REWRITE_SYSTEM = (
    "You are a search query assistant. "
    "Given a user question and a short passage that answers it, "
    "write the best web search query (5 words or fewer) to find that answer. "
    "Reply with ONLY the search query — no explanation, no punctuation, no quotes."
)


def _rewrite_as_search_query(
    query: str,
    passage: str,
    *,
    model: str,
    url: str,
    timeout_seconds: float,
) -> str:
    """Second LLM call while the model is still warm."""
    user_msg = f"Question: {query}\nPassage: {passage}"
    payload = json.dumps({
        "model": model,
        "stream": False,
        "keep_alive": -1,  # keep pinned — inference service startup already set this
        "think": False,
        "options": {"temperature": 0.0, "num_predict": 20, "num_ctx": 512},
        "messages": [
            {"role": "system", "content": _SEARCH_REWRITE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    result = ((body.get("message") or {}).get("content") or "").strip().strip("\"'")
    return result if (result and len(result) <= 120) else query


def generate_hyde_query(
    query: str,
    *,
    model: str,
    base_url: str,
    timeout_seconds: float = 15.0,
    temperature: float = 0.4,
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate a hypothetical answer passage for a user query.

    Returns ``(embed_text, trace)`` where ``embed_text`` is either the
    generated passage (on success) or the original query (on any failure).
    Retrieval is therefore never blocked by HyDE errors.

    Trace keys:
        enabled (bool)       — always True when this function is called
        applied (bool)       — True if a passage was successfully generated
        passage (str)        — generated text (only when applied=True)
        latency_seconds (float)
        model (str)
        reason (str)         — failure reason (only when applied=False)
        error (str)          — exception message (only on llm_error)
    """
    if not (query or "").strip():
        return query, {"enabled": True, "applied": False, "reason": "empty_query"}

    # Free the intent classifier from VRAM before the LLM loads.
    try:
        from retrieval.intent_classifier import unload as _unload_classifier
        _unload_classifier()
    except Exception:
        pass

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": -1,  # keep pinned in VRAM — inference service startup maintains this
        "think": False,  # disable chain-of-thought — we want direct output
        "options": {
            "temperature": temperature,
            "num_predict": 160,  # short passage — enough substance, not too broad
            "num_ctx": 1024,     # HyDE never needs a large context window; keeps KV cache small so model weights fit in VRAM
        },
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.monotonic() - t0
        passage = ((body.get("message") or {}).get("content") or "").strip()
        if not passage:
            return query, {
                "enabled": True,
                "applied": False,
                "reason": "empty_response",
                "latency_seconds": round(elapsed, 3),
                "model": model,
            }

        # While the model is still warm, ask it to rewrite the query as a
        # focused web search string. Model stays pinned (keep_alive=-1).
        search_query = query
        try:
            search_query = _rewrite_as_search_query(
                query, passage, model=model, url=url, timeout_seconds=min(timeout_seconds, 10.0),
            )
        except Exception as rewrite_exc:
            logger.warning("HyDE search query rewrite failed (falling back to original query): %s", rewrite_exc)

        return passage, {
            "enabled": True,
            "applied": True,
            "passage": passage,
            "search_query": search_query,
            "latency_seconds": round(elapsed, 3),
            "model": model,
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return query, {
            "enabled": True,
            "applied": False,
            "reason": "llm_error",
            "error": str(exc),
            "latency_seconds": round(elapsed, 3),
            "model": model,
        }


# ── Step-back prompting ───────────────────────────────────────────────────────
# Reference: "Take a Step Back: Evoking Reasoning via Abstraction in LLMs"
# Zheng et al. (2023) — https://arxiv.org/abs/2310.06117
#
# Given a specific question, ask an LLM to rephrase it as a broader, more
# abstract question whose answer would provide the background context needed to
# answer the original.  Retrieve on BOTH the original and the step-back query
# and merge the candidate sets.
#
# Model routing: avoid loading the large 9B model cold just for a query rewrite.
# Check Ollama /api/ps first — use whichever model is already warm.

_STEPBACK_SYSTEM_PROMPT = (
    "You are a research assistant. "
    "Given a specific question, rewrite it as a broader, more abstract question "
    "whose answer provides the background knowledge needed to answer the original. "
    "Reply with ONLY the rephrased question — no explanation, no preamble."
)

# Intents that are already high-level enough — step-back would just paraphrase
# them with no benefit (or introduce noise).
_STEPBACK_SKIP_INTENTS: frozenset = frozenset({
    "metadata_lookup",
    "conversational",
    "conversational_meta",
    "user_profile",
    "greeting",
    "out_of_scope",
    "current_data_lookup",
})

# Queries shorter than this probably lack enough specificity to benefit.
_STEPBACK_MIN_WORDS: int = 6


def _check_server_reachable(base_url: str, timeout: float = 2.0) -> bool:
    """Return True if the Ollama server at *base_url* responds at all."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/api/tags")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:
        return False


def _check_ollama_model_loaded(base_url: str, model: str, timeout: float = 2.0) -> bool:
    """Return True if *model* is currently pinned in Ollama VRAM (/api/ps)."""
    try:
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/api/ps",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        loaded_names = [m.get("name", "") for m in (data.get("models") or [])]
        return any(model in name for name in loaded_names)
    except Exception:
        return False


def generate_stepback_query(
    query: str,
    *,
    llm_model: str,
    llm_base_url: str,
    hyde_model: str = "",
    hyde_base_url: str = "",
    timeout_seconds: float = 15.0,
    intent: str = "",
) -> Tuple[str, Dict[str, Any]]:
    """Generate a step-back (broader) rewrite of *query*.

    Model routing (in priority order):
      1. Local 9B LLM — if already warm in Ollama (free — model is pinned)
      2. HyDE model   — if already warm (avoids cold-loading the 9B)
      3. HyDE model   — cold load accepted if neither is warm (small model)

    Returns ``(stepback_query, trace)``.  On any failure returns the original
    query so retrieval is never blocked.

    Trace keys: applied, model_used, warm, latency_seconds, reason (on failure)
    """
    if not (query or "").strip():
        return query, {"applied": False, "reason": "empty_query"}

    if intent in _STEPBACK_SKIP_INTENTS:
        return query, {"applied": False, "reason": f"skip_intent:{intent}"}

    if len(query.split()) < _STEPBACK_MIN_WORDS:
        return query, {"applied": False, "reason": "query_too_short"}

    # ── Model selection ───────────────────────────────────────────────────────
    model: str = ""
    base_url: str = ""
    warm: bool = False

    llm_warm = bool(
        llm_model and llm_base_url
        and _check_ollama_model_loaded(llm_base_url, llm_model)
    )
    hyde_warm = bool(
        hyde_model and hyde_base_url
        and _check_ollama_model_loaded(hyde_base_url, hyde_model)
    )
    # Only commit to cold-loading HyDE if its server is actually reachable;
    # if the satellite is down, fall through to the local LLM instead.
    hyde_reachable = bool(
        hyde_model and hyde_base_url
        and (hyde_warm or _check_server_reachable(hyde_base_url))
    )

    if llm_warm:
        model, base_url, warm = llm_model, llm_base_url, True
    elif hyde_warm:
        model, base_url, warm = hyde_model, hyde_base_url, True
    elif hyde_reachable:
        # HyDE server is up but model not yet loaded — cheap cold load
        model, base_url, warm = hyde_model, hyde_base_url, False
    elif llm_model and llm_base_url:
        # HyDE server unreachable — fall back to local LLM
        model, base_url, warm = llm_model, llm_base_url, False
    else:
        return query, {"applied": False, "reason": "no_model_configured"}

    # ── LLM call ──────────────────────────────────────────────────────────────
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = json.dumps({
        "model": model,
        "stream": False,
        "keep_alive": -1,
        "think": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 60,
            "num_ctx": 512,
        },
        "messages": [
            {"role": "system", "content": _STEPBACK_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )

    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.monotonic() - t0
        result = ((body.get("message") or {}).get("content") or "").strip().strip('"\'')

        # Reject degenerate outputs: same as original, too short, or too long
        if (
            not result
            or result.lower() == query.lower()
            or len(result) < 10
            or len(result) > 300
        ):
            return query, {
                "applied": False,
                "reason": "degenerate_output",
                "raw": result,
                "latency_seconds": round(elapsed, 3),
                "model_used": model,
                "warm": warm,
            }

        return result, {
            "applied": True,
            "stepback_query": result,
            "model_used": model,
            "warm": warm,
            "latency_seconds": round(elapsed, 3),
        }
    except Exception as exc:
        elapsed = time.monotonic() - t0
        return query, {
            "applied": False,
            "reason": "llm_error",
            "error": str(exc),
            "latency_seconds": round(elapsed, 3),
            "model_used": model,
            "warm": warm,
        }
