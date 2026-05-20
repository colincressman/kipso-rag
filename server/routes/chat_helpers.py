"""
Shared helpers for the chat route.

Extracted from chat.py to keep the main route file focused on the
streaming pipeline (_stream_unified) and non-streaming handler (_handle_unified).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from db.client import (
    _CONV_KEEP_RECENT,
    _CONV_SUMMARIZE_THRESHOLD,
    add_conversation_message,
    compress_conversation,
    get_conversation,
    get_conversation_message_count,
    set_conversation_title,
)
from retrieval.internet_fallback import _filter_search_results
from retrieval.web_search import fetch_page as _fetch_page, search_web as _search_web
from server.shared import (
    DEFAULT_DB_DSN,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    _INTERNET_MAX_RESULTS,
    _INTERNET_TIMEOUT,
    ollama_chat,
)


# ── SSE helper ────────────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# ── Query / history helpers ───────────────────────────────────────────────────

_USER_PROFILE_RE = re.compile(
    r"""(?xi)
    \b(what\s+(do\s+you\s+know|does\s+(my|your|the)\s+(context|profile|info|background)|
       know\s+about\s+me|can\s+you\s+tell\s+me\s+about\s+myself)|
       tell\s+me\s+about\s+myself|
       who\s+am\s+i|
       my\s+(context|profile|background|info|information)\s+say|
       what\s+(is|are)\s+my\s+(name|role|job|occupation|projects?|background)|
       remind\s+me\s+of\s+my\s+(context|profile|info))\b
    """,
    re.IGNORECASE,
)


def _history_meta_prompt(history: List[Dict[str, str]], question: str) -> str:
    if not history:
        return f"The user asked: {question}\n\nThere is no prior conversation history. Let them know politely."
    lines = ["Here is the conversation history so far:\n"]
    for msg in history:
        role = msg.get("role", "")
        content = (msg.get("content") or "")[:800]
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")
    lines.append(f"\nNow answer this question about the conversation above: {question}")
    return "\n".join(lines)


def _history_to_prompt(
    history: List[Any],
    current_message: str,
    web_sources: Optional[List[Dict[str, str]]] = None,
) -> str:
    lines: List[str] = []
    if web_sources:
        lines.append("--- Web search results ---")
        for i, src in enumerate(web_sources, 1):
            title   = src.get("title", "").strip()
            url     = src.get("url", "")
            snippet = src.get("snippet", "").strip()
            content = src.get("content", "").strip()
            lines.append(f"[{i}] {title} ({url})")
            body = content if content else snippet
            if body:
                lines.append(f"    {body}")
        lines.append("--- End web results ---\n")
    for msg in (history or [])[-10:]:
        role    = msg.get("role")    if isinstance(msg, dict) else msg.role
        content = msg.get("content") if isinstance(msg, dict) else msg.content
        tag = "You" if role == "user" else "Assistant"
        lines.append(f"{tag}: {content}")
    lines.append(f"You: {current_message}")
    lines.append("Assistant:")
    return "\n".join(lines)


# ── Web context fetching ──────────────────────────────────────────────────────

_CONV_PREFIX_RE = re.compile(
    r"^(?:(?:can|could|would)\s+you\s+)?(?:please\s+)?"
    r"(?:tell\s+me|explain|describe|show\s+me|give\s+me|find|list|search\s+for)"
    r"(?:\s+(?:about|me|us|on|for|regarding))?"
    r"\s+",
    re.IGNORECASE,
)


def _extract_search_query(message: str) -> str:
    stripped = _CONV_PREFIX_RE.sub("", message.strip())
    return stripped if len(stripped) >= 3 else message.strip()


_PAGE_FETCH_MAX_CHARS = 8000
_PAGE_FETCH_TOP_N     = 3


def _fetch_web_context(query: str) -> tuple[List[Dict[str, str]], bool]:
    try:
        search_query = _extract_search_query(query)
        search_trace = _search_web(search_query, max_results=_INTERNET_MAX_RESULTS, timeout=_INTERNET_TIMEOUT)
        raw = list(search_trace.get("search_results") or [])
        filtered, _ = _filter_search_results(search_query, raw)
        sources: List[Dict[str, str]] = []
        for i, r in enumerate(filtered[:_INTERNET_MAX_RESULTS]):
            url = str(r.get("url") or "")
            if not url:
                continue
            src: Dict[str, str] = {
                "title":   str(r.get("title") or ""),
                "url":     url,
                "snippet": str(r.get("snippet") or ""),
                "content": "",
            }
            if i < _PAGE_FETCH_TOP_N:
                src["content"] = _fetch_page(url, timeout=_INTERNET_TIMEOUT, max_chars=_PAGE_FETCH_MAX_CHARS)
            sources.append(src)
        return sources, bool(sources)
    except Exception:
        return [], False


# ── Citation building ─────────────────────────────────────────────────────────

def _build_citations(
    answer: Dict[str, Any],
    retrieval: Dict[str, Any],
) -> List[Dict[str, Any]]:
    cited_ids: set[str] = set(answer.get("citations") or [])
    hits: List[Dict[str, Any]] = retrieval.get("hits") or []
    citations: List[Dict[str, Any]] = []
    seen: set[str] = set()
    ordered = sorted(hits, key=lambda h: (h.get("chunk_id") not in cited_ids, 0))
    for hit in ordered[:8]:
        chunk_id = hit.get("chunk_id", "")
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        source = (
            hit.get("document_title") or hit.get("source_name") or hit.get("doc_id", "Unknown")
        )
        page = hit.get("page_start") or hit.get("page_number")
        text_snippet = (hit.get("text") or "")[:300].strip()
        if len(hit.get("text") or "") > 300:
            text_snippet += "…"
        citations.append({
            "chunk_id":      chunk_id,
            "source":        source,
            "page":          page,
            "score":         round(float(hit.get("score") or 0), 4),
            "snippet":       text_snippet,
            "cited":         chunk_id in cited_ids,
            "source_type":   hit.get("source_type", "pdf_book"),
            "collection_id": hit.get("collection_id"),
        })
    return citations


# ── Background persistence ────────────────────────────────────────────────────

def _generate_title(conversation_id: str, first_message: str) -> None:
    title = first_message[:60].strip()
    if len(first_message) > 60:
        title += "…"
    try:
        set_conversation_title(DEFAULT_DB_DSN, conversation_id, title)
    except Exception:
        pass


def _maybe_compress(db_path: str, conversation_id: str) -> None:
    try:
        count = get_conversation_message_count(db_path, conversation_id)
        if count < _CONV_SUMMARIZE_THRESHOLD:
            return
        conv = get_conversation(db_path, conversation_id)
        if not conv:
            return
        msgs = conv.get("messages") or []
        to_compress = msgs[:-_CONV_KEEP_RECENT] if len(msgs) > _CONV_KEEP_RECENT else []
        if not to_compress:
            return
        keep_from_seq = to_compress[-1]["sequence"] + 1
        lines = ["Summarize the following conversation excerpt in 3-5 concise sentences:\n"]
        for m in to_compress:
            role_label = "User" if m["role"] == "user" else "Assistant"
            lines.append(f"{role_label}: {(m['content'] or '')[:400]}")
        summary = ollama_chat(
            model=DEFAULT_LLM_MODEL,
            system_prompt="You are a precise conversation summarizer. Be concise.",
            user_prompt="\n".join(lines),
            base_url=DEFAULT_LLM_BASE_URL,
            timeout_seconds=60.0,
            temperature=0.1,
            think=False,
        )
        if summary:
            compress_conversation(db_path, conversation_id, summary, keep_from_sequence=keep_from_seq)
    except Exception:
        pass


def _bg_persist(
    conversation_id: Optional[str],
    full_answer: str,
    mode: str,
    is_first: bool,
    first_message: str,
) -> None:
    import threading as _t
    if not conversation_id or not full_answer:
        return
    try:
        add_conversation_message(DEFAULT_DB_DSN, conversation_id, "assistant", full_answer, mode=mode)
    except Exception:
        pass
    if is_first:
        _t.Thread(target=_generate_title, args=(conversation_id, first_message), daemon=True).start()
    _t.Thread(target=_maybe_compress, args=(DEFAULT_DB_DSN, conversation_id), daemon=True).start()
