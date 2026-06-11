"""Chat endpoint — unified streaming + non-streaming query pipeline."""
from __future__ import annotations

import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import StreamingResponse

from db.client import (
    add_conversation_message,
    create_conversation,
    get_conversation,
)
from services.llm import answer as llm_answer
from services.rag import rag_retrieve
from llm.citations import strip_trailing_citations_block
from server.shared import (
    CHAT_SYSTEM_PROMPT,
    CHAT_WEB_SYSTEM_PROMPT,
    DEFAULT_DB_DSN,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    ChatRequest,
    ChatResponse,
    _gpu_busy_response,
    _gpu_is_tight,
    _gpu_state,
    _log_routing_trace,
    _resolve_query,
    _with_context,
    ollama_chat,
    ollama_stream,
)
from server.routes.chat_helpers import (
    _sse,
    _USER_PROFILE_RE,
    _history_meta_prompt,
    _history_to_prompt,
    _fetch_web_context,
    _build_citations,
    _generate_title,
    _maybe_compress,
    _bg_persist,
)

router = APIRouter()


# ── Streaming handler ─────────────────────────────────────────────────────────

def _stream_unified(
    req: ChatRequest,
    t0: float,
    history: List[Dict[str, str]] | None = None,
    conversation_id: Optional[str] = None,
    is_first: bool = False,
):
    from pipeline.plan import plan_query, TOOL_ICON  # noqa: PLC0415

    trace_id = str(uuid.uuid4())
    history_dicts = [{"role": m["role"], "content": m["content"]} for m in (history or [])]
    effective_query = _resolve_query(req.message, history_dicts)

    plan = plan_query(
        effective_query,
        force_rag=req.force_rag,
        force_web=req.force_web,
        history_available=bool(history),
        clarification_pending=req.clarification_pending,
        collection_id=req.collection_id,
        doc_ids=req.doc_ids or None,
        prior_sources=req.prior_sources or None,
    )

    yield _sse({
        "type":   "plan",
        "steps":  [{"tool": s.tool, "label": s.label, "icon": TOOL_ICON.get(s.tool, "")} for s in plan.steps],
        "intent": plan.intent,
    })

    rag_result: Optional[Dict[str, Any]] = None
    web_sources: List[Dict[str, str]] = []
    internet_used = False

    for i, step in enumerate(plan.steps):
        yield _sse({"type": "status", "message": step.label, "step_index": i})

        if step.tool == "history":
            if not history or _USER_PROFILE_RE.search(req.message):
                user_prompt = req.message
            else:
                user_prompt = _history_meta_prompt(history, req.message)
            full_answer = ""
            try:
                for token in ollama_stream(
                    model=DEFAULT_LLM_MODEL,
                    system_prompt=_with_context(CHAT_SYSTEM_PROMPT),
                    user_prompt=user_prompt,
                    base_url=DEFAULT_LLM_BASE_URL,
                    timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                    temperature=DEFAULT_LLM_TEMPERATURE,
                ):
                    full_answer += token
                    yield _sse({"type": "token", "content": token})
            except Exception as exc:
                yield _sse({"type": "error", "message": str(exc)})
                return
            _bg_persist(conversation_id, full_answer, "chat", is_first, req.message)
            _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], full_answer, round(time.perf_counter() - t0, 2), conversation_id)
            yield _sse({
                "type": "done", "answer": full_answer, "mode": "chat",
                "intent": plan.intent, "citations": [], "routing": None,
                "internet_used": False, "web_sources": [],
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "conversation_id": conversation_id, "trace_id": trace_id,
            })
            return

        elif step.tool == "clarify":
            clarify_topic = plan.meta.get("clarify_topic", "")
            clarify_prompt = (
                f'The user said: "{req.message}".\n'
                f'Your knowledge base covers "{clarify_topic}" in a machine learning '
                f'and deep learning context. Ask the user in a friendly, natural '
                f'one-sentence question whether they mean "{clarify_topic}" in the '
                f'technical ML sense, or something else entirely.'
            )
            full_answer = ""
            try:
                for token in ollama_stream(
                    model=DEFAULT_LLM_MODEL,
                    system_prompt=_with_context(CHAT_SYSTEM_PROMPT),
                    user_prompt=clarify_prompt,
                    base_url=DEFAULT_LLM_BASE_URL,
                    timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                    temperature=DEFAULT_LLM_TEMPERATURE,
                ):
                    full_answer += token
                    yield _sse({"type": "token", "content": token})
            except Exception as exc:
                yield _sse({"type": "error", "message": str(exc)})
                return
            _bg_persist(conversation_id, full_answer, "chat", is_first, req.message)
            _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], full_answer, round(time.perf_counter() - t0, 2), conversation_id)
            yield _sse({
                "type": "done", "answer": full_answer, "mode": "chat",
                "intent": plan.intent, "citations": [], "routing": None,
                "internet_used": False, "web_sources": [],
                "clarification_asked": True,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "conversation_id": conversation_id, "trace_id": trace_id,
            })
            return

        elif step.tool == "rag":
            try:
                yield _sse({"type": "status", "message": "Generating query variants…", "step_index": i})
                rag_result = rag_retrieve(
                    effective_query,
                    top_k=req.top_k,
                    collection=req.collection_id or None,
                    doc_ids=req.doc_ids or None,
                    prior_intents=req.prior_intents or None,
                )
                hit_count = len(rag_result.get("hits") or [])
                s = "s" if hit_count != 1 else ""
                _routing = rag_result.get("routing") or {}
                if _routing.get("rag_fusion_applied"):
                    variant_count = _routing.get("variant_count", 1)
                    yield _sse({"type": "status", "message": f"Retrieved via {variant_count} query variants (RAG Fusion) — {hit_count} chunk{s}…", "step_index": i})
                elif _routing.get("decomposition_applied"):
                    yield _sse({"type": "status", "message": f"Retrieved using query decomposition — {hit_count} chunk{s}…", "step_index": i})
                else:
                    yield _sse({"type": "status", "message": f"Found {hit_count} relevant chunk{s}…", "step_index": i})
            except Exception as exc:
                traceback.print_exc()
                yield _sse({"type": "status", "message": "Document search failed — continuing…", "step_index": i})

        elif step.tool == "web":
            try:
                web_sources, internet_used = _fetch_web_context(effective_query)
                src_count = len(web_sources)
                s = "s" if src_count != 1 else ""
                yield _sse({"type": "status", "message": f"Found {src_count} web source{s}…", "step_index": i})
            except Exception:
                yield _sse({"type": "status", "message": "Web search failed — continuing…", "step_index": i})

    yield _sse({"type": "status", "message": "Generating answer…"})

    if rag_result is not None:
        if web_sources:
            web_hits = []
            for j, src in enumerate(web_sources):
                body = (src.get("content") or src.get("snippet") or "").strip()
                if body:
                    web_hits.append({
                        "chunk_id": f"web_{j}", "text": body[:2000], "score": 0.85,
                        "source_type": "internet",
                        "document_title": src.get("title", "Web result"),
                        "metadata": {"url": src.get("url", ""), "title": src.get("title", "")},
                    })
            if web_hits:
                rag_result = dict(rag_result, hits=web_hits + list(rag_result.get("hits") or []))
        try:
            from llm.answer import prepare_rag_answer, finalize_rag_answer, _RagGenCtx  # noqa: PLC0415
            rag_intent = (rag_result.get("routing") or {}).get("intent") or plan.intent
            ctx = prepare_rag_answer(effective_query, rag_result, history=history, intent=rag_intent)
            if not isinstance(ctx, _RagGenCtx):
                answer = ctx
                full_answer = answer.get("answer", "")
            else:
                full_answer = ""
                for token in ollama_stream(
                    model=ctx.model,
                    system_prompt=_with_context(ctx.system_prompt),
                    user_prompt=ctx.user_prompt,
                    base_url=ctx.base_url,
                    timeout_seconds=ctx.timeout_seconds,
                    temperature=ctx.temperature,
                ):
                    full_answer += token
                    yield _sse({"type": "token", "content": token})
                answer = finalize_rag_answer(ctx, full_answer)
                full_answer = answer.get("answer", "")
            full_answer = strip_trailing_citations_block(full_answer)
            citations = _build_citations(answer, rag_result)
            internet_used_rag = bool((answer.get("internet_fallback") or {}).get("used", False))
            internet_used = internet_used or internet_used_rag
            _bg_persist(conversation_id, full_answer, "rag", is_first, req.message)
            _chunk_ids = [h.get("chunk_id", "") for h in (rag_result.get("hits") or [])]
            _log_routing_trace(trace_id, req.message, plan.intent, "rag", _chunk_ids, full_answer, round(time.perf_counter() - t0, 2), conversation_id)
            yield _sse({
                "type": "done", "answer": full_answer, "mode": "rag",
                "intent": plan.intent, "citations": citations,
                "routing": rag_result.get("routing"),
                "internet_used": internet_used, "web_sources": web_sources,
                "elapsed_seconds": round(time.perf_counter() - t0, 2),
                "conversation_id": conversation_id, "trace_id": trace_id,
            })
        except Exception as exc:
            traceback.print_exc()
            yield _sse({"type": "error", "message": str(exc)})
    else:
        system_prompt = _with_context(CHAT_WEB_SYSTEM_PROMPT if web_sources else CHAT_SYSTEM_PROMPT)
        user_prompt = _history_to_prompt(history or [], req.message, web_sources=web_sources)
        full_answer = ""
        try:
            for token in ollama_stream(
                model=DEFAULT_LLM_MODEL,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_url=DEFAULT_LLM_BASE_URL,
                timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                temperature=DEFAULT_LLM_TEMPERATURE,
            ):
                full_answer += token
                yield _sse({"type": "token", "content": token})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return
        _bg_persist(conversation_id, full_answer, "chat", is_first, req.message)
        _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], full_answer, round(time.perf_counter() - t0, 2), conversation_id)
        yield _sse({
            "type": "done", "answer": full_answer, "mode": "chat",
            "intent": plan.intent, "citations": [], "routing": None,
            "internet_used": internet_used, "web_sources": web_sources,
            "elapsed_seconds": round(time.perf_counter() - t0, 2),
            "conversation_id": conversation_id, "trace_id": trace_id,
        })


# ── Non-streaming handler ─────────────────────────────────────────────────────

def _handle_unified(
    req: ChatRequest,
    t0: float,
    history: List[Dict[str, str]] | None = None,
    conversation_id: Optional[str] = None,
    is_first: bool = False,
    background_tasks: Optional[BackgroundTasks] = None,
) -> ChatResponse:
    import threading as _t
    from pipeline.plan import plan_query  # noqa: PLC0415

    trace_id = str(uuid.uuid4())
    history_dicts = [m if isinstance(m, dict) else {"role": m["role"], "content": m["content"]} for m in (history or [])]
    effective_query = _resolve_query(req.message, history_dicts)

    plan = plan_query(
        effective_query,
        force_rag=req.force_rag,
        force_web=req.force_web,
        history_available=bool(history),
        clarification_pending=req.clarification_pending,
        collection_id=req.collection_id,
        doc_ids=req.doc_ids or None,
        prior_sources=req.prior_sources or None,
    )

    rag_result: Optional[Dict[str, Any]] = None
    web_sources: List[Dict[str, str]] = []
    internet_used = False

    for step in plan.steps:
        if step.tool == "history":
            if not history or _USER_PROFILE_RE.search(req.message):
                user_prompt = req.message
            else:
                user_prompt = _history_meta_prompt(history, req.message)
            full_answer = ollama_chat(
                model=DEFAULT_LLM_MODEL,
                system_prompt=_with_context(CHAT_SYSTEM_PROMPT),
                user_prompt=user_prompt,
                base_url=DEFAULT_LLM_BASE_URL,
                timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                temperature=DEFAULT_LLM_TEMPERATURE,
                think=False,
            )
            if conversation_id and full_answer:
                try:
                    add_conversation_message(DEFAULT_DB_DSN, conversation_id, "assistant", full_answer, mode="chat")
                except Exception:
                    pass
                if background_tasks:
                    if is_first:
                        background_tasks.add_task(_generate_title, conversation_id, req.message)
                    background_tasks.add_task(_maybe_compress, DEFAULT_DB_DSN, conversation_id)
            _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], full_answer or "", round(time.perf_counter() - t0, 2), conversation_id)
            return ChatResponse(answer=full_answer, mode="chat", citations=[], elapsed_seconds=round(time.perf_counter() - t0, 2), trace_id=trace_id)

        elif step.tool == "clarify":
            clarify_topic = plan.meta.get("clarify_topic", "")
            clarify_prompt = (
                f'The user said: "{req.message}".\n'
                f'Your knowledge base covers "{clarify_topic}" in a machine learning '
                f'and deep learning context. Ask the user in a friendly, natural '
                f'one-sentence question whether they mean "{clarify_topic}" in the '
                f'technical ML sense, or something else entirely.'
            )
            full_answer = ollama_chat(
                model=DEFAULT_LLM_MODEL,
                system_prompt=_with_context(CHAT_SYSTEM_PROMPT),
                user_prompt=clarify_prompt,
                base_url=DEFAULT_LLM_BASE_URL,
                timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
                temperature=DEFAULT_LLM_TEMPERATURE,
                think=False,
            )
            if conversation_id and full_answer:
                try:
                    add_conversation_message(DEFAULT_DB_DSN, conversation_id, "assistant", full_answer, mode="chat")
                except Exception:
                    pass
                if background_tasks:
                    if is_first:
                        background_tasks.add_task(_generate_title, conversation_id, req.message)
                    background_tasks.add_task(_maybe_compress, DEFAULT_DB_DSN, conversation_id)
            _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], full_answer or "", round(time.perf_counter() - t0, 2), conversation_id)
            return ChatResponse(answer=full_answer, mode="chat", citations=[], elapsed_seconds=round(time.perf_counter() - t0, 2), trace_id=trace_id)

        elif step.tool == "rag":
            try:
                rag_result = rag_retrieve(
                    effective_query,
                    top_k=req.top_k,
                    collection=req.collection_id or None,
                    doc_ids=req.doc_ids or None,
                    prior_intents=req.prior_intents or None,
                )
            except Exception:
                traceback.print_exc()
        elif step.tool == "web":
            try:
                web_sources, internet_used = _fetch_web_context(effective_query)
            except Exception:
                pass

    if rag_result is not None:
        if web_sources:
            web_hits = []
            for j, src in enumerate(web_sources):
                body = (src.get("content") or src.get("snippet") or "").strip()
                if body:
                    web_hits.append({
                        "chunk_id": f"web_{j}", "text": body[:2000], "score": 0.85,
                        "source_type": "internet",
                        "document_title": src.get("title", "Web result"),
                        "metadata": {"url": src.get("url", ""), "title": src.get("title", "")},
                    })
            if web_hits:
                rag_result = dict(rag_result, hits=web_hits + list(rag_result.get("hits") or []))
        answer = llm_answer(effective_query, rag_result, history=history)
        full_answer = strip_trailing_citations_block(answer.get("answer", ""))
        citations = _build_citations(answer, rag_result)
        internet_used_rag = bool((answer.get("internet_fallback") or {}).get("used", False))
        internet_used = internet_used or internet_used_rag
        if conversation_id and full_answer:
            try:
                add_conversation_message(DEFAULT_DB_DSN, conversation_id, "assistant", full_answer, mode="rag")
            except Exception:
                pass
            if background_tasks:
                if is_first:
                    background_tasks.add_task(_generate_title, conversation_id, req.message)
                background_tasks.add_task(_maybe_compress, DEFAULT_DB_DSN, conversation_id)
        _chunk_ids = [h.get("chunk_id", "") for h in (rag_result.get("hits") or [])]
        _log_routing_trace(trace_id, req.message, plan.intent, "rag", _chunk_ids, full_answer, round(time.perf_counter() - t0, 2), conversation_id)
        return ChatResponse(
            answer=full_answer, mode="rag", citations=citations,
            routing=rag_result.get("routing"), internet_used=internet_used,
            elapsed_seconds=round(time.perf_counter() - t0, 2), trace_id=trace_id,
        )
    else:
        system_prompt = _with_context(CHAT_WEB_SYSTEM_PROMPT if web_sources else CHAT_SYSTEM_PROMPT)
        user_prompt = _history_to_prompt(history or [], req.message, web_sources=web_sources)
        answer_text = ollama_chat(
            model=DEFAULT_LLM_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base_url=DEFAULT_LLM_BASE_URL,
            timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            temperature=DEFAULT_LLM_TEMPERATURE,
            think=False,
        )
        if conversation_id and answer_text:
            try:
                add_conversation_message(DEFAULT_DB_DSN, conversation_id, "assistant", answer_text, mode="chat")
            except Exception:
                pass
            if background_tasks:
                if is_first:
                    background_tasks.add_task(_generate_title, conversation_id, req.message)
                background_tasks.add_task(_maybe_compress, DEFAULT_DB_DSN, conversation_id)
        _log_routing_trace(trace_id, req.message, plan.intent, "chat", [], answer_text or "", round(time.perf_counter() - t0, 2), conversation_id)
        return ChatResponse(
            answer=answer_text, mode="chat", internet_used=internet_used,
            web_sources=web_sources,
            elapsed_seconds=round(time.perf_counter() - t0, 2), trace_id=trace_id,
        )


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/api/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> Any:
    lock = _gpu_state["lock"]
    if not lock._value and _gpu_is_tight():  # type: ignore[attr-defined]
        return _gpu_busy_response()

    t0 = time.perf_counter()

    # Ensure a conversation exists
    conversation_id = req.conversation_id
    conv_data: Optional[Dict[str, Any]] = None
    if conversation_id:
        try:
            conv_data = get_conversation(DEFAULT_DB_DSN, conversation_id)
        except Exception:
            conv_data = None
    if not conversation_id or conv_data is None:
        try:
            conversation_id = create_conversation(DEFAULT_DB_DSN)
            conv_data = {"messages": [], "summary": None}
        except Exception:
            conversation_id = None
            conv_data = {"messages": [], "summary": None}

    # Build history for LLM
    history_for_llm: List[Dict[str, str]] = []
    if conv_data:
        raw_msgs = (conv_data.get("messages") or [])[-20:]
        if conv_data.get("summary"):
            history_for_llm.append({
                "role": "assistant",
                "content": f"[Summary of earlier messages: {conv_data['summary']}]",
            })
        history_for_llm.extend({"role": m["role"], "content": m["content"]} for m in raw_msgs)

    is_first_message = not conv_data or not conv_data.get("messages")
    if conversation_id:
        try:
            add_conversation_message(DEFAULT_DB_DSN, conversation_id, "user", req.message, mode=req.mode)
        except Exception:
            pass

    if req.stream:
        return StreamingResponse(
            _stream_unified(req, t0, history=history_for_llm, conversation_id=conversation_id, is_first=is_first_message),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    try:
        return _handle_unified(req, t0, history=history_for_llm, conversation_id=conversation_id, is_first=is_first_message, background_tasks=background_tasks)
    except Exception as exc:
        traceback.print_exc()
        return ChatResponse(
            answer="Sorry, something went wrong on the server.",
            mode="chat",
            error=str(exc),
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )
