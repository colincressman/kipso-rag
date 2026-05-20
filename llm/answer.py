"""RAG answer generation - thin orchestration layer.

Context preparation lives in llm.answer_context.
Post-LLM finalization lives in llm.answer_generate.
All sub-module symbols are re-exported here for backward compatibility.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

# ── Sub-module re-exports (public API + backward compat) ─────────────────────
from llm.answer_context import (
    _RagGenCtx,
    _TOKEN_WARN_THRESHOLD,
    _estimate_prompt_tokens,
    prepare_rag_answer,
)
from llm.answer_generate import finalize_rag_answer
from llm.citations import (
    CITATION_RE,
    normalize_answer_citations,
    select_citations,
    ensure_inline_sentence_citations,
)
from llm.coverage import (
    determine_confidence_band,
    is_external_fact_query,
    is_factoid_query,
    is_metadata_fact_query,
)
from llm.extraction import (
    _has_explicit_formula_for_query,
    _has_formula_content,
    extract_metadata_field_answer,
    extractive_evidence_facts,
)
from llm.generation import (
    fallback_answer,
    grounded_citation_fallback,
    load_llm_config,
    ollama_chat,
)
from llm.grounding import (
    _MIN_COVERAGE_SCORE,
    _MIN_LEXICAL_COVERAGE,
    lexical_coverage_score,
    safe_no_coverage_answer,
    sentence_faithfulness_scores,
    unsupported_answer_entities,
)
from llm.prompt_templates import build_system_prompt, build_user_prompt
from llm.tracing import append_query_trace, chunk_trace_rows
from utils.runtime_defaults import (
    DEFAULT_CONTEXTUAL_COMPRESSION_ENABLED,
    DEFAULT_CONTEXTUAL_COMPRESSION_TIMEOUT,
    DEFAULT_CONTEXTUAL_COMPRESSION_TOP_N,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TEMPERATURE,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)

# Monkeypatch-friendly binding - tests patch `llm.answer._ollama_chat`.
_ollama_chat = ollama_chat

# Backwards-compat aliases used by tests that import private names directly.
_safe_no_coverage_answer = safe_no_coverage_answer
_lexical_coverage_score = lexical_coverage_score

logger = logging.getLogger(__name__)


def answer_query_with_retrieval(
    query: str,
    retrieval_result: Dict[str, Any],
    *,
    history: List[Dict[str, str]] | None = None,
    intent: str | None = None,
    llm_model: str | None = None,
    llm_base_url: str | None = None,
    llm_timeout_seconds: float | None = None,
    llm_temperature: float | None = None,
    config_path: str = "configs/llm.yaml",
) -> Dict[str, Any]:
    """Generate final grounded answer using retrieved hits.

    Parameters
    ----------
    query            : user query string
    retrieval_result : output from retrieve_as_dict()
    intent           : optional intent label from route_query()
    """
    ctx = prepare_rag_answer(
        query,
        retrieval_result,
        history=history,
        intent=intent,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
        config_path=config_path,
    )

    # Early exits (no_coverage, extractive metadata, etc.) return a dict directly.
    if not isinstance(ctx, _RagGenCtx):
        return ctx

    # LLM generation
    try:
        llm_text = _ollama_chat(
            model=ctx.model,
            system_prompt=ctx.system_prompt,
            user_prompt=ctx.user_prompt,
            base_url=ctx.base_url,
            timeout_seconds=ctx.timeout_seconds,
            temperature=ctx.temperature,
        )
    except Exception:
        llm_text = ""
        ctx.routing["fallback_reason"] = "llm_exception"

    return finalize_rag_answer(ctx, llm_text)
