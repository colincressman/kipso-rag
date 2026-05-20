"""Post-LLM RAG answer finalization — grounding, citation enforcement, faithfulness.

Extracted from llm.answer to keep individual modules under ~300 lines.
Public surface is imported and re-exported by llm.answer for backward compat.
"""

from __future__ import annotations

from typing import Any, Dict

from llm.answer_context import _RagGenCtx
from llm.citations import (
	CITATION_RE,
	ensure_inline_sentence_citations,
	normalize_answer_citations,
)
from llm.generation import fallback_answer, grounded_citation_fallback
from llm.grounding import sentence_faithfulness_scores, unsupported_answer_entities
from llm.tracing import append_query_trace


def finalize_rag_answer(ctx: _RagGenCtx, llm_text: str) -> Dict[str, Any]:
	"""Post-process LLM output and return the final answer dict.

	Call this after ``prepare_rag_answer()`` returns a ``_RagGenCtx`` and you have
	collected the full LLM text (whether via ``ollama_chat`` or ``ollama_stream``).
	"""
	mode = ctx.mode
	answer = llm_text
	routing = ctx.routing
	citations = ctx.citations
	hits = ctx.hits
	raw_hits = ctx.raw_hits
	confidence_band = ctx.confidence_band
	evidence_facts = ctx.evidence_facts
	prompt_cfg = ctx.prompt_cfg
	decision_cfg = ctx.decision_cfg

	if not answer:
		mode = "fallback"
		routing["fallback_reason"] = "empty_llm_response"
		answer = fallback_answer(ctx.query, hits)
	elif bool(prompt_cfg.get("force_grounded_fallback_when_uncited", False)) and not CITATION_RE.search(answer):
		allow_low = bool(decision_cfg.get("allow_low_confidence_answer", True))
		require_fallback = (routing.get("rule") == "medium_ambiguous")
		if (
			(confidence_band == "low" and not allow_low)
			or require_fallback
			or (
				not bool(decision_cfg.get("allow_uncited_if_confident", True))
				and confidence_band != "high"
			)
		):
			mode = "grounded_fallback"
			routing["fallback_reason"] = "uncited_low_or_disallowed"
			answer = grounded_citation_fallback(hits)

	if bool(prompt_cfg.get("enforce_sentence_citations", True)) and mode in {
		"high_confidence",
		"medium_confidence",
		"low_confidence",
	}:
		answer, added_inline = ensure_inline_sentence_citations(answer, citations)
		routing["inline_citations_added"] = int(added_inline)

	enforce_entity_grounding = bool(decision_cfg.get("enforce_entity_grounding", True))
	bands = decision_cfg.get("entity_grounding_bands", ["high", "medium"])
	if not isinstance(bands, list):
		bands = ["high", "medium"]
	bands_set = {str(b).strip().lower() for b in bands if str(b).strip()}
	if enforce_entity_grounding and confidence_band in bands_set and mode in {
		"high_confidence",
		"medium_confidence",
	} and not ctx.internet_only_for_answer:
		unsupported_entities = unsupported_answer_entities(answer, ctx.query, hits, citations)
		if unsupported_entities:
			mode = "grounded_fallback"
			routing["fallback_reason"] = "unsupported_entities"
			routing["unsupported_entities"] = unsupported_entities[:8]
			answer = grounded_citation_fallback(hits)

	answer = normalize_answer_citations(answer, citations)
	if confidence_band in {"medium", "low"} and evidence_facts:
		routing["extractive_facts_used"] = True
		routing["extractive_fact_count"] = len([ln for ln in evidence_facts.splitlines() if ln.strip().startswith("-")])

	faithfulness = sentence_faithfulness_scores(answer, hits)
	flagged_count = sum(1 for s in faithfulness if s.get("flagged"))
	routing["faithfulness_sentence_count"] = len(faithfulness)
	routing["faithfulness_flagged_count"] = flagged_count

	payload: Dict[str, Any] = {
		"query": ctx.query,
		"mode": mode,
		"answer": answer,
		"citations": citations,
		"retrieved_count": len(hits),
		"llm_model": ctx.model,
		"confidence": ctx.confidence_meta,
		"routing": routing,
		"faithfulness": faithfulness,
		"retrieved_chunks": ctx.retrieved_trace,
		"llm_input_chunks": ctx.llm_input_trace,
	}
	append_query_trace(
		query=ctx.query,
		mode=mode,
		llm_used=True,
		retrieved_rows=ctx.retrieved_trace,
		llm_input_rows=ctx.llm_input_trace,
		intent=ctx.intent,
		internet_triggered=bool(routing.get("internet_triggered")),
		hyde_applied=ctx.hyde_applied,
	)
	return payload
