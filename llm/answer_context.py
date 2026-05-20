"""Pre-LLM RAG answer preparation — context selection & routing decisions.

Extracted from llm.answer to keep individual modules under ~300 lines.
Public surface is imported and re-exported by llm.answer for backward compat.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm.citations import normalize_answer_citations, select_citations
from llm.coverage import (
	determine_confidence_band,
	is_external_fact_query,
	is_factoid_query,
	is_metadata_fact_query,
)
from llm.extraction import (
	_has_explicit_formula_for_query,
	extract_metadata_field_answer,
	extractive_evidence_facts,
)
from llm.generation import load_llm_config
from llm.grounding import (
	_MIN_COVERAGE_SCORE,
	_MIN_LEXICAL_COVERAGE,
	lexical_coverage_score,
	safe_no_coverage_answer,
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

logger = logging.getLogger(__name__)

# Rough token budget: warn when the combined prompt looks like it may be
# approaching the model's context window.  This threshold is intentionally
# conservative — we warn at 75% rather than blocking.
_TOKEN_WARN_THRESHOLD = 24_576   # ~75% of typical 32k context window


def _estimate_prompt_tokens(text: str) -> int:
	"""Fast, allocation-cheap token estimate: chars / 4 (GPT-style average)."""
	return max(1, len(text) // 4)


# ── Two-phase generation context ─────────────────────────────────────────────
# Used by _stream_unified (server) to split pre-LLM prep from the LLM call so
# tokens can be streamed before finalize_rag_answer() runs post-processing.

@dataclass
class _RagGenCtx:
	"""Intermediate state between prepare_rag_answer() and finalize_rag_answer()."""
	model: str
	base_url: str
	timeout_seconds: float
	temperature: float
	system_prompt: str
	user_prompt: str
	citations: List[Dict[str, Any]]
	confidence_band: str
	mode: str
	routing: Dict[str, Any]
	hits: List[Dict[str, Any]]
	raw_hits: List[Dict[str, Any]]
	prompt_cfg: Dict[str, Any]
	decision_cfg: Dict[str, Any]
	query: str
	intent: str
	hyde_applied: bool
	retrieved_trace: List[Any]
	llm_input_trace: List[Any]
	internet_only_for_answer: bool = False
	evidence_facts: str = ""
	confidence_meta: Dict[str, Any] = field(default_factory=dict)


def prepare_rag_answer(
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
) -> "Dict[str, Any] | _RagGenCtx":
	"""Run all pre-LLM answer preparation.

	Returns either:
	- A complete answer dict (early exit: no_coverage, extractive metadata, etc.)
	- A ``_RagGenCtx`` containing everything needed to call the LLM and finalize.

	Callers that want to stream tokens should check ``isinstance(result, _RagGenCtx)``,
	then call ``ollama_stream(model=ctx.model, ...)`` and pass the buffered text to
	``finalize_rag_answer(ctx, llm_text)``.
	"""
	if intent is None:
		intent = "fact_lookup"

	raw_hits: List[Dict[str, Any]] = retrieval_result.get("hits", [])
	internet_trace = retrieval_result.get("internet_fallback") if isinstance(retrieval_result.get("internet_fallback"), dict) else {}
	internet_triggered = bool(internet_trace.get("triggered"))
	external_fact_query = is_external_fact_query(query)
	internet_only_for_answer = bool(external_fact_query and internet_triggered)
	internet_hits = [h for h in raw_hits if str(h.get("source_type") or "") == "internet"]
	if internet_only_for_answer and internet_hits:
		hits = sorted(internet_hits, key=lambda h: float(h.get("score") or 0.0), reverse=True)
	else:
		hits = list(raw_hits)

	cfg = load_llm_config(config_path)
	llm_cfg: Dict[str, Any] = cfg.get("llm", {})
	prompt_cfg: Dict[str, Any] = cfg.get("prompt", {})
	decision_cfg: Dict[str, Any] = cfg.get("decision", {})
	always_use_llm = bool(decision_cfg.get("always_use_llm", True))

	model = llm_model or str(llm_cfg.get("model", DEFAULT_LLM_MODEL))
	base_url = llm_base_url or str(llm_cfg.get("base_url", DEFAULT_LLM_BASE_URL))
	timeout_seconds = float(llm_timeout_seconds if llm_timeout_seconds is not None else llm_cfg.get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS))
	temperature = float(llm_temperature if llm_temperature is not None else llm_cfg.get("temperature", DEFAULT_LLM_TEMPERATURE))

	if DEFAULT_CONTEXTUAL_COMPRESSION_ENABLED and hits:
		from retrieval.context_compress import compress_chunks  # noqa: PLC0415
		hits = compress_chunks(
			query,
			hits,
			model=model,
			base_url=base_url,
			timeout_seconds=DEFAULT_CONTEXTUAL_COMPRESSION_TIMEOUT,
			top_n=DEFAULT_CONTEXTUAL_COMPRESSION_TOP_N,
		)

	citations = select_citations(hits, prompt_cfg)
	confidence_band, confidence_meta = determine_confidence_band(query, hits, decision_cfg)
	evidence_facts = extractive_evidence_facts(query, hits, citations, max_facts=6, confidence_band=confidence_band)
	factoid_query = is_factoid_query(query)
	metadata_fact_query = is_metadata_fact_query(query)

	top_score = float(confidence_meta.get("top_score") or (hits[0].get("score") if hits else 0.0) or 0.0)

	prompt_cfg_runtime: Dict[str, Any] = dict(prompt_cfg)
	if internet_only_for_answer:
		prompt_cfg_runtime["max_chunks"] = max(3, int(prompt_cfg_runtime.get("max_chunks", 6)))
		prompt_cfg_runtime["max_chars_per_chunk"] = max(6000, int(prompt_cfg_runtime.get("max_chars_per_chunk", 1600)))
		prompt_cfg_runtime["include_neighbor_context"] = False
	elif confidence_band == "high" and evidence_facts:
		prompt_cfg_runtime["max_chunks"] = min(3, int(prompt_cfg_runtime.get("max_chunks", 6)))

	# Determine which prompt persona/style to use based on where the answer comes from.
	if internet_only_for_answer:
		source_mode = "web"
	elif not hits or all(h.get("score", 1.0) is None for h in hits):
		source_mode = "general"
	else:
		source_mode = "corpus"

	system_prompt = build_system_prompt(prompt_cfg_runtime, source_mode=source_mode)
	user_prompt = build_user_prompt(
		query,
		hits,
		prompt_cfg_runtime,
		confidence_band=confidence_band,
		evidence_facts=evidence_facts,
		history=history,
		source_mode=source_mode,
		intent=intent,
	)

	_prompt_tokens = _estimate_prompt_tokens(system_prompt) + _estimate_prompt_tokens(user_prompt)
	if _prompt_tokens > _TOKEN_WARN_THRESHOLD:
		logger.warning(
			"Token budget warning: estimated prompt tokens %d exceeds threshold %d "
			"(query=%r, hits=%d, source_mode=%s)",
			_prompt_tokens, _TOKEN_WARN_THRESHOLD, query[:80], len(hits), source_mode,
		)

	mode = f"{confidence_band}_confidence"
	routing: Dict[str, Any] = {
		"intent": intent,
		"confidence_band": confidence_band,
		"top_score": confidence_meta.get("top_score"),
		"score_gap": confidence_meta.get("score_gap"),
		"rule": confidence_meta.get("rule"),
		"override": bool(confidence_meta.get("override", False)),
		"max_ambiguous_gap": float(decision_cfg.get("max_ambiguous_gap", 0.03)),
		"medium_confidence_score": float(decision_cfg.get("medium_confidence_score", 0.55)),
		"high_confidence_score": float(decision_cfg.get("high_confidence_score", 0.70)),
		"always_use_llm": always_use_llm,
		"factoid_query": factoid_query,
		"metadata_fact_query": metadata_fact_query,
		"internet_triggered": internet_triggered,
		"external_fact_query": external_fact_query,
		"internet_only_for_answer": internet_only_for_answer,
		"estimated_prompt_tokens": _prompt_tokens,
	}
	llm_max_chunks = int(prompt_cfg_runtime.get("max_chunks", 6))
	retrieved_trace = chunk_trace_rows(raw_hits)
	llm_input_trace = chunk_trace_rows(hits[:llm_max_chunks])

	hyde_applied = bool(
		retrieval_result.get("hyde_trace") and
		retrieval_result.get("hyde_trace", {}).get("applied")
	)

	def _with_trace(payload: Dict[str, Any], *, llm_used: bool) -> Dict[str, Any]:
		payload["retrieved_chunks"] = retrieved_trace
		payload["llm_input_chunks"] = llm_input_trace if llm_used else []
		append_query_trace(
			query=query,
			mode=str(payload.get("mode", mode)),
			llm_used=llm_used,
			retrieved_rows=retrieved_trace,
			llm_input_rows=(llm_input_trace if llm_used else []),
			intent=intent,
			internet_triggered=internet_triggered,
			hyde_applied=hyde_applied,
		)
		return payload

	if internet_only_for_answer and not internet_hits:
		routing["answer_policy"] = "internet_no_evidence"
		routing["no_coverage_reason"] = "internet_fallback_triggered_but_no_qualified_web_hits"
		return _with_trace({
			"query": query,
			"mode": "internet_no_evidence",
			"answer": (
				"The internet fallback was triggered for this fact query, but no reliable internet evidence "
				"passed relevance checks. I cannot produce a grounded answer — no answer fabricated."
			),
			"citations": [],
			"retrieved_count": len(raw_hits),
			"llm_model": model,
			"confidence": confidence_meta,
			"routing": routing,
		}, llm_used=False)

	has_explicit_scores = bool(hits and any(h.get("score") is not None for h in hits))

	lex_coverage = lexical_coverage_score(query, hits)
	routing["lexical_coverage_score"] = round(lex_coverage, 3)

	# 1. Score too low
	if not always_use_llm and has_explicit_scores and top_score < _MIN_COVERAGE_SCORE and not metadata_fact_query and intent != "section_lookup":
		routing["answer_policy"] = "no_coverage"
		routing["no_coverage_reason"] = "low_score"
		answer = safe_no_coverage_answer(query, intent, top_score)
		return _with_trace({
			"query": query,
			"mode": "no_coverage",
			"answer": answer,
			"citations": citations,
			"retrieved_count": len(hits),
			"llm_model": model,
			"confidence": confidence_meta,
			"routing": routing,
		}, llm_used=False)

	_skip_lexical_gate_intents = {"formula_lookup", "section_lookup"}
	force_zero_coverage_fallback = (
		has_explicit_scores
		and not metadata_fact_query
		and intent not in _skip_lexical_gate_intents
		and confidence_band != "high"
		and lex_coverage <= 0.0
		and not internet_triggered
		and not internet_only_for_answer
	)
	if force_zero_coverage_fallback:
		routing["answer_policy"] = "no_coverage"
		routing["no_coverage_reason"] = "zero_lexical_coverage"
		answer = safe_no_coverage_answer(query, intent, top_score)
		return _with_trace({
			"query": query,
			"mode": "no_coverage",
			"answer": answer,
			"citations": citations,
			"retrieved_count": len(hits),
			"llm_model": model,
			"confidence": confidence_meta,
			"routing": routing,
		}, llm_used=False)

	if (
		not always_use_llm
		and has_explicit_scores
		and not metadata_fact_query
		and intent not in _skip_lexical_gate_intents
		and confidence_band != "high"
		and lex_coverage < _MIN_LEXICAL_COVERAGE
		and not internet_triggered   # internet hits use different vocab (e.g. "CEO" vs "chief executive")
	):
		routing["answer_policy"] = "no_coverage"
		routing["no_coverage_reason"] = "low_lexical_coverage"
		answer = safe_no_coverage_answer(query, intent, top_score)
		return _with_trace({
			"query": query,
			"mode": "no_coverage",
			"answer": answer,
			"citations": citations,
			"retrieved_count": len(hits),
			"llm_model": model,
			"confidence": confidence_meta,
			"routing": routing,
		}, llm_used=False)

	# 2. Formula intent: verify explicit equation present
	if not always_use_llm and has_explicit_scores and intent == "formula_lookup" and not _has_explicit_formula_for_query(query, hits):
		routing["answer_policy"] = "formula_not_found"
		routing["formula_reason"] = "missing_explicit_equation"
		answer = safe_no_coverage_answer(query, intent, top_score)
		return _with_trace({
			"query": query,
			"mode": "formula_not_found",
			"answer": answer,
			"citations": citations,
			"retrieved_count": len(hits),
			"llm_model": model,
			"confidence": confidence_meta,
			"routing": routing,
		}, llm_used=False)

	# 3. Metadata extractive path
	if not always_use_llm and metadata_fact_query:
		metadata_answer = extract_metadata_field_answer(query, hits, citations)
		if metadata_answer:
			routing["answer_policy"] = "extractive_metadata"
			answer = normalize_answer_citations(metadata_answer, citations)
			if confidence_band in {"medium", "low"} and evidence_facts:
				routing["extractive_facts_used"] = True
				routing["extractive_fact_count"] = len([ln for ln in evidence_facts.splitlines() if ln.strip().startswith("-")])
			return _with_trace({
				"query": query,
				"mode": mode,
				"answer": answer,
				"citations": citations,
				"retrieved_count": len(hits),
				"llm_model": model,
				"confidence": confidence_meta,
				"routing": routing,
			}, llm_used=False)

	# Needs LLM generation — return context object for the caller to drive.
	return _RagGenCtx(
		model=model,
		base_url=base_url,
		timeout_seconds=timeout_seconds,
		temperature=temperature,
		system_prompt=system_prompt,
		user_prompt=user_prompt,
		citations=citations,
		confidence_band=confidence_band,
		mode=mode,
		routing=routing,
		hits=hits,
		raw_hits=raw_hits,
		prompt_cfg=prompt_cfg,
		decision_cfg=decision_cfg,
		query=query,
		intent=intent,
		hyde_applied=hyde_applied,
		retrieved_trace=retrieved_trace,
		llm_input_trace=llm_input_trace,
		internet_only_for_answer=internet_only_for_answer,
		evidence_facts=evidence_facts,
		confidence_meta=confidence_meta,
	)
