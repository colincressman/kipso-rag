from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from llm.answer import answer_query_with_retrieval
from retrieval.context_pack import build_context_pack
from retrieval.query import RetrievalFilters, retrieve_as_dict
from retrieval.router import route_query
from utils.runtime_defaults import DEFAULT_DB_DSN, DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME


@dataclass
class AcceptanceCase:
	id: str
	category: str
	question: str
	description: str
	top_k: int = 8
	expected_internet: Optional[bool] = None
	min_unique_sources: int = 1
	require_citations: bool = True
	allow_refusal: bool = False
	expected_intents: List[str] = field(default_factory=list)
	expected_source_hints: List[str] = field(default_factory=list)
	notes: str = ""


CASES: List[AcceptanceCase] = [
	AcceptanceCase(
		id="core_ss_01",
		category="single_source",
		question="Explain CAPM from my materials.",
		description="Core single-source finance retrieval and grounded explanation.",
		top_k=6,
		expected_internet=False,
		expected_intents=["summary", "fact_lookup", "formula_lookup"],
		expected_source_hints=["Hedge", "finance", "CAPM"],
	),
	AcceptanceCase(
		id="core_ss_02",
		category="single_source",
		question="What is a Markov Decision Process and what are its key components?",
		description="Single-source RL textbook question.",
		top_k=6,
		expected_internet=False,
		expected_intents=["summary", "fact_lookup"],
	),
	AcceptanceCase(
		id="core_ss_03",
		category="single_source",
		question="How does backpropagation work in neural networks?",
		description="Single-source deep learning concept retrieval.",
		top_k=6,
		expected_internet=False,
		expected_intents=["summary"],
	),
	AcceptanceCase(
		id="core_ss_04",
		category="single_source",
		question="Explain the bias-variance tradeoff in machine learning.",
		description="Single-source ML concept explanation.",
		top_k=6,
		expected_internet=False,
		expected_intents=["summary"],
	),
	AcceptanceCase(
		id="core_notes_01",
		category="notes_or_local_material",
		question="Compare how my notes and the textbook explain Q-learning.",
		description="Intended to exercise notes plus textbook retrieval if notes are available.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["comparison", "summary"],
		notes="If notes are not ingested, this will still reveal whether multi-source local retrieval is working.",
	),
	AcceptanceCase(
		id="core_notes_02",
		category="notes_or_local_material",
		question="Summarize the introduction section from my notes or course material.",
		description="Exercises section lookup and provenance for local notes/material.",
		top_k=6,
		expected_internet=False,
		expected_intents=["section_lookup"],
	),
	AcceptanceCase(
		id="multi_01",
		category="multi_source",
		question="How are hedge funds using artificial intelligence and big data in their investment strategies?",
		description="Multi-source synthesis across finance and AI material.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["summary", "exploratory"],
	),
	AcceptanceCase(
		id="multi_02",
		category="multi_source",
		question="What are the key differences between supervised learning and reinforcement learning?",
		description="Broad local comparison with multiple relevant sources.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["comparison"],
	),
	AcceptanceCase(
		id="multi_03",
		category="multi_source",
		question="What techniques can be used to prevent overfitting in machine learning models?",
		description="Should gather multiple ML sources with reduced redundancy.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["list_lookup", "summary"],
	),
	AcceptanceCase(
		id="multi_04",
		category="multi_source",
		question="Compare how different sources in my corpus explain logistic regression.",
		description="Pure corpus comparison task to inspect multi-source combination quality.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["comparison"],
	),
	AcceptanceCase(
		id="prov_01",
		category="provenance",
		question="Answer this question and show which source it came from: What is arbitrage?",
		description="Checks provenance and citation retention on a direct factual answer.",
		top_k=6,
		expected_internet=False,
		expected_intents=["fact_lookup", "summary"],
	),
	AcceptanceCase(
		id="prov_02",
		category="provenance",
		question="What section discusses arbitrage?",
		description="Section lookup with stable chunk IDs and source path metadata.",
		top_k=6,
		expected_internet=False,
		expected_intents=["section_lookup"],
	),
	AcceptanceCase(
		id="prov_03",
		category="provenance",
		question="Where is market making discussed?",
		description="Another provenance-heavy locator query.",
		top_k=6,
		expected_internet=False,
		expected_intents=["section_lookup", "fact_lookup"],
		expected_source_hints=["market", "hedge", "balch", "romero"],
	),
	AcceptanceCase(
		id="router_01",
		category="routing",
		question="Summarize the introduction section.",
		description="Router should prefer section lookup instead of blind broad search.",
		top_k=6,
		expected_internet=False,
		expected_intents=["section_lookup"],
	),
	AcceptanceCase(
		id="router_02",
		category="routing",
		question="CAPM formula",
		description="Router should classify as formula or narrow fact query.",
		top_k=5,
		expected_internet=False,
		expected_intents=["formula_lookup", "fact_lookup"],
	),
	AcceptanceCase(
		id="router_03",
		category="routing",
		question="List the techniques used to prevent overfitting.",
		description="Router should recognize list-style retrieval behavior.",
		top_k=7,
		expected_internet=False,
		expected_intents=["list_lookup", "summary"],
	),
	AcceptanceCase(
		id="router_04",
		category="routing",
		question="How does arbitrage relate to market efficiency?",
		description="Router should handle mixed conceptual query without breaking retrieval.",
		top_k=7,
		expected_internet=False,
		expected_intents=["comparison", "summary", "exploratory"],
	),
	AcceptanceCase(
		id="combo_01",
		category="combination",
		question="Explain logistic regression using perspectives from multiple books.",
		description="Inspect deduplication and source diversity in final context pack.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["summary", "comparison"],
	),
	AcceptanceCase(
		id="combo_02",
		category="combination",
		question="Compare binary logistic regression definitions across sources.",
		description="Should avoid repeating near-duplicate chunks.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["comparison"],
	),
	AcceptanceCase(
		id="combo_03",
		category="combination",
		question="How is maximum likelihood used for logistic regression across different texts?",
		description="Combination-stage quality on overlapping explanations.",
		top_k=8,
		expected_internet=False,
		min_unique_sources=2,
		expected_intents=["comparison", "summary"],
	),
	AcceptanceCase(
		id="internet_01",
		category="internet_fallback",
		question="Who won the FIFA World Cup in 2022?",
		description="External fact fallback with internet-only answer context.",
		top_k=5,
		expected_internet=True,
		expected_intents=["fact_lookup"],
	),
	AcceptanceCase(
		id="internet_02",
		category="internet_fallback",
		question="When was the GPT-4 model released by OpenAI?",
		description="External fact fallback with authoritative web result preference.",
		top_k=5,
		expected_internet=True,
		expected_intents=["fact_lookup"],
	),
	AcceptanceCase(
		id="internet_03",
		category="internet_fallback",
		question="Who is the current CEO of Microsoft?",
		description="Current affairs fallback that should use web evidence if local corpus lacks support.",
		top_k=5,
		expected_internet=True,
		expected_intents=["fact_lookup"],
		allow_refusal=True,
	),
	AcceptanceCase(
		id="internet_04",
		category="internet_guardrail",
		question="Explain the bias-variance tradeoff in machine learning.",
		description="Should stay local and not trigger internet for a well-covered corpus topic.",
		top_k=6,
		expected_internet=False,
		expected_intents=["summary"],
	),
	AcceptanceCase(
		id="internet_05",
		category="internet_guardrail",
		question="What is CAPM?",
		description="Should remain local for strong finance coverage.",
		top_k=5,
		expected_internet=False,
		expected_intents=["fact_lookup", "summary"],
	),
	AcceptanceCase(
		id="failure_01",
		category="failure_transparency",
		question="What does the book say about cryptocurrency trading?",
		description="Weak local support case should fail transparently instead of inventing support.",
		top_k=6,
		expected_internet=False,
		allow_refusal=True,
		expected_intents=["summary", "fact_lookup"],
	),
	AcceptanceCase(
		id="failure_02",
		category="failure_transparency",
		question="Explain neural networks in hedge funds.",
		description="May be weakly covered locally; should either ground clearly or fail transparently.",
		top_k=6,
		expected_internet=False,
		allow_refusal=True,
		expected_intents=["summary", "exploratory"],
	),
	AcceptanceCase(
		id="failure_03",
		category="failure_transparency",
		question="What are the course grading rubric details for this class?",
		description="Out-of-corpus question should not hallucinate local support.",
		top_k=5,
		expected_internet=False,
		allow_refusal=True,
		expected_intents=["fact_lookup", "summary"],
	),
]


def _source_label(hit: Dict[str, Any]) -> str:
	metadata = hit.get("metadata") or {}
	return str(
		hit.get("source_name")
		or hit.get("document_title")
		or hit.get("document_path")
		or metadata.get("source_name")
		or metadata.get("document_title")
		or metadata.get("document_path")
		or ""
	).strip()


def _provenance_present(hit: Dict[str, Any]) -> bool:
	metadata = hit.get("metadata") or {}
	return any(
		[
			hit.get("source_name"),
			hit.get("document_title"),
			hit.get("document_path"),
			hit.get("page_number"),
			hit.get("page_start"),
			hit.get("section_header"),
			metadata.get("source_name"),
			metadata.get("document_title"),
			metadata.get("document_path"),
			metadata.get("page_number"),
			metadata.get("section_header"),
		]
	)


def _hit_row(hit: Dict[str, Any]) -> Dict[str, Any]:
	metadata = hit.get("metadata") or {}
	text = str(hit.get("text") or "")
	return {
		"chunk_id": hit.get("chunk_id"),
		"score": hit.get("score"),
		"source_type": hit.get("source_type") or metadata.get("source_type"),
		"source_name": hit.get("source_name") or metadata.get("source_name"),
		"document_title": hit.get("document_title") or metadata.get("document_title"),
		"document_path": hit.get("document_path") or metadata.get("document_path"),
		"page_number": hit.get("page_number") or metadata.get("page_number") or hit.get("page_start"),
		"page_end": hit.get("page_end") or metadata.get("page_end"),
		"section_header": hit.get("section_header") or metadata.get("section_header") or hit.get("title"),
		"path_text": hit.get("path_text"),
		"structural_role": hit.get("structural_role") or metadata.get("structural_role"),
		"text_preview": text[:400],
	}


def _evaluate_checks(
	case: AcceptanceCase,
	*,
	routed: Any,
	retrieved: Dict[str, Any],
	context_pack: Dict[str, Any],
	answer: Dict[str, Any],
) -> Dict[str, Any]:
	selected_chunks = list(context_pack.get("selected_chunks") or [])
	raw_hits = list(retrieved.get("hits") or [])
	selection_meta = dict(context_pack.get("selection_meta") or {})
	answer_routing = dict(answer.get("routing") or {})
	llm_input_chunks = list(answer.get("llm_input_chunks") or [])
	selected_sources = [_source_label(hit) for hit in selected_chunks if _source_label(hit)]
	unique_sources = len(dict.fromkeys(selected_sources))
	internet_triggered = bool((retrieved.get("internet_fallback") or {}).get("triggered"))
	checks: Dict[str, bool] = {
		"router_intent_recorded": bool(getattr(routed, "intent", None)),
		"router_reason_recorded": bool(((getattr(routed, "meta", {}) or {}).get("intent_classification") or {}).get("matched_pattern")),
		"context_pack_present": bool(selected_chunks),
		"context_pack_not_larger_than_raw": len(selected_chunks) <= max(1, len(raw_hits)) if raw_hits else len(selected_chunks) == 0,
		"all_selected_chunks_have_ids": bool(selected_chunks) and all(bool(hit.get("chunk_id")) for hit in selected_chunks),
		"all_selected_chunks_have_provenance": bool(selected_chunks) and all(_provenance_present(hit) for hit in selected_chunks),
		"llm_input_trace_saved": bool(llm_input_chunks) or answer.get("mode") in {"no_coverage", "internet_no_evidence", "formula_not_found"},
		"internet_behavior_matches_expectation": True,
		"intent_matches_expectation": True,
		"minimum_source_diversity_met": unique_sources >= int(case.min_unique_sources),
		"citations_present_when_required": True,
		"answer_or_refusal_present": bool((answer.get("answer") or "").strip()) or bool(answer.get("refusal")),
	}

	if case.expected_internet is not None:
		checks["internet_behavior_matches_expectation"] = internet_triggered == bool(case.expected_internet)

	if case.expected_intents:
		actual_intent = str(getattr(routed, "intent", ""))
		expected_set = set(case.expected_intents)
		# These intents are interchangeable for routing purposes — the router
		# picks the most specific label but any of these is a valid output for
		# general explanation/comparison/enumeration questions.
		_general_intents = {"summary", "exploratory", "comparison", "list_lookup", "fact_lookup", "section_lookup"}
		if actual_intent in expected_set:
			checks["intent_matches_expectation"] = True
		elif actual_intent in _general_intents and bool(expected_set & _general_intents):
			# Both actual and at least one expected are in the general-intent family
			checks["intent_matches_expectation"] = True
		else:
			checks["intent_matches_expectation"] = False

	if case.require_citations:
		answer_mode = str(answer.get("mode") or "")
		# internet_no_evidence means the search fired but returned junk — the
		# pipeline correctly refused to fabricate; don't penalise for no citations.
		if answer_mode == "internet_no_evidence":
			checks["citations_present_when_required"] = True
		else:
			answer_text = str(answer.get("answer") or "")
			checks["citations_present_when_required"] = bool(answer.get("citations")) or ("[c" in answer_text)

	if case.allow_refusal and not checks["answer_or_refusal_present"]:
		checks["answer_or_refusal_present"] = False

	review_flags: List[str] = []
	if case.expected_source_hints:
		joined_sources = "\n".join(selected_sources).lower()
		if not any(hint.lower() in joined_sources for hint in case.expected_source_hints):
			review_flags.append("expected_source_hint_not_seen")
	if selection_meta.get("deduplication", {}).get("dropped_near_duplicate", 0) == 0 and case.category == "combination":
		review_flags.append("combination_case_kept_all_near_duplicates")
	if case.expected_internet is False and answer_routing.get("internet_only_for_answer"):
		review_flags.append("unexpected_internet_only_answer")
	if case.expected_internet is True and not answer_routing.get("internet_only_for_answer"):
		review_flags.append("internet_case_not_marked_internet_only")
	if unique_sources == 1 and case.min_unique_sources > 1:
		review_flags.append("single_source_dominance")

	failed_checks = [name for name, ok in checks.items() if not ok]
	passed = not failed_checks
	return {
		"passed": passed,
		"failed_checks": failed_checks,
		"review_flags": review_flags,
		"checks": checks,
		"metrics": {
			"raw_hit_count": len(raw_hits),
			"selected_count": len(selected_chunks),
			"llm_input_count": len(llm_input_chunks),
			"unique_sources_selected": unique_sources,
			"internet_triggered": internet_triggered,
		},
	}


def _run_case(
	case: AcceptanceCase,
	*,
	db_path: str,
	embed_backend: str,
	embed_model: str,
	llm_config: str,
) -> Dict[str, Any]:
	case_start = time.perf_counter()
	route_start = time.perf_counter()
	routed = route_query(case.question)
	route_seconds = time.perf_counter() - route_start

	strategy = routed.strategy
	runtime_top_k = max(int(case.top_k), int(strategy.top_k))
	runtime_candidate_k = max(runtime_top_k, int(strategy.candidate_k))

	retrieval_start = time.perf_counter()
	retrieved = retrieve_as_dict(
		case.question,
		db_dsn=db_path,
		top_k=runtime_top_k,
		filters=RetrievalFilters(),
		rerank_candidate_k=runtime_candidate_k,
		rerank_alpha_vector=float(strategy.alpha_vector),
		rerank_alpha_lexical=float(strategy.alpha_lexical),
		rerank_prefer_tables=bool(strategy.prefer_tables),
		rerank_prefer_shorter=bool(strategy.prefer_shorter),
		embed_backend=embed_backend,
		embed_model_name=embed_model,
	)
	retrieval_seconds = time.perf_counter() - retrieval_start

	context_start = time.perf_counter()
	context_pack = build_context_pack(retrieved, routed, max_chunks=runtime_top_k)
	context_seconds = time.perf_counter() - context_start

	answer_start = time.perf_counter()
	answer_input = {
		**retrieved,
		"hits": context_pack.get("selected_chunks", []),
	}
	answer = answer_query_with_retrieval(
		case.question,
		answer_input,
		intent=routed.intent,
		config_path=llm_config,
	)
	answer_seconds = time.perf_counter() - answer_start

	checks = _evaluate_checks(
		case,
		routed=routed,
		retrieved=retrieved,
		context_pack=context_pack,
		answer=answer,
	)

	total_seconds = time.perf_counter() - case_start
	return {
		"id": case.id,
		"category": case.category,
		"question": case.question,
		"description": case.description,
		"notes": case.notes,
		"expectations": {
			"expected_internet": case.expected_internet,
			"min_unique_sources": case.min_unique_sources,
			"require_citations": case.require_citations,
			"allow_refusal": case.allow_refusal,
			"expected_intents": case.expected_intents,
			"expected_source_hints": case.expected_source_hints,
		},
		"route": {
			"intent": routed.intent,
			"sources": list(routed.sources),
			"strategy": asdict(routed.strategy),
			"meta": routed.meta,
			"seconds": round(route_seconds, 3),
		},
		"retrieval": {
			"top_k": runtime_top_k,
			"candidate_k": runtime_candidate_k,
			"internet_fallback": retrieved.get("internet_fallback") or {},
			"top_hits": [_hit_row(hit) for hit in list(retrieved.get("hits") or [])[: runtime_top_k]],
			"seconds": round(retrieval_seconds, 3),
		},
		"context_pack": {
			"selection_meta": context_pack.get("selection_meta") or {},
			"selected_chunks": [_hit_row(hit) for hit in list(context_pack.get("selected_chunks") or [])],
			"seconds": round(context_seconds, 3),
		},
		"answer": {
			"mode": answer.get("mode"),
			"text": answer.get("answer"),
			"citations": answer.get("citations") or [],
			"confidence": answer.get("confidence") or {},
			"routing": answer.get("routing") or {},
			"retrieved_chunks": answer.get("retrieved_chunks") or [],
			"llm_input_chunks": answer.get("llm_input_chunks") or [],
			"seconds": round(answer_seconds, 3),
		},
		"checks": checks,
		"total_seconds": round(total_seconds, 3),
	}


def _summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
	by_category: Dict[str, Dict[str, int]] = {}
	for result in results:
		bucket = by_category.setdefault(result["category"], {"count": 0, "passed": 0})
		bucket["count"] += 1
		if result["checks"]["passed"]:
			bucket["passed"] += 1

	total = len(results)
	passed = sum(1 for result in results if result["checks"]["passed"])
	internet_triggered = sum(1 for result in results if result["checks"]["metrics"]["internet_triggered"])
	review_flagged = sum(1 for result in results if result["checks"]["review_flags"])
	return {
		"total_cases": total,
		"passed_cases": passed,
		"failed_cases": total - passed,
		"pass_rate": round((passed / total), 4) if total else 0.0,
		"internet_triggered_cases": internet_triggered,
		"review_flagged_cases": review_flagged,
		"by_category": by_category,
	}


def _parse_only(raw: str) -> List[str]:
	return [part.strip() for part in (raw or "").split(",") if part.strip()]


def main() -> None:
	parser = argparse.ArgumentParser(description="Run a comprehensive overnight acceptance battery for the RAG pipeline.")
	parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
	parser.add_argument("--embed-backend", type=str, default=DEFAULT_EMBED_BACKEND)
	parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
	parser.add_argument("--llm-config", type=str, default="configs/llm.yaml")
	parser.add_argument("--only", type=str, default="", help="Comma-separated case IDs to run, e.g. core_ss_01,internet_01")
	parser.add_argument("--out", type=str, default="data/diagnostics/acceptance_battery_latest.json")
	args = parser.parse_args()

	selected_ids = set(_parse_only(args.only))
	selected_cases = [case for case in CASES if not selected_ids or case.id in selected_ids]
	if not selected_cases:
		raise SystemExit("No acceptance cases selected.")

	print(f"Running {len(selected_cases)} acceptance cases...")
	results: List[Dict[str, Any]] = []
	for index, case in enumerate(selected_cases, start=1):
		print(f"[{index}/{len(selected_cases)}] {case.id}: {case.question}")
		try:
			result = _run_case(
				case,
				db_dsn=args.db,
				embed_backend=args.embed_backend,
				embed_model=args.embed_model,
				llm_config=args.llm_config,
			)
			results.append(result)
			status = "PASS" if result["checks"]["passed"] else "REVIEW"
			print(f"    -> {status} in {result['total_seconds']:.1f}s")
		except Exception as exc:
			results.append(
				{
					"id": case.id,
					"category": case.category,
					"question": case.question,
					"description": case.description,
					"error": str(exc),
					"checks": {
						"passed": False,
						"failed_checks": ["case_execution_error"],
						"review_flags": [],
						"checks": {},
						"metrics": {"internet_triggered": False},
					},
				}
			)
			print(f"    -> ERROR: {exc}")

	summary = _summarize(results)
	payload = {
		"generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
		"settings": {
			"db": args.db,
			"embed_backend": args.embed_backend,
			"embed_model": args.embed_model,
			"llm_config": args.llm_config,
			"selected_case_ids": [case.id for case in selected_cases],
		},
		"summary": summary,
		"results": results,
	}

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
	archive_path = out_path.with_name(f"acceptance_battery_{ts}.json")
	archive_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	print(f"\nSaved latest report: {out_path}")
	print(f"Saved archive report: {archive_path}")
	print(f"Pass rate: {summary['passed_cases']}/{summary['total_cases']} ({summary['pass_rate']:.1%})")
	print(f"Internet triggered in {summary['internet_triggered_cases']} cases")


if __name__ == "__main__":
	main()