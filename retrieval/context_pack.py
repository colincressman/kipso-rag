from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set, Tuple

from utils.text_utils import tokenize, tokenize_all, normalize_tokens as _normalize_text, jaccard as _jaccard
from utils.runtime_defaults import (
	DEFAULT_MAX_SCORE_GAP,
	DEFAULT_NEAR_DUP_THRESHOLD,
	DEFAULT_SOURCE_AUTHORITY,
)


_DIVERSIFY_INTENTS = {"summary", "comparison", "list_lookup", "exploratory"}


def _safe_float(v: Any, default: float = 0.0) -> float:
	try:
		return float(v)
	except Exception:
		return default


def _provenance_signature(hit: Dict[str, Any]) -> Tuple[Any, ...]:
	md = hit.get("metadata") or {}
	return (
		hit.get("chunk_id"),
		hit.get("doc_id"),
		hit.get("collection_id") or md.get("collection_id"),
		hit.get("source_name") or md.get("source_name"),
		hit.get("document_title") or md.get("document_title"),
		hit.get("document_path") or md.get("document_path"),
		hit.get("page_number") or md.get("page_number") or hit.get("page_start"),
		hit.get("section_header") or md.get("section_header") or hit.get("title"),
	)


def _deduplicate_hits(hits: List[Dict[str, Any]], near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
	kept: List[Dict[str, Any]] = []
	seen_chunk_ids: Set[str] = set()
	seen_fingerprints: Set[Tuple[Any, ...]] = set()
	stats = {"dropped_duplicate_chunk_id": 0, "dropped_near_duplicate": 0}

	for hit in hits:
		chunk_id = str(hit.get("chunk_id") or "")
		if chunk_id and chunk_id in seen_chunk_ids:
			stats["dropped_duplicate_chunk_id"] += 1
			continue

		sig = _provenance_signature(hit)
		if sig in seen_fingerprints:
			stats["dropped_near_duplicate"] += 1
			continue

		cand_toks = _normalize_text(str(hit.get("text") or ""))
		cand_doc = str(hit.get("doc_id") or "")
		is_near_dup = False
		for prior in kept:
			if str(prior.get("doc_id") or "") != cand_doc:
				continue
			prior_toks = _normalize_text(str(prior.get("text") or ""))
			if _jaccard(cand_toks, prior_toks) >= near_dup_threshold:
				is_near_dup = True
				break

		if is_near_dup:
			stats["dropped_near_duplicate"] += 1
			continue

		kept.append(hit)
		if chunk_id:
			seen_chunk_ids.add(chunk_id)
		seen_fingerprints.add(sig)

	return kept, stats


def _source_authority_for_hit(hit: Dict[str, Any], authority_map: Dict[str, float]) -> float:
	md = hit.get("metadata") or {}
	source_type = str(hit.get("source_type") or md.get("source_type") or "").strip().lower()
	if source_type:
		if source_type in authority_map:
			return float(authority_map[source_type])
		if source_type.startswith("pdf"):
			return 0.90
	return float(authority_map.get("general", 0.50))


def _rank_with_authority(
	hits: List[Dict[str, Any]],
	*,
	route_info: Dict[str, Any],
	authority_map: Dict[str, float],
	authority_weight: float,
	preferred_source_bonus: float,
) -> List[Dict[str, Any]]:
	preferred = set(route_info.get("preferred_sources") or [])
	ranked: List[Tuple[float, Dict[str, Any]]] = []
	for idx, hit in enumerate(hits):
		base_score = _safe_float(hit.get("score"), 0.0)
		authority = _source_authority_for_hit(hit, authority_map)
		bonus = preferred_source_bonus if (preferred and "corpus" in preferred) else 0.0
		combined = base_score + authority_weight * authority + bonus - (idx * 1e-6)
		item = copy.deepcopy(hit)
		md = item.setdefault("metadata", {})
		md["context_pack_base_score"] = float(base_score)
		md["context_pack_source_authority"] = float(authority)
		md["context_pack_combined_score"] = float(combined)
		ranked.append((combined, item))

	ranked.sort(key=lambda x: x[0], reverse=True)
	return [h for _, h in ranked]


def _apply_conditional_diversification(
	selected: List[Dict[str, Any]],
	candidates: List[Dict[str, Any]],
	*,
	intent: str,
	max_score_gap: float = DEFAULT_MAX_SCORE_GAP,
) -> Tuple[List[Dict[str, Any]], bool]:
	if not selected or intent not in _DIVERSIFY_INTENTS:
		return selected, False

	selected_doc_ids = {str(h.get("doc_id") or "") for h in selected}
	if len(selected_doc_ids) >= 2:
		return selected, False

	tail_score = _safe_float(selected[-1].get("score"), 0.0)
	alt: Optional[Dict[str, Any]] = None
	for cand in candidates:
		cand_doc = str(cand.get("doc_id") or "")
		if not cand_doc or cand_doc in selected_doc_ids:
			continue
		cand_score = _safe_float(cand.get("score"), 0.0)
		if (tail_score - cand_score) <= max_score_gap:
			alt = cand
			break

	if alt is None:
		return selected, False

	replaced = list(selected)
	replaced[-1] = alt
	return replaced, True


def _lost_in_middle_reorder(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Reorder chunks so the most relevant appear at the beginning and end.

	Liu et al. (2023) "Lost in the Middle" showed LLMs attend primarily to
	context boundaries and miss information placed in the middle.  This
	interleaving puts rank-0 first, rank-1 last, rank-2 second, rank-3
	second-to-last, etc., so the weakest chunks land in the centre.

	Example (5 chunks ranked [a,b,c,d,e] by score):
	  → [a, c, e, d, b]  — a/b at boundary, c/d/e in middle
	"""
	n = len(chunks)
	if n <= 2:
		return chunks
	result: List[Dict[str, Any]] = [{}] * n
	lo, hi = 0, n - 1
	for i, chunk in enumerate(chunks):
		if i % 2 == 0:
			result[lo] = chunk
			lo += 1
		else:
			result[hi] = chunk
			hi -= 1
	return result


def _apply_internet_priority(
	selected: List[Dict[str, Any]],
	ranked: List[Dict[str, Any]],
	*,
	priority_applied: bool,
	target_k: int,
) -> Tuple[List[Dict[str, Any]], bool]:
	if not priority_applied or not selected:
		return selected, False

	internet_in_selected = [h for h in selected if str(h.get("source_type") or "") == "internet"]
	if not internet_in_selected:
		best_internet = next((h for h in ranked if str(h.get("source_type") or "") == "internet"), None)
		if best_internet is None:
			return selected, False
		updated = list(selected[: max(1, int(target_k))])
		if not updated:
			updated = [best_internet]
		else:
			updated[-1] = best_internet
		internet_in_selected = [best_internet]
	else:
		updated = list(selected)

	best_internet = max(internet_in_selected, key=lambda h: _safe_float(h.get("score"), 0.0))
	remaining = [h for h in updated if h is not best_internet]
	return [best_internet] + remaining, True


def build_context_pack(
	retrieval_result: Dict[str, Any],
	routed_query: Any,
	*,
	max_chunks: Optional[int] = None,
	authority_weight: float = 0.20,
	preferred_source_bonus: float = 0.02,
) -> Dict[str, Any]:
	"""Build a post-retrieval context pack with deduplication and weighted selection.

	The output is designed for downstream answer generation and diagnostics.
	All selected chunks preserve full provenance metadata from retrieval hits.
	"""
	hits = list(retrieval_result.get("hits", []) or [])
	target_k = max_chunks if max_chunks is not None else int(retrieval_result.get("top_k") or 5)
	target_k = max(1, int(target_k))

	route_meta = dict(getattr(routed_query, "meta", {}) or {})
	llm_route = dict(route_meta.get("llm_routing") or {})
	routing_authority = str(route_meta.get("routing_authority") or "heuristic_strategy")
	llm_can_control_strategy = bool(route_meta.get("llm_can_control_strategy", False))
	use_llm_strategy = routing_authority == "llm_strategy" and llm_can_control_strategy

	base_route_type = getattr(routed_query, "intent", None)
	if use_llm_strategy:
		base_route_type = llm_route.get("route_type") or base_route_type

	base_preferred_sources = list(getattr(routed_query, "sources", []) or [])
	if use_llm_strategy and llm_route.get("preferred_sources"):
		base_preferred_sources = llm_route.get("preferred_sources") or base_preferred_sources

	effective_route: Dict[str, Any] = {
		"route_type": base_route_type,
		"preferred_sources": base_preferred_sources,
		"confidence": _safe_float(llm_route.get("confidence"), 0.0),
		"valid": bool(llm_route.get("valid", False)),
		"advisory_route_type": llm_route.get("route_type"),
		"advisory_preferred_sources": llm_route.get("preferred_sources") or [],
		"use_llm_strategy": use_llm_strategy,
	}

	deduped, dedup_stats = _deduplicate_hits(hits)
	ranked = _rank_with_authority(
		deduped,
		route_info=effective_route,
		authority_map=DEFAULT_SOURCE_AUTHORITY,
		authority_weight=authority_weight,
		preferred_source_bonus=preferred_source_bonus,
	)

	selected = ranked[:target_k]
	selected, diversified = _apply_conditional_diversification(
		selected,
		ranked[target_k:],
		intent=str(effective_route.get("route_type") or ""),
	)

	internet_fallback = dict(retrieval_result.get("internet_fallback") or {})
	internet_priority_flag = bool(internet_fallback.get("priority_applied", False))
	selected, internet_priority_honored = _apply_internet_priority(
		selected,
		ranked,
		priority_applied=internet_priority_flag,
		target_k=target_k,
	)

	presented = _lost_in_middle_reorder(selected)

	return {
		"query": retrieval_result.get("query"),
		"target_k": target_k,
		"selected_chunks": presented,
		"selection_meta": {
			"original_hit_count": len(hits),
			"deduped_hit_count": len(deduped),
			"selected_count": len(selected),
			"deduplication": dedup_stats,
			"authority_weighting_applied": True,
			"conditional_diversification_applied": bool(diversified),
			"internet_priority_requested": bool(internet_priority_flag),
			"internet_priority_honored": bool(internet_priority_honored),
			"lost_in_middle_reorder_applied": len(selected) > 2,
			"route_signal": effective_route,
			"routing_authority": routing_authority,
			"llm_can_control_strategy": llm_can_control_strategy,
		},
	}
