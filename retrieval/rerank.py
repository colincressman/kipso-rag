"""
Simple reranking helpers.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from utils.text_utils import jaccard, tokenize
from utils.runtime_defaults import (
	DEFAULT_RERANK_ALPHA_LEXICAL,
	DEFAULT_RERANK_ALPHA_VECTOR,
	DEFAULT_RERANK_DIVERSITY_PENALTY,
	DEFAULT_RERANK_EXACT_PHRASE_BONUS,
	DEFAULT_RERANK_HEADER_BONUS_CAP,
	DEFAULT_RERANK_HEADER_BONUS_PER_MATCH,
	DEFAULT_RERANK_KEYWORD_BONUS_CAP,
	DEFAULT_RERANK_KEYWORD_BONUS_PER_MATCH,
	DEFAULT_RERANK_SHORT_STUB_PENALTY,
)


def _token_set(text: str) -> set[str]:
	return set(tokenize((text or "").lower()))


def rerank_by_preference(
	hits: List[Dict[str, Any]],
	*,
	prefer_tables: bool = False,
	prefer_shorter: bool = False,
) -> List[Dict[str, Any]]:
	"""
	Rule-based reranker for quick tuning without extra models.
	"""
	def score(h: Dict[str, Any]) -> float:
		base = float(h.get("score", 0.0))
		md = h.get("metadata", {}) or {}
		if prefer_tables and md.get("has_table"):
			base += 0.03
		if prefer_shorter:
			tok = md.get("token_count_est") or 0
			if tok and tok < 250:
				base += 0.02
		return base

	out = list(hits)
	out.sort(key=score, reverse=True)
	return out


def rerank_by_query(
	query: str,
	hits: List[Dict[str, Any]],
	*,
	alpha_vector: float = DEFAULT_RERANK_ALPHA_VECTOR,
	alpha_lexical: float = DEFAULT_RERANK_ALPHA_LEXICAL,
	prefer_tables: bool = False,
	prefer_shorter: bool = False,
	keyword_bonus_per_match: float = DEFAULT_RERANK_KEYWORD_BONUS_PER_MATCH,
	keyword_bonus_cap: float = DEFAULT_RERANK_KEYWORD_BONUS_CAP,
	header_bonus_per_match: float = DEFAULT_RERANK_HEADER_BONUS_PER_MATCH,
	header_bonus_cap: float = DEFAULT_RERANK_HEADER_BONUS_CAP,
	exact_phrase_bonus: float = DEFAULT_RERANK_EXACT_PHRASE_BONUS,
	diversity_penalty: float = DEFAULT_RERANK_DIVERSITY_PENALTY,
	max_select: Optional[int] = None,
	progress_fn: Optional[Callable[[str], None]] = None,
) -> List[Dict[str, Any]]:
	"""
	Query-aware reranker over retrieved candidates.

	final_score = alpha_vector * vector_score + alpha_lexical * lexical_jaccard + rule_boost
	"""
	qset = _token_set(query)
	query_norm = " ".join(tokenize((query or "").lower()))
	query_term_count = len(qset)

	def score(h: Dict[str, Any]) -> float:
		vector_score = float(h.get("score", 0.0))
		text = str(h.get("text", ""))
		text_norm = " ".join(tokenize(text.lower()))
		cset = _token_set(text)
		inter = len(qset.intersection(cset))
		lexical_jaccard = jaccard(qset, cset)
		exact_match_count = inter
		keyword_bonus = min(keyword_bonus_cap, exact_match_count * keyword_bonus_per_match)

		rule_boost = 0.0
		md = h.get("metadata", {}) or {}
		title = str(h.get("title") or "")
		path_text = str(h.get("path_text") or "")
		header_tokens = _token_set(f"{title} {path_text}")
		header_match_count = len(qset.intersection(header_tokens))
		header_bonus = min(header_bonus_cap, header_match_count * header_bonus_per_match)
		phrase_bonus = 0.0
		if query_term_count >= 2 and query_norm and query_norm in text_norm:
			phrase_bonus = exact_phrase_bonus
		if prefer_tables and md.get("has_table"):
			rule_boost += 0.03
		if prefer_shorter:
			tok = md.get("token_count_est") or 0
			if tok and tok < 250:
				rule_boost += 0.02

		# Short-stub penalty: title-only fragments (token_count_est < 20) earn
		# inflated header/lexical bonuses because their text IS the title.
		# Apply a final-score penalty so they don't displace substantive chunks.
		if 0 < (md.get("token_count_est") or 0) < 20:
			rule_boost -= DEFAULT_RERANK_SHORT_STUB_PENALTY

		final = (
			alpha_vector * vector_score
			+ alpha_lexical * lexical_jaccard
			+ keyword_bonus
			+ header_bonus
			+ phrase_bonus
			+ rule_boost
		)
		md["vector_score"] = vector_score
		md["lexical_jaccard"] = lexical_jaccard
		md["keyword_bonus"] = keyword_bonus
		md["header_bonus"] = header_bonus
		md["header_match_count"] = header_match_count
		md["exact_phrase_bonus"] = phrase_bonus
		md["rerank_score"] = final
		h["metadata"] = md
		h["score"] = final
		return final

	out = list(hits)
	out.sort(key=score, reverse=True)

	# MMR-style novelty pass: penalize near-duplicate chunks so top-k covers
	# more distinct evidence snippets. Stops early at max_select to avoid
	# O(n²) work when the caller only needs the top N results.
	# Skip entirely when diversity_penalty=0 — already sorted above, no MMR needed.
	_pg = progress_fn or (lambda _: None)
	_limit = max_select if (max_select is not None and max_select > 0) else len(out)
	if float(diversity_penalty) == 0.0:
		return out[:_limit] if _limit < len(out) else out

	_total = len(out)
	_report_every = max(1, _limit // 10)  # ~10 progress ticks
	selected: List[Dict[str, Any]] = []
	remaining = list(out)
	while remaining and len(selected) < _limit:
		best_idx = 0
		best_adj = -1e9
		for idx, cand in enumerate(remaining):
			cand_text = str(cand.get("text") or "")
			cand_tokens = _token_set(cand_text)
			max_sim = 0.0
			for chosen in selected:
				chosen_tokens = _token_set(str(chosen.get("text") or ""))
				union = len(cand_tokens.union(chosen_tokens)) or 1
				sim = len(cand_tokens.intersection(chosen_tokens)) / union
				if sim > max_sim:
					max_sim = sim
			adj = float(cand.get("score", 0.0)) - float(diversity_penalty) * max_sim
			if adj > best_adj:
				best_adj = adj
				best_idx = idx

		chosen = remaining.pop(best_idx)
		md = chosen.get("metadata", {}) or {}
		md["diversity_adjusted_score"] = best_adj
		chosen["metadata"] = md
		selected.append(chosen)
		if len(selected) % _report_every == 0:
			_pg(f"  → Reranking {len(selected)} / {_limit}…")

	# Append any unprocessed tail when early-exit fires (already score-sorted)
	selected.extend(remaining)
	return selected

