"""Query-classification and confidence-band helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from utils.text_utils import tokenize
from llm.extraction import path_title_match_override, query_keywords, query_terms
from llm.grounding import ANSWER_STOPWORDS

from utils.text_utils import YEAR_RE

_OVERVIEW_PATTERNS = re.compile(
	r"""
	(?:
		# "what is/does this/the book/document/paper/chapter about"
		\b what \s+ (is|does) \s+ (this|the) \s+ \w+ \s+ about \b
		|
		# "what is this about"
		\b what \s+ is \s+ this \s+ about \b
		|
		# "what is [Name of Book] about"
		\b what \s+ is \b .{1,80} \b about \b
		|
		# "what does this book cover"
		\b what \s+ does \s+ (this|the) \s+ \w+ \s+ cover \b
		|
		# "summarize (this|the)" or "give me a summary"
		\b (summarize|summarise) \b
		|
		\b give \s+ (me|us) \s+ (a|an) \s+ (brief\s+)? (summary|overview|synopsis) \b
		|
		# "overview of" or "high.level overview"
		\b overview \s+ of \b
		|
		# "what topics does this cover"
		\b what \s+ topics \b
		|
		# "what is covered in"
		\b what \s+ is \s+ covered \b
		|
		# "describe this book/document"
		\b describe \s+ (this|the) \s+ (book|document|paper|article|chapter|text) \b
		|
		# "tell me about this book"
		\b tell \s+ (me|us) \s+ about \s+ (this|the) \s+ (book|document|paper|article|chapter|text) \b
	)
	""",
	re.IGNORECASE | re.VERBOSE,
)


def is_overview_query(query: str, intent: Optional[str] = None) -> bool:
	"""Return True for broad overview/summary questions that need synthesized prose.

	When ``intent`` is provided (from the routing pipeline), use it directly —
	no regex needed.  Falls back to ``_OVERVIEW_PATTERNS`` only for standalone
	callers (tests, scripts) where routing context is unavailable.
	"""
	if intent == "summary":
		return True
	return bool(_OVERVIEW_PATTERNS.search((query or "").strip()))


FACTOID_HINTS = {
	"isbn",
	"publisher",
	"published",
	"publication",
	"year",
	"author",
	"authors",
	"title",
	"chapter",
	"section",
	"where",
	"when",
	"who",
	"define",
	"definition",
	"meaning",
}
METADATA_HINTS = {"isbn", "publisher", "published", "publication", "author", "authors", "title"}
FACT_QUERY_OPENERS = {"what", "who", "when", "where", "which", "whom"}


def is_external_fact_query(query: str) -> bool:
	q = (query or "").strip()
	if not q:
		return False
	tokens = tokenize(q)
	if not tokens:
		return False
	if tokens[0].lower() not in FACT_QUERY_OPENERS:
		return False
	has_year = bool(YEAR_RE.search(q))
	has_current_hint = "current" in {t.lower() for t in tokens}
	has_entity_hint = any(
		(tok.isupper() and len(tok) >= 2)
		or (tok[:1].isupper() and len(tok) >= 3 and tok.lower() not in ANSWER_STOPWORDS)
		for tok in tokens[1:]
	)
	return bool(has_year or has_current_hint or has_entity_hint)


def is_factoid_query(query: str) -> bool:
	q = (query or "").strip().lower()
	if not q:
		return False
	tokens = tokenize(q)
	if tokens and tokens[0] in {"what", "who", "when", "where", "which"} and len(tokens) <= 14:
		return True
	if any(tok in FACTOID_HINTS for tok in tokens):
		return True
	if len(tokens) <= 8 and q.endswith("?"):
		return True
	return False


def is_metadata_fact_query(query: str) -> bool:
	q = (query or "").lower()
	tokens = set(tokenize(q))
	return bool(tokens.intersection(METADATA_HINTS))


def determine_confidence_band(
	query: str,
	hits: List[Dict[str, Any]],
	decision_cfg: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
	if not hits:
		return "low", {"reason": "no_hits", "rule": "no_hits"}

	top = hits[0]
	top_score = float(top.get("score") or 0.0)
	metadata = top.get("metadata") or {}
	if isinstance(metadata.get("score_gap_to_second"), (int, float)):
		score_gap = float(metadata.get("score_gap_to_second"))
	elif len(hits) > 1:
		score_gap = float(top_score - float(hits[1].get("score") or 0.0))
	else:
		score_gap = 1.0

	min_conf = float(decision_cfg.get("medium_confidence_score", 0.55))
	high_conf = float(decision_cfg.get("high_confidence_score", 0.70))
	borderline = float(decision_cfg.get("borderline_confidence_score", 0.62))
	max_gap = float(decision_cfg.get("max_ambiguous_gap", 0.03))
	top_role = str(top.get("structural_role") or metadata.get("structural_role") or "body").lower()
	is_metadata_q = is_metadata_fact_query(query)
	is_broad_explanatory = bool(re.search(r"\b(explain|summary|summarize|overview|describe|in\s+detail)\b", (query or "").lower()))
	override = path_title_match_override(
		query,
		top,
		int(decision_cfg.get("path_override_min_term_matches", 1)),
	)

	if top_score < min_conf:
		return "low", {"top_score": top_score, "score_gap": score_gap, "override": False, "rule": "top_score_below_medium"}

	if override:
		if top_score >= high_conf:
			if top_role in {"index_noise", "frontmatter"} and not is_metadata_q:
				return "medium", {
					"top_score": top_score,
					"score_gap": score_gap,
					"override": True,
					"rule": "override_demoted_aux_role",
				}
			return "high", {"top_score": top_score, "score_gap": score_gap, "override": True, "rule": "override_header_match"}
		return "medium", {"top_score": top_score, "score_gap": score_gap, "override": True, "rule": "override_header_match"}
	if top_score >= high_conf:
		if top_role in {"index_noise", "frontmatter"} and not is_metadata_q:
			return "medium", {
				"top_score": top_score,
				"score_gap": score_gap,
				"override": False,
				"rule": "top_score_high_demoted_aux_role",
			}
		if is_broad_explanatory and score_gap < max(max_gap, 0.06):
			return "medium", {
				"top_score": top_score,
				"score_gap": score_gap,
				"override": False,
				"rule": "top_score_high_demoted_low_gap_broad_query",
			}
		return "high", {"top_score": top_score, "score_gap": score_gap, "override": False, "rule": "top_score_high"}
	if top_score < borderline and score_gap < max_gap:
		return "medium", {
			"top_score": top_score,
			"score_gap": score_gap,
			"override": False,
			"rule": "medium_ambiguous",
		}
	return "medium", {"top_score": top_score, "score_gap": score_gap, "override": False, "rule": "top_score_medium"}
