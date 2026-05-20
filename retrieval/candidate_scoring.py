"""Query classification and candidate scoring helpers for the retrieval layer.

Provides:
  - Intent-detection predicates: metadata, numeric, external-fact queries
  - Structural role score adjustments based on query type
  - Query-overlap bonus for lexical signal
  - Lexical-only candidate scoring (cross-encoder-only mode)
"""

from __future__ import annotations

from typing import Dict, List, Set

from utils.text_utils import tokenize
from utils.text_utils import YEAR_RE
from retrieval.acronym_expand import STOPWORDS

# ---------------------------------------------------------------------------
# Query-type vocabularies
# ---------------------------------------------------------------------------

METADATA_QUERY_TERMS: Set[str] = {
	"isbn",
	"publisher",
	"published",
	"publication",
	"year",
	"author",
	"authors",
	"title",
	"edition",
}

# Query terms that indicate the user wants a book-overview / table-of-contents answer.
OVERVIEW_QUERY_TERMS: Set[str] = {
	"topics", "topic", "chapters", "chapter", "contents", "outline",
	"covered", "covers", "about", "overview", "sections", "table",
}

NUMERIC_QUERY_TERMS: Set[str] = {
	"calculate", "compute", "equation", "formula", "derive", "proof",
	"odds", "probability", "logit", "coef", "coefficient", "regression",
	"value", "estimate", "fit",
}

FACT_QUERY_OPENERS: Set[str] = {"who", "when", "where", "which", "whom"}

# Openers that indicate an information request (not ML-corpus questions)
INFORMATIONAL_OPENERS = frozenset({"tell", "explain", "describe", "show", "give", "find", "list"})

# Terms that signal the user wants real-world / current-events information.
# Keep narrow: avoid terms that appear heavily in ML/math questions.
NEWS_INTENT_TERMS = frozenset({
	"news", "politics", "political", "election", "elections", "happening",
	# sports / competitions — specific enough to not fire on ML corpus questions
	"world\u00a0cup", "championship", "tournament", "league", "playoffs", "semifinal",
	"final", "olympics", "superbowl", "premier\u00a0league", "winner", "champion",
	"champions", "scored", "standings",
})

# Single tokens that alone make a query time-sensitive regardless of opener.
RECENCY_KEYWORDS = frozenset({
	"today", "yesterday", "tonight", "currently", "nowadays",
	"last",    # "who won the last world cup", "who was the last president"
	"latest",  # "who is the latest champion"  (safe under who/when/where)
	"recent",  # "where was the recent summit"
	"now",     # "who is president now"
})

# Product/model release verbs: "what has [Company] released/launched/unveiled/announced"
PRODUCT_RELEASE_VERBS = frozenset({
	"released", "release",
	"launched", "launch",
	"unveiled", "unveil",
	"announced", "announce",
})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Query-type predicates
# ---------------------------------------------------------------------------

def _is_metadata_query(query: str) -> bool:
	tokens = {t.lower() for t in tokenize(query)}
	return bool(tokens.intersection(METADATA_QUERY_TERMS))


def _is_numeric_query(query: str) -> bool:
	q = (query or "").lower()
	tokens = {t.lower() for t in tokenize(q)}
	if any(ch.isdigit() for ch in q):
		return True
	if bool(tokens.intersection(NUMERIC_QUERY_TERMS)) and any(
		t in tokens for t in {"calculate", "compute", "derive", "equation", "formula"}
	):
		return True
	return False


def _is_external_fact_query(query: str) -> bool:
	q = (query or "").strip()
	if not q:
		return False
	tokens = tokenize(q)
	if not tokens:
		return False
	opener = tokens[0].lower()
	has_year = bool(YEAR_RE.search(q))
	has_entity_hint = any(tok.isupper() and len(tok) >= 2 for tok in tokens)
	tokens_lower = {t.lower() for t in tokens}
	has_recency_word = bool(tokens_lower & RECENCY_KEYWORDS)
	has_news_intent = bool(tokens_lower & NEWS_INTENT_TERMS)
	# Detect proper-noun entities: title-case tokens after the first position
	has_proper_noun = any(
		tok[0].isupper() and not tok.isupper()
		for tok in tokens[1:]
		if len(tok) >= 2
	)
	if opener in FACT_QUERY_OPENERS:
		# who/when/where/which/whom + year, all-caps entity, proper noun, or recency word.
		return bool(has_year or has_entity_hint or has_proper_noun or has_recency_word)
	if opener == "what":
		# "what" is common for ML questions — only fire on strong recency/news signal,
		# or a product-release query ("what has/did [Company] release/announce/...").
		recency_strong = bool(tokens_lower & {"today", "yesterday", "tonight", "currently", "nowadays", "now"})
		# "what has/did [Entity] release/announce/launch/unveil" — requires a proper noun
		# AND a product-release verb (any tense/form).
		has_product_release = bool(
			has_proper_noun and (tokens_lower & PRODUCT_RELEASE_VERBS)
		)
		return bool(recency_strong or has_news_intent or has_product_release)
	if opener in INFORMATIONAL_OPENERS:
		# "tell me about X today" / "explain the news" — require recency or news intent
		# to avoid false-triggering on "explain backpropagation"
		return bool(has_recency_word or has_news_intent)
	# Any query containing explicit news/current-events intent, regardless of opener
	if has_news_intent:
		return True
	return False


# ---------------------------------------------------------------------------
# Candidate scoring adjustments
# ---------------------------------------------------------------------------

def _structural_role_score(query: str, row: dict) -> float:
	"""
	Adjust chunk score based on structural role — replaces the old set of
	hand-coded scoring shims.  Logic is entirely generic (no corpus-specific
	terms).

	  metadata queries   → prioritise 'metadata' chunks, penalise others
	  overview queries   → prioritise 'heading_overview' / 'frontmatter'
	  substantive queries → penalise 'metadata' and 'index_noise' noise
	"""
	role = str(row["structural_role"] or "body")
	token_est = int(row["token_count_est"] or 0)

	is_meta_q = _is_metadata_query(query)
	query_tokens = {t.lower() for t in tokenize(query)}
	is_overview_q = len(query_tokens.intersection(OVERVIEW_QUERY_TERMS)) >= 2
	is_numeric_q = _is_numeric_query(query)

	adj = 0.0

	if is_meta_q:
		if role == "metadata":
			adj = 0.60
		elif role == "frontmatter":
			adj = 0.08
		elif role == "toc":
			adj = 0.25  # TOC pages are directly useful for structure/metadata questions
		elif role in {"index_noise", "body", "table_data", "heading_overview"}:
			adj = -0.20

	elif is_overview_q:
		if role in {"document_summary", "page_range_summary"}:
			adj = 0.25  # Summaries are ideal for overview queries
		elif role == "heading_overview":
			adj = 0.20
		elif role == "frontmatter":
			adj = 0.12
		elif role == "toc":
			adj = 0.15  # TOC is useful for "what does this book cover?" queries
		elif role == "body":
			page = int(row["page_start"] or -1)
			if page > 30:
				adj = -0.06

	else:
		# Substantive query — push noise roles down
		if role in {"document_summary", "page_range_summary"}:
			adj = -0.15  # Penalise synthetic summaries — avoid LLM reading back its own output
		elif role == "metadata":
			adj = -0.12
		elif role == "index_noise":
			adj = -0.08
		elif role == "toc":
			adj = -0.30  # Strong penalty: CE is also excluded for toc chunks
		elif role == "table_data" and not is_numeric_q:
			adj = -0.06
		elif role == "frontmatter":
			page = int(row["page_start"] or -1)
			if page <= 20:
				adj = -0.04

	# Short stub penalty: title-only / near-empty chunks add no answer substance.
	# header_bonus and lexical bonuses over-inflate them in the reranker, so
	# suppress them early.
	if 0 < token_est < 20:
		adj -= 0.20

	return adj


def _query_overlap_bonus(query: str, row: dict) -> float:
	"""Small lexical match bonus to reduce semantic drift on broad queries."""
	query_terms = {
		t for t in tokenize((query or "").lower())
		if t not in STOPWORDS and len(t) > 3
	}
	if not query_terms:
		return 0.0

	blob = " ".join([
		str(row["title"] or "").lower(),
		str(row["path_text"] or "").lower(),
		str(row["text"] or "").lower(),
	])
	matches = sum(1 for term in query_terms if term in blob)
	if matches <= 0:
		return 0.0
	return min(0.015 * matches, 0.10)


def _lexical_candidate_score(query: str, row: dict) -> float:
	"""Lexical-only candidate score for cross-encoder-only retrieval mode."""
	query_terms = [
		t for t in tokenize((query or "").lower())
		if t not in STOPWORDS and len(t) > 2
	]
	if not query_terms:
		return 0.0

	blob = " ".join([
		str(row["title"] or "").lower(),
		str(row["path_text"] or "").lower(),
		str(row["text"] or "").lower(),
	])
	if not blob:
		return 0.0

	matches = sum(1 for term in query_terms if term in blob)
	coverage = matches / max(1, len(set(query_terms)))
	return float(coverage)
