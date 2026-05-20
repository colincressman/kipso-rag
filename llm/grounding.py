"""Grounding and lexical coverage checks for RAG answers."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from llm.citations import (
	collect_source_text_for_validation,
	strip_trailing_citations_block,
)
from utils.text_utils import tokenize
from utils.runtime_defaults import (
	DEFAULT_ENTITY_SUPPORT_THRESHOLD,
	DEFAULT_MIN_COVERAGE_SCORE,
	DEFAULT_MIN_LEXICAL_COVERAGE,
)

ANSWER_STOPWORDS: frozenset[str] = frozenset({
	"the", "a", "an", "of", "and", "for", "to", "in", "on", "at",
	"by", "with", "is", "are", "was", "were", "be", "does", "do",
	"what", "who", "when", "where", "why", "how",
})

ENTITY_TOKEN_RE = re.compile(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,10})\b")

_MIN_COVERAGE_SCORE: float = DEFAULT_MIN_COVERAGE_SCORE
_MIN_LEXICAL_COVERAGE: float = DEFAULT_MIN_LEXICAL_COVERAGE

_COVERAGE_EXTRA_STOPWORDS: frozenset[str] = frozenset({
	"book", "chapter", "section", "page", "text", "document",
	"say", "says", "tell", "tells", "explain", "explains",
	"describe", "describes", "discuss", "discusses", "according",
	"states", "about", "cover", "covers", "mention", "mentions",
	"define", "defines", "mean", "means", "meaning", "overview",
	"basics", "explanation", "detail", "details", "summary",
	"main", "key", "primary", "overall", "goal", "take", "advantage", "full", "management",
	# conversational scaffolding — function words that are never corpus topic terms
	"can", "you", "get", "please", "help", "show", "find", "need",
	"want", "know", "give", "information", "more", "could", "would",
})

_ENTITY_GROUNDING_IGNORE: set[str] = {
	# stopwords / sentence-starters that ENTITY_TOKEN_RE matches
	"the", "this", "that", "these", "those", "and", "but", "for",
	"with", "from", "into", "low", "high", "answer", "citations",
	# generic terms that are always capitalised in headings/titles
	"model", "models", "method", "methods", "approach", "approaches",
	"chapter", "section", "figure", "table", "note", "notes",
	"introduction", "conclusion", "summary", "overview", "example",
	"definition", "theorem", "proof", "result", "results",
	"algorithm", "function", "equation", "formula", "matrix",
	# domain-generic terms that appear capitalised in technical writing
	"machine", "learning", "neural", "network", "networks", "deep",
	"data", "training", "test", "loss", "gradient", "weight", "weights",
	"investment", "fund", "funds", "market", "return", "returns",
	"risk", "asset", "assets", "portfolio", "equity",
	"really",  # title word from "What Hedge Funds Really Do"
}


def term_variants(term: str) -> List[str]:
	base = (term or "").strip().lower()
	if not base:
		return []
	variants = {base}
	if base.endswith("ies") and len(base) > 4:
		variants.add(base[:-3] + "y")
	if base.endswith("s") and len(base) > 4:
		variants.add(base[:-1])
	if base.endswith("ing") and len(base) > 5:
		variants.add(base[:-3])
	if base.endswith("ed") and len(base) > 4:
		variants.add(base[:-2])
	return [v for v in variants if v]


def term_present_in_text(term: str, text: str) -> bool:
	text_l = (text or "").lower()
	for candidate in term_variants(term):
		if candidate and candidate in text_l:
			return True
	return False


def safe_no_coverage_answer(query: str, intent: str | None, top_score: float) -> str:
	"""Return an honest, non-fabricated response when the corpus lacks coverage."""
	if intent == "formula_lookup":
		return (
			"The formula or equation you asked about was not found in the loaded documents. "
			"I cannot derive or produce it from the available context."
		)
	if top_score < _MIN_COVERAGE_SCORE:
		return (
			"This topic does not appear to be covered in the loaded documents. "
			"I cannot answer without relevant source material — no answer fabricated."
		)
	return (
		"The loaded documents do not appear to contain information about this specific topic. "
		"I cannot produce a grounded answer \u2014 no answer fabricated."
	)


def lexical_coverage_score(query: str, hits: List[Dict[str, Any]], top_n: int = 3) -> float:
	"""
	Fraction of query topic-keywords present in the best single retrieved chunk.

	Returns a value in [0.0, 1.0]; returns 1.0 when no topic terms can be
	extracted so the gate never fires on term-free queries.
	"""
	all_stops = ANSWER_STOPWORDS | _COVERAGE_EXTRA_STOPWORDS
	terms = [
		t for t in tokenize((query or "").lower())
		if t not in all_stops and len(t) > 2
	]
	if not terms:
		return 1.0

	if len(terms) == 1:
		return 1.0

	top_text_len = len(str(hits[0].get("text") or "")) if hits else 0
	if top_text_len < 40:
		return 1.0

	if len(terms) == 2:
		most_specific = max(terms, key=len)
		for h in hits[:top_n]:
			if term_present_in_text(most_specific, str(h.get("text") or "")):
				return 1.0
		return 0.0

	best = 0.0
	aggregated_text = "\n".join(str(h.get("text") or "") for h in hits[:top_n])
	aggregated_match = sum(1 for t in terms if term_present_in_text(t, aggregated_text))
	aggregated_ratio = aggregated_match / len(terms)
	for h in hits[:top_n]:
		text = str(h.get("text") or "")
		matched = sum(1 for t in terms if term_present_in_text(t, text))
		ratio = matched / len(terms)
		if ratio > best:
			best = ratio

	return max(best, aggregated_ratio)


def unsupported_answer_entities(
	answer: str,
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
) -> List[str]:
	body = strip_trailing_citations_block(answer)
	if not body:
		return []

	source_text = collect_source_text_for_validation(hits, citation_ids)
	if not source_text:
		return []

	query_terms = {t.lower() for t in tokenize(query or "")}

	all_entities = sorted(set(ENTITY_TOKEN_RE.findall(body)))
	unsupported: List[str] = []
	for ent in all_entities:
		ent_l = ent.lower()
		if ent_l in _ENTITY_GROUNDING_IGNORE or ent_l in query_terms:
			continue
		if ent_l in source_text:
			continue
		unsupported.append(ent)

	if len(all_entities) > 0 and len(unsupported) < 3:
		return []
	if len(all_entities) > 0 and len(unsupported) / len(all_entities) < DEFAULT_ENTITY_SUPPORT_THRESHOLD:
		return []

	return unsupported


def sentence_faithfulness_scores(
	answer: str,
	hits: List[Dict[str, Any]],
	*,
	threshold: float = 0.35,
) -> List[Dict[str, Any]]:
	"""Score each answer sentence by its lexical support in the retrieved chunks.

	For each sentence with ≥3 content terms, compute the fraction of those
	terms that appear in any retrieved chunk text.  Sentences below *threshold*
	are flagged as potentially unsupported.

	Returns a list of ``{sentence, coverage, flagged}`` dicts.  Sentences that
	are too short to score reliably (fewer than 3 content terms after stopword
	removal) are omitted from the output.

	This is a lexical faithfulness check — a fast complement to entity
	grounding that works at sentence granularity without requiring an NLI model.
	"""
	from llm.citations import SENTENCE_RE, strip_trailing_citations_block  # noqa: PLC0415

	if not answer or not hits:
		return []

	all_stops = ANSWER_STOPWORDS | _COVERAGE_EXTRA_STOPWORDS
	body = strip_trailing_citations_block(answer)
	# Remove citation tags [cNNNNNN] before scoring
	body = re.sub(r"\[[^\[\]]{1,20}\]", "", body)

	raw_sentences = SENTENCE_RE.split(body.strip())
	if not raw_sentences:
		return []

	context = "\n".join(str(h.get("text") or "") for h in hits)

	results: List[Dict[str, Any]] = []
	for sent in raw_sentences:
		sent = sent.strip()
		if not sent:
			continue
		terms = [
			t for t in tokenize(sent.lower())
			if t not in all_stops and len(t) > 2
		]
		if len(terms) < 3:
			continue
		matched = sum(1 for t in terms if term_present_in_text(t, context))
		coverage = matched / len(terms)
		results.append({
			"sentence": sent,
			"coverage": round(coverage, 3),
			"flagged": coverage < threshold,
		})

	return results
