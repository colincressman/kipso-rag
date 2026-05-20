"""Acronym mining and query expansion for the retrieval layer.

Mines acronym-to-phrase mappings directly from corpus chunk text, then uses
them to expand short or acronym-heavy queries before embedding and reranking.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import psycopg

from utils.text_utils import tokenize

# Small stop-word set used to filter trivial words from acronym matching and
# query-overlap bonus calculations.
STOPWORDS = {"the", "a", "an", "of", "and", "for", "to", "in", "on", "at", "by", "with"}

ACRONYM_TOKEN_RE = re.compile(r"^[A-Za-z]{2,8}$")
PHRASE_WITH_ACRONYM_RE = re.compile(r"\b([A-Za-z][A-Za-z\- ]{3,80}?)\s*\(([A-Z]{2,10})\)")
ACRONYM_WITH_PHRASE_RE = re.compile(r"\b([A-Z]{2,10})\s*\(([A-Za-z][A-Za-z\- ]{3,80}?)\)")


def _tokenize_words(text: str) -> List[str]:
	return tokenize(text or "")


def _normalize_phrase(phrase: str) -> str:
	return " ".join(_tokenize_words(phrase.lower())).strip()


def _initials(phrase: str) -> str:
	words = [w for w in _tokenize_words(phrase) if len(w) > 1 and w.lower() not in STOPWORDS]
	return "".join(w[0].upper() for w in words)


def _acronym_matches_phrase(acronym: str, phrase: str) -> bool:
	initials = _initials(phrase)
	acr = acronym.upper()
	return initials == acr or initials.endswith(acr)


def _mine_acronym_expansions(conn: psycopg.Connection) -> Dict[str, List[str]]:
	"""Mine acronym expansions from chunk text/title/path patterns in the corpus."""
	counts: Dict[str, Counter[str]] = defaultdict(Counter)
	rows = conn.execute("SELECT title, path_text, text FROM chunks").fetchall()
	for r in rows:
		blob = "\n".join([str(r["title"] or ""), str(r["path_text"] or ""), str(r["text"] or "")])

		for m in PHRASE_WITH_ACRONYM_RE.finditer(blob):
			phrase = _normalize_phrase(m.group(1))
			acr = m.group(2).upper()
			if len(phrase.split()) >= 2 and _acronym_matches_phrase(acr, phrase):
				counts[acr][phrase] += 1

		for m in ACRONYM_WITH_PHRASE_RE.finditer(blob):
			acr = m.group(1).upper()
			phrase = _normalize_phrase(m.group(2))
			if len(phrase.split()) >= 2 and _acronym_matches_phrase(acr, phrase):
				counts[acr][phrase] += 1

	result: Dict[str, List[str]] = {}
	for acr, counter in counts.items():
		best = [p for p, _ in counter.most_common(3)]
		if best:
			result[acr] = best
	return result


def _expand_query_with_acronyms(query: str, conn: psycopg.Connection) -> Tuple[str, Dict[str, Any]]:
	"""Expand short/acronym-heavy queries using corpus-mined acronym dictionary."""
	tokens = _tokenize_words(query)
	if not tokens:
		return query, {"expanded": False, "expansions": {}}

	acronym_map = _mine_acronym_expansions(conn)
	expansions: Dict[str, List[str]] = {}
	query_terms = set(_tokenize_words(query.lower()))
	parts: List[str] = [query]

	for tok in tokens:
		candidate = tok.upper()
		if not ACRONYM_TOKEN_RE.match(tok):
			continue
		# Only expand explicit acronym-like tokens, not ordinary lowercase words
		# such as "is" or "do" that happen to match the acronym token regex.
		if tok.lower() in STOPWORDS or not tok.isupper():
			continue
		if candidate not in acronym_map:
			continue

		choices = []
		for phrase in acronym_map[candidate]:
			p_terms = set(_tokenize_words(phrase.lower()))
			if p_terms and not p_terms.issubset(query_terms):
				choices.append(phrase)
		if choices:
			expansions[candidate] = choices
			parts.extend(choices[:2])

	expanded_query = " ".join(parts)
	return expanded_query, {
		"expanded": bool(expansions),
		"expansions": expansions,
		"original_query": query,
		"expanded_query": expanded_query,
	}
