"""Shared query tokenisation helpers used across extraction sub-modules."""

from __future__ import annotations

from utils.text_utils import tokenize
from llm.grounding import ANSWER_STOPWORDS


def query_terms(query: str) -> set[str]:
	return set(tokenize((query or "").lower()))


def query_keywords(query: str) -> set[str]:
	terms = query_terms(query)
	return {t for t in terms if t not in ANSWER_STOPWORDS and len(t) > 2}
