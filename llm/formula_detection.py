"""Formula / mathematical content detection helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from llm.extraction_helpers import query_keywords


# Patterns that indicate mathematical / formula content in a chunk
_MATH_PATTERNS = [
	re.compile(r"[A-Za-zΑ-Ωα-ω]\s*=\s*[\d\(\[A-Za-zΑ-Ωα-ω]"),  # X = ...
	re.compile(r"[αβγδεζηθμσρλπΣΠΔΩ]"),                             # greek letters
	re.compile(r"\b\d+\s*[\+\-×÷\*\/]\s*\d+"),               # arithmetic
	re.compile(r"\bE\s*\[|\bVar\s*\(|\bCov\s*\("),               # stats notation
	re.compile(r"r_[ipm]\b|\br_f\b|\bbeta\b"),                    # finance notation
	re.compile(r"\bsqrt\b|\bln\b|\blog\b|\bexp\b"),               # math functions
]


def _has_formula_content(hits: List[Dict[str, Any]]) -> bool:
	"""Return True if any top hit contains recognisable mathematical notation."""
	for h in hits[:3]:
		text = str(h.get("text") or "")
		if any(p.search(text) for p in _MATH_PATTERNS):
			return True
	return False


def _has_explicit_formula_for_query(query: str, hits: List[Dict[str, Any]]) -> bool:
	"""Return True when formula evidence is explicit for the requested concept."""
	from llm.grounding import term_present_in_text  # noqa: PLC0415

	q_terms = query_keywords(query)
	q_lower = (query or "").lower()
	formula_request = bool(
		re.search(r"\b(formula|equation|calculate|calculation|derive|derivation|expression)\b", q_lower)
	)

	for h in hits[:3]:
		text = str(h.get("text") or "")
		header = f"{h.get('title') or ''} {h.get('path_text') or ''}"
		full = f"{header} {text}".strip()
		if not any(p.search(full) for p in _MATH_PATTERNS):
			continue
		if formula_request or not q_terms:
			return True
		text_l = full.lower()
		if any(term_present_in_text(term, text_l) for term in q_terms):
			return True
	return False
