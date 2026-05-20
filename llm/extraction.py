"""Thin re-exporter — aggregates the extraction sub-modules.

The actual logic lives in:
  llm.extraction_helpers    — query_terms, query_keywords
  llm.formula_detection     — _MATH_PATTERNS, _has_formula_content,
                              _has_explicit_formula_for_query
  llm.metadata_extraction   — clean_publisher_name, extract_title_candidate,
                              is_section_summary_query,
                              extract_metadata_field_answer,
                              extractive_factoid_answer
  llm.evidence_extraction   — extract_section_summary_answer,
                              extract_section_locator_answer,
                              extractive_evidence_facts,
                              path_title_match_override
"""

from __future__ import annotations

from llm.extraction_helpers import query_keywords, query_terms
from llm.formula_detection import (
	_MATH_PATTERNS,
	_has_explicit_formula_for_query,
	_has_formula_content,
)
from llm.metadata_extraction import (
	clean_publisher_name,
	extract_metadata_field_answer,
	extract_title_candidate,
	extractive_factoid_answer,
	is_section_summary_query,
)
from llm.evidence_extraction import (
	extract_section_locator_answer,
	extract_section_summary_answer,
	extractive_evidence_facts,
	path_title_match_override,
)

__all__ = [
	"query_terms",
	"query_keywords",
	"_MATH_PATTERNS",
	"_has_formula_content",
	"_has_explicit_formula_for_query",
	"clean_publisher_name",
	"extract_title_candidate",
	"is_section_summary_query",
	"extract_metadata_field_answer",
	"extractive_factoid_answer",
	"extract_section_summary_answer",
	"extract_section_locator_answer",
	"extractive_evidence_facts",
	"path_title_match_override",
]
