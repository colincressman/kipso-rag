"""Unit tests for the llm extraction sub-modules.

Covers:
  llm.extraction_helpers   — query_terms, query_keywords
  llm.formula_detection    — _has_formula_content, _has_explicit_formula_for_query
  llm.metadata_extraction  — clean_publisher_name, extract_title_candidate,
                             is_section_summary_query, extract_metadata_field_answer,
                             extractive_factoid_answer
  llm.evidence_extraction  — extract_section_summary_answer,
                             extract_section_locator_answer,
                             extractive_evidence_facts, path_title_match_override

All tests are pure-Python / no DB / no network.
"""
from __future__ import annotations

import pytest

# ── helpers ──────────────────────────────────────────────────────────────────

from llm.extraction_helpers import query_keywords, query_terms


def test_query_terms_basic():
    assert "backpropagation" in query_terms("What is backpropagation?")


def test_query_terms_empty():
    assert query_terms("") == set()


def test_query_keywords_removes_stopwords():
    kw = query_keywords("what is the backpropagation algorithm?")
    assert "backpropagation" in kw
    assert "algorithm" in kw
    assert "the" not in kw
    assert "is" not in kw


def test_query_keywords_min_length():
    kw = query_keywords("a to do")
    assert all(len(t) > 2 for t in kw)


# ── formula detection ─────────────────────────────────────────────────────────

from llm.formula_detection import _has_explicit_formula_for_query, _has_formula_content


def _hit(text: str, chunk_id: str = "c1", title: str = "", path_text: str = "") -> dict:
    return {"chunk_id": chunk_id, "text": text, "title": title, "path_text": path_text}


def test_has_formula_content_detects_greek():
    hits = [_hit("The loss is L = Σ(yᵢ - ŷᵢ)² where α is the learning rate.")]
    assert _has_formula_content(hits)


def test_has_formula_content_detects_assignment():
    hits = [_hit("We define x = 3 * y + 2.")]
    assert _has_formula_content(hits)


def test_has_formula_content_plain_prose():
    hits = [_hit("Neural networks are powerful function approximators.")]
    assert not _has_formula_content(hits)


def test_has_formula_content_empty():
    assert not _has_formula_content([])


def test_has_explicit_formula_no_formula_in_text():
    hits = [_hit("Backpropagation updates weights using gradient descent.")]
    assert not _has_explicit_formula_for_query("backpropagation formula", hits)


def test_has_explicit_formula_formula_present():
    hits = [_hit("The weight update rule is Δw = -η * ∂L/∂w where η = learning rate.")]
    assert _has_explicit_formula_for_query("weight update formula", hits)


def test_has_explicit_formula_empty_hits():
    assert not _has_explicit_formula_for_query("formula", [])


# ── metadata extraction ───────────────────────────────────────────────────────

from llm.metadata_extraction import (
    clean_publisher_name,
    extract_metadata_field_answer,
    extract_title_candidate,
    extractive_factoid_answer,
    is_section_summary_query,
)


def test_clean_publisher_name_strips_url():
    result = clean_publisher_name("Cambridge University Press www.cambridge.org")
    assert "cambridge.org" not in result.lower()
    assert "Cambridge" in result


def test_clean_publisher_name_strips_address():
    result = clean_publisher_name("MIT Press 55 Hayward Street, Cambridge")
    assert "55" not in result


def test_clean_publisher_name_empty():
    assert clean_publisher_name("") == ""


def test_extract_title_candidate_from_by_pattern():
    blob = "Deep Learning: Foundations and Concepts by Christopher Bishop 2024"
    title = extract_title_candidate(blob)
    assert title != ""
    assert "Bishop" not in title


def test_extract_title_candidate_empty():
    assert extract_title_candidate("") == ""


def test_is_section_summary_query_true():
    assert is_section_summary_query("Summarize chapter 3")
    assert is_section_summary_query("Give me an overview of transformers")
    assert is_section_summary_query("Explain what backpropagation does")


def test_is_section_summary_query_false():
    assert not is_section_summary_query("Who wrote this book?")
    assert not is_section_summary_query("What is the ISBN?")


# ── extract_metadata_field_answer ─────────────────────────────────────────────

def _meta_hit(text: str, cid: str = "c1", title: str = "", path_text: str = "") -> dict:
    return {"chunk_id": cid, "text": text, "title": title, "path_text": path_text}


def test_metadata_isbn_found():
    hits = [_meta_hit("ISBN-13: 978-1-107-01821-0 Published 2016.")]
    result = extract_metadata_field_answer("What is the ISBN?", hits, ["c1"])
    assert "978" in result
    assert "ISBN" in result


def test_metadata_isbn_not_found():
    hits = [_meta_hit("This book covers deep learning fundamentals.")]
    result = extract_metadata_field_answer("What is the ISBN?", hits, ["c1"])
    assert "Low confidence" in result


def test_metadata_publisher_found():
    hits = [_meta_hit("First published in 2016 by Cambridge University Press.")]
    result = extract_metadata_field_answer("Who is the publisher?", hits, ["c1"])
    assert "Cambridge" in result


def test_metadata_year_found():
    hits = [_meta_hit("First published in 2016 by Cambridge University Press.")]
    result = extract_metadata_field_answer("What year was it published?", hits, ["c1"])
    assert "2016" in result


def test_metadata_year_not_found():
    hits = [_meta_hit("This chapter discusses neural networks.")]
    result = extract_metadata_field_answer("When was it published?", hits, ["c1"])
    assert "Low confidence" in result


def test_metadata_author_from_path():
    hits = [_meta_hit(
        "Deep Learning: Foundations and Concepts by Christopher Bishop and Hugh Bishop.",
        cid="c1",
        title="Deep Learning",
        path_text="Chapter 1 > Introduction",
    )]
    result = extract_metadata_field_answer("Who are the authors?", hits, ["c1"])
    assert "Bishop" in result


def test_metadata_empty_hits():
    result = extract_metadata_field_answer("What is the ISBN?", [], [])
    assert result == ""


# ── extractive_factoid_answer ─────────────────────────────────────────────────

def test_factoid_returns_matching_sentence():
    hits = [_meta_hit(
        "The book was first published in 2016 by Cambridge University Press. "
        "It covers deep learning theory and practice.",
        cid="c1",
    )]
    result = extractive_factoid_answer("When was this published?", hits, ["c1"], "medium")
    assert "2016" in result


def test_factoid_empty_hits():
    assert extractive_factoid_answer("What is this?", [], [], "medium") == ""


def test_factoid_no_match_returns_empty_or_low_confidence():
    hits = [_meta_hit("Neural networks learn representations from data.", cid="c1")]
    result = extractive_factoid_answer("isbn", hits, ["c1"], "medium")
    # Either empty string or low-confidence message
    assert result == "" or "Low confidence" in result


# ── evidence extraction ───────────────────────────────────────────────────────

from llm.evidence_extraction import (
    extract_section_locator_answer,
    extract_section_summary_answer,
    extractive_evidence_facts,
    path_title_match_override,
)


def _ev_hit(
    text: str,
    cid: str = "c1",
    title: str = "",
    path_text: str = "",
    page_start: int = 1,
) -> dict:
    return {
        "chunk_id": cid,
        "text": text,
        "title": title,
        "path_text": path_text,
        "page_start": page_start,
    }


def test_section_summary_returns_scored_sentence():
    hits = [_ev_hit(
        "This chapter introduces the concept of backpropagation and gradient descent. "
        "It covers topics including forward pass and weight updates.",
        cid="c1",
        title="Introduction",
        path_text="Chapter 1 > Introduction",
    )]
    result = extract_section_summary_answer("backpropagation", hits, ["c1"], "medium")
    assert result != ""
    assert "[" in result  # citation present


def test_section_summary_empty_hits():
    assert extract_section_summary_answer("anything", [], [], "medium") == ""


def test_section_summary_low_confidence_prefix():
    hits = [_ev_hit(
        "This chapter introduces backpropagation and gradient descent thoroughly.",
        cid="c1",
        title="Introduction",
        path_text="Chapter 1 > Introduction",
    )]
    result = extract_section_summary_answer("backpropagation", hits, ["c1"], "low")
    if result:
        assert result.startswith("Low confidence:")


def test_section_summary_high_confidence_no_prefix():
    hits = [_ev_hit(
        "This chapter introduces backpropagation and gradient descent thoroughly. "
        "The topics covered include weight initialisation and activation functions.",
        cid="c1",
        title="Introduction",
        path_text="Chapter 1 > Introduction",
    )]
    result = extract_section_summary_answer("backpropagation", hits, ["c1"], "high")
    if result:
        assert not result.startswith("Low confidence:")


def test_section_locator_finds_chapter():
    hits = [_ev_hit(
        "This section covers backpropagation.",
        cid="c1",
        title="Backpropagation",
        path_text="Chapter 5 > Backpropagation",
        page_start=82,
    )]
    result = extract_section_locator_answer("Where is backpropagation covered?", hits, ["c1"], "medium")
    assert result != ""
    assert "Backpropagation" in result or "Chapter" in result


def test_section_locator_empty_hits():
    assert extract_section_locator_answer("anything", [], [], "medium") == ""


def test_section_locator_filters_noisy_paths():
    hits = [_ev_hit(
        "See references for more.",
        cid="c1",
        title="References",
        path_text="references > index",
        page_start=300,
    )]
    result = extract_section_locator_answer("backpropagation", hits, ["c1"], "medium")
    # Noisy path (references/index) heavily penalised — should return empty
    assert result == ""


def test_section_locator_page_number_in_output():
    hits = [_ev_hit(
        "Gradient descent optimises the loss function.",
        cid="c1",
        title="Gradient Descent",
        path_text="Chapter 3 > Gradient Descent",
        page_start=45,
    )]
    result = extract_section_locator_answer("gradient descent chapter", hits, ["c1"], "medium")
    if result:
        assert "45" in result  # page number surfaced


def test_extractive_evidence_facts_returns_bulleted():
    hits = [_ev_hit(
        "Backpropagation computes gradients efficiently using the chain rule. "
        "It is the foundation of training deep neural networks.",
        cid="c1",
    )]
    result = extractive_evidence_facts("backpropagation", hits, ["c1"], max_facts=3, confidence_band="medium")
    if result:
        assert result.startswith("- ")


def test_extractive_evidence_facts_empty():
    assert extractive_evidence_facts("anything", [], [], max_facts=3) == ""


def test_extractive_evidence_high_confidence_passage():
    hits = [_ev_hit(
        "The backpropagation algorithm uses the chain rule to propagate errors backward "
        "through the network layers and compute parameter gradients.",
        cid="c1",
    )]
    result = extractive_evidence_facts(
        "backpropagation chain rule", hits, ["c1"], max_facts=3, confidence_band="high"
    )
    if result:
        assert "backpropagation" in result.lower() or "chain" in result.lower()


def test_extractive_evidence_respects_max_facts():
    long_text = ". ".join(
        [f"Backpropagation fact number {i} about gradient descent" for i in range(20)]
    ) + "."
    hits = [_ev_hit(long_text, cid="c1")]
    result = extractive_evidence_facts("backpropagation gradient", hits, ["c1"], max_facts=3, confidence_band="medium")
    if result:
        assert result.count("\n- ") < 3  # at most 3 bullet points (≤ max_facts)


def test_path_title_match_override_true():
    hit = {"title": "Backpropagation", "path_text": "Chapter 5 > Backpropagation > Gradient Descent"}
    assert path_title_match_override("backpropagation gradient", hit, min_matches=2)


def test_path_title_match_override_false():
    hit = {"title": "Introduction", "path_text": "Chapter 1 > Introduction"}
    assert not path_title_match_override("quantum entanglement photon", hit, min_matches=2)


def test_path_title_match_override_empty_query():
    hit = {"title": "Introduction", "path_text": "Chapter 1"}
    assert not path_title_match_override("", hit, min_matches=1)


def test_path_title_match_override_min_matches_respected():
    hit = {"title": "Backpropagation", "path_text": "Chapter 5"}
    # Only 1 term matches, but min_matches=2 → should be False
    assert not path_title_match_override("backpropagation quantum", hit, min_matches=2)


# ── re-export smoke test ──────────────────────────────────────────────────────

def test_extraction_reexport():
    """extraction.py thin shim exports everything expected."""
    import llm.extraction as ext  # noqa: PLC0415

    for name in [
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
    ]:
        assert hasattr(ext, name), f"llm.extraction missing: {name}"
