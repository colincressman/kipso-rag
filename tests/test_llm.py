from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pytest
import yaml

import llm.answer as answer_mod
from llm.answer import answer_query_with_retrieval
from llm.prompt_templates import build_user_prompt, format_context_blocks
from scripts.ops.query_cli import _load_cli_llm_defaults


def _sample_hits():
    return [
        {
            "chunk_id": "book:sec_001:chunk_0001",
            "path_text": "Chapter 1 > Intro",
            "page_start": 1,
            "page_end": 2,
            "text": "Capital markets allocate risk and return through tradable securities.",
        },
        {
            "chunk_id": "book:sec_002:chunk_0001",
            "path_text": "Chapter 2 > Portfolio",
            "page_start": 3,
            "page_end": 4,
            "text": "Diversification reduces idiosyncratic risk in a portfolio.",
        },
    ]


def test_format_context_blocks_includes_chunk_markers():
    rendered = format_context_blocks(_sample_hits(), max_chunks=2)
    assert "[CHUNK book:sec_001:chunk_0001]" in rendered
    assert "[CHUNK book:sec_002:chunk_0001]" in rendered


def test_build_user_prompt_contains_query_and_context():
    prompt = build_user_prompt("What is diversification?", _sample_hits())
    assert "What is diversification?" in prompt
    assert "Context:" in prompt
    assert "[CHUNK book:sec_001:chunk_0001]" in prompt


def test_answer_falls_back_when_llm_unreachable():
    retrieval_result = {"hits": _sample_hits()}
    result = answer_query_with_retrieval(
        "Explain diversification.",
        retrieval_result,
        llm_model="qwen3:latest",
        llm_base_url="http://localhost:1",
        llm_timeout_seconds=0.2,
    )
    assert result["mode"] in {"fallback", "low_confidence"}
    if result["mode"] == "fallback":
        assert "best matching context excerpt" in result["answer"]
        assert "book:sec_001:chunk_0001" in result["citations"]


def test_answer_with_empty_hits_returns_missing_context_message():
    retrieval_result = {"hits": []}
    result = answer_query_with_retrieval(
        "Explain diversification.",
        retrieval_result,
        llm_model="qwen3:latest",
        llm_base_url="http://localhost:1",
        llm_timeout_seconds=0.2,
    )
    assert result["mode"] == "fallback"
    assert "could not find relevant context" in result["answer"].lower()


def test_cli_defaults_loaded_from_llm_config():
    defaults = _load_cli_llm_defaults()
    assert defaults["llm_model"] == "rag-llm"
    assert str(defaults["llm_base_url"]).startswith("http")


def test_llm_config_path_is_applied_for_model_override(tmp_path: Path):
    config = tmp_path / "llm.yaml"
    config.write_text(
        """
llm:
  model: tiny-test-model
  base_url: http://localhost:1
  timeout_seconds: 0.1
  temperature: 0.0
prompt:
  force_grounded_fallback_when_uncited: true
""".strip(),
        encoding="utf-8",
    )

    retrieval_result = {"hits": _sample_hits()}
    result = answer_query_with_retrieval(
        "Explain diversification.",
        retrieval_result,
        config_path=str(config),
    )
    assert result["llm_model"] == "tiny-test-model"
    assert result["mode"] in {"fallback", "grounded_fallback", "high_confidence", "medium_confidence", "low_confidence"}


def test_uncited_answer_kept_in_llm_mode_when_confident(monkeypatch: pytest.MonkeyPatch):
    def _fake_chat(**_: object) -> str:
        return "CAPM links expected return to beta and market risk."

    monkeypatch.setattr(answer_mod, "_ollama_chat", _fake_chat)

    retrieval_result = {
        "hits": [
            {
                "chunk_id": "c1",
                "title": "CAPM",
                "path_text": "Finance > CAPM",
                "text": "Capital Asset Pricing Model relates expected return to beta.",
                "score": 0.74,
                "metadata": {"score_gap_to_second": 0.04},
            },
            {"chunk_id": "c2", "text": "Other", "score": 0.70, "metadata": {}},
        ]
    }

    result = answer_query_with_retrieval("What is CAPM?", retrieval_result)
    assert result["mode"] == "high_confidence"
    assert result["confidence"]["top_score"] >= 0.70
    assert result["routing"]["confidence_band"] == "high"


def test_uncited_answer_becomes_grounded_fallback_when_low_confidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def _fake_chat(**_: object) -> str:
        return "It is about arbitrage."

    monkeypatch.setattr(answer_mod, "_ollama_chat", _fake_chat)

    config = tmp_path / "llm_decision.yaml"
    config.write_text(
        json.dumps(
            {
                "llm": {
                    "model": "qwen2.5:3b-instruct",
                    "base_url": "http://localhost:11434",
                    "timeout_seconds": 5,
                    "temperature": 0.0,
                },
                "prompt": {
                    "force_grounded_fallback_when_uncited": True,
                    "max_chunks": 6,
                    "max_chars_per_chunk": 1600,
                    "require_inline_citations": True,
                    "min_citations": 2,
                    "max_citations": 6,
                    "include_retrieval_score": True,
                },
                "decision": {
                    "medium_confidence_score": 0.55,
                    "high_confidence_score": 0.70,
                    "borderline_confidence_score": 0.62,
                    "max_ambiguous_gap": 0.03,
                    "path_override_min_term_matches": 2,
                    "allow_uncited_if_confident": True,
                    "allow_low_confidence_answer": False,
                },
            }
        ),
        encoding="utf-8",
    )

    retrieval_result = {
        "hits": [
            {
                "chunk_id": "c1",
                "title": "General",
                "path_text": "Finance > Concepts",
                "text": "Arbitrage is discussed.",
                "score": 0.56,
                "metadata": {"score_gap_to_second": 0.01},
            },
            {"chunk_id": "c2", "text": "Other", "score": 0.55, "metadata": {}},
        ]
    }

    result = answer_query_with_retrieval(
        "arbitrage",
        retrieval_result,
        config_path=str(config),
    )
    assert result["mode"] == "grounded_fallback"
    assert result["confidence"]["rule"] in {"medium_ambiguous", "top_score_below_medium"}


def test_path_title_override_does_not_bypass_low_confidence_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    def _fake_chat(**_: object) -> str:
        return "Arbitrage exploits temporary mispricing."

    monkeypatch.setattr(answer_mod, "_ollama_chat", _fake_chat)

    config = tmp_path / "llm_override.yaml"
    config.write_text(
        json.dumps(
            {
                "prompt": {"force_grounded_fallback_when_uncited": True},
                "decision": {
                    "medium_confidence_score": 0.70,
                    "high_confidence_score": 0.80,
                    "borderline_confidence_score": 0.75,
                    "max_ambiguous_gap": 0.05,
                    "path_override_min_term_matches": 1,
                    "allow_uncited_if_confident": True,
                    "allow_low_confidence_answer": True,
                },
            }
        ),
        encoding="utf-8",
    )

    retrieval_result = {
        "hits": [
            {
                "chunk_id": "c1",
                "title": "Arbitrage Basics",
                "path_text": "Markets > Arbitrage",
                "text": "Arbitrage exploits temporary price discrepancies.",
                "score": 0.58,
                "metadata": {"score_gap_to_second": 0.01},
            }
        ]
    }

    result = answer_query_with_retrieval(
        "arbitrage",
        retrieval_result,
        config_path=str(config),
    )
    assert result["mode"] == "low_confidence"
    assert result["confidence"]["rule"] == "top_score_below_medium"


def test_answer_citation_cleanup_removes_chunk_noise_and_malformed_ids(monkeypatch: pytest.MonkeyPatch):
    def _fake_chat(**_: object) -> str:
        return (
            "Answer text [chunk_id] [chunk_abcdef] and "
            "(chunk 4a58a34d7699bfc03317af15fcd9f5cb4b2e3204f4a5bb92253c98489cc6034d-c0a58a34d)."
        )

    monkeypatch.setattr(answer_mod, "_ollama_chat", _fake_chat)
    retrieval_result = {
        "hits": [
            {"chunk_id": "abc-c000035", "text": "x", "score": 0.9, "metadata": {}},
            {"chunk_id": "abc-c000034", "text": "y", "score": 0.86, "metadata": {}},
        ]
    }
    result = answer_query_with_retrieval("What is arbitrage?", retrieval_result)
    assert "[chunk_id]" not in result["answer"]


def test_formula_query_uses_generic_equation_evidence() -> None:
    retrieval_result = {
        "hits": [
            {
                "chunk_id": "m1",
                "title": "Risk Model",
                "path_text": "Quant > Risk",
                "text": "Expected loss = probability of default * exposure at default.",
                "score": 0.82,
                "metadata": {"score_gap_to_second": 0.12},
            },
            {
                "chunk_id": "m2",
                "title": "Context",
                "path_text": "Quant > Notes",
                "text": "General context.",
                "score": 0.70,
                "metadata": {},
            },
        ]
    }
    result = answer_query_with_retrieval("risk model formula", retrieval_result, intent="formula_lookup")
    assert result["mode"] != "formula_not_found"


def test_section_summary_query_reaches_llm() -> None:
    retrieval_result = {
        "hits": [
            {
                "chunk_id": "doc-c000012",
                "title": "Introduction",
                "path_text": "Part I > Chapter 1 > Introduction",
                "text": "This introduction explains the goals of the book and outlines the main themes for readers.",
                "score": 0.62,
                "metadata": {"score_gap_to_second": 0.08},
            },
            {
                "chunk_id": "doc-c000013",
                "title": "Introduction",
                "path_text": "Part I > Chapter 1 > Introduction",
                "text": "It also summarizes how the later sections develop practical strategy and risk concepts.",
                "score": 0.58,
                "metadata": {},
            },
        ]
    }
    config_path = _write_llm_config_override({"decision": {"always_use_llm": False}})
    result = answer_query_with_retrieval(
        "Summarize the introduction section.",
        retrieval_result,
        intent="section_lookup",
        config_path=config_path,
    )
    # Extractive shortcuts removed — section_lookup goes to the LLM.
    # Verify the pipeline reached a confidence mode (not no_coverage/formula_not_found).
    assert result["mode"] in {"high_confidence", "medium_confidence", "low_confidence", "fallback", "grounded_fallback"}



def _write_llm_config_override(overrides: dict) -> str:
    base = {
        "llm": {
            "model": "qwen2.5:3b-instruct",
            "base_url": "http://localhost:11434",
            "timeout_seconds": 180,
            "temperature": 0.05,
        },
        "prompt": {
            "max_chunks": 6,
            "max_chars_per_chunk": 1600,
            "include_neighbor_context": True,
            "max_neighbors_per_chunk": 2,
            "max_chars_per_neighbor": 280,
            "require_inline_citations": True,
            "enforce_sentence_citations": True,
            "min_citations": 2,
            "max_citations": 3,
            "include_retrieval_score": True,
            "force_grounded_fallback_when_uncited": False,
            "citation_score_window": 0.06,
        },
        "decision": {
            "medium_confidence_score": 0.55,
            "high_confidence_score": 0.70,
            "borderline_confidence_score": 0.62,
            "max_ambiguous_gap": 0.03,
            "path_override_min_term_matches": 1,
            "always_use_llm": True,
            "allow_uncited_if_confident": True,
            "allow_low_confidence_answer": True,
            "enforce_entity_grounding": True,
            "entity_grounding_bands": ["high", "medium"],
        },
    }
    for section, values in overrides.items():
        base.setdefault(section, {}).update(values)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(base, handle, sort_keys=False)
        return handle.name


def test_dynamic_citation_selection_not_always_five(monkeypatch: pytest.MonkeyPatch):
    def _fake_chat(**_: object) -> str:
        return "Short grounded answer."

    monkeypatch.setattr(answer_mod, "_ollama_chat", _fake_chat)
    retrieval_result = {
        "hits": [
            {"chunk_id": "doc-c000001", "text": "a", "score": 0.80, "metadata": {}},
            {"chunk_id": "doc-c000002", "text": "b", "score": 0.77, "metadata": {}},
            {"chunk_id": "doc-c000003", "text": "c", "score": 0.70, "metadata": {}},
            {"chunk_id": "doc-c000004", "text": "d", "score": 0.60, "metadata": {}},
            {"chunk_id": "doc-c000005", "text": "e", "score": 0.50, "metadata": {}},
        ]
    }
    result = answer_query_with_retrieval("test question", retrieval_result)
    assert 2 <= len(result["citations"]) <= 3
    assert len(result["citations"]) == 2


def test_external_fact_query_uses_internet_only_context_for_answer() -> None:
    retrieval_result = {
        "hits": [
            {
                "chunk_id": "local-c000111",
                "title": "Unrelated",
                "path_text": "ML > Causality",
                "text": "Counterfactual outcomes are cross-world quantities.",
                "score": 0.70,
                "source_type": "pdf_book",
                "metadata": {"score_gap_to_second": 0.05},
            },
            {
                "chunk_id": "internet-c000003",
                "title": "2022 FIFA World Cup - Britannica",
                "path_text": "https://www.britannica.com/sports/2022-FIFA-World-Cup",
                "text": "Argentina won the 2022 FIFA World Cup, defeating France in the final.",
                "score": 0.59,
                "source_type": "internet",
                "metadata": {},
            },
        ],
        "internet_fallback": {"triggered": True, "priority_applied": True},
    }

    result = answer_query_with_retrieval(
        "Who won the FIFA World Cup in 2022?",
        retrieval_result,
        intent="fact_lookup",
    )

    assert result["routing"]["internet_only_for_answer"] is True
    assert "internet-c000003" in result["citations"]
    assert "local-c000111" not in result["citations"]


def test_external_fact_query_abstains_when_no_qualified_internet_hits() -> None:
    retrieval_result = {
        "hits": [
            {
                "chunk_id": "local-c000111",
                "title": "Unrelated",
                "path_text": "ML > Causality",
                "text": "Counterfactual outcomes are cross-world quantities.",
                "score": 0.70,
                "source_type": "pdf_book",
                "metadata": {"score_gap_to_second": 0.05},
            }
        ],
        "internet_fallback": {"triggered": True, "priority_applied": False},
    }

    result = answer_query_with_retrieval(
        "Who is the current CEO of Microsoft?",
        retrieval_result,
        intent="fact_lookup",
    )

    assert result["mode"] == "internet_no_evidence"
    assert result["citations"] == []
    assert "no answer fabricated" in result["answer"].lower()
    assert result["routing"]["answer_policy"] == "internet_no_evidence"


# ── Per-intent style customization ────────────────────────────────────────────

def test_formula_lookup_style_mentions_formula_instruction():
    prompt = build_user_prompt(
        "What is the CAPM formula?",
        _sample_hits(),
        intent="formula_lookup",
        confidence_band="medium",
    )
    style_section = prompt[prompt.find("Style:"):]
    assert "formula" in style_section.lower() or "equation" in style_section.lower()
    assert "notation" in style_section.lower()


def test_list_lookup_style_mentions_bullet():
    prompt = build_user_prompt(
        "What are the assumptions of CAPM?",
        _sample_hits(),
        intent="list_lookup",
        confidence_band="medium",
    )
    style_section = prompt[prompt.find("Style:"):]
    assert "bullet" in style_section.lower() or "list" in style_section.lower()


def test_comparison_style_mentions_compare():
    prompt = build_user_prompt(
        "Compare CAPM and APT.",
        _sample_hits(),
        intent="comparison",
        confidence_band="medium",
    )
    style_section = prompt[prompt.find("Style:"):]
    assert "compar" in style_section.lower() or "contrast" in style_section.lower()


def test_intent_style_does_not_override_low_confidence():
    """Low-confidence trust signal must be preserved even when intent is known."""
    prompt = build_user_prompt(
        "What is the CAPM formula?",
        _sample_hits(),
        intent="formula_lookup",
        confidence_band="low",
    )
    style_section = prompt[prompt.find("Style:"):]
    assert "low confidence" in style_section.lower()
    # Formula-specific language should NOT appear when confidence is low.
    assert "lead with the formula" not in style_section.lower()


def test_intent_style_overrides_high_confidence_band():
    """Intent style should take priority over the generic high-confidence style."""
    prompt_with_intent = build_user_prompt(
        "What is the CAPM formula?",
        _sample_hits(),
        intent="formula_lookup",
        confidence_band="high",
    )
    prompt_no_intent = build_user_prompt(
        "What is the CAPM formula?",
        _sample_hits(),
        intent=None,
        confidence_band="high",
    )
    style_with = prompt_with_intent[prompt_with_intent.find("Style:"):]
    style_without = prompt_no_intent[prompt_no_intent.find("Style:"):]
    assert "lead with the formula" in style_with.lower() or "notation" in style_with.lower()
    assert "2" in style_without  # generic "2–4 sentences" wording


def test_unknown_intent_falls_through_to_confidence_band():
    """An unrecognised intent should not break anything — falls through to confidence band."""
    prompt = build_user_prompt(
        "General question.",
        _sample_hits(),
        intent="exploratory",
        confidence_band="medium",
    )
    assert "Style:" in prompt
    assert "colleague" in prompt.lower()


def test_no_intent_uses_confidence_band_style():
    prompt_medium = build_user_prompt(
        "Explain diversification.",
        _sample_hits(),
        intent=None,
        confidence_band="medium",
    )
    assert "colleague" in prompt_medium.lower()


# ── Grounding evaluation unit tests ───────────────────────────────────────────

from scripts.eval.evaluate_grounding import (
    _confidence_band,
    _norm_tokens,
    _split_sentences,
    _unsupported_counts,
    evaluate_grounding,
)


class TestConfidenceBand:
    def test_high_at_threshold(self):
        assert _confidence_band(0.70, 0.55, 0.70) == "high"

    def test_high_above_threshold(self):
        assert _confidence_band(0.85, 0.55, 0.70) == "high"

    def test_medium_between_thresholds(self):
        assert _confidence_band(0.62, 0.55, 0.70) == "medium"

    def test_medium_at_lower_threshold(self):
        assert _confidence_band(0.55, 0.55, 0.70) == "medium"

    def test_low_below_medium(self):
        assert _confidence_band(0.40, 0.55, 0.70) == "low"

    def test_zero_score_is_low(self):
        assert _confidence_band(0.0, 0.55, 0.70) == "low"


class TestNormTokens:
    def test_removes_stopwords(self):
        tokens = _norm_tokens("the capital asset pricing model")
        assert "the" not in tokens
        assert "capital" in tokens
        assert "asset" in tokens

    def test_removes_short_words(self):
        tokens = _norm_tokens("a big dog")
        assert "big" in tokens
        assert "dog" in tokens

    def test_lowercases(self):
        tokens = _norm_tokens("CAPM Model")
        assert "capm" in tokens
        assert "model" in tokens

    def test_empty_string(self):
        assert _norm_tokens("") == []


class TestSplitSentences:
    def test_splits_on_period(self):
        sents = _split_sentences("First sentence. Second sentence.")
        assert len(sents) == 2

    def test_splits_on_newline(self):
        sents = _split_sentences("Line one.\nLine two.")
        assert len(sents) == 2

    def test_empty_string(self):
        assert _split_sentences("") == []

    def test_single_sentence_no_trailing_period(self):
        sents = _split_sentences("Just one sentence")
        assert sents == ["Just one sentence"]


class TestUnsupportedCounts:
    def test_clean_answer_passes(self):
        answer = "Diversification reduces idiosyncratic risk in a portfolio."
        cited = "Diversification reduces idiosyncratic risk in a portfolio of securities."
        counts = _unsupported_counts(answer, cited, sentence_overlap_threshold=0.35)
        assert counts["weak_sentence_count"] == 0
        assert counts["malformed_citation_count"] == 0

    def test_unsupported_year_flagged(self):
        answer = "This happened in 2019."
        cited = "No year mentioned here at all."
        counts = _unsupported_counts(answer, cited, sentence_overlap_threshold=0.35)
        assert counts["unsupported_year_count"] == 1

    def test_supported_year_not_flagged(self):
        answer = "This happened in 2019."
        cited = "Events in 2019 were significant."
        counts = _unsupported_counts(answer, cited, sentence_overlap_threshold=0.35)
        assert counts["unsupported_year_count"] == 0

    def test_malformed_citation_detected(self):
        answer = "See [c0000000000000000000000] for details."
        cited = "Some source text."
        counts = _unsupported_counts(answer, cited, sentence_overlap_threshold=0.35)
        assert counts["malformed_citation_count"] >= 1

    def test_short_sentences_skipped_for_weak_check(self):
        # Sentences with < 6 tokens are not checked for overlap (too short to be meaningful)
        answer = "Yes it does."
        cited = "Completely unrelated source text about other topics."
        counts = _unsupported_counts(answer, cited, sentence_overlap_threshold=0.35)
        assert counts["weak_sentence_count"] == 0


class TestEvaluateGroundingIntegration:
    """End-to-end test using a temp SQLite DB and in-memory QA fixture."""

    def _build_db(self, pg_dsn: str) -> str:
        import psycopg as _psycopg
        with _psycopg.connect(pg_dsn, autocommit=True) as conn:
            conn.execute(
                "INSERT INTO documents (doc_id, filename, source_path, num_pages, metadata_json) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                ("test_doc", "test.pdf", "/test.pdf", 1, "{}"),
            )
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, text) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                ("c000001", "test_doc", "Capital markets allocate risk and return through tradable securities."),
            )
            conn.execute(
                "INSERT INTO chunks (chunk_id, doc_id, text) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                ("c000002", "test_doc", "Diversification reduces idiosyncratic risk in a portfolio."),
            )
        return pg_dsn

    def _build_qa(self, tmp_path: Path) -> Path:
        qa = tmp_path / "qa.json"
        qa.write_text(
            json.dumps({
                "items": [
                    {
                        "index": 0,
                        "question": "What is diversification?",
                        "answer": "Diversification reduces idiosyncratic risk in a portfolio of securities.",
                        "citations": ["c000002"],
                        "mode": "high_confidence",
                        "top_hits": [{"score": 0.82}],
                    },
                    {
                        "index": 1,
                        "question": "What is arbitrage?",
                        "answer": "Arbitrage exploits price discrepancies in the 2099 market.",
                        "citations": ["c000001"],
                        "mode": "medium_confidence",
                        "top_hits": [{"score": 0.61}],
                    },
                    {
                        "index": 2,
                        "question": "What is the formula?",
                        "answer": "No coverage found.",
                        "citations": [],
                        "mode": "no_coverage",
                        "top_hits": [{"score": 0.20}],
                    },
                ]
            }),
            encoding="utf-8",
        )
        return qa

    def test_high_confidence_clean_answer_passes(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        item = result["items"][0]
        assert item["band"] == "high"
        assert item["pass"] is True

    def test_unsupported_year_in_medium_band_fails(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        item = result["items"][1]
        assert item["band"] == "medium"
        assert item["unsupported_year_count"] >= 1
        assert item["pass"] is False

    def test_safe_refusal_always_passes(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        item = result["items"][2]
        assert item["band"] == "refusal"
        assert item["safe_refusal"] is True
        assert item["pass"] is True

    def test_summary_by_band_present(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        assert "summary_by_band" in result
        assert "high" in result["summary_by_band"]
        assert "medium" in result["summary_by_band"]
        assert "refusal" in result["summary_by_band"]

    def test_pass_rate_correct_for_high_band(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        high_summary = result["summary_by_band"]["high"]
        assert high_summary["pass_count"] == 1
        assert high_summary["pass_rate"] == 1.0

    def test_thresholds_recorded_in_result(self, pg_dsn: str, tmp_path: Path):
        db = self._build_db(pg_dsn)
        qa = self._build_qa(tmp_path)
        result = evaluate_grounding(
            qa_path=qa, db_dsn=db,
            medium_threshold=0.55, high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
        thresholds = result["thresholds"]
        assert thresholds["medium_confidence_score"] == 0.55
        assert thresholds["high_confidence_score"] == 0.70
        assert thresholds["sentence_overlap_threshold"] == 0.35


# ── prepare_rag_answer / finalize_rag_answer unit tests ───────────────────────

from llm.answer import prepare_rag_answer, finalize_rag_answer, _RagGenCtx


def _make_simple_hits(score: float, text: str, count: int = 1):
    return [
        {
            "chunk_id": f"doc-c{i:06d}",
            "title": "Chapter 1",
            "path_text": "Book > Chapter 1",
            "text": text,
            "score": score,
            "metadata": {"score_gap_to_second": 0.15},
        }
        for i in range(count)
    ]


class TestPrepareRagAnswer:
    def test_returns_raggenctx_for_normal_path(self):
        """Normal high-score hits return a _RagGenCtx for LLM generation."""
        hits = _make_simple_hits(0.80, "Diversification reduces idiosyncratic risk in a portfolio.")
        result = prepare_rag_answer("What is diversification?", {"hits": hits})
        assert isinstance(result, _RagGenCtx)

    def test_no_coverage_low_score(self):
        """Score below _MIN_COVERAGE_SCORE with always_use_llm=False → no_coverage/low_score."""
        hits = _make_simple_hits(0.25, "Hedging uses derivatives to reduce risk exposure.")
        config_path = _write_llm_config_override({"decision": {"always_use_llm": False}})
        result = prepare_rag_answer("What is portfolio theory?", {"hits": hits}, config_path=config_path)
        assert isinstance(result, dict)
        assert result["mode"] == "no_coverage"
        assert result["routing"]["no_coverage_reason"] == "low_score"

    def test_no_coverage_zero_lexical(self):
        """Query keywords absent from hits → force_zero_coverage_fallback fires."""
        # Query: 4 non-stopword domain terms; hit text is about an unrelated subject
        hits = _make_simple_hits(
            0.60,
            "Dogs are loyal companions providing comfort and affection to their owners in many ways.",
        )
        result = prepare_rag_answer(
            "gradient backpropagation activation sigmoid",
            {"hits": hits},
        )
        assert isinstance(result, dict)
        assert result["mode"] == "no_coverage"
        assert result["routing"]["no_coverage_reason"] == "zero_lexical_coverage"

    def test_internet_no_evidence_when_external_query_has_no_web_hits(self):
        """External fact query + internet triggered + no internet hits → internet_no_evidence."""
        local_hit = {
            "chunk_id": "doc-c000001",
            "title": "Local Chapter",
            "path_text": "Book > Intro",
            "text": "Capital markets allocate risk and return through tradable securities.",
            "score": 0.70,
            "source_type": "pdf_book",
            "metadata": {},
        }
        retrieval_result = {
            "hits": [local_hit],
            "internet_fallback": {"triggered": True},
        }
        result = prepare_rag_answer(
            "What did Microsoft announce at Build?",
            retrieval_result,
        )
        assert isinstance(result, dict)
        assert result["mode"] == "internet_no_evidence"

    def test_token_budget_warning_logged(self, monkeypatch, caplog):
        """Lowering the token-warn threshold triggers a budget warning."""
        import logging
        import llm.answer_context as answer_ctx_mod
        monkeypatch.setattr(answer_mod, "_TOKEN_WARN_THRESHOLD", 10)
        monkeypatch.setattr(answer_ctx_mod, "_TOKEN_WARN_THRESHOLD", 10)
        hits = _make_simple_hits(0.80, "Diversification reduces idiosyncratic risk in a portfolio.")
        with caplog.at_level(logging.WARNING, logger="llm.answer_context"):
            result = prepare_rag_answer("What is diversification?", {"hits": hits})
        assert any("Token budget warning" in r.message for r in caplog.records)
        if isinstance(result, _RagGenCtx):
            assert result.routing["estimated_prompt_tokens"] > 10


class TestFinalizeRagAnswer:
    def test_empty_llm_response_produces_fallback_mode(self):
        """finalize_rag_answer with empty text returns mode='fallback'."""
        hits = _make_simple_hits(0.80, "Diversification reduces idiosyncratic risk in a portfolio.")
        ctx = prepare_rag_answer("What is diversification?", {"hits": hits})
        assert isinstance(ctx, _RagGenCtx), "expected _RagGenCtx for high-score hits"
        result = finalize_rag_answer(ctx, "")
        assert result["mode"] == "fallback"
        assert result["routing"]["fallback_reason"] == "empty_llm_response"

    def test_non_empty_llm_response_is_not_fallback(self, monkeypatch):
        """A real LLM answer does not produce a fallback mode."""
        monkeypatch.setattr(answer_mod, "_ollama_chat", lambda **_: "Diversification reduces risk.")
        hits = _make_simple_hits(0.80, "Diversification reduces idiosyncratic risk in a portfolio.")
        result = answer_query_with_retrieval("What is diversification?", {"hits": hits})
        assert result["mode"] != "fallback"
