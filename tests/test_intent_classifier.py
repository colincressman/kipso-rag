"""Unit tests for retrieval.intent_classifier.

All tests mock out the heavy NLI pipeline (DeBERTa) and the remote inference
service so they run fast and without GPU/network dependencies.
"""
from __future__ import annotations

import pytest

import retrieval.intent_classifier as clf_mod
from retrieval.intent_classifier import (
    classify_intent_ml,
    _LABEL_MAP,
    _CANDIDATES,
    _LABELS,
    _CONFIDENCE_THRESHOLD,
    _SECONDARY_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pipeline_for(scores: dict[str, float]):
    """
    Return a callable that mimics the Transformers zero-shot pipeline.

    Parameters
    ----------
    scores : dict mapping *label key* (e.g. "fact_lookup") to float score
    """
    def _pipe(query: str, candidates: list[str], **kwargs) -> dict:
        # Build (candidate_text, score) pairs for every key provided, plus
        # zeroes for any candidate not in `scores`.
        pairs = []
        for key, score in scores.items():
            candidate_text = _LABEL_MAP[key]
            pairs.append((candidate_text, score))
        # Fill in zeros for missing labels so the result covers all candidates
        present_texts = {t for t, _ in pairs}
        for candidate_text in candidates:
            if candidate_text not in present_texts:
                pairs.append((candidate_text, 0.0))
        pairs.sort(key=lambda x: -x[1])
        return {
            "labels": [t for t, _ in pairs],
            "scores": [s for _, s in pairs],
        }
    return _pipe


# ---------------------------------------------------------------------------
# classify_intent_ml: pipeline-unavailable path
# ---------------------------------------------------------------------------

class TestClassifyIntentMlPipelineUnavailable:
    def test_returns_none_triple_when_both_unavailable(self, monkeypatch):
        """When the pipeline can't load and remote is unavailable, all None."""
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: None)
        primary, secondary, conf = classify_intent_ml("What is gradient descent?")
        assert primary is None
        assert secondary is None
        assert conf == 0.0

    def test_empty_query_returns_none_triple(self, monkeypatch):
        """Empty query returns (None, None, 0.0) without touching the pipeline."""
        called = {"value": False}

        def _fake_pipe(*a, **kw):
            called["value"] = True
            raise RuntimeError("should not be called")

        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: _fake_pipe)
        primary, secondary, conf = classify_intent_ml("")
        assert primary is None
        assert secondary is None
        assert conf == 0.0
        assert called["value"] is False


# ---------------------------------------------------------------------------
# classify_intent_ml: confidence threshold
# ---------------------------------------------------------------------------

class TestClassifyIntentMlConfidenceThreshold:
    def test_below_threshold_returns_none_primary(self, monkeypatch):
        """Score below _CONFIDENCE_THRESHOLD → primary=None, conf=that score."""
        low_score = max(0.0, _CONFIDENCE_THRESHOLD - 0.15)
        pipe = _make_pipeline_for({"fact_lookup": low_score, "summary": low_score / 2})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, secondary, conf = classify_intent_ml("What is gradient descent?")
        assert primary is None
        assert conf == pytest.approx(low_score)

    def test_at_threshold_returns_primary(self, monkeypatch):
        """Score exactly at _CONFIDENCE_THRESHOLD → primary is returned."""
        pipe = _make_pipeline_for({"fact_lookup": _CONFIDENCE_THRESHOLD, "summary": 0.1})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, _, conf = classify_intent_ml("What is gradient descent?")
        assert primary == "fact_lookup"
        assert conf == pytest.approx(_CONFIDENCE_THRESHOLD)

    def test_above_threshold_returns_primary(self, monkeypatch):
        """Score above threshold → correct primary label returned."""
        pipe = _make_pipeline_for({"current_data_lookup": 0.90, "fact_lookup": 0.05})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, _, conf = classify_intent_ml("What is the current Fed funds rate?")
        assert primary == "current_data_lookup"
        assert conf == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# classify_intent_ml: secondary intent threshold
# ---------------------------------------------------------------------------

class TestClassifyIntentMlSecondaryIntent:
    def test_secondary_returned_when_gap_within_threshold(self, monkeypatch):
        """Gap between top and second < _SECONDARY_THRESHOLD → secondary set."""
        top = _CONFIDENCE_THRESHOLD + 0.10        # e.g. 0.75
        second = top - (_SECONDARY_THRESHOLD / 2) # e.g. 0.675 — within gap
        pipe = _make_pipeline_for({"comparison": top, "formula_lookup": second, "fact_lookup": 0.05})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, secondary, _ = classify_intent_ml("Compare the formulas for X and Y.")
        assert primary == "comparison"
        assert secondary == "formula_lookup"

    def test_secondary_none_when_gap_exceeds_threshold(self, monkeypatch):
        """Gap between top and second > _SECONDARY_THRESHOLD → secondary=None."""
        top = _CONFIDENCE_THRESHOLD + 0.25        # e.g. 0.90
        second = top - (_SECONDARY_THRESHOLD + 0.10)  # far below
        pipe = _make_pipeline_for({"fact_lookup": top, "summary": second})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, secondary, _ = classify_intent_ml("What is backpropagation?")
        assert primary == "fact_lookup"
        assert secondary is None

    def test_secondary_none_when_only_one_label(self, monkeypatch):
        """Only one label provided → secondary=None (no second label to compare)."""
        # Build a pipeline result with only one label above zero
        def _single_label_pipe(q, candidates, **kw):
            return {"labels": [_LABEL_MAP["fact_lookup"]], "scores": [0.95]}
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: _single_label_pipe)
        primary, secondary, conf = classify_intent_ml("What is momentum in neural networks?")
        assert primary == "fact_lookup"
        assert secondary is None
        assert conf == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# classify_intent_ml: remote inference service path
# ---------------------------------------------------------------------------

class TestClassifyIntentMlRemoteService:
    def test_remote_result_used_when_available(self, monkeypatch):
        """When _remote_classify returns a result it is passed through directly."""
        fake_remote = ("current_data_lookup", "fact_lookup", 0.88, {})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: fake_remote)
        # Pipeline must NOT be called
        called = {"value": False}

        def _fail_pipeline():
            called["value"] = True
            return None

        monkeypatch.setattr(clf_mod, "_get_pipeline", _fail_pipeline)
        primary, secondary, conf = classify_intent_ml("What is the current inflation rate?")
        assert primary == "current_data_lookup"
        assert secondary == "fact_lookup"
        assert conf == pytest.approx(0.88)
        assert called["value"] is False

    def test_local_pipeline_used_when_remote_returns_none(self, monkeypatch):
        """When _remote_classify returns None, local pipeline is used."""
        pipe = _make_pipeline_for({"fact_lookup": 0.82, "summary": 0.10})
        monkeypatch.setattr(clf_mod, "_remote_classify", lambda q: None)
        monkeypatch.setattr(clf_mod, "_get_pipeline", lambda: pipe)
        primary, _, conf = classify_intent_ml("What is gradient descent?")
        assert primary == "fact_lookup"
        assert conf == pytest.approx(0.82)
