from __future__ import annotations

from unittest.mock import patch

from pipeline.plan import plan_query


def _intent_meta(**overrides):
    base = {
        "ml_confidence": 0.88,
        "secondary_intent": None,
        "nli_scores": {},
    }
    base.update(overrides)
    return base


def test_active_collection_prefers_rag_even_when_scope_is_weak():
    with (
        patch("retrieval.router.classify_intent", return_value=("fact_lookup", _intent_meta())),
        patch("retrieval.router._needs_web", return_value=(False, None)),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(False, 0.0, [])),
    ):
        plan = plan_query(
            "What does it say about cybersecurity?",
            collection_id="collection-1",
        )

    assert [step.tool for step in plan.steps] == ["rag"]
    assert plan.meta.get("selected_source") == "rag"


def test_doc_cue_overrides_conversational_tone_and_uses_rag():
    with (
        patch("retrieval.router.classify_intent", return_value=("conversational", _intent_meta())),
        patch("retrieval.router._needs_web", return_value=(False, None)),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(False, 0.0, [])),
    ):
        plan = plan_query("Can you check the spec for fiber replacement?")

    assert [step.tool for step in plan.steps] == ["rag"]
    assert "doc_verbs" in (plan.meta.get("source_signals") or {}).get("doc_cues", [])


def test_social_acknowledgement_stays_in_chat_even_with_collection_active():
    with (
        patch("retrieval.router.classify_intent", return_value=("conversational", _intent_meta())),
        patch("retrieval.router._needs_web", return_value=(False, None)),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(True, 1.0, ["thanks"])),
    ):
        plan = plan_query("Thanks", collection_id="collection-1")

    assert [step.tool for step in plan.steps] == ["history"]
    assert plan.meta.get("selected_source") == "chat"


def test_recent_rag_session_keeps_ambiguous_followup_in_rag():
    with (
        patch("retrieval.router.classify_intent", return_value=("fact_lookup", _intent_meta(ml_confidence=0.41))),
        patch("retrieval.router._needs_web", return_value=(False, None)),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(False, 0.0, [])),
    ):
        plan = plan_query("What about reporting?", prior_sources=["chat", "rag", "rag"])

    assert [step.tool for step in plan.steps] == ["rag"]
    assert (plan.meta.get("source_signals") or {}).get("rag_momentum") == 2


def test_fresh_current_query_prefers_web_without_doc_context():
    with (
        patch("retrieval.router.classify_intent", return_value=("current_data_lookup", _intent_meta())),
        patch("retrieval.router._needs_web", return_value=(True, "current_data_lookup")),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(False, 0.0, [])),
    ):
        plan = plan_query("What is the latest VTScada version?")

    assert [step.tool for step in plan.steps] == ["web"]
    assert plan.meta.get("selected_source") == "web"


def test_current_public_fact_stays_web_even_with_rag_momentum():
    with (
        patch("retrieval.router.classify_intent", return_value=("current_data_lookup", _intent_meta())),
        patch("retrieval.router._needs_web", return_value=(True, "current_data_lookup")),
        patch("retrieval.corpus_scope.is_in_scope", return_value=(True, 0.5, ["pope"])),
    ):
        plan = plan_query(
            "Who is the current pope?",
            collection_id="collection-1",
            prior_sources=["rag", "rag", "rag"],
        )

    assert [step.tool for step in plan.steps] == ["web"]
    assert plan.meta.get("web_priority_override") is True
