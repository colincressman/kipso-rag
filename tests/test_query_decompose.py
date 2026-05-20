"""Unit tests for retrieval/query_decompose.py.

These tests do not hit the database; they exercise the pure-Python logic
(sub-topic extraction, sub-query building, RRF merge, and the
retrieve_decomposed orchestrator with a stubbed retrieve function).
"""

from __future__ import annotations

import pytest
from retrieval.query_decompose import (
    extract_sub_topics,
    rrf_merge,
    retrieve_decomposed,
    _build_sub_query,
)


# ── extract_sub_topics ────────────────────────────────────────────────────────

class TestExtractSubTopics:
    def test_vs_pattern(self):
        topics = extract_sub_topics("dropout vs batch normalization")
        assert len(topics) == 2
        assert any("dropout" in t.lower() for t in topics)
        assert any("batch normalization" in t.lower() for t in topics)

    def test_versus_pattern(self):
        topics = extract_sub_topics("CAPM versus APT — what's the difference?")
        assert len(topics) == 2

    def test_compare_and_pattern(self):
        topics = extract_sub_topics("compare transformers and RNNs")
        assert len(topics) == 2
        assert any("transformer" in t.lower() for t in topics)
        assert any("rnn" in t.lower() for t in topics)

    def test_difference_between_pattern(self):
        topics = extract_sub_topics("what is the difference between precision and recall?")
        assert len(topics) == 2

    def test_how_does_differ_from(self):
        topics = extract_sub_topics("how does L1 regularization differ from L2?")
        assert len(topics) == 2

    def test_no_comparison(self):
        topics = extract_sub_topics("what is backpropagation?")
        assert topics == []

    def test_vague_pronouns_rejected(self):
        # "compare them" should not produce topics
        topics = extract_sub_topics("compare them")
        assert len(topics) < 2

    def test_max_three_topics(self):
        # Even if the regex grabs more groups we should cap at _MAX_SUB_TOPICS
        topics = extract_sub_topics("Adam vs SGD vs RMSProp")
        assert len(topics) <= 3

    def test_compare_with(self):
        topics = extract_sub_topics("compare gradient boosting with random forests")
        assert len(topics) == 2


# ── _build_sub_query ──────────────────────────────────────────────────────────

class TestBuildSubQuery:
    def test_preserves_what_is_opener(self):
        sq = _build_sub_query("what is the difference between A and B", "dropout")
        assert sq.lower().startswith("what is")
        assert "dropout" in sq

    def test_long_topic_used_verbatim(self):
        long_topic = "the role of attention mechanisms in transformer architectures"
        sq = _build_sub_query("compare X and Y", long_topic)
        assert sq == long_topic


# ── rrf_merge ─────────────────────────────────────────────────────────────────

class TestRrfMerge:
    def _make_hit(self, cid: str, score: float = 1.0) -> dict:
        return {"chunk_id": cid, "text": f"text for {cid}", "score": score}

    def test_deduplicates(self):
        list1 = [self._make_hit("a"), self._make_hit("b"), self._make_hit("c")]
        list2 = [self._make_hit("b"), self._make_hit("a"), self._make_hit("d")]
        merged = rrf_merge([list1, list2], top_k=10)
        ids = [h["chunk_id"] for h in merged]
        assert len(ids) == len(set(ids)), "should have no duplicate chunk_ids"

    def test_consistent_top_hit_boosted_by_both_lists(self):
        # chunk "a" is rank-0 in both lists → should score highest
        list1 = [self._make_hit("a"), self._make_hit("b")]
        list2 = [self._make_hit("a"), self._make_hit("c")]
        merged = rrf_merge([list1, list2], top_k=5)
        assert merged[0]["chunk_id"] == "a"

    def test_top_k_respected(self):
        hits = [self._make_hit(str(i)) for i in range(20)]
        merged = rrf_merge([hits], top_k=5)
        assert len(merged) <= 5

    def test_empty_lists_handled(self):
        merged = rrf_merge([[], []])
        assert merged == []

    def test_single_list_passthrough(self):
        hits = [self._make_hit("x"), self._make_hit("y")]
        merged = rrf_merge([hits], top_k=10)
        assert [h["chunk_id"] for h in merged] == ["x", "y"]

    def test_rrf_score_attached(self):
        hits = [self._make_hit("z")]
        merged = rrf_merge([hits])
        assert "rrf_score" in merged[0]


# ── retrieve_decomposed ───────────────────────────────────────────────────────

class TestRetrieveDecomposed:
    def _stub_retrieve(self, query: str, **kwargs) -> dict:
        """Minimal stub that returns deterministic hits based on the query."""
        return {
            "hits": [
                {"chunk_id": f"chunk-for-{query[:10]}-0", "text": "text A", "score": 0.9},
                {"chunk_id": f"chunk-for-{query[:10]}-1", "text": "text B", "score": 0.7},
            ],
            "query": query,
            "top_k": kwargs.get("top_k", 10),
            "filters": {},
        }

    def test_returns_none_for_non_comparison(self):
        result = retrieve_decomposed(
            "what is backpropagation?",
            self._stub_retrieve,
            {},
            top_k=10,
        )
        assert result is None

    def test_returns_dict_for_comparison(self):
        result = retrieve_decomposed(
            "dropout vs batch normalization",
            self._stub_retrieve,
            {},
            top_k=10,
        )
        assert result is not None
        assert "hits" in result

    def test_decomposition_metadata_present(self):
        result = retrieve_decomposed(
            "compare Adam and SGD optimizers",
            self._stub_retrieve,
            {},
            top_k=10,
        )
        assert result is not None
        decomp = result.get("decomposition", {})
        assert "sub_topics" in decomp
        assert "sub_queries" in decomp
        assert len(decomp["sub_topics"]) >= 2

    def test_hits_deduplicated(self):
        def _same_hits(query, **kwargs):
            return {
                "hits": [{"chunk_id": "shared-001", "text": "t", "score": 1.0}],
                "query": query,
                "top_k": 10,
                "filters": {},
            }
        result = retrieve_decomposed(
            "dropout vs batch normalization",
            _same_hits,
            {},
            top_k=10,
        )
        assert result is not None
        ids = [h["chunk_id"] for h in result["hits"]]
        assert ids.count("shared-001") == 1

    def test_failed_subtopic_falls_back_gracefully(self):
        call_count = [0]

        def _flaky_retrieve(query, **kwargs):
            call_count[0] += 1
            if call_count[0] % 2 == 0:
                raise RuntimeError("DB error")
            return {
                "hits": [{"chunk_id": "ok-001", "text": "ok", "score": 0.8}],
                "query": query,
                "top_k": kwargs.get("top_k", 10),
                "filters": {},
            }

        result = retrieve_decomposed(
            "dropout vs batch normalization",
            _flaky_retrieve,
            {},
            top_k=10,
        )
        # Should succeed with at least the one working sub-result
        assert result is not None
        assert len(result["hits"]) > 0

    def test_all_subtopics_fail_returns_none(self):
        def _always_fail(query, **kwargs):
            raise RuntimeError("always fails")

        result = retrieve_decomposed(
            "dropout vs batch normalization",
            _always_fail,
            {},
            top_k=10,
        )
        assert result is None
