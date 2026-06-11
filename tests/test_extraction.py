"""Unit tests for extraction pipeline utilities.

Covers:
  - extraction/batch.py: _strip_leading_artifacts, _make_preview,
    parse_scan_response, assign_ids_and_batch, rehydrate,
    format_scan_batch_text
  - extraction/dedup.py: jaccard, dedup_items
"""

from __future__ import annotations

import json

import pytest

from extraction.batch import (
    _make_preview,
    _strip_leading_artifacts,
    assign_ids_and_batch,
    format_scan_batch_text,
    parse_scan_response,
    rehydrate,
)
from extraction.branch_config import (
    BranchResult,
    BranchStats,
    ExtractionItem,
    ProjectConfig,
    SecondPassConfig,
)
from extraction.dedup import dedup_items, jaccard
import extraction.project_runner as project_runner
from extraction.project_runner import _run_post_passes
from extraction.project_runner import rerun_reports_from_checkpoint
from extraction.source_kinds import infer_source_kind


# ---------------------------------------------------------------------------
# _strip_leading_artifacts
# ---------------------------------------------------------------------------

class TestStripLeadingArtifacts:
    def test_no_artifact(self):
        text = "Backpropagation is used to train neural networks."
        assert _strip_leading_artifacts(text) == text

    def test_single_letter_label(self):
        text = "N  Batch gradient descent updates weights once per epoch."
        result = _strip_leading_artifacts(text)
        assert result.startswith("Batch")

    def test_equation_fragment(self):
        text = "( x , y ):  Chain rule applies when composing functions."
        result = _strip_leading_artifacts(text)
        assert result.startswith("Chain")

    def test_real_word_stops_stripping(self):
        # "Hello  world" — first segment has a real word, must NOT be stripped
        text = "Hello  world with normal text."
        assert _strip_leading_artifacts(text) == text

    def test_empty_string(self):
        assert _strip_leading_artifacts("") == ""


# ---------------------------------------------------------------------------
# _make_preview
# ---------------------------------------------------------------------------

class TestMakePreview:
    def test_short_text_returned_whole(self):
        text = "Short text."
        assert _make_preview(text, 200) == text

    def test_truncation_at_max_chars(self):
        text = "a" * 100
        result = _make_preview(text, 50)
        assert len(result) <= 52  # 50 chars + ellipsis
        assert result.endswith("…")

    def test_zero_means_no_truncation(self):
        text = "a" * 1000
        result = _make_preview(text, 0)
        assert result == text

    def test_negative_means_no_truncation(self):
        text = "x" * 500
        assert _make_preview(text, -1) == text

    def test_newlines_collapsed(self):
        text = "line one\nline two\nline three"
        result = _make_preview(text, 500)
        assert "\n" not in result


# ---------------------------------------------------------------------------
# parse_scan_response
# ---------------------------------------------------------------------------

class TestParseScanResponse:
    def test_bracketed_ids(self):
        assert parse_scan_response("[1], [4], [7]") == [1, 4, 7]

    def test_bracketed_no_commas(self):
        assert parse_scan_response("[1] [4] [7]") == [1, 4, 7]

    def test_none_response(self):
        assert parse_scan_response("NONE") == []

    def test_none_lowercase(self):
        assert parse_scan_response("none") == []

    def test_deduplication(self):
        assert parse_scan_response("[1], [1], [3]") == [1, 3]

    def test_fallback_to_bare_integers(self):
        result = parse_scan_response("Chunks 2 and 5 are relevant.")
        assert 2 in result
        assert 5 in result

    def test_bullet_points(self):
        response = "• [1]\n• [4]\n• [9]"
        assert parse_scan_response(response) == [1, 4, 9]

    def test_empty_string(self):
        assert parse_scan_response("") == []


# ---------------------------------------------------------------------------
# assign_ids_and_batch
# ---------------------------------------------------------------------------

class TestAssignIdsAndBatch:
    def _make_chunks(self, n: int, text_len: int = 100) -> list:
        return [{"chunk_id": f"c{i}", "text": "x" * text_len} for i in range(n)]

    def test_single_batch(self):
        chunks = self._make_chunks(5, text_len=50)
        id_map, batches = assign_ids_and_batch(chunks, batch_max_chars=10000, preview_chars=80)
        assert len(batches) == 1
        assert len(id_map) == 5
        assert list(id_map.keys()) == [1, 2, 3, 4, 5]

    def test_splits_into_multiple_batches(self):
        # Each chunk preview = 80 chars + ~10 overhead = ~90 chars.
        # batch_max_chars=200 means ~2 chunks per batch for 10 chunks.
        chunks = self._make_chunks(10, text_len=200)
        id_map, batches = assign_ids_and_batch(chunks, batch_max_chars=200, preview_chars=80)
        assert len(batches) > 1
        assert sum(len(b) for b in batches) == 10

    def test_id_map_contains_all_chunks(self):
        chunks = self._make_chunks(8)
        id_map, batches = assign_ids_and_batch(chunks, batch_max_chars=50000, preview_chars=80)
        assert set(id_map.keys()) == set(range(1, 9))
        for sid, chunk in id_map.items():
            assert chunk["chunk_id"] == f"c{sid - 1}"

    def test_empty_pool(self):
        id_map, batches = assign_ids_and_batch([])
        assert id_map == {}
        assert batches == []


# ---------------------------------------------------------------------------
# rehydrate
# ---------------------------------------------------------------------------

class TestRehydrate:
    def test_basic_lookup(self):
        chunks = [{"chunk_id": "a", "text": "alpha"}, {"chunk_id": "b", "text": "beta"}]
        id_map = {1: chunks[0], 2: chunks[1]}
        result = rehydrate([2, 1], id_map)
        assert [c["chunk_id"] for c in result] == ["b", "a"]

    def test_unknown_ids_ignored(self):
        id_map = {1: {"chunk_id": "x", "text": "text"}}
        result = rehydrate([1, 99, 42], id_map)
        assert len(result) == 1

    def test_deduplication(self):
        chunk = {"chunk_id": "dup", "text": "duplicate text"}
        id_map = {1: chunk, 2: chunk}
        result = rehydrate([1, 2], id_map)
        assert len(result) == 1

    def test_empty_ids(self):
        id_map = {1: {"chunk_id": "x", "text": "text"}}
        assert rehydrate([], id_map) == []


# ---------------------------------------------------------------------------
# format_scan_batch_text
# ---------------------------------------------------------------------------

class TestFormatScanBatchText:
    def test_format(self):
        batch = [(1, "First preview"), (2, "Second preview")]
        result = format_scan_batch_text(batch)
        assert result == "[1] First preview\n[2] Second preview"

    def test_with_prefix(self):
        batch = [(3, "Some text")]
        result = format_scan_batch_text(batch, id_prefix="B1-")
        assert result == "[B1-3] Some text"

    def test_empty_batch(self):
        assert format_scan_batch_text([]) == ""


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_identical(self):
        assert jaccard("hello world", "hello world") == pytest.approx(1.0)

    def test_disjoint(self):
        assert jaccard("foo bar", "baz qux") == pytest.approx(0.0)

    def test_partial_overlap(self):
        # {"a", "b"} ∩ {"b", "c"} = {"b"}, union = {"a","b","c"} → 1/3
        score = jaccard("a b", "b c")
        assert score == pytest.approx(1 / 3)

    def test_both_empty(self):
        assert jaccard("", "") == pytest.approx(1.0)

    def test_one_empty(self):
        assert jaccard("hello", "") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# dedup_items
# ---------------------------------------------------------------------------

class TestDedupItems:
    def _item(self, text: str, priority: float = 1.0) -> ExtractionItem:
        return ExtractionItem(text=text, branch_name="test", priority_weight=priority)

    def test_no_duplicates_unchanged(self):
        items = [self._item("alpha beta gamma"), self._item("delta epsilon zeta")]
        result = dedup_items(items, threshold=0.70)
        assert len(result) == 2

    def test_exact_duplicate_removed(self):
        items = [self._item("the quick brown fox"), self._item("the quick brown fox")]
        result = dedup_items(items, threshold=0.70)
        assert len(result) == 1

    def test_near_duplicate_removed(self):
        # Overlap > 0.70 → should be deduped
        a = "The system shall provide remote monitoring of all field devices"
        b = "The system shall provide remote monitoring of all field devices in real time"
        items = [self._item(a), self._item(b)]
        result = dedup_items(items, threshold=0.70)
        assert len(result) == 1

    def test_higher_priority_wins(self):
        low = self._item("The valve shall close within 5 seconds", priority=1.0)
        high = self._item("The valve shall close within 5 seconds of signal", priority=2.0)
        result = dedup_items([low, high], threshold=0.70)
        assert len(result) == 1
        assert result[0].priority_weight == pytest.approx(2.0)

    def test_empty_list(self):
        assert dedup_items([], threshold=0.70) == []

    def test_threshold_respected(self):
        # With threshold=1.0 only exact duplicates are removed
        a = "alpha beta gamma delta"
        b = "alpha beta gamma epsilon"
        items = [self._item(a), self._item(b)]
        result = dedup_items(items, threshold=1.0)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# branch_runner — pure-function tests (no LLM / no DB)
# ---------------------------------------------------------------------------

from extraction.branch_runner import (
    ExtractionConfig,
    _apply_keyword_filter_and_boost,
    _call_llm_with_retry,
    compile_keyword_regexes,
)
from extraction.branch_config import BranchConfig


class TestCompileKeywordRegexes:
    def test_single_keyword(self):
        pat = compile_keyword_regexes(["backpropagation"])
        assert pat is not None
        assert pat.search("Backpropagation is key")

    def test_multiple_keywords(self):
        pat = compile_keyword_regexes(["alpha", "beta"])
        assert pat.search("alpha blending")
        assert pat.search("beta testing")

    def test_special_chars_escaped(self):
        pat = compile_keyword_regexes(["C++"])
        assert pat is not None
        assert pat.search("C++ is fast")

    def test_empty_list_returns_none(self):
        assert compile_keyword_regexes([]) is None

    def test_regex_mode_passthrough(self):
        pat = compile_keyword_regexes([r"\bGPT-\d+\b"], keywords_are_regex=True)
        assert pat.search("GPT-4 was released")
        assert not pat.search("GPT is a model family")


class TestApplyKeywordFilterAndBoost:
    def _chunk(self, text: str, score: float = 0.5, cid: str = "c1") -> dict:
        return {"chunk_id": cid, "text": text, "score": score, "source_filename": "doc.pdf"}

    def _branch(self, mode: str = "keyword", keywords=None) -> BranchConfig:
        return BranchConfig(
            name="test",
            topic_description="test topic",
            mode=mode,
            keywords=keywords or ["gradient"],
        )

    def test_keyword_mode_filters_non_matching(self):
        chunks = [
            self._chunk("gradient descent is used for optimisation", cid="c1"),
            self._chunk("this sentence has nothing relevant", cid="c2"),
        ]
        config = ExtractionConfig()
        result = _apply_keyword_filter_and_boost(chunks, self._branch(), config)
        assert len(result) == 1
        assert result[0]["chunk_id"] == "c1"

    def test_keyword_mode_boosts_score(self):
        chunks = [self._chunk("gradient descent gradient optimisation gradient", cid="c1")]
        config = ExtractionConfig(keyword_score_boost=0.10)
        result = _apply_keyword_filter_and_boost(chunks, self._branch(), config)
        # 3 keyword hits × 0.10 boost = +0.30
        assert result[0]["score"] > 0.5

    def test_semantic_mode_skips_filter(self):
        chunks = [
            self._chunk("neural networks learn representations", cid="c1"),
            self._chunk("no keywords here at all", cid="c2"),
        ]
        branch = self._branch(mode="semantic", keywords=["gradient"])
        config = ExtractionConfig()
        result = _apply_keyword_filter_and_boost(chunks, branch, config)
        assert len(result) == 2  # semantic mode: no hard filter

    def test_sorted_descending_by_score(self):
        chunks = [
            self._chunk("gradient good", score=0.3, cid="c1"),
            self._chunk("gradient better", score=0.7, cid="c2"),
            self._chunk("gradient best", score=0.5, cid="c3"),
        ]
        config = ExtractionConfig()
        result = _apply_keyword_filter_and_boost(chunks, self._branch(), config)
        scores = [c["score"] for c in result]
        assert scores == sorted(scores, reverse=True)

    def test_max_candidate_chunks_respected(self):
        chunks = [self._chunk(f"gradient chunk {i}", score=float(i) / 100, cid=f"c{i}") for i in range(20)]
        config = ExtractionConfig(max_candidate_chunks=5)
        result = _apply_keyword_filter_and_boost(chunks, self._branch(), config)
        assert len(result) <= 5


class TestExtractionConfig:
    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.batch_max_chars == 10000
        assert cfg.scan_chunk_preview_chars == 80
        assert cfg.max_candidate_chunks == 500

    def test_from_yaml_missing_file_returns_defaults(self):
        cfg = ExtractionConfig.from_yaml(path="nonexistent_file_that_does_not_exist.yaml")
        assert cfg.batch_max_chars == 10000

    def test_from_yaml_partial_override(self, tmp_path):
        import yaml
        data = {"batching": {"batch_max_chars": 5000, "scan_chunk_preview_chars": 120}}
        yaml_file = tmp_path / "extraction.yaml"
        yaml_file.write_text(yaml.dump(data), encoding="utf-8")
        cfg = ExtractionConfig.from_yaml(path=str(yaml_file))
        assert cfg.batch_max_chars == 5000
        assert cfg.scan_chunk_preview_chars == 120
        assert cfg.max_candidate_chunks == 500  # unchanged default


class TestCallLlmWithRetry:
    def test_succeeds_on_first_attempt(self):
        calls = []

        def llm_fn(**kw):
            calls.append(1)
            return "result"

        out = _call_llm_with_retry(llm_fn, "sys", "usr", temperature=0.0, timeout_seconds=5.0, max_retries=2)
        assert out == "result"
        assert len(calls) == 1

    def test_retries_on_exception(self):
        attempts = []

        def flaky(**kw):
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("transient error")
            return "ok"

        out = _call_llm_with_retry(flaky, "sys", "usr", temperature=0.0, timeout_seconds=5.0, max_retries=2)
        assert out == "ok"
        assert len(attempts) == 3

    def test_raises_after_max_retries_exhausted(self):
        def always_fails(**kw):
            raise ValueError("boom")

        with pytest.raises(RuntimeError, match="LLM call failed"):
            _call_llm_with_retry(always_fails, "sys", "usr", temperature=0.0, timeout_seconds=5.0, max_retries=1)


# ---------------------------------------------------------------------------
# branch_runner — LLM integration tests (injected fake LLM, no Ollama)
# ---------------------------------------------------------------------------

from extraction.branch_runner import (
    CorpusHandle,
    _scan_batch,
    _synthesis_batch,
    run_branch,
)
from extraction.branch_config import BranchResult, BranchStats
from extraction.batch import assign_ids_and_batch


def _make_corpus(chunks: list[dict]) -> CorpusHandle:
    """Build a minimal CorpusHandle with in-memory chunk rows."""
    return CorpusHandle(
        db_dsn="postgresql://unused/unused",
        collection_id="col-test",
        chunk_rows=chunks,
    )


def _make_branch(name: str = "test", keywords: list[str] | None = None) -> BranchConfig:
    return BranchConfig(
        name=name,
        topic_description="Gradient descent and optimisation",
        mode="keyword",
        keywords=keywords or ["gradient"],
    )


def _make_chunk(cid: str, text: str, score: float = 0.7) -> dict:
    return {
        "chunk_id": cid,
        "text": text,
        "score": score,
        "source_filename": "book.pdf",
        "doc_id": "doc-1",
    }


class TestScanBatch:
    """Tests for _scan_batch with a stubbed llm_fn."""

    def _batch(self, chunks: list[dict]) -> list[tuple[int, dict]]:
        id_map, batches = assign_ids_and_batch(
            chunks, batch_max_chars=10000, preview_chars=80
        )
        return batches[0] if batches else []

    def test_returns_selected_ids_and_prompts(self):
        chunks = [_make_chunk("c1", "gradient descent optimisation algorithm")]
        branch = _make_branch()
        config = ExtractionConfig(max_retries=0)
        batch = self._batch(chunks)

        def llm_fn(**kw):
            # Return a scan response that selects ID 1
            return "1"

        selected, sys_p, usr_p, raw = _scan_batch(batch, branch, config, llm_fn)
        assert 1 in selected
        assert isinstance(sys_p, str) and len(sys_p) > 0
        assert isinstance(usr_p, str) and len(usr_p) > 0
        assert raw == "1"

    def test_empty_llm_response_selects_nothing(self):
        chunks = [_make_chunk("c1", "gradient learning rate")]
        branch = _make_branch()
        config = ExtractionConfig(max_retries=0)
        batch = self._batch(chunks)

        selected, _, _, _ = _scan_batch(batch, branch, config, lambda **kw: "")
        assert selected == []

    def test_llm_fn_receives_temperature_and_timeout(self):
        received = {}
        chunks = [_make_chunk("c1", "gradient descent step")]
        branch = _make_branch()
        config = ExtractionConfig(scan_pass_temperature=0.03, scan_pass_timeout_seconds=45.0, max_retries=0)
        batch = self._batch(chunks)

        def capturing_llm(**kw):
            received.update(kw)
            return "1"

        _scan_batch(batch, branch, config, capturing_llm)
        assert received.get("temperature") == pytest.approx(0.03)
        assert received.get("timeout_seconds") == pytest.approx(45.0)


class TestSynthesisBatch:
    """Tests for _synthesis_batch with a stubbed llm_fn."""

    def _synthesis_batch_input(self, chunks: list[dict]) -> list[tuple[int, dict]]:
        id_map, batches = assign_ids_and_batch(
            chunks, batch_max_chars=10000, preview_chars=80
        )
        # Re-pair with sequential IDs for synthesis pass
        return list(id_map.items())

    def test_returns_ids_and_raw_response(self):
        chunks = [_make_chunk("c1", "stochastic gradient descent with momentum")]
        branch = _make_branch()
        config = ExtractionConfig(max_retries=0)
        batch = self._synthesis_batch_input(chunks)

        selected, sys_p, usr_p, raw = _synthesis_batch(batch, branch, config, lambda **kw: "1")
        assert isinstance(selected, list)
        assert isinstance(raw, str)

    def test_synthesis_temperature_forwarded(self):
        received = {}
        chunks = [_make_chunk("c1", "gradient vanishing problem in deep nets")]
        branch = _make_branch()
        config = ExtractionConfig(synthesis_pass_temperature=0.10, max_retries=0)
        batch = self._synthesis_batch_input(chunks)

        def capturing(**kw):
            received.update(kw)
            return ""

        _synthesis_batch(batch, branch, config, capturing)
        assert received.get("temperature") == pytest.approx(0.10)


class TestRunBranch:
    """Integration tests for run_branch with injected fake LLMs and in-memory corpus."""

    def _chunks(self, n: int = 3) -> list[dict]:
        return [
            _make_chunk(f"c{i}", f"gradient descent iteration {i} learning", score=0.8 - i * 0.05)
            for i in range(1, n + 1)
        ]

    def _corpus(self, n: int = 3) -> CorpusHandle:
        return _make_corpus(self._chunks(n))

    def test_disabled_branch_returns_empty(self):
        branch = BranchConfig(
            name="disabled-branch",
            topic_description="Anything",
            mode="keyword",
            keywords=["x"],
            enabled=False,
        )
        corpus = self._corpus()
        config = ExtractionConfig()
        result = run_branch(branch, corpus, config, llm_fn=lambda **kw: "")
        assert result.status == "empty"
        assert result.branch_name == "disabled-branch"

    def test_no_matching_chunks_returns_empty(self):
        """Keyword filter removes all chunks when none match, returning empty without LLM call."""
        branch = _make_branch(keywords=["xyzzy_nonexistent_keyword_zz9"])
        # Patch _retrieve_candidates to return chunks that don't match the keyword
        from unittest.mock import patch
        chunks = [_make_chunk("c1", "neural network backpropagation")]
        with patch("extraction.branch_runner._retrieve_candidates", return_value=chunks):
            config = ExtractionConfig(max_retries=0)
            result = run_branch(branch, corpus=self._corpus(), config=config, llm_fn=lambda **kw: "1")
        assert result.status == "empty"

    def test_scan_pass_selects_chunks(self):
        branch = _make_branch(keywords=["gradient"])
        config = ExtractionConfig(max_retries=0)
        scan_calls: list[str] = []

        def llm_fn(**kw):
            scan_calls.append(kw.get("user_prompt", ""))
            return "1"  # always select ID 1

        from unittest.mock import patch
        with patch("extraction.branch_runner._retrieve_candidates", return_value=self._chunks(2)):
            result = run_branch(branch, corpus=self._corpus(), config=config, llm_fn=llm_fn)

        assert len(scan_calls) >= 1
        assert result.status in ("ok", "empty")

    def test_emit_callback_receives_progress_messages(self):
        branch = _make_branch(keywords=["gradient"])
        config = ExtractionConfig(max_retries=0)
        messages: list[str] = []

        from unittest.mock import patch
        with patch("extraction.branch_runner._retrieve_candidates", return_value=self._chunks(2)):
            run_branch(branch, corpus=self._corpus(), config=config, llm_fn=lambda **kw: "1", emit=messages.append)

        assert len(messages) > 0
        assert any("chunk" in m.lower() or "scan" in m.lower() or "→" in m for m in messages)

    def test_result_branch_name_matches_config(self):
        branch = _make_branch(name="gradient-methods")
        config = ExtractionConfig(max_retries=0)

        from unittest.mock import patch
        with patch("extraction.branch_runner._retrieve_candidates", return_value=self._chunks(1)):
            result = run_branch(branch, corpus=self._corpus(), config=config, llm_fn=lambda **kw: "")
        assert result.branch_name == "gradient-methods"

    def test_llm_error_sets_error_status(self):
        branch = _make_branch(keywords=["gradient"])
        config = ExtractionConfig(max_retries=0)

        def always_raises(**kw):
            raise RuntimeError("LLM timeout")

        from unittest.mock import patch
        with patch("extraction.branch_runner._retrieve_candidates", return_value=self._chunks(2)):
            result = run_branch(branch, corpus=self._corpus(), config=config, llm_fn=always_raises)
        assert result.status == "error"
        assert result.stats.error is not None


class TestSecondPassPipeline:
    def _write_branch_checkpoint(self, tmp_path) -> object:
        ckpt_path = tmp_path / "checkpoint.jsonl"
        branch_results = [
            BranchResult(
                branch_name="Scope",
                output_heading="Scope",
                items=[
                    ExtractionItem(
                        text="PLC cabinet required",
                        branch_name="Scope",
                        source_chunk_id="doc-c000001",
                        source_filename="spec.pdf",
                        source_page=12,
                    )
                ],
                stats=BranchStats(),
                status="ok",
            ),
            BranchResult(
                branch_name="Requirements",
                output_heading="Requirements",
                items=[
                    ExtractionItem(
                        text="SCADA integration required",
                        branch_name="Requirements",
                        source_chunk_id="doc-c000002",
                        source_filename="spec.pdf",
                        source_page=14,
                    )
                ],
                stats=BranchStats(),
                status="ok",
            ),
        ]
        ckpt_path.write_text(
            "\n".join(json.dumps({"type": "branch_result", "data": result.to_dict()}) for result in branch_results) + "\n",
            encoding="utf-8",
        )
        return ckpt_path

    def test_report_plan_is_created_before_configured_second_passes(self, tmp_path):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="second-pass-plan",
            name="Second Pass Plan",
            second_passes=[
                SecondPassConfig(name="Organize by Category", pass_type="organize_by_category"),
            ],
        )

        results = _run_post_passes(project, ckpt_path, lambda **_kw: "unused", lambda _: None)
        assert [result.pass_name for result in results] == [
            "report_plan",
            "Organize by Category",
            "Assemble Report",
        ]
        assert results[0].artifact_type == "report_plan_v1"
        assert results[0].artifact_data["sections"][0]["heading"] == "Organize by Category"

    def test_organize_by_category_builds_category_artifact(self, tmp_path):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="organize-pass-test",
            name="Organize Pass Test",
            second_passes=[
                SecondPassConfig(name="Organize by Category", pass_type="organize_by_category"),
            ],
        )

        results = _run_post_passes(project, ckpt_path, lambda **_kw: "unused", lambda _: None)
        organized = results[1]
        assert organized.artifact_type == "category_organizer_v1"
        categories = organized.artifact_data["categories"]
        assert [cat["name"] for cat in categories] == ["Scope", "Requirements"]
        assert any("[chunk:doc-c000001]" in line for line in categories[0]["evidence_lines"])
        assert "### Scope" in organized.response_text
        assert "### Requirements" in organized.response_text

    def test_summarize_by_category_auto_organizes_when_needed(self, tmp_path):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="summarize-pass-test",
            name="Summarize Pass Test",
            second_passes=[
                SecondPassConfig(name="Summarize by Category", pass_type="summarize_by_category"),
            ],
        )
        prompts = []

        def fake_llm(**kw):
            prompts.append(kw["user_prompt"])
            if "Category: Scope" in kw["user_prompt"]:
                return "Scope summary. [chunk:doc-c000001]"
            return "Requirements summary. [chunk:doc-c000002]"

        results = _run_post_passes(project, ckpt_path, fake_llm, lambda _: None)
        summarized = results[1]
        assert summarized.artifact_type == "category_summaries_v1"
        assert any("Category: Scope" in prompt for prompt in prompts)
        assert any("Category: Requirements" in prompt for prompt in prompts)
        assert "### Scope" in summarized.response_text
        assert "### Requirements" in summarized.response_text
        assert any("Source types represented in this category:" in prompt for prompt in prompts)
        assert any("Cite substantive claims" in prompt for prompt in prompts)

    def test_assemble_report_follows_planned_block_order(self, tmp_path):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="assemble-pass-test",
            name="Assemble Pass Test",
            second_passes=[
                SecondPassConfig(name="Executive Summary", pass_type="executive_summary"),
                SecondPassConfig(name="Summarize by Category", pass_type="summarize_by_category"),
                SecondPassConfig(name="Organize by Category", pass_type="organize_by_category"),
                SecondPassConfig(name="Assemble Report", pass_type="assemble_report"),
            ],
        )

        def fake_llm(**kw):
            prompt = kw["user_prompt"]
            if "Category: Scope" in prompt:
                return "Scope summary. [chunk:doc-c000001]"
            if "Category: Requirements" in prompt:
                return "Requirements summary. [chunk:doc-c000002]"
            if "executive summary" in prompt.lower():
                return "Top-line summary."
            return "Unexpected"

        results = _run_post_passes(project, ckpt_path, fake_llm, lambda _: None)
        assembled = results[-1]
        assert assembled.artifact_type == "assembled_report_v1"
        text = assembled.response_text
        assert text.startswith("# Assemble Pass Test Report")
        assert text.index("### Executive Summary") < text.index("### Scope")
        assert text.index("### Scope") < text.index("## Organized Evidence by Category")


class TestSourceKinds:
    def test_detects_rfq(self):
        assert infer_source_kind(text="Request for Qualifications for SCADA migration", filename="spec.pdf") == "rfq"

    def test_detects_workshop_note(self):
        assert infer_source_kind(text="Workshop No. 2\nAttendees:\nDiscussion Items:", filename="notes.pdf") == "workshop_note"

    def test_detects_standard_without_overmatching_appendix(self):
        assert infer_source_kind(
            text="SCADA Standards Volume 3 HMI software display definitions and ChangeSet files.",
            filename="reference.pdf",
        ) == "scada_standard"

    def test_detects_appendix_form_from_form_signals(self):
        assert infer_source_kind(
            text="Respondent Certification\nSignature of Respondent\nNotary Public",
            filename="appendix.pdf",
        ) == "appendix_form"

    def test_key_findings_and_next_actions_use_prior_outputs(self, tmp_path):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="findings-actions",
            name="Findings and Actions",
            second_passes=[
                SecondPassConfig(name="Summarize by Category", pass_type="summarize_by_category"),
                SecondPassConfig(name="Key Findings", pass_type="key_findings"),
                SecondPassConfig(name="Next Actions", pass_type="next_actions"),
                SecondPassConfig(name="Assemble Report", pass_type="assemble_report"),
            ],
        )
        prompts = []

        def fake_llm(**kw):
            prompts.append(kw["user_prompt"])
            prompt = kw["user_prompt"]
            if "Category: Scope" in prompt:
                return "Scope summary. [chunk:doc-c000001]"
            if "Category: Requirements" in prompt:
                return "Requirements summary. [chunk:doc-c000002]"
            if "Key Findings" in prompt:
                return "- Finding one\n- Finding two"
            if "Next Actions" in prompt:
                return "- Review PLC cabinet scope\n- Confirm SCADA interfaces"
            return "Fallback"

        results = _run_post_passes(project, ckpt_path, fake_llm, lambda _: None)
        key_findings = next(result for result in results if result.pass_name == "Key Findings")
        next_actions = next(result for result in results if result.pass_name == "Next Actions")
        assert key_findings.artifact_type == "key_findings_v1"
        assert next_actions.artifact_type == "next_actions_v1"
        assert any("Scope summary." in prompt for prompt in prompts if "Key Findings" in prompt)
        assert any("Finding one" in prompt for prompt in prompts if "Next Actions" in prompt)
        assert any("Each bullet should include at least one lightweight source citation" in prompt for prompt in prompts if "Key Findings" in prompt)
        assert any("Each action should include a lightweight source citation" in prompt for prompt in prompts if "Next Actions" in prompt)

    def test_rerun_reports_from_checkpoint_reuses_branch_outputs(self, tmp_path, monkeypatch):
        ckpt_path = self._write_branch_checkpoint(tmp_path)
        project = ProjectConfig(
            slug="rerun-second-passes",
            name="Rerun Second Passes",
            second_passes=[
                SecondPassConfig(name="Executive Summary", pass_type="executive_summary"),
            ],
        )

        monkeypatch.setattr(project_runner, "_make_llm_fn", lambda: (lambda **_kw: "checkpoint-based summary"))

        result = rerun_reports_from_checkpoint(
            project,
            checkpoint_path=str(ckpt_path),
            emit=lambda _: None,
        )

        assert result.second_pass_results[1].status == "ok"
        assert result.second_pass_results[1].response_text == "### Executive Summary\ncheckpoint-based summary"
        assert len(result.branch_results) == 2
        assert result.branch_results[0].items[0].text == "PLC cabinet required"
        assert result.checkpoint_path
        assert "checkpoint-based summary" in result.report_markdown


