"""
Tests for query_cli.py — verifies that source-type routing and filter wiring
work correctly without hitting the DB or Ollama.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters


# ── Fixtures ───────────────────────────────────────────────────────────────────

_EMPTY_RESULT = {
    "query": "test",
    "top_k": 5,
    "filters": {},
    "hits": [],
    "context_pack": {},
}


def _run_cli(argv: list[str]) -> dict:
    """Run query_cli.main() with the given argv, return parsed stdout JSON."""
    captured: list[str] = []

    with patch("sys.argv", ["query_cli.py"] + argv), \
         patch("scripts.ops.query_cli.retrieve_as_dict", return_value=_EMPTY_RESULT) as mock_retrieve, \
         patch("scripts.ops.query_cli.build_context_pack", return_value={"selected_chunks": []}) as mock_pack, \
         patch("retrieval.router.classify_intent", return_value=("factual_lookup", {"matched_pattern": "mock"})), \
         patch("builtins.print", side_effect=lambda *a, **kw: captured.append(str(a[0]))):
        from scripts.ops.query_cli import main
        main()

    return mock_retrieve, captured


# ── Source-type filter injection ───────────────────────────────────────────────

class TestSourceTypeRouting:
    """Verify the detected source-type filter reaches retrieve_as_dict."""

    def test_notes_query_sets_source_type_filter(self):
        """'my notes' in query → filters.source_type == 'notes'."""
        mock_retrieve, _ = _run_cli(["what do my notes say about attention?"])
        assert mock_retrieve.call_count == 1
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type == "notes"

    def test_pdf_query_sets_source_type_filter(self):
        """'the books' in query → filters.source_type == 'pdf_book'."""
        mock_retrieve, _ = _run_cli(["what do the books say about backpropagation?"])
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type == "pdf_book"

    def test_generic_query_no_source_type_filter(self):
        """No source hint → filters.source_type is None."""
        mock_retrieve, _ = _run_cli(["what is gradient descent?"])
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type is None

    def test_explicit_source_type_flag_takes_priority(self):
        """--source-type flag overrides auto-detected filter."""
        mock_retrieve, _ = _run_cli([
            "what do my notes say about attention?",
            "--source-type", "pdf_book",
        ])
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type == "pdf_book"

    def test_explicit_source_type_flag_generic_query(self):
        """--source-type flag works on a generic query too."""
        mock_retrieve, _ = _run_cli([
            "what is gradient descent?",
            "--source-type", "notes",
        ])
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type == "notes"

    def test_only_one_retrieve_call_made(self):
        """CLI should make exactly one retrieve_as_dict call (no wasted pre-route call)."""
        mock_retrieve, _ = _run_cli(["what do my notes say about attention?"])
        assert mock_retrieve.call_count == 1

    def test_only_one_retrieve_call_generic(self):
        """Single retrieve call for generic queries too."""
        mock_retrieve, _ = _run_cli(["what is backpropagation?"])
        assert mock_retrieve.call_count == 1


# ── Hard filters bypass routing ───────────────────────────────────────────────

class TestHardFiltersSkipStrategyOverride:
    """When hard filters (--doc-id etc.) are set, strategy overrides are skipped."""

    def test_doc_id_uses_user_top_k(self):
        mock_retrieve, _ = _run_cli([
            "what is attention?",
            "--doc-id", "abc123",
            "--top-k", "3",
        ])
        assert mock_retrieve.call_args[1]["top_k"] == 3

    def test_doc_id_with_source_type_flag_respected(self):
        mock_retrieve, _ = _run_cli([
            "what is attention?",
            "--doc-id", "abc123",
            "--source-type", "notes",
        ])
        filters: RetrievalFilters = mock_retrieve.call_args[1]["filters"]
        assert filters.source_type == "notes"
        assert filters.doc_id == "abc123"
