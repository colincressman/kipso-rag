"""Tests for overview-query detection and document summary DB helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.coverage import is_overview_query


# ── is_overview_query ─────────────────────────────────────────────────────────

class TestIsOverviewQuery:
    def test_what_is_this_book_about(self):
        assert is_overview_query("what is this book about?")

    def test_what_is_the_book_about(self):
        assert is_overview_query("What is the book about?")

    def test_what_does_this_book_cover(self):
        assert is_overview_query("What does this book cover?")

    def test_summarize(self):
        assert is_overview_query("Summarize this document")

    def test_summarise_british(self):
        assert is_overview_query("summarise the paper")

    def test_give_me_a_summary(self):
        assert is_overview_query("give me a summary of this")

    def test_give_me_an_overview(self):
        assert is_overview_query("give me an overview of the textbook")

    def test_overview_of(self):
        assert is_overview_query("overview of this document")

    def test_what_topics_does_this_cover(self):
        assert is_overview_query("what topics does this cover?")

    def test_what_is_covered_in(self):
        assert is_overview_query("what is covered in this book?")

    def test_describe_this_book(self):
        assert is_overview_query("describe this book")

    def test_tell_me_about_this_document(self):
        assert is_overview_query("tell me about this document")

    def test_what_is_X_about(self):
        assert is_overview_query("What is Introduction to Statistical Learning about?")

    def test_give_me_a_brief_overview(self):
        assert is_overview_query("give me a brief overview of the content")

    # Negative cases — these should NOT trigger overview mode
    def test_specific_question_not_overview(self):
        assert not is_overview_query("what is gradient descent?")

    def test_factoid_not_overview(self):
        assert not is_overview_query("who wrote this book?")

    def test_formula_not_overview(self):
        assert not is_overview_query("what is the formula for cross entropy?")

    def test_definition_not_overview(self):
        assert not is_overview_query("what is a neural network?")

    def test_year_not_overview(self):
        assert not is_overview_query("when was this published?")

    def test_empty_string(self):
        assert not is_overview_query("")

    def test_none_like_empty(self):
        assert not is_overview_query("   ")


# ── DB summary helpers ────────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestSummaryChunkDB:
    """Tests for upsert_summary_chunk, fetch_representative_chunks, get_docs_without_summary."""

    @pytest.fixture()
    def db(self, pg_dsn):
        """Return an initialized DB DSN with one document and some chunks."""
        from db.client import init_db, upsert_document_record, upsert_chunks_from_index

        db_path = pg_dsn
        init_db(db_path)

        upsert_document_record(
            db_path,
            doc_id="doc001",
            filename="mybook.pdf",
            source_path="/data/mybook.pdf",
            source_type="pdf_book",
            num_pages=10,
            metadata={},
        )

        payload = {
            "full_document": False,
            "items": [
                {
                    "chunk_id": "doc001_c000001",
                    "doc_id": "doc001",
                    "section_id": "s1",
                    "collection_id": "books",
                    "source_name": "My Book",
                    "document_title": "My Book Title",
                    "document_path": "/data/mybook.pdf",
                    "path_text": "Introduction",
                    "title": "Introduction",
                    "level": 1,
                    "page_start": 1,
                    "page_end": 2,
                    "has_table": False,
                    "token_count_est": 20,
                    "source_type": "pdf_book",
                    "structural_role": "introduction",
                    "text": "This book covers deep learning fundamentals.",
                    "embedding": [0.1, 0.2, 0.3] + [0.0] * 4093,
                },
                {
                    "chunk_id": "doc001_c000002",
                    "doc_id": "doc001",
                    "section_id": "s2",
                    "collection_id": "books",
                    "source_name": "My Book",
                    "document_title": "My Book Title",
                    "document_path": "/data/mybook.pdf",
                    "path_text": "Chapter 1",
                    "title": "Chapter 1",
                    "level": 1,
                    "page_start": 3,
                    "page_end": 10,
                    "has_table": False,
                    "token_count_est": 30,
                    "source_type": "pdf_book",
                    "structural_role": "body",
                    "text": "Neural networks are composed of layers.",
                    "embedding": [0.4, 0.5, 0.6] + [0.0] * 4093,
                },
            ],
        }
        upsert_chunks_from_index(db_path, payload)
        return db_path

    def test_get_docs_without_summary_returns_doc(self, db):
        from db.client import get_docs_without_summary
        pending = get_docs_without_summary(db)
        assert len(pending) == 1
        assert pending[0]["doc_id"] == "doc001"

    def test_fetch_representative_chunks(self, db):
        from db.client import fetch_representative_chunks
        chunks = fetch_representative_chunks(db, "doc001")
        assert len(chunks) >= 1
        # Introduction should come first (structural_role priority)
        assert chunks[0]["structural_role"] == "introduction"

    def test_fetch_representative_chunks_respects_max_chars(self, db):
        from db.client import fetch_representative_chunks
        # Max chars smaller than the first chunk text
        chunks = fetch_representative_chunks(db, "doc001", max_chars=10)
        # Should still return at least one chunk (first chunk always included)
        assert len(chunks) >= 1

    def test_upsert_summary_chunk_creates_chunk(self, db):
        from db.client import upsert_summary_chunk, get_docs_without_summary
        upsert_summary_chunk(
            db,
            doc_id="doc001",
            summary_text="This book is a comprehensive guide to deep learning.",
            embedding=[0.7, 0.8, 0.9] + [0.0] * 4093,
            collection_id="books",
            document_title="My Book Title",
            source_name="My Book",
            document_path="/data/mybook.pdf",
            source_type="pdf_book",
        )
        # doc should no longer appear in pending list
        pending = get_docs_without_summary(db)
        assert not any(d["doc_id"] == "doc001" for d in pending)

    def test_upsert_summary_chunk_is_idempotent(self, db):
        from db.client import upsert_summary_chunk
        for _ in range(2):
            upsert_summary_chunk(
                db,
                doc_id="doc001",
                summary_text="Updated summary text.",
                embedding=[0.1, 0.1, 0.1] + [0.0] * 4093,
                source_type="pdf_book",
            )
        # No exception → idempotent


# ── Page-range summary DB helpers ─────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestPageRangeSummaryDB:
    """Tests for upsert_page_range_summary_chunk, list_page_range_summaries,
    get_page_range_summary_text, and fetch_chunks_in_page_range."""

    @pytest.fixture()
    def db(self, pg_dsn):
        """DB with one document and three body chunks spread across pages 1-30."""
        from db.client import init_db, upsert_document_record, upsert_chunks_from_index

        init_db(pg_dsn)
        upsert_document_record(
            pg_dsn,
            doc_id="rdoc001",
            filename="rangebook.pdf",
            source_path="/data/rangebook.pdf",
            source_type="pdf_book",
            num_pages=30,
            metadata={},
        )
        payload = {
            "full_document": False,
            "items": [
                {
                    "chunk_id": f"rdoc001_c{i:06d}",
                    "doc_id": "rdoc001",
                    "section_id": f"s{i}",
                    "collection_id": "books",
                    "source_name": "Range Book",
                    "document_title": "Range Book Title",
                    "document_path": "/data/rangebook.pdf",
                    "path_text": f"Chapter {i}",
                    "title": f"Chapter {i}",
                    "level": 1,
                    "page_start": (i - 1) * 10 + 1,
                    "page_end": i * 10,
                    "has_table": False,
                    "token_count_est": 50,
                    "source_type": "pdf_book",
                    "structural_role": "body",
                    "text": f"Content of chapter {i} covering pages {(i-1)*10+1} to {i*10}.",
                    "embedding": [float(i) * 0.1] * 10 + [0.0] * 4086,
                }
                for i in range(1, 4)
            ],
        }
        upsert_chunks_from_index(pg_dsn, payload)
        return pg_dsn

    # ── fetch_chunks_in_page_range ────────────────────────────────────────────

    def test_fetch_chunks_in_full_range(self, db):
        from db.client import fetch_chunks_in_page_range
        chunks = fetch_chunks_in_page_range(db, "rdoc001", 1, 30)
        assert len(chunks) == 3

    def test_fetch_chunks_partial_range(self, db):
        from db.client import fetch_chunks_in_page_range
        chunks = fetch_chunks_in_page_range(db, "rdoc001", 1, 10)
        assert len(chunks) == 1
        assert chunks[0]["page_start"] == 1

    def test_fetch_chunks_empty_range(self, db):
        from db.client import fetch_chunks_in_page_range
        chunks = fetch_chunks_in_page_range(db, "rdoc001", 100, 200)
        assert chunks == []

    def test_fetch_chunks_wrong_doc(self, db):
        from db.client import fetch_chunks_in_page_range
        chunks = fetch_chunks_in_page_range(db, "nonexistent", 1, 30)
        assert chunks == []

    # ── upsert_page_range_summary_chunk ──────────────────────────────────────

    def test_upsert_creates_chunk(self, db):
        from db.client import upsert_page_range_summary_chunk, list_page_range_summaries
        upsert_page_range_summary_chunk(
            db,
            doc_id="rdoc001",
            page_start=1,
            page_end=10,
            summary_text="Summary of pages 1-10.",
            embedding=[0.5] * 4096,
            document_title="Range Book Title",
            source_name="Range Book",
        )
        rows = list_page_range_summaries(db, "rdoc001")
        assert len(rows) == 1
        assert rows[0]["page_start"] == 1
        assert rows[0]["page_end"] == 10
        assert "Summary" in rows[0]["preview"]

    def test_upsert_is_idempotent(self, db):
        from db.client import upsert_page_range_summary_chunk, list_page_range_summaries
        for text in ("First version.", "Second version."):
            upsert_page_range_summary_chunk(
                db,
                doc_id="rdoc001",
                page_start=1,
                page_end=10,
                summary_text=text,
                embedding=[0.1] * 4096,
            )
        rows = list_page_range_summaries(db, "rdoc001")
        assert len(rows) == 1  # No duplicate
        assert "Second" in rows[0]["preview"]

    def test_upsert_multiple_ranges(self, db):
        from db.client import upsert_page_range_summary_chunk, list_page_range_summaries
        for start, end in [(1, 10), (11, 20), (21, 30)]:
            upsert_page_range_summary_chunk(
                db,
                doc_id="rdoc001",
                page_start=start,
                page_end=end,
                summary_text=f"Summary of pages {start}-{end}.",
                embedding=[0.1] * 4096,
            )
        rows = list_page_range_summaries(db, "rdoc001")
        assert len(rows) == 3
        assert [r["page_start"] for r in rows] == [1, 11, 21]

    # ── list_page_range_summaries ─────────────────────────────────────────────

    def test_list_all_returns_all_docs(self, db):
        from db.client import (
            upsert_document_record,
            upsert_page_range_summary_chunk,
            list_page_range_summaries,
        )
        upsert_document_record(
            db,
            doc_id="rdoc002",
            filename="other.pdf",
            source_path="/data/other.pdf",
            source_type="pdf_book",
            num_pages=5,
            metadata={},
        )
        for doc_id in ("rdoc001", "rdoc002"):
            upsert_page_range_summary_chunk(
                db,
                doc_id=doc_id,
                page_start=1,
                page_end=5,
                summary_text=f"Summary for {doc_id}.",
                embedding=[0.1] * 4096,
            )
        all_rows = list_page_range_summaries(db)  # no doc_id filter
        doc_ids = {r["doc_id"] for r in all_rows}
        assert "rdoc001" in doc_ids
        assert "rdoc002" in doc_ids

    def test_list_filtered_by_doc(self, db):
        from db.client import upsert_page_range_summary_chunk, list_page_range_summaries
        upsert_page_range_summary_chunk(
            db, doc_id="rdoc001", page_start=1, page_end=10,
            summary_text="For rdoc001.", embedding=[0.1] * 4096,
        )
        rows = list_page_range_summaries(db, "rdoc001")
        assert all(r["doc_id"] == "rdoc001" for r in rows)

    def test_list_empty_doc(self, db):
        from db.client import list_page_range_summaries
        rows = list_page_range_summaries(db, "nonexistent")
        assert rows == []

    # ── get_page_range_summary_text ───────────────────────────────────────────

    def test_get_text_returns_full_text(self, db):
        from db.client import upsert_page_range_summary_chunk, get_page_range_summary_text
        upsert_page_range_summary_chunk(
            db, doc_id="rdoc001", page_start=1, page_end=10,
            summary_text="Full summary text here.", embedding=[0.1] * 4096,
        )
        text = get_page_range_summary_text(db, "rdoc001", 1, 10)
        assert text == "Full summary text here."

    def test_get_text_missing_range_returns_none(self, db):
        from db.client import get_page_range_summary_text
        text = get_page_range_summary_text(db, "rdoc001", 99, 999)
        assert text is None

