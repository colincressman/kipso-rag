import json
import sys
from pathlib import Path

import psycopg
from psycopg.rows import dict_row
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.models import IngestedDocument, IngestedPage
from db.client import (
    init_db,
    persist_pipeline_outputs,
    upsert_document_record,
    upsert_chunks_from_index,
    is_document_ingested,
    create_collection,
    list_collections,
    get_collection,
    get_collection_scope,
    delete_collection,
    assign_to_collection,
    unassign_from_collection,
    upsert_chunk_questions,
    create_conversation,
    add_conversation_message,
    get_conversation,
    list_conversations,
    delete_document,
    get_doc_id_for_path,
)


def _doc() -> IngestedDocument:
    return IngestedDocument(
        doc_id="abc123",
        source_path="C:/spec.pdf",
        filename="spec.pdf",
        num_pages=1,
        metadata={"title": "Spec"},
        pages=[
            IngestedPage(
                page_num=0,
                width=100,
                height=100,
                raw_text="hello",
                blocks=[],
                tables=[],
                image_count=0,
            )
        ],
    )


@pytest.mark.requires_postgres
def test_db_persistence(pg_dsn: str, tmp_path: Path):
    index_path = tmp_path / "index.json"
    index_payload = {
        "items": [
            {
                "chunk_id": "abc123-c000000",
                "doc_id": "abc123",
                "section_id": "s1",
                "path_text": "A",
                "title": "T",
                "level": 2,
                "page_start": 1,
                "page_end": 1,
                "has_table": False,
                "token_count_est": 12,
                "text": "chunk text",
                "embedding": [0.1, 0.2, 0.3] + [0.0] * 4093,
            }
        ]
    }
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")

    init_db(pg_dsn)
    stats = persist_pipeline_outputs(
        pg_dsn,
        _doc(),
        extracted_path="data/extracted/spec.json",
        markdown_path="data/markdown/spec.md",
        structured_path="data/structured/spec.structured.json",
        chunks_path="data/chunks/spec.chunks.json",
        index_path=str(index_path),
    )
    assert stats["documents"] == 1
    assert stats["chunks"] == 1

    with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        art_count = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    assert doc_count == 1
    assert art_count == 5
    assert chunk_count == 1


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_chunk_payload(doc_id: str, chunk_id: str, text: str = "hello", emb: list | None = None) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "section_id": "s1",
        "path_text": "A > B",
        "title": "Title",
        "level": 1,
        "page_start": 1,
        "page_end": 1,
        "has_table": False,
        "token_count_est": 10,
        "text": text,
        "embedding": emb or ([0.1] + [0.0] * 4095),
    }


def _insert_doc_and_chunk(pg_dsn: str, doc_id: str, chunk_id: str, collection_id: str | None = None) -> None:
    upsert_document_record(
        pg_dsn,
        doc_id=doc_id,
        filename=f"{doc_id}.pdf",
        source_path=f"/tmp/{doc_id}.pdf",
        source_type="pdf_book",
    )
    item = _make_chunk_payload(doc_id, chunk_id)
    item["collection_id"] = collection_id
    upsert_chunks_from_index(pg_dsn, {"items": [item]})


# ── upsert_document_record ────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestUpsertDocumentRecord:

    def test_insert_creates_row(self, pg_dsn):
        upsert_document_record(
            pg_dsn,
            doc_id="doc001",
            filename="book.pdf",
            source_path="/data/book.pdf",
        )
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT doc_id, filename, source_type FROM documents WHERE doc_id = 'doc001'"
            ).fetchone()
        assert row is not None
        assert row["filename"] == "book.pdf"
        assert row["source_type"] == "pdf_book"

    def test_conflict_updates_filename(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="doc002", filename="v1.pdf", source_path="/v1.pdf")
        upsert_document_record(pg_dsn, doc_id="doc002", filename="v2.pdf", source_path="/v2.pdf")
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT filename FROM documents WHERE doc_id = 'doc002'"
            ).fetchone()
        assert row["filename"] == "v2.pdf"

    def test_custom_source_type_stored(self, pg_dsn):
        upsert_document_record(
            pg_dsn,
            doc_id="doc003",
            filename="wiki.md",
            source_path="/wiki.md",
            source_type="web_article",
        )
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT source_type FROM documents WHERE doc_id = 'doc003'"
            ).fetchone()
        assert row["source_type"] == "web_article"


# ── upsert_chunks_from_index ─────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestUpsertChunksFromIndex:

    def test_returns_chunk_count(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="d1", filename="a.pdf", source_path="/a.pdf")
        payload = {
            "items": [
                _make_chunk_payload("d1", "d1-c1"),
                _make_chunk_payload("d1", "d1-c2", text="world"),
            ]
        }
        count = upsert_chunks_from_index(pg_dsn, payload)
        assert count == 2

    def test_empty_payload_returns_zero(self, pg_dsn):
        assert upsert_chunks_from_index(pg_dsn, {"items": []}) == 0
        assert upsert_chunks_from_index(pg_dsn, {}) == 0

    def test_replace_doc_chunks_clears_old(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="d2", filename="b.pdf", source_path="/b.pdf")
        # First ingest: 3 chunks
        payload1 = {"items": [_make_chunk_payload("d2", f"d2-c{i}") for i in range(3)]}
        upsert_chunks_from_index(pg_dsn, payload1)

        # Re-ingest with replace_doc_chunks=True and only 1 chunk
        payload2 = {"items": [_make_chunk_payload("d2", "d2-new")]}
        upsert_chunks_from_index(pg_dsn, payload2, replace_doc_chunks=True)

        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            n = conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = 'd2'"
            ).fetchone()["n"]
        assert n == 1

    def test_conflict_updates_text(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="d3", filename="c.pdf", source_path="/c.pdf")
        upsert_chunks_from_index(pg_dsn, {"items": [_make_chunk_payload("d3", "d3-c1", text="original")]})
        upsert_chunks_from_index(pg_dsn, {"items": [_make_chunk_payload("d3", "d3-c1", text="updated")]})
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute("SELECT text FROM chunks WHERE chunk_id = 'd3-c1'").fetchone()
        assert row["text"] == "updated"


# ── is_document_ingested ──────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestIsDocumentIngested:

    def test_returns_false_when_not_ingested(self, pg_dsn):
        assert is_document_ingested(pg_dsn, "/some/path.pdf", "coll1") is False

    def test_returns_true_after_ingest(self, pg_dsn):
        _insert_doc_and_chunk(pg_dsn, "ingdoc1", "ingdoc1-c1", collection_id="coll1")
        assert is_document_ingested(pg_dsn, "/tmp/ingdoc1.pdf", "coll1") is True

    def test_wrong_collection_returns_false(self, pg_dsn):
        _insert_doc_and_chunk(pg_dsn, "ingdoc2", "ingdoc2-c1", collection_id="coll_a")
        assert is_document_ingested(pg_dsn, "/tmp/ingdoc2.pdf", "coll_b") is False


# ── Collection CRUD ───────────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestCollections:

    def test_create_and_list(self, pg_dsn):
        create_collection(pg_dsn, "col1", "My Collection", description="A test collection")
        cols = list_collections(pg_dsn)
        ids = [c["collection_id"] for c in cols]
        assert "col1" in ids

    def test_create_duplicate_raises(self, pg_dsn):
        create_collection(pg_dsn, "col_dup", "First")
        with pytest.raises(ValueError, match="already exists"):
            create_collection(pg_dsn, "col_dup", "Second")

    def test_create_with_invalid_parent_raises(self, pg_dsn):
        with pytest.raises(ValueError, match="Parent collection"):
            create_collection(pg_dsn, "col_child", "Child", parent_id="nonexistent")

    def test_create_requires_collection_id(self, pg_dsn):
        with pytest.raises(ValueError):
            create_collection(pg_dsn, "", "No ID")

    def test_get_collection_returns_none_for_missing(self, pg_dsn):
        assert get_collection(pg_dsn, "does_not_exist") is None

    def test_get_collection_structure(self, pg_dsn):
        create_collection(pg_dsn, "gc1", "Get Test")
        result = get_collection(pg_dsn, "gc1")
        assert result is not None
        assert result["collection_id"] == "gc1"
        assert result["name"] == "Get Test"
        assert "sub_collections" in result
        assert "documents" in result

    def test_get_collection_scope_includes_children(self, pg_dsn):
        create_collection(pg_dsn, "parent1", "Parent")
        create_collection(pg_dsn, "child1", "Child", parent_id="parent1")
        scope = get_collection_scope(pg_dsn, "parent1")
        assert "parent1" in scope
        assert "child1" in scope

    def test_get_collection_scope_no_children(self, pg_dsn):
        create_collection(pg_dsn, "solo1", "Solo")
        scope = get_collection_scope(pg_dsn, "solo1")
        assert scope == ["solo1"]

    def test_list_collections_chunk_counts(self, pg_dsn):
        create_collection(pg_dsn, "cnt1", "Count Test")
        upsert_document_record(pg_dsn, doc_id="cntdoc", filename="cnt.pdf", source_path="/cnt.pdf")
        item = _make_chunk_payload("cntdoc", "cntdoc-c1")
        item["collection_id"] = "cnt1"
        upsert_chunks_from_index(pg_dsn, {"items": [item]})
        cols = {c["collection_id"]: c for c in list_collections(pg_dsn)}
        assert cols["cnt1"]["chunk_count"] == 1
        assert cols["cnt1"]["doc_count"] == 1


# ── delete_collection ─────────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestDeleteCollection:

    def test_delete_clears_chunks_by_default(self, pg_dsn):
        create_collection(pg_dsn, "del1", "Delete Me")
        _insert_doc_and_chunk(pg_dsn, "deldoc1", "deldoc1-c1", collection_id="del1")
        cleared = delete_collection(pg_dsn, "del1")
        assert cleared == 1
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT collection_id FROM chunks WHERE chunk_id = 'deldoc1-c1'"
            ).fetchone()
        assert row["collection_id"] is None

    def test_delete_with_clear_chunks_false(self, pg_dsn):
        create_collection(pg_dsn, "del2", "Delete No Clear")
        _insert_doc_and_chunk(pg_dsn, "deldoc2", "deldoc2-c1", collection_id="del2")
        cleared = delete_collection(pg_dsn, "del2", clear_chunks=False)
        assert cleared == 0


# ── assign / unassign ─────────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestAssignUnassign:

    def test_assign_by_doc_ids(self, pg_dsn):
        create_collection(pg_dsn, "asgn1", "Assign Test")
        _insert_doc_and_chunk(pg_dsn, "asgdoc1", "asgdoc1-c1")
        n = assign_to_collection(pg_dsn, "asgn1", doc_ids=["asgdoc1"])
        assert n >= 1

    def test_unassign_clears_collection(self, pg_dsn):
        create_collection(pg_dsn, "unasgn1", "Unassign Test")
        _insert_doc_and_chunk(pg_dsn, "unadoc1", "unadoc1-c1", collection_id="unasgn1")
        cleared = unassign_from_collection(pg_dsn, ["unadoc1"])
        assert cleared >= 1
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT collection_id FROM chunks WHERE chunk_id = 'unadoc1-c1'"
            ).fetchone()
        assert row["collection_id"] is None

    def test_unassign_empty_list_returns_zero(self, pg_dsn):
        assert unassign_from_collection(pg_dsn, []) == 0

    def test_assign_requires_scope(self, pg_dsn):
        with pytest.raises(ValueError, match="doc_ids or source_type"):
            assign_to_collection(pg_dsn, "any_col")


# ── upsert_chunk_questions ────────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestUpsertChunkQuestions:

    def test_inserts_questions(self, pg_dsn):
        _insert_doc_and_chunk(pg_dsn, "qdoc1", "qdoc1-c1")
        questions = ["What is X?", "How does Y work?"]
        embeddings = [[0.1] * 4096, [0.2] * 4096]
        upsert_chunk_questions(pg_dsn, "qdoc1-c1", questions, embeddings)
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                "SELECT question FROM chunk_questions WHERE chunk_id = 'qdoc1-c1' ORDER BY question"
            ).fetchall()
        assert len(rows) == 2

    def test_replace_on_second_call(self, pg_dsn):
        _insert_doc_and_chunk(pg_dsn, "qdoc2", "qdoc2-c1")
        upsert_chunk_questions(pg_dsn, "qdoc2-c1", ["Q1", "Q2"], [[0.1] * 4096, [0.2] * 4096])
        upsert_chunk_questions(pg_dsn, "qdoc2-c1", ["Q_new"], [[0.3] * 4096])
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            rows = conn.execute(
                "SELECT question FROM chunk_questions WHERE chunk_id = 'qdoc2-c1'"
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["question"] == "Q_new"

    def test_mismatched_lengths_raises(self, pg_dsn):
        _insert_doc_and_chunk(pg_dsn, "qdoc3", "qdoc3-c1")
        with pytest.raises(ValueError, match="same length"):
            upsert_chunk_questions(pg_dsn, "qdoc3-c1", ["Q1", "Q2"], [[0.1] * 4096])

    def test_empty_questions_noop(self, pg_dsn):
        # Should not raise
        upsert_chunk_questions(pg_dsn, "nonexistent-chunk", [], [])


# ── Conversation persistence ──────────────────────────────────────────────────

@pytest.mark.requires_postgres
class TestConversations:

    def test_create_and_get(self, pg_dsn):
        cid = create_conversation(pg_dsn, title="Test Conv")
        result = get_conversation(pg_dsn, cid)
        assert result is not None
        assert result["title"] == "Test Conv"
        assert result["messages"] == []

    def test_get_missing_returns_none(self, pg_dsn):
        assert get_conversation(pg_dsn, "nonexistent-conv-id") is None

    def test_add_message_and_retrieve(self, pg_dsn):
        cid = create_conversation(pg_dsn, title="Msg Test")
        mid = add_conversation_message(pg_dsn, cid, "user", "Hello?")
        conv = get_conversation(pg_dsn, cid)
        assert len(conv["messages"]) == 1
        msg = conv["messages"][0]
        assert msg["role"] == "user"
        assert msg["content"] == "Hello?"

    def test_messages_ordered_by_sequence(self, pg_dsn):
        cid = create_conversation(pg_dsn, title="Order Test")
        add_conversation_message(pg_dsn, cid, "user", "First")
        add_conversation_message(pg_dsn, cid, "assistant", "Second")
        add_conversation_message(pg_dsn, cid, "user", "Third")
        conv = get_conversation(pg_dsn, cid)
        contents = [m["content"] for m in conv["messages"]]
        assert contents == ["First", "Second", "Third"]

    def test_list_conversations(self, pg_dsn):
        create_conversation(pg_dsn, title="Conv A")
        create_conversation(pg_dsn, title="Conv B")
        convs = list_conversations(pg_dsn)
        titles = [c["title"] for c in convs]
        assert "Conv A" in titles
        assert "Conv B" in titles


# ── delete_document / get_doc_id_for_path ─────────────────────────────────────

@pytest.mark.requires_postgres
class TestDeleteDocument:

    def test_delete_removes_row(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="deldoc99", filename="del99.pdf", source_path="/del99.pdf")
        delete_document(pg_dsn, "deldoc99")
        with psycopg.connect(pg_dsn, row_factory=dict_row) as conn:
            row = conn.execute(
                "SELECT doc_id FROM documents WHERE doc_id = 'deldoc99'"
            ).fetchone()
        assert row is None

    def test_get_doc_id_for_path(self, pg_dsn):
        upsert_document_record(pg_dsn, doc_id="pathtest1", filename="path.pdf", source_path="/special/path.pdf")
        result = get_doc_id_for_path(pg_dsn, "/special/path.pdf")
        assert result == "pathtest1"

    def test_get_doc_id_for_missing_path_returns_none(self, pg_dsn):
        assert get_doc_id_for_path(pg_dsn, "/nonexistent/file.pdf") is None
