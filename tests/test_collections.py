"""Tests for the collections feature and intent-gate HyDE."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from db.client import (
    assign_to_collection,
    create_collection,
    delete_collection,
    get_collection,
    get_collection_scope,
    init_db,
    list_collections,
    list_unassigned_documents,
    upsert_chunks_from_index,
    upsert_document_record,
    unassign_from_collection,
)
from pipeline.ingest_multisource import SourceInput, ingest_text_sources
from retrieval.query import RetrievalFilters, _HYDE_SKIP_INTENTS, retrieve

_DIMS = 4096


def _e(*v):
    """Build a 4096-dim embedding vector with the given leading values."""
    return list(v) + [0.0] * (_DIMS - len(v))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_db(db_path: str) -> None:
    """Seed a test DB with two docs in different collections."""
    init_db(str(db_path))
    upsert_document_record(
        str(db_path),
        doc_id="docA",
        filename="alpha.pdf",
        source_path="/data/alpha.pdf",
        source_type="pdf_book",
    )
    upsert_document_record(
        str(db_path),
        doc_id="docB",
        filename="beta.md",
        source_path="/data/beta.md",
        source_type="notes",
    )
    upsert_chunks_from_index(str(db_path), {
        "items": [
            {
                "chunk_id": "docA-c000000", "doc_id": "docA",
                "text": "Neural networks learn via backpropagation.",
                "embedding": _e(1.0),
                "collection_id": "ml-books",
                "source_type": "pdf_book",
                "structural_role": "body",
                "token_count_est": 10,
            },
            {
                "chunk_id": "docA-c000001", "doc_id": "docA",
                "text": "Gradient descent minimises the loss function.",
                "embedding": _e(0.9, 0.1),
                "collection_id": "ml-books",
                "source_type": "pdf_book",
                "structural_role": "body",
                "token_count_est": 10,
            },
            {
                "chunk_id": "docB-c000000", "doc_id": "docB",
                "text": "CS7646 lecture notes on reinforcement learning.",
                "embedding": _e(0.0, 0.0, 1.0),
                "collection_id": "cs7646",
                "source_type": "notes",
                "structural_role": "body",
                "token_count_est": 10,
            },
        ]
    })


# ---------------------------------------------------------------------------
# DB collection management
# ---------------------------------------------------------------------------

@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_list_collections_returns_created(pg_dsn: str):
    db = pg_dsn
    assert list_collections(db) == []

    create_collection(db, "cs7646", "Machine Learning for Trading")
    create_collection(db, "ml-books", "ML Textbooks", description="Deep Learning etc.")

    cols = list_collections(db)
    assert len(cols) == 2
    ids = {c["collection_id"] for c in cols}
    assert ids == {"cs7646", "ml-books"}


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_create_collection_duplicate_raises(pg_dsn: str):
    db = pg_dsn
    create_collection(db, "cs7646", "ML for Trading")
    with pytest.raises(ValueError, match="already exists"):
        create_collection(db, "cs7646", "Duplicate")


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_returns_none_for_missing(pg_dsn: str):
    assert get_collection(pg_dsn, "nonexistent") is None


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_returns_documents(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)
    create_collection(str(db), "ml-books", "ML Books")

    info = get_collection(str(db), "ml-books")
    assert info is not None
    assert info["collection_id"] == "ml-books"
    assert len(info["documents"]) == 1
    assert info["documents"][0]["filename"] == "alpha.pdf"
    assert info["documents"][0]["chunk_count"] == 2


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_delete_collection_removes_row_and_clears_chunks(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)
    create_collection(str(db), "ml-books", "ML Books")

    cleared = delete_collection(str(db), "ml-books", clear_chunks=True)
    assert cleared == 2  # two chunks from docA

    # Collection row gone
    assert get_collection(str(db), "ml-books") is None

    # Chunks still exist, just no collection tag
    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(db, row_factory=dict_row) as conn:
        rows = conn.execute(
            "SELECT collection_id FROM chunks WHERE doc_id = 'docA'"
        ).fetchall()
    assert all(r["collection_id"] is None for r in rows)


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_assign_to_collection_by_doc_ids(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)
    create_collection(str(db), "new-col", "New Collection")

    count = assign_to_collection(str(db), "new-col", doc_ids=["docA"])
    assert count == 2  # docA has 2 chunks


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_assign_to_collection_by_source_type(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)
    create_collection(str(db), "all-notes", "All Notes")

    count = assign_to_collection(str(db), "all-notes", source_type="notes")
    assert count == 1  # docB has 1 chunk


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_unassign_from_collection(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)

    cleared = unassign_from_collection(str(db), ["docA"])
    assert cleared == 2


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_list_unassigned_documents(pg_dsn: str):
    db = pg_dsn
    # Seed with no collection_id on some chunks
    init_db(str(db))
    upsert_document_record(str(db), doc_id="docX", filename="x.pdf",
                            source_path="/x.pdf", source_type="pdf_book")
    upsert_chunks_from_index(str(db), {"items": [
        {"chunk_id": "docX-c000000", "doc_id": "docX",
         "text": "some text", "embedding": _e(1.0),
         "source_type": "pdf_book", "structural_role": "body", "token_count_est": 5}
    ]})

    docs = list_unassigned_documents(str(db))
    assert any(d["doc_id"] == "docX" for d in docs)


# ---------------------------------------------------------------------------
# Collection filter in retrieval
# ---------------------------------------------------------------------------

@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_retrieve_filters_to_collection(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)

    # Filter to ml-books — should only return docA chunks
    result = retrieve(
        "neural network training",
        db_dsn=str(db),
        top_k=5,
        filters=RetrievalFilters(collection_id="ml-books"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    assert result.hits
    assert all(
        h.metadata.get("collection_id") == "ml-books" for h in result.hits
    )


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_retrieve_filters_to_cs7646(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)

    result = retrieve(
        "reinforcement learning notes",
        db_dsn=str(db),
        top_k=5,
        filters=RetrievalFilters(collection_id="cs7646"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    assert result.hits
    assert all(
        h.metadata.get("collection_id") == "cs7646" for h in result.hits
    )
    assert all(h.doc_id == "docB" for h in result.hits)


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_retrieve_no_filter_returns_all(pg_dsn: str):
    db = pg_dsn
    _seed_db(db)

    result = retrieve(
        "learning",
        db_dsn=str(db),
        top_k=10,
        filters=RetrievalFilters(),
        embed_backend="_test",
        embed_dimension=4096,
    )
    doc_ids = {h.doc_id for h in result.hits}
    assert "docA" in doc_ids
    assert "docB" in doc_ids


# ---------------------------------------------------------------------------
# collection_id propagation through ingest_text_sources
# ---------------------------------------------------------------------------

@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_ingest_text_sources_collection_id_per_source(pg_dsn: str, tmp_path):
    db = pg_dsn
    sources = [
        SourceInput(
            title="Lecture 1",
            text="Q-learning is a model-free reinforcement learning algorithm.",
            source_path=str(tmp_path / "lec1.md"),
            source_type="notes",
            metadata={},
            collection_id="cs7646",
        ),
        SourceInput(
            title="Lecture 2",
            text="Policy gradient methods directly optimise the policy.",
            source_path=str(tmp_path / "lec2.md"),
            source_type="notes",
            metadata={},
            collection_id="cs7646",
        ),
    ]
    ingest_text_sources(sources, db_dsn=db, embed_backend="_test", embed_dimension=4096)

    result = retrieve(
        "reinforcement learning",
        db_dsn=db,
        top_k=5,
        filters=RetrievalFilters(collection_id="cs7646"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    assert result.hits
    assert all(h.metadata.get("collection_id") == "cs7646" for h in result.hits)


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_ingest_text_sources_collection_id_kwarg(pg_dsn: str, tmp_path):
    """collection_id kwarg on ingest_text_sources applies to all sources."""
    db = pg_dsn
    sources = [
        SourceInput(
            title="Note A",
            text="Backtesting evaluates strategy performance on historical data.",
            source_path=str(tmp_path / "a.md"),
            source_type="notes",
            metadata={},
        ),
    ]
    ingest_text_sources(
        sources, db_dsn=db, embed_backend="_test", embed_dimension=4096,
        collection_id="trading-notes",
    )

    result = retrieve(
        "backtesting",
        db_dsn=db,
        top_k=3,
        filters=RetrievalFilters(collection_id="trading-notes"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    assert result.hits


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_ingest_text_sources_no_collection_id(pg_dsn: str, tmp_path):
    """Without collection_id, chunks should have collection_id = NULL."""
    db = pg_dsn
    sources = [
        SourceInput(
            title="Orphan",
            text="Some unassigned content.",
            source_path=str(tmp_path / "orphan.md"),
            source_type="notes",
            metadata={},
        ),
    ]
    ingest_text_sources(sources, db_dsn=db, embed_backend="_test", embed_dimension=4096)

    import psycopg
    from psycopg.rows import dict_row
    with psycopg.connect(db, row_factory=dict_row) as conn:
        rows = conn.execute("SELECT collection_id FROM chunks").fetchall()
    assert all(r["collection_id"] is None for r in rows)


# ---------------------------------------------------------------------------
# Intent-gate HyDE
# ---------------------------------------------------------------------------

def test_hyde_skip_intents_contains_metadata_and_formula():
    assert "metadata_lookup" in _HYDE_SKIP_INTENTS
    assert "formula_lookup" in _HYDE_SKIP_INTENTS


def test_hyde_skip_intents_does_not_contain_exploratory():
    """Exploratory and fact queries should NOT skip HyDE."""
    assert "exploratory" not in _HYDE_SKIP_INTENTS
    assert "fact_lookup" not in _HYDE_SKIP_INTENTS
    assert "comparison" not in _HYDE_SKIP_INTENTS


# ---------------------------------------------------------------------------
# Hierarchical collections (sub-collections / parent-child)
# ---------------------------------------------------------------------------

@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_hierarchical_collections_creation(pg_dsn: str):
    db = pg_dsn
    create_collection(db, "CS7646", "ML for Trading")
    create_collection(db, "CS7646/notes", "Lecture Notes", parent_id="CS7646")
    create_collection(db, "CS7646/books", "Textbooks", parent_id="CS7646")

    cols = list_collections(db)
    assert len(cols) == 3
    parent = next(c for c in cols if c["collection_id"] == "CS7646")
    assert parent["parent_id"] is None
    notes = next(c for c in cols if c["collection_id"] == "CS7646/notes")
    assert notes["parent_id"] == "CS7646"
    books = next(c for c in cols if c["collection_id"] == "CS7646/books")
    assert books["parent_id"] == "CS7646"


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_hierarchical_parent_not_found_raises(pg_dsn: str):
    db = pg_dsn
    with pytest.raises(ValueError, match="Parent collection.*does not exist"):
        create_collection(db, "CS7646/notes", "Lecture Notes", parent_id="CS7646")


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_includes_sub_collections(pg_dsn: str):
    db = pg_dsn
    create_collection(db, "CS7646", "ML for Trading")
    create_collection(db, "CS7646/notes", "Lecture Notes", parent_id="CS7646")
    create_collection(db, "CS7646/books", "Textbooks", parent_id="CS7646")

    info = get_collection(db, "CS7646")
    assert info is not None
    assert info["parent_id"] is None
    subs = {s["collection_id"] for s in info["sub_collections"]}
    assert subs == {"CS7646/notes", "CS7646/books"}


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_scope_returns_all_descendants(pg_dsn: str):
    db = pg_dsn
    create_collection(db, "CS7646", "ML for Trading")
    create_collection(db, "CS7646/notes", "Notes", parent_id="CS7646")
    create_collection(db, "CS7646/books", "Books", parent_id="CS7646")

    scope = get_collection_scope(db, "CS7646")
    assert set(scope) == {"CS7646", "CS7646/notes", "CS7646/books"}


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_scope_leaf_returns_self(pg_dsn: str):
    db = pg_dsn
    create_collection(db, "CS7646", "ML for Trading")
    create_collection(db, "CS7646/notes", "Notes", parent_id="CS7646")

    scope = get_collection_scope(db, "CS7646/notes")
    assert scope == ["CS7646/notes"]


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_get_collection_scope_unknown_id_returns_self(pg_dsn: str):
    """Unknown collection ID falls back to [collection_id] for legacy support."""
    db = pg_dsn
    scope = get_collection_scope(db, "some-unregistered-id")
    assert scope == ["some-unregistered-id"]


def _seed_hierarchical(db_path: Path) -> None:
    """Seed: parent=CS7646, children=CS7646/notes + CS7646/books."""
    init_db(str(db_path))
    # Create collections
    create_collection(str(db_path), "CS7646", "ML for Trading")
    create_collection(str(db_path), "CS7646/notes", "Lecture Notes", parent_id="CS7646")
    create_collection(str(db_path), "CS7646/books", "Textbooks", parent_id="CS7646")
    # docs
    upsert_document_record(str(db_path), doc_id="note1", filename="lec1.md",
                            source_path="/lec1.md", source_type="notes")
    upsert_document_record(str(db_path), doc_id="book1", filename="sutton.pdf",
                            source_path="/sutton.pdf", source_type="pdf_book")
    # chunks
    upsert_chunks_from_index(str(db_path), {"items": [
        {
            "chunk_id": "note1-c000000", "doc_id": "note1",
            "text": "Q-learning updates the action-value function iteratively.",
            "embedding": _e(0.0, 1.0),
            "collection_id": "CS7646/notes",
            "source_type": "notes", "structural_role": "body", "token_count_est": 10,
        },
        {
            "chunk_id": "book1-c000000", "doc_id": "book1",
            "text": "Reinforcement learning: an introduction by Sutton and Barto.",
            "embedding": _e(0.0, 0.8, 0.2),
            "collection_id": "CS7646/books",
            "source_type": "pdf_book", "structural_role": "body", "token_count_est": 10,
        },
    ]})


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_hierarchical_retrieve_parent_returns_both_sub_collections(pg_dsn: str):
    """Querying parent CS7646 should return hits from both sub-collections."""
    db = pg_dsn
    _seed_hierarchical(db)

    result = retrieve(
        "reinforcement learning",
        db_dsn=str(db),
        top_k=10,
        filters=RetrievalFilters(collection_id="CS7646"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    collection_ids = {h.metadata.get("collection_id") for h in result.hits}
    # Both sub-collections should be represented
    assert "CS7646/notes" in collection_ids
    assert "CS7646/books" in collection_ids


@pytest.mark.requires_postgres
@pytest.mark.requires_postgres
def test_hierarchical_retrieve_child_excludes_sibling(pg_dsn: str):
    """Querying CS7646/notes should NOT return CS7646/books chunks."""
    db = pg_dsn
    _seed_hierarchical(db)

    result = retrieve(
        "reinforcement learning",
        db_dsn=str(db),
        top_k=10,
        filters=RetrievalFilters(collection_id="CS7646/notes"),
        embed_backend="_test",
        embed_dimension=4096,
    )
    assert result.hits
    assert all(h.metadata.get("collection_id") == "CS7646/notes" for h in result.hits)
