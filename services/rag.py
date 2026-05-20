"""RAG service facade.

All retrieval, ingest, and collection operations go through here.
server/ and main.py import from this module only — never from retrieval/,
pipeline/, or db/ directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

# Re-export the public API functions so callers only need `services.rag`
from api import rag_retrieve, llm_answer as _llm_answer  # noqa: F401

from db.client import (
    init_db,
    list_collections,
    list_documents,
    list_unassigned_documents,
    get_collection,
    create_collection,
    delete_collection,
    assign_to_collection,
    unassign_from_collection,
    delete_document,
    add_conversation_message,
    archive_conversation,
    archive_stale_conversations,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversation_message_count,
    compress_conversation,
    list_conversations,
    set_conversation_title,
    _CONV_SUMMARIZE_THRESHOLD,
    _CONV_KEEP_RECENT,
)

from utils.runtime_defaults import DEFAULT_DB_DSN


def retrieve(
    query: str,
    *,
    top_k: int = 5,
    collection_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    prior_intents: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Retrieve context chunks for a query.

    Thin wrapper around api.rag_retrieve — use this instead of importing
    api directly so server code stays decoupled from retrieval internals.
    """
    return rag_retrieve(
        query,
        top_k=top_k,
        collection=collection_id,
        doc_ids=doc_ids,
        prior_intents=prior_intents,
        **kwargs,
    )


def ingest_file(
    file_path: str,
    *,
    collection_id: Optional[str] = None,
    db_dsn: str = DEFAULT_DB_DSN,
) -> Dict[str, Any]:
    """Ingest a single document into the RAG corpus.

    Returns a summary dict: {doc_id, chunk_count, status}.
    """
    from pipeline.ingest_v3 import ingest_file as _ingest
    return _ingest(file_path, collection_id=collection_id, db_dsn=db_dsn)


def ingest_files(
    file_paths: List[str],
    *,
    collection_id: Optional[str] = None,
    db_dsn: str = DEFAULT_DB_DSN,
) -> List[Dict[str, Any]]:
    """Ingest multiple documents. Returns one summary dict per file."""
    return [ingest_file(p, collection_id=collection_id, db_dsn=db_dsn) for p in file_paths]
