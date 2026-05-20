"""
PostgreSQL persistence client for RAG pipeline artifacts.

Uses psycopg3 for all database access.  Pass connection strings in libpq DSN
format, e.g. ``postgresql://localhost/rag`` or
``postgresql://user:pass@host:5432/rag``.

This module is a backward-compatible re-exporter.  Implementation lives in:
  db.init          — connection helpers, schema init, migrations
  db.documents     — document and chunk CRUD
  db.collections   — collection CRUD
  db.chunk_questions — hypothetical question index CRUD
  db.conversations — conversation CRUD
"""

from db.init import _connect, init_db  # noqa: F401
from db.documents import (  # noqa: F401
	is_document_ingested,
	upsert_document,
	upsert_document_record,
	upsert_artifact,
	_fmt_vector,
	upsert_chunks_from_index,
	persist_pipeline_outputs,
	list_unassigned_documents,
	list_documents,
	get_doc_id_for_path,
	delete_document,
	get_docs_without_summary,
	fetch_representative_chunks,
	fetch_all_chunks_ordered,
	upsert_summary_chunk,
	get_document_file_hash,
	get_summary_text,
	fetch_chunks_in_page_range,
	upsert_page_range_summary_chunk,
	list_page_range_summaries,
	get_page_range_summary_text,
)
from db.collections import (  # noqa: F401
	create_collection,
	list_collections,
	get_collection,
	get_collection_scope,
	assign_to_collection,
	unassign_from_collection,
	delete_collection,
)
from db.chunk_questions import (  # noqa: F401
	upsert_chunk_questions,
	search_chunks_by_question_embedding,
	count_chunk_questions,
)
from db.conversations import (  # noqa: F401
	_CONV_SUMMARIZE_THRESHOLD,
	_CONV_KEEP_RECENT,
	create_conversation,
	get_conversation,
	list_conversations,
	add_conversation_message,
	set_conversation_title,
	compress_conversation,
	get_conversation_message_count,
	archive_conversation,
	delete_conversation,
	archive_stale_conversations,
)

