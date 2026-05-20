"""Document and chunk CRUD operations."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from db.init import _connect, init_db


def is_document_ingested(db_dsn: str, source_path: str, collection_id: str) -> bool:
	"""Return True if *source_path* already has chunks in *collection_id*."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		filename = Path(source_path).name
		row = conn.execute(
			"""
			SELECT d.doc_id
			FROM documents d
			JOIN chunks c ON c.doc_id = d.doc_id
			WHERE d.filename = %s AND c.collection_id = %s
			LIMIT 1
			""",
			(filename, collection_id),
		).fetchone()
		return row is not None
	except Exception:
		return False
	finally:
		conn.close()


def upsert_document(db_dsn: str, document: Any) -> None:
	"""Store/replace ingest document metadata row."""
	init_db(db_dsn)
	payload = asdict(document)
	metadata = payload.get("metadata", {})
	source_type = str(metadata.get("source_type", "pdf_book"))

	upsert_document_record(
		db_dsn,
		doc_id=str(payload["doc_id"]),
		filename=str(payload["filename"]),
		source_path=str(payload["source_path"]),
		source_type=source_type,
		num_pages=int(payload["num_pages"]),
		metadata=metadata,
		ingested_at=str(payload["ingested_at"]),
	)


def upsert_document_record(
	db_dsn: str,
	*,
	doc_id: str,
	filename: str,
	source_path: str,
	source_type: str = "pdf_book",
	num_pages: int = 1,
	metadata: Optional[Dict[str, Any]] = None,
	ingested_at: Optional[str] = None,
	file_hash: Optional[str] = None,
) -> None:
	"""Store/replace a document row without requiring an IngestedDocument object."""
	init_db(db_dsn)
	payload_metadata = metadata or {}

	timestamp = ingested_at or datetime.now(timezone.utc).isoformat()

	conn = _connect(db_dsn)
	try:
		conn.execute(
			"""
			INSERT INTO documents (doc_id, filename, source_path, source_type, num_pages, metadata_json, ingested_at, file_hash)
			VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
			ON CONFLICT(doc_id) DO UPDATE SET
				filename=excluded.filename,
				source_path=excluded.source_path,
				source_type=excluded.source_type,
				num_pages=excluded.num_pages,
				metadata_json=excluded.metadata_json,
				ingested_at=excluded.ingested_at,
				file_hash=COALESCE(excluded.file_hash, documents.file_hash)
			""",
			(
				doc_id,
				filename,
				source_path,
				source_type,
				int(num_pages),
				json.dumps(payload_metadata, ensure_ascii=False),
				timestamp,
				file_hash,
			),
		)
		conn.commit()
	finally:
		conn.close()


def upsert_artifact(db_dsn: str, doc_id: str, artifact_type: str, artifact_path: str) -> None:
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"""
			INSERT INTO artifacts (doc_id, artifact_type, artifact_path)
			VALUES (%s, %s, %s)
			ON CONFLICT(doc_id, artifact_type) DO UPDATE SET
				artifact_path=excluded.artifact_path,
				created_at=NOW()
			""",
			(doc_id, artifact_type, artifact_path),
		)
		conn.commit()
	finally:
		conn.close()


def _fmt_vector(vec: list | None) -> str | None:
	"""Format a Python float list as a pgvector literal string, or None."""
	if not vec:
		return None
	return "[" + ",".join(str(float(v)) for v in vec) + "]"


def upsert_chunks_from_index(
	db_dsn: str,
	index_payload: Dict[str, Any],
	*,
	replace_doc_chunks: bool | None = None,
) -> int:
	"""
	Store chunk rows, including embedding vectors from index payload.

	Returns inserted/updated row count.
	"""
	items = index_payload.get("items", [])
	if not isinstance(items, list) or not items:
		return 0

	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if replace_doc_chunks is None:
			replace_doc_chunks = bool(index_payload.get("full_document", False))

		doc_ids = sorted({str(it.get("doc_id")) for it in items if it.get("doc_id")})
		doc_cache: Dict[str, Dict[str, Any]] = {}
		if doc_ids:
			placeholders = ",".join(["%s"] * len(doc_ids))
			rows = conn.execute(
				f"SELECT doc_id, filename, source_path, source_type, metadata_json FROM documents WHERE doc_id IN ({placeholders})",
				doc_ids,
			).fetchall()
			for row in rows:
				meta: Dict[str, Any] = {}
				try:
					meta = json.loads(row["metadata_json"] or "{}")
				except json.JSONDecodeError:
					meta = {}
				doc_cache[str(row["doc_id"])] = {
					"filename": row["filename"],
					"source_path": row["source_path"],
					"source_type": str(row["source_type"] or "pdf_book"),
					"metadata": meta,
				}

		if replace_doc_chunks:
			doc_ids = sorted({str(it.get("doc_id")) for it in items if it.get("doc_id")})
			for doc_id in doc_ids:
				conn.execute("DELETE FROM chunks WHERE doc_id = %s", (doc_id,))

		for item in items:
			doc_id = item.get("doc_id")
			doc_info = doc_cache.get(str(doc_id), {}) if doc_id else {}
			doc_meta: Dict[str, Any] = dict(doc_info.get("metadata") or {})
			doc_title = doc_meta.get("title")
			doc_path = doc_info.get("source_path")
			doc_filename = doc_info.get("filename")
			doc_source_type = str(doc_info.get("source_type") or "pdf_book")

			collection_id = (
				item.get("collection_id")
				or doc_meta.get("collection_id")
				or None
			)
			source_name = (
				item.get("source_name")
				or doc_meta.get("source_name")
				or doc_filename
			)
			document_title = (
				item.get("document_title")
				or doc_title
				or doc_meta.get("document_title")
				or source_name
			)
			document_path = (
				item.get("document_path")
				or doc_path
			)

			vec = _fmt_vector(item.get("embedding"))
			conn.execute(
				"""
				INSERT INTO chunks (
					chunk_id, doc_id, collection_id, source_name, document_title, document_path,
					section_id, path_text, title, level,
					page_start, page_end, has_table, token_count_est,
					source_type, structural_role, text, embedding
				) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
				ON CONFLICT(chunk_id) DO UPDATE SET
					doc_id=excluded.doc_id,
					collection_id=excluded.collection_id,
					source_name=excluded.source_name,
					document_title=excluded.document_title,
					document_path=excluded.document_path,
					section_id=excluded.section_id,
					path_text=excluded.path_text,
					title=excluded.title,
					level=excluded.level,
					page_start=excluded.page_start,
					page_end=excluded.page_end,
					has_table=excluded.has_table,
					token_count_est=excluded.token_count_est,
					source_type=excluded.source_type,
					structural_role=excluded.structural_role,
					text=excluded.text,
					embedding=excluded.embedding
				""",
				(
					item.get("chunk_id"),
					doc_id,
					collection_id,
					source_name,
					document_title,
					document_path,
					item.get("section_id"),
					item.get("path_text"),
					item.get("title"),
					item.get("level"),
					item.get("page_start"),
					item.get("page_end"),
					1 if item.get("has_table") else 0,
					item.get("token_count_est"),
					item.get("source_type", doc_source_type),
					item.get("structural_role", "body"),
					item.get("text", ""),
					vec,
				),
			)
		conn.commit()
		return len(items)
	finally:
		conn.close()


def persist_pipeline_outputs(
	db_dsn: str,
	document: Any,
	*,
	extracted_path: Optional[str] = None,
	markdown_path: Optional[str] = None,
	structured_path: Optional[str] = None,
	chunks_path: Optional[str] = None,
	index_path: Optional[str] = None,
) -> Dict[str, int]:
	"""Persist ingest document + artifact pointers + embedded chunks."""
	upsert_document(db_dsn, document)
	doc_id = getattr(document, "doc_id")

	if extracted_path:
		upsert_artifact(db_dsn, doc_id, "extracted", extracted_path)
	if markdown_path:
		upsert_artifact(db_dsn, doc_id, "markdown", markdown_path)
	if structured_path:
		upsert_artifact(db_dsn, doc_id, "structured", structured_path)
	if chunks_path:
		upsert_artifact(db_dsn, doc_id, "chunks", chunks_path)
	if index_path:
		upsert_artifact(db_dsn, doc_id, "index", index_path)

	chunk_rows = 0
	if index_path:
		payload = json.loads(Path(index_path).read_text(encoding="utf-8"))
		chunk_rows = upsert_chunks_from_index(db_dsn, payload, replace_doc_chunks=True)

	return {"documents": 1, "chunks": chunk_rows}


def list_unassigned_documents(db_dsn: str) -> list:
	"""Return documents where ALL chunks have no collection assigned."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT d.doc_id,
			       COALESCE(MIN(c.document_title), d.filename) AS title,
			       COALESCE(MIN(c.document_title), d.filename) AS document_title,
			       d.filename,
			       d.source_path,
			       d.source_type,
			       COUNT(c.chunk_id) AS chunk_count
			FROM documents d
			JOIN chunks c ON c.doc_id = d.doc_id
			WHERE NOT EXISTS (
			    SELECT 1 FROM chunks cx
			    WHERE cx.doc_id = d.doc_id
			      AND cx.collection_id IS NOT NULL
			)
			GROUP BY d.doc_id, d.filename, d.source_path, d.source_type
			HAVING COUNT(c.chunk_id) > 0
			ORDER BY d.filename
			"""
		).fetchall()
		return [
			{
				"doc_id": r["doc_id"],
				"title": r["title"],
				"document_title": r["document_title"],
				"filename": r["filename"],
				"source_path": r["source_path"],
				"source_type": r["source_type"],
				"chunk_count": r["chunk_count"],
			}
			for r in rows
		]
	finally:
		conn.close()


def list_documents(db_dsn: str, collection_id: Optional[str] = None) -> list:
	"""Return all documents with chunk counts. Optionally scoped to a collection."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if collection_id:
			rows = conn.execute(
				"""
				SELECT d.doc_id,
				       COALESCE(MIN(c.document_title), d.filename) AS title,
				       COALESCE(MIN(c.document_title), d.filename) AS document_title,
				       d.filename,
				       d.source_path,
				       d.source_type,
				       COUNT(c.chunk_id) AS chunk_count
				FROM documents d
				JOIN chunks c ON c.doc_id = d.doc_id
				WHERE c.collection_id = %s
				GROUP BY d.doc_id, d.filename, d.source_path, d.source_type
				ORDER BY d.source_type, title
				""",
				(collection_id,),
			).fetchall()
		else:
			rows = conn.execute(
				"""
				SELECT d.doc_id,
				       COALESCE(MIN(c.document_title), d.filename) AS title,
				       COALESCE(MIN(c.document_title), d.filename) AS document_title,
				       d.filename,
				       d.source_path,
				       d.source_type,
				       COUNT(c.chunk_id) AS chunk_count
				FROM documents d
				JOIN chunks c ON c.doc_id = d.doc_id
				GROUP BY d.doc_id, d.filename, d.source_path, d.source_type
				ORDER BY d.source_type, title
				"""
			).fetchall()
		return [
			{
				"doc_id": r["doc_id"],
				"title": r["title"],
				"document_title": r["document_title"],
				"filename": r["filename"],
				"source_path": r["source_path"],
				"source_type": r["source_type"],
				"chunk_count": r["chunk_count"],
			}
			for r in rows
		]
	finally:
		conn.close()


def get_doc_id_for_path(db_dsn: str, source_path: str) -> Optional[str]:
	"""Return the doc_id stored for *source_path*, or None if not found.

	Used by the ingest pipeline to detect whether a file has changed: if
	the stored doc_id (SHA-256 of old content) differs from the current
	file hash, the old document is stale and should be retired before the
	new content is ingested.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT doc_id FROM documents WHERE source_path = %s LIMIT 1",
			(source_path,),
		).fetchone()
		return str(row["doc_id"]) if row else None
	finally:
		conn.close()


def delete_document(db_dsn: str, doc_id: str) -> None:
	"""Delete a document and all its chunks (ON DELETE CASCADE) from the DB."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute("DELETE FROM documents WHERE doc_id = %s", (doc_id,))
		conn.commit()
	finally:
		conn.close()


def get_docs_without_summary(db_dsn: str) -> list:
	"""Return (doc_id, filename, source_type) for documents lacking a summary chunk."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT d.doc_id, d.filename, d.source_type,
			       COALESCE(MIN(c.collection_id), '') AS collection_id,
			       COALESCE(MIN(c.document_title), d.filename) AS document_title,
			       COALESCE(MIN(c.source_name), d.filename) AS source_name,
			       COALESCE(MIN(c.document_path), d.filename) AS document_path
			FROM documents d
			JOIN chunks c ON c.doc_id = d.doc_id
			WHERE NOT EXISTS (
				SELECT 1 FROM chunks s
				WHERE s.doc_id = d.doc_id
				AND s.structural_role = 'document_summary'
			)
			GROUP BY d.doc_id, d.filename, d.source_type
			ORDER BY d.filename
			"""
		).fetchall()
		return [
			{
				"doc_id": r["doc_id"],
				"filename": r["filename"],
				"source_type": r["source_type"],
				"collection_id": r["collection_id"],
				"document_title": r["document_title"],
				"source_name": r["source_name"],
				"document_path": r["document_path"],
			}
			for r in rows
		]
	finally:
		conn.close()


def fetch_representative_chunks(db_dsn: str, doc_id: str, max_chars: int = 80000) -> list:
	"""Fetch chunks for single-pass summary generation (small docs only).

	Priority order:
	  1. Structural front-matter (toc / abstract / preface / introduction / summary)
	  2. Body chunks by page order
	Caps total text at max_chars.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT chunk_id, text, structural_role, path_text, page_start
			FROM chunks
			WHERE doc_id = %s
			  AND structural_role != 'document_summary'
			ORDER BY
			  CASE structural_role
			    WHEN 'toc'          THEN 0
			    WHEN 'abstract'     THEN 1
			    WHEN 'preface'      THEN 2
			    WHEN 'introduction' THEN 3
			    WHEN 'summary'      THEN 4
			    ELSE 10
			  END,
			  COALESCE(page_start, 9999)
			""",
			(doc_id,),
		).fetchall()

		selected: list = []
		total = 0
		for r in rows:
			text = (r["text"] or "").strip()
			if not text:
				continue
			if total + len(text) > max_chars and selected:
				break
			selected.append({"chunk_id": r["chunk_id"], "text": text,
			                  "structural_role": r["structural_role"],
			                  "path_text": r["path_text"]})
			total += len(text)
		return selected
	finally:
		conn.close()


def fetch_all_chunks_ordered(db_dsn: str, doc_id: str) -> list:
	"""Return ALL non-summary chunks for a document sorted by page then position.

	Used by the map-reduce summarizer to ensure full-book coverage.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT chunk_id, text, structural_role, path_text,
			       COALESCE(page_start, 0) AS page_start
			FROM chunks
			WHERE doc_id = %s
			  AND structural_role != 'document_summary'
			ORDER BY
			  CASE structural_role
			    WHEN 'toc'          THEN -4
			    WHEN 'abstract'     THEN -3
			    WHEN 'preface'      THEN -2
			    WHEN 'introduction' THEN -1
			    ELSE COALESCE(page_start, 9999)
			  END
			""",
			(doc_id,),
		).fetchall()
		return [
			{
				"chunk_id": r["chunk_id"],
				"text": (r["text"] or "").strip(),
				"structural_role": r["structural_role"],
				"path_text": r["path_text"],
				"page_start": r["page_start"],
			}
			for r in rows
			if (r["text"] or "").strip()
		]
	finally:
		conn.close()


def upsert_summary_chunk(
	db_dsn: str,
	*,
	doc_id: str,
	summary_text: str,
	embedding: list,
	collection_id: str = "",
	document_title: str = "",
	source_name: str = "",
	document_path: str = "",
	source_type: str = "pdf_book",
) -> None:
	"""Insert or replace the document_summary chunk for a document."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		chunk_id = f"{doc_id}_summary"
		vec = _fmt_vector(embedding)
		conn.execute(
			"""
			INSERT INTO chunks (
				chunk_id, doc_id, collection_id, source_name, document_title, document_path,
				section_id, path_text, title, level,
				page_start, page_end, has_table, token_count_est,
				source_type, structural_role, text, embedding
			) VALUES (%s, %s, %s, %s, %s, %s, NULL, 'Summary', 'Document Summary', 0,
			          NULL, NULL, 0, %s,
			          %s, 'document_summary', %s, %s::vector)
			ON CONFLICT(chunk_id) DO UPDATE SET
				text=excluded.text,
				embedding=excluded.embedding,
				document_title=excluded.document_title,
				source_name=excluded.source_name
			""",
			(
				chunk_id,
				doc_id,
				collection_id or None,
				source_name or document_title,
				document_title,
				document_path,
				len(summary_text.split()),
				source_type,
				summary_text,
				vec,
			),
		)
		conn.commit()
	finally:
		conn.close()


def get_document_file_hash(db_dsn: str, doc_id: str) -> Optional[str]:
	"""Return the stored SHA-256 file_hash for *doc_id*, or None if not present."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT file_hash FROM documents WHERE doc_id = %s LIMIT 1",
			(doc_id,),
		).fetchone()
		return str(row["file_hash"]) if row and row["file_hash"] else None
	finally:
		conn.close()


def get_summary_text(db_dsn: str, doc_id: str) -> Optional[str]:
	"""Return the text of the document_summary chunk for *doc_id*, or None."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"""
			SELECT text FROM chunks
			WHERE doc_id = %s AND structural_role = 'document_summary'
			LIMIT 1
			""",
			(doc_id,),
		).fetchone()
		return str(row["text"]) if row and row["text"] else None
	finally:
		conn.close()


# ── Page-range summaries ──────────────────────────────────────────────────────

def fetch_chunks_in_page_range(db_dsn: str, doc_id: str, page_start: int, page_end: int) -> list:
	"""Return body chunks whose page_start falls within [page_start, page_end]."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT chunk_id, text, structural_role, path_text,
			       COALESCE(page_start, 0) AS page_start
			FROM chunks
			WHERE doc_id = %s
			  AND structural_role NOT IN ('document_summary', 'page_range_summary')
			  AND COALESCE(page_start, 0) >= %s
			  AND COALESCE(page_start, 0) <= %s
			ORDER BY COALESCE(page_start, 0)
			""",
			(doc_id, page_start, page_end),
		).fetchall()
		return [
			{
				"chunk_id": r["chunk_id"],
				"text": (r["text"] or "").strip(),
				"structural_role": r["structural_role"],
				"path_text": r["path_text"],
				"page_start": r["page_start"],
			}
			for r in rows
		]
	finally:
		conn.close()


def upsert_page_range_summary_chunk(
	db_dsn: str,
	*,
	doc_id: str,
	page_start: int,
	page_end: int,
	summary_text: str,
	embedding: Any,
	collection_id: str = "",
	document_title: str = "",
	source_name: str = "",
	document_path: str = "",
	source_type: str = "pdf_book",
) -> None:
	"""Insert or replace a page_range_summary chunk for the given range."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		chunk_id = f"{doc_id}_pages_{page_start}_{page_end}"
		vec = _fmt_vector(embedding)
		range_label = f"Pages {page_start}\u2013{page_end}"
		conn.execute(
			"""
			INSERT INTO chunks (
				chunk_id, doc_id, collection_id, source_name, document_title, document_path,
				section_id, path_text, title, level,
				page_start, page_end, has_table, token_count_est,
				source_type, structural_role, text, embedding
			) VALUES (%s, %s, %s, %s, %s, %s, NULL, %s, %s, 0,
			          %s, %s, 0, %s,
			          %s, 'page_range_summary', %s, %s::vector)
			ON CONFLICT(chunk_id) DO UPDATE SET
				text=excluded.text,
				embedding=excluded.embedding,
				document_title=excluded.document_title,
				source_name=excluded.source_name
			""",
			(
				chunk_id,
				doc_id,
				collection_id or None,
				source_name or document_title,
				document_title,
				document_path,
				range_label,
				range_label,
				page_start,
				page_end,
				len(summary_text.split()),
				source_type,
				summary_text,
				vec,
			),
		)
		conn.commit()
	finally:
		conn.close()


def list_page_range_summaries(db_dsn: str, doc_id: Optional[str] = None) -> list:
	"""Return page_range_summary chunks, optionally filtered to one doc_id."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if doc_id:
			rows = conn.execute(
				"""
				SELECT chunk_id, doc_id, document_title, page_start, page_end, text
				FROM chunks
				WHERE doc_id = %s AND structural_role = 'page_range_summary'
				ORDER BY COALESCE(page_start, 0)
				""",
				(doc_id,),
			).fetchall()
		else:
			rows = conn.execute(
				"""
				SELECT chunk_id, doc_id, document_title, page_start, page_end, text
				FROM chunks
				WHERE structural_role = 'page_range_summary'
				ORDER BY doc_id, COALESCE(page_start, 0)
				"""
			).fetchall()
		return [
			{
				"chunk_id": r["chunk_id"],
				"doc_id": r["doc_id"],
				"document_title": r["document_title"] or "",
				"page_start": r["page_start"],
				"page_end": r["page_end"],
				"preview": (r["text"] or "")[:150].rstrip() + "…" if r["text"] else "",
			}
			for r in rows
		]
	finally:
		conn.close()


def get_page_range_summary_text(db_dsn: str, doc_id: str, page_start: int, page_end: int) -> Optional[str]:
	"""Return summary text for a specific page range, or None if not found."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		chunk_id = f"{doc_id}_pages_{page_start}_{page_end}"
		row = conn.execute(
			"SELECT text FROM chunks WHERE chunk_id = %s",
			(chunk_id,),
		).fetchone()
		return str(row["text"]) if row and row["text"] else None
	finally:
		conn.close()
