"""Database query functions for the retrieval layer.

Provides:
  - _connect()          — open a pgvector-enabled psycopg connection
  - _text_rows()        — fetch all chunk rows matching RetrievalFilters
  - _vector_candidates() — ANN cosine search via pgvector
  - _chunk_neighbors()  — fetch adjacent chunks for context window expansion
  - _format_document_path() — apply document path display mode
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from db.client import init_db
from retrieval.models import CHUNK_ID_RE, RetrievalFilters
from utils.runtime_defaults import DEFAULT_DOCUMENT_PATH_MODE


def _connect(db_dsn: str) -> psycopg.Connection:
	# Ensure schema + idempotent migrations are applied before retrieval queries.
	init_db(db_dsn)
	conn = psycopg.connect(db_dsn, row_factory=dict_row)
	try:
		from pgvector.psycopg import register_vector
		register_vector(conn)
	except Exception:
		pass
	return conn


def _format_document_path(path: Optional[str]) -> Optional[str]:
	if not path:
		return path
	mode = (DEFAULT_DOCUMENT_PATH_MODE or "full").strip().lower()
	if mode == "full":
		return path
	if mode == "redact":
		return "<redacted>"
	if mode == "basename":
		return Path(path).name
	if mode == "relative":
		try:
			return Path(path).resolve().relative_to(Path.cwd().resolve()).as_posix()
		except Exception:
			return Path(path).name
	return path


def _text_rows(conn: psycopg.Connection, filters: RetrievalFilters) -> List[dict]:
	query = """
		SELECT c.chunk_id, c.doc_id,
			   c.collection_id,
			   c.source_name,
			   c.document_title,
			   c.document_path,
			   d.source_path AS doc_source_path,
			   d.filename AS doc_filename,
			   d.metadata_json AS doc_metadata_json,
			   c.section_id, c.path_text, c.title, c.level,
			   c.page_start, c.page_end, c.has_table, c.token_count_est,
			   c.source_type, c.structural_role, c.text
		FROM chunks c
		LEFT JOIN documents d ON d.doc_id = c.doc_id
		WHERE 1=1
	"""
	args: List[Any] = []

	if filters.doc_ids:
		placeholders = ",".join("%s" for _ in filters.doc_ids)
		query += f" AND c.doc_id IN ({placeholders})"
		args.extend(filters.doc_ids)
	elif filters.doc_id:
		query += " AND c.doc_id = %s"
		args.append(filters.doc_id)
	if filters.path_prefix:
		query += " AND c.path_text LIKE %s"
		args.append(f"{filters.path_prefix}%")
	if filters.min_page is not None:
		query += " AND (c.page_end IS NULL OR c.page_end >= %s)"
		args.append(int(filters.min_page))
	if filters.max_page is not None:
		query += " AND (c.page_start IS NULL OR c.page_start <= %s)"
		args.append(int(filters.max_page))
	if filters.has_table is not None:
		query += " AND c.has_table = %s"
		args.append(1 if filters.has_table else 0)
	if filters.source_type:
		query += " AND c.source_type = %s"
		args.append(filters.source_type)
	if filters.structural_role:
		query += " AND c.structural_role = %s"
		args.append(filters.structural_role)
	if filters.collection_id:
		children = conn.execute(
			"SELECT collection_id FROM collections WHERE parent_id = %s",
			(filters.collection_id,),
		).fetchall()
		scoped_ids = [filters.collection_id] + [r["collection_id"] for r in children]
		placeholders = ",".join("%s" for _ in scoped_ids)
		query += f" AND c.collection_id IN ({placeholders})"
		args.extend(scoped_ids)

	query += " ORDER BY c.chunk_id"
	return list(conn.execute(query, args).fetchall())


def _vector_candidates(
	conn: psycopg.Connection,
	filters: RetrievalFilters,
	qvec: List[float],
	*,
	stage2_vec: Optional[List[float]] = None,
	two_stage_alpha: float = 0.6,
	limit: int = 300,
) -> List[dict]:
	"""Return top-limit chunks by vector cosine similarity using pgvector ANN."""
	q1_str = "[" + ",".join(str(float(v)) for v in qvec) + "]"

	# Build WHERE clause (same filter logic as _text_rows)
	where_clauses: List[str] = []
	args: List[Any] = []

	if filters.doc_ids:
		ph = ",".join("%s" for _ in filters.doc_ids)
		where_clauses.append(f"c.doc_id IN ({ph})")
		args.extend(filters.doc_ids)
	elif filters.doc_id:
		where_clauses.append("c.doc_id = %s")
		args.append(filters.doc_id)
	if filters.path_prefix:
		where_clauses.append("c.path_text LIKE %s")
		args.append(f"{filters.path_prefix}%")
	if filters.min_page is not None:
		where_clauses.append("(c.page_end IS NULL OR c.page_end >= %s)")
		args.append(int(filters.min_page))
	if filters.max_page is not None:
		where_clauses.append("(c.page_start IS NULL OR c.page_start <= %s)")
		args.append(int(filters.max_page))
	if filters.has_table is not None:
		where_clauses.append("c.has_table = %s")
		args.append(1 if filters.has_table else 0)
	if filters.source_type:
		where_clauses.append("c.source_type = %s")
		args.append(filters.source_type)
	if filters.structural_role:
		where_clauses.append("c.structural_role = %s")
		args.append(filters.structural_role)
	if filters.collection_id:
		children = conn.execute(
			"SELECT collection_id FROM collections WHERE parent_id = %s",
			(filters.collection_id,),
		).fetchall()
		scoped_ids = [filters.collection_id] + [r["collection_id"] for r in children]
		ph = ",".join("%s" for _ in scoped_ids)
		where_clauses.append(f"c.collection_id IN ({ph})")
		args.extend(scoped_ids)

	where_clause = ("AND " + " AND ".join(where_clauses)) if where_clauses else ""

	if stage2_vec is not None:
		s2_str = "[" + ",".join(str(float(v)) for v in stage2_vec) + "]"
		score_expr = (
			f"({two_stage_alpha} * (1.0 - (c.embedding <=> %s::vector))"
			f" + {1.0 - two_stage_alpha} * (1.0 - (c.embedding <=> %s::vector)))"
		)
		sql = f"""
			SELECT c.chunk_id, {score_expr} AS score
			FROM chunks c
			WHERE c.embedding IS NOT NULL {where_clause}
			ORDER BY score DESC
			LIMIT %s
		"""
		vec_args = [q1_str, s2_str] + args + [limit]
	else:
		sql = f"""
			SELECT c.chunk_id, (1.0 - (c.embedding <=> %s::vector)) AS score
			FROM chunks c
			WHERE c.embedding IS NOT NULL {where_clause}
			ORDER BY c.embedding <=> %s::vector
			LIMIT %s
		"""
		vec_args = [q1_str] + args + [q1_str, limit]

	return list(conn.execute(sql, vec_args).fetchall())


def _chunk_neighbors(conn: psycopg.Connection, chunk_id: str, window: int = 1) -> List[Tuple[str, str]]:
	m = CHUNK_ID_RE.match(chunk_id)
	if not m:
		return []
	doc_id = m.group("doc")
	idx = int(m.group("idx"))

	neighbors: List[Tuple[str, str]] = []
	for offset in range(-window, window + 1):
		if offset == 0:
			continue
		nidx = idx + offset
		if nidx < 0:
			continue
		nid = f"{doc_id}-c{nidx:06d}"
		row = conn.execute("SELECT chunk_id, text FROM chunks WHERE chunk_id = %s", (nid,)).fetchone()
		if row:
			neighbors.append((row["chunk_id"], row["text"]))
	return neighbors
