"""Chunk questions (hypothetical question index) CRUD operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.init import _connect, init_db


def upsert_chunk_questions(
	db_dsn: str,
	chunk_id: str,
	questions: List[str],
	embeddings: List[List[float]],
) -> None:
	"""Insert (chunk_id, question, embedding) rows, replacing existing ones.

	Existing questions for *chunk_id* are deleted first so a re-run is safe.
	"""
	if not questions or not embeddings:
		return
	if len(questions) != len(embeddings):
		raise ValueError("questions and embeddings must have the same length")
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute("DELETE FROM chunk_questions WHERE chunk_id = %s", (chunk_id,))
		for question, embedding in zip(questions, embeddings):
			conn.execute(
				"INSERT INTO chunk_questions (chunk_id, question, embedding) VALUES (%s, %s, %s)",
				(chunk_id, question, embedding),
			)
		conn.commit()
	finally:
		conn.close()


def search_chunks_by_question_embedding(
	db_dsn: str,
	query_vector: List[float],
	*,
	limit: int = 20,
	collection_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
	"""ANN search over chunk_questions embeddings.

	Returns rows with keys: chunk_id, question, score (cosine similarity).
	Deduplicated by chunk_id — the highest-scoring question per chunk is kept.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		# Cast to halfvec so the IVFFlat halfvec index can be used.
		coll_filter = "AND c.collection_id = %(cid)s" if collection_id else ""
		sql = f"""
		SELECT DISTINCT ON (cq.chunk_id)
			cq.chunk_id,
			cq.question,
			1 - (cq.embedding::halfvec(4096) <=> %(vec)s::halfvec(4096)) AS score
		FROM chunk_questions cq
		JOIN chunks c ON c.chunk_id = cq.chunk_id
		WHERE cq.embedding IS NOT NULL
		{coll_filter}
		ORDER BY cq.chunk_id, score DESC
		LIMIT %(lim)s
		"""
		params: Dict[str, Any] = {"vec": query_vector, "lim": int(limit)}
		if collection_id:
			params["cid"] = collection_id
		rows = conn.execute(sql, params).fetchall()
		return [dict(r) for r in rows]
	except Exception:
		return []
	finally:
		conn.close()


def count_chunk_questions(db_dsn: str) -> int:
	"""Return the total number of rows in chunk_questions."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute("SELECT COUNT(*) AS n FROM chunk_questions").fetchone()
		return int((row or {}).get("n", 0))
	except Exception:
		return 0
	finally:
		conn.close()
