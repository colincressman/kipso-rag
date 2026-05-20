"""Collection CRUD operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from db.init import _connect, init_db


def create_collection(
	db_dsn: str,
	collection_id: str,
	name: str,
	description: Optional[str] = None,
	parent_id: Optional[str] = None,
) -> None:
	"""Create a named collection. Raises ValueError if collection_id already exists."""
	if not collection_id or not name:
		raise ValueError("collection_id and name are required")
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		existing = conn.execute(
			"SELECT collection_id FROM collections WHERE collection_id = %s", (collection_id,)
		).fetchone()
		if existing:
			raise ValueError(f"Collection '{collection_id}' already exists")
		if parent_id is not None:
			parent_row = conn.execute(
				"SELECT collection_id FROM collections WHERE collection_id = %s", (parent_id,)
			).fetchone()
			if not parent_row:
				raise ValueError(f"Parent collection '{parent_id}' does not exist")
		conn.execute(
			"INSERT INTO collections (collection_id, name, description, parent_id) VALUES (%s, %s, %s, %s)",
			(collection_id, name, description, parent_id),
		)
		conn.commit()
	finally:
		conn.close()


def list_collections(db_dsn: str) -> list:
	"""Return all collections as a list of dicts, with chunk counts."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		rows = conn.execute(
			"""
			SELECT
				c.collection_id,
				c.name,
				c.description,
				c.parent_id,
				c.created_at,
				COUNT(DISTINCT ch.doc_id) AS doc_count,
				COUNT(ch.chunk_id) AS chunk_count
			FROM collections c
			LEFT JOIN chunks ch ON ch.collection_id = c.collection_id
			GROUP BY c.collection_id
			ORDER BY COALESCE(c.parent_id, c.collection_id), c.parent_id IS NOT NULL, c.collection_id
			"""
		).fetchall()
		return [
			{
				"collection_id": r["collection_id"],
				"name": r["name"],
				"description": r["description"],
				"parent_id": r["parent_id"],
				"created_at": r["created_at"],
				"doc_count": r["doc_count"],
				"chunk_count": r["chunk_count"],
			}
			for r in rows
		]
	finally:
		conn.close()


def get_collection(db_dsn: str, collection_id: str) -> Optional[Dict[str, Any]]:
	"""Return collection metadata + document list + sub-collections, or None if not found."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT collection_id, name, description, parent_id, created_at FROM collections WHERE collection_id = %s",
			(collection_id,),
		).fetchone()
		if not row:
			return None
		docs = conn.execute(
			"""
			SELECT DISTINCT d.doc_id, d.filename, d.source_type, COUNT(c.chunk_id) as chunk_count
			FROM chunks c
			JOIN documents d ON d.doc_id = c.doc_id
			WHERE c.collection_id = %s
			GROUP BY d.doc_id, d.filename, d.source_type
			ORDER BY d.filename
			""",
			(collection_id,),
		).fetchall()
		children = conn.execute(
			"SELECT collection_id, name FROM collections WHERE parent_id = %s ORDER BY collection_id",
			(collection_id,),
		).fetchall()
		return {
			"collection_id": row["collection_id"],
			"name": row["name"],
			"description": row["description"],
			"parent_id": row["parent_id"],
			"created_at": row["created_at"],
			"sub_collections": [
				{"collection_id": c["collection_id"], "name": c["name"]}
				for c in children
			],
			"documents": [
				{
					"doc_id": d["doc_id"],
					"filename": d["filename"],
					"source_type": d["source_type"],
					"chunk_count": d["chunk_count"],
				}
				for d in docs
			],
		}
	finally:
		conn.close()


def get_collection_scope(db_dsn: str, collection_id: str) -> List[str]:
	"""
	Return all collection_ids that should be searched when filtering by collection_id.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		children = conn.execute(
			"SELECT collection_id FROM collections WHERE parent_id = %s",
			(collection_id,),
		).fetchall()
		ids = [collection_id] + [r["collection_id"] for r in children]
		return ids
	finally:
		conn.close()


def assign_to_collection(
	db_dsn: str,
	collection_id: str,
	*,
	doc_ids: Optional[list] = None,
	source_type: Optional[str] = None,
) -> int:
	"""
	Assign existing chunks to a collection.

	At least one of doc_ids or source_type must be provided.
	Returns the number of chunks updated.
	"""
	if not doc_ids and not source_type:
		raise ValueError("Provide doc_ids or source_type to scope the assignment")
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		if doc_ids:
			placeholders = ",".join(["%s"] * len(doc_ids))
			conn.execute(
				f"UPDATE chunks SET collection_id = %s WHERE doc_id IN ({placeholders})",
				[collection_id, *doc_ids],
			)
		if source_type:
			conn.execute(
				"UPDATE chunks SET collection_id = %s WHERE source_type = %s",
				(collection_id, source_type),
			)
		conn.commit()
		return conn.execute(
			"SELECT COUNT(*) AS n FROM chunks WHERE collection_id = %s", (collection_id,)
		).fetchone()["n"]
	finally:
		conn.close()


def unassign_from_collection(db_dsn: str, doc_ids: list) -> int:
	"""Clear collection_id for given doc_ids. Returns number of chunks updated."""
	if not doc_ids:
		return 0
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		placeholders = ",".join(["%s"] * len(doc_ids))
		cur = conn.execute(
			f"UPDATE chunks SET collection_id = NULL WHERE doc_id IN ({placeholders})",
			doc_ids,
		)
		conn.commit()
		return cur.rowcount
	finally:
		conn.close()


def delete_collection(db_dsn: str, collection_id: str, *, clear_chunks: bool = True) -> int:
	"""
	Delete a collection record.

	If clear_chunks=True (default), sets collection_id=NULL on all chunks that
	belonged to it. Returns number of chunks cleared.
	"""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		cleared = 0
		if clear_chunks:
			cur = conn.execute(
				"UPDATE chunks SET collection_id = NULL WHERE collection_id = %s",
				(collection_id,),
			)
			cleared = cur.rowcount
		conn.execute(
			"DELETE FROM collections WHERE collection_id = %s", (collection_id,)
		)
		conn.commit()
		return cleared
	finally:
		conn.close()
