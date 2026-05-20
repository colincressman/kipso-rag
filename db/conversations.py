"""Conversation persistence operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from db.init import _connect, init_db


_CONV_SUMMARIZE_THRESHOLD = 40   # total messages (20 turns) before compressing
_CONV_KEEP_RECENT         = 20   # keep the most recent N messages live after compression


def create_conversation(db_dsn: str, *, conversation_id: Optional[str] = None, title: Optional[str] = None) -> str:
	"""Create a new conversation row. Returns the conversation_id."""
	import uuid
	init_db(db_dsn)
	cid = conversation_id or str(uuid.uuid4())
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"INSERT INTO conversations (conversation_id, title) VALUES (%s, %s)",
			(cid, title),
		)
		conn.commit()
	finally:
		conn.close()
	return cid


def get_conversation(db_dsn: str, conversation_id: str) -> Optional[Dict[str, Any]]:
	"""Return conversation dict with messages list, or None if not found."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT conversation_id, title, created_at, updated_at, archived, summary "
			"FROM conversations WHERE conversation_id = %s",
			(conversation_id,),
		).fetchone()
		if not row:
			return None
		msgs = conn.execute(
			"SELECT message_id, role, content, mode, sequence, created_at "
			"FROM conversation_messages WHERE conversation_id = %s ORDER BY sequence",
			(conversation_id,),
		).fetchall()
		return {
			"conversation_id": row["conversation_id"],
			"title": row["title"],
			"created_at": row["created_at"],
			"updated_at": row["updated_at"],
			"archived": bool(row["archived"]),
			"summary": row["summary"],
			"messages": [dict(m) for m in msgs],
		}
	finally:
		conn.close()


def list_conversations(db_dsn: str, *, limit: int = 30, include_archived: bool = False) -> List[Dict[str, Any]]:
	"""Return conversations, most recent first, with message counts."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		where = "" if include_archived else "WHERE c.archived = 0"
		rows = conn.execute(
			f"""
			SELECT c.conversation_id, c.title, c.created_at, c.updated_at, c.archived,
			       COUNT(m.message_id) AS message_count
			FROM conversations c
			LEFT JOIN conversation_messages m ON m.conversation_id = c.conversation_id
			{where}
			GROUP BY c.conversation_id, c.title, c.created_at, c.updated_at, c.archived
			ORDER BY c.updated_at DESC
			LIMIT %s
			""",
			(limit,),
		).fetchall()
		return [
			{
				"conversation_id": r["conversation_id"],
				"title": r["title"],
				"created_at": r["created_at"],
				"updated_at": r["updated_at"],
				"archived": bool(r["archived"]),
				"message_count": r["message_count"],
			}
			for r in rows
		]
	finally:
		conn.close()


def add_conversation_message(
	db_dsn: str,
	conversation_id: str,
	role: str,
	content: str,
	*,
	mode: Optional[str] = None,
) -> str:
	"""Append a message to a conversation. Returns message_id."""
	import uuid
	mid = str(uuid.uuid4()).replace("-", "")[:16]
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		seq = conn.execute(
			"SELECT COALESCE(MAX(sequence), 0) + 1 AS seq FROM conversation_messages WHERE conversation_id = %s",
			(conversation_id,),
		).fetchone()["seq"]
		conn.execute(
			"INSERT INTO conversation_messages (message_id, conversation_id, role, content, mode, sequence) "
			"VALUES (%s, %s, %s, %s, %s, %s)",
			(mid, conversation_id, role, content, mode, seq),
		)
		conn.execute(
			"UPDATE conversations SET updated_at = NOW() WHERE conversation_id = %s",
			(conversation_id,),
		)
		conn.commit()
	finally:
		conn.close()
	return mid


def set_conversation_title(db_dsn: str, conversation_id: str, title: str) -> None:
	"""Set the display title for a conversation."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"UPDATE conversations SET title = %s WHERE conversation_id = %s",
			(title, conversation_id),
		)
		conn.commit()
	finally:
		conn.close()


def compress_conversation(
	db_dsn: str,
	conversation_id: str,
	summary: str,
	*,
	keep_from_sequence: int,
) -> None:
	"""Store a summary and delete all messages with sequence < keep_from_sequence."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"UPDATE conversations SET summary = %s WHERE conversation_id = %s",
			(summary, conversation_id),
		)
		conn.execute(
			"DELETE FROM conversation_messages WHERE conversation_id = %s AND sequence < %s",
			(conversation_id, keep_from_sequence),
		)
		conn.commit()
	finally:
		conn.close()


def get_conversation_message_count(db_dsn: str, conversation_id: str) -> int:
	"""Return the total number of messages in a conversation."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		row = conn.execute(
			"SELECT COUNT(*) FROM conversation_messages WHERE conversation_id = %s",
			(conversation_id,),
		).fetchone()
		return row[0] if row else 0
	finally:
		conn.close()


def archive_conversation(db_dsn: str, conversation_id: str) -> None:
	"""Mark a conversation as archived (hidden from default list)."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute(
			"UPDATE conversations SET archived = 1 WHERE conversation_id = %s",
			(conversation_id,),
		)
		conn.commit()
	finally:
		conn.close()


def delete_conversation(db_dsn: str, conversation_id: str) -> None:
	"""Permanently delete a conversation and all its messages."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		conn.execute("DELETE FROM conversations WHERE conversation_id = %s", (conversation_id,))
		conn.commit()
	finally:
		conn.close()


def archive_stale_conversations(db_dsn: str, *, days: int = 7) -> int:
	"""Archive conversations with no activity in the last `days` days. Returns count archived."""
	init_db(db_dsn)
	conn = _connect(db_dsn)
	try:
		cutoff = datetime.now(timezone.utc) - timedelta(days=days)
		result = conn.execute(
			"UPDATE conversations SET archived = 1 "
			"WHERE archived = 0 AND updated_at < %s",
			(cutoff,),
		)
		conn.commit()
		return result.rowcount
	finally:
		conn.close()
