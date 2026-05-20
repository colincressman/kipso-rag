"""Query trace logging — records every retrieval + LLM call to a JSONL file."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.runtime_defaults import (
	DEFAULT_QUERY_TRACE_BACKUPS,
	DEFAULT_QUERY_TRACE_MAX_MB,
	DEFAULT_QUERY_TRACE_PATH,
)


def chunk_trace_rows(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
	"""Convert a list of retrieved hit dicts to compact trace row dicts."""
	rows: List[Dict[str, Any]] = []
	for h in hits:
		md = h.get("metadata") or {}
		rows.append(
			{
				"chunk_id": h.get("chunk_id"),
				"doc_id": h.get("doc_id"),
				"collection_id": h.get("collection_id") or md.get("collection_id"),
				"source_name": h.get("source_name") or md.get("source_name"),
				"document_title": h.get("document_title") or md.get("document_title"),
				"document_path": h.get("document_path") or md.get("document_path"),
				"section_header": h.get("section_header") or md.get("section_header") or h.get("title"),
				"page_number": h.get("page_number") or md.get("page_number") or h.get("page_start"),
				"retrieval_score": float(h.get("score", 0.0) or 0.0),
			}
		)
	return rows


def append_query_trace(
	*,
	query: str,
	mode: str,
	llm_used: bool,
	retrieved_rows: List[Dict[str, Any]],
	llm_input_rows: List[Dict[str, Any]],
	intent: Optional[str] = None,
	internet_triggered: bool = False,
	hyde_applied: bool = False,
	trace_path: str = DEFAULT_QUERY_TRACE_PATH,
) -> None:
	"""Append a single trace entry to the JSONL trace file, rotating when needed."""
	max_bytes = max(1, int(DEFAULT_QUERY_TRACE_MAX_MB)) * 1024 * 1024
	backups = max(1, int(DEFAULT_QUERY_TRACE_BACKUPS))

	entry = {
		"timestamp": datetime.now(timezone.utc).isoformat(),
		"query": query,
		"mode": mode,
		"intent": intent,
		"llm_used": llm_used,
		"internet_triggered": internet_triggered,
		"hyde_applied": hyde_applied,
		"retrieved_chunks": retrieved_rows,
		"llm_input_chunks": llm_input_rows,
	}
	path = Path(trace_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	if path.exists() and path.stat().st_size >= max_bytes:
		for idx in range(backups, 0, -1):
			older = path.with_suffix(path.suffix + f".{idx}")
			if idx == backups and older.exists():
				older.unlink()
			prev = path.with_suffix(path.suffix + f".{idx - 1}") if idx > 1 else path
			if prev.exists():
				os.replace(prev, older)
	with path.open("a", encoding="utf-8") as f:
		f.write(json.dumps(entry, ensure_ascii=False) + "\n")
