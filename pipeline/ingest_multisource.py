"""
Multi-source text ingestion utilities.

This module ingests non-PDF sources (notes, web snippets, QA files) into the
same SQLite chunk store used by the PDF pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from db.client import init_db, upsert_chunks_from_index, upsert_document_record, get_document_file_hash
from pipeline.chunk.strategies import _split_by_paragraphs, estimate_tokens as _estimate_tokens_chunker
from pipeline.embed.embedder import create_embedder
from pipeline.ingest_v3 import _extract_docx_text
from utils.runtime_defaults import (
	DEFAULT_BOOK_REGISTRY_PATH,
	DEFAULT_CHUNK_MAX_TOKENS,
	DEFAULT_CHUNK_OVERLAP_TOKENS,
	DEFAULT_EMBED_BACKEND,
	DEFAULT_DB_DSN,
	DEFAULT_EMBED_DIMENSION,
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_OLLAMA_BASE_URL,
	DEFAULT_OLLAMA_TIMEOUT_SECONDS,
)


def _read_source_text(path: Path) -> str:
	"""Read text from supported source files for multisource ingestion."""
	ext = path.suffix.lower()
	if ext == ".docx":
		return _extract_docx_text(path)
	return path.read_text(encoding="utf-8", errors="ignore").strip()


@dataclass
class SourceInput:
	"""Single source item for ingestion."""

	title: str
	text: str
	source_path: str
	source_type: str
	metadata: Dict[str, Any]
	collection_id: Optional[str] = None


@dataclass
class IngestStats:
	documents: int = 0
	chunks: int = 0


def _hash_doc_id(source_type: str, source_path: str) -> str:
	raw = f"{source_type}:{source_path}".encode("utf-8")
	return hashlib.sha256(raw).hexdigest()


def _estimate_tokens(text: str) -> int:
	return _estimate_tokens_chunker(text)


def _chunk_source_item(
	item: SourceInput,
	*,
	doc_id: str,
	max_tokens: int,
	overlap_tokens: int,
) -> List[Dict[str, Any]]:
	segments = _split_by_paragraphs(item.text.strip(), max_tokens=max_tokens)
	chunks: List[Dict[str, Any]] = []
	for idx, seg in enumerate(segments):
		chunk_id = f"{doc_id}-c{idx:06d}"
		chunks.append(
			{
				"chunk_id": chunk_id,
				"doc_id": doc_id,
				"section_id": f"s{idx:04d}",
				"path_text": item.title,
				"title": item.title,
				"level": 1,
				"page_start": None,
				"page_end": None,
				"has_table": False,
				"token_count_est": _estimate_tokens(seg),
				"source_type": item.source_type,
				"structural_role": "body",
				"text": seg,
			}
		)
	return chunks


def ingest_text_sources(
	sources: Iterable[SourceInput],
	*,
	db_dsn: str = DEFAULT_DB_DSN,
	embed_backend: str = DEFAULT_EMBED_BACKEND,
	embed_dimension: int = DEFAULT_EMBED_DIMENSION,
	embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
	embed_ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
	embed_ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
	max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
	overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
	collection_id: Optional[str] = None,
) -> IngestStats:
	"""Ingest arbitrary text sources into the shared chunk table."""
	init_db(db_dsn)
	embedder = create_embedder(
		backend=embed_backend,
		dimension=embed_dimension,
		model_name=embed_model_name,
		ollama_base_url=embed_ollama_base_url,
		ollama_timeout_seconds=embed_ollama_timeout_seconds,
	)

	stats = IngestStats()
	for source in sources:
		txt = (source.text or "").strip()
		if not txt:
			continue

		# collection_id precedence: per-source > function-level > None
		effective_collection = source.collection_id or collection_id or None

		doc_id = _hash_doc_id(source.source_type, source.source_path)

		# Skip re-ingest if file content is unchanged.
		content_hash = hashlib.sha256(txt.encode("utf-8")).hexdigest()
		stored_hash = get_document_file_hash(db_dsn, doc_id)
		if stored_hash is not None and stored_hash == content_hash:
			continue

		upsert_document_record(
			db_dsn,
			doc_id=doc_id,
			filename=Path(source.source_path).name or source.title,
			source_path=source.source_path,
			source_type=source.source_type,
			num_pages=1,
			metadata={
				**source.metadata,
				"title": source.title,
				"source_type": source.source_type,
				"collection_id": effective_collection,
				"source_name": source.metadata.get("source_name", Path(source.source_path).name or source.title),
				"document_title": source.metadata.get("document_title", source.title),
				"document_path": source.metadata.get("document_path", source.source_path),
			},
			ingested_at=datetime.now(timezone.utc).isoformat(),
			file_hash=content_hash,
		)

		raw_chunks = _chunk_source_item(
			source,
			doc_id=doc_id,
			max_tokens=max_tokens,
			overlap_tokens=overlap_tokens,
		)
		if not raw_chunks:
			continue

		vectors = embedder.embed_texts([c["text"] for c in raw_chunks])
		items: List[Dict[str, Any]] = []
		for c, vec in zip(raw_chunks, vectors):
			row = dict(c)
			row["embedding"] = vec
			row["embedding_dim"] = len(vec)
			items.append(row)

		inserted = upsert_chunks_from_index(db_dsn, {"items": items}, replace_doc_chunks=True)
		stats.documents += 1
		stats.chunks += inserted

	if stats.documents > 0:
		try:
			from utils.book_registry import refresh_registry_from_db
			refresh_registry_from_db(db_dsn, DEFAULT_BOOK_REGISTRY_PATH)
		except Exception:
			pass
		try:
			from retrieval.corpus_scope import update_manifest_for_doc, invalidate_cache
			# Update manifest for each newly ingested document using the same
			# hash function that ingest_text_sources uses for doc_id.
			for src in sources:
				doc_id = _hash_doc_id(src.source_type, src.source_path)
				update_manifest_for_doc(db_dsn, doc_id)
			invalidate_cache()
		except Exception:
			pass
	return stats


def load_text_sources_from_dir(
	input_dir: str,
	*,
	source_type: str = "text",
	patterns: Optional[List[str]] = None,
) -> List[SourceInput]:
	"""Load plain text-like files from a directory into SourceInput objects."""
	base = Path(input_dir)
	globs = patterns or ["*.md", "*.txt", "*.docx"]

	sources: List[SourceInput] = []
	for pattern in globs:
		for path in sorted(base.glob(pattern)):
			text = _read_source_text(path)
			if not text:
				continue
			sources.append(
				SourceInput(
					title=path.stem,
					text=text,
					source_path=str(path.resolve()),
					source_type=source_type,
					metadata={"ext": path.suffix.lower()},
				)
			)
	return sources


def load_qa_sources_from_json(
	json_path: str,
	*,
	source_type: str = "qa_pairs",
) -> List[SourceInput]:
	"""Load Q/A json rows into chunkable text sources.

	Supported shapes:
	- [{"question": "...", "answer": "..."}, ...]
	- {"items": [{"question": "...", "answer": "..."}, ...]}
	"""
	path = Path(json_path)
	payload = json.loads(path.read_text(encoding="utf-8"))
	if isinstance(payload, dict):
		rows = payload.get("items", [])
	elif isinstance(payload, list):
		rows = payload
	else:
		rows = []

	sources: List[SourceInput] = []
	for idx, row in enumerate(rows):
		if not isinstance(row, dict):
			continue
		question = str(row.get("question") or "").strip()
		answer = str(row.get("answer") or "").strip()
		if not question and not answer:
			continue
		body = f"Q: {question}\nA: {answer}".strip()
		sources.append(
			SourceInput(
				title=f"qa_{idx + 1:04d}",
				text=body,
				source_path=f"{path.resolve()}#item-{idx + 1}",
				source_type=source_type,
				metadata={"qa_index": idx + 1},
			)
		)
	return sources
