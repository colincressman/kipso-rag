from __future__ import annotations

import json
import re
import psycopg
from psycopg.rows import dict_row
from pathlib import Path
from typing import Any, Dict, Optional


_YEAR_PUBLISHER_RE = re.compile(r"\((?P<year>(?:19|20)\d{2})(?:,\s*(?P<publisher>[^)]+))?\)")


def _clean(value: Any) -> Optional[str]:
	if value is None:
		return None
	text = str(value).strip()
	return text or None


def _strip_suffixes(filename: str) -> str:
	stem = Path(filename).stem
	for suffix in (" - libgen.li", " - libgen.is", " - libgen.rs"):
		if stem.endswith(suffix):
			stem = stem[: -len(suffix)]
	return stem.strip()


def _infer_filename_parts(filename: str) -> Dict[str, Optional[str]]:
	stem = _strip_suffixes(filename)
	year = None
	publisher = None
	match = _YEAR_PUBLISHER_RE.search(stem)
	if match:
		year = _clean(match.group("year"))
		publisher = _clean(match.group("publisher"))
		stem = (stem[: match.start()] + stem[match.end() :]).strip(" -_")

	authors = None
	title = None
	parts = [part.strip() for part in stem.split(" - ") if part.strip()]
	if len(parts) >= 2:
		authors = parts[0]
		title = " - ".join(parts[1:]).strip()
	elif parts:
		title = parts[0]

	return {
		"authors": _clean(authors),
		"title": _clean(title),
		"publisher": publisher,
		"year": year,
	}


def summarize_registry_entry(entry: Dict[str, Any]) -> Dict[str, str]:
	summary: Dict[str, str] = {}
	for key in ("title", "authors", "publisher", "year"):
		value = _clean(entry.get(key))
		if value:
			summary[key] = value
	return summary


def normalize_document_registry_entry(
	*,
	doc_id: str,
	filename: str,
	source_path: str,
	source_type: str,
	metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
	meta = dict(metadata or {})
	inferred = _infer_filename_parts(filename)
	title = _clean(meta.get("title") or meta.get("document_title") or inferred.get("title") or Path(filename).stem)
	authors = _clean(meta.get("author") or meta.get("authors") or inferred.get("authors"))
	publisher = _clean(meta.get("publisher") or inferred.get("publisher"))
	year = _clean(meta.get("year") or meta.get("publication_year") or inferred.get("year"))

	entry = {
		"doc_id": doc_id,
		"filename": _clean(filename),
		"source_path": _clean(source_path),
		"source_type": _clean(source_type) or "pdf_book",
		"title": title,
		"authors": authors,
		"publisher": publisher,
		"year": year,
		"subject": _clean(meta.get("subject")),
		"creator": _clean(meta.get("creator")),
		"producer": _clean(meta.get("producer")),
		"collection_id": _clean(meta.get("collection_id")),
		"source_name": _clean(meta.get("source_name") or filename),
		"document_title": _clean(meta.get("document_title") or title),
		"document_path": _clean(meta.get("document_path") or source_path),
		"is_book": bool((source_type or "").startswith("pdf") or str(source_type or "") == "docx"),
	}
	entry["registry_summary"] = summarize_registry_entry(entry)
	return entry


def load_registry_from_db(db_dsn: str) -> Dict[str, Dict[str, Any]]:
	conn = psycopg.connect(db_dsn, row_factory=dict_row)
	try:
		rows = conn.execute(
			"SELECT doc_id, filename, source_path, source_type, metadata_json FROM documents ORDER BY filename"
		).fetchall()
		registry: Dict[str, Dict[str, Any]] = {}
		for row in rows:
			try:
				meta = json.loads(row["metadata_json"] or "{}")
			except json.JSONDecodeError:
				meta = {}
			entry = normalize_document_registry_entry(
				doc_id=str(row["doc_id"]),
				filename=str(row["filename"] or ""),
				source_path=str(row["source_path"] or ""),
				source_type=str(row["source_type"] or "pdf_book"),
				metadata=meta,
			)
			registry[str(row["doc_id"])] = entry
		return registry
	finally:
		conn.close()


def save_registry_file(registry: Dict[str, Dict[str, Any]], *, db_dsn: str, out_path: str) -> Dict[str, Any]:
	payload = {
		"db": db_dsn,
		"document_count": len(registry),
		"documents": list(registry.values()),
	}
	path = Path(out_path)
	path.parent.mkdir(parents=True, exist_ok=True)
	path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
	return payload


def refresh_registry_from_db(db_dsn: str, out_path: str) -> Dict[str, Any]:
	registry = load_registry_from_db(db_dsn)
	return save_registry_file(registry, db_dsn=db_dsn, out_path=out_path)