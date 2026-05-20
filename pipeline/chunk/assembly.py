"""
Chunk assembly — Chunk dataclass and top-level document chunking functions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.runtime_defaults import (
	DEFAULT_CHUNK_MAX_TOKENS,
	DEFAULT_CHUNK_MIN_TOKENS,
	DEFAULT_CHUNK_OVERLAP_TOKENS,
	DEFAULT_OVERSIZED_SEGMENT_CHARS,
)

from pipeline.chunk.filters import _chunk_is_low_value, _section_is_bibliography
from pipeline.chunk.strategies import (
	_split_by_paragraphs,
	_split_by_simulated_pages,
	_split_table_block,
	estimate_tokens,
)


@dataclass
class Chunk:
	chunk_id: str
	doc_id: str
	source_path: str
	section_id: str
	chunk_index_in_section: int
	text: str
	token_count_est: int
	word_count: int
	title: str
	path: List[str]
	path_text: str
	level: int
	page_start: Optional[int]
	page_end: Optional[int]
	has_table: bool
	structural_role: str = "body"
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


def chunk_structured_document(
	structured: Dict[str, Any],
	*,
	max_tokens: int = DEFAULT_CHUNK_MAX_TOKENS,
	overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
	min_chunk_tokens: int = DEFAULT_CHUNK_MIN_TOKENS,
	include_heading_in_chunk: bool = True,
) -> List[Dict[str, Any]]:
	"""
	Chunk an enriched structured document.

	Args:
		structured: output dict from structure.enrich
		max_tokens: target max chunk size
		overlap_tokens: overlap between adjacent chunks
		min_chunk_tokens: chunks below this can be merged in merge pass
		include_heading_in_chunk: prepend section title to each chunk body
	"""
	sections = structured.get("sections", [])
	metadata = structured.get("metadata", {})
	source_path = structured.get("source_path", "")
	doc_id = metadata.get("doc_id") or Path(source_path).stem.split("_", 1)[0]

	chunks: List[Chunk] = []
	global_idx = 0

	for sec in sections:
		content = (sec.get("content") or "").strip()
		if not content:
			continue

		title = sec.get("title", "")
		section_id = sec.get("section_id", "")
		level = sec.get("level", 0)
		page_start = sec.get("page_start")
		page_end = sec.get("page_end")
		path = sec.get("path", []) or []
		path_text = sec.get("path_text", " > ".join(path))
		has_table = bool(sec.get("has_table", False))
		structural_role = str(sec.get("structural_role", "body"))

		# Promote bibliography sections so they can be excluded from extraction.
		if structural_role == "body" and _section_is_bibliography(title, path_text):
			structural_role = "bibliography"

		# Table-aware chunking: preserve row boundaries by splitting only at
		# blank lines; add a [ TABLE ] prefix so retrievers know the chunk
		# contains structured tabular data.
		_table_max_tokens = max_tokens * 2  # allow larger chunks for tables
		if has_table:
			segments = _split_table_block(
				content,
				max_tokens=_table_max_tokens,
			)
		else:
			segments = _split_by_paragraphs(content, max_tokens=max_tokens)
		for in_sec_idx, segment in enumerate(segments):
			# Check if segment is oversized and needs page-aware splitting
			segment_len = len(segment)
			needs_page_split = segment_len > DEFAULT_OVERSIZED_SEGMENT_CHARS

			if needs_page_split:
				# Use simulated page splitting (fall back to page 1 if no page info)
				p_start = page_start if page_start is not None else 1
				p_end = page_end if page_end is not None else p_start
				page_chunks = _split_by_simulated_pages(segment, p_start, p_end, title)
				for page_seg, page_s, page_e in page_chunks:
					_table_prefix = "[ TABLE ]\n" if has_table else ""
					text = f"{title}\n\n{_table_prefix}{page_seg}" if include_heading_in_chunk and title else f"{_table_prefix}{page_seg}"
					if _chunk_is_low_value(page_seg):
						continue
					tok_est = estimate_tokens(text)
					chunk = Chunk(
						chunk_id=f"{doc_id}-c{global_idx:06d}",
						doc_id=doc_id,
						source_path=source_path,
						section_id=section_id,
						chunk_index_in_section=in_sec_idx,
						text=text,
						token_count_est=tok_est,
						word_count=len(text.split()),
						title=title,
						path=path,
						path_text=path_text,
						level=int(level or 0),
						page_start=page_s,
						page_end=page_e,
						has_table=has_table,
						structural_role=structural_role,
						metadata={
							"min_chunk_tokens": min_chunk_tokens,
							"page_split": True,
						},
					)
					chunks.append(chunk)
					global_idx += 1
			else:
				# Normal single chunk
				_table_prefix = "[ TABLE ]\n" if has_table else ""
				text = f"{title}\n\n{_table_prefix}{segment}" if include_heading_in_chunk and title else f"{_table_prefix}{segment}"
				if _chunk_is_low_value(segment):
					continue
				tok_est = estimate_tokens(text)

				chunk = Chunk(
					chunk_id=f"{doc_id}-c{global_idx:06d}",
					doc_id=doc_id,
					source_path=source_path,
					section_id=section_id,
					chunk_index_in_section=in_sec_idx,
					text=text,
					token_count_est=tok_est,
					word_count=len(text.split()),
					title=title,
					path=path,
					path_text=path_text,
					level=int(level or 0),
					page_start=page_start,
					page_end=page_end,
					has_table=has_table,
					structural_role=structural_role,
					metadata={
						"min_chunk_tokens": min_chunk_tokens,
					},
				)
				chunks.append(chunk)
				global_idx += 1

	return [c.to_dict() for c in chunks]


def chunk_structured_file(
	structured_json_path: str,
	output_path: Optional[str] = None,
	**kwargs: Any,
) -> List[Dict[str, Any]]:
	"""Load structured JSON, chunk it, optionally save chunks JSON."""
	path = Path(structured_json_path)
	structured = json.loads(path.read_text(encoding="utf-8"))
	chunks = chunk_structured_document(structured, **kwargs)

	if output_path:
		out = Path(output_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		payload = {
			"source_structured_path": str(path),
			"chunk_count": len(chunks),
			"chunks": chunks,
		}
		out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

	return chunks
