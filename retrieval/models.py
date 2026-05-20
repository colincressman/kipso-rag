"""Shared dataclasses and regex constants for the retrieval layer."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

# Matches canonical chunk IDs of the form "<doc_id>-c<6-digit-index>".
CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)-c(?P<idx>\d{6})$")


@dataclass
class RetrievalFilters:
	doc_id: Optional[str] = None
	doc_ids: Optional[List[str]] = None   # filter to a specific set of documents
	path_prefix: Optional[str] = None
	min_page: Optional[int] = None
	max_page: Optional[int] = None
	has_table: Optional[bool] = None
	source_type: Optional[str] = None
	structural_role: Optional[str] = None
	collection_id: Optional[str] = None


@dataclass
class RetrievedChunk:
	chunk_id: str
	doc_id: str
	collection_id: Optional[str]
	source_name: Optional[str]
	document_title: Optional[str]
	document_path: Optional[str]
	section_id: Optional[str]
	title: Optional[str]
	path_text: Optional[str]
	page_number: Optional[int]
	section_header: Optional[str]
	page_start: Optional[int]
	page_end: Optional[int]
	text: str
	score: float
	source_type: str = "pdf_book"
	structural_role: str = "body"
	metadata: Dict[str, Any] = field(default_factory=dict)

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class RetrievalResult:
	query: str
	top_k: int
	filters: Dict[str, Any]
	hits: List[RetrievedChunk]
	internet_fallback: Optional[Dict[str, Any]] = None
	hyde_trace: Optional[Dict[str, Any]] = None
	stepback_trace: Optional[Dict[str, Any]] = None
	perf_ms: Optional[Dict[str, float]] = None

	def to_dict(self) -> Dict[str, Any]:
		result = {
			"query": self.query,
			"top_k": self.top_k,
			"filters": self.filters,
			"hits": [h.to_dict() for h in self.hits],
		}
		if self.internet_fallback is not None:
			result["internet_fallback"] = self.internet_fallback
		if self.hyde_trace is not None:
			result["hyde_trace"] = self.hyde_trace
		if self.stepback_trace is not None:
			result["stepback_trace"] = self.stepback_trace
		if self.perf_ms is not None:
			result["perf_ms"] = self.perf_ms
		return result
