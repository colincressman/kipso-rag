from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.context_pack import build_context_pack
from retrieval.router import RoutedQuery, RetrievalStrategy


def _mk_hit(
	*,
	chunk_id: str,
	doc_id: str,
	text: str,
	score: float,
	source_type: str = "pdf_book",
) -> dict:
	return {
		"chunk_id": chunk_id,
		"doc_id": doc_id,
		"collection_id": "pdf_book",
		"source_name": f"{doc_id}.pdf",
		"document_title": "Title",
		"document_path": f"/tmp/{doc_id}.pdf",
		"section_id": "s1",
		"title": "Section",
		"path_text": "Book > Section",
		"page_number": 10,
		"section_header": "Section",
		"page_start": 10,
		"page_end": 11,
		"text": text,
		"score": score,
		"source_type": source_type,
		"structural_role": "body",
		"metadata": {
			"source_type": source_type,
			"collection_id": "pdf_book",
			"source_name": f"{doc_id}.pdf",
			"document_title": "Title",
			"document_path": f"/tmp/{doc_id}.pdf",
		},
	}


def _routed(intent: str = "summary") -> RoutedQuery:
	return RoutedQuery(
		original_query="q",
		intent=intent,
		sources=["corpus"],
		strategy=RetrievalStrategy(top_k=3),
		meta={"llm_routing": {"route_type": intent, "preferred_sources": ["corpus"], "confidence": 0.8, "valid": True}},
	)


def test_context_pack_deduplicates_redundant_chunks() -> None:
	h1 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Logistic regression uses sigmoid for binary classification.", score=0.80)
	h2 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Logistic regression uses sigmoid for binary classification.", score=0.79)
	result = {"query": "Explain logistic regression", "top_k": 3, "hits": [h1, h2]}

	pack = build_context_pack(result, _routed("summary"), max_chunks=3)
	assert len(pack["selected_chunks"]) == 1
	assert pack["selection_meta"]["deduplication"]["dropped_duplicate_chunk_id"] >= 1


def test_context_pack_prioritizes_high_authority_source() -> None:
	web_hit = _mk_hit(
		chunk_id="dweb-c000001",
		doc_id="dweb",
		text="Short web snippet",
		score=0.90,
		source_type="web",
	)
	pdf_hit = _mk_hit(
		chunk_id="dpdf-c000001",
		doc_id="dpdf",
		text="Authoritative textbook explanation",
		score=0.86,
		source_type="pdf_book",
	)
	result = {"query": "Explain logistic regression", "top_k": 1, "hits": [web_hit, pdf_hit]}

	pack = build_context_pack(result, _routed("fact_lookup"), max_chunks=1)
	assert pack["selected_chunks"][0]["chunk_id"] == "dpdf-c000001"


def test_context_pack_applies_conditional_diversification() -> None:
	h1 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Topic details A", score=0.82)
	h2 = _mk_hit(chunk_id="d1-c000002", doc_id="d1", text="Topic details B", score=0.80)
	h3 = _mk_hit(chunk_id="d2-c000001", doc_id="d2", text="Complementary perspective", score=0.77)
	result = {"query": "Compare approaches", "top_k": 2, "hits": [h1, h2, h3]}

	pack = build_context_pack(result, _routed("comparison"), max_chunks=2)
	doc_ids = [h.get("doc_id") for h in pack["selected_chunks"]]
	assert len(set(doc_ids)) >= 2
	assert pack["selection_meta"]["conditional_diversification_applied"] is True


def test_context_pack_retains_provenance_metadata() -> None:
	h1 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Provenance test", score=0.80)
	result = {"query": "q", "top_k": 1, "hits": [h1]}
	pack = build_context_pack(result, _routed("fact_lookup"), max_chunks=1)

	selected = pack["selected_chunks"][0]
	for key in ["collection_id", "source_name", "document_title", "document_path", "page_number", "section_header"]:
		assert key in selected
	assert "metadata" in selected
	assert selected["metadata"].get("document_path")


def test_context_pack_honors_internet_priority_signal() -> None:
	web_hit = _mk_hit(
		chunk_id="web-c000001",
		doc_id="web",
		text="Argentina won the 2022 FIFA World Cup final.",
		score=0.35,
		source_type="internet",
	)
	pdf_hit = _mk_hit(
		chunk_id="pdf-c000001",
		doc_id="pdf",
		text="Unrelated local chunk",
		score=0.88,
		source_type="pdf_book",
	)
	result = {
		"query": "Who won the FIFA World Cup in 2022?",
		"top_k": 2,
		"hits": [web_hit, pdf_hit],
		"internet_fallback": {"priority_applied": True},
	}

	pack = build_context_pack(result, _routed("fact_lookup"), max_chunks=2)
	assert pack["selected_chunks"][0]["source_type"] == "internet"
	assert pack["selection_meta"]["internet_priority_requested"] is True
	assert pack["selection_meta"]["internet_priority_honored"] is True
