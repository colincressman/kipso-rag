import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.chunk.assembly import chunk_structured_document, chunk_structured_file
from pipeline.chunk.merge import merge_small_chunks


def _structured_fixture() -> dict:
	return {
		"source_path": "data/markdown/spec.md",
		"metadata": {"doc_id": "abc123", "filename": "spec.md"},
		"sections": [
			{
				"section_id": "s0001-title",
				"title": "1 Scope",
				"level": 2,
				"parent_id": None,
				"path": ["1 Scope"],
				"path_text": "1 Scope",
				"content": " ".join(["scope"] * 500),
				"page_start": 1,
				"page_end": 2,
				"has_table": False,
			},
			{
				"section_id": "s0002-req",
				"title": "2 Requirements",
				"level": 2,
				"parent_id": None,
				"path": ["2 Requirements"],
				"path_text": "2 Requirements",
				"content": "short section",
				"page_start": 3,
				"page_end": 3,
				"has_table": True,
			},
		],
		"stats": {"section_count": 2},
	}


def test_chunk_structured_document_generates_chunks():
	chunks = chunk_structured_document(
		_structured_fixture(),
		max_tokens=120,
		overlap_tokens=20,
		min_chunk_tokens=20,
	)
	assert len(chunks) >= 2
	assert chunks[0]["doc_id"] == "abc123"
	assert chunks[0]["section_id"] == "s0001-title"
	assert chunks[0]["token_count_est"] > 0


def test_chunk_structured_document_skips_empty_sections():
	payload = _structured_fixture()
	payload["sections"].append(
		{
			"section_id": "s0003-empty",
			"title": "Empty",
			"level": 2,
			"parent_id": None,
			"path": ["Empty"],
			"path_text": "Empty",
			"content": "",
			"page_start": 4,
			"page_end": 4,
			"has_table": False,
		}
	)
	chunks = chunk_structured_document(payload)
	assert all(c["section_id"] != "s0003-empty" for c in chunks)


def test_chunk_structured_file_roundtrip(tmp_path: Path):
	in_path = tmp_path / "structured.json"
	out_path = tmp_path / "chunks.json"
	in_path.write_text(json.dumps(_structured_fixture()), encoding="utf-8")

	chunks = chunk_structured_file(str(in_path), output_path=str(out_path), max_tokens=120)
	assert len(chunks) > 0
	payload = json.loads(out_path.read_text(encoding="utf-8"))
	assert payload["chunk_count"] == len(chunks)
	assert "chunks" in payload


def test_merge_small_chunks_reduces_count():
	chunks = [
		{
			"chunk_id": "abc123-c000000",
			"doc_id": "abc123",
			"section_id": "s1",
			"path_text": "A",
			"text": "tiny",
			"token_count_est": 5,
			"word_count": 1,
			"page_start": 1,
			"page_end": 1,
			"metadata": {},
		},
		{
			"chunk_id": "abc123-c000001",
			"doc_id": "abc123",
			"section_id": "s1",
			"path_text": "A",
			"text": "this is a much larger chunk with enough content",
			"token_count_est": 60,
			"word_count": 9,
			"page_start": 1,
			"page_end": 2,
			"metadata": {},
		},
	]
	merged = merge_small_chunks(chunks, min_tokens=20, max_tokens_after_merge=200)
	assert len(merged) == 1
	assert "tiny" in merged[0]["text"]
	assert merged[0]["chunk_id"] == "abc123-c000000"


def test_merge_small_chunks_respects_section_boundary():
	chunks = [
		{
			"chunk_id": "abc123-c000000",
			"doc_id": "abc123",
			"section_id": "s1",
			"path_text": "A",
			"text": "tiny",
			"token_count_est": 5,
			"word_count": 1,
			"page_start": 1,
			"page_end": 1,
			"metadata": {},
		},
		{
			"chunk_id": "abc123-c000001",
			"doc_id": "abc123",
			"section_id": "s2",
			"path_text": "B",
			"text": "another section content",
			"token_count_est": 50,
			"word_count": 3,
			"page_start": 2,
			"page_end": 2,
			"metadata": {},
		},
	]
	merged = merge_small_chunks(chunks, min_tokens=20, max_tokens_after_merge=200)
	assert len(merged) == 2

