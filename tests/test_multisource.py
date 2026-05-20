from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest_multisource import (
	SourceInput,
	ingest_text_sources,
	load_qa_sources_from_json,
	load_text_sources_from_dir,
)
from retrieval.query import RetrievalFilters, retrieve


@pytest.mark.requires_postgres
def test_ingest_text_sources_persists_source_types(pg_dsn: str, tmp_path: Path):
	sources = [
		SourceInput(
			title="Ops note",
			text="Pump maintenance checklist and torque settings.",
			source_path=str(tmp_path / "notes" / "ops.md"),
			source_type="notes",
			metadata={"team": "ops"},
		),
		SourceInput(
			title="Web article",
			text="Market microstructure and liquidity providers overview.",
			source_path="https://example.com/liquidity",
			source_type="web_snippet",
			metadata={"domain": "example.com"},
		),
	]

	stats = ingest_text_sources(
		sources,
		db_dsn=pg_dsn,
		embed_backend="_test",
		embed_dimension=4096,
	)

	assert stats.documents == 2
	assert stats.chunks >= 2

	notes_only = retrieve(
		"torque settings",
		db_dsn=pg_dsn,
		top_k=3,
		filters=RetrievalFilters(source_type="notes"),
		embed_backend="_test",
		embed_dimension=4096,
	)
	assert notes_only.hits
	assert all(h.source_type == "notes" for h in notes_only.hits)

	web_only = retrieve(
		"liquidity providers",
		db_dsn=pg_dsn,
		top_k=3,
		filters=RetrievalFilters(source_type="web_snippet"),
		embed_backend="_test",
		embed_dimension=4096,
	)
	assert web_only.hits
	assert all(h.source_type == "web_snippet" for h in web_only.hits)


def test_loaders_for_notes_and_qa(tmp_path: Path):
	notes_dir = tmp_path / "notes"
	notes_dir.mkdir(parents=True, exist_ok=True)
	(notes_dir / "desk_note.md").write_text("Risk budget checklist", encoding="utf-8")
	(notes_dir / "todo.txt").write_text("Rebalance portfolio monthly", encoding="utf-8")

	note_sources = load_text_sources_from_dir(str(notes_dir), source_type="notes")
	assert len(note_sources) == 2
	assert all(s.source_type == "notes" for s in note_sources)

	qa_payload = {
		"items": [
			{"question": "What is CAPM?", "answer": "A model linking expected return and beta."},
			{"question": "", "answer": ""},
		]
	}
	qa_path = tmp_path / "qa.json"
	qa_path.write_text(json.dumps(qa_payload), encoding="utf-8")

	qa_sources = load_qa_sources_from_json(str(qa_path))
	assert len(qa_sources) == 1
	assert qa_sources[0].source_type == "qa_pairs"
	assert "Q:" in qa_sources[0].text
	assert "A:" in qa_sources[0].text
