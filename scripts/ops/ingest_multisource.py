from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingest_multisource import (
	ingest_text_sources,
	load_qa_sources_from_json,
	load_text_sources_from_dir,
)
from utils.runtime_defaults import (
	DEFAULT_CHUNK_MAX_TOKENS,
	DEFAULT_CHUNK_OVERLAP_TOKENS,
	DEFAULT_DB_DSN,
	DEFAULT_EMBED_BACKEND,
	DEFAULT_EMBED_DIMENSION,
	DEFAULT_EMBED_MODEL_NAME,
)


def main() -> None:
	parser = argparse.ArgumentParser(description="Ingest non-PDF sources into RAG SQLite store")
	parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
	parser.add_argument("--notes-dir", type=str, default="")
	parser.add_argument("--web-dir", type=str, default="")
	parser.add_argument("--qa-json", type=str, default="")
	parser.add_argument("--backend", type=str, default=DEFAULT_EMBED_BACKEND)
	parser.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
	parser.add_argument("--dimension", type=int, default=DEFAULT_EMBED_DIMENSION)
	parser.add_argument("--max-tokens", type=int, default=DEFAULT_CHUNK_MAX_TOKENS)
	parser.add_argument("--overlap-tokens", type=int, default=DEFAULT_CHUNK_OVERLAP_TOKENS)
	parser.add_argument(
		"--collection",
		type=str,
		default=None,
		help="Assign all ingested sources to a named collection (e.g. 'CS7646', 'My Notes').",
	)
	args = parser.parse_args()

	sources = []
	if args.notes_dir:
		sources.extend(load_text_sources_from_dir(args.notes_dir, source_type="notes"))
	if args.web_dir:
		sources.extend(load_text_sources_from_dir(args.web_dir, source_type="web_snippet"))
	if args.qa_json:
		sources.extend(load_qa_sources_from_json(args.qa_json, source_type="qa_pairs"))

	stats = ingest_text_sources(
		sources,
		db_dsn=args.db,
		embed_backend=args.backend,
		embed_dimension=args.dimension,
		embed_model_name=args.model,
		max_tokens=args.max_tokens,
		overlap_tokens=args.overlap_tokens,
		collection_id=args.collection,
	)
	print(json.dumps({"documents": stats.documents, "chunks": stats.chunks, "source_count": len(sources)}, indent=2))


if __name__ == "__main__":
	main()
