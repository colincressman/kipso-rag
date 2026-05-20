#!/usr/bin/env python3
"""Ingest exactly one file (PDF/TXT/MD/DOCX/PY/…) using pipeline.ingest_file."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.ingest_v3 import ingest_file
from scripts.generate_summaries import generate_summary_for_doc
from utils.book_registry import refresh_registry_from_db
from utils.runtime_defaults import DEFAULT_BOOK_REGISTRY_PATH, DEFAULT_DB_DSN


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one source file")
    parser.add_argument("file", help="Path to source file")
    parser.add_argument("--db-dsn", default=None, help="PostgreSQL DSN (default: from runtime_defaults)")
    parser.add_argument(
        "--source-type",
        default=None,
        help="Override source_type stored in DB (e.g. 'notes', 'docx', 'pdf_book'). "
             "Useful when the file extension does not reflect the intended source type.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Assign ingested document to a named collection (e.g. 'CS7646', 'RL Books').",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip automatic document summary generation after ingest.",
    )
    args = parser.parse_args()

    src = Path(args.file)
    if not src.exists():
        print(f"[ingest-one] ERROR: file not found: {src}", flush=True)
        return 1

    print(f"[ingest-one] START {src}", flush=True)

    result = ingest_file(
        str(src),
        db_dsn=args.db_dsn,
        source_type=args.source_type,
        collection_id=args.collection,
    )
    print(f"[ingest-one] DONE {src.name} | result={result}", flush=True)

    db = args.db_dsn or DEFAULT_DB_DSN
    registry_payload = refresh_registry_from_db(db, DEFAULT_BOOK_REGISTRY_PATH)
    print(f"[ingest-one] Registry refreshed: {registry_payload['document_count']} document(s)", flush=True)

    # Auto-generate document summary unless suppressed
    doc_id = (result or {}).get("doc_id") if isinstance(result, dict) else None
    if not args.no_summary and doc_id:
        from db.client import get_docs_without_summary
        pending = get_docs_without_summary(db)
        to_summarize = [d for d in pending if d["doc_id"] == doc_id]
        for doc in to_summarize:
            print(f"[ingest-one] Generating summary for {doc.get('document_title') or doc.get('filename')}…", flush=True)
            try:
                generate_summary_for_doc(doc, db_dsn=db)
            except Exception as exc:
                print(f"[ingest-one] Summary generation failed (non-fatal): {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
