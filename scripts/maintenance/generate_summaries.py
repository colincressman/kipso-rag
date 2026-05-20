"""Generate and store document-level summary chunks for all documents in the DB.

Strategy:
  - Small docs  (total text ≤ SINGLE_PASS_THRESHOLD): single LLM call.
  - Large docs  (total text  > SINGLE_PASS_THRESHOLD): map-reduce.
      Stage 1 (map):    Divide all chunks into N equal-span segments.
                        Summarize each segment independently (1-2 paragraphs).
      Stage 2 (reduce): Feed all segment summaries to LLM → final 4-5 paragraph summary.

Usage:
    python scripts/generate_summaries.py
    python scripts/generate_summaries.py --force
    python scripts/generate_summaries.py --doc-id 277ac35c
    python scripts/generate_summaries.py --dry-run
    python scripts/generate_summaries.py --list
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from db.client import (
    get_docs_without_summary,
    list_documents,
)
from llm.summarize import (
    MAX_CHARS_PER_SEGMENT_PROMPT,
    MAX_SEGMENTS,
    SEGMENT_CHARS,
    SINGLE_PASS_THRESHOLD,
    SUMMARY_TEMPERATURE,
    SUMMARY_TIMEOUT,
    _SYSTEM_REDUCE,
    _SYSTEM_SEGMENT,
    _SYSTEM_SINGLE,
    _llm,
    generate_doc_summary,
    reduce_prompt,
    segment_prompt,
    single_pass_prompt,
    split_into_segments,
)
from utils.runtime_defaults import DEFAULT_DB_DSN

# ── Core function (thin wrapper around llm.summarize.generate_doc_summary) ───

def generate_summary_for_doc(
    doc: dict,
    *,
    dry_run: bool = False,
    db_path: str = DEFAULT_DB_DSN,
) -> bool:
    from db.client import fetch_all_chunks_ordered

    doc_id = doc["doc_id"]
    title  = doc.get("document_title") or doc.get("filename") or doc_id

    if dry_run:
        all_chunks = fetch_all_chunks_ordered(db_path, doc_id)
        if not all_chunks:
            print(f"  ⚠  No chunks found for {doc_id!r} — skipping")
            return False
        total_chars = sum(len(c["text"]) for c in all_chunks)
        print(f"  → {len(all_chunks)} chunks, {total_chars:,} chars total")
        if total_chars <= SINGLE_PASS_THRESHOLD:
            print(f"  [dry-run] Strategy: single-pass")
        else:
            segs = split_into_segments(all_chunks, SEGMENT_CHARS, MAX_SEGMENTS)
            print(f"  [dry-run] Strategy: map-reduce → {len(segs)} segments")
        return True

    success = generate_doc_summary(
        doc_id,
        db_path,
        document_title=title,
        collection_id=doc.get("collection_id") or "",
        source_name=doc.get("source_name") or title,
        document_path=doc.get("document_path") or "",
        source_type=doc.get("source_type") or "pdf_book",
    )
    if success:
        print(f"  ✓  Summary generated for {title!r}")
    else:
        print(f"  ✗  Summary failed for {title!r}")
    return success


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate document summary chunks.")
    parser.add_argument("--force",   action="store_true", help="Re-generate existing summaries.")
    parser.add_argument("--doc-id",  default="",          help="Only process docs whose doc_id starts with this prefix.")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without writing to DB.")
    parser.add_argument("--list",    action="store_true", help="List pending docs and exit.")
    parser.add_argument("--db",      default=DEFAULT_DB_DSN, help="Path to SQLite DB.")
    args = parser.parse_args()

    db_path = args.db

    docs = list_documents(db_path) if args.force else get_docs_without_summary(db_path)

    if args.doc_id:
        docs = [d for d in docs if d["doc_id"].startswith(args.doc_id)]

    if args.list:
        if not docs:
            print("No pending documents.")
        else:
            print(f"{len(docs)} document(s) pending summary:\n")
            for d in docs:
                print(f"  {d['doc_id'][:12]}  {d.get('document_title') or d['filename']}")
        return

    if not docs:
        print("All documents already have summaries. Use --force to regenerate.")
        return

    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}Processing {len(docs)} document(s)…\n")

    ok = fail = 0
    for i, doc in enumerate(docs, 1):
        title = doc.get("document_title") or doc.get("filename") or doc["doc_id"]
        print(f"[{i}/{len(docs)}] {title}")
        try:
            success = generate_summary_for_doc(doc, dry_run=args.dry_run, db_dsn=db_path)
            ok += success
            fail += not success
        except Exception as exc:
            import traceback
            print(f"  ✗  Error: {exc}")
            traceback.print_exc()
            fail += 1
        print()

    print(f"Done. {ok} succeeded, {fail} failed.")


if __name__ == "__main__":
    main()
