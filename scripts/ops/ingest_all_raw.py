#!/usr/bin/env python3
"""Batch-ingest files from data/raw one file at a time (repeatable, no timeout)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.ingest_v3 import ingest_file
from utils.book_registry import refresh_registry_from_db
from utils.runtime_defaults import DEFAULT_BOOK_REGISTRY_PATH, DEFAULT_DB_DSN


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


def _already_ingested_paths(db_dsn: str) -> set[str]:
    """Return the set of source_path values already recorded in the documents table."""
    try:
        import psycopg
        with psycopg.connect(db_dsn) as conn:
            rows = conn.execute("SELECT source_path FROM documents").fetchall()
            return {r[0] for r in rows}
    except Exception:
        return set()


def _gather_files(raw_dir: Path) -> list[Path]:
    files = [
        p for p in sorted(raw_dir.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest all supported files from data/raw")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing source files")
    parser.add_argument("--db-dsn", default=None, help="PostgreSQL DSN (default: from runtime.yaml)")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-ingest files that are already in the database (default: skip them)",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"[ingest-all] ERROR: raw directory not found: {raw_dir}")
        return 1

    files = _gather_files(raw_dir)
    print(f"[ingest-all] Found {len(files)} supported files in {raw_dir}", flush=True)
    for p in files:
        print(f"[ingest-all]   - {p.name}", flush=True)

    db = args.db_dsn or DEFAULT_DB_DSN
    already_done: set[str] = set() if args.force else _already_ingested_paths(db)
    if already_done:
        print(f"[ingest-all] {len(already_done)} file(s) already in DB — skipping (use --force to re-ingest)", flush=True)

    success = 0
    failed = 0
    skipped = 0

    for idx, file_path in enumerate(files, start=1):
        if str(file_path.resolve()) in already_done:
            print(f"[ingest-all] [{idx}/{len(files)}] SKIP {file_path.name} (already ingested)", flush=True)
            skipped += 1
            continue

        print("\n" + "=" * 88, flush=True)
        print(f"[ingest-all] [{idx}/{len(files)}] START {file_path.name}", flush=True)
        start_ts = time.time()

        try:
            result = ingest_file(
                str(file_path),
                db_dsn=db,
            )
            elapsed = time.time() - start_ts
            print(f"[ingest-all] [{idx}/{len(files)}] OK {file_path.name} | result={result} | {elapsed:.1f}s", flush=True)
            success += 1
        except Exception as exc:
            elapsed = time.time() - start_ts
            print(f"[ingest-all] [{idx}/{len(files)}] FAIL {file_path.name} | {elapsed:.1f}s", flush=True)
            print(f"[ingest-all] ERROR: {exc}", flush=True)
            failed += 1

    print("\n" + "=" * 88, flush=True)
    print(f"[ingest-all] COMPLETE | success={success} failed={failed} skipped={skipped} total={len(files)}", flush=True)

    if success > 0:
        db = args.db_dsn or DEFAULT_DB_DSN
        registry_payload = refresh_registry_from_db(db, DEFAULT_BOOK_REGISTRY_PATH)
        print(f"[ingest-all] Registry refreshed: {registry_payload['document_count']} document(s)", flush=True)

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
