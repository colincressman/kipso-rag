#!/usr/bin/env python3
"""Delete the RAG SQLite database so ingestion can rebuild from scratch."""

from __future__ import annotations

from pathlib import Path


def main() -> int:
    db_files = [
        Path("data/db/rag.sqlite"),
        Path("data/db/rag.sqlite-shm"),
        Path("data/db/rag.sqlite-wal"),
    ]

    print("[reset-db] Starting DB reset")
    removed = 0
    for db_file in db_files:
        if db_file.exists():
            db_file.unlink()
            removed += 1
            print(f"[reset-db] Removed: {db_file}")
        else:
            print(f"[reset-db] Not found (skip): {db_file}")

    print(f"[reset-db] Done. Files removed: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
