#!/usr/bin/env python3
"""Show current document/chunk counts in the RAG database."""

from __future__ import annotations

import psycopg
from psycopg.rows import dict_row

from utils.runtime_defaults import DEFAULT_DB_DSN


def main() -> int:
    try:
        conn = psycopg.connect(DEFAULT_DB_DSN, row_factory=dict_row)
    except Exception as exc:
        print(f"[db-status] Cannot connect to DB: {exc}")
        return 1

    with conn:
        doc_count = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        rows = conn.execute("SELECT filename, num_pages FROM documents ORDER BY filename").fetchall()

    print(f"[db-status] documents={doc_count}")
    print(f"[db-status] chunks={chunk_count}")
    for row in rows:
        print(f"[db-status] - {row['filename']} | pages={row['num_pages']}")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
