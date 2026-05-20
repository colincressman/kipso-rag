#!/usr/bin/env python3
"""Repeatable rebuild: delete DB, then ingest all supported files from data/raw."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    py = sys.executable
    root = Path(__file__).resolve().parents[1]

    print("[rebuild] Step 1/2: Reset DB")
    reset_code = subprocess.call([py, "scripts/reset_db.py"], cwd=str(root))
    if reset_code != 0:
        print(f"[rebuild] DB reset failed with code={reset_code}")
        return reset_code

    print("[rebuild] Step 2/2: Ingest all files from RAW")
    ingest_code = subprocess.call([py, "scripts/ingest_all_raw.py"], cwd=str(root))
    print(f"[rebuild] Finished with code={ingest_code}")
    return ingest_code


if __name__ == "__main__":
    raise SystemExit(main())
