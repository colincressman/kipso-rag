#!/usr/bin/env python3
"""Check Windows path safety for ingest artifact filenames."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.ingest_v3 import _artifact_stem


def _paths_for_pdf(root: Path, file_name: str) -> list[Path]:
    # Use deterministic long hash to represent doc_id width.
    stem = _artifact_stem("f" * 64, file_name)
    return [
        root / "data" / "extracted" / f"{stem}.pdf.json",
        root / "data" / "markdown" / f"{stem}.md",
        root / "data" / "structured" / f"{stem}.structured.json",
        root / "data" / "chunks" / f"{stem}.structured.chunks.json",
        root / "data" / "chunks" / f"{stem}.structured.chunks.merged.json",
        root / "data" / "index" / f"{stem}.structured.chunks.merged.index.json",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Check path lengths for ingest outputs")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory containing source files")
    parser.add_argument("--warn", type=int, default=220, help="Warning threshold")
    parser.add_argument("--fail", type=int, default=240, help="Failure threshold")
    args = parser.parse_args()

    raw_dir = (ROOT_DIR / args.raw_dir).resolve()
    if not raw_dir.exists():
        print(f"[path-check] ERROR: raw dir not found: {raw_dir}", flush=True)
        return 1

    pdfs = sorted(raw_dir.glob("*.pdf"))
    print(f"[path-check] Checking {len(pdfs)} PDF files", flush=True)
    print(f"[path-check] Thresholds: warn>={args.warn}, fail>={args.fail}", flush=True)

    warn_count = 0
    fail_count = 0
    max_len = -1
    max_path = ""
    max_file = ""

    for pdf in pdfs:
        paths = _paths_for_pdf(ROOT_DIR, pdf.name)
        path_lens = [(str(p), len(str(p))) for p in paths]
        local_max = max(path_lens, key=lambda x: x[1])

        if local_max[1] > max_len:
            max_len = local_max[1]
            max_path = local_max[0]
            max_file = pdf.name

        flagged = False
        for path_str, path_len in path_lens:
            if path_len >= args.fail:
                if not flagged:
                    print(f"[path-check] FAIL {pdf.name}", flush=True)
                    flagged = True
                print(f"[path-check]   len={path_len} {path_str}", flush=True)
                fail_count += 1
            elif path_len >= args.warn:
                if not flagged:
                    print(f"[path-check] WARN {pdf.name}", flush=True)
                    flagged = True
                print(f"[path-check]   len={path_len} {path_str}", flush=True)
                warn_count += 1

    print("[path-check] --- SUMMARY ---", flush=True)
    print(f"[path-check] max_len={max_len}", flush=True)
    print(f"[path-check] max_file={max_file}", flush=True)
    print(f"[path-check] max_path={max_path}", flush=True)
    print(f"[path-check] warns={warn_count} fails={fail_count}", flush=True)

    if fail_count > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
