"""
Rebuild the corpus scope manifest from scratch by scanning all chunks in the DB.

Usage
-----
    .venv\Scripts\python.exe scripts/rebuild_corpus_manifest.py
    .venv\Scripts\python.exe scripts/rebuild_corpus_manifest.py --db data/db/rag.sqlite
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.runtime_defaults import DEFAULT_DB_DSN
from retrieval.corpus_scope import rebuild_manifest, MANIFEST_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the corpus scope manifest.")
    parser.add_argument("--db", default=DEFAULT_DB_DSN, help="Path to the SQLite database.")
    parser.add_argument("--out", default=str(MANIFEST_PATH), help="Output path for the manifest JSON.")
    args = parser.parse_args()

    print(f"Rebuilding corpus manifest from: {args.db}")
    manifest = rebuild_manifest(args.db, Path(args.out))
    print(f"Done. {len(manifest.documents)} documents, {len(manifest.topics)} topics.")
    print()
    print(manifest.topic_summary())


if __name__ == "__main__":
    main()
