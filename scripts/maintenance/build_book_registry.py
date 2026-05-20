from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
	sys.path.insert(0, str(ROOT))

from utils.book_registry import refresh_registry_from_db
from utils.runtime_defaults import DEFAULT_BOOK_REGISTRY_PATH, DEFAULT_DB_DSN


def main() -> None:
	parser = argparse.ArgumentParser(description="Build a normalized metadata registry for documents in the RAG library.")
	parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
	parser.add_argument("--out", type=str, default=DEFAULT_BOOK_REGISTRY_PATH)
	args = parser.parse_args()

	payload = refresh_registry_from_db(args.db, args.out)
	print(f"Saved registry: {Path(args.out)}")
	print(f"Documents: {payload['document_count']}")


if __name__ == "__main__":
	main()