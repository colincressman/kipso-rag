"""RAG retrieval CLI — thin wrapper around api.rag_retrieve().

For programmatic / multi-call use, import directly:

    from api import rag_retrieve
    result = rag_retrieve("query", top_k=20, source_type="spec")
    hits   = result["hits"]

CLI hand-off usage:

    python scripts/rag_retrieve.py "what is gradient descent?" --out retrieval.json
    python scripts/rag_retrieve.py "what is gradient descent?" | \\
        python scripts/llm_answer.py "what is gradient descent?"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import rag_retrieve
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
)


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def main() -> None:
    _configure_utf8_stdio()

    parser = argparse.ArgumentParser(
        description="RAG retrieval — outputs JSON for downstream LLM consumption"
    )
    parser.add_argument("query", type=str, help="Natural language query")
    parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--doc-id", type=str, default=None)
    parser.add_argument("--path-prefix", type=str, default=None)
    parser.add_argument("--min-page", type=int, default=None)
    parser.add_argument("--max-page", type=int, default=None)
    parser.add_argument("--has-table", action="store_true")
    parser.add_argument("--source-type", type=str, default=None,
                        help="Filter chunks by source_type (e.g. 'notes', 'pdf_book'). "
                             "Auto-detected from query if omitted.")
    parser.add_argument("--structural-role", type=str, default=None)
    parser.add_argument("--backend", type=str, default=DEFAULT_EMBED_BACKEND)
    parser.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
    parser.add_argument("--cross-encoder", dest="cross_encoder", action="store_true")
    parser.add_argument("--no-cross-encoder", dest="cross_encoder", action="store_false")
    parser.set_defaults(cross_encoder=None)
    parser.add_argument("--cross-only", dest="cross_only", action="store_true")
    parser.add_argument("--no-cross-only", dest="cross_only", action="store_false")
    parser.set_defaults(cross_only=None)
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON to FILE instead of stdout")
    args = parser.parse_args()

    result = rag_retrieve(
        args.query,
        top_k=args.top_k,
        db_dsn=args.db,
        source_type=args.source_type,
        doc_id=args.doc_id,
        path_prefix=args.path_prefix,
        min_page=args.min_page,
        max_page=args.max_page,
        has_table=True if args.has_table else None,
        structural_role=args.structural_role,
        embed_backend=args.backend,
        embed_model_name=args.model,
        cross_encoder_enabled=args.cross_encoder,
        cross_encoder_only=args.cross_only,
    )

    output = json.dumps(result, indent=2, ensure_ascii=False)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
        print(f"[rag_retrieve] Written to {args.out}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
