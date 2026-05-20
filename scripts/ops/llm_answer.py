"""LLM answer CLI — thin wrapper around api.llm_answer().

For programmatic / multi-call use, import directly:

    from api import rag_retrieve, llm_answer
    result = rag_retrieve("query")
    answer = llm_answer("query", result)
    print(answer["answer"])

CLI usage:

    python scripts/llm_answer.py "query" --retrieval-file retrieval.json
    python scripts/rag_retrieve.py "query" | python scripts/llm_answer.py "query"

The query string must be provided as a positional argument even when
reading the retrieval JSON from stdin.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import llm_answer as _llm_answer
from utils.config import load_yaml_config
from utils.runtime_defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
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


def _load_llm_defaults() -> dict:
    cfg = load_yaml_config("configs/llm.yaml", default={})
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return {
        "llm_model": llm_cfg.get("model", DEFAULT_LLM_MODEL),
        "llm_base_url": llm_cfg.get("base_url", DEFAULT_LLM_BASE_URL),
        "llm_timeout": llm_cfg.get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS),
    }


def main() -> None:
    _configure_utf8_stdio()

    defaults = _load_llm_defaults()
    parser = argparse.ArgumentParser(
        description="LLM answer generation — consumes retrieval JSON, calls LLM"
    )
    parser.add_argument("query", type=str, help="Natural language query (must match the one used in rag_retrieve.py)")
    parser.add_argument(
        "--retrieval-file", type=str, default="-",
        help="Path to retrieval JSON produced by rag_retrieve.py. Use '-' (default) to read from stdin.",
    )
    parser.add_argument("--llm-model", type=str, default=defaults["llm_model"])
    parser.add_argument("--llm-base-url", type=str, default=defaults["llm_base_url"])
    parser.add_argument("--llm-timeout", type=float, default=float(defaults["llm_timeout"]))
    parser.add_argument("--llm-config", type=str, default="configs/llm.yaml")
    args = parser.parse_args()

    # Load retrieval JSON from file or stdin.
    if args.retrieval_file == "-":
        raw = sys.stdin.read()
    else:
        raw = Path(args.retrieval_file).read_text(encoding="utf-8")

    retrieval_result: dict = json.loads(raw)

    answer = _llm_answer(
        args.query,
        retrieval_result,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        llm_timeout_seconds=args.llm_timeout,
        config_path=args.llm_config,
    )

    print(json.dumps(answer, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
