"""Trace reader — prints the last N entries from a JSONL trace file in a readable format.

Two trace files exist:
  - data/diagnostics/query_trace.jsonl  — LLM query traces (DEFAULT_QUERY_TRACE_PATH)
  - data/feedback/routing_trace.jsonl   — routing + feedback traces from the UI

Usage:
    python scripts/read_trace.py                                     # last 10 queries (query_trace)
    python scripts/read_trace.py -n 20                               # last 20 entries
    python scripts/read_trace.py --all                               # all entries
    python scripts/read_trace.py --path data/feedback/routing_trace.jsonl  # routing trace
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_defaults import DEFAULT_QUERY_TRACE_PATH


def _fmt_score(score: float | None) -> str:
    if score is None:
        return "  n/a"
    return f"{score:5.3f}"


def _top_score(entry: dict) -> float | None:
    chunks = entry.get("retrieved_chunks") or []
    if not chunks:
        return None
    return chunks[0].get("retrieval_score")


def _top_source(entry: dict) -> str:
    chunks = entry.get("retrieved_chunks") or []
    if not chunks:
        return "-"
    sources = []
    seen: set[str] = set()
    for c in chunks:
        s = c.get("source_name") or c.get("document_title") or "?"
        short = s[:25]
        if short not in seen:
            sources.append(short)
            seen.add(short)
        if len(seen) >= 2:
            break
    return ", ".join(sources)


def print_entry(entry: dict, idx: int) -> None:
    ts = entry.get("timestamp", "")[:19].replace("T", " ")
    query = entry.get("query", "")
    mode = entry.get("mode", "?")
    intent = entry.get("intent") or "-"
    llm_used = "llm" if entry.get("llm_used") else "ext"
    internet = "🌐" if entry.get("internet_triggered") else "  "
    hyde = "H" if entry.get("hyde_applied") else " "
    top_score = _top_score(entry)
    n_retrieved = len(entry.get("retrieved_chunks") or [])
    n_llm = len(entry.get("llm_input_chunks") or [])

    mode_short = mode.replace("_confidence", "").replace("high", "HIGH").replace("medium", "MED ").replace("low", "LOW ").replace("no_coverage", "NONE").replace("zero_lexical_coverage", "ZERO")[:8]

    query_preview = query[:72] + "…" if len(query) > 72 else query

    print(
        f"[{idx:3}] {ts}  "
        f"{internet}{hyde}  "
        f"{mode_short:<8}  "
        f"score={_fmt_score(top_score)}  "
        f"ret={n_retrieved:2} llm={n_llm:2}  "
        f"intent={intent:<16}  "
        f"{llm_used}  "
        f"{query_preview}"
    )
    top_src = _top_source(entry)
    if top_src != "-":
        print(f"       └─ top source: {top_src}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and summarise query_trace.jsonl")
    parser.add_argument("-n", type=int, default=10, metavar="N", help="Number of recent entries to show (default: 10)")
    parser.add_argument("--all", action="store_true", help="Show all entries")
    parser.add_argument("--path", default=DEFAULT_QUERY_TRACE_PATH, help="Path to the trace file")
    args = parser.parse_args()

    trace_path = Path(args.path)
    if not trace_path.exists():
        print(f"Trace file not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    entries: list[dict] = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        print("Trace file is empty.")
        return

    limit = len(entries) if args.all else args.n
    shown = entries[-limit:]

    print(f"Query trace — {trace_path}  ({len(entries)} total, showing {len(shown)})\n")
    print(f"{'':5}  {'timestamp':<19}  {'I H':<3}  {'mode':<8}  {'score':>9}  {'ret  llm':>8}  {'intent':<18}  {'use':<3}  query")
    print("─" * 120)
    for i, entry in enumerate(shown, start=len(entries) - len(shown) + 1):
        print_entry(entry, i)
    print()


if __name__ == "__main__":
    main()
