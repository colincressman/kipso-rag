"""Comprehensive manual diagnostic: test providers + pipeline logic across query types.

Usage:
    .venv\Scripts\python.exe scripts/test_web_providers.py
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval.web_search import BingRSSProvider, DuckDuckGoProvider, BingHTMLProvider
from retrieval.internet_fallback import _is_external_fact_query, _filter_search_results

# ── Query categories ──────────────────────────────────────────────────────────
QUERIES = [
    # (category, query)
    ("biographical",     "What is Donald Trump's youngest son's name?"),
    ("biographical",     "Who is Elon Musk's mother?"),
    ("fact-person",      "Who is the CEO of OpenAI?"),
    ("fact-financial",   "What is the current federal funds rate?"),
    ("current-events",   "What happened at the White House Correspondents Dinner 2026?"),
    ("ml-corpus",        "What is backpropagation?"),
    ("ml-corpus",        "Explain the attention mechanism in transformers"),
]

PROVIDERS = [
    BingRSSProvider(),
    DuckDuckGoProvider(),
    BingHTMLProvider(),
]

TIMEOUT = 15.0
MAX_RESULTS = 4


def run():
    # ── Section 1: _is_external_fact_query classification ─────────────────
    print("\n" + "=" * 70)
    print("SECTION 1: _is_external_fact_query classification")
    print("=" * 70)
    for category, query in QUERIES:
        result = _is_external_fact_query(query)
        flag = "FACT" if result else "----"
        print(f"  [{flag}] ({category:<16}) {query}")

    # ── Section 2: Provider results + filter output ────────────────────────
    for category, query in QUERIES:
        print(f"\n\n{'#' * 70}")
        print(f"CATEGORY: {category}")
        print(f"QUERY:    {query}")
        print(f"FACT?:    {_is_external_fact_query(query)}")
        print("#" * 70)

        all_raw = []
        for provider in PROVIDERS:
            print(f"\n  [{provider.name}]")
            print(f"  search URL: {provider.search_url(query)}")
            try:
                results = provider.search(query, max_results=MAX_RESULTS, timeout=TIMEOUT)
            except Exception as exc:
                print(f"  ERROR: {type(exc).__name__}: {exc}")
                continue
            if not results:
                print("  RESULT: 0 results (blocked or empty)")
                continue
            print(f"  RESULT: {len(results)} result(s)")
            for i, r in enumerate(results, 1):
                raw = {"url": r.url, "title": r.title, "snippet": r.snippet,
                       "provider": r.provider, "search_url": r.search_url}
                all_raw.append(raw)
                print(f"    [{i}] {r.url}")
                if r.title:
                    print(f"         title:   {r.title[:80]}")
                if r.snippet:
                    print(f"         snippet: {r.snippet[:120]}")

        if all_raw:
            filtered, rejected = _filter_search_results(query, all_raw)
            print(f"\n  -- Filter: {len(all_raw)} raw → {len(filtered)} kept, {len(rejected)} rejected --")
            for r in rejected:
                print(f"     REJECTED ({r.get('reject_reason','?')}): {r.get('url','')[:70]}")
            print(f"  -- Top kept after sort: --")
            for r in filtered[:3]:
                print(f"     KEPT: {r.get('url','')[:70]}")


if __name__ == "__main__":
    run()