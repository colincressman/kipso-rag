"""RAG Fusion — multi-query expansion via paraphrase generation.

Standard technique from Raudaschl (2023) "RAG Fusion":
  1. Generate N paraphrase variants of the original query using the LLM.
  2. Run retrieve() once per paraphrase in parallel threads.
  3. RRF-merge all result lists (same formula as query_decompose.py).
  4. Re-rank the merged pool against the *original* query via the cross-encoder.

This improves recall for single-topic queries by covering vocabulary mismatches
between user phrasing and corpus language.  It is complementary to
query_decompose.py (which handles multi-topic/comparison queries).

Design choices:
- Paraphrase generation is a single LLM call that returns N variants in one
  response, not N separate calls.
- If the LLM call fails, we fall back to single-query retrieval silently.
- A `progress_fn` callback is threaded through so the UI can show
  "Generating query variants…" / "Expanding retrieval…" status messages.
- Activated only for intents where vocabulary lift helps:
  fact_lookup, exploratory, list_lookup, section_lookup.
  Skipped for: comparison (handled by query_decompose), formula_lookup,
  metadata_lookup, summary, conversational variants.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from retrieval.query_decompose import rrf_merge
from utils.runtime_defaults import (
    DEFAULT_HYDE_BASE_URL,
    DEFAULT_HYDE_MODEL,
    DEFAULT_HYDE_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

# Number of paraphrase variants to generate.
_N_VARIANTS = 3

# Intents for which RAG Fusion is enabled.
_FUSION_INTENTS = frozenset({
    "fact_lookup",
    "exploratory",
    "list_lookup",
    "section_lookup",
    "general",
    None,  # unknown intent — apply fusion as a safe default
})

_PARAPHRASE_SYSTEM_PROMPT = (
    "You are a query rewriting assistant. Given a user question, produce "
    f"{_N_VARIANTS} alternative phrasings of the same question. "
    "Each variant should use different vocabulary and sentence structure "
    "while preserving the exact meaning. "
    "Output ONLY the variants, one per line, with no numbering, bullets, or "
    "explanations. Do not include the original question."
)


# ── Paraphrase generation ──────────────────────────────────────────────────────

def _generate_paraphrases(
    query: str,
    *,
    model: str = DEFAULT_HYDE_MODEL,
    base_url: str = DEFAULT_HYDE_BASE_URL,
    timeout_seconds: float = DEFAULT_HYDE_TIMEOUT_SECONDS,
    n: int = _N_VARIANTS,
) -> List[str]:
    """Call the LLM to generate *n* paraphrase variants of *query*.

    Returns a list of variant strings (may be shorter than *n* on partial output).
    Returns an empty list on any LLM error so the caller can fall back gracefully.
    """
    if not (query or "").strip():
        return []

    # Free intent classifier VRAM before LLM loads (mirrors HyDE behaviour).
    try:
        from retrieval.intent_classifier import unload as _unload_classifier  # noqa: PLC0415
        _unload_classifier()
    except Exception:
        pass

    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": -1,
        "think": False,
        "options": {
            "temperature": 0.7,  # slightly higher than HyDE for lexical variety
            "num_predict": 200,
            "num_ctx": 512,
        },
        "messages": [
            {"role": "system", "content": _PARAPHRASE_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        elapsed = time.monotonic() - t0
        raw = ((body.get("message") or {}).get("content") or "").strip()
        if not raw:
            return []
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        logger.info(
            "rag_fusion: generated %d variants in %.2fs for query %r",
            len(variants), elapsed, query,
        )
        return variants[:n]
    except Exception as exc:
        logger.warning("rag_fusion: paraphrase LLM call failed: %s", exc)
        return []


# ── Orchestrator ───────────────────────────────────────────────────────────────

def retrieve_with_fusion(
    query: str,
    retrieve_fn: Any,
    retrieve_kwargs: Dict[str, Any],
    *,
    intent: Optional[str] = None,
    top_k: int = 10,
    model: str = DEFAULT_HYDE_MODEL,
    base_url: str = DEFAULT_HYDE_BASE_URL,
    timeout_seconds: float = DEFAULT_HYDE_TIMEOUT_SECONDS,
    progress_fn: Optional[Callable[[str], None]] = None,
) -> Optional[Dict[str, Any]]:
    """Run RAG Fusion: generate query paraphrases, retrieve in parallel, RRF-merge.

    Parameters
    ----------
    query           : original user query
    retrieve_fn     : callable with signature ``retrieve_fn(query, **kwargs) -> dict``
    retrieve_kwargs : keyword args forwarded to each retrieve call
    intent          : routed intent — fusion is skipped for non-qualifying intents
    top_k           : number of hits in the merged result
    model / base_url / timeout_seconds : LLM connection params
    progress_fn     : optional callback(message: str) for UI status updates

    Returns ``None`` when fusion is not applied (caller falls back to normal retrieval).
    """
    _pg = progress_fn or (lambda _: None)

    # Skip fusion for intents where it adds noise rather than value.
    if intent not in _FUSION_INTENTS:
        return None

    _pg("Generating query variants…")
    variants = _generate_paraphrases(
        query,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )

    if not variants:
        logger.info("rag_fusion: no variants generated, skipping fusion")
        return None

    all_queries = [query] + variants  # original always included
    logger.info("rag_fusion: expanding retrieval with %d queries", len(all_queries))
    _pg(f"Expanding retrieval ({len(all_queries)} variants)…")

    # ── Parallel retrieval ────────────────────────────────────────────────────
    sub_results: List[Optional[Dict[str, Any]]] = [None] * len(all_queries)

    def _run(idx: int, q: str) -> None:
        try:
            kw = dict(retrieve_kwargs)
            kw["top_k"] = max(int(top_k), 10)
            sub_results[idx] = retrieve_fn(q, **kw)
        except Exception as exc:
            logger.warning("rag_fusion: sub-query %r failed: %s", q, exc)

    threads = [threading.Thread(target=_run, args=(i, q)) for i, q in enumerate(all_queries)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    valid = [r for r in sub_results if r is not None]
    if not valid:
        logger.warning("rag_fusion: all sub-queries failed, falling back")
        return None

    # ── RRF merge ─────────────────────────────────────────────────────────────
    hit_lists = [r.get("hits") or [] for r in valid]
    merged_hits = rrf_merge(hit_lists, top_k=top_k)

    base = dict(valid[0])
    base["hits"] = merged_hits
    base["query"] = query
    base["top_k"] = top_k
    base["rag_fusion"] = {
        "variants": variants,
        "all_queries": all_queries,
        "sub_result_count": len(valid),
        "pre_merge_hit_counts": [len(hl) for hl in hit_lists],
        "merged_hit_count": len(merged_hits),
    }
    return base
