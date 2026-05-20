"""Query decomposition for comparison and multi-topic queries.

When a query compares or asks about multiple distinct items (e.g. "compare
dropout vs batch normalization", "what is the difference between CAPM and APT?"),
a single embedding vector lands in a blended space between the topics and often
misses chunks that are specific to each individual item.

This module:
  1. Detects whether a query contains multiple distinct sub-topics.
  2. Extracts those sub-topics as separate focused queries.
  3. Runs retrieve() once per sub-query in parallel threads.
  4. Merges all result lists via Reciprocal Rank Fusion (RRF).

The merged result is a plain dict compatible with the existing retrieval result
format — callers receive it as if a single retrieve() call had run.

Reference: Raudaschl (2023) "RAG Fusion"; Cormack et al. (2009) "RRF".
"""

from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import logging

logger = logging.getLogger(__name__)

# ── RRF constant ───────────────────────────────────────────────────────────────
# k=60 is the standard RRF constant from Cormack et al. (2009).
_RRF_K = 60

# Maximum number of sub-topics to decompose into (keeps latency bounded).
_MAX_SUB_TOPICS = 3

# ── Pattern library for sub-topic extraction ──────────────────────────────────
# Ordered from most-specific to least-specific.  First match wins.

# "compare X and Y" / "compare X with Y" / "compare X to Y"
_PAT_COMPARE_AND = re.compile(
    r"\bcompare\s+(.+?)\s+(?:and|with|to|versus|vs\.?)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)

# "X vs Y" / "X versus Y"
_PAT_VS = re.compile(
    r"(.+?)\s+(?:vs\.?|versus)\s+(.+?)(?:\?|$)",
    re.IGNORECASE,
)

# "difference between X and Y" / "differences between X and Y"
_PAT_DIFF_BETWEEN = re.compile(
    r"\bdifferences?\s+between\s+(.+?)\s+and\s+([^,?]+?)(?:[,?]|$)",
    re.IGNORECASE,
)

# "how does X differ from Y" / "how do X and Y differ"
_PAT_DIFFER = re.compile(
    r"\bhow\s+(?:does\s+(.+?)\s+differ\s+from\s+(.+?)|do\s+(.+?)\s+and\s+(.+?)\s+differ)(?:\?|$)",
    re.IGNORECASE,
)

# "X and Y" at the end of a what/explain question — weakest signal, only used
# when nothing else matched and the query length suggests a dual-topic structure.
_PAT_AND_FALLBACK = re.compile(
    r"^(?:what\s+(?:is|are)|explain|describe|define)\b.+?\b(.{4,40}?)\s+and\s+(.{4,40})(?:\?|$)",
    re.IGNORECASE,
)

_ALL_PATTERNS: List[re.Pattern] = [
    _PAT_COMPARE_AND,
    _PAT_VS,
    _PAT_DIFF_BETWEEN,
    _PAT_DIFFER,
    _PAT_AND_FALLBACK,
]

# Stop-fragments that indicate a match group is too vague to be a useful sub-topic.
_VAGUE_FRAGMENTS = frozenset({
    "them", "they", "it", "these", "those", "this", "that",
    "the two", "both", "each", "either", "the difference",
})


def _clean_fragment(s: str) -> str:
    """Strip leading articles/filler and trailing punctuation from a match group."""
    s = s.strip().rstrip("?.!,;")
    s = re.sub(r"^(?:the|a|an)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


def extract_sub_topics(query: str) -> List[str]:
    """Return a list of sub-topic strings extracted from *query*.

    Returns an empty list when the query does not appear to contain multiple
    distinct topics — callers should fall back to normal single-query retrieval.
    """
    for pat in _ALL_PATTERNS:
        m = pat.search(query)
        if not m:
            continue
        # Gather all non-None groups (differ pattern has 4 groups, two alt branches)
        groups = [g for g in m.groups() if g is not None]
        topics = [_clean_fragment(g) for g in groups]
        topics = [t for t in topics if t and t.lower() not in _VAGUE_FRAGMENTS]
        # Require at least 2 non-trivially-short topics
        topics = [t for t in topics if len(t.split()) >= 1 and len(t) >= 2]
        if len(topics) >= 2:
            return topics[:_MAX_SUB_TOPICS]

    return []


def _build_sub_query(
    original: str,
    topic: str,
    *,
    all_topics: Optional[List[str]] = None,
) -> str:
    """Construct a focused retrieval query for a single *topic*.

    Keeps the original query's context words (what/how/explain/define) so the
    embedding still reflects the question type, but centres on one topic.
    When exactly two short topics are being compared, the sibling topic is
    included ("what is precision versus recall") so the sub-query is more
    specific and less likely to match unrelated content that happens to share
    the topic word.
    """
    # If the topic is already a complete sentence-like string, use as-is.
    if len(topic.split()) > 6:
        return topic

    # Detect question opener so we can preserve it.
    opener_m = re.match(
        r"^(what\s+is|what\s+are|how\s+does|how\s+do|explain|describe|define)\b",
        original,
        re.IGNORECASE,
    )
    opener = opener_m.group(0) if opener_m else "what is"

    # When exactly two short topics exist, anchor the sub-query against the
    # sibling so the embedding is more discriminating (e.g. "what is precision"
    # can match unrelated ML algorithm text; "what is precision versus recall"
    # points specifically at the evaluation metric comparison).
    if all_topics and len(all_topics) == 2:
        sibling = next(
            (t for t in all_topics if t.lower() != topic.lower()),
            None,
        )
        if sibling and len(sibling.split()) <= 4:
            return f"{opener} {topic} versus {sibling}"

    return f"{opener} {topic}"


def rrf_merge(
    result_lists: List[List[Dict[str, Any]]],
    *,
    k: int = _RRF_K,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """Merge multiple ranked hit lists via Reciprocal Rank Fusion.

    Each list is a list of chunk dicts (with a ``chunk_id`` key).
    Returns a single merged, re-ranked list of at most *top_k* unique chunks.
    The chunk dict from the list where a chunk scored highest is preserved.
    """
    scores: Dict[str, float] = {}
    best_hit: Dict[str, Dict[str, Any]] = {}

    for ranked_list in result_lists:
        for rank, hit in enumerate(ranked_list):
            cid = str(hit.get("chunk_id") or "")
            if not cid:
                continue
            rrf_score = 1.0 / (k + rank + 1)
            prev = scores.get(cid, 0.0)
            scores[cid] = prev + rrf_score
            # Keep the hit dict from whichever list had it ranked highest
            if cid not in best_hit or rrf_score > (1.0 / (k + best_hit[cid].get("_rrf_rank", 999) + 1)):
                hit_copy = dict(hit)
                hit_copy["_rrf_rank"] = rank
                best_hit[cid] = hit_copy

    merged = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, rrf_score in merged[:top_k]:
        hit = dict(best_hit[cid])
        hit.pop("_rrf_rank", None)
        hit["rrf_score"] = round(rrf_score, 6)
        # Keep original retrieval score but annotate it
        hit["score"] = float(hit.get("score") or rrf_score)
        result.append(hit)
    return result


def retrieve_decomposed(
    query: str,
    retrieve_fn: Any,
    retrieve_kwargs: Dict[str, Any],
    *,
    top_k: int = 10,
) -> Optional[Dict[str, Any]]:
    """Run decomposed parallel retrieval and return a merged result dict.

    Parameters
    ----------
    query           : original user query
    retrieve_fn     : callable matching the signature of ``retrieve_as_dict``
    retrieve_kwargs : keyword args to pass through to each retrieve call
    top_k           : number of hits to return in the merged result

    Returns ``None`` when the query does not decompose (caller should fall back
    to normal single-query retrieval).
    """
    topics = extract_sub_topics(query)
    if len(topics) < 2:
        return None

    sub_queries = [_build_sub_query(query, t, all_topics=topics) for t in topics]
    logger.info(
        "query_decompose: decomposed %r into %d sub-queries: %s",
        query,
        len(sub_queries),
        sub_queries,
    )

    # ── Parallel retrieval ────────────────────────────────────────────────────
    sub_results: List[Optional[Dict[str, Any]]] = [None] * len(sub_queries)
    errors: List[Optional[Exception]] = [None] * len(sub_queries)

    def _run(idx: int, sq: str) -> None:
        try:
            kw = dict(retrieve_kwargs)
            # Use a larger candidate pool per sub-query so RRF has enough to work with
            kw["top_k"] = max(int(top_k), 10)
            sub_results[idx] = retrieve_fn(sq, **kw)
        except Exception as exc:  # noqa: BLE001
            errors[idx] = exc
            logger.warning("query_decompose: sub-query %r failed: %s", sq, exc)

    threads = [threading.Thread(target=_run, args=(i, sq)) for i, sq in enumerate(sub_queries)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    # Filter out failed retrievals
    valid_results = [r for r in sub_results if r is not None]
    if not valid_results:
        logger.warning("query_decompose: all sub-queries failed, falling back to single-query")
        return None

    # ── RRF merge ─────────────────────────────────────────────────────────────
    hit_lists = [r.get("hits") or [] for r in valid_results]
    merged_hits = rrf_merge(hit_lists, top_k=top_k)

    # Build a merged result dict using the first valid result as the base
    base = dict(valid_results[0])
    base["hits"] = merged_hits
    base["query"] = query
    base["top_k"] = top_k
    base["decomposition"] = {
        "sub_topics": topics,
        "sub_queries": sub_queries,
        "sub_result_count": len(valid_results),
        "pre_merge_hit_counts": [len(hl) for hl in hit_lists],
        "merged_hit_count": len(merged_hits),
    }
    # Merge internet_fallback flags from all sub-results
    internet_flags = [r.get("internet_fallback") for r in valid_results if r.get("internet_fallback")]
    if internet_flags:
        base["internet_fallback"] = internet_flags[0]

    return base
