"""
Query router v3 — multi-label intent detection.

Change from router2
-------------------
The NLI classifier now returns a *secondary* intent when the second-ranked
label score is within 0.15 of the top score.  This lets the router detect
compound queries such as "compare the formulas for X and Y" and adjust
retrieval mechanics without changing the primary intent label.

Specific rule added to route_query():
  - If secondary_intent == "formula_lookup", disable HyDE (use_hyde=False)
    and record the secondary in meta.  The formula_lookup strategy already
    favours exact-match with prefer_shorter=True; the HyDE step would
    generate a prose passage that drifts away from the exact formula text.

RoutedQuery changes:
  - meta["secondary_intent"] (str | None) — secondary label when present.
  - meta["use_hyde"] (bool) — False when formula_lookup secondary detected.

All other public symbols and behaviour are identical to router2.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.vector_ops import cosine as _cosine
from retrieval.router_scope import (
    classify_source_type,
    classify_collection_from_query,
    detect_book_scope,
    _STRONG_TEMPORAL_RE,
    _LIVE_LOOKUP_RE,
    _CHAT_OPENER_RE,
    _NON_FACTUAL_INTENTS,
)
from utils.runtime_defaults import (
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_ROUTING_TRACE_BACKUPS,
    DEFAULT_ROUTING_TRACE_MAX_MB,
    DEFAULT_ROUTING_TRACE_PATH,
    RUNTIME_DEFAULTS,
)

# ── Example-based routing ──────────────────────────────────────────────────────

_EXAMPLES_PATH = Path(__file__).parent.parent / "data" / "intent_examples.json"
_EXAMPLE_BLEND_NLI = 0.6   # weight for soft NLI score
_EXAMPLE_BLEND_SIM = 0.4   # weight for cosine similarity vote

# CDI score threshold: read from scoring.yaml routing.cdi_threshold (fallback 0.15)
_CDI_THRESHOLD: float = float(
    (RUNTIME_DEFAULTS.get("scoring", {}).get("routing") or {}).get("cdi_threshold", 0.15)
)

_example_embedder = None
_example_embedder_lock = threading.Lock()
_example_cache: Optional[Dict[str, List[List[float]]]] = None
_example_cache_lock = threading.Lock()


def _get_example_embedder():
    """Lazy singleton embedder for example-based routing."""
    global _example_embedder
    if _example_embedder is not None:
        return _example_embedder
    with _example_embedder_lock:
        if _example_embedder is not None:
            return _example_embedder
        try:
            from pipeline.embed.embedder import create_embedder  # noqa: PLC0415
            _example_embedder = create_embedder(
                backend=DEFAULT_EMBED_BACKEND,
                model_name=DEFAULT_EMBED_MODEL_NAME,
                ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
                ollama_timeout_seconds=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
            )
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger(__name__).warning(
                "router2: example embedder unavailable: %s", _exc
            )
    return _example_embedder


def _get_example_vecs() -> Optional[Dict[str, List[List[float]]]]:
    """Lazily load and cache embeddings for all intent examples."""
    global _example_cache
    if _example_cache is not None:
        return _example_cache
    with _example_cache_lock:
        if _example_cache is not None:
            return _example_cache
        if not _EXAMPLES_PATH.exists():
            return None
        try:
            data = json.loads(_EXAMPLES_PATH.read_text(encoding="utf-8"))
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger(__name__).warning(
                "router2: failed to load intent_examples.json: %s", _exc
            )
            return None
        embedder = _get_example_embedder()
        if embedder is None:
            return None
        try:
            cache: Dict[str, List[List[float]]] = {}
            for label, texts in data.items():
                if texts:
                    cache[label] = embedder.embed_texts([str(t) for t in texts])
            _example_cache = cache
        except Exception as _exc:  # noqa: BLE001
            import logging as _logging  # noqa: PLC0415
            _logging.getLogger(__name__).warning(
                "router2: failed to embed intent examples: %s", _exc
            )
            return None
    return _example_cache




def _example_vote_scores(query_vec: List[float]) -> Dict[str, float]:
    """
    Per-intent score based on max cosine similarity to example queries.

    Returns ``{intent: score}`` normalised so the highest score is 1.0.
    Returns an empty dict when no examples are available.
    """
    vecs = _get_example_vecs()
    if not vecs:
        return {}
    raw: Dict[str, float] = {
        label: max(_cosine(query_vec, ev) for ev in example_vecs)
        for label, example_vecs in vecs.items()
        if example_vecs
    }
    max_score = max(raw.values()) if raw else 0.0
    if max_score <= 0.0:
        return raw
    return {label: score / max_score for label, score in raw.items()}


# ── Intent labels ──────────────────────────────────────────────────────────────
INTENTS = frozenset(
    {
        "metadata_lookup",
        "formula_lookup",
        "section_lookup",
        "comparison",
        "list_lookup",
        "summary",
        "fact_lookup",
        "exploratory",
        "conversational_meta",
        "conversational",
        "user_profile",
        "current_data_lookup",
        "implicit_followup",
    }
)


# (source-type, temporal, book-scope, and collection-scope are in router_scope.py)


# ── Retrieval strategy ─────────────────────────────────────────────────────────

@dataclass
class RetrievalStrategy:
    """Parameters forwarded directly to retrieve()."""

    top_k: int = 5
    candidate_k: int = 40
    alpha_vector: float = 0.68
    alpha_lexical: float = 0.32
    prefer_tables: bool = False
    prefer_shorter: bool = False
    path_prefix: Optional[str] = None
    skip_retrieval: bool = False
    notes: str = ""


_STRATEGY_MAP: Dict[str, RetrievalStrategy] = {
    "metadata_lookup": RetrievalStrategy(
        top_k=5, candidate_k=20,
        alpha_vector=0.58, alpha_lexical=0.42,
        notes="metadata structural-role weighting applied downstream",
    ),
    "formula_lookup": RetrievalStrategy(
        top_k=3, candidate_k=20,
        alpha_vector=0.72, alpha_lexical=0.28,
        prefer_shorter=True,
        notes="tight precision; answer policy checks for math content",
    ),
    "section_lookup": RetrievalStrategy(
        top_k=8, candidate_k=40,
        alpha_vector=0.65, alpha_lexical=0.24,
        notes="rerank header bonus handles section alignment",
    ),
    "comparison": RetrievalStrategy(
        top_k=10, candidate_k=60,
        alpha_vector=0.65, alpha_lexical=0.24,
    ),
    "list_lookup": RetrievalStrategy(
        top_k=10, candidate_k=60,
        alpha_vector=0.65, alpha_lexical=0.24,
    ),
    "summary": RetrievalStrategy(
        top_k=12, candidate_k=70,
        alpha_vector=0.62, alpha_lexical=0.24,
    ),
    "fact_lookup": RetrievalStrategy(
        top_k=10, candidate_k=50,
        alpha_vector=0.68, alpha_lexical=0.24,
    ),
    "exploratory": RetrievalStrategy(
        top_k=10, candidate_k=60,
        alpha_vector=0.62, alpha_lexical=0.24,
    ),
    "conversational_meta": RetrievalStrategy(
        skip_retrieval=True,
        notes="conversation history query — no corpus retrieval needed",
    ),
    "conversational": RetrievalStrategy(
        skip_retrieval=True,
        notes="greeting or small-talk — answer from LLM directly",
    ),
    "user_profile": RetrievalStrategy(
        skip_retrieval=True,
        notes="user profile / personal context question — answered from system context",
    ),
    "current_data_lookup": RetrievalStrategy(
        top_k=5, candidate_k=40,
        alpha_vector=0.68, alpha_lexical=0.32,
        notes="requires live data — routing will send to web",
    ),
    "implicit_followup": RetrievalStrategy(
        top_k=5, candidate_k=40,
        alpha_vector=0.68, alpha_lexical=0.32,
        notes="follow-up referencing prior context — query resolver injects entity",
    ),
}


def _apply_intent_alpha_overrides() -> None:
    _intent_alphas: Dict[str, Any] = (
        RUNTIME_DEFAULTS.get("scoring", {}).get("intent_alphas") or {}
    )
    for intent, blends in _intent_alphas.items():
        if intent not in _STRATEGY_MAP:
            continue
        if not isinstance(blends, dict):
            continue
        strat = _STRATEGY_MAP[intent]
        if "alpha_vector" in blends:
            strat.alpha_vector = float(blends["alpha_vector"])
        if "alpha_lexical" in blends:
            strat.alpha_lexical = float(blends["alpha_lexical"])

_apply_intent_alpha_overrides()


# ── Routed query result ────────────────────────────────────────────────────────

@dataclass
class RoutedQuery:
    original_query: str
    intent: str
    sources: List[str] = field(default_factory=lambda: ["corpus"])
    strategy: RetrievalStrategy = field(default_factory=RetrievalStrategy)
    source_type_filter: Optional[str] = None
    collection_id: Optional[str] = None
    rewritten_query: Optional[str] = None
    doc_ids: Optional[List[str]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_query(self) -> str:
        """The query to use for retrieval — stripped of any scoping phrase."""
        return self.rewritten_query or self.original_query


# ── Temporal / freshness decision ─────────────────────────────────────────────

def _needs_web(
    query: str,
    *,
    intent: Optional[str] = None,
    secondary_intent: Optional[str] = None,
    nli_scores: Optional[Dict[str, float]] = None,
) -> Tuple[bool, Optional[str]]:
    """Return ``(True, reason)`` if this query requires live web information.

    When called from the routing pipeline, pass ``intent`` and
    ``secondary_intent`` (already computed) so no extra model call is needed.
    When called standalone (e.g. tests or scripts), omits intent and falls back
    to a quick NLI check via the cached DeBERTa pipeline.
    """
    q = (query or "").strip()
    if not q:
        return False, None

    if intent in ("conversational", "user_profile") and secondary_intent is None:
        return False, None

    if _STRONG_TEMPORAL_RE.search(q):
        return True, "strong_temporal"

    if _LIVE_LOOKUP_RE.search(q):
        return True, "live_lookup"

    if intent == "current_data_lookup" or secondary_intent == "current_data_lookup":
        return True, "current_data_lookup"

    if intent is not None and intent not in _NON_FACTUAL_INTENTS:
        cdi_score = (nli_scores or {}).get("current_data_lookup", 0.0)
        if cdi_score > _CDI_THRESHOLD:
            return True, "current_data_lookup"
        return False, None

    if intent is not None:
        return False, None

    if _CHAT_OPENER_RE.match(q):
        return False, None

    try:
        _intent, _meta = classify_intent(q)
        _sec = _meta.get("secondary_intent")

        if _intent in ("conversational", "user_profile") and _sec != "current_data_lookup":
            return False, None

        if _intent == "current_data_lookup" or _sec == "current_data_lookup":
            return True, "current_data_lookup"

        if _intent not in _NON_FACTUAL_INTENTS:
            cdi_score = (_meta.get("nli_scores") or {}).get("current_data_lookup", 0.0)
            if cdi_score > _CDI_THRESHOLD:
                return True, "current_data_lookup"
    except Exception:
        pass

    return False, None


def _count_sentences(text: str) -> int:
    return max(1, len(re.split(r"[.!?]+\s+", text.strip())))


def classify_intent(query: str) -> Tuple[str, Dict[str, Any]]:
    """
    Classify query into one primary intent label plus an optional secondary.

    Uses ``classify_intent_full_scores()`` which returns
    ``(primary, secondary, confidence, scores_dict)``.

    High-confidence path (confidence >= threshold):
        Returns the NLI result unchanged.  Secondary intent recorded in meta.

    Low-confidence path (ml_fallback):
        Attempts example-based blending — embeds the query and computes cosine
        similarity to curated example queries for each intent.  The blended
        score is ``0.6 * nli_score + 0.4 * sim_score`` for every label and
        ``argmax`` selects the winner.  Falls back to structural heuristics
        when the embedder or example file is unavailable.

    Returns ``(intent_label, meta_dict)`` — same public signature as before.
    ``meta["secondary_intent"]`` carries the secondary label when present.
    """
    from retrieval.intent_classifier import classify_intent_full_scores  # noqa: PLC0415
    import logging as _logging  # noqa: PLC0415

    _log = _logging.getLogger(__name__)
    q = (query or "").strip()
    meta: Dict[str, Any] = {"query_words": len(q.split())}

    ml_primary, ml_secondary, ml_confidence, nli_scores = classify_intent_full_scores(q)

    if ml_primary is not None:
        # High-confidence NLI result — use it directly, skip example blending.
        meta["matched_pattern"] = "ml_zeroshot"
        meta["ml_confidence"] = ml_confidence
        meta["secondary_intent"] = ml_secondary
        meta["nli_scores"] = nli_scores or {}
        return ml_primary, meta

    # ── Low-confidence path ────────────────────────────────────────────────────
    meta["ml_confidence"] = ml_confidence
    meta["ml_fallback"] = True
    meta["secondary_intent"] = None
    meta["nli_scores"] = nli_scores or {}

    if nli_scores:
        embedder = _get_example_embedder()
        if embedder is not None:
            try:
                query_vec = embedder.embed_query(q)
                sim_scores = _example_vote_scores(query_vec)
                if sim_scores:
                    blended = {
                        label: (
                            _EXAMPLE_BLEND_NLI * nli_scores.get(label, 0.0)
                            + _EXAMPLE_BLEND_SIM * sim_scores.get(label, 0.0)
                        )
                        for label in INTENTS
                    }
                    best_label = max(blended, key=blended.__getitem__)
                    meta["matched_pattern"] = "example_blend"
                    meta["example_blend_winner"] = best_label
                    meta["example_blend_score"] = round(blended[best_label], 4)
                    return best_label, meta
            except Exception as _exc:  # noqa: BLE001
                _log.warning("router2: example blending failed: %s", _exc)

    # ── Structural fallback (no NLI scores or embedder unavailable) ────────────
    word_count = len(q.split())
    sentence_count = _count_sentences(q)

    if sentence_count >= 2 or word_count > 25:
        meta["matched_pattern"] = "exploratory_long"
        meta["sentence_count"] = sentence_count
        return "exploratory", meta

    meta["matched_pattern"] = "default_fact"
    return "fact_lookup", meta


# ── Main routing entry point ───────────────────────────────────────────────────

# Intents that should never be carried forward into a new turn — they have
# no retrieval semantics and would mis-steer an unrelated follow-up question.
_NO_CARRYFORWARD_INTENTS: frozenset = frozenset({
    "conversational",
    "conversational_meta",
})


def route_query(
    query: str,
    available_sources: Optional[List[str]] = None,
    *,
    db_dsn: Optional[str] = None,
    prior_intents: Optional[List[str]] = None,
) -> RoutedQuery:
    """
    Classify intent, detect source requirements, and select retrieval strategy.

    v3 additions vs router2:
    - meta["secondary_intent"] — second-ranked NLI label when within 0.15 gap.
    - meta["use_hyde"] — set to False when secondary_intent == "formula_lookup"
      (HyDE generates prose that drifts from exact formula notation).

    Parameters
    ----------
    prior_intents : optional list of intent labels from recent turns (most
        recent last).  When the NLI classifier fires a low-confidence heuristic
        fallback, the last non-conversational prior intent is inherited and
        meta["intent_carry_forward"] is set to True.  This prevents ambiguous
        follow-up questions (e.g. "and the second one?") from being mis-routed.
    """
    sources = list(available_sources or ["corpus"])
    q = (query or "").strip()

    # ── Step 1: Explicit source-type detection ─────────────────────────────────
    source_type_filter = classify_source_type(query)

    # ── Step 2: NLI intent classifier ─────────────────────────────────────────
    intent, intent_meta = classify_intent(q)
    secondary_intent: Optional[str] = intent_meta.get("secondary_intent")

    # ── Step 2a: Conversation-aware carry-forward ──────────────────────────────
    # When the NLI model fires a low-confidence heuristic fallback AND the
    # caller provides prior-turn intents, inherit the most recent
    # non-conversational intent from the history instead.  This handles
    # ambiguous follow-ups like "what about the second one?" or "can you
    # elaborate?" that have no strong classification signal on their own.
    intent_carry_forward = False
    if intent_meta.get("ml_fallback") and prior_intents:
        inherited = next(
            (i for i in reversed(prior_intents) if i not in _NO_CARRYFORWARD_INTENTS),
            None,
        )
        if inherited is not None and inherited in INTENTS:
            intent = inherited
            intent_carry_forward = True
            intent_meta["carry_forward_from"] = inherited

    # ── Step 2b: secondary_intent adjustments ─────────────────────────────────
    # When formula_lookup appears as a secondary intent alongside any
    # primary, disable HyDE — prose generation drifts from exact notation.
    use_hyde: bool = True
    if secondary_intent == "formula_lookup":
        use_hyde = False

    # ── Step 3: Temporal/freshness analysis ───────────────────────────────────
    if source_type_filter == "internet":
        needs_web = True
        needs_web_reason: Optional[str] = "explicit_source_type"
    else:
        needs_web, needs_web_reason = _needs_web(
            q,
            intent=intent,
            secondary_intent=secondary_intent,
            nli_scores=intent_meta.get("nli_scores"),
        )

    # ── Step 4: Collection-scope detection ────────────────────────────────────
    collection_id, rewritten_query = classify_collection_from_query(query, db_dsn)

    # ── Step 4b: Source-book scope detection ──────────────────────────────────
    # Detect "According to [Book]," style phrasing and resolve the named book
    # to its doc_id(s).  Only applies when collection scope is not already set.
    # When a book is matched the scoping preamble is also stripped from the
    # rewritten query so the embedding search targets the core topic.
    scoped_doc_ids: Optional[List[str]] = None
    if not collection_id and db_dsn:
        _scope_ids, _scope_rewrite = detect_book_scope(query, db_dsn)
        if _scope_ids:
            scoped_doc_ids = _scope_ids
            # Only override rewritten_query when collection detection didn't
            # already produce one.
            if not rewritten_query:
                rewritten_query = _scope_rewrite

    strategy = _STRATEGY_MAP.get(intent, RetrievalStrategy())

    if collection_id and needs_web:
        needs_web = False
        needs_web_reason = "overridden_by_collection_scope"

    # Book-scope also suppresses internet fallback — user explicitly asked
    # about a corpus document, so web results would violate the scope.
    if scoped_doc_ids and needs_web:
        needs_web = False
        needs_web_reason = "overridden_by_book_scope"

    effective_sources = list(sources)
    routing_meta: Dict[str, Any] = {
        "intent_classification": intent_meta,
        "secondary_intent": secondary_intent,
        "use_hyde": use_hyde,
        "needs_web": needs_web,
        "needs_web_reason": needs_web_reason,
        "routing_authority": "v2_heuristic",
        "source_type_filter": source_type_filter,
        "collection_id": collection_id,
        "rewritten_query": rewritten_query,
        "router_version": "router2",
        "intent_carry_forward": intent_carry_forward,
        "scoped_doc_ids": scoped_doc_ids,
    }

    _append_routing_trace(
        query=query,
        intent=intent,
        secondary_intent=secondary_intent,
        needs_web=needs_web,
        needs_web_reason=needs_web_reason,
        selected_sources=effective_sources,
        intent_meta=intent_meta,
    )

    return RoutedQuery(
        original_query=query,
        intent=intent,
        sources=effective_sources,
        strategy=strategy,
        source_type_filter=source_type_filter,
        collection_id=collection_id,
        rewritten_query=rewritten_query,
        doc_ids=scoped_doc_ids,
        meta=routing_meta,
    )


# ── Routing trace ──────────────────────────────────────────────────────────────

def _append_routing_trace(
    *,
    query: str,
    intent: str,
    secondary_intent: Optional[str],
    needs_web: bool,
    needs_web_reason: Optional[str],
    selected_sources: List[str],
    intent_meta: Dict[str, Any],
    trace_path: str = DEFAULT_ROUTING_TRACE_PATH,
) -> None:
    """Append a single routing decision to the rolling JSONL trace file."""
    max_bytes = max(1, int(DEFAULT_ROUTING_TRACE_MAX_MB)) * 1024 * 1024
    backups = max(1, int(DEFAULT_ROUTING_TRACE_BACKUPS))

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "router_version": "router2",
        "query": query,
        "intent": intent,
        "secondary_intent": secondary_intent,
        "needs_web": needs_web,
        "needs_web_reason": needs_web_reason,
        "matched_pattern": intent_meta.get("matched_pattern"),
        "ml_confidence": intent_meta.get("ml_confidence"),
        "selected_sources": selected_sources,
    }

    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size >= max_bytes:
        for idx in range(backups, 0, -1):
            older = path.with_suffix(path.suffix + f".{idx}")
            if idx == backups and older.exists():
                older.unlink()
            prev = path.with_suffix(path.suffix + f".{idx - 1}") if idx > 1 else path
            if prev.exists():
                os.replace(prev, older)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
