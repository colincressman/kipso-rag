"""RAG + LLM public API — importable entry point for both pipelines.

Both pipelines import from here rather than calling subprocesses or CLI scripts.

    from api import rag_retrieve, llm_answer

Scorecard pipeline (retrieve + answer per field):
    result  = rag_retrieve("query", top_k=5, source_type="spec")
    answer  = llm_answer("query", result)
    text    = answer["answer"]
    sources = answer["citations"]

Spec sieve pipeline (retrieve only, hand off to 270B):
    result  = rag_retrieve("safety-critical requirements", top_k=50, source_type="spec")
    hits    = result["hits"]   # list of dicts — feed straight to 270B model
    routing = result["routing"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.query import RetrievalFilters, retrieve_as_dict
from retrieval.query_decompose import retrieve_decomposed
from retrieval.rag_fusion import retrieve_with_fusion, _FUSION_INTENTS
from retrieval.context_pack import build_context_pack
from retrieval.router import route_query
from llm.answer import answer_query_with_retrieval
from db.client import init_db as _init_db
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
    DEFAULT_RETRIEVAL_TOP_K,
)


def rag_retrieve(
    query: str,
    *,
    # Core options
    top_k: int = DEFAULT_RETRIEVAL_TOP_K,
    db_dsn: str = DEFAULT_DB_DSN,
    # Filtering
    source_type: Optional[str] = None,
    doc_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    path_prefix: Optional[str] = None,
    min_page: Optional[int] = None,
    max_page: Optional[int] = None,
    has_table: Optional[bool] = None,
    structural_role: Optional[str] = None,
    collection: Optional[str] = None,
    # Embedding
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    # Reranking
    cross_encoder_enabled: Optional[bool] = None,
    cross_encoder_only: Optional[bool] = None,
    # Routing overrides (leave None to let the router decide)
    force_source_type: Optional[str] = None,
    # Conversation carry-forward — last N intent labels from prior turns.
    # When the NLI classifier fires a low-confidence fallback, the most
    # recent non-conversational prior intent is inherited.
    prior_intents: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full retrieval pipeline for a query and return a result dict.

    Parameters
    ----------
    query            : natural language query string
    top_k            : maximum number of hits to return
    source_type      : filter chunks by source_type ('notes', 'pdf_book', 'spec', …).
                       Auto-detected from the query if omitted.
    force_source_type: override auto-detection and always apply this filter.
                       Takes priority over `source_type`.

    Returns
    -------
    dict with keys:
        "hits"      — ranked list of chunk dicts (text, score, metadata, …)
        "routing"   — intent, source_type_filter, strategy used
        "context_pack" — deduplicated, authority-weighted selection for LLM use
    """
    routed = route_query(query, db_dsn=db_dsn, prior_intents=prior_intents or None)
    strategy = routed.strategy

    effective_source_type = force_source_type or source_type or routed.source_type_filter
    effective_collection = collection or routed.collection_id

    # When the router detected an explicit "According to [Book]" scope, use
    # those doc_ids as a filter — unless the caller already supplied their own.
    effective_doc_ids = doc_ids or routed.doc_ids or None

    filters = RetrievalFilters(
        doc_id=doc_id,
        doc_ids=effective_doc_ids,
        path_prefix=path_prefix,
        min_page=min_page,
        max_page=max_page,
        has_table=has_table,
        source_type=effective_source_type,
        structural_role=structural_role,
        collection_id=effective_collection,
    )

    hard_filters_set = any([doc_id, doc_ids, path_prefix, min_page is not None,
                             max_page is not None, has_table])

    retrieve_kwargs: Dict[str, Any] = {
        "db_dsn": db_dsn,
        "filters": filters,
        "embed_backend": embed_backend,
        "embed_model_name": embed_model_name,
    }

    if hard_filters_set:
        retrieve_kwargs["top_k"] = top_k
    else:
        runtime_top_k = max(top_k, int(strategy.top_k))
        runtime_candidate_k = max(runtime_top_k, int(strategy.candidate_k))
        retrieve_kwargs.update({
            "top_k": runtime_top_k,
            "rerank_candidate_k": runtime_candidate_k,
            "rerank_alpha_vector": float(strategy.alpha_vector),
            "rerank_alpha_lexical": float(strategy.alpha_lexical),
            "rerank_prefer_tables": bool(strategy.prefer_tables),
            "rerank_prefer_shorter": bool(strategy.prefer_shorter),
        })

    if cross_encoder_enabled is not None:
        retrieve_kwargs["cross_encoder_enabled"] = cross_encoder_enabled
    if cross_encoder_only is not None:
        retrieve_kwargs["cross_encoder_only"] = cross_encoder_only

    retrieve_kwargs["intent"] = routed.intent
    retrieve_kwargs["needs_web"] = bool((routed.meta or {}).get("needs_web", False))
    # Honour the router's HyDE suppression signal (set when secondary_intent ==
    # "formula_lookup" — HyDE prose drifts from exact formula notation).
    if not (routed.meta or {}).get("use_hyde", True):
        retrieve_kwargs["hyde_enabled"] = False
    effective_query = routed.effective_query

    # ── Query decomposition for comparison / multi-topic queries ──────────────
    # When the router signals a comparison intent and the query mentions two or
    # more distinct items, run one retrieve() call per sub-topic in parallel
    # and merge the results via RRF so each topic gets dedicated coverage.
    _decomp_result: Dict[str, Any] | None = None
    if routed.intent == "comparison":
        _decomp_result = retrieve_decomposed(
            effective_query,
            retrieve_as_dict,
            retrieve_kwargs,
            top_k=max(top_k, int(strategy.top_k)),
        )

    # ── RAG Fusion for single-topic queries (all other qualifying intents) ────
    # Generates N paraphrase variants, retrieves in parallel, RRF-merges.
    # Skipped when query decomposition already ran (comparison intents).
    _fusion_result: Dict[str, Any] | None = None
    if _decomp_result is None and routed.intent in _FUSION_INTENTS:
        from utils.runtime_defaults import (
            DEFAULT_HYDE_BASE_URL,
            DEFAULT_HYDE_MODEL,
            DEFAULT_HYDE_TIMEOUT_SECONDS,
        )
        _fusion_result = retrieve_with_fusion(
            effective_query,
            retrieve_as_dict,
            retrieve_kwargs,
            intent=routed.intent,
            top_k=max(top_k, int(strategy.top_k)),
            model=DEFAULT_HYDE_MODEL,
            base_url=DEFAULT_HYDE_BASE_URL,
            timeout_seconds=DEFAULT_HYDE_TIMEOUT_SECONDS,
        )

    if _decomp_result is not None:
        result = _decomp_result
        result.setdefault("routing", {})["decomposition_applied"] = True
        result.setdefault("routing", {})["sub_queries"] = (
            result.get("decomposition", {}).get("sub_queries", [])
        )
    elif _fusion_result is not None:
        result = _fusion_result
    else:
        result = retrieve_as_dict(effective_query, **retrieve_kwargs)

    # ── Summary boost: for summary-intent queries, prepend the document_summary
    # chunk(s) so they always appear at the top of the context, regardless of
    # whether they won the embedding race against body chunks.
    if routed.intent == "summary":
        import psycopg
        from psycopg.rows import dict_row
        _db = retrieve_kwargs.get("db_dsn", db_dsn)
        try:
            _init_db(_db)
            _conn = psycopg.connect(_db, row_factory=dict_row)
            # Scope to the same doc_ids / collection as the retrieval.
            _where_parts = ["structural_role = 'document_summary'"]
            _args: list = []
            _scope_doc_ids = doc_ids or ([doc_id] if doc_id else None)
            if _scope_doc_ids:
                _ph = ",".join("%s" for _ in _scope_doc_ids)
                _where_parts.append(f"doc_id IN ({_ph})")
                _args.extend(_scope_doc_ids)
            elif effective_collection:
                _where_parts.append("collection_id = %s")
                _args.append(effective_collection)
            _where_sql = " AND ".join(_where_parts)
            _rows = _conn.execute(
                f"SELECT chunk_id, doc_id, collection_id, source_name, document_title, "
                f"document_path, structural_role, text, token_count_est, source_type "
                f"FROM chunks WHERE {_where_sql} ORDER BY document_title",
                _args,
            ).fetchall()
            _conn.close()
            _summary_hits = [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "collection_id": r["collection_id"],
                    "source_name": r["source_name"],
                    "document_title": r["document_title"],
                    "document_path": r["document_path"],
                    "structural_role": r["structural_role"],
                    "text": r["text"] or "",
                    "token_count_est": r["token_count_est"],
                    "source_type": r["source_type"],
                    "score": 1.0,
                    "path_text": "Document Summary",
                    "page_start": None,
                    "page_end": None,
                    "level": 0,
                    "title": "Document Summary",
                }
                for r in _rows
            ]
            if _summary_hits:
                existing_ids = {h.get("chunk_id") for h in result.get("hits", [])}
                new_hits = [h for h in _summary_hits if h["chunk_id"] not in existing_ids]
                result["hits"] = new_hits + (result.get("hits") or [])
        except Exception:
            pass  # never break retrieval due to summary boost failure

    context_pack = build_context_pack(result, routed, max_chunks=top_k)
    result["context_pack"] = context_pack

    # ── CRAG: Corrective-RAG — if retrieval quality is low, fall back to web ─
    # Requires cross-encoder to be enabled so scores exist on hits.
    # Skipped when internet was already triggered by the router/retrieval.
    try:
        from utils.config import load_yaml_config  # noqa: PLC0415, F401
        import yaml as _yaml  # noqa: PLC0415
        _llm_cfg_path = Path(__file__).parent / "configs" / "llm.yaml"
        _llm_cfg = _yaml.safe_load(_llm_cfg_path.read_text(encoding="utf-8")) or {} if _llm_cfg_path.exists() else {}
        _crag_threshold = float((_llm_cfg.get("crag") or {}).get("min_relevance_score", 0.0))
        if _crag_threshold > 0:
            _already_web = bool((result.get("routing") or {}).get("internet_triggered"))
            _hits = result.get("hits") or []
            _max_ce = max(
                (float(h.get("cross_encoder_score") or 0.0) for h in _hits),
                default=0.0,
            )
            if not _already_web and _max_ce < _crag_threshold:
                # Re-run retrieval forcing internet fallback
                _crag_kwargs = dict(retrieve_kwargs)
                _crag_kwargs["needs_web"] = True
                _crag_result = retrieve_as_dict(effective_query, **_crag_kwargs)
                # Merge web hits after existing hits (deduplication by chunk_id)
                _existing_ids = {h.get("chunk_id") for h in _hits}
                _web_hits = [h for h in (_crag_result.get("hits") or [])
                             if h.get("chunk_id") not in _existing_ids]
                result["hits"] = _hits + _web_hits
                result.setdefault("routing", {})["crag_triggered"] = True
                result.setdefault("routing", {})["crag_max_ce_score"] = _max_ce
                # Rebuild context pack with merged hits
                from retrieval.context_pack import build_context_pack as _bcp  # noqa: PLC0415
                result["context_pack"] = _bcp(result, routed, max_chunks=top_k)
    except Exception:
        pass  # CRAG is best-effort; never break retrieval

    # Expose routing metadata under a clean top-level key.
    _rag_fusion_meta = (result.get("rag_fusion") or {})
    result["routing"] = {
        "intent": routed.intent,
        "source_type_filter": effective_source_type,
        "collection_id": effective_collection,
        "rewritten_query": routed.rewritten_query,
        "strategy": routed.strategy.route_type if hasattr(routed.strategy, "route_type") else None,
        "rag_fusion_applied": bool(_rag_fusion_meta),
        "variant_count": len(_rag_fusion_meta.get("all_queries", [])) if _rag_fusion_meta else 0,
        "decomposition_applied": bool(_decomp_result is not None),
    }

    return result


def llm_answer(
    query: str,
    retrieval_result: Dict[str, Any],
    *,
    history: Optional[List[Dict[str, str]]] = None,
    llm_model: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    llm_timeout_seconds: Optional[float] = None,
    llm_temperature: Optional[float] = None,
    config_path: str = "configs/llm.yaml",
) -> Dict[str, Any]:
    """Generate a grounded answer from a retrieval result.

    Parameters
    ----------
    query            : the original query string (must match rag_retrieve call)
    retrieval_result : dict returned by rag_retrieve()
    llm_model        : override the model name (default: from configs/llm.yaml)
    llm_base_url     : override the Ollama base URL
    llm_timeout_seconds : override per-call timeout
    config_path      : path to llm.yaml config file

    Returns
    -------
    dict with keys:
        "answer"    — grounded answer text with inline citations
        "citations" — list of chunk IDs cited
        "grounded"  — bool if answer was verified against corpus; None in
                      high_confidence mode (grounding check is skipped — this
                      is correct by design, not a missing value)
        "mode"      — confidence band: high_confidence / medium_confidence /
                      low_confidence / no_coverage
    """
    routing = retrieval_result.get("routing", {})
    intent: Optional[str] = routing.get("intent")

    context_pack = retrieval_result.get("context_pack", {})
    answer_input = {
        **retrieval_result,
        "hits": context_pack.get("selected_chunks", retrieval_result.get("hits", [])),
    }

    answer = answer_query_with_retrieval(
        query,
        answer_input,
        history=history,
        intent=intent,
        llm_model=llm_model,
        llm_base_url=llm_base_url,
        llm_timeout_seconds=llm_timeout_seconds,
        llm_temperature=llm_temperature,
        config_path=config_path,
    )
    answer["context_pack"] = context_pack
    if "internet_fallback" in retrieval_result:
        answer["internet_fallback"] = retrieval_result["internet_fallback"]

    return answer
