"""Vector retrieval over PostgreSQL/pgvector-stored chunks.

Stage 7 responsibilities:
  1) Embed user query
  2) Similarity search against stored vectors
  3) Optional metadata filters
  4) Return top chunks with contextual neighbors

Implementation is split across focused sub-modules:
  retrieval.models           — RetrievalFilters, RetrievedChunk, RetrievalResult
  retrieval.chunk_normalize  — PDF-artifact normalisation
  retrieval.acronym_expand   — corpus-mined acronym expansion
  retrieval.candidate_scoring — query classification + scoring helpers
  retrieval.db_search        — DB connection, SQL query helpers, vector ANN
  retrieval.lexical_search   — BM25 index + warm_bm25_index
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

from pipeline.embed.embedder import create_embedder
from retrieval.acronym_expand import _expand_query_with_acronyms
from retrieval.candidate_scoring import (
	_is_external_fact_query,
	_lexical_candidate_score,
	_query_overlap_bonus,
	_structural_role_score,
)
from retrieval.chunk_normalize import normalize_chunk_text
from retrieval.cross_encoder import rerank_with_cross_encoder
from retrieval.db_search import (
	_chunk_neighbors,
	_connect,
	_format_document_path,
	_text_rows,
	_vector_candidates,
)
from retrieval.hyde import generate_hyde_query, generate_stepback_query
from retrieval.internet_fallback import retrieve_internet_chunks
from retrieval.lexical_search import _bm25_scores, warm_bm25_index
from retrieval.models import (
	CHUNK_ID_RE,
	RetrievalFilters,
	RetrievedChunk,
	RetrievalResult,
)
from retrieval.rerank import rerank_by_query
from utils.book_registry import normalize_document_registry_entry
from utils.runtime_defaults import (
	DEFAULT_BM25_ENABLED,
	DEFAULT_BM25_RRF_K,
	DEFAULT_DB_DSN,
	DEFAULT_EMBED_BACKEND,
	DEFAULT_EMBED_DIMENSION,
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_HYDE_BASE_URL,
	DEFAULT_HYDE_ENABLED,
	DEFAULT_HYDE_MODEL,
	DEFAULT_HYDE_TEMPERATURE,
	DEFAULT_HYDE_TIMEOUT_SECONDS,
	DEFAULT_INTERNET_FALLBACK_ENABLED,
	DEFAULT_INTERNET_MAX_CHUNKS,
	DEFAULT_INTERNET_MAX_RESULTS,
	DEFAULT_INTERNET_MIN_RELEVANCE_SCORE,
	DEFAULT_INTERNET_OVERRIDE_GUARD_THRESHOLD,
	DEFAULT_INTERNET_SCORE_WEIGHT,
	DEFAULT_INTERNET_TIMEOUT_SECONDS,
	DEFAULT_INTERNET_TRIGGER_GAP,
	DEFAULT_INTERNET_TRIGGER_ON_LOW_CONFIDENCE,
	DEFAULT_INTERNET_TRIGGER_TOP_SCORE,
	DEFAULT_LOW_CONFIDENCE_GAP_THRESHOLD,
	DEFAULT_OLLAMA_BASE_URL,
	DEFAULT_OLLAMA_TIMEOUT_SECONDS,
	DEFAULT_PRF_ENABLED,
	DEFAULT_RERANK_ALPHA_LEXICAL,
	DEFAULT_RERANK_ALPHA_VECTOR,
	DEFAULT_RERANK_CANDIDATE_K,
	DEFAULT_RERANK_CROSS_ENCODER_MODEL,
	DEFAULT_RERANK_CROSS_ENCODER_TOP_N,
	DEFAULT_RERANK_CROSS_ENCODER_WEIGHT,
	DEFAULT_RERANK_DIVERSITY_PENALTY,
	DEFAULT_RERANK_USE_CROSS_ENCODER,
	DEFAULT_RERANK_USE_CROSS_ENCODER_ONLY,
	DEFAULT_RETRIEVAL_TOP_K,
	DEFAULT_STEPBACK_ENABLED,
	DEFAULT_STEPBACK_TIMEOUT_SECONDS,
	DEFAULT_TWO_STAGE_ALPHA,
	DEFAULT_TWO_STAGE_ENABLED,
	DEFAULT_HYP_QUESTIONS_ENABLED,
	DEFAULT_HYP_QUESTIONS_TOP_K,
)

# ---------------------------------------------------------------------------
# HyDE configuration
# ---------------------------------------------------------------------------

# Intents for which HyDE should be suppressed: precise lookups where a
# hypothetical passage adds noise rather than semantic lift.  Also skipped for
# short conversational / meta queries where the overhead isn't worth it.
_HYDE_SKIP_INTENTS: frozenset = frozenset({
    "metadata_lookup",
    "formula_lookup",
    "factoid_lookup",
    "conversational",
    "conversational_meta",
    "user_profile",
    "current_data_lookup",
    "greeting",
    "out_of_scope",
})

# Queries shorter than this word count also skip HyDE (too little context to
# generate a useful hypothetical passage).
_HYDE_MIN_WORDS: int = 6


# ---------------------------------------------------------------------------
# Main retrieval entry point
# ---------------------------------------------------------------------------

def retrieve(
	query: str,
	*,
	db_dsn: str = DEFAULT_DB_DSN,
	top_k: int = DEFAULT_RETRIEVAL_TOP_K,
	filters: Optional[RetrievalFilters] = None,
	include_neighbors: bool = True,
	neighbor_window: int = 1,
	rerank_enabled: bool = True,
	rerank_candidate_k: int = DEFAULT_RERANK_CANDIDATE_K,
	rerank_alpha_vector: float = DEFAULT_RERANK_ALPHA_VECTOR,
	rerank_alpha_lexical: float = DEFAULT_RERANK_ALPHA_LEXICAL,
	rerank_prefer_tables: bool = False,
	rerank_prefer_shorter: bool = False,
	cross_encoder_enabled: bool = DEFAULT_RERANK_USE_CROSS_ENCODER,
	cross_encoder_only: bool = DEFAULT_RERANK_USE_CROSS_ENCODER_ONLY,
	cross_encoder_model: str = DEFAULT_RERANK_CROSS_ENCODER_MODEL,
	cross_encoder_top_n: int = DEFAULT_RERANK_CROSS_ENCODER_TOP_N,
	cross_encoder_weight: float = DEFAULT_RERANK_CROSS_ENCODER_WEIGHT,
	low_confidence_gap_threshold: float = DEFAULT_LOW_CONFIDENCE_GAP_THRESHOLD,
	internet_fallback_enabled: bool = DEFAULT_INTERNET_FALLBACK_ENABLED,
	internet_trigger_on_low_confidence: bool = DEFAULT_INTERNET_TRIGGER_ON_LOW_CONFIDENCE,
	internet_trigger_top_score: float = DEFAULT_INTERNET_TRIGGER_TOP_SCORE,
	internet_trigger_gap: float = DEFAULT_INTERNET_TRIGGER_GAP,
	internet_max_results: int = DEFAULT_INTERNET_MAX_RESULTS,
	internet_max_chunks: int = DEFAULT_INTERNET_MAX_CHUNKS,
	internet_timeout_seconds: float = DEFAULT_INTERNET_TIMEOUT_SECONDS,
	internet_score_weight: float = DEFAULT_INTERNET_SCORE_WEIGHT,
	internet_override_guard_threshold: float = DEFAULT_INTERNET_OVERRIDE_GUARD_THRESHOLD,
	internet_min_relevance_score: float = DEFAULT_INTERNET_MIN_RELEVANCE_SCORE,
	embed_backend: str = DEFAULT_EMBED_BACKEND,
	embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
	embed_dimension: int = DEFAULT_EMBED_DIMENSION,
	ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
	ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
	docx_boost_k: int = 2,
	intent: Optional[str] = None,
	hyde_enabled: bool = DEFAULT_HYDE_ENABLED,
	hyde_model: str = DEFAULT_HYDE_MODEL,
	hyde_base_url: str = DEFAULT_HYDE_BASE_URL,
	hyde_temperature: float = DEFAULT_HYDE_TEMPERATURE,
	hyde_timeout_seconds: float = DEFAULT_HYDE_TIMEOUT_SECONDS,
	two_stage_enabled: bool = DEFAULT_TWO_STAGE_ENABLED,
	two_stage_alpha: float = DEFAULT_TWO_STAGE_ALPHA,
	bm25_enabled: bool = DEFAULT_BM25_ENABLED,
	bm25_rrf_k: int = DEFAULT_BM25_RRF_K,
	rerank_diversity_penalty: float = DEFAULT_RERANK_DIVERSITY_PENALTY,
	needs_web: bool = False,
	progress_fn: Optional[Callable[[str], None]] = None,
	stepback_enabled: bool = DEFAULT_STEPBACK_ENABLED,
	stepback_timeout_seconds: float = DEFAULT_STEPBACK_TIMEOUT_SECONDS,
	prf_enabled: bool = DEFAULT_PRF_ENABLED,
	hyp_questions_enabled: bool = DEFAULT_HYP_QUESTIONS_ENABLED,
	hyp_questions_top_k: int = DEFAULT_HYP_QUESTIONS_TOP_K,
) -> RetrievalResult:
	"""
	Retrieve top-k chunks for a user query.

	If ``hyde_enabled`` is True, a small LLM call generates a hypothetical
	answer passage which is embedded in place of the raw question.  On any
	LLM failure the original query is used transparently.

	If ``two_stage_enabled`` is True *and* HyDE was successfully applied, a
	second embedding of the original (acronym-expanded) query is computed and
	blended with the HyDE vector score:

	    final_vector_score = alpha * cosine(hyde_vec, chunk)
	                       + (1 - alpha) * cosine(orig_vec, chunk)

	This ensures chunks that match the *question wording* (not just the
	hypothetical answer) are still surfaced, improving recall for precise
	keyword queries inside long documents.

	Returns ranked chunks with optional neighboring chunk context.
	"""
	if not query.strip():
		return RetrievalResult(query=query, top_k=top_k, filters={}, hits=[])

	_pg = progress_fn or (lambda _: None)
	f = filters or RetrievalFilters()
	_t0 = time.perf_counter()
	_perf_marks: List[tuple] = []

	# ── HyDE: generate hypothetical passage to use as the embed query ─────────
	_hyde_trace: Optional[Dict[str, Any]] = None
	embed_query_text = query
	_effective_hyde = (
		hyde_enabled
		and intent not in _HYDE_SKIP_INTENTS
		and len(query.split()) >= _HYDE_MIN_WORDS
	)
	if _effective_hyde and not cross_encoder_only:
		_pg("  → HyDE: generating passage…")
		# Prefer the remote Ollama URL (same host as inference service) over the
		# static config value so discovery keeps HyDE and HF models in sync.
		_hyde_url = hyde_base_url
		try:
			from utils.service_discovery import get_remote_ollama_url  # noqa: PLC0415
			_remote_ol = get_remote_ollama_url()
			if _remote_ol:
				_hyde_url = _remote_ol
		except Exception:
			pass
		embed_query_text, _hyde_trace = generate_hyde_query(
			query,
			model=hyde_model,
			base_url=_hyde_url,
			timeout_seconds=hyde_timeout_seconds,
			temperature=hyde_temperature,
		)
		_perf_marks.append(("hyde", time.perf_counter()))

	# ── Step-back: generate a broader query to merge with original candidates ──
	_stepback_trace: Optional[Dict[str, Any]] = None
	_stepback_query: Optional[str] = None
	_effective_stepback = (
		stepback_enabled
		and intent not in _HYDE_SKIP_INTENTS
		and len(query.split()) >= _HYDE_MIN_WORDS
	)
	if _effective_stepback and not cross_encoder_only:
		_pg("  → Step-back: broadening query…")
		_llm_url = ollama_base_url
		_hyde_url_sb = hyde_base_url
		try:
			from utils.service_discovery import get_remote_ollama_url  # noqa: PLC0415
			_remote_ol = get_remote_ollama_url()
			if _remote_ol:
				_hyde_url_sb = _remote_ol
		except Exception:
			pass
		_stepback_result, _stepback_trace = generate_stepback_query(
			query,
			llm_model="rag-llm",
			llm_base_url=_llm_url,
			hyde_model=hyde_model,
			hyde_base_url=_hyde_url_sb,
			timeout_seconds=stepback_timeout_seconds,
			intent=intent or "",
		)
		if _stepback_trace and _stepback_trace.get("applied"):
			_stepback_query = _stepback_result
		_perf_marks.append(("stepback", time.perf_counter()))

	use_cross_only = bool(cross_encoder_only)
	if use_cross_only and not cross_encoder_enabled:
		cross_encoder_enabled = True

	embedder = None
	qvec: Optional[List[float]] = None
	if not use_cross_only:
		embedder = create_embedder(
			backend=embed_backend,
			model_name=embed_model_name,
			dimension=embed_dimension,
			ollama_base_url=ollama_base_url,
			ollama_timeout_seconds=ollama_timeout_seconds,
		)

	conn = _connect(db_dsn)
	try:
		# Acronym expansion still runs on the original query so that
		# domain-specific abbreviations are resolved for lexical reranking;
		# the resulting expanded query is merged with the HyDE passage for
		# embedding so both signals are present.
		expanded_query, expansion_info = _expand_query_with_acronyms(query, conn)
		hyde_applied = False
		if embedder is not None:
			# When HyDE is active embed the hypothetical passage; fall back to
			# the acronym-expanded original query when HyDE was not applied.
			hyde_applied = bool(_hyde_trace and _hyde_trace.get("applied"))
			embed_text = embed_query_text if hyde_applied else expanded_query
			_pg("  → Embedding query…")
			qvec = embedder.embed_query(embed_text)

		# ── Stage 2 vector: embed original query for dual-vector blending ─────
		# Only computed when HyDE actually fired; otherwise qvec already encodes
		# the original query and a second embed call would be redundant.
		stage2_qvec: Optional[List[float]] = None
		if two_stage_enabled and hyde_applied and embedder is not None:
			stage2_qvec = embedder.embed_query(expanded_query)

		rows = _text_rows(conn, f)
		_perf_marks.append(("embed", time.perf_counter()))
		_pg(f"  → Scoring {len(rows)} candidates…")

		# ── Vector scoring via pgvector ANN ───────────────────────────────────
		scored: List[Tuple[float, dict]] = []
		if use_cross_only:
			for r in rows:
				score = _lexical_candidate_score(expanded_query, r)
				score += _structural_role_score(query, r)
				score += _query_overlap_bonus(query, r)
				scored.append((score, r))
		elif qvec is not None:
			vec_limit = max(int(rerank_candidate_k), int(top_k)) * 2
			vec_hits = _vector_candidates(
				conn, f, qvec,
				stage2_vec=stage2_qvec,
				two_stage_alpha=two_stage_alpha,
				limit=vec_limit,
			)
			score_map: Dict[str, float] = {r["chunk_id"]: float(r["score"]) for r in vec_hits}
			for r in rows:
				cid = r["chunk_id"]
				if cid not in score_map:
					continue
				score = score_map[cid]
				score += _structural_role_score(query, r)
				score += _query_overlap_bonus(query, r)
				scored.append((score, r))

		scored.sort(key=lambda x: x[0], reverse=True)
		_perf_marks.append(("ann", time.perf_counter()))

		# ── Step-back candidate injection ────────────────────────────────────────
		# Embed the broader step-back query and retrieve its top candidates.
		# Any chunk that is in the step-back set but not yet in the scored list
		# is appended (score=0.0) so the cross-encoder can evaluate it.
		if _stepback_query and embedder is not None and qvec is not None:
			try:
				sb_vec = embedder.embed_query(_stepback_query)
				sb_limit = max(int(rerank_candidate_k), int(top_k))
				sb_hits = _vector_candidates(conn, f, sb_vec, limit=sb_limit)
				scored_ids: set = {r["chunk_id"] for _, r in scored}
				row_map_sb: Dict[str, dict] = {r["chunk_id"]: r for r in rows}
				sb_injected = 0
				for sb_row in sb_hits:
					cid = sb_row["chunk_id"]
					if cid not in scored_ids and cid in row_map_sb:
						scored.append((0.0, row_map_sb[cid]))
						scored_ids.add(cid)
						sb_injected += 1
						if sb_injected >= top_k:
							break
				if _stepback_trace:
					_stepback_trace["injected_chunks"] = sb_injected
			except Exception:
				pass
			_perf_marks.append(("stepback_ann", time.perf_counter()))

		# ── BM25 + RRF candidate expansion ──────────────────────────────────────
		# When BM25 is enabled, blend the vector rank and BM25 rank via Reciprocal
		# Rank Fusion (RRF) so that chunks with strong exact-token matches are
		# surfaced even when their cosine score falls below the candidate_k cut.
		# The cosine scores are restored after reordering so the downstream reranker
		# (which blends vector + lexical) still receives meaningful vector scores.
		bm25_inject: List[dict] = []
		if bm25_enabled and not use_cross_only and rows:
			from retrieval.acronym_expand import _tokenize_words  # noqa: PLC0415
			orig_toks = [t.lower() for t in _tokenize_words(query)]
			hyde_toks = [t.lower() for t in _tokenize_words(expanded_query)]
			query_toks = list(dict.fromkeys(orig_toks + hyde_toks))
			bm25_chunk_ids, raw_bm25 = _bm25_scores(rows, query_toks, db_dsn=db_dsn, conn=conn)
			_rrf_candidate_k = max(int(top_k), int(rerank_candidate_k))
			# Keep the top vector candidates as-is; BM25 only injects chunks that
			# ranked below _rrf_candidate_k by cosine (i.e. ones the vector scan missed).
			vec_top_ids: set = {r["chunk_id"] for _, r in scored[:_rrf_candidate_k]}
			row_map: Dict[str, dict] = {r["chunk_id"]: r for r in rows}
			for cid, _ in sorted(zip(bm25_chunk_ids, raw_bm25), key=lambda x: -x[1]):
				if cid not in vec_top_ids and cid in row_map:
					bm25_inject.append(row_map[cid])
				if len(bm25_inject) >= top_k:
					break
			# Append BM25-only additions (score=0.0; reranker will decide their value).
			scored = scored[:_rrf_candidate_k] + [
				(0.0, r) for r in bm25_inject
			]

		# ── PRF: pseudo-relevance feedback term expansion ─────────────────────
		# Mine high-frequency terms from the top-scored initial candidates,
		# add novel terms to the BM25 token list, and inject any new chunks.
		# Gate is off by default (prf_enabled=False).
		_prf_trace: Dict[str, Any] = {"applied": False}
		if prf_enabled and not use_cross_only and rows and scored:
			try:
				from retrieval.acronym_expand import _tokenize_words as _prf_tok  # noqa: PLC0415
				_PRF_STOPWORDS = frozenset({
					"the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
					"for", "of", "with", "by", "from", "that", "this", "it", "is",
					"are", "was", "were", "be", "been", "being", "have", "has",
					"had", "do", "does", "did", "will", "would", "could", "should",
					"may", "might", "must", "can", "not", "no", "nor", "so", "yet",
					"both", "either", "neither", "each", "few", "more", "most",
					"other", "some", "such", "than", "too", "very", "just", "also",
					"they", "their", "there", "which", "when", "where", "what",
					"who", "how", "all", "any", "each", "into", "then", "than",
				})
				_prf_top_rows = [r for _, r in scored[:3]]
				_prf_freq: Dict[str, int] = {}
				for _pr in _prf_top_rows:
					for _tok in _prf_tok(_pr.get("text", "")):
						_tl = _tok.lower()
						if len(_tl) >= 4 and _tl not in _PRF_STOPWORDS:
							_prf_freq[_tl] = _prf_freq.get(_tl, 0) + 1
				_existing_toks = {t.lower() for t in _prf_tok(expanded_query)}
				_prf_new_terms = [
					t for t, _ in sorted(_prf_freq.items(), key=lambda x: -x[1])
					if t not in _existing_toks
				][:5]
				if _prf_new_terms:
					_prf_all_toks = list(dict.fromkeys(
						[t.lower() for t in _prf_tok(expanded_query)] + _prf_new_terms
					))
					_prf_scored_ids = {r["chunk_id"] for _, r in scored}
					_prf_row_map = {r["chunk_id"]: r for r in rows}
					_prf_chunk_ids, _ = _bm25_scores(rows, _prf_all_toks, db_dsn=db_dsn, conn=conn)
					_prf_injected: List[dict] = []
					for _pcid in _prf_chunk_ids:
						if _pcid not in _prf_scored_ids and _pcid in _prf_row_map:
							_prf_injected.append(_prf_row_map[_pcid])
							if len(_prf_injected) >= int(top_k):
								break
					if _prf_injected:
						scored = list(scored) + [(0.0, r) for r in _prf_injected]
					_prf_trace = {
						"applied": bool(_prf_injected),
						"new_terms": _prf_new_terms,
						"injected": len(_prf_injected),
					}
			except Exception:
				pass
		_perf_marks.append(("prf", time.perf_counter()))

		# ── Hypothetical question retrieval ──────────────────────────────────
		# A second ANN pass over the chunk_questions table uses pre-generated
		# question embeddings to surface chunks whose text never overlaps with
		# the query phrasing but whose questions do.  Results are merged into
		# `scored` with a sentinel score of 0.0 (they will be re-ranked).
		if hyp_questions_enabled and query_vec is not None:
			try:
				from db.client import search_chunks_by_question_embedding, _connect as _db_connect
				_already_ids = {r.get("chunk_id") for _, r in scored}
				_q_hits = search_chunks_by_question_embedding(
					db_dsn,
					list(query_vec),
					limit=int(hyp_questions_top_k),
					collection_id=collection_id,
				)
				_injected_q: List[Any] = []
				for _qh in _q_hits:
					_qcid = _qh.get("chunk_id")
					if _qcid and _qcid not in _already_ids:
						# Fetch the full row to make it compatible with `scored`
						_own_conn = conn is None
						_conn_q = _db_connect(db_dsn) if _own_conn else conn
						try:
							_qrow = _conn_q.execute(
								"""
								SELECT c.*, d.filename AS doc_filename,
								       d.source_path AS doc_source_path,
								       d.metadata_json AS doc_metadata_json
								FROM chunks c
								JOIN documents d ON d.doc_id = c.doc_id
								WHERE c.chunk_id = %s
								""",
								(_qcid,),
							).fetchone()
						finally:
							if _own_conn:
								_conn_q.close()
						if _qrow:
							_injected_q.append((0.0, dict(_qrow)))
							_already_ids.add(_qcid)
				if _injected_q:
					scored = list(scored) + _injected_q
			except Exception:
				pass
		_perf_marks.append(("hyp_q", time.perf_counter()))

		candidate_k = max(int(top_k), int(rerank_candidate_k))
		top = scored[: max(1, candidate_k + len(bm25_inject))] if bm25_inject else scored[: max(1, candidate_k)]

		_perf_marks.append(("bm25", time.perf_counter()))
		candidate_hits: List[RetrievedChunk] = []
		for score, r in top:
			try:
				doc_meta = json.loads(r["doc_metadata_json"] or "{}")
			except Exception:
				doc_meta = {}
			page_start = r["page_start"]
			page_end = r["page_end"]
			page_number = page_start if page_start == page_end else page_start
			section_header = r["title"] or r["path_text"]
			collection_id = r["collection_id"] or r["source_type"]
			source_name = r["source_name"] or r["doc_filename"]
			document_title = r["document_title"] or r["title"] or r["doc_filename"]
			document_path = _format_document_path(r["document_path"] or r["doc_source_path"])

			metadata: Dict[str, Any] = {
				"level": r["level"],
				"token_count_est": r["token_count_est"],
				"has_table": bool(r["has_table"]),
				"source_type": str(r["source_type"] or "pdf_book"),
				"structural_role": str(r["structural_role"] or "body"),
				"collection_id": collection_id,
				"source_name": source_name,
				"document_title": document_title,
				"document_path": document_path,
				"page_number": page_number,
				"section_header": section_header,
				"retrieval_score": float(score),
				"document_registry": normalize_document_registry_entry(
					doc_id=str(r["doc_id"] or ""),
					filename=str(r["doc_filename"] or source_name or ""),
					source_path=str(r["doc_source_path"] or document_path or ""),
					source_type=str(r["source_type"] or "pdf_book"),
					metadata=doc_meta,
				),
			}
			if include_neighbors:
				metadata["neighbors"] = [
					{"chunk_id": cid, "text": txt}
					for cid, txt in _chunk_neighbors(conn, r["chunk_id"], window=neighbor_window)
				]

			candidate_hits.append(
				RetrievedChunk(
					chunk_id=r["chunk_id"],
					doc_id=r["doc_id"],
					collection_id=collection_id,
					source_name=source_name,
					document_title=document_title,
					document_path=document_path,
					section_id=r["section_id"],
					title=r["title"],
					path_text=r["path_text"],
					page_number=page_number,
					section_header=section_header,
					page_start=page_start,
					page_end=page_end,
					text=normalize_chunk_text(r["text"]),
					score=float(score),
					source_type=str(r["source_type"] or "pdf_book"),
					structural_role=str(r["structural_role"] or "body"),
					metadata=metadata,
				)
			)

		reranked = [h.to_dict() for h in candidate_hits]
		if rerank_enabled and not use_cross_only:
			reranked = rerank_by_query(
				expanded_query,
				reranked,
				alpha_vector=rerank_alpha_vector,
				alpha_lexical=rerank_alpha_lexical,
				prefer_tables=rerank_prefer_tables,
				prefer_shorter=rerank_prefer_shorter,
				diversity_penalty=rerank_diversity_penalty,
				max_select=int(top_k),
				progress_fn=_pg,
			)

		_perf_marks.append(("rerank", time.perf_counter()))
		if cross_encoder_enabled:
			reranked, ce_trace = rerank_with_cross_encoder(
				expanded_query,
				reranked,
				model_name=cross_encoder_model,
				top_n=cross_encoder_top_n,
				weight=cross_encoder_weight,
			)
			if reranked:
				md0 = dict(reranked[0].get("metadata") or {})
				md0["cross_encoder"] = ce_trace
				md0["cross_encoder_only"] = bool(use_cross_only)
				reranked[0]["metadata"] = md0
		_perf_marks.append(("cross_encoder", time.perf_counter()))

		hits = [RetrievedChunk(**h) for h in reranked[: max(1, int(top_k))]]

		local_top_score = float(hits[0].score) if hits else 0.0
		local_score_gap = (
			float(hits[0].score) - float(hits[1].score)
			if len(hits) > 1
			else 1.0
		)
		local_low_confidence = bool(local_score_gap < float(low_confidence_gap_threshold)) if hits else True

		hard_local_filters = any([
			bool(f.doc_id),
			bool(f.doc_ids),
			bool(f.path_prefix),
			f.min_page is not None,
			f.max_page is not None,
			f.has_table is not None,
			bool(f.structural_role),
			bool(f.source_type),
			bool(f.collection_id),
		])
		prioritize_internet = _is_external_fact_query(query) or needs_web
		low_confidence_trigger = (
			bool(internet_trigger_on_low_confidence)
			and (
				local_low_confidence
				or local_top_score < float(internet_trigger_top_score)
				or local_score_gap < float(internet_trigger_gap)
			)
		)
		trigger_internet_fallback = (
			bool(internet_fallback_enabled)
			and not bool(use_cross_only)
			and not hard_local_filters
			and (
				not hits
				or prioritize_internet
				or low_confidence_trigger
			)
		)

		internet_used = False
		internet_count = 0
		internet_hits: List[Dict[str, Any]] = []
		internet_trace: Optional[Dict[str, Any]] = None
		internet_priority_applied = False
		internet_relevance_rejected_count = 0
		internet_relevance_accepted_count = 0
		if trigger_internet_fallback:
			try:
				effective_internet_max_results = max(1, int(internet_max_results))
				if prioritize_internet:
					effective_internet_max_results = max(effective_internet_max_results, 8)
				_hyde_passage_for_search = (
					embed_query_text
					if (bool(_hyde_trace and _hyde_trace.get("applied")) and embed_query_text != query)
					else ""
				)
				_hyde_search_query = (
					str(_hyde_trace.get("search_query") or "")
					if (_hyde_trace and _hyde_trace.get("search_query"))
					else ""
				)
				internet_result = retrieve_internet_chunks(
					query=query,
					query_vector=(qvec or []),
					embedder=embedder,
					max_results=effective_internet_max_results,
					max_chunks=max(int(top_k), int(internet_max_chunks)),
					timeout_seconds=float(internet_timeout_seconds),
					score_weight=float(internet_score_weight),
					hyde_passage=_hyde_passage_for_search,
					hyde_search_query=_hyde_search_query,
					llm_model=hyde_model,
					llm_base_url=hyde_base_url,
				)
			except Exception:
				internet_result = {"hits": [], "trace": None}

			if isinstance(internet_result, dict):
				internet_hits = list(internet_result.get("hits") or [])
				internet_trace = internet_result.get("trace") if isinstance(internet_result.get("trace"), dict) else None
			else:
				internet_hits = list(internet_result or [])

			if internet_hits:
				qualified_internet_hits: List[Dict[str, Any]] = []
				for wh in internet_hits:
					score = float(wh.get("score") or 0.0)
					if score < float(internet_min_relevance_score):
						wmd = wh.setdefault("metadata", {})
						wmd["internet_relevance_rejected"] = True
						wmd["internet_relevance_reason"] = "min_score_not_met"
						wmd["internet_min_relevance_score"] = float(internet_min_relevance_score)
						internet_relevance_rejected_count += 1
						continue
					qualified_internet_hits.append(wh)

				internet_hits = qualified_internet_hits
				internet_relevance_accepted_count = len(internet_hits)

			if internet_hits:
				if local_top_score >= float(internet_override_guard_threshold) and not prioritize_internet:
					cap_score = local_top_score - 0.02
					for wh in internet_hits:
						wh["score"] = min(float(wh.get("score") or 0.0), cap_score)
						wmd = wh.setdefault("metadata", {})
						wmd["internet_score_capped"] = True
						wmd["internet_cap_score"] = float(cap_score)

				merged: List[Dict[str, Any]] = [h.to_dict() for h in hits]
				merged.extend(internet_hits)
				merged.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)

				uniq: List[Dict[str, Any]] = []
				seen_ids: set[str] = set()
				for row in merged:
					cid = str(row.get("chunk_id") or "")
					if cid and cid in seen_ids:
						continue
					if cid:
						seen_ids.add(cid)
					uniq.append(row)

				if prioritize_internet:
					internet_rows = [row for row in uniq if (row.get("source_type") or "") == "internet"]
					local_rows = [row for row in uniq if (row.get("source_type") or "") != "internet"]
					selected_rows = (internet_rows + local_rows)[: max(1, int(top_k))]
					internet_priority_applied = bool(internet_rows)
				else:
					selected_rows = uniq[: max(1, int(top_k))]
					selected_internet_count = sum(1 for row in selected_rows if (row.get("source_type") or "") == "internet")
					if selected_internet_count == 0 and internet_hits:
						best_internet = max(internet_hits, key=lambda row: float(row.get("score") or 0.0))
						if not selected_rows:
							selected_rows = [best_internet]
						else:
							selected_rows = list(selected_rows[: max(1, int(top_k))])
							selected_rows[-1] = best_internet

				hits = [RetrievedChunk(**h) for h in selected_rows]
				internet_used = True
				internet_count = sum(1 for h in hits if (h.source_type or "") == "internet")

		if internet_trace is not None:
			internet_trace = {
				**internet_trace,
				"triggered": bool(trigger_internet_fallback),
				"used": bool(internet_used),
				"selected_count": int(internet_count),
				"min_relevance_score": float(internet_min_relevance_score),
				"relevance_accepted_count": int(internet_relevance_accepted_count),
				"relevance_rejected_count": int(internet_relevance_rejected_count),
				"priority_applied": bool(internet_priority_applied),
			}
		_perf_marks.append(("internet", time.perf_counter()))

		# === DOCX boost pass ===
		# Run a second retrieval pass restricted to DOCX chunks so that notes
		# always have representation alongside the larger PDF corpus.
		if docx_boost_k > 0 and not f.source_type and not f.doc_id and qvec is not None:
			from retrieval.models import RetrievalFilters as _RF  # noqa: PLC0415
			docx_f = _RF(source_type="docx")
			docx_rows_meta = {r["chunk_id"]: r for r in _text_rows(conn, docx_f)}
			docx_vec_hits = _vector_candidates(conn, docx_f, qvec, limit=docx_boost_k * 3)
			docx_scored: List[Tuple[float, dict]] = []
			for _vh in docx_vec_hits:
				_cid = _vh["chunk_id"]
				_dr = docx_rows_meta.get(_cid)
				if _dr is None:
					continue
				_s = float(_vh["score"])
				_s += _structural_role_score(query, _dr)
				_s += _query_overlap_bonus(query, _dr)
				docx_scored.append((_s, _dr))
			docx_scored.sort(key=lambda x: x[0], reverse=True)

			existing_ids = {h.chunk_id for h in hits}
			injected = 0
			for _s, _dr in docx_scored:
				if injected >= docx_boost_k:
					break
				_cid = _dr["chunk_id"]
				if _cid in existing_ids:
					continue
				_page_start = _dr["page_start"]
				_page_end = _dr["page_end"]
				_page_number = _page_start if _page_start == _page_end else _page_start
				_section_header = _dr["title"] or _dr["path_text"]
				_collection_id = _dr["collection_id"] or _dr["source_type"]
				_source_name = _dr["source_name"] or _dr["doc_filename"]
				_document_title = _dr["document_title"] or _dr["title"] or _dr["doc_filename"]
				_document_path = _format_document_path(_dr["document_path"] or _dr["doc_source_path"])
				_meta: Dict[str, Any] = {
					"level": _dr["level"],
					"token_count_est": _dr["token_count_est"],
					"has_table": bool(_dr["has_table"]),
					"source_type": "docx",
					"structural_role": str(_dr["structural_role"] or "body"),
					"collection_id": _collection_id,
					"source_name": _source_name,
					"document_title": _document_title,
					"document_path": _document_path,
					"page_number": _page_number,
					"section_header": _section_header,
					"retrieval_score": float(_s),
					"docx_boost_injected": True,
				}
				if include_neighbors:
					_meta["neighbors"] = [
						{"chunk_id": _nc_id, "text": _nc_txt}
						for _nc_id, _nc_txt in _chunk_neighbors(conn, _cid, window=neighbor_window)
					]
				hits = list(hits) + [
					RetrievedChunk(
						chunk_id=_cid,
						doc_id=_dr["doc_id"],
						collection_id=_collection_id,
						source_name=_source_name,
						document_title=_document_title,
						document_path=_document_path,
						section_id=_dr["section_id"],
						title=_dr["title"],
						path_text=_dr["path_text"],
						page_number=_page_number,
						section_header=_section_header,
						page_start=_page_start,
						page_end=_page_end,
						text=_dr["text"],
						score=float(_s),
						source_type="docx",
						structural_role=str(_dr["structural_role"] or "body"),
						metadata=_meta,
					)
				]
				existing_ids.add(_cid)
				injected += 1

		# Keep retrieval contract: return at most top_k hits while preserving
		# any DOCX boosts that were explicitly injected.
		if len(hits) > max(1, int(top_k)):
			boosted_hits = [h for h in hits if bool(h.metadata.get("docx_boost_injected"))]
			non_boosted_hits = [h for h in hits if not bool(h.metadata.get("docx_boost_injected"))]
			boosted_hits.sort(key=lambda h: float(h.score), reverse=True)
			non_boosted_hits.sort(key=lambda h: float(h.score), reverse=True)

			target_k = max(1, int(top_k))
			if len(boosted_hits) >= target_k:
				hits = boosted_hits[:target_k]
			else:
				remaining = target_k - len(boosted_hits)
				hits = boosted_hits + non_boosted_hits[:remaining]
			hits.sort(key=lambda h: float(h.score), reverse=True)

		if hits:
			if len(hits) > 1:
				score_gap = float(hits[0].score) - float(hits[1].score)
			else:
				score_gap = 1.0
			hits[0].metadata["query_expansion"] = expansion_info
			hits[0].metadata["score_gap_to_second"] = score_gap
			hits[0].metadata["low_confidence"] = bool(score_gap < float(low_confidence_gap_threshold))
			hits[0].metadata["low_confidence_gap_threshold"] = float(low_confidence_gap_threshold)
			hits[0].metadata["internet_fallback_triggered"] = bool(trigger_internet_fallback)
			hits[0].metadata["internet_fallback_used"] = bool(internet_used)
			hits[0].metadata["internet_fallback_selected_count"] = int(internet_count)
			hits[0].metadata["internet_trigger_top_score"] = float(internet_trigger_top_score)
			hits[0].metadata["internet_trigger_gap"] = float(internet_trigger_gap)
			if two_stage_enabled:
				hits[0].metadata["two_stage_applied"] = bool(stage2_qvec is not None)
				hits[0].metadata["two_stage_alpha"] = float(two_stage_alpha) if stage2_qvec is not None else None
			if prf_enabled or _prf_trace.get("applied"):
				hits[0].metadata["prf_trace"] = _prf_trace

		_t_end = time.perf_counter()
		_perf_ms: Dict[str, float] = {"total": round((_t_end - _t0) * 1000, 1)}
		_prev = _t0
		for _pname, _pt in _perf_marks:
			_perf_ms[_pname] = round((_pt - _prev) * 1000, 1)
			_prev = _pt
		return RetrievalResult(
			query=query,
			top_k=top_k,
			filters=asdict(f),
			hits=hits,
			internet_fallback=internet_trace,
			hyde_trace=_hyde_trace,
			stepback_trace=_stepback_trace,
			perf_ms=_perf_ms,
		)
	finally:
		conn.close()


def retrieve_as_dict(*args: Any, **kwargs: Any) -> Dict[str, Any]:
	return retrieve(*args, **kwargs).to_dict()
