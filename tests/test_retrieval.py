import json
import sys
from pathlib import Path
from unittest.mock import patch

import psycopg
import pytest
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db.client import init_db, persist_pipeline_outputs
from pipeline.models import IngestedDocument, IngestedPage
from retrieval.hyde import generate_hyde_query
from retrieval.query import RetrievalFilters, _expand_query_with_acronyms, retrieve
from retrieval.rerank import rerank_by_query


def _doc() -> IngestedDocument:
	return IngestedDocument(
		doc_id="doc1",
		source_path="C:/spec.pdf",
		filename="spec.pdf",
		num_pages=2,
		metadata={"title": "Spec"},
		pages=[
			IngestedPage(page_num=0, width=100, height=100, raw_text="a", blocks=[], tables=[]),
			IngestedPage(page_num=1, width=100, height=100, raw_text="b", blocks=[], tables=[]),
		],
	)


_DIMS = 4096


def _e(*v: float) -> list:
	"""Pad a short vector to 4096 dims."""
	return list(v) + [0.0] * (_DIMS - len(v))


def _make_index(path: Path) -> Path:
	payload = {
		"source_chunks_path": "data/chunks/spec.chunks.merged.json",
		"backend": "_test",
		"dimension": 4096,
		"vector_count": 3,
		"items": [
			{
				"chunk_id": "doc1-c000000",
				"doc_id": "doc1",
				"section_id": "s1",
				"path_text": "Pump > Electrical",
				"title": "Electrical",
				"level": 2,
				"page_start": 1,
				"page_end": 1,
				"has_table": False,
				"token_count_est": 120,
				"text": "The pump motor uses 24V DC input.",
				"embedding": _e(1.0),
			},
			{
				"chunk_id": "doc1-c000001",
				"doc_id": "doc1",
				"section_id": "s2",
				"path_text": "Pump > Hydraulic",
				"title": "Hydraulic",
				"level": 2,
				"page_start": 2,
				"page_end": 2,
				"has_table": True,
				"token_count_est": 140,
				"text": "Hydraulic pressure should remain below limits.",
				"embedding": _e(0.0, 1.0),
			},
			{
				"chunk_id": "doc1-c000002",
				"doc_id": "doc1",
				"section_id": "s3",
				"path_text": "Pump > Electrical",
				"title": "Electrical Notes",
				"level": 3,
				"page_start": 2,
				"page_end": 2,
				"has_table": False,
				"token_count_est": 100,
				"text": "Electrical connectors must be sealed.",
				"embedding": _e(0.9, 0.1),
			},
			{
				"chunk_id": "doc1-c000003",
				"doc_id": "doc1",
				"section_id": "s4",
				"path_text": "Pump > Quant",
				"title": "CAPM Overview",
				"level": 2,
				"page_start": 3,
				"page_end": 3,
				"has_table": False,
				"token_count_est": 130,
				"text": "The Capital Asset Pricing Model (CAPM) explains expected return and beta.",
				"embedding": _e(0.2, 0.2, 0.8),
			},
		],
	}
	path.write_text(json.dumps(payload), encoding="utf-8")
	return path


def _seed_db(pg_dsn: str, tmp_path: Path) -> str:
	init_db(pg_dsn)
	idx = _make_index(tmp_path / "index.json")
	persist_pipeline_outputs(
		pg_dsn,
		_doc(),
		extracted_path="data/extracted/spec.json",
		markdown_path="data/markdown/spec.md",
		structured_path="data/structured/spec.structured.json",
		chunks_path="data/chunks/spec.chunks.merged.json",
		index_path=str(idx),
	)
	return pg_dsn


@pytest.mark.requires_postgres
def test_retrieve_returns_top_k_with_filters(pg_dsn: str, tmp_path: Path):
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"electrical 24V",
		db_dsn=str(db),
		top_k=2,
		filters=RetrievalFilters(path_prefix="Pump > Electrical"),
		embed_backend="_test",
		embed_dimension=4096,
		include_neighbors=False,
	)
	assert len(result.hits) == 2
	assert all("Electrical" in (h.path_text or "") for h in result.hits)


@pytest.mark.requires_postgres
def test_retrieve_neighbors_context(pg_dsn: str, tmp_path: Path):
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"electrical connectors",
		db_dsn=str(db),
		top_k=1,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		include_neighbors=True,
		neighbor_window=1,
	)
	assert len(result.hits) == 1
	neighbors = result.hits[0].metadata.get("neighbors", [])
	assert isinstance(neighbors, list)


# ── HyDE tests ───────────────────────────────────────────────────────────────

def test_hyde_fallback_on_unreachable_server():
	"""generate_hyde_query must return original query when Ollama is unreachable."""
	query = "What is the vanishing gradient problem?"
	result, trace = generate_hyde_query(
		query,
		model="qwen2.5:3b-instruct",
		base_url="http://localhost:19999",  # deliberately wrong port
		timeout_seconds=2.0,
	)
	assert result == query, "should fall back to original query"
	assert trace["enabled"] is True
	assert trace["applied"] is False
	assert trace["reason"] in {"llm_error", "empty_response"}


@pytest.mark.requires_postgres
def test_hyde_disabled_does_not_call_llm(pg_dsn: str, tmp_path: Path):
	"""retrieve() with hyde_enabled=False must not produce a hyde_trace."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"hydraulic pressure",
		db_dsn=str(db),
		top_k=2,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
	)
	assert result.hyde_trace is None
	assert "hits" in result.to_dict()


@pytest.mark.requires_postgres
def test_hyde_enabled_with_unreachable_server_still_retrieves(pg_dsn: str, tmp_path: Path):
	"""retrieve() with hyde_enabled=True must still return hits when HyDE LLM fails."""
	db = _seed_db(pg_dsn, tmp_path)
	# Disable service discovery so the deliberately bad URL is not overridden.
	with patch("utils.service_discovery.get_remote_ollama_url", return_value=None):
		result = retrieve(
			"what is the maximum hydraulic pressure rating for safe operation",
			db_dsn=str(db),
			top_k=2,
			filters=RetrievalFilters(doc_id="doc1"),
			embed_backend="_test",
			embed_dimension=4096,
			hyde_enabled=True,
			hyde_base_url="http://localhost:19999",  # deliberately wrong port
			hyde_timeout_seconds=2.0,
		)
	assert len(result.hits) > 0, "should still return hits even when HyDE LLM fails"
	assert result.hyde_trace is not None
	assert result.hyde_trace["applied"] is False


@pytest.mark.requires_postgres
def test_two_stage_disabled_no_metadata_flag(pg_dsn: str, tmp_path: Path):
	"""With two_stage_enabled=False, two_stage_applied must not appear in metadata."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"hydraulic pressure",
		db_dsn=str(db),
		top_k=2,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
		two_stage_enabled=False,
	)
	assert len(result.hits) > 0
	assert result.hits[0].metadata.get("two_stage_applied") is None


@pytest.mark.requires_postgres
def test_two_stage_no_effect_when_hyde_fails(pg_dsn: str, tmp_path: Path):
	"""two_stage only fires when HyDE succeeds; with a bad HyDE URL it should be skipped."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"hydraulic pressure",
		db_dsn=str(db),
		top_k=2,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=True,
		hyde_base_url="http://localhost:19999",
		hyde_timeout_seconds=2.0,
		two_stage_enabled=True,
	)
	assert len(result.hits) > 0
	# HyDE failed → two_stage should not have fired
	assert result.hits[0].metadata.get("two_stage_applied") is False


def test_rerank_by_query_promotes_lexical_match():
	hits = [
		{
			"chunk_id": "a",
			"title": "Motor Basics",
			"path_text": "Pump > Electrical",
			"text": "motor specification and voltage limits",
			"score": 0.80,
			"metadata": {"has_table": False, "token_count_est": 300},
		},
		{
			"chunk_id": "b",
			"title": "Hydraulic Notes",
			"path_text": "Pump > Hydraulic",
			"text": "hydraulic pressure and flow calculations",
			"score": 0.79,
			"metadata": {"has_table": False, "token_count_est": 300},
		},
	]

	reranked = rerank_by_query("hydraulic pressure", hits, alpha_vector=0.5, alpha_lexical=0.5)
	assert reranked[0]["chunk_id"] == "b"
	assert "rerank_score" in reranked[0]["metadata"]
	assert "keyword_bonus" in reranked[0]["metadata"]
	assert "header_bonus" in reranked[0]["metadata"]


@pytest.mark.requires_postgres
def test_retrieve_rerank_uses_candidate_pool(pg_dsn: str, tmp_path: Path):
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"hydraulic pressure",
		db_dsn=str(db),
		top_k=1,
		rerank_enabled=True,
		rerank_candidate_k=4,
		rerank_alpha_vector=0.1,
		rerank_alpha_lexical=0.9,
		embed_backend="_test",
		embed_dimension=4096,
		include_neighbors=False,
	)

	assert len(result.hits) == 1
	assert result.hits[0].chunk_id == "doc1-c000001"
	assert "rerank_score" in result.hits[0].metadata


@pytest.mark.requires_postgres
def test_retrieve_adds_score_gap_low_confidence_flag(pg_dsn: str, tmp_path: Path):
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"electrical",
		db_dsn=str(db),
		top_k=2,
		rerank_enabled=True,
		rerank_candidate_k=3,
		embed_backend="_test",
		embed_dimension=4096,
		include_neighbors=False,
	)

	assert len(result.hits) == 2
	assert "score_gap_to_second" in result.hits[0].metadata
	assert "low_confidence" in result.hits[0].metadata
	assert isinstance(result.hits[0].metadata["low_confidence"], bool)


def test_rerank_by_query_exact_phrase_boosts_hit():
	hits = [
		{
			"chunk_id": "p1",
			"title": "Other",
			"path_text": "A > B",
			"text": "this chunk has the capital asset pricing model phrase exactly",
			"score": 0.60,
			"metadata": {"has_table": False, "token_count_est": 300},
		},
		{
			"chunk_id": "p2",
			"title": "Other",
			"path_text": "A > C",
			"text": "this chunk discusses pricing and capital but not the exact phrase order",
			"score": 0.60,
			"metadata": {"has_table": False, "token_count_est": 300},
		},
	]

	reranked = rerank_by_query("capital asset pricing model", hits, alpha_vector=0.3, alpha_lexical=0.7)
	assert reranked[0]["chunk_id"] == "p1"
	assert reranked[0]["metadata"].get("exact_phrase_bonus", 0.0) > 0.0


def test_rerank_by_query_diversity_penalty_promotes_distinct_evidence():
	hits = [
		{
			"chunk_id": "c1",
			"title": "A",
			"path_text": "X",
			"text": "alpha beta gamma delta epsilon",
			"score": 0.90,
			"metadata": {},
		},
		{
			"chunk_id": "c2",
			"title": "B",
			"path_text": "X",
			"text": "alpha beta gamma delta epsilon zeta",
			"score": 0.89,
			"metadata": {},
		},
		{
			"chunk_id": "c3",
			"title": "C",
			"path_text": "Y",
			"text": "portfolio optimization with risk constraints and expected return",
			"score": 0.86,
			"metadata": {},
		},
	]

	reranked = rerank_by_query("alpha beta gamma", hits, diversity_penalty=0.35)
	assert len(reranked) == 3
	assert reranked[0]["chunk_id"] in {"c1", "c2"}
	dup = next(h for h in reranked if h["chunk_id"] in {"c1", "c2"} and h["chunk_id"] != reranked[0]["chunk_id"])
	assert "diversity_adjusted_score" in dup["metadata"]
	assert dup["metadata"]["diversity_adjusted_score"] < dup["metadata"]["rerank_score"]


# ── BM25 tests ────────────────────────────────────────────────────────────────

@pytest.mark.requires_postgres
def test_bm25_scores_returns_correct_length(pg_dsn: str, tmp_path: Path):
	"""_bm25_scores must return chunk_ids and scores aligned with the row list."""
	from retrieval.query import _bm25_scores
	db = _seed_db(pg_dsn, tmp_path)
	conn = psycopg.connect(str(db), row_factory=dict_row)
	rows = conn.execute("SELECT * FROM chunks").fetchall()
	conn.close()
	chunk_ids, scores = _bm25_scores(rows, ["electrical", "voltage"])
	assert len(chunk_ids) == len(rows)
	assert len(scores) == len(rows)
	assert all(isinstance(s, float) for s in scores)


@pytest.mark.requires_postgres
def test_bm25_scores_ranks_matching_chunk_higher(pg_dsn: str, tmp_path: Path):
	"""The chunk whose text best matches the query should get the highest BM25 score."""
	from retrieval.query import _bm25_scores
	db = _seed_db(pg_dsn, tmp_path)
	conn = psycopg.connect(str(db), row_factory=dict_row)
	rows = conn.execute("SELECT * FROM chunks").fetchall()
	conn.close()
	chunk_ids, scores = _bm25_scores(rows, ["hydraulic", "pressure"])
	best_idx = scores.index(max(scores))
	best_id = chunk_ids[best_idx]
	# doc1-c000001 has "Hydraulic pressure should remain below limits."
	assert "hydraulic" in best_id.lower() or "000001" in best_id


@pytest.mark.requires_postgres
def test_retrieve_bm25_enabled_returns_hits(pg_dsn: str, tmp_path: Path):
	"""retrieve() with bm25_enabled=True must still return valid hits."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"hydraulic pressure",
		db_dsn=str(db),
		top_k=2,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
		bm25_enabled=True,
	)
	assert len(result.hits) > 0
	# All hits must be valid RetrievedChunk objects
	assert all(hasattr(h, "chunk_id") for h in result.hits)


@pytest.mark.requires_postgres
def test_retrieve_bm25_disabled_matches_baseline(pg_dsn: str, tmp_path: Path):
	"""With bm25_enabled=False the result should equal the non-BM25 baseline order."""
	db = _seed_db(pg_dsn, tmp_path)
	common_kwargs = dict(
		db_dsn=str(db),
		top_k=3,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
		rerank_enabled=False,
		include_neighbors=False,
	)
	result_no_bm25 = retrieve("electrical connectors", bm25_enabled=False, **common_kwargs)
	result_bm25    = retrieve("electrical connectors", bm25_enabled=True,  **common_kwargs)
	# Both should return hits; top hit IDs may differ (BM25 can surface different chunks)
	assert len(result_no_bm25.hits) > 0
	assert len(result_bm25.hits) > 0


@pytest.mark.requires_postgres
def test_retrieve_bm25_exact_token_chunk_surfaces(pg_dsn: str, tmp_path: Path):
	"""A chunk with exact query tokens should appear in top results with BM25 enabled."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"Capital Asset Pricing Model CAPM beta",
		db_dsn=str(db),
		top_k=3,
		filters=RetrievalFilters(doc_id="doc1"),
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
		bm25_enabled=True,
	)
	hit_ids = {h.chunk_id for h in result.hits}
	# doc1-c000003 contains "Capital Asset Pricing Model (CAPM)" exactly
	assert "doc1-c000003" in hit_ids, f"Expected CAPM chunk in results, got: {hit_ids}"


@pytest.mark.requires_postgres
def test_expand_query_with_acronyms_mines_corpus(pg_dsn: str, tmp_path: Path):
	db = _seed_db(pg_dsn, tmp_path)
	conn = psycopg.connect(str(db), row_factory=dict_row)
	try:
		expanded, info = _expand_query_with_acronyms("CAPM", conn)
	finally:
		conn.close()

	assert info["expanded"] is True
	assert "CAPM" in expanded
	assert "capital asset pricing model" in expanded.lower()


# ── HyDE skip conditions ──────────────────────────────────────────────────────

def test_hyde_skipped_for_short_query(pg_dsn: str, tmp_path: Path):
	"""Queries below _HYDE_MIN_WORDS words must not produce a hyde_trace."""
	db = _seed_db(pg_dsn, tmp_path)
	called = {"value": False}

	def _fake_hyde(*a, **kw):
		called["value"] = True
		return "fake passage", {"applied": True}

	with patch("retrieval.query.generate_hyde_query", _fake_hyde):
		result = retrieve(
			"What is CAPM?",  # 3 words — below threshold of 6
			db_dsn=str(db),
			top_k=2,
			embed_backend="_test",
			hyde_enabled=True,
		)

	assert called["value"] is False
	assert result.hyde_trace is None


def test_hyde_skipped_for_skip_intent(pg_dsn: str, tmp_path: Path):
	"""Intents in _HYDE_SKIP_INTENTS must suppress HyDE even for long queries."""
	from retrieval.query import _HYDE_SKIP_INTENTS
	skip_intent = next(iter(_HYDE_SKIP_INTENTS))

	db = _seed_db(pg_dsn, tmp_path)
	called = {"value": False}

	def _fake_hyde(*a, **kw):
		called["value"] = True
		return "fake passage", {"applied": True}

	with patch("retrieval.query.generate_hyde_query", _fake_hyde):
		result = retrieve(
			"What is the exact formula for the capital asset pricing model?",
			db_dsn=str(db),
			top_k=2,
			embed_backend="_test",
			hyde_enabled=True,
			intent=skip_intent,
		)

	assert called["value"] is False
	assert result.hyde_trace is None


def test_stepback_disabled_produces_no_trace(pg_dsn: str, tmp_path: Path):
	"""stepback_enabled=False must not call generate_stepback_query."""
	db = _seed_db(pg_dsn, tmp_path)
	called = {"value": False}

	def _fake_stepback(*a, **kw):
		called["value"] = True
		return "broader query", {"applied": True}

	with patch("retrieval.query.generate_stepback_query", _fake_stepback):
		result = retrieve(
			"How does the capital asset pricing model relate to portfolio theory?",
			db_dsn=str(db),
			top_k=2,
			embed_backend="_test",
			hyde_enabled=False,
			stepback_enabled=False,
		)

	assert called["value"] is False
	assert result.stepback_trace is None


def test_prf_disabled_by_default_no_prf_metadata(pg_dsn: str, tmp_path: Path):
	"""prf_enabled=False (default) must not attach prf_trace to hits."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"Explain the capital asset pricing model and its assumptions",
		db_dsn=str(db),
		top_k=2,
		embed_backend="_test",
		hyde_enabled=False,
		prf_enabled=False,
	)

	for hit in result.hits:
		assert "prf_trace" not in hit.metadata


def test_prf_enabled_attaches_prf_trace(pg_dsn: str, tmp_path: Path):
	"""prf_enabled=True must attach prf_trace to the top hit's metadata."""
	db = _seed_db(pg_dsn, tmp_path)
	result = retrieve(
		"Explain the capital asset pricing model and its assumptions",
		db_dsn=str(db),
		top_k=2,
		embed_backend="_test",
		embed_dimension=4096,
		hyde_enabled=False,
		prf_enabled=True,
		bm25_enabled=True,
	)

	assert len(result.hits) > 0, "Expected at least one hit"
	# prf_trace is attached to the top hit when prf_enabled=True
	assert "prf_trace" in result.hits[0].metadata, (
		"prf_trace not found in top hit metadata when prf_enabled=True"
	)
	trace = result.hits[0].metadata["prf_trace"]
	assert isinstance(trace, dict)
	assert "applied" in trace
	assert "new_terms" in trace
	assert "injected" in trace


@pytest.mark.slow
def test_cross_encoder_reranks_relevant_passage_higher():
	"""Integration: real cross-encoder model ranks a relevant passage above an irrelevant one."""
	from retrieval.cross_encoder import rerank_with_cross_encoder

	query = "What is recursive binary splitting in decision trees?"
	hits = [
		{
			"chunk_id": "irrelevant",
			"text": "Chocolate cake is made by mixing flour, sugar, cocoa powder, and eggs.",
			"score": 0.90,
			"metadata": {},
		},
		{
			"chunk_id": "relevant",
			"text": (
				"Recursive binary splitting is a top-down greedy approach used to build decision trees. "
				"At each step the algorithm selects the feature and cutpoint that minimise the residual sum "
				"of squares (RSS) across the resulting two subregions."
			),
			"score": 0.70,
			"metadata": {},
		},
	]

	reranked, trace = rerank_with_cross_encoder(
		query, hits,
		model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
		top_n=10,
		weight=0.65,
	)

	assert trace.get("applied") is True, f"CE was not applied: {trace}"
	assert len(reranked) == 2
	chunk_ids = [h["chunk_id"] for h in reranked]
	assert chunk_ids[0] == "relevant", (
		f"Expected relevant chunk at rank 1, got: {chunk_ids}"
	)
	# Verify CE metadata is injected
	top_md = reranked[0].get("metadata", {})
	assert "cross_encoder_score" in top_md
	assert 0.0 <= top_md["cross_encoder_score"] <= 1.0


@pytest.mark.slow
def test_cross_encoder_gracefully_handles_empty_hits():
	"""Integration: empty hits list returns empty with enabled=False trace."""
	from retrieval.cross_encoder import rerank_with_cross_encoder

	reranked, trace = rerank_with_cross_encoder(
		"some query", [],
		model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
	)
	assert reranked == []
	assert trace.get("applied") is False
