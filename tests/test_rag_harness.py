"""
Professional RAG Evaluation Harness
=====================================
Tests that separate a professional-grade RAG from a naive implementation.
Each section targets a specific capability that a naive system gets wrong.

Sections
--------
  A  BM25 Lexical Scoring      — IDF weighting, title boost, path_text scoring
  B  Reranking & Score Fusion  — alpha balance, phrase bonus, stub penalty, header boost
  C  Deduplication & Near-Dup  — same-doc vs. cross-doc near-dup semantics
  D  Context Pack Quality      — diversification gap, authority weighting, meta completeness
  E  Retrieval Filters         — doc_id, path_prefix, source_type, top_k precision
  F  Edge Cases & Robustness   — empty inputs, single-chunk corpus, zero-match queries
  G  Query Routing & Intent    — conversational bypass, temporal web routing, strategy map
  H  Pipeline Chunking         — sequential IDs, title injection, over-size splitting
  I  Citations & Source Policy — short_citation, score window, inline enforcement
  J  Coverage & Confidence     — overview detection, lexical coverage, confidence bands
  K  Grounding & Hallucination — morphological variants, entity support, off-topic coverage
  L  Text Utilities & Normalization — tokenize boundaries, markdown cleaning, token estimation

Scoring
-------
  • Each passing test earns 1 point.
  • Professional threshold: ≥ 85% overall, no section below 60%.
  • Score card is printed and appended to the assertion message.

Usage
-----
  pytest tests/test_rag_harness.py -v           # score in error if threshold missed
  pytest tests/test_rag_harness.py -v -s        # always prints score card
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.client import init_db, persist_pipeline_outputs
from llm.citations import (
    ensure_inline_sentence_citations,
    is_factual_sentence,
    normalize_answer_citations,
    select_citations,
    short_citation,
)
from llm.coverage import (
    determine_confidence_band,
    is_factoid_query,
    is_overview_query,
)
from llm.grounding import (
    lexical_coverage_score,
    term_present_in_text,
    term_variants,
    unsupported_answer_entities,
)
from pipeline.chunk.assembly import chunk_structured_document
from pipeline.chunk.strategies import estimate_tokens
from pipeline.models import IngestedDocument, IngestedPage
from pipeline.normalize.clean_markdown import clean_markdown
from retrieval.context_pack import build_context_pack
from retrieval.query import RetrievalFilters, _bm25_scores, retrieve
from retrieval.rerank import rerank_by_query
from retrieval.router import (
    RetrievalStrategy,
    RoutedQuery,
    _STRATEGY_MAP,
    _needs_web,
    classify_source_type,
    route_query,
)
from utils.text_utils import tokenize, tokenize_all

# ── Score tracking ────────────────────────────────────────────────────────────

_SECTION_LABELS: Dict[str, str] = {
    "A": "BM25 Lexical Scoring    ",
    "B": "Reranking & Score Fusion",
    "C": "Dedup & Near-Dup        ",
    "D": "Context Pack Quality    ",
    "E": "Retrieval Filters       ",
    "F": "Edge Cases & Robustness ",
    "G": "Query Routing & Intent  ",
    "H": "Pipeline Chunking       ",
    "I": "Citations & Source Policy",
    "J": "Coverage & Confidence   ",
    "K": "Grounding & Hallucination",
    "L": "Text Utils & Normalizatn",
}
_REGISTERED: Dict[str, List[str]] = {}
_PASSED: set[Tuple[str, str]] = set()


def _t(section: str, name: str) -> None:
    """Register a test case (call at the very start of each test body)."""
    _REGISTERED.setdefault(section, [])
    if name not in _REGISTERED[section]:
        _REGISTERED[section].append(name)


def _ok(section: str, name: str) -> None:
    """Mark a test case as passed (call at the very end of each test body)."""
    _PASSED.add((section, name))


# ── Shared helpers ────────────────────────────────────────────────────────────

class _Row(dict):
    """Minimal dict-based row stand-in for unit-testing _bm25_scores() directly."""
    def __getitem__(self, key: str) -> Any:  # type: ignore[override]
        return self.get(key)


def _mk_row(chunk_id: str, text: str, title: str = "", path_text: str = "") -> _Row:
    return _Row(chunk_id=chunk_id, text=text, title=title, path_text=path_text)


def _mk_hit(
    *,
    chunk_id: str,
    doc_id: str,
    text: str,
    score: float,
    source_type: str = "pdf_book",
    title: str = "Section",
    path_text: str = "Book > Section",
    token_count_est: int = 60,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "collection_id": "col1",
        "source_name": f"{doc_id}.pdf",
        "document_title": "Title",
        "document_path": f"/tmp/{doc_id}.pdf",
        "section_id": "s1",
        "title": title,
        "path_text": path_text,
        "page_number": 1,
        "section_header": title,
        "page_start": 1,
        "page_end": 1,
        "text": text,
        "score": score,
        "source_type": source_type,
        "structural_role": "body",
        "metadata": {
            "source_type": source_type,
            "collection_id": "col1",
            "source_name": f"{doc_id}.pdf",
            "document_title": "Title",
            "document_path": f"/tmp/{doc_id}.pdf",
            "token_count_est": token_count_est,
        },
    }


def _routed(intent: str = "summary") -> RoutedQuery:
    return RoutedQuery(
        original_query="q",
        intent=intent,
        sources=["corpus"],
        strategy=RetrievalStrategy(top_k=5),
        meta={
            "llm_routing": {
                "route_type": intent,
                "preferred_sources": ["corpus"],
                "confidence": 0.9,
                "valid": True,
            }
        },
    )


# ── Retrieval fixture (E tests) ───────────────────────────────────────────────

def _ingest_doc(db_path: str, base: Path, doc_id: str, title: str, chunks: List[Dict]) -> None:
    doc = IngestedDocument(
        doc_id=doc_id,
        source_path=f"C:/{doc_id}.pdf",
        filename=f"{doc_id}.pdf",
        num_pages=1,
        metadata={"title": title},
        pages=[IngestedPage(page_num=0, width=100, height=100, raw_text="x", blocks=[], tables=[])],
    )
    idx_path = base / f"{doc_id}_idx.json"
    idx_path.write_text(json.dumps({
        "source_chunks_path": f"data/chunks/{doc_id}.json",
        "backend": "_test",
        "dimension": 4096,
        "vector_count": len(chunks),
        "items": chunks,
    }), encoding="utf-8")
    persist_pipeline_outputs(
        db_path, doc,
        extracted_path=f"data/extracted/{doc_id}.json",
        markdown_path=f"data/markdown/{doc_id}.md",
        structured_path=f"data/structured/{doc_id}.json",
        chunks_path=f"data/chunks/{doc_id}.json",
        index_path=str(idx_path),
    )


@pytest.fixture(scope="module")
def retrieval_db(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Module-scoped multi-document DB for filter / retrieval tests."""
    import uuid
    import psycopg as _psycopg  # noqa: PLC0415

    _ADMIN_DSN = "postgresql://postgres:postgres@localhost/postgres"
    base = tmp_path_factory.mktemp("harness_e")
    db_name = f"rag_test_{uuid.uuid4().hex[:12]}"
    try:
        with _psycopg.connect(_ADMIN_DSN, autocommit=True, connect_timeout=2) as _ac:
            _ac.execute(f'CREATE DATABASE "{db_name}"')
    except Exception:
        pytest.skip("PostgreSQL not available")

    _DIMS = 4096

    def _e(*v: float) -> list:
        return list(v) + [0.0] * (_DIMS - len(v))

    pg_dsn = f"postgresql://postgres:postgres@localhost/{db_name}"
    try:
        init_db(pg_dsn)

        _ingest_doc(pg_dsn, base, "eng", "Engineering Spec", [
            {"chunk_id": "eng-c000000", "doc_id": "eng", "section_id": "s1",
             "path_text": "Spec > Hydraulic", "title": "Hydraulic System",
             "level": 2, "page_start": 1, "page_end": 1, "has_table": False,
             "token_count_est": 15,
             "text": "Hydraulic pressure must not exceed the rated operating limit under any load condition.",
             "embedding": _e(0.0, 1.0)},
            {"chunk_id": "eng-c000001", "doc_id": "eng", "section_id": "s1",
             "path_text": "Spec > Hydraulic > Pump", "title": "Pump Specifications",
             "level": 3, "page_start": 2, "page_end": 2, "has_table": False,
             "token_count_est": 12,
             "text": "The hydraulic pump delivers fluid at variable pressure and flow rate.",
             "embedding": _e(0.0, 0.9, 0.1)},
            {"chunk_id": "eng-c000002", "doc_id": "eng", "section_id": "s2",
             "path_text": "Spec > Electrical", "title": "Electrical System",
             "level": 2, "page_start": 3, "page_end": 3, "has_table": False,
             "token_count_est": 10,
             "text": "The motor controller operates on 24V DC with PWM speed control.",
             "embedding": _e(1.0)},
            {"chunk_id": "eng-c000003", "doc_id": "eng", "section_id": "s3",
             "path_text": "Spec > Safety", "title": "Safety Limits",
             "level": 2, "page_start": 4, "page_end": 4, "has_table": False,
             "token_count_est": 14,
             "text": "All safety limits must be enforced in hardware independent of software state.",
             "embedding": _e(0.5, 0.5)},
        ])

        _ingest_doc(pg_dsn, base, "ml", "ML Notes", [
            {"chunk_id": "ml-c000000", "doc_id": "ml", "section_id": "s1",
             "path_text": "Notes > Optimization", "title": "Gradient Descent",
             "level": 2, "page_start": 1, "page_end": 1, "has_table": False,
             "token_count_est": 14,
             "text": "Gradient descent iteratively updates model weights to minimise the loss function.",
             "embedding": _e(0.0, 0.0, 1.0)},
            {"chunk_id": "ml-c000001", "doc_id": "ml", "section_id": "s2",
             "path_text": "Notes > Regularization", "title": "Overfitting",
             "level": 2, "page_start": 2, "page_end": 2, "has_table": False,
             "token_count_est": 13,
             "text": "Overfitting occurs when a model memorises training data and fails to generalise.",
             "embedding": _e(0.0, 0.0, 0.0, 1.0)},
        ])
        yield pg_dsn
    finally:
        try:
            with _psycopg.connect(_ADMIN_DSN, autocommit=True, connect_timeout=2) as _ac:
                _ac.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Section A — BM25 Lexical Scoring
# A professional RAG builds IDF-weighted BM25 over text+title×5+path_text.
# Naive implementations use TF-only or ignore title/path contributions.
# ─────────────────────────────────────────────────────────────────────────────

def test_a1_idf_weights_rare_terms_over_high_tf() -> None:
    """IDF wins: rare term in one chunk beats a common term repeated 6×.

    chunk_freq has 'neural' 6× (high TF, low IDF — appears in 9 of 10 docs).
    chunk_rare has 'eigenvalue' once (low TF, very high IDF — appears once).
    Query: ['neural', 'eigenvalue'].
    BM25 IDF penalises the ubiquitous term; chunk_rare must win.
    A TF-only scorer would give chunk_freq an advantage.
    """
    _t("A", "idf_rare_beats_high_tf")
    # 10 corpus chunks; 'neural' appears in 9, 'eigenvalue' appears in 1
    rows = [
        _mk_row("c_rare", "Eigenvalue decomposition reveals principal components in the neural matrix."),
        _mk_row("c_freq", "Neural neural neural neural neural neural networks transform inputs."),
        *[_mk_row(f"c_filler_{i}", f"Neural networks learn hierarchical representations from training data example {i}.")
          for i in range(8)],
    ]
    ids, scores = _bm25_scores(rows, ["neural", "eigenvalue"])
    # 'eigenvalue' has IDF ≈ 42× higher than 'neural' in this corpus
    best_id = ids[scores.index(max(scores))]
    assert best_id == "c_rare", (
        f"Expected 'c_rare' (eigenvalue — high IDF) to win; got '{best_id}'. "
        "IDF weighting is not applied correctly."
    )
    _ok("A", "idf_rare_beats_high_tf")


def test_a2_title_5x_boost_beats_body_only_match() -> None:
    """Title 5× boost: query term in title beats same term only in body text.

    chunk_title has 'backpropagation' in its title (counts 5× in BM25 corpus).
    chunk_body has 'backpropagation' in its body text once.
    Both have comparable body text length.
    """
    _t("A", "title_boost")
    # Filler rows contain no 'backpropagation' — push corpus to N=8 so that
    # IDF for 'backpropagation' (n=2) is positive: log(6.5/2.5) ≈ 0.96.
    # With only 2 rows n=N=2 → IDF = log(0.2) < 0, and title-boost *hurts*.
    _fillers = [
        _mk_row("a2_f0", "Convolutional filters detect local patterns using shared weights."),
        _mk_row("a2_f1", "Recurrent networks process sequences by maintaining hidden state."),
        _mk_row("a2_f2", "Dropout randomly disables neurons to prevent co-adaptation."),
        _mk_row("a2_f3", "Batch normalisation standardises activations to accelerate training."),
        _mk_row("a2_f4", "Residual connections allow skip pathways through deep networks."),
        _mk_row("a2_f5", "Pooling layers reduce spatial dimensions while preserving features."),
    ]
    rows = [
        _mk_row("c_title",
                title="Backpropagation",
                text="The algorithm computes gradients layer by layer through the network chain."),
        _mk_row("c_body",
                title="Training Algorithms",
                text="Backpropagation computes gradients through the network using the chain rule."),
        *_fillers,
    ]
    ids, scores = _bm25_scores(rows, ["backpropagation"])
    best_id = ids[scores.index(max(scores))]
    assert best_id == "c_title", (
        f"Expected 'c_title' (title 5× boost) to win; got '{best_id}'. "
        "Title boosting is not implemented in BM25 corpus construction."
    )
    _ok("A", "title_boost")


def test_a3_path_text_contributes_to_bm25_score() -> None:
    """Path text is included in the BM25 corpus.

    chunk_path has 'backpropagation' in its path_text only.
    chunk_plain has equal body text but no path match.
    Query includes 'backpropagation' → chunk_path must score higher.
    """
    _t("A", "path_text_scored")
    rows = [
        _mk_row("c_path",
                text="The algorithm computes gradients through each layer sequentially.",
                path_text="Deep Learning > Training > Backpropagation"),
        _mk_row("c_plain",
                text="The algorithm computes gradients through each layer sequentially.",
                path_text="Deep Learning > Training > General"),
    ]
    ids, scores = _bm25_scores(rows, ["backpropagation"])
    best_id = ids[scores.index(max(scores))]
    assert best_id == "c_path", (
        f"Expected 'c_path' (backpropagation in path_text) to win; got '{best_id}'. "
        "path_text is not included in the BM25 corpus."
    )
    _ok("A", "path_text_scored")


def test_a4_full_match_beats_partial_match() -> None:
    """Chunk matching all three query terms beats chunk matching only one."""
    _t("A", "full_beats_partial")
    rows = [
        _mk_row("c_full",  text="Transformer attention mechanisms use multi-head self-attention layers."),
        _mk_row("c_partial", text="The transformer is a powerful sequence model."),
        _mk_row("c_none",  text="The butterfly effect demonstrates sensitivity to initial conditions."),
    ]
    ids, scores = _bm25_scores(rows, ["transformer", "attention", "mechanisms"])
    order = sorted(zip(scores, ids), reverse=True)
    ranked_ids = [i for _, i in order]
    assert ranked_ids[0] == "c_full", "Full-match chunk must rank first."
    assert ranked_ids[-1] == "c_none", "Zero-match chunk must rank last."
    _ok("A", "full_beats_partial")


def test_a5_empty_query_tokens_returns_zeros_no_crash() -> None:
    """_bm25_scores with empty token list returns zero scores without raising."""
    _t("A", "empty_tokens_robust")
    rows = [_mk_row("c0", "Some text about transformers and attention.")]
    ids, scores = _bm25_scores(rows, [])
    assert ids == ["c0"]
    assert scores == [0.0]
    _ok("A", "empty_tokens_robust")


# ─────────────────────────────────────────────────────────────────────────────
# Section B — Reranking & Score Fusion
# Professional reranking uses vector + lexical alpha blend, phrase bonuses,
# header bonuses, and short-stub penalties. Naive rankers use score-only.
# ─────────────────────────────────────────────────────────────────────────────

def test_b1_vector_and_lexical_agreement_produces_top_rank() -> None:
    """A chunk that wins both vector score AND lexical overlap must rank first."""
    _t("B", "agreement_wins")
    hits = [
        _mk_hit(chunk_id="h_win", doc_id="d1",
                text="Transformer attention mechanisms process sequences in parallel using multi-head self-attention.",
                score=0.92),
        _mk_hit(chunk_id="h_mid", doc_id="d1",
                text="Attention heads in transformers attend to different subspaces of the representation.",
                score=0.65),
        _mk_hit(chunk_id="h_low", doc_id="d2",
                text="Monarch butterflies migrate thousands of kilometres each autumn season.",
                score=0.20),
    ]
    ranked = rerank_by_query("transformer attention mechanisms", hits,
                             alpha_vector=0.6, alpha_lexical=0.4)
    assert ranked[0]["chunk_id"] == "h_win", (
        f"Expected h_win (top vector + top lexical) first; got {ranked[0]['chunk_id']}."
    )
    assert ranked[-1]["chunk_id"] == "h_low", (
        f"Expected h_low (no lexical match, low vector) last; got {ranked[-1]['chunk_id']}."
    )
    _ok("B", "agreement_wins")


def test_b2_alpha_lexical_1_orders_by_lexical_only() -> None:
    """With alpha_lexical=1.0, alpha_vector=0.0, ranking follows lexical overlap only."""
    _t("B", "alpha_controls_order")
    hits = [
        _mk_hit(chunk_id="h_highvec", doc_id="d1",
                text="Unrelated content about butterfly migration patterns.",
                score=0.98),   # very high vector but zero lexical
        _mk_hit(chunk_id="h_highlex", doc_id="d2",
                text="Gradient descent optimises the loss using gradient steps.",
                score=0.10),   # low vector but strong lexical for 'gradient descent'
    ]
    ranked = rerank_by_query(
        "gradient descent optimisation",
        hits,
        alpha_vector=0.0,
        alpha_lexical=1.0,
        diversity_penalty=0.0,
    )
    assert ranked[0]["chunk_id"] == "h_highlex", (
        "With alpha_lexical=1.0, the lexically-strong chunk must rank first. "
        "alpha weighting is not being applied."
    )
    _ok("B", "alpha_controls_order")


def test_b3_exact_phrase_bonus_lifts_phrase_match() -> None:
    """A chunk containing the exact query phrase gets the phrase bonus.

    Both chunks contain all three query tokens with identical Jaccard overlap;
    only h_phrase presents them as the contiguous phrase 'attention mechanisms allow'.
    With equal vector scores and equal jaccard, the phrase bonus (0.14) is the
    sole differentiator — h_phrase must rank first.
    """
    _t("B", "exact_phrase_bonus")
    # Both texts have exactly 9 tokens, all 3 query tokens present → same jaccard.
    # h_phrase: tokens start with 'attention mechanisms allow ...' → phrase fires.
    # h_scattered: tokens are 'allow precise mechanisms of attention ...' → no match.
    hits = [
        _mk_hit(chunk_id="h_phrase", doc_id="d1",
                text="Attention mechanisms allow transformers to model sequence dependencies effectively.",
                score=0.70),
        _mk_hit(chunk_id="h_scattered", doc_id="d1",
                text="Allow precise mechanisms of attention in transformer model processing.",
                score=0.70),  # same vector score & same jaccard — phrase bonus differentiates
    ]
    ranked = rerank_by_query(
        "attention mechanisms allow",
        hits,
        alpha_vector=0.5,
        alpha_lexical=0.5,
        diversity_penalty=0.0,
    )
    assert ranked[0]["chunk_id"] == "h_phrase", (
        "Chunk with exact phrase match must score higher than scattered-token chunk."
    )
    _ok("B", "exact_phrase_bonus")


def test_b4_header_bonus_lifts_title_match() -> None:
    """A chunk whose title matches the query gets a header bonus over body-only matches.

    h_titled has all 4 query tokens in its title + path (header_bonus = 0.24) but
    none in its body text.  h_body has no title match and its text contains only
    one query token ('rate').  With equal vector scores the header bonus of 0.24
    overwhelms h_body's tiny jaccard advantage and h_titled must rank first.
    """
    _t("B", "header_bonus")
    hits = [
        _mk_hit(chunk_id="h_titled", doc_id="d1",
                title="Gradient Descent Convergence Rate",
                path_text="ML > Optimisation > Convergence",
                text="The iterative minimisation procedure adjusts model parameters "
                     "by following the steepest downward slope at each update step.",
                score=0.70),
        _mk_hit(chunk_id="h_body", doc_id="d2",
                title="Parameter Update Methods",
                path_text="ML > Training Algorithms",
                text="The learning rate must be carefully tuned so that model "
                     "parameters converge smoothly during training.",
                score=0.70),  # 'rate' matches but no title match; 'converge' != 'convergence'
    ]
    ranked = rerank_by_query(
        "gradient descent convergence rate",
        hits,
        alpha_vector=0.5,
        alpha_lexical=0.5,
        diversity_penalty=0.0,
    )
    assert ranked[0]["chunk_id"] == "h_titled", (
        "Chunk with all 4 query tokens in title+path must rank higher via header bonus (0.24)."
    )
    _ok("B", "header_bonus")


def test_b5_short_stub_penalty_demotes_tiny_chunk() -> None:
    """Chunks with token_count_est < 20 receive a stub penalty in final score.

    Both chunks share identical text, identical vector score, and identical metadata
    — the ONLY difference is token_count_est (3 vs 50).  The stub (< 20 tokens)
    gets rule_boost -= 0.15 and must rank below the full-length chunk.
    """
    _t("B", "stub_penalty")
    _shared_text = (
        "Attention mechanisms allow transformer models to focus selectively "
        "on the most relevant portions of the input sequence during inference."
    )
    hits = [
        _mk_hit(chunk_id="h_stub", doc_id="d1",
                text=_shared_text, score=0.80, token_count_est=3),
        _mk_hit(chunk_id="h_full", doc_id="d2",
                text=_shared_text, score=0.80, token_count_est=50),
    ]
    ranked = rerank_by_query("attention mechanisms", hits,
                             alpha_vector=0.5, alpha_lexical=0.5,
                             diversity_penalty=0.0)
    assert ranked[0]["chunk_id"] == "h_full", (
        "The full-length chunk (token_count_est=50) must beat the stub (token_count_est=3). "
        "The stub penalty (-0.15) must be applied when 0 < token_count_est < 20."
    )
    _ok("B", "stub_penalty")


# ─────────────────────────────────────────────────────────────────────────────
# Section C — Deduplication & Near-Duplicate Detection
# Professional RAG deduplicates within the same document but preserves
# cross-document near-duplicates (diversity of sources is valuable).
# Naive dedupers treat all near-dups the same regardless of source.
# ─────────────────────────────────────────────────────────────────────────────

# 22-token template for near-dup tests; swapping one word gives Jaccard ≈ 0.91
_NEAR_DUP_BASE = (
    "Overfitting occurs when the model fits training data too closely and "
    "the learned parameters fail to generalise well to held-out unseen "
    "examples in the evaluation dataset."
)
_NEAR_DUP_ALT = (
    "Overfitting occurs when the model fits training data too closely and "
    "the learned parameters fail to generalise well to held-out future "
    "examples in the evaluation dataset."
)  # 'unseen' → 'future'; Jaccard ≈ 0.913 > 0.90 threshold


def test_c1_exact_duplicate_chunk_id_removed() -> None:
    """A chunk_id that appears twice keeps only the first occurrence."""
    _t("C", "exact_dup_chunk_id")
    hits = [
        _mk_hit(chunk_id="d1-c000001", doc_id="d1",
                text="Logistic regression uses sigmoid activation for binary classification.", score=0.85),
        _mk_hit(chunk_id="d1-c000001", doc_id="d1",
                text="Logistic regression uses sigmoid activation for binary classification.", score=0.83),
    ]
    pack = build_context_pack({"query": "logistic regression", "top_k": 5, "hits": hits},
                              _routed("fact_lookup"), max_chunks=5)
    assert len(pack["selected_chunks"]) == 1
    assert pack["selection_meta"]["deduplication"]["dropped_duplicate_chunk_id"] >= 1
    _ok("C", "exact_dup_chunk_id")


def test_c2_near_dup_same_document_is_removed() -> None:
    """Two near-duplicate chunks from the same document: only the first is kept."""
    _t("C", "near_dup_same_doc_removed")
    hits = [
        _mk_hit(chunk_id="d1-c000010", doc_id="d1", text=_NEAR_DUP_BASE, score=0.80),
        _mk_hit(chunk_id="d1-c000011", doc_id="d1", text=_NEAR_DUP_ALT,  score=0.79),
    ]
    pack = build_context_pack({"query": "overfitting", "top_k": 5, "hits": hits},
                              _routed("fact_lookup"), max_chunks=5)
    selected_ids = {h["chunk_id"] for h in pack["selected_chunks"]}
    assert "d1-c000010" in selected_ids, "First chunk must be retained."
    assert "d1-c000011" not in selected_ids, (
        "Near-duplicate from the SAME document must be removed. "
        "Naive dedupers that ignore doc_id would incorrectly remove cross-doc pairs too."
    )
    _ok("C", "near_dup_same_doc_removed")


def test_c3_near_dup_cross_document_both_kept() -> None:
    """Two near-duplicate chunks from DIFFERENT documents must BOTH be retained.

    Cross-document similarity is valuable — it represents corroboration from
    multiple sources. A naive deduper that ignores doc_id would wrongly drop one.
    """
    _t("C", "near_dup_cross_doc_kept")
    hits = [
        _mk_hit(chunk_id="d1-c000020", doc_id="d1", text=_NEAR_DUP_BASE, score=0.82),
        _mk_hit(chunk_id="d2-c000001", doc_id="d2", text=_NEAR_DUP_ALT,  score=0.81),
    ]
    pack = build_context_pack({"query": "overfitting", "top_k": 5, "hits": hits},
                              _routed("fact_lookup"), max_chunks=5)
    selected_ids = {h["chunk_id"] for h in pack["selected_chunks"]}
    assert "d1-c000020" in selected_ids, "Cross-doc chunk 1 must be retained."
    assert "d2-c000001" in selected_ids, (
        "Cross-doc near-duplicate must ALSO be retained. "
        "Near-dup dedup must be scoped to same-document pairs only."
    )
    _ok("C", "near_dup_cross_doc_kept")


def test_c4_short_distinct_texts_not_flagged_as_near_dup() -> None:
    """Short texts that differ by a single token are distinct — Jaccard < 0.9."""
    _t("C", "short_distinct_not_deduped")
    hits = [
        _mk_hit(chunk_id="d1-c000030", doc_id="d1",
                text="The function returns the maximum value in the input array.",
                score=0.75),
        _mk_hit(chunk_id="d1-c000031", doc_id="d1",
                text="The function returns the minimum value in the input array.",
                score=0.74),
        # 'maximum' vs 'minimum' — one-word change in a 13-word sentence; Jaccard = 12/14 ≈ 0.86 < 0.9
    ]
    pack = build_context_pack({"query": "array value", "top_k": 5, "hits": hits},
                              _routed("fact_lookup"), max_chunks=5)
    assert len(pack["selected_chunks"]) == 2, (
        "Both chunks must survive deduplication. "
        "The near-dup threshold is 0.90; these texts have Jaccard ≈ 0.86."
    )
    _ok("C", "short_distinct_not_deduped")


# ─────────────────────────────────────────────────────────────────────────────
# Section D — Context Pack Quality
# Tests diversification (with score gap guard), authority weighting, and
# selection metadata completeness.
# ─────────────────────────────────────────────────────────────────────────────

def test_d1_diversification_fires_on_comparison_intent() -> None:
    """For comparison intent, a mono-source selection is diversified if a suitable alt exists."""
    _t("D", "diversification_fires")
    h1 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Approach A uses gradient descent.", score=0.82)
    h2 = _mk_hit(chunk_id="d1-c000002", doc_id="d1", text="Approach A converges slowly.", score=0.80)
    h3 = _mk_hit(chunk_id="d2-c000001", doc_id="d2", text="Approach B uses Newton's method.", score=0.78)
    pack = build_context_pack(
        {"query": "Compare optimisation approaches", "top_k": 2, "hits": [h1, h2, h3]},
        _routed("comparison"), max_chunks=2,
    )
    doc_ids = {h["doc_id"] for h in pack["selected_chunks"]}
    assert len(doc_ids) >= 2, "Diversification must inject a second-document chunk."
    assert pack["selection_meta"]["conditional_diversification_applied"] is True
    _ok("D", "diversification_fires")


def test_d2_score_gap_blocks_diversification_when_alt_too_weak() -> None:
    """If the best alt chunk is more than max_score_gap below the tail, no swap occurs.

    max_score_gap default is 0.06. Alt at 0.80 - 0.20 = 0.60 gap → blocked.
    """
    _t("D", "score_gap_blocks_diversification")
    h1 = _mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Topic explanation part one.", score=0.82)
    h2 = _mk_hit(chunk_id="d1-c000002", doc_id="d1", text="Topic explanation part two.", score=0.80)
    h_weak = _mk_hit(chunk_id="d2-c000001", doc_id="d2", text="Unrelated distant topic.", score=0.55)
    # Gap from tail (0.80) to alt (0.55) = 0.25 >> 0.06 → diversification must NOT fire
    pack = build_context_pack(
        {"query": "Compare approaches", "top_k": 2, "hits": [h1, h2, h_weak]},
        _routed("comparison"), max_chunks=2,
    )
    assert pack["selection_meta"]["conditional_diversification_applied"] is False, (
        "Diversification must be blocked when the alt candidate score gap exceeds the threshold. "
        "Sacrificing quality for forced diversity is incorrect."
    )
    _ok("D", "score_gap_blocks_diversification")


def test_d3_authority_weighting_lifts_pdf_over_notes() -> None:
    """pdf_book has higher authority than notes; pdf chunk should rank above notes at equal vector score."""
    _t("D", "authority_pdf_over_notes")
    notes_hit = _mk_hit(chunk_id="n-c000001", doc_id="n1",
                        text="Regularisation prevents overfitting by penalising large weights.",
                        score=0.80, source_type="notes")
    pdf_hit   = _mk_hit(chunk_id="p-c000001", doc_id="p1",
                        text="Regularisation prevents overfitting by penalising large weights.",
                        score=0.80, source_type="pdf_book")
    pack = build_context_pack(
        {"query": "regularisation", "top_k": 1, "hits": [notes_hit, pdf_hit]},
        _routed("fact_lookup"), max_chunks=1,
    )
    assert pack["selected_chunks"][0]["chunk_id"] == "p-c000001", (
        "pdf_book chunk must rank above notes chunk at equal vector score. "
        "Source authority weighting is not applied."
    )
    _ok("D", "authority_pdf_over_notes")


def test_d4_selection_meta_contains_required_keys() -> None:
    """selection_meta must contain all keys required by downstream answer generation."""
    _t("D", "selection_meta_completeness")
    required = {
        "original_hit_count", "deduped_hit_count", "selected_count",
        "deduplication", "authority_weighting_applied",
        "conditional_diversification_applied", "route_signal",
    }
    hits = [_mk_hit(chunk_id="d1-c000001", doc_id="d1", text="Example content.", score=0.75)]
    pack = build_context_pack({"query": "test", "top_k": 1, "hits": hits},
                              _routed("fact_lookup"), max_chunks=1)
    missing = required - pack["selection_meta"].keys()
    assert not missing, f"selection_meta is missing required keys: {missing}"
    _ok("D", "selection_meta_completeness")


def test_d5_max_chunks_is_strictly_respected() -> None:
    """build_context_pack must never return more than max_chunks selected chunks."""
    _t("D", "max_chunks_respected")
    hits = [
        _mk_hit(chunk_id=f"d{i}-c000001", doc_id=f"d{i}",
                text=f"Distinct content paragraph {i} about machine learning.",
                score=0.9 - i * 0.05)
        for i in range(8)
    ]
    for k in (1, 3, 5):
        pack = build_context_pack({"query": "machine learning", "top_k": k, "hits": hits},
                                  _routed("summary"), max_chunks=k)
        assert len(pack["selected_chunks"]) <= k, (
            f"max_chunks={k} must be respected; got {len(pack['selected_chunks'])} chunks."
        )
    _ok("D", "max_chunks_respected")


# ─────────────────────────────────────────────────────────────────────────────
# Section E — Retrieval Filters & Scoping
# ─────────────────────────────────────────────────────────────────────────────

def test_e1_doc_id_filter_excludes_other_documents(retrieval_db: str) -> None:
    """Filtering by doc_id must return only chunks from that document."""
    _t("E", "doc_id_filter")
    result = retrieve(
        "hydraulic pressure",
        db_dsn=retrieval_db,
        top_k=10,
        filters=RetrievalFilters(doc_id="eng"),
        embed_backend="_test", embed_dimension=4096,
        hyde_enabled=False, bm25_enabled=True,
        include_neighbors=False,
    )
    assert all(h.doc_id == "eng" for h in result.hits), (
        "doc_id filter must exclude all chunks from other documents."
    )
    _ok("E", "doc_id_filter")


def test_e2_path_prefix_filter_restricts_to_section(retrieval_db: str) -> None:
    """path_prefix filter must include only chunks whose path_text starts with the prefix."""
    _t("E", "path_prefix_filter")
    result = retrieve(
        "hydraulic pump",
        db_dsn=retrieval_db,
        top_k=10,
        filters=RetrievalFilters(path_prefix="Spec > Hydraulic"),
        embed_backend="_test", embed_dimension=4096,
        hyde_enabled=False, bm25_enabled=True,
        include_neighbors=False,
    )
    assert len(result.hits) > 0, "Should find hydraulic chunks."
    assert all("Hydraulic" in (h.path_text or "") for h in result.hits), (
        "path_prefix filter must exclude chunks outside the specified section."
    )
    _ok("E", "path_prefix_filter")


def test_e3_top_k_is_respected_exactly(retrieval_db: str) -> None:
    """top_k=1 must return at most 1 hit even when many candidates match."""
    _t("E", "top_k_respected")
    result = retrieve(
        "training model",
        db_dsn=retrieval_db,
        top_k=1,
        embed_backend="_test", embed_dimension=4096,
        hyde_enabled=False, bm25_enabled=True,
        include_neighbors=False,
    )
    assert len(result.hits) <= 1, f"top_k=1 must return at most 1 hit; got {len(result.hits)}."
    _ok("E", "top_k_respected")


def test_e4_bm25_surfaces_exact_keyword_match(retrieval_db: str) -> None:
    """BM25-enabled retrieve must surface the chunk whose text best matches the query terms."""
    _t("E", "bm25_surfaces_match")
    result = retrieve(
        "hydraulic pressure rated operating limit",
        db_dsn=retrieval_db,
        top_k=4,
        filters=RetrievalFilters(doc_id="eng"),
        embed_backend="_test", embed_dimension=4096,
        hyde_enabled=False, bm25_enabled=True,
        include_neighbors=False,
    )
    assert len(result.hits) > 0, "BM25 must return at least one hit."
    chunk_ids = [h.chunk_id for h in result.hits]
    assert "eng-c000000" in chunk_ids, (
        "The chunk containing 'hydraulic pressure rated operating limit' must be retrieved."
    )
    _ok("E", "bm25_surfaces_match")


def test_e5_nonmatching_doc_filter_returns_empty(retrieval_db: str) -> None:
    """A doc_id filter for a non-existent document must return an empty hit list."""
    _t("E", "empty_on_no_match")
    result = retrieve(
        "hydraulic pressure",
        db_dsn=retrieval_db,
        top_k=5,
        filters=RetrievalFilters(doc_id="nonexistent_doc_xyz"),
        embed_backend="_test", embed_dimension=4096,
        hyde_enabled=False, bm25_enabled=True,
        include_neighbors=False,
    )
    assert len(result.hits) == 0, (
        f"Filter for non-existent doc must return empty hits; got {len(result.hits)}."
    )
    _ok("E", "empty_on_no_match")


# ─────────────────────────────────────────────────────────────────────────────
# Section F — Edge Cases & Robustness
# Professional RAG is robust to degenerate inputs. Naive implementations crash.
# ─────────────────────────────────────────────────────────────────────────────

def test_f1_empty_query_does_not_raise(retrieval_db: str) -> None:
    """retrieve('') must not raise; it should return an empty or minimal result."""
    _t("F", "empty_query_no_crash")
    try:
        result = retrieve(
            "",
            db_dsn=retrieval_db,
            top_k=3,
            embed_backend="_test", embed_dimension=4096,
            hyde_enabled=False, bm25_enabled=False,
            include_neighbors=False,
        )
        assert hasattr(result, "hits")
    except Exception as exc:
        pytest.fail(f"retrieve('') raised {type(exc).__name__}: {exc}")
    _ok("F", "empty_query_no_crash")


def test_f2_build_context_pack_with_empty_hits() -> None:
    """build_context_pack with zero hits must return valid structure, not crash."""
    _t("F", "empty_hits_pack")
    pack = build_context_pack({"query": "something", "top_k": 3, "hits": []},
                              _routed("fact_lookup"), max_chunks=3)
    assert pack["selected_chunks"] == []
    assert isinstance(pack["selection_meta"], dict)
    _ok("F", "empty_hits_pack")


def test_f3_bm25_scores_with_zero_chunks() -> None:
    """_bm25_scores on an empty row list must return empty lists, not crash."""
    _t("F", "bm25_empty_corpus")
    ids, scores = _bm25_scores([], ["transformer", "attention"])
    assert ids == []
    assert scores == []
    _ok("F", "bm25_empty_corpus")


def test_f4_rerank_with_single_chunk_is_stable() -> None:
    """rerank_by_query with a single-chunk input must return that chunk unchanged."""
    _t("F", "rerank_single_chunk")
    hits = [_mk_hit(chunk_id="only-c000001", doc_id="only",
                    text="The only chunk in the result set.", score=0.75)]
    ranked = rerank_by_query("only chunk result", hits)
    assert len(ranked) == 1
    assert ranked[0]["chunk_id"] == "only-c000001"
    _ok("F", "rerank_single_chunk")


def test_f5_bm25_scores_all_mismatch_returns_noncrash() -> None:
    """Query with zero matches in corpus returns 0.0 scores without raising."""
    _t("F", "bm25_zero_match")
    rows = [
        _mk_row("c0", "The butterfly population migrates to warmer climates."),
        _mk_row("c1", "Annual rainfall affects agricultural yield and crop rotation."),
    ]
    ids, scores = _bm25_scores(rows, ["eigenvalue", "backpropagation", "transformer"])
    assert len(ids) == 2
    # BM25 returns 0.0 for terms not in the corpus; must not raise
    assert all(isinstance(s, float) for s in scores)
    _ok("F", "bm25_zero_match")


# ─────────────────────────────────────────────────────────────────────────────
# Score Summary — always runs last (file-order; put nothing after this)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Section G — Query Routing & Intent Detection
# Professional RAG routes conversational queries without touching the index,
# detects temporal/volatile topics and routes them to the web, maps each
# intent to a calibrated retrieval strategy, and exposes source-type filters.
# A naive router sends every query to the corpus.
# ─────────────────────────────────────────────────────────────────────────────

def test_g1_conversational_regex_skips_retrieval() -> None:
    """Greeting queries must never touch the corpus (skip_retrieval=True).

    A naive router would embed the greeting and run a vector search,
    wasting VRAM and producing irrelevant chunks.
    """
    _t("G", "conversational_skips_retrieval")
    for greeting in ("Hi there!", "Hello!", "Hey, how are you?", "Good morning"):
        rq = route_query(greeting)
        assert rq.intent == "conversational", (
            f"Greeting {greeting!r} must be classified as 'conversational'; got {rq.intent!r}."
        )
        assert rq.strategy.skip_retrieval is True, (
            f"Greeting {greeting!r} must set skip_retrieval=True."
        )
    _ok("G", "conversational_skips_retrieval")


def test_g2_conv_meta_skips_retrieval() -> None:
    """Queries referencing previous conversation turns must skip corpus retrieval."""
    _t("G", "conv_meta_skips_retrieval")
    for q in (
        "What did you just say?",
        "What was my last question?",
        "can we recap this conversation",
    ):
        rq = route_query(q)
        assert rq.intent in ("conversational", "conversational_meta"), (
            f"{q!r} must be a conversational variant; got {rq.intent!r}."
        )
        assert rq.strategy.skip_retrieval is True, (
            f"{q!r} must set skip_retrieval=True."
        )
    _ok("G", "conv_meta_skips_retrieval")


def test_g3_source_type_notes_detected() -> None:
    """Explicit 'my notes' phrase must set source_type_filter='notes', not 'pdf_book'."""
    _t("G", "source_type_notes")
    assert classify_source_type("what do my notes say about transformers") == "notes"
    assert classify_source_type("from my notes, explain gradient descent") == "notes"
    # No scoping phrase → None (let default collection apply)
    assert classify_source_type("explain gradient descent") is None
    _ok("G", "source_type_notes")


def test_g4_temporal_volatile_triggers_web() -> None:
    """Queries with temporal + volatile-topic signals must route to the web.

    A naive router has no temporal awareness and would search the static corpus,
    returning stale data for live-price or live-event queries.
    """
    _t("G", "temporal_volatile_web")
    # 'current' (soft temporal) + 'stock' / 'price' (volatile topic)
    needs, reason = _needs_web("what is the current stock price for Apple")
    assert needs is True, "'current stock price' must trigger web routing."
    assert reason is not None

    # 'today' alone is a strong temporal signal
    needs2, _ = _needs_web("what happened today in the markets")
    assert needs2 is True, "'today' is a strong temporal signal that must trigger web routing."

    # Phrased conversationally — "I was hoping to talk about X" is a chat
    # opener, NOT a live-data lookup.  The word 'current' here means
    # 'present-day' in a general sense; the corpus is irrelevant but this
    # does not require a web search either.
    needs3, _ = _needs_web("I was hoping to talk about the current political atmosphere in the US")
    assert needs3 is False, (
        "Conversationally framed queries must NOT force web routing — "
        "'current' as a general adjective is not a live-data signal."
    )

    # 'current' (soft temporal) + 'policy' (volatile topic)
    needs4, _ = _needs_web("what is the current government policy on tariffs")
    assert needs4 is True, "'current government policy' must trigger web routing."

    # Purely academic query — must NOT trigger web
    needs5, _ = _needs_web("explain gradient descent with momentum")
    assert needs5 is False, "Academic query must not trigger web routing."
    _ok("G", "temporal_volatile_web")


def test_g5_strategy_map_calibrated_per_intent() -> None:
    """Each intent has a dedicated retrieval strategy with sensible defaults.

    A naive router uses the same top_k / alpha for every intent.
    A professional system calibrates per-intent:
      - formula_lookup: prefer_shorter=True (exact notation, not prose)
      - comparison:     top_k >= 6 (needs multiple sources)
      - conversational: skip_retrieval=True (no index access)
    """
    _t("G", "strategy_map_calibrated")
    formula = _STRATEGY_MAP["formula_lookup"]
    assert formula.prefer_shorter is True, (
        "formula_lookup must use prefer_shorter=True to surface notation chunks."
    )
    assert formula.top_k <= 5, (
        "formula_lookup needs tight precision; top_k should be <= 5."
    )
    comparison = _STRATEGY_MAP["comparison"]
    assert comparison.top_k >= 6, (
        "comparison needs broad coverage; top_k must be >= 6."
    )
    conv = _STRATEGY_MAP["conversational"]
    assert conv.skip_retrieval is True, (
        "conversational must skip retrieval entirely."
    )
    _ok("G", "strategy_map_calibrated")


# ─────────────────────────────────────────────────────────────────────────────
# Section H — Pipeline Chunking
# A professional chunker emits sequential deterministic chunk IDs, injects
# the section title into each chunk, never produces empty chunks, and splits
# oversized sections while preserving paragraph boundaries.
# ─────────────────────────────────────────────────────────────────────────────

def _mk_structured(sections: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "source_path": "/tmp/doc.pdf",
        "metadata": {"doc_id": "testdoc"},
        "sections": sections,
    }


def test_h1_chunk_ids_are_sequential_and_formatted() -> None:
    """Chunk IDs must follow the `{doc_id}-c{idx:06d}` format, starting at 0."""
    _t("H", "sequential_chunk_ids")
    structured = _mk_structured([
        {"section_id": "s1", "title": "Intro",   "content": "This is section one.", "level": 1},
        {"section_id": "s2", "title": "Methods", "content": "This is section two.", "level": 2},
        {"section_id": "s3", "title": "Results", "content": "This is section three.", "level": 2},
    ])
    chunks = chunk_structured_document(structured, include_heading_in_chunk=False)
    ids = [c["chunk_id"] for c in chunks]
    assert ids == ["testdoc-c000000", "testdoc-c000001", "testdoc-c000002"], (
        f"Expected sequential IDs ['testdoc-c000000', ...]; got {ids}"
    )
    _ok("H", "sequential_chunk_ids")


def test_h2_title_prepended_when_include_heading_true() -> None:
    """With include_heading_in_chunk=True every chunk must start with the section title."""
    _t("H", "title_prepended")
    structured = _mk_structured([
        {"section_id": "s1", "title": "Backpropagation Algorithm",
         "content": "The algorithm iteratively computes gradients.", "level": 2},
    ])
    chunks = chunk_structured_document(structured, include_heading_in_chunk=True)
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Backpropagation Algorithm"), (
        f"Chunk text must start with the section title; got: {chunks[0]['text'][:80]!r}"
    )
    _ok("H", "title_prepended")


def test_h3_empty_section_produces_no_chunk() -> None:
    """Sections with empty or whitespace-only content must be silently skipped."""
    _t("H", "empty_section_skipped")
    structured = _mk_structured([
        {"section_id": "s1", "title": "Empty",    "content": "",    "level": 1},
        {"section_id": "s2", "title": "Whitespace", "content": "   ", "level": 1},
        {"section_id": "s3", "title": "Real",     "content": "Actual content appears here.", "level": 2},
    ])
    chunks = chunk_structured_document(structured)
    assert len(chunks) == 1, (
        f"Empty sections must not produce chunks; got {len(chunks)} chunks."
    )
    assert chunks[0]["section_id"] == "s3"
    _ok("H", "empty_section_skipped")


def test_h4_long_section_splits_into_multiple_chunks() -> None:
    """A section exceeding max_tokens must be split into >= 2 chunks."""
    _t("H", "long_section_split")
    # Build a content block well over the 400-token default max
    paragraph = " ".join(["The quick brown fox jumps over the lazy dog."] * 12)
    long_content = "\n\n".join([paragraph] * 8)  # 8 paragraphs ~ 960 words
    structured = _mk_structured([
        {"section_id": "s1", "title": "Long Section", "content": long_content, "level": 2},
    ])
    chunks = chunk_structured_document(structured, max_tokens=100)
    assert len(chunks) >= 2, (
        f"Long section must produce >= 2 chunks at max_tokens=100; got {len(chunks)}."
    )
    # No chunk should massively exceed the token budget
    for c in chunks:
        assert c["token_count_est"] < 400, (
            f"No chunk should exceed 400 tokens; got {c['token_count_est']}."
        )
    _ok("H", "long_section_split")


def test_h5_token_estimate_is_conservative() -> None:
    """estimate_tokens must scale with content and use a conservative char-floor.

    A naive implementation just counts whitespace-separated words.
    The professional implementation also checks chars/3 as a floor, preventing
    gross underestimates for table rows or formula strings with no spaces.
    """
    _t("H", "token_estimate_conservative")
    short_text = "Hello world."
    long_text = " ".join(["word"] * 100)
    # Longer text must estimate more tokens
    assert estimate_tokens(long_text) > estimate_tokens(short_text), (
        "estimate_tokens must scale with content length."
    )
    # Dense no-space content (table pipe row) must NOT estimate near-zero
    dense = "|col1|col2|col3|" * 30   # 510 chars, no spaces
    est = estimate_tokens(dense)
    # char-based floor: 510 chars / 3 ≈ 170 tokens minimum
    assert est >= 100, (
        f"Dense no-space content (510 chars) must estimate >= 100 tokens; got {est}."
    )
    _ok("H", "token_estimate_conservative")


# ─────────────────────────────────────────────────────────────────────────────
# Section I — Citations & Answer Source Policy
# Professional RAG normalises citation IDs to short form, applies score-window
# selection, strips LLM hallucinated citation formats, and enforces that every
# factual sentence carries at least one inline citation.
# ─────────────────────────────────────────────────────────────────────────────

def test_i1_short_citation_format() -> None:
    """short_citation must extract the zero-padded 6-digit suffix from a standard chunk_id."""
    _t("I", "short_citation_format")
    assert short_citation("mybook-c000042") == "c000042"
    assert short_citation("doc_x-c000001") == "c000001"
    assert short_citation("doc-c000000") == "c000000"
    # Non-standard fallback: take last 12 chars
    result = short_citation("someunknownformat")
    assert result == "someunknownformat"[-12:]
    _ok("I", "short_citation_format")


def test_i2_select_citations_respects_score_window() -> None:
    """Chunks within the score window are selected; outside chunks are excluded.

    Default window is 0.06.  A chunk with score top-0.05 is in; top-0.10 is out
    (unless padding to min_citations forces it in).
    """
    _t("I", "score_window_selection")
    hits = [
        {"chunk_id": "c_top",    "score": 0.90},
        {"chunk_id": "c_in",     "score": 0.85},   # 0.90 - 0.85 = 0.05 ≤ 0.06 → selected
        {"chunk_id": "c_border", "score": 0.84},   # 0.90 - 0.84 = 0.06 ≤ 0.06 → selected
        {"chunk_id": "c_out",    "score": 0.70},   # 0.90 - 0.70 = 0.20 > 0.06 → excluded
    ]
    cfg = {"min_citations": 2, "max_citations": 3, "citation_score_window": 0.06}
    selected = select_citations(hits, cfg)
    assert "c_top" in selected
    assert "c_in" in selected
    assert "c_out" not in selected, (
        f"c_out (gap=0.20) must not be selected; got {selected}."
    )
    _ok("I", "score_window_selection")


def test_i3_normalize_strips_noisy_citation_formats() -> None:
    """Hallucinated citation formats from the LLM must be stripped from the answer.

    Naive post-processing only strips the exact [c000001] format; a professional
    system also strips [chunk_id], [chunk_xyz], 'chunk_1234abcd', etc.
    """
    _t("I", "noisy_citation_stripped")
    noisy = (
        "The model is trained [chunk_id] using backpropagation [CHUNK 00abc]. "
        "Results are stored chunk_1234567890abcdef in memory."
    )
    cleaned = normalize_answer_citations(noisy, [])
    assert "chunk_id" not in cleaned.lower(), "[chunk_id] must be stripped."
    assert "CHUNK" not in cleaned, "[CHUNK ...] must be stripped."
    assert "chunk_1234567890abcdef" not in cleaned, "bare chunk_... ID must be stripped."
    _ok("I", "noisy_citation_stripped")


def test_i4_is_factual_sentence_classification() -> None:
    """Factual sentences (>= 5 tokens OR contains a digit) must be identified.

    A naive system cites every sentence including short fillers; a professional
    one skips non-factual transitions.
    """
    _t("I", "factual_sentence_classification")
    # Long sentence → True
    assert is_factual_sentence(
        "Gradient descent iteratively adjusts weights to minimise the loss function."
    ) is True
    # Short sentence with a digit → True
    assert is_factual_sentence("The year is 2024.") is True
    # Very short, no digit → False
    assert is_factual_sentence("Yes.") is False
    assert is_factual_sentence("In summary.") is False
    _ok("I", "factual_sentence_classification")


def test_i5_ensure_inline_adds_tags_to_uncited_sentences() -> None:
    """ensure_inline_sentence_citations appends default tag to every uncited factual sentence."""
    _t("I", "inline_citation_enforcement")
    answer = (
        "Gradient descent minimises the loss function iteratively. "
        "Each step moves parameters in the direction of steepest descent."
    )
    citation_ids = ["doc-c000001", "doc-c000002"]
    result, added = ensure_inline_sentence_citations(answer, citation_ids)
    assert added >= 1, "At least one citation must be added to uncited factual sentences."
    # Result must contain the short citation tag
    assert "[c000001]" in result, f"Default citation tag must appear in result; got: {result!r}"
    _ok("I", "inline_citation_enforcement")


# ─────────────────────────────────────────────────────────────────────────────
# Section J — Coverage & Confidence Scoring
# A professional RAG classifies query intent before choosing how to answer,
# checks whether the corpus actually covers the topic before generating, and
# maps retrieval confidence to three bands (high / medium / low).
# Naive systems always attempt to answer regardless of coverage.
# ─────────────────────────────────────────────────────────────────────────────

def test_j1_is_overview_query_fires_correctly() -> None:
    """Broad overview/summary queries must be detected and handled differently.

    A naive system treats them like fact-lookups; a professional system expands
    top_k and synthesises across chunks rather than extracting a single fact.
    """
    _t("J", "overview_query_detection")
    # Must fire
    assert is_overview_query("summarize this book") is True
    assert is_overview_query("what is this document about") is True
    assert is_overview_query("give me an overview of chapter three") is True
    # Must NOT fire (specific factual question)
    assert is_overview_query("what is backpropagation") is False
    assert is_overview_query("who invented the transformer architecture") is False
    _ok("J", "overview_query_detection")


def test_j2_is_factoid_query_detection() -> None:
    """Short who/what/when/where questions and metadata lookups are factoid."""
    _t("J", "factoid_query_detection")
    assert is_factoid_query("who wrote this book") is True
    assert is_factoid_query("what is the ISBN?") is True
    assert is_factoid_query("when was this published") is True
    # Long exploratory question → NOT factoid
    assert is_factoid_query(
        "explain in depth how convolutional neural networks work including "
        "the role of pooling layers stride and feature maps"
    ) is False
    _ok("J", "factoid_query_detection")


def test_j3_lexical_coverage_one_when_terms_present() -> None:
    """lexical_coverage_score returns 1.0 when the best chunk covers the query terms."""
    _t("J", "coverage_one_when_present")
    hits = [
        {"text": "Backpropagation computes gradients for each layer using the chain rule."},
        {"text": "Stochastic gradient descent updates parameters on mini-batches."},
    ]
    # Both terms present across chunks
    score = lexical_coverage_score("backpropagation chain rule", hits)
    assert score == 1.0, f"Expected 1.0 when query terms present in chunks; got {score}."
    _ok("J", "coverage_one_when_present")


def test_j4_lexical_coverage_low_when_terms_absent() -> None:
    """lexical_coverage_score returns 0.0 when query terms are missing from all chunks."""
    _t("J", "coverage_zero_when_absent")
    hits = [
        {"text": "Butterfly migration patterns follow seasonal temperature gradients."},
        {"text": "Annual rainfall affects crop rotation in temperate climates."},
    ]
    # Query terms ('eigenvalue', 'decomposition') are completely absent
    score = lexical_coverage_score("eigenvalue decomposition covariance matrix", hits)
    assert score == 0.0, (
        f"Expected 0.0 when query terms absent from all chunks; got {score}."
    )
    _ok("J", "coverage_zero_when_absent")


def test_j5_confidence_band_classification() -> None:
    """determine_confidence_band maps score thresholds to 'high', 'medium', 'low'.

    A professional system uses confidence bands to decide whether to answer,
    add a caveat, or refuse. A naive system always answers regardless.
    """
    _t("J", "confidence_band_classification")
    cfg = {
        "medium_confidence_score": 0.55,
        "high_confidence_score": 0.70,
        "borderline_confidence_score": 0.62,
        "max_ambiguous_gap": 0.03,
        "path_override_min_term_matches": 1,
    }
    high_hit = {"score": 0.85, "structural_role": "body", "title": "Gradient Descent",
                "path_text": "ML > Training", "metadata": {"score_gap_to_second": 0.15}}
    band, detail = determine_confidence_band("gradient descent", [high_hit], cfg)
    assert band == "high", f"Score=0.85, gap=0.15 must produce 'high' band; got {band!r}."

    low_hit = {"score": 0.40, "structural_role": "body", "title": "",
               "path_text": "", "metadata": {"score_gap_to_second": 0.01}}
    band2, _ = determine_confidence_band("butterfly migration", [low_hit], cfg)
    assert band2 == "low", f"Score=0.40 must produce 'low' band; got {band2!r}."
    _ok("J", "confidence_band_classification")


# ─────────────────────────────────────────────────────────────────────────────
# Section K — Grounding & Hallucination Detection
# A professional RAG checks that named entities in the generated answer are
# supported by the source text, uses morphological variants for coverage
# scoring, and returns a 0.0 coverage score for truly off-topic corpora.
# A naive system skips post-generation verification.
# ─────────────────────────────────────────────────────────────────────────────

def test_k1_term_variants_generates_morphological_forms() -> None:
    """term_variants must produce stem forms so morphological matches succeed."""
    _t("K", "term_variants")
    # 'training' → {'training', 'train'} (strip -ing)
    assert "train" in term_variants("training"), (
        "'training' must have 'train' as a variant."
    )
    # 'learning' → {'learning', 'learn'}
    assert "learn" in term_variants("learning")
    # 'networks' → {'networks', 'network'} (strip -s)
    assert "network" in term_variants("networks")
    # Short word: 'go' → no false truncation
    variants = term_variants("go")
    assert "go" in variants
    _ok("K", "term_variants")


def test_k2_term_present_matches_morphological_variant() -> None:
    """term_present_in_text must match when the text contains a morphological variant.

    The query says 'training'; the source text says 'trained'.
    A naive system does an exact substring match and misses this.
    """
    _t("K", "morphological_match")
    # text contains 'trained' (past tense); term is 'training' → variant 'train' matches 'trained'
    # Actually term_variants('training') = {'training', 'train'}, and 'train' in 'trained' → True
    assert term_present_in_text("training", "The model was trained on ImageNet.") is True, (
        "'training' must match 'trained' via morphological variant 'train'."
    )
    # Completely unrelated term must NOT match
    assert term_present_in_text("backpropagation", "The butterfly migrates south.") is False
    _ok("K", "morphological_match")


def test_k3_coverage_zero_for_off_topic_corpus() -> None:
    """lexical_coverage_score must return 0.0 when no chunk covers the query terms."""
    _t("K", "off_topic_coverage_zero")
    off_topic_hits = [
        {"text": "Butterfly populations decline due to habitat loss and pesticide use."},
        {"text": "Migratory birds adapt to temperature changes across seasons."},
        {"text": "Rainfall patterns determine soil moisture and agricultural yield."},
    ]
    score = lexical_coverage_score(
        "eigenvalue decomposition principal component analysis covariance",
        off_topic_hits,
    )
    assert score == 0.0, (
        f"Off-topic corpus must yield coverage 0.0; got {score}."
    )
    _ok("K", "off_topic_coverage_zero")


def test_k4_unsupported_entities_empty_when_grounded() -> None:
    """When all answer entities appear in the source text, no hallucination is flagged."""
    _t("K", "grounded_entities_not_flagged")
    answer = "The Transformer architecture was introduced by Vaswani at Google."
    query = "who invented the transformer"
    hits = [{
        "chunk_id": "doc-c000001",
        "text": "The Transformer architecture was introduced by Vaswani and colleagues at Google.",
        "metadata": {},
    }]
    unsupported = unsupported_answer_entities(answer, query, hits, ["doc-c000001"])
    assert unsupported == [], (
        f"All entities grounded in source; expected []; got {unsupported}."
    )
    _ok("K", "grounded_entities_not_flagged")


def test_k5_hallucinated_entities_flagged() -> None:
    """Named entities in the answer that do NOT appear in source text must be flagged.

    The answer introduces 'Hinton', 'Bengio', 'Stanford', 'Berkeley', 'Werbos' which
    are not in the source.  A professional RAG flags these for review; a naive one
    silently returns them.

    ENTITY_TOKEN_RE: r'\\b(?:[A-Z][a-z]{2,}|[A-Z]{2,10})\\b' — requires >=3-char
    Title-case tokens.  'LeCun' has only 1 lowercase after 'Le' → not matched.
    The function returns [] when unsupported < 3 OR ratio < 0.40, so we supply
    5 clearly hallucinated entities (none appearing in the query or source text).
    """
    _t("K", "hallucinated_entities_flagged")
    # 5 Title-case entities, none in source, none in query
    answer = (
        "The algorithm was popularised by Hinton and Bengio at Stanford. "
        "Berkeley and Werbos also contributed foundational theory. "
        "Bengio published several key papers alongside Hinton."
    )
    query = "who invented attention mechanisms"
    hits = [{
        "chunk_id": "doc-c000001",
        "text": "Attention mechanisms allow models to attend to relevant parts of the input.",
        "metadata": {},
    }]
    unsupported = unsupported_answer_entities(answer, query, hits, ["doc-c000001"])
    # At least 3 hallucinated entities (Hinton, Bengio, Stanford / Berkeley / Werbos)
    assert len(unsupported) >= 3, (
        "Named entities not in source text ('Hinton', 'Bengio', 'Stanford', ...) "
        f"must be flagged as hallucinations; got {unsupported!r}."
    )
    _ok("K", "hallucinated_entities_flagged")


# ─────────────────────────────────────────────────────────────────────────────
# Section L — Text Utilities & Normalization
# tokenize() and tokenize_all() have different semantics that are critical to
# correctness; the wrong choice causes BM25 inconsistencies or near-dup bugs.
# clean_markdown handles PDF extraction artifacts that corrupt chunk quality.
# ─────────────────────────────────────────────────────────────────────────────

def test_l1_tokenize_excludes_digit_start_and_single_char() -> None:
    """tokenize() must exclude digit-start tokens and single-character tokens.

    '24V' starts with a digit → excluded (consistent with BM25 query tokenization).
    'A' is a single char → excluded by the regex ('[A-Za-z][A-Za-z0-9_\\-.]+' requires >= 2 chars).
    'DC' has 2 chars, letter-start → included.
    """
    _t("L", "tokenize_boundaries")
    tokens = tokenize("The 24V DC motor system has type A connector.")
    assert "24V" not in tokens, "digit-start token '24V' must be excluded."
    assert "A" not in tokens,   "single-char token 'A' must be excluded."
    assert "DC" in tokens,      "two-char letter-start token 'DC' must be included."
    assert "motor" in tokens,   "normal word 'motor' must be included."
    _ok("L", "tokenize_boundaries")


def test_l2_tokenize_all_includes_digit_start_and_single_char() -> None:
    """tokenize_all() must include digit-start and single-char tokens.

    Used for near-dup detection where 'Topic A' vs 'Topic B' must not deduplicate.
    """
    _t("L", "tokenize_all_inclusive")
    tokens = tokenize_all("The 24V DC motor system has type A connector.")
    assert "24V" in tokens, "digit-start token '24V' must be included by tokenize_all."
    assert "A" in tokens,   "single-char token 'A' must be included by tokenize_all."
    assert "DC" in tokens
    assert "motor" in tokens
    _ok("L", "tokenize_all_inclusive")


def test_l3_tokenize_on_empty_and_whitespace_input() -> None:
    """tokenize and tokenize_all must handle empty/None/whitespace without raising."""
    _t("L", "tokenize_empty_safe")
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize_all("") == []
    assert tokenize_all("   ") == []
    _ok("L", "tokenize_empty_safe")


def test_l4_clean_markdown_joins_soft_hyphen_wordbreaks() -> None:
    """Hyphenated word-breaks from PDF extraction must be joined across lines.

    PDF text extraction often breaks 'back-\npropagation' across lines.
    A naive cleaner leaves the hyphen; a professional one joins the word.
    """
    _t("L", "clean_markdown_hyphen_join")
    raw = "The back-\npropagation algorithm computes gradients layer by layer."
    cleaned = clean_markdown(raw)
    assert "back-\npropagation" not in cleaned, "Hyphen+newline must be joined."
    assert "backpropagation" in cleaned, f"Word must be rejoined; got: {cleaned!r}"
    _ok("L", "clean_markdown_hyphen_join")


def test_l5_clean_markdown_normalizes_bullets_and_blank_lines() -> None:
    """Non-standard bullet characters must be normalised to '- ' and excess blank lines collapsed."""
    _t("L", "clean_markdown_artifacts")
    raw = "\n".join([
        "• First item with bullet",
        "◦ Second item",
        "▪ Third item",
        "",
        "",
        "",
        "",
        "Paragraph after four blank lines.",
    ])
    cleaned = clean_markdown(raw)
    # All bullet variants → '- '
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped.startswith("-") or any(
            stripped.startswith(c) for c in ["•", "◦", "▪", "►", "▶"]
        ):
            assert stripped.startswith("- "), (
                f"Bullet must be normalised to '- '; got {stripped!r}"
            )
    # No run of 3+ consecutive blank lines
    assert "\n\n\n" not in cleaned, (
        "Three or more consecutive blank lines must be collapsed."
    )
    _ok("L", "clean_markdown_artifacts")


PROFESSIONAL_THRESHOLD = 0.85
SECTION_MINIMUM = 0.60


def test_z_score_summary() -> None:
    """Print section-by-section scores and assert the professional-grade threshold."""
    _t("Z", "summary")  # sentinel

    width = 60
    lines: List[str] = []
    lines.append("+" + "=" * width + "+")
    lines.append("|" + "  PROFESSIONAL RAG EVALUATION HARNESS".center(width) + "|")
    lines.append("+" + "-" * width + "+")

    total_pass = total_reg = 0
    section_failures: List[str] = []

    for sec in sorted(k for k in _SECTION_LABELS):
        label = _SECTION_LABELS[sec]
        registered = _REGISTERED.get(sec, [])
        passed = sum(1 for name in registered if (sec, name) in _PASSED)
        reg = len(registered)
        total_pass += passed
        total_reg += reg
        pct = int(100 * passed / reg) if reg else 0
        tick = "PASS" if (reg == 0 or passed / reg >= SECTION_MINIMUM) else "FAIL"
        bar = f"  {sec}  {label}  {passed:2d} / {reg:2d}  ({pct:3d}%)  [{tick}]"
        lines.append("|" + bar.ljust(width) + "|")
        if reg > 0 and passed / reg < SECTION_MINIMUM:
            section_failures.append(f"Section {sec} ({label.strip()}): {passed}/{reg}")

    overall_pct = int(100 * total_pass / total_reg) if total_reg else 0
    verdict = "PROFESSIONAL GRADE" if (
        total_reg > 0
        and total_pass / total_reg >= PROFESSIONAL_THRESHOLD
        and not section_failures
    ) else "NOT YET PROFESSIONAL"
    tick = "PASS" if verdict == "PROFESSIONAL GRADE" else "FAIL"

    lines.append("+" + "-" * width + "+")
    overall_bar = f"  OVERALL  {total_pass:2d} / {total_reg:2d}  ({overall_pct:3d}%)  [{tick}]"
    lines.append("|" + overall_bar.ljust(width) + "|")
    verdict_bar = f"  VERDICT: {verdict}  (threshold: {int(PROFESSIONAL_THRESHOLD*100)}%)"
    lines.append("|" + verdict_bar.ljust(width) + "|")
    lines.append("+" + "=" * width + "+")

    report = "\n".join(lines)
    print("\n" + report)

    problems: List[str] = []
    if total_reg > 0 and total_pass / total_reg < PROFESSIONAL_THRESHOLD:
        problems.append(
            f"Overall score {total_pass}/{total_reg} ({overall_pct}%) "
            f"is below the {int(PROFESSIONAL_THRESHOLD*100)}% professional threshold."
        )
    for sf in section_failures:
        problems.append(f"Section below {int(SECTION_MINIMUM*100)}% minimum: {sf}")

    assert not problems, "\n" + report + "\n\nFailures:\n" + "\n".join(problems)
    _ok("Z", "summary")
