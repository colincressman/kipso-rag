from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from utils.config import load_yaml_config


_FALLBACK_DEFAULTS: Dict[str, Any] = {
    "paths": {
        "db_dsn": "postgresql://postgres:postgres@localhost/rag",
        "diagnostics_dir": "data/diagnostics",
        "query_trace": "data/diagnostics/query_trace.jsonl",
        "routing_trace": "data/diagnostics/routing_trace.jsonl",
        "book_registry": "data/metadata/book_registry.json",
    },
    "embedding": {
        "backend": "ollama",
        "model_name": "qwen3-embedding",
        "dimension": 384,
        "ollama_base_url": "http://localhost:11434",
        "ollama_timeout_seconds": 60.0,
    },
    "llm": {
        "model": "qwen2.5:3b-instruct",
        "base_url": "http://localhost:11434",
        "timeout_seconds": 180.0,
        "temperature": 0.05,
    },
    "retrieval": {
        "top_k": 5,
        "rerank_candidate_k": 40,
        "alpha_vector": 0.68,
        "alpha_lexical": 0.24,  # Grid-confirmed sweet spot (was 0.32)
        "low_confidence_gap_threshold": 0.03,
        "bm25_enabled": True,
        "bm25_rrf_k": 60,
        "internet_fallback_enabled": True,
        "internet_trigger_on_low_confidence": False,
        "internet_trigger_top_score": 0.72,
        "internet_trigger_gap": 0.03,
        "internet_max_results": 3,
        "internet_max_chunks": 6,
        "internet_timeout_seconds": 8.0,
        "internet_score_weight": 0.72,
        "internet_override_guard_threshold": 0.62,
        "internet_min_relevance_score": 0.42,
    },
    "chunking": {
        "max_tokens": 400,
        "overlap_tokens": 60,
        "min_chunk_tokens": 40,
        "merge_max_tokens_after": 520,
        "max_chars": 4000,
        "overlap_chars": 200,
        "chars_per_page": 3500,
        "oversized_segment_chars": 6000,
        "include_heading_in_chunk": True,
    },
    "logging": {
        "query_trace_max_mb": 20,
        "query_trace_backups": 3,
        "routing_trace_max_mb": 20,
        "routing_trace_backups": 3,
    },
    "routing_llm": {
        "enabled": True,
        "model": "qwen2.5:3b-instruct",
        "base_url": "http://localhost:11434",
        "timeout_seconds": 8.0,
        "temperature": 0.0,
    },
    "provenance": {
        "document_path_mode": "full",
    },
    "index": {
        "source_pattern": "*.chunks.merged.json",
    },
    "hyde": {
        # HyDE is opt-in. Model + base_url inherit from the llm section at load time.
        "enabled": False,
        "temperature": 0.4,
        "timeout_seconds": 15.0,
    },
    "two_stage": {
        # Two-stage retrieval: when HyDE fires, also embed the original query and
        # blend both cosine scores so chunks relevant to either signal surface.
        # alpha = weight on Stage-1 (HyDE) score; (1-alpha) = weight on Stage-2
        # (original query) score.  0.0 = pure original query, 1.0 = pure HyDE.
        "enabled": True,
        "alpha": 0.6,
    },
    # ── Scoring / tuning parameters (from configs/scoring.yaml) ───────────────
    "scoring": {
        "coverage": {
            "min_coverage_score": 0.38,
            "min_lexical_coverage": 0.55,
            "entity_support_threshold": 0.40,
        },
        "citations": {
            "url_noise_penalty": 0.35,
        },
        "context_pack": {
            "near_dup_threshold": 0.90,
            "max_score_gap": 0.06,
            "source_authority": {
                "pdf_book": 1.00,
                "pdf_paper": 0.95,
                "pdf_report": 0.92,
                "docx": 0.90,
                "notes": 0.70,
                "web": 0.72,
                "internet": 0.60,
                "general": 0.50,
            },
        },
        "reranking": {
            "use_cross_encoder": False,
            "use_cross_encoder_only": False,
            "cross_encoder_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "cross_encoder_top_n": 24,
            "cross_encoder_weight": 0.65,
            "keyword_bonus_per_match": 0.05,
            "keyword_bonus_cap": 0.25,
            "header_bonus_per_match": 0.06,
            "header_bonus_cap": 0.24,
            "exact_phrase_bonus": 0.14,
            "diversity_penalty": 0.10,
            "short_stub_penalty": 0.15,
        },
        "internet": {
            "max_article_chars": 12000,
        },
        "inference_service": {
            "url": "",
            "timeout_seconds": 30.0,
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@lru_cache(maxsize=1)
def _load_all_configs() -> Dict[str, Any]:
    """Load runtime.yaml then layer scoring.yaml on top."""
    runtime = load_yaml_config("configs/runtime.yaml", default={})
    scoring = load_yaml_config("configs/scoring.yaml", default={})
    if not isinstance(runtime, dict):
        runtime = {}
    if not isinstance(scoring, dict):
        scoring = {}
    merged = _deep_merge(_FALLBACK_DEFAULTS, runtime)
    # scoring.yaml keys nest under "scoring" to avoid key collisions
    if scoring:
        merged = _deep_merge(merged, {"scoring": scoring})
    return merged


RUNTIME_DEFAULTS = _load_all_configs()

# ── Paths ──────────────────────────────────────────────────────────────────────
DEFAULT_DB_DSN = str(RUNTIME_DEFAULTS["paths"]["db_dsn"])
DEFAULT_DIAGNOSTICS_DIR = str(RUNTIME_DEFAULTS["paths"]["diagnostics_dir"])
DEFAULT_QUERY_TRACE_PATH = str(RUNTIME_DEFAULTS["paths"]["query_trace"])
DEFAULT_ROUTING_TRACE_PATH = str(RUNTIME_DEFAULTS["paths"]["routing_trace"])
DEFAULT_BOOK_REGISTRY_PATH = str(RUNTIME_DEFAULTS["paths"]["book_registry"])

# ── Embedding / agent ──────────────────────────────────────────────────────────
DEFAULT_EMBED_BACKEND = str(RUNTIME_DEFAULTS["embedding"]["backend"])
DEFAULT_EMBED_MODEL_NAME = str(RUNTIME_DEFAULTS["embedding"]["model_name"])
DEFAULT_EMBED_DIMENSION = int(RUNTIME_DEFAULTS["embedding"]["dimension"])
DEFAULT_OLLAMA_BASE_URL = str(RUNTIME_DEFAULTS["embedding"]["ollama_base_url"])
DEFAULT_OLLAMA_TIMEOUT_SECONDS = float(RUNTIME_DEFAULTS["embedding"]["ollama_timeout_seconds"])

# ── LLM ───────────────────────────────────────────────────────────────────────
DEFAULT_LLM_MODEL = str(RUNTIME_DEFAULTS["llm"]["model"])
DEFAULT_LLM_BASE_URL = str(RUNTIME_DEFAULTS["llm"]["base_url"])
DEFAULT_LLM_TIMEOUT_SECONDS = float(RUNTIME_DEFAULTS["llm"]["timeout_seconds"])
DEFAULT_LLM_TEMPERATURE = float(RUNTIME_DEFAULTS["llm"]["temperature"])

# ── Retrieval ─────────────────────────────────────────────────────────────────
DEFAULT_RETRIEVAL_TOP_K = int(RUNTIME_DEFAULTS["retrieval"]["top_k"])
DEFAULT_RERANK_CANDIDATE_K = int(RUNTIME_DEFAULTS["retrieval"]["rerank_candidate_k"])
DEFAULT_RERANK_ALPHA_VECTOR = float(RUNTIME_DEFAULTS["retrieval"]["alpha_vector"])
DEFAULT_RERANK_ALPHA_LEXICAL = float(RUNTIME_DEFAULTS["retrieval"]["alpha_lexical"])
DEFAULT_LOW_CONFIDENCE_GAP_THRESHOLD = float(RUNTIME_DEFAULTS["retrieval"]["low_confidence_gap_threshold"])
DEFAULT_INTERNET_FALLBACK_ENABLED = bool(RUNTIME_DEFAULTS["retrieval"]["internet_fallback_enabled"])
DEFAULT_INTERNET_TRIGGER_ON_LOW_CONFIDENCE = bool(RUNTIME_DEFAULTS["retrieval"]["internet_trigger_on_low_confidence"])
DEFAULT_INTERNET_TRIGGER_TOP_SCORE = float(RUNTIME_DEFAULTS["retrieval"]["internet_trigger_top_score"])
DEFAULT_INTERNET_TRIGGER_GAP = float(RUNTIME_DEFAULTS["retrieval"]["internet_trigger_gap"])
DEFAULT_INTERNET_MAX_RESULTS = int(RUNTIME_DEFAULTS["retrieval"]["internet_max_results"])
DEFAULT_INTERNET_MAX_CHUNKS = int(RUNTIME_DEFAULTS["retrieval"]["internet_max_chunks"])
DEFAULT_INTERNET_TIMEOUT_SECONDS = float(RUNTIME_DEFAULTS["retrieval"]["internet_timeout_seconds"])
DEFAULT_INTERNET_SCORE_WEIGHT = float(RUNTIME_DEFAULTS["retrieval"]["internet_score_weight"])
DEFAULT_INTERNET_OVERRIDE_GUARD_THRESHOLD = float(RUNTIME_DEFAULTS["retrieval"]["internet_override_guard_threshold"])
DEFAULT_INTERNET_MIN_RELEVANCE_SCORE = float(RUNTIME_DEFAULTS["retrieval"]["internet_min_relevance_score"])
DEFAULT_BM25_ENABLED = bool(RUNTIME_DEFAULTS["retrieval"].get("bm25_enabled", True))
DEFAULT_BM25_RRF_K = int(RUNTIME_DEFAULTS["retrieval"].get("bm25_rrf_k", 60))

# ── Chunking ──────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_MAX_TOKENS = int(RUNTIME_DEFAULTS["chunking"]["max_tokens"])
DEFAULT_CHUNK_OVERLAP_TOKENS = int(RUNTIME_DEFAULTS["chunking"]["overlap_tokens"])
DEFAULT_CHUNK_MIN_TOKENS = int(RUNTIME_DEFAULTS["chunking"]["min_chunk_tokens"])
DEFAULT_MERGE_MAX_TOKENS_AFTER = int(RUNTIME_DEFAULTS["chunking"]["merge_max_tokens_after"])
DEFAULT_CHUNK_MAX_CHARS = int(RUNTIME_DEFAULTS["chunking"]["max_chars"])
DEFAULT_CHUNK_OVERLAP_CHARS = int(RUNTIME_DEFAULTS["chunking"]["overlap_chars"])
DEFAULT_CHARS_PER_PAGE = int(RUNTIME_DEFAULTS["chunking"]["chars_per_page"])
DEFAULT_OVERSIZED_SEGMENT_CHARS = int(RUNTIME_DEFAULTS["chunking"]["oversized_segment_chars"])

# ── Logging ───────────────────────────────────────────────────────────────────
DEFAULT_QUERY_TRACE_MAX_MB = int(RUNTIME_DEFAULTS["logging"]["query_trace_max_mb"])
DEFAULT_QUERY_TRACE_BACKUPS = int(RUNTIME_DEFAULTS["logging"]["query_trace_backups"])
DEFAULT_ROUTING_TRACE_MAX_MB = int(RUNTIME_DEFAULTS["logging"]["routing_trace_max_mb"])
DEFAULT_ROUTING_TRACE_BACKUPS = int(RUNTIME_DEFAULTS["logging"]["routing_trace_backups"])

# ── HyDE ─────────────────────────────────────────────────────────────────────
_hyde = RUNTIME_DEFAULTS.get("hyde", {})
DEFAULT_HYDE_ENABLED = bool(_hyde.get("enabled", False))
DEFAULT_HYDE_TEMPERATURE = float(_hyde.get("temperature", 0.4))
DEFAULT_HYDE_TIMEOUT_SECONDS = float(_hyde.get("timeout_seconds", 15.0))
# Model and base_url for HyDE — defaults to main LLM but can be overridden
# to use a smaller model on a separate machine (e.g. Ubuntu inference box).
DEFAULT_HYDE_MODEL = str(_hyde.get("model") or DEFAULT_LLM_MODEL)
DEFAULT_HYDE_BASE_URL = str(_hyde.get("base_url") or DEFAULT_LLM_BASE_URL)

# ── Step-back prompting ───────────────────────────────────────────────────────
_stepback = RUNTIME_DEFAULTS.get("stepback", {})
DEFAULT_STEPBACK_ENABLED = bool(_stepback.get("enabled", False))
DEFAULT_STEPBACK_TIMEOUT_SECONDS = float(_stepback.get("timeout_seconds", 15.0))

# ── Two-stage retrieval ───────────────────────────────────────────────────────
_two_stage = RUNTIME_DEFAULTS.get("two_stage", {})
DEFAULT_TWO_STAGE_ENABLED = bool(_two_stage.get("enabled", True))
# alpha = weight on Stage-1 (HyDE) score; (1-alpha) = weight on Stage-2 (original query).
DEFAULT_TWO_STAGE_ALPHA = float(_two_stage.get("alpha", 0.6))

# ── Routing LLM ───────────────────────────────────────────────────────────────
DEFAULT_ROUTING_LLM_ENABLED = bool(RUNTIME_DEFAULTS["routing_llm"]["enabled"])
DEFAULT_ROUTING_LLM_MODEL = str(RUNTIME_DEFAULTS["routing_llm"]["model"])
DEFAULT_ROUTING_LLM_BASE_URL = str(RUNTIME_DEFAULTS["routing_llm"]["base_url"])
DEFAULT_ROUTING_LLM_TIMEOUT_SECONDS = float(RUNTIME_DEFAULTS["routing_llm"]["timeout_seconds"])
DEFAULT_ROUTING_LLM_TEMPERATURE = float(RUNTIME_DEFAULTS["routing_llm"]["temperature"])

# ── Provenance ────────────────────────────────────────────────────────────────
DEFAULT_DOCUMENT_PATH_MODE = str(RUNTIME_DEFAULTS["provenance"]["document_path_mode"])

# ── Scoring (from configs/scoring.yaml) ──────────────────────────────────────
_scoring = RUNTIME_DEFAULTS.get("scoring", {})
_coverage = _scoring.get("coverage", {})
_citations = _scoring.get("citations", {})
_ctxpack = _scoring.get("context_pack", {})
_rerank = _scoring.get("reranking", {})
_inet = _scoring.get("internet", {})

DEFAULT_MIN_COVERAGE_SCORE = float(_coverage.get("min_coverage_score", 0.38))
DEFAULT_MIN_LEXICAL_COVERAGE = float(_coverage.get("min_lexical_coverage", 0.55))
DEFAULT_ENTITY_SUPPORT_THRESHOLD = float(_coverage.get("entity_support_threshold", 0.40))

DEFAULT_CITATION_URL_NOISE_PENALTY = float(_citations.get("url_noise_penalty", 0.35))

DEFAULT_NEAR_DUP_THRESHOLD = float(_ctxpack.get("near_dup_threshold", 0.90))
DEFAULT_MAX_SCORE_GAP = float(_ctxpack.get("max_score_gap", 0.06))
DEFAULT_SOURCE_AUTHORITY: Dict[str, float] = dict(_ctxpack.get("source_authority", {
    "pdf_book": 1.00, "pdf_paper": 0.95, "pdf_report": 0.92,
    "docx": 0.90, "notes": 0.70, "web": 0.72, "internet": 0.60, "general": 0.50,
}))

DEFAULT_RERANK_KEYWORD_BONUS_PER_MATCH = float(_rerank.get("keyword_bonus_per_match", 0.05))
DEFAULT_RERANK_KEYWORD_BONUS_CAP = float(_rerank.get("keyword_bonus_cap", 0.25))
DEFAULT_RERANK_HEADER_BONUS_PER_MATCH = float(_rerank.get("header_bonus_per_match", 0.06))
DEFAULT_RERANK_HEADER_BONUS_CAP = float(_rerank.get("header_bonus_cap", 0.24))
DEFAULT_RERANK_EXACT_PHRASE_BONUS = float(_rerank.get("exact_phrase_bonus", 0.14))
DEFAULT_RERANK_DIVERSITY_PENALTY = float(_rerank.get("diversity_penalty", 0.10))
DEFAULT_RERANK_SHORT_STUB_PENALTY = float(_rerank.get("short_stub_penalty", 0.15))
DEFAULT_RERANK_USE_CROSS_ENCODER = bool(_rerank.get("use_cross_encoder", False))
DEFAULT_RERANK_USE_CROSS_ENCODER_ONLY = bool(_rerank.get("use_cross_encoder_only", False))
DEFAULT_RERANK_CROSS_ENCODER_MODEL = str(_rerank.get("cross_encoder_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
DEFAULT_RERANK_CROSS_ENCODER_TOP_N = int(_rerank.get("cross_encoder_top_n", 24))
DEFAULT_RERANK_CROSS_ENCODER_WEIGHT = float(_rerank.get("cross_encoder_weight", 0.65))

DEFAULT_INTERNET_MAX_ARTICLE_CHARS = int(_inet.get("max_article_chars", 12000))

_infsvc = RUNTIME_DEFAULTS.get("inference_service", {})
DEFAULT_INFERENCE_SERVICE_URL     = str(_infsvc.get("url", "")).strip()
DEFAULT_INFERENCE_SERVICE_TIMEOUT = float(_infsvc.get("timeout_seconds", 30.0))
DEFAULT_INFERENCE_SERVICE_NLI_MODEL = str(_infsvc.get("nli_model", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"))
DEFAULT_INFERENCE_SERVICE_CE_MODEL  = str(_infsvc.get("ce_model",  "cross-encoder/ms-marco-MiniLM-L-6-v2"))

# ── PRF (pseudo-relevance feedback) ───────────────────────────────────────────────────────
DEFAULT_PRF_ENABLED = bool(RUNTIME_DEFAULTS.get("prf", {}).get("enabled", False))

# ── Contextual compression ─────────────────────────────────────────────────────
_ctxcomp = RUNTIME_DEFAULTS.get("contextual_compression", {})
DEFAULT_CONTEXTUAL_COMPRESSION_ENABLED = bool(_ctxcomp.get("enabled", False))
DEFAULT_CONTEXTUAL_COMPRESSION_TOP_N = int(_ctxcomp.get("top_n", 3))
DEFAULT_CONTEXTUAL_COMPRESSION_TIMEOUT = float(_ctxcomp.get("timeout_seconds", 10.0))

# ── Hypothetical question retrieval ───────────────────────────────────────────
_hypq = RUNTIME_DEFAULTS.get("hyp_questions", {})
DEFAULT_HYP_QUESTIONS_ENABLED = bool(_hypq.get("enabled", False))
DEFAULT_HYP_QUESTIONS_TOP_K = int(_hypq.get("top_k", 10))

