"""
HuggingFace inference service — runs on the Ubuntu box.

Exposes two endpoints:
  POST /classify_intent   — DeBERTa zero-shot NLI intent classification
  POST /rerank            — cross-encoder reranking (ms-marco-MiniLM-L-6-v2)

Start with:
  source .venv/bin/activate
  uvicorn server:app --host 0.0.0.0 --port 8100

The RAG app on Windows points to this via INFERENCE_SERVICE_URL in runtime.yaml.
"""

from __future__ import annotations

import gc
import json as _json
import logging
import os
import threading
import urllib.request as _ur
import warnings
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

warnings.filterwarnings("ignore", message=".*pipelines sequentially.*")
logger = logging.getLogger("inference_service")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Inference Service")

# ── Device ─────────────────────────────────────────────────────────────────────
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("inference_service: using device %s", _DEVICE)

# ── DeBERTa intent classifier ──────────────────────────────────────────────────
# Override with env vars to run different models on different machines:
#   INFERENCE_NLI_MODEL    — zero-shot NLI model (HuggingFace model ID)
#   INFERENCE_CE_MODEL     — cross-encoder model (HuggingFace model ID)
#   INFERENCE_OLLAMA_MODEL — Ollama model to pin in VRAM on this machine
#   INFERENCE_OLLAMA_URL   — Ollama API base URL on this machine
#   INFERENCE_VRAM_KEEP_MB — free VRAM threshold below which HF models are
#                            offloaded to CPU after each request (default 1500).
#                            Set to 0 to keep models warm always.
_MODEL_ID    = os.environ.get("INFERENCE_NLI_MODEL", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0")
_CE_MODEL_ID = os.environ.get("INFERENCE_CE_MODEL",  "cross-encoder/ms-marco-MiniLM-L-6-v2")

_LABEL_MAP: dict[str, str] = {
    "metadata_lookup":     "look up bibliographic or publication metadata such as author, title, publisher, or ISBN",
    "formula_lookup":      "find or derive a specific mathematical formula, equation, or numerical calculation",
    "section_lookup":      "locate a specific chapter, section, or named part of a document",
    "comparison":          "compare or contrast two or more distinct concepts, methods, or items",
    "list_lookup":         "enumerate or list multiple items, types, steps, examples, or categories",
    "summary":             "get a broad, general overview or explanation of a whole topic, concept, or document — such as 'tell me about X', 'explain X', or 'describe X'",
    "fact_lookup":         "find a single precise, narrow fact or definition in response to a specific question",
    "exploratory":         "investigate an open-ended or multi-part topic in depth across several related questions",
    "conversational_meta": "ask about something said earlier in this conversation, reference previous messages, or ask what was previously asked or answered",
    "conversational":      "a greeting, farewell, acknowledgement, or casual social exchange with no question or information need — such as 'hi', 'thanks', 'okay', 'sounds good', or 'let's chat'",
    "user_profile":        "a question about the user's own personal context, background, role, name, or what the assistant knows about them — such as 'who am I', 'what do you know about me', or 'tell me about my background'",
    "current_data_lookup": "a question requiring real-time or recently changed information — such as live stock prices, current events, breaking news, who currently holds a corporate or political position, or the latest software version",
    "implicit_followup":   "a short follow-up message that refers to a subject from earlier in the conversation without naming it — such as 'what about their CEO?', 'and the CFO?', 'how old is it?', or 'is it profitable?'",
}
_CANDIDATES = list(_LABEL_MAP.values())
_LABELS     = list(_LABEL_MAP.keys())

_HYPOTHESIS_TEMPLATE  = "The intent of this question is to {}."
_CONFIDENCE_THRESHOLD = 0.65
_SECONDARY_THRESHOLD  = 0.15

# ── VRAM headroom manager ─────────────────────────────────────────────────────
# After each request, free VRAM is checked. If it drops below this threshold
# both HF models are moved to CPU and their globals cleared so the next request
# reloads them from disk. On machines with enough headroom they stay warm.
#
# Override: INFERENCE_VRAM_KEEP_MB (default 1500 = 1.5 GB).
# Set to 0 to never offload (always keep warm). Set very high to always offload.
_VRAM_KEEP_MB = int(os.environ.get("INFERENCE_VRAM_KEEP_MB", "1500"))


def _ollama_vram_used_mb() -> int:
    """Return total VRAM (MB) currently consumed by Ollama-loaded models."""
    try:
        req = _ur.Request(f"{_OLLAMA_BASE_URL}/api/ps", method="GET")
        with _ur.urlopen(req, timeout=3) as resp:
            data = _json.loads(resp.read())
        return sum(m.get("size_vram", 0) for m in data.get("models", [])) // (1024 * 1024)
    except Exception:
        return 0


def _vram_stats() -> tuple[int, int, int]:
    """Return (total_mb, hf_allocated_mb, free_mb_before_hf_offload)."""
    if not torch.cuda.is_available():
        return 0, 0, 0
    total_mb  = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    hf_mb     = torch.cuda.memory_allocated(0) // (1024 * 1024)
    ollama_mb = _ollama_vram_used_mb()
    free_mb   = total_mb - ollama_mb - hf_mb
    return total_mb, hf_mb, free_mb


def _offload_hf_models() -> None:
    """Move HF models off GPU → CPU and clear globals. Lazy loaders handle reload."""
    global _nli_pipeline, _ce_model
    offloaded: list[str] = []
    with _nli_lock:
        if _nli_pipeline is not None:
            try:
                _nli_pipeline.model.to("cpu")
            except Exception:
                pass
            _nli_pipeline = None
            offloaded.append("DeBERTa")
    with _ce_lock:
        if _ce_model is not None:
            try:
                _ce_model.model.to("cpu")
            except Exception:
                pass
            _ce_model = None
            offloaded.append("CrossEncoder")
    if offloaded:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "inference_service: offloaded %s from GPU (free VRAM < %d MB)",
            " + ".join(offloaded), _VRAM_KEEP_MB,
        )


def _check_vram_headroom() -> None:
    """Called in a background thread after each request.
    Offloads HF models if remaining VRAM headroom is below the threshold.
    No-op when VRAM_KEEP_MB == 0 (keep warm always) or no CUDA device.
    """
    if _VRAM_KEEP_MB == 0 or not torch.cuda.is_available():
        return
    total_mb, hf_mb, free_mb = _vram_stats()
    logger.debug(
        "inference_service: VRAM total=%d MB  hf=%d MB  free=%d MB  threshold=%d MB",
        total_mb, hf_mb, free_mb, _VRAM_KEEP_MB,
    )
    if free_mb < _VRAM_KEEP_MB:
        _offload_hf_models()


_nli_pipeline        = None
_nli_lock            = threading.Lock()

def _get_nli_pipeline():
    global _nli_pipeline
    if _nli_pipeline is not None:
        return _nli_pipeline
    with _nli_lock:
        if _nli_pipeline is not None:
            return _nli_pipeline
        from transformers import pipeline as hf_pipeline
        logger.info("Loading DeBERTa model %s onto %s…", _MODEL_ID, _DEVICE)
        _nli_pipeline = hf_pipeline(
            "zero-shot-classification",
            model=_MODEL_ID,
            device=_DEVICE,
        )
        logger.info("DeBERTa loaded.")
        return _nli_pipeline


# ── Cross-encoder reranker ─────────────────────────────────────────────────────
_ce_model     = None
_ce_lock      = threading.Lock()

def _get_cross_encoder():
    global _ce_model
    if _ce_model is not None:
        return _ce_model
    with _ce_lock:
        if _ce_model is not None:
            return _ce_model
        from sentence_transformers import CrossEncoder
        logger.info("Loading CrossEncoder %s…", _CE_MODEL_ID)
        _ce_model = CrossEncoder(_CE_MODEL_ID)
        logger.info("CrossEncoder loaded.")
        return _ce_model


# ── Request / response models ──────────────────────────────────────────────────

class IntentRequest(BaseModel):
    query: str

class IntentResponse(BaseModel):
    primary: Optional[str]
    secondary: Optional[str]
    confidence: float
    all_scores: Dict[str, float]

class RerankRequest(BaseModel):
    query: str
    pairs: List[List[str]]  # [[query, text], …]

class RerankResponse(BaseModel):
    scores: List[float]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    info: dict = {"status": "ok", "device": _DEVICE}
    if torch.cuda.is_available():
        props     = torch.cuda.get_device_properties(0)
        total_mb  = props.total_memory // (1024 * 1024)
        hf_mb     = torch.cuda.memory_allocated(0) // (1024 * 1024)
        ollama_mb = _ollama_vram_used_mb()
        free_mb   = total_mb - ollama_mb - hf_mb
        info["gpu_name"]            = props.name
        info["vram_total_mb"]       = total_mb
        info["vram_hf_mb"]          = hf_mb
        info["vram_ollama_mb"]      = ollama_mb
        info["vram_free_mb"]        = free_mb
        info["vram_keep_threshold"] = _VRAM_KEEP_MB
        info["hf_nli_loaded"]       = _nli_pipeline is not None
        info["hf_ce_loaded"]        = _ce_model is not None
    return info


@app.get("/capabilities")
def capabilities():
    """Return what this inference node is serving — used by service_discovery."""
    total_mb, hf_mb, free_mb = _vram_stats()
    return {
        "service": "rag-inference",
        "device": _DEVICE,
        "endpoints": {
            "classify_intent": {
                "model": _MODEL_ID,
                "loaded": _nli_pipeline is not None,
            },
            "rerank": {
                "model": _CE_MODEL_ID,
                "loaded": _ce_model is not None,
            },
        },
        "ollama": {
            "url":          _OLLAMA_BASE_URL,
            "pinned_model": _OLLAMA_KEEPALIVE_MODEL,
        },
        "vram": {
            "total_mb":    total_mb,
            "hf_mb":       hf_mb,
            "free_mb":     free_mb,
            "keep_threshold_mb": _VRAM_KEEP_MB,
        },
    }


@app.post("/classify_intent", response_model=IntentResponse)
def classify_intent(req: IntentRequest):
    pipe = _get_nli_pipeline()
    result = pipe(
        req.query,
        candidate_labels=_CANDIDATES,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=False,
    )

    scores_by_label: Dict[str, float] = {}
    for label_text, score in zip(result["labels"], result["scores"]):
        idx = _CANDIDATES.index(label_text)
        scores_by_label[_LABELS[idx]] = float(score)

    sorted_labels = sorted(scores_by_label.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_label, second_score = sorted_labels[1] if len(sorted_labels) > 1 else (None, 0.0)

    primary   = top_label if top_score >= _CONFIDENCE_THRESHOLD else None
    secondary = (
        second_label
        if second_label and (top_score - second_score) <= _SECONDARY_THRESHOLD
        else None
    )

    response = IntentResponse(
        primary=primary,
        secondary=secondary,
        confidence=float(top_score),
        all_scores=scores_by_label,
    )
    threading.Thread(target=_check_vram_headroom, daemon=True).start()
    return response


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest):
    model = _get_cross_encoder()
    scores = model.predict(req.pairs)
    response = RerankResponse(scores=[float(s) for s in scores])
    threading.Thread(target=_check_vram_headroom, daemon=True).start()
    return response


# ── Warmup on startup ──────────────────────────────────────────────────────────

_OLLAMA_KEEPALIVE_MODEL = os.environ.get("INFERENCE_OLLAMA_MODEL", "hyde-model")
_OLLAMA_BASE_URL        = os.environ.get("INFERENCE_OLLAMA_URL",   "http://localhost:11434")

def _ollama_keepalive():
    """Ping Ollama with keep_alive=-1 so the HyDE model stays resident in VRAM indefinitely."""
    import json as _json
    import urllib.request as _ur
    payload = _json.dumps({
        "model": _OLLAMA_KEEPALIVE_MODEL,
        "keep_alive": -1,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "options": {"num_predict": 1, "num_ctx": 512},
    }).encode()
    req = _ur.Request(
        f"{_OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _ur.urlopen(req, timeout=120) as resp:
            resp.read()
        logger.info("inference_service: Ollama %s loaded and pinned in VRAM (keep_alive=-1)", _OLLAMA_KEEPALIVE_MODEL)
    except Exception as exc:
        logger.warning("inference_service: Ollama keepalive ping failed: %s", exc)


@app.on_event("startup")
def warmup():
    """Load both HuggingFace models and pin the Ollama HyDE model into VRAM at startup."""
    def _load():
        _get_nli_pipeline()
        _get_cross_encoder()
        _ollama_keepalive()
    threading.Thread(target=_load, daemon=True).start()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8100)
