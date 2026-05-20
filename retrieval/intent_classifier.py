"""
Zero-shot intent classifier v3.

Model: MoritzLaurer/deberta-v3-large-zeroshot-v2.0

Change from v2
--------------
``classify_intent_ml`` now returns a **secondary intent** in addition to the
primary.  When the second-ranked label's score is within
``_SECONDARY_THRESHOLD`` (0.15) of the top score, it is returned as
``secondary_intent``; otherwise ``secondary_intent`` is ``None``.

This allows the router to detect compound queries such as
"compare the formulas for X and Y" (primary: comparison, secondary:
formula_lookup) and adjust retrieval mechanics accordingly — e.g. disable HyDE
and boost exact-match when a formula_lookup secondary is detected alongside
any primary intent.

Return signature change
-----------------------
  v2:  classify_intent_ml(query) -> (intent | None, confidence)
  v3:  classify_intent_ml(query) -> (primary | None, secondary | None, confidence)

All other public symbols (warmup, unload, _MODEL_ID, _LABEL_MAP, _CANDIDATES,
_LABELS) are unchanged.
"""
from __future__ import annotations

import logging
import threading
import warnings
from typing import Any, Dict, Optional, Tuple

warnings.filterwarnings(
    "ignore",
    message=".*pipelines sequentially.*",
    category=UserWarning,
)

logger = logging.getLogger(__name__)

# ── Remote inference service ───────────────────────────────────────────────────
# When INFERENCE_SERVICE_URL is set in runtime.yaml the NLI calls are sent to
# the Ubuntu inference box instead of loading DeBERTa locally.

def _remote_classify(query: str) -> Optional[Tuple[Optional[str], Optional[str], float, dict]]:
    """
    Call the remote inference service.  Returns (primary, secondary, confidence,
    scores_dict) on success, or None if the service is unavailable.
    """
    try:
        from utils.service_discovery import get_inference_url  # noqa: PLC0415
        from utils.runtime_defaults import DEFAULT_INFERENCE_SERVICE_TIMEOUT  # noqa: PLC0415
        url = get_inference_url()
        if not url:
            return None
        import httpx  # noqa: PLC0415
        resp = httpx.post(
            f"{url}/classify_intent",
            json={"query": query},
            timeout=DEFAULT_INFERENCE_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("primary"),
            data.get("secondary"),
            float(data.get("confidence", 0.0)),
            {k: float(v) for k, v in (data.get("all_scores") or {}).items()},
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("intent_classifier2: remote service unavailable, falling back to local: %s", exc)
        return None

# ── Model config ───────────────────────────────────────────────────────────────
try:
    from utils.runtime_defaults import DEFAULT_INFERENCE_SERVICE_NLI_MODEL as _MODEL_ID  # noqa: PLC0415
except Exception:
    _MODEL_ID = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"
_DEVICE = "cuda"              # pinned permanently; falls back to "cpu" if unavailable
try:
    from utils.runtime_defaults import RUNTIME_DEFAULTS as _RUNTIME_DEFAULTS  # noqa: PLC0415
    _clf_cfg = (_RUNTIME_DEFAULTS.get("scoring") or {}).get("intent_classification") or {}
    _CONFIDENCE_THRESHOLD: float = float(_clf_cfg.get("confidence_threshold", 0.65))
    _SECONDARY_THRESHOLD:  float = float(_clf_cfg.get("secondary_threshold", 0.15))
except Exception:
    _CONFIDENCE_THRESHOLD = 0.65
    _SECONDARY_THRESHOLD  = 0.15

_HYPOTHESIS_TEMPLATE = "The intent of this question is to {}."

# ── Candidate label map ────────────────────────────────────────────────────────
_LABEL_MAP: dict[str, str] = {
    "metadata_lookup":    "look up bibliographic or publication metadata such as author, title, publisher, or ISBN",
    "formula_lookup":     "find or derive a specific mathematical formula, equation, or numerical calculation",
    "section_lookup":     "locate a specific chapter, section, or named part of a document",
    "comparison":         "compare or contrast two or more distinct concepts, methods, or items",
    "list_lookup":        "enumerate or list multiple items, types, steps, examples, or categories",
    "summary":            "get a broad, general overview or explanation of a whole topic, concept, or document — such as 'tell me about X', 'explain X', or 'describe X'",
    "fact_lookup":        "find a single precise, narrow fact or definition in response to a specific question",
    "exploratory":        "investigate an open-ended or multi-part topic in depth across several related questions",
    "conversational_meta": "ask about something said earlier in this conversation, reference previous messages, or ask what was previously asked or answered",
    "conversational":     "a greeting, farewell, acknowledgement, or casual social exchange with no question or information need — such as 'hi', 'thanks', 'okay', 'sounds good', or 'let's chat'",
    "user_profile":       "a question about the user's own personal context, background, role, name, or what the assistant knows about them — such as 'who am I', 'what do you know about me', or 'tell me about my background'",
    "current_data_lookup": "a question requiring real-time, recently changed, or current-state information — such as live stock prices, current events, breaking news, who currently holds a position or role, the latest version or release of software or firmware, what products a company has recently released or announced, or the current value of a financial indicator such as an interest rate, inflation rate, Federal Reserve funds rate, mortgage rate, or exchange rate",
    "implicit_followup":  "a short follow-up message that refers to a subject from earlier in the conversation without naming it — such as 'what about their CEO?', 'and the CFO?', 'how old is it?', or 'is it profitable?'",
}

_CANDIDATES = list(_LABEL_MAP.values())
_LABELS     = list(_LABEL_MAP.keys())

# ── Singleton pipeline ─────────────────────────────────────────────────────────
_pipeline        = None
_pipeline_lock   = threading.Lock()
_pipeline_failed = False


def _get_pipeline():
    """Return the cached pipeline, loading it on first call (thread-safe)."""
    global _pipeline, _pipeline_failed

    if _pipeline is not None:
        return _pipeline

    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline
        if _pipeline_failed:
            return None

        try:
            import torch
            from transformers import pipeline as hf_pipeline

            device = _DEVICE
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("intent_classifier2: CUDA not available, falling back to CPU")
                device = "cpu"

            logger.info(
                "intent_classifier2: loading %s onto %s (first call, then cached)…",
                _MODEL_ID,
                device,
            )
            _pipeline = hf_pipeline(
                "zero-shot-classification",
                model=_MODEL_ID,
                device=device,
            )
            logger.info("intent_classifier2: model loaded successfully")
            return _pipeline

        except Exception as exc:  # noqa: BLE001
            import traceback
            logger.error(
                "intent_classifier2: failed to load model — heuristic fallback active.\n%s",
                traceback.format_exc(),
            )
            print(f"[intent_classifier2] LOAD ERROR: {exc}")
            traceback.print_exc()
            _pipeline_failed = True
            return None


# ── Public API ─────────────────────────────────────────────────────────────────

def classify_intent_ml(query: str) -> Tuple[Optional[str], Optional[str], float]:
    """
    Classify *query* using the zero-shot NLI model.

    Returns ``(primary_intent, secondary_intent, confidence)`` where:
    - ``primary_intent`` is one of the keys in ``_LABEL_MAP``, or ``None``
      when the model is not loaded or confidence is below
      ``_CONFIDENCE_THRESHOLD``.
    - ``secondary_intent`` is the second-ranked label key when its score is
      within ``_SECONDARY_THRESHOLD`` of the top score; otherwise ``None``.
    - ``confidence`` is the top-label score (0.0 on failure).

    The caller should fall back to structural heuristics when primary is None.
    """
    remote = _remote_classify(query)
    if remote is not None:
        primary, secondary, confidence, _ = remote
        return primary, secondary, confidence

    pipe = _get_pipeline()
    if pipe is None:
        return None, None, 0.0

    q = (query or "").strip()
    if not q:
        return None, None, 0.0

    try:
        result = pipe(
            q,
            _CANDIDATES,
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        top_label_text: str   = result["labels"][0]
        top_score: float      = float(result["scores"][0])

        intent_idx  = _CANDIDATES.index(top_label_text)
        primary_key = _LABELS[intent_idx]

        if top_score < _CONFIDENCE_THRESHOLD:
            logger.debug(
                "intent_classifier2: low confidence %.2f for %r → heuristic fallback",
                top_score,
                query,
            )
            return None, None, top_score

        # Determine secondary intent if second score is close enough.
        secondary_key: Optional[str] = None
        if len(result["labels"]) >= 2:
            second_label_text: str = result["labels"][1]
            second_score: float    = float(result["scores"][1])
            if (top_score - second_score) <= _SECONDARY_THRESHOLD:
                second_idx    = _CANDIDATES.index(second_label_text)
                secondary_key = _LABELS[second_idx]
                logger.debug(
                    "intent_classifier2: secondary=%r gap=%.3f (top=%.3f second=%.3f)",
                    secondary_key,
                    top_score - second_score,
                    top_score,
                    second_score,
                )

        return primary_key, secondary_key, top_score

    except Exception as exc:  # noqa: BLE001
        logger.warning("intent_classifier2: inference error: %s", exc)
        return None, None, 0.0


def classify_intent_full_scores(
    query: str,
) -> Tuple[Optional[str], Optional[str], float, dict]:
    """
    Like ``classify_intent_ml`` but also returns the full label score
    distribution as a ``{intent_label: score}`` dict.

    This is used by the example-based routing blend in router2.py to mix
    soft NLI scores with cosine similarity scores over intent examples,
    producing a better signal when NLI confidence is below the threshold.

    Returns ``(primary|None, secondary|None, confidence, scores_dict)`` where:
    - ``scores_dict`` maps every intent label key to its raw NLI score.
    - An empty dict is returned on pipeline failure.
    """
    remote = _remote_classify(query)
    if remote is not None:
        return remote

    pipe = _get_pipeline()
    if pipe is None:
        return None, None, 0.0, {}

    q = (query or "").strip()
    if not q:
        return None, None, 0.0, {}

    try:
        result = pipe(
            q,
            _CANDIDATES,
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )

        scores_dict: dict = {
            _LABELS[_CANDIDATES.index(label_text)]: float(score)
            for label_text, score in zip(result["labels"], result["scores"])
        }

        top_label_text: str = result["labels"][0]
        top_score: float = float(result["scores"][0])
        intent_idx = _CANDIDATES.index(top_label_text)
        primary_key = _LABELS[intent_idx]

        if top_score < _CONFIDENCE_THRESHOLD:
            return None, None, top_score, scores_dict

        secondary_key: Optional[str] = None
        if len(result["labels"]) >= 2:
            second_label_text: str = result["labels"][1]
            second_score: float = float(result["scores"][1])
            if (top_score - second_score) <= _SECONDARY_THRESHOLD:
                second_idx = _CANDIDATES.index(second_label_text)
                secondary_key = _LABELS[second_idx]

        return primary_key, secondary_key, top_score, scores_dict

    except Exception as exc:  # noqa: BLE001
        logger.warning("intent_classifier2: inference error (full_scores): %s", exc)
        return None, None, 0.0, {}


def warmup() -> bool:
    """
    Pre-load the model now (blocking).  Call at server startup so the first
    real query isn't slow.  Returns True if the model loaded successfully.

    When a remote inference service URL is configured, skips local model
    loading and pings the remote /health endpoint instead.
    """
    try:
        from utils.service_discovery import get_inference_url  # noqa: PLC0415
        from utils.runtime_defaults import DEFAULT_INFERENCE_SERVICE_TIMEOUT  # noqa: PLC0415
        url = get_inference_url()
        if url:
            import httpx  # noqa: PLC0415
            resp = httpx.get(
                f"{url}/health",
                timeout=DEFAULT_INFERENCE_SERVICE_TIMEOUT,
            )
            resp.raise_for_status()
            logger.info("intent_classifier2: remote inference service healthy at %s — skipping local model load", url)
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("intent_classifier2: remote service unreachable during warmup: %s — loading locally", exc)

    pipe = _get_pipeline()
    if pipe is None:
        return False
    try:
        pipe(
            "What did I ask you earlier?",
            [_CANDIDATES[8]],  # conversational_meta candidate
            hypothesis_template=_HYPOTHESIS_TEMPLATE,
            multi_label=False,
        )
        logger.info("intent_classifier2: warmup inference OK")
        return True
    except Exception as exc:  # noqa: BLE001
        import traceback
        logger.error(
            "intent_classifier2: warmup inference failed: %s\n%s",
            exc,
            traceback.format_exc(),
        )
        return True


def unload() -> None:
    """
    Release the pipeline from VRAM immediately.

    Call this before loading any other large model (e.g. the LLM) to free
    GPU memory.  The classifier will reload lazily on the next classify call.
    """
    global _pipeline, _pipeline_failed
    with _pipeline_lock:
        if _pipeline is not None:
            try:
                import torch
                del _pipeline
                _pipeline = None
                _pipeline_failed = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("intent_classifier2: unloaded from VRAM")
            except Exception as exc:  # noqa: BLE001
                logger.warning("intent_classifier2: unload error: %s", exc)
                _pipeline = None
                _pipeline_failed = False
