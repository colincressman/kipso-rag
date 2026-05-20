from __future__ import annotations

import logging
import math
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


def _remote_rerank(pairs: List[Tuple[str, str]]) -> Optional[List[float]]:
    """Call the remote inference service for cross-encoder scores.
    Returns a list of raw scores on success, or None if unavailable.
    """
    try:
        from utils.service_discovery import get_inference_url  # noqa: PLC0415
        from utils.runtime_defaults import DEFAULT_INFERENCE_SERVICE_TIMEOUT  # noqa: PLC0415
        url = get_inference_url()
        if not url:
            return None
        import httpx  # noqa: PLC0415
        resp = httpx.post(
            f"{url}/rerank",
            json={"query": pairs[0][0] if pairs else "", "pairs": [list(p) for p in pairs]},
            timeout=DEFAULT_INFERENCE_SERVICE_TIMEOUT,
        )
        resp.raise_for_status()
        return [float(s) for s in resp.json()["scores"]]
    except Exception as exc:  # noqa: BLE001
        logger.warning("cross_encoder: remote service unavailable, falling back to local: %s", exc)
        return None


def _min_max_normalize(values: Sequence[float]) -> List[float]:
	vals = [float(v) for v in values]
	if not vals:
		return []
	lo = min(vals)
	hi = max(vals)
	if hi <= lo:
		return [0.5 for _ in vals]
	span = hi - lo
	return [(v - lo) / span for v in vals]


def _sigmoid_normalize(values: Sequence[float]) -> List[float]:
	"""Apply sigmoid to each score, mapping raw logits to (0, 1).

	Makes the cross-encoder weight model-agnostic: a weight of 0.65 means the
	same thing regardless of the score range the underlying model produces.
	"""
	return [1.0 / (1.0 + math.exp(-float(v))) for v in values]


_ce_lock: threading.Lock = threading.Lock()
_ce_model: Optional[Any] = None
_ce_model_name: Optional[str] = None


def _load_cross_encoder(model_name: str):
	global _ce_model, _ce_model_name
	if _ce_model is not None and _ce_model_name == model_name:
		return _ce_model
	with _ce_lock:
		if _ce_model is not None and _ce_model_name == model_name:
			return _ce_model
		from sentence_transformers import CrossEncoder  # type: ignore
		_ce_model = CrossEncoder(model_name)
		_ce_model_name = model_name
		return _ce_model


def unload() -> None:
	"""Release the CrossEncoder from memory.  Safe to call at any time."""
	global _ce_model, _ce_model_name
	with _ce_lock:
		if _ce_model is not None:
			try:
				import torch
				del _ce_model
				_ce_model = None
				_ce_model_name = None
				if torch.cuda.is_available():
					torch.cuda.empty_cache()
			except Exception:  # noqa: BLE001
				_ce_model = None
				_ce_model_name = None


def rerank_with_cross_encoder(
	query: str,
	hits: List[Dict[str, Any]],
	*,
	model_name: str,
	top_n: int = 24,
	weight: float = 0.65,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
	"""
	Optional second-stage reranker.

	Returns:
		(reranked_hits, trace)
		- If dependency/model load fails, returns original hits with trace.enabled=False.
	"""
	if not hits:
		return hits, {"enabled": False, "applied": False, "reason": "no_hits"}

	top_n = max(1, int(top_n))
	w = max(0.0, min(1.0, float(weight)))
	_all_active = [dict(h) for h in hits[:top_n]]
	# Exclude structural TOC/index chunks from CE scoring: they score extremely
	# high on keyword matches (they literally list topic headings) but contain
	# no answer content.  They are placed after CE-scored chunks in the result.
	_toc_holdout = [h for h in _all_active if h.get("structural_role") == "toc"]
	active = [h for h in _all_active if h.get("structural_role") != "toc"]
	tail = [dict(h) for h in hits[top_n:]]

	if not active:
		return _toc_holdout + tail, {"enabled": False, "applied": False, "reason": "no_non_toc_hits"}

	pairs = [(query or "", str(h.get("text") or "")) for h in active]
	try:
		raw_scores_remote = _remote_rerank(pairs)
		if raw_scores_remote is not None:
			raw_scores = raw_scores_remote
		else:
			model = _load_cross_encoder(str(model_name))
			raw_scores = model.predict(pairs)
		norm_scores = _sigmoid_normalize([float(s) for s in raw_scores])
	except Exception as exc:
		return hits, {
			"enabled": False,
			"applied": False,
			"reason": "cross_encoder_unavailable",
			"error": str(exc),
			"model": str(model_name),
		}

	for item, ce_score in zip(active, norm_scores):
		base = float(item.get("score", 0.0))
		blended = (1.0 - w) * base + w * float(ce_score)
		item["score"] = blended
		md = dict(item.get("metadata") or {})
		md["cross_encoder_score"] = float(ce_score)
		md["cross_encoder_base_score"] = base
		md["cross_encoder_blend_weight"] = w
		md["cross_encoder_model"] = str(model_name)
		item["metadata"] = md

	active.sort(key=lambda h: float(h.get("score", 0.0)), reverse=True)
	result = active + _toc_holdout + tail

	return result, {
		"enabled": True,
		"applied": True,
		"model": str(model_name),
		"top_n": int(top_n),
		"weight": w,
		"rescored": len(active),
	}
