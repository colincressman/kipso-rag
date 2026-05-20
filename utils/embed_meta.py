"""Persist and verify the embedding model's output dimension.

The meta file at ``data/index/embed_meta.json`` records which model produced
the current index and its vector dimension.  Whenever an embed call is made,
the detected dimension is compared with the stored value.  A mismatch means
the index was built with a different model and should be rebuilt.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_META_PATH = os.path.join("data", "index", "embed_meta.json")


def load(meta_path: str = _DEFAULT_META_PATH) -> dict:
    """Return the stored embed meta dict, or ``{}`` if file absent/corrupt."""
    try:
        return json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save(
    model: str,
    dimension: int,
    meta_path: str = _DEFAULT_META_PATH,
) -> None:
    """Persist model name and dimension to the meta file."""
    p = Path(meta_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"model": model, "dimension": dimension}, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "embed_meta: saved model=%s dimension=%d to %s", model, dimension, meta_path
    )


def detect_and_save(
    model: str,
    detected_dim: int,
    meta_path: str = _DEFAULT_META_PATH,
) -> None:
    """Compare *detected_dim* against stored meta; save on first call.

    Logs a warning if the stored dimension differs from the detected value
    (indicates the index was built with a different model).  Updates the
    stored model name silently when only the name changed but dimension is
    the same.
    """
    stored = load(meta_path)
    if not stored:
        save(model, detected_dim, meta_path)
        return

    stored_dim = int(stored.get("dimension") or 0)
    stored_model = str(stored.get("model") or "")

    if stored_dim and stored_dim != detected_dim:
        logger.warning(
            "embed_meta: DIMENSION MISMATCH — stored=%d detected=%d "
            "(stored_model=%r current_model=%r). "
            "Re-index with scripts/ingest_all_raw.py to fix.",
            stored_dim,
            detected_dim,
            stored_model,
            model,
        )
    elif stored_model != model:
        # Dimension is the same but model name changed — update silently.
        logger.warning(
            "embed_meta: MODEL CHANGED — stored=%r current=%r "
            "(dimension %d unchanged). Re-index if vectors are stale.",
            stored_model,
            model,
            detected_dim,
        )
        save(model, detected_dim, meta_path)
