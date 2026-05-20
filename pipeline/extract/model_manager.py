"""
Mutual-exclusion model manager for pipeline2.

Marker and pix2tex are NEVER loaded into memory at the same time.
Acquiring one model automatically unloads the other and frees GPU cache.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Module-level state — use a lock for thread safety
_lock = threading.Lock()
_active_model: Optional[str] = None   # "marker" | "pix2tex" | None
_marker_models: Optional[Any] = None
_pix2tex_model: Optional[Any] = None


def _free_cuda_cache() -> None:
    """Release CUDA memory if PyTorch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def get_marker() -> Any:
    """
    Return the loaded Marker models, loading them if necessary.

    If pix2tex is currently loaded, it is unloaded first.
    Before loading, all Ollama models and the local intent classifier are
    evicted from VRAM so Marker has maximum headroom.
    """
    global _active_model, _marker_models, _pix2tex_model
    with _lock:
        if _active_model == "marker":
            return _marker_models

        # Unload pix2tex before loading Marker
        if _active_model == "pix2tex":
            logger.info("model_manager: unloading pix2tex to make room for Marker")
            _pix2tex_model = None
            gc.collect()
            _free_cuda_cache()
            _active_model = None
            try:
                from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
                _get_vram().on_torch_unloaded("pix2tex")
            except Exception:
                pass

        # Evict Ollama models + intent classifier before loading Marker
        try:
            from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
            from utils.runtime_defaults import DEFAULT_OLLAMA_BASE_URL  # noqa: PLC0415
            _get_vram().evict_all_for_marker(ollama_base_url=DEFAULT_OLLAMA_BASE_URL)
        except Exception as _vram_exc:
            logger.warning("model_manager: VRAM eviction failed (non-fatal): %s", _vram_exc)

        logger.info("model_manager: loading Marker models")
        try:
            from marker.models import create_model_dict  # marker >= 1.0
            _marker_models = create_model_dict()
        except ImportError:
            from marker.models import load_all_models  # marker < 1.0 fallback
            _marker_models = load_all_models()

        _active_model = "marker"

        # Register with VRAM manager for status tracking
        try:
            from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
            _get_vram().on_torch_loaded("marker")
        except Exception:
            pass

        return _marker_models


def get_pix2tex() -> Any:
    """
    Return the loaded pix2tex LatexOCR model, loading it if necessary.

    If Marker is currently loaded, it is unloaded first.

    Raises
    ------
    ImportError
        If pix2tex is not installed.
    """
    global _active_model, _marker_models, _pix2tex_model
    with _lock:
        if _active_model == "pix2tex":
            return _pix2tex_model

        # Unload Marker before loading pix2tex
        if _active_model == "marker":
            logger.info("model_manager: unloading Marker to make room for pix2tex")
            _marker_models = None
            gc.collect()
            _free_cuda_cache()
            _active_model = None
            try:
                from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
                _get_vram().on_torch_unloaded("marker")
            except Exception:
                pass

        logger.info("model_manager: loading pix2tex LatexOCR")
        try:
            from pix2tex.cli import LatexOCR  # type: ignore[import]
        except ImportError:
            raise ImportError(
                "pix2tex is not installed. Install it with:\n"
                "    pip install pix2tex[gui]"
            ) from None

        _pix2tex_model = LatexOCR()
        _active_model = "pix2tex"

        # Register with VRAM manager for status tracking
        try:
            from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
            _get_vram().on_torch_loaded("pix2tex")
        except Exception:
            pass

        return _pix2tex_model


def unload_all() -> None:
    """Unload whichever model is currently resident and free memory."""
    global _active_model, _marker_models, _pix2tex_model
    with _lock:
        if _active_model is None:
            return
        logger.info("model_manager: unloading %s", _active_model)
        _unloaded_name = _active_model
        _marker_models = None
        _pix2tex_model = None
        _active_model = None
        gc.collect()
        _free_cuda_cache()

    # Deregister from VRAM manager (outside lock to avoid deadlock)
    try:
        from utils.vram_manager import get_manager as _get_vram  # noqa: PLC0415
        _get_vram().on_torch_unloaded(_unloaded_name)
    except Exception:
        pass
