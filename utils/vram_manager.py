"""VRAM Budget Manager.

Central coordinator for GPU memory across all local model types:
  - Ollama-served models  (LLM, embedder) — live state read from /api/ps
  - PyTorch models loaded in-process  (Marker, pix2tex, intent classifier)

Design
------
The manager is reactive, not predictive.  It does not try to schedule model
loads; it evicts on demand when a caller needs a known large amount of VRAM.

Two main entry points for callers:

  evict_all_for_marker(...)
      Call this synchronously before loading Marker (create_model_dict /
      load_all_models).  It:
        1. Unloads the local intent classifier (if resident on CUDA).
        2. Sends keep_alive=0 to every loaded Ollama model to force-evict.
        3. Sleeps briefly so Ollama releases VRAM before Marker loads.
      On high-VRAM machines (≥ _HIGH_VRAM_THRESHOLD_MB total) this is a
      no-op — there is room for everything simultaneously.

  after_llm_complete(...)
      Call this (or spawn a thread that calls it) after the LLM finishes.
      It reloads the intent classifier and pre-warms the embedder — but only
      when Marker is not currently resident (avoids kicking Marker out).

Tracking
--------
  on_torch_loaded(name, est_vram_mb)   — call after loading a PyTorch model
  on_torch_unloaded(name)              — call after freeing one

Status
------
  get_status(ollama_base_url)   — returns a dict suitable for /api/status
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

# Machines with this much total VRAM (MB) can keep all models resident at once.
_HIGH_VRAM_THRESHOLD_MB: int = 20_000

# nvidia-smi cache lifetime.  The call takes ~100 ms; cache it.
_GPU_CACHE_TTL_S: float = 30.0

# How long to sleep after sending Ollama eviction requests before returning.
# Gives Ollama time to actually release the VRAM.
_EVICT_WAIT_S: float = 2.5

# Estimated VRAM costs for locally-loaded PyTorch models (MB).
# These are used only for status display; actual eviction uses /api/ps truth.
MODEL_VRAM_ESTIMATES: Dict[str, int] = {
    "marker":             7_000,   # full Marker suite (OCR + layout + surya)
    "pix2tex":            1_200,   # LatexOCR
    "intent_classifier":  2_000,   # DeBERTa-v3-large zero-shot
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class GpuInfo:
    name: str
    total_mb: int
    free_mb: int


@dataclass
class OllamaModelInfo:
    name: str
    vram_mb: int    # VRAM portion only
    loaded_mb: int  # total (VRAM + CPU RAM)


@dataclass
class _TorchEntry:
    name: str
    est_vram_mb: int
    loaded_at: float = field(default_factory=time.time)


# ── VRAMManager ────────────────────────────────────────────────────────────────

class VRAMManager:
    """Thread-safe VRAM coordinator. Use get_manager() to get the singleton."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._torch_models: Dict[str, _TorchEntry] = {}
        self._gpu_cache: Optional[List[GpuInfo]] = None
        self._gpu_cache_ts: float = 0.0

    # ── GPU info ───────────────────────────────────────────────────────────────

    def get_gpu_info(self, force: bool = False) -> List[GpuInfo]:
        """Return local GPU list from nvidia-smi. Cached for _GPU_CACHE_TTL_S seconds."""
        now = time.monotonic()
        with self._lock:
            if (
                not force
                and self._gpu_cache is not None
                and (now - self._gpu_cache_ts) < _GPU_CACHE_TTL_S
            ):
                return list(self._gpu_cache)

        gpus = self._query_nvidia_smi()

        with self._lock:
            self._gpu_cache = gpus
            self._gpu_cache_ts = now

        return gpus

    @staticmethod
    def _query_nvidia_smi() -> List[GpuInfo]:
        try:
            _kwargs: dict = {}
            if sys.platform == "win32":
                _kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=10,
                **_kwargs,
            )
            if result.returncode != 0:
                return []
            gpus: List[GpuInfo] = []
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        gpus.append(
                            GpuInfo(
                                name=parts[0],
                                total_mb=int(parts[1]),
                                free_mb=int(parts[2]),
                            )
                        )
                    except ValueError:
                        pass
            return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return []

    def _total_vram_mb(self) -> int:
        return sum(g.total_mb for g in self.get_gpu_info())

    def _is_high_vram(self) -> bool:
        total = self._total_vram_mb()
        return total >= _HIGH_VRAM_THRESHOLD_MB

    # ── Ollama state ───────────────────────────────────────────────────────────

    @staticmethod
    def get_ollama_loaded(base_url: str) -> List[OllamaModelInfo]:
        """Query /api/ps to get currently-loaded Ollama models. Returns [] on any error."""
        try:
            req = urllib.request.Request(
                f"{base_url.rstrip('/')}/api/ps",
                headers={"Accept": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            return [
                OllamaModelInfo(
                    name=m.get("name") or "",
                    vram_mb=(m.get("size_vram") or m.get("size") or 0) // (1024 * 1024),
                    loaded_mb=(m.get("size") or 0) // (1024 * 1024),
                )
                for m in (data.get("models") or [])
            ]
        except Exception:
            return []

    @staticmethod
    def evict_ollama_model(model_name: str, base_url: str) -> bool:
        """
        Force-evict one model from Ollama by sending keep_alive=0.

        Tries /api/generate first (works for LLMs), then /api/embed
        (embedding models that don't respond to generate).

        Returns True if at least one request was accepted.  The model may
        still be unloading asynchronously.
        """
        base = base_url.rstrip("/")
        attempts = [
            (
                f"{base}/api/generate",
                json.dumps(
                    {"model": model_name, "prompt": "", "keep_alive": 0, "stream": False}
                ).encode(),
            ),
            (
                f"{base}/api/embed",
                json.dumps(
                    {"model": model_name, "input": [""], "keep_alive": 0}
                ).encode(),
            ),
        ]
        for url, payload in attempts:
            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    resp.read()
                logger.info(
                    "vram_manager: evicted %s from Ollama via %s",
                    model_name,
                    url.split("/")[-1],
                )
                return True
            except Exception:
                continue
        logger.warning("vram_manager: could not evict %s from Ollama", model_name)
        return False

    # ── PyTorch model tracking ─────────────────────────────────────────────────

    def on_torch_loaded(self, name: str, est_vram_mb: int = 0) -> None:
        """Register a PyTorch model as loaded in-process."""
        if est_vram_mb == 0:
            est_vram_mb = MODEL_VRAM_ESTIMATES.get(name, 0)
        with self._lock:
            self._torch_models[name] = _TorchEntry(name=name, est_vram_mb=est_vram_mb)
        logger.debug("vram_manager: torch model loaded: %s (~%d MB)", name, est_vram_mb)

    def on_torch_unloaded(self, name: str) -> None:
        """Deregister a PyTorch model."""
        with self._lock:
            self._torch_models.pop(name, None)
        logger.debug("vram_manager: torch model unloaded: %s", name)

    def _is_marker_active(self) -> bool:
        with self._lock:
            return "marker" in self._torch_models

    # ── Eviction for Marker ────────────────────────────────────────────────────

    def evict_all_for_marker(
        self,
        ollama_base_url: str = "http://localhost:11434",
    ) -> None:
        """
        Prepare VRAM for Marker loading. Call synchronously before
        create_model_dict() / load_all_models().

        On high-VRAM machines (≥ _HIGH_VRAM_THRESHOLD_MB) this is a no-op.

        Otherwise:
          1. Unloads the local intent classifier (releases CUDA memory).
          2. Force-evicts every currently-loaded Ollama model via keep_alive=0.
          3. Sleeps _EVICT_WAIT_S seconds for Ollama to actually release VRAM.
        """
        if self._is_high_vram():
            logger.info(
                "vram_manager: high-VRAM machine (%d MB total) — skipping eviction for Marker",
                self._total_vram_mb(),
            )
            return

        logger.info("vram_manager: evicting models to make room for Marker")

        # 1. Unload local intent classifier
        try:
            from retrieval.intent_classifier import unload as _unload_cls  # noqa: PLC0415
            _unload_cls()
        except Exception:
            pass

        # 2. Evict every Ollama model that is currently loaded
        loaded = self.get_ollama_loaded(ollama_base_url)
        if loaded:
            for m in loaded:
                self.evict_ollama_model(m.name, ollama_base_url)
        else:
            # /api/ps returned nothing (Ollama may be unreachable or empty).
            # Nothing to evict.
            logger.debug("vram_manager: Ollama reports no loaded models — nothing to evict")

        # 3. Brief pause for Ollama to release GPU memory
        if loaded:
            time.sleep(_EVICT_WAIT_S)
            logger.info("vram_manager: eviction complete")

    # ── After-LLM reload ───────────────────────────────────────────────────────

    def after_llm_complete(
        self,
        base_url: str = "http://localhost:11434",
        embed_model: str = "qwen3-embedding:latest",
    ) -> None:
        """
        Non-blocking. Spawns a background thread to reload the intent
        classifier and pre-warm the embedder after the LLM releases VRAM.

        Replaces _reload_after_llm() in llm/generation.py.

        Skips all reloads when Marker is currently loaded — reloading the
        classifier or the embedder at that point would compete for Marker's VRAM.
        """
        if self._is_marker_active():
            logger.debug(
                "vram_manager: after_llm_complete skipped — Marker is resident"
            )
            return

        def _worker() -> None:
            # Give Ollama a moment to finish evicting the LLM.
            time.sleep(2)

            # Skip all reloads if Marker has since started loading
            if self._is_marker_active():
                return

            # Reload intent classifier (warmup() forces immediate load;
            # without it the classifier reloads lazily on the next query).
            try:
                from retrieval.intent_classifier import warmup as _warmup  # noqa: PLC0415
                _warmup()
            except Exception:
                pass

            # Pre-warm embedder — keep_alive=300 keeps it hot for 5 min.
            try:
                payload = json.dumps(
                    {"model": embed_model, "input": ["warmup"], "keep_alive": 300}
                ).encode()
                req = urllib.request.Request(
                    f"{base_url.rstrip('/')}/api/embed",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp.read()
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True, name="vram-after-llm").start()

    # ── Status ─────────────────────────────────────────────────────────────────

    def get_status(
        self,
        ollama_base_url: str = "http://localhost:11434",
    ) -> Dict[str, Any]:
        """
        Return a combined GPU + model status dict suitable for /api/status.

        All queries are best-effort with short timeouts; partial data is always
        returned even when Ollama or nvidia-smi are unreachable.
        """
        gpus = self.get_gpu_info()
        ollama_models = self.get_ollama_loaded(ollama_base_url)

        with self._lock:
            torch_models = list(self._torch_models.values())

        total_vram_mb = sum(g.total_mb for g in gpus)
        total_free_mb = sum(g.free_mb for g in self.get_gpu_info(force=True))
        total_ollama_vram = sum(m.vram_mb for m in ollama_models)
        total_torch_vram = sum(m.est_vram_mb for m in torch_models)

        return {
            "gpus": [
                {"name": g.name, "total_mb": g.total_mb, "free_mb": g.free_mb}
                for g in gpus
            ],
            "total_vram_mb": total_vram_mb,
            "free_vram_mb": total_free_mb,
            "used_vram_mb": total_ollama_vram + total_torch_vram,
            "ollama_models": [
                {
                    "name": m.name,
                    "vram_mb": m.vram_mb,
                    "loaded_mb": m.loaded_mb,
                }
                for m in ollama_models
            ],
            "torch_models": [
                {"name": m.name, "est_vram_mb": m.est_vram_mb}
                for m in torch_models
            ],
        }


# ── Singleton ──────────────────────────────────────────────────────────────────

_manager: Optional[VRAMManager] = None
_manager_lock = threading.Lock()


def get_manager() -> VRAMManager:
    """Return the process-wide VRAMManager singleton."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = VRAMManager()
    return _manager
