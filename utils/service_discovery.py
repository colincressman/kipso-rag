"""utils/service_discovery.py
----------------------------
Discovers and caches the remote HuggingFace inference service URL.

Priority order:
  1. In-process override (set via configure(), resets on restart)
  2. data/inference_override.json  (persisted from UI)
  3. runtime.yaml  inference_service.url
       - if empty or "auto"  →  scan local /24 subnet on port 8100
  4. ""  (no remote service — callers fall back to local models)

Cache: successful resolution is cached for CACHE_TTL_SECONDS (300 s).
Re-scan happens automatically when the cache expires or reset_cache() is called.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS   = 300    # 5 minutes
DEFAULT_PORT        = 8100
SCAN_TIMEOUT        = 0.5    # per-host probe during subnet scan
PROBE_TIMEOUT       = 2.0    # targeted probe
OVERRIDE_FILE       = Path("data/inference_override.json")

_lock = threading.Lock()
_cache: Dict[str, Any] = {
    "url":          None,   # None = not yet resolved; "" = resolved but nothing found
    "capabilities": {},
    "expires":      0.0,
}
_in_process_override: Optional[str] = None   # "" = explicitly disabled


# ── Persistence helpers ────────────────────────────────────────────────────────

def _read_override_file() -> Optional[str]:
    """Return the persisted URL override, or None if the file is absent/invalid."""
    try:
        if OVERRIDE_FILE.exists():
            data = json.loads(OVERRIDE_FILE.read_text(encoding="utf-8"))
            val = str(data.get("url", "")).strip()
            return val  # may be "" (disabled) or a real URL
    except Exception:
        pass
    return None


def _write_override_file(url: str) -> None:
    OVERRIDE_FILE.parent.mkdir(parents=True, exist_ok=True)
    OVERRIDE_FILE.write_text(json.dumps({"url": url}, indent=2), encoding="utf-8")


# ── Network helpers ────────────────────────────────────────────────────────────

def _local_subnet_prefix() -> Optional[str]:
    """Return the /24 prefix of the primary LAN IP, e.g. '192.168.1.' """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(1.0)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        parts = ip.split(".")
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.{parts[2]}."
    except Exception:
        pass
    return None


def _probe_url(url: str, timeout: float = PROBE_TIMEOUT) -> Optional[Dict]:
    """
    Attempt to GET /capabilities from url.
    Falls back to /health for older service versions.
    Returns the response dict on success, None on failure.
    """
    base = url.rstrip("/")
    for path in ("/capabilities", "/health"):
        try:
            req = urllib.request.Request(
                f"{base}{path}",
                headers={"User-Agent": "rag-discovery/1.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
                # /health must return {"status": "ok"} to be considered ours
                if path == "/health" and data.get("status") != "ok":
                    continue
                return data
        except Exception:
            continue
    return None


def _scan_subnet(port: int = DEFAULT_PORT) -> Optional[str]:
    """
    Parallel-probe every host on the local /24 for our inference service.
    Returns the first URL that responds with a valid service signature.
    """
    prefix = _local_subnet_prefix()
    if not prefix:
        logger.warning("service_discovery: could not determine local subnet for scan")
        return None

    logger.info("service_discovery: scanning %s0/24 port %d …", prefix, port)
    candidates = [f"http://{prefix}{i}:{port}" for i in range(1, 255)]
    found_event = threading.Event()
    result: list[str] = []

    def _try(url: str) -> Optional[str]:
        if found_event.is_set():
            return None
        caps = _probe_url(url, timeout=SCAN_TIMEOUT)
        if caps is not None and (
            caps.get("service") == "rag-inference" or caps.get("status") == "ok"
        ):
            return url
        return None

    with ThreadPoolExecutor(max_workers=64) as pool:
        futs = {pool.submit(_try, u): u for u in candidates}
        for fut in as_completed(futs):
            r = fut.result()
            if r and not found_event.is_set():
                found_event.set()
                result.append(r)
                # Cancel remaining (best-effort)
                for f in futs:
                    f.cancel()
                break

    if result:
        logger.info("service_discovery: found inference service at %s", result[0])
        return result[0]

    logger.info("service_discovery: no inference service found on subnet")
    return None


# ── Core resolution logic ──────────────────────────────────────────────────────

def _resolve() -> Tuple[str, Dict]:
    """
    Walk the priority chain and return (url, capabilities).
    url may be "" if nothing is available.
    """
    global _in_process_override

    # 1. In-process override (set via configure() this session)
    if _in_process_override is not None:
        url = _in_process_override
        if not url:
            return ("", {})
        caps = _probe_url(url) or {}
        return (url, caps)

    # 2. Persisted override file
    override = _read_override_file()
    if override is not None:
        url = override
        if not url:
            return ("", {})
        caps = _probe_url(url) or {}
        return (url, caps)

    # 3. runtime.yaml
    configured_url = ""
    try:
        from utils.config import load_yaml_config  # noqa: PLC0415
        rt = load_yaml_config("configs/runtime.yaml") or {}
        configured_url = str((rt.get("inference_service") or {}).get("url", "")).strip()
    except Exception:
        pass

    if configured_url and configured_url.lower() != "auto":
        caps = _probe_url(configured_url) or {}
        return (configured_url, caps)

    # 4. Auto-scan
    found = _scan_subnet()
    if found:
        caps = _probe_url(found) or {}
        return (found, caps)

    return ("", {})


# ── Public API ─────────────────────────────────────────────────────────────────

def get_inference_url() -> str:
    """Return the active inference service URL, or '' if none is available."""
    with _lock:
        now = time.monotonic()
        if _cache["url"] is not None and now < _cache["expires"]:
            return _cache["url"]

    # Resolve outside the lock to avoid holding it during network I/O
    url, caps = _resolve()

    with _lock:
        _cache["url"]          = url
        _cache["capabilities"] = caps
        _cache["expires"]      = time.monotonic() + CACHE_TTL_SECONDS

    return url


def get_cached_status() -> Dict[str, Any]:
    """Return the last cached discovery result without triggering a network call."""
    with _lock:
        return {
            "url":              _cache["url"] or "",
            "capabilities":    _cache["capabilities"],
            "cache_expires_in": max(0.0, _cache["expires"] - time.monotonic()),
            "connected":       bool(_cache["url"]),
        }


def probe_now(url: str) -> Dict[str, Any]:
    """Probe a specific URL and return its capabilities (or {} on failure)."""
    return _probe_url(url, timeout=PROBE_TIMEOUT) or {}


def scan_now() -> Optional[str]:
    """
    Trigger a fresh subnet scan regardless of cache state.
    Updates the cache on success.  Returns the found URL or None.
    """
    found = _scan_subnet()
    caps  = _probe_url(found) or {} if found else {}
    with _lock:
        _cache["url"]          = found or ""
        _cache["capabilities"] = caps
        _cache["expires"]      = time.monotonic() + CACHE_TTL_SECONDS
    return found


def configure(url: str) -> Dict[str, Any]:
    """
    Manually set the inference service URL and persist to disk.
    Pass "" to disable remote inference entirely.
    Returns the probed capabilities dict (may be {} if unreachable or disabled).
    """
    global _in_process_override
    url  = url.strip().rstrip("/")
    _write_override_file(url)
    caps = _probe_url(url) if url else {}

    with _lock:
        _in_process_override   = url
        _cache["url"]          = url
        _cache["capabilities"] = caps or {}
        _cache["expires"]      = time.monotonic() + CACHE_TTL_SECONDS

    return caps or {}


def reset_cache() -> None:
    """Clear the cache so the next call to get_inference_url() re-discovers."""
    with _lock:
        _cache["url"]          = None
        _cache["capabilities"] = {}
        _cache["expires"]      = 0.0


def get_remote_ollama_url() -> str:
    """
    Return the Ollama base URL on the remote inference node, or '' if no
    remote service is available.

    Derives the host from the discovered inference service URL, then uses
    the Ollama port advertised in /capabilities (default 11434).
    e.g. inference at http://10.0.0.5:8100  →  http://10.0.0.5:11434
    """
    url = get_inference_url()
    if not url:
        return ""
    with _lock:
        caps = _cache.get("capabilities") or {}
    ollama_info = caps.get("ollama") or {}
    # The ollama.url in capabilities is localhost-relative (from the remote
    # box's own perspective).  Extract just the port from it and combine
    # with the remote host we already know.
    try:
        from urllib.parse import urlparse  # noqa: PLC0415
        parsed_inf = urlparse(url)
        remote_host = parsed_inf.hostname or ""
        if not remote_host:
            return ""
        ollama_raw = str(ollama_info.get("url", "")).strip()
        if ollama_raw:
            parsed_ol = urlparse(ollama_raw)
            port = parsed_ol.port or 11434
        else:
            port = 11434
        scheme = parsed_inf.scheme or "http"
        return f"{scheme}://{remote_host}:{port}"
    except Exception:
        return ""
