"""GPU / inference-node detection and VRAM feasibility checker.

Two modes
---------
Status (default):
    python utils/gpu_detect.py
    python utils/gpu_detect.py --json      # machine-readable JSON

Discovery (measures actual loaded VRAM, writes configs/models.yaml):
    python utils/gpu_detect.py --discover

Discovery loads each configured Ollama model one at a time, reads size_vram
and size from /api/ps, saves the measured values to configs/models.yaml, then
unloads the model before moving to the next one.  This will briefly displace
any models currently loaded -- warn the user before running in production.

The feasibility report groups models by host and swap_group (models that are
never loaded simultaneously), computes peak VRAM per group, and compares
against detected GPU VRAM.  If a model's loaded_mb exceeds the GPU's total
VRAM, the overflow spills to CPU RAM -- functional but degraded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.request import urlopen, Request

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if getattr(__import__("sys"), "frozen", False):
    from utils.frozen import get_install_dir as _get_install_dir
    _PROJECT_ROOT = _get_install_dir()
_DEFAULT_RUNTIME  = _PROJECT_ROOT / "configs" / "runtime.yaml"
_DEFAULT_MODELS   = _PROJECT_ROOT / "configs" / "models.yaml"


def _load_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        print(f"[warn] Could not read {path}: {exc}", file=sys.stderr)
        return {}


def _save_yaml(path: Path, data: dict) -> None:
    import yaml  # type: ignore
    path.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def _runtime_urls(runtime: dict) -> Dict[str, str]:
    """Extract all unique Ollama + inference-service URLs from runtime.yaml."""
    urls: Dict[str, str] = {}
    urls["llm"]       = (runtime.get("llm") or {}).get("base_url", "http://localhost:11434")
    urls["embedding"] = (runtime.get("embedding") or {}).get("ollama_base_url", "http://localhost:11434")
    urls["hyde"]      = (runtime.get("hyde") or {}).get("base_url", "")
    urls["inference_service"] = (runtime.get("inference_service") or {}).get("url", "")
    return urls


# ---------------------------------------------------------------------------
# HTTP utilities
# ---------------------------------------------------------------------------

def _http_get(url: str, timeout: int = 6) -> Optional[dict]:
    try:
        req = Request(url, headers={"Accept": "application/json"})
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception:
        return None


def _http_post(url: str, payload: dict, timeout: int = 120) -> Optional[dict]:
    try:
        data = json.dumps(payload).encode()
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode()
            # Ollama generate streams NDJSON; take the last non-empty line
            lines = [l for l in raw.strip().splitlines() if l.strip()]
            return json.loads(lines[-1]) if lines else {}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# nvidia-smi (local only)
# ---------------------------------------------------------------------------

def _local_gpus() -> List[Dict]:
    """Return list of {name, vram_total_mb, vram_free_mb} from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    gpus.append({
                        "name": parts[0],
                        "vram_total_mb": int(parts[1]),
                        "vram_free_mb":  int(parts[2]),
                    })
                except ValueError:
                    pass
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


# ---------------------------------------------------------------------------
# Ollama probing
# ---------------------------------------------------------------------------

def _ollama_status(base_url: str) -> Dict:
    """Return {reachable, loaded_models, available_models} for an Ollama host."""
    base_url = base_url.rstrip("/")
    result = {"url": base_url, "reachable": False, "loaded_models": [], "available_models": []}

    tags = _http_get(f"{base_url}/api/tags")
    if tags is None:
        return result
    result["reachable"] = True

    for m in tags.get("models") or []:
        name = m.get("name") or ""
        size_mb = (m.get("size") or 0) // (1024 * 1024)
        result["available_models"].append({"name": name, "size_mb": size_mb})

    ps = _http_get(f"{base_url}/api/ps")
    if ps:
        for m in ps.get("models") or []:
            name  = m.get("name") or ""
            total = (m.get("size") or 0) // (1024 * 1024)
            vram  = (m.get("size_vram") or m.get("size") or 0) // (1024 * 1024)
            result["loaded_models"].append({
                "name": name, "loaded_mb": total, "vram_mb": vram,
                "cpu_overflow_mb": max(0, total - vram),
            })
    return result


# ---------------------------------------------------------------------------
# Ollama load / measure / unload  (used by --discover)
# ---------------------------------------------------------------------------

def _ollama_load(base_url: str, model: str, keep_alive: int = 300) -> bool:
    """Load a model into VRAM.

    Tries /api/generate first (generative models). If that fails, falls back to
    /api/embed (embedding models that don't support generate).
    """
    print(f"    Loading {model}...", end=" ", flush=True)
    resp = _http_post(f"{base_url}/api/generate", {
        "model": model,
        "prompt": "",
        "keep_alive": keep_alive,
        "stream": False,
        "options": {"num_predict": 1},
    }, timeout=180)
    if resp is None:
        # Embedding models don't respond to /api/generate -- try /api/embed
        resp = _http_post(f"{base_url}/api/embed", {
            "model": model,
            "input": "hello",
            "keep_alive": keep_alive,
        }, timeout=180)
    ok = resp is not None
    print("ok" if ok else "FAILED")
    return ok


def _ollama_measure(base_url: str, model: str) -> Tuple[int, int]:
    """
    Read /api/ps and return (loaded_mb, vram_mb) for the named model.
    loaded_mb = total memory (VRAM + CPU RAM).
    vram_mb   = VRAM-only portion.

    Waits for the reported size to stabilise (two consecutive reads agree)
    to avoid undercounting during mid-load layer placement.
    """
    model_stem = model.split(":")[0].lower()
    prev: Tuple[int, int] = (0, 0)

    for _ in range(20):         # up to 20s
        ps = _http_get(f"{base_url}/api/ps")
        if ps:
            for m in ps.get("models") or []:
                name = (m.get("name") or "").lower()
                if model_stem in name or name.startswith(model_stem):
                    total = (m.get("size") or 0) // (1024 * 1024)
                    vram  = (m.get("size_vram") or m.get("size") or 0) // (1024 * 1024)
                    current = (total, vram)
                    if current == prev and total > 0:
                        return total, vram   # stable -- two reads match
                    prev = current
        time.sleep(1)

    # Return whatever we last saw even if not fully stable
    return prev


def _ollama_unload(base_url: str, model: str) -> None:
    """Unload model from VRAM immediately (keep_alive=0)."""
    _http_post(f"{base_url}/api/generate", {
        "model": model,
        "prompt": "",
        "keep_alive": 0,
        "stream": False,
        "options": {"num_predict": 1},
    }, timeout=30)


# ---------------------------------------------------------------------------
# Inference service probing
# ---------------------------------------------------------------------------

def _inference_service_status(url: str) -> Dict:
    url = url.rstrip("/")
    result = {"url": url, "reachable": False}
    health = _http_get(f"{url}/health")
    if health is None:
        return result
    result["reachable"] = True
    result.update(health)
    return result


# ---------------------------------------------------------------------------
# Feasibility check
# ---------------------------------------------------------------------------

def _feasibility(models_cfg: dict, runtime: dict) -> List[Dict]:
    """
    Per-host VRAM feasibility report.

    Returns a list of dicts, one per distinct host:
        host_url, gpu_vram_mb, peak_needed_mb, overflow_mb, models, warnings
    """
    urls = _runtime_urls(runtime)
    local_gpus = _local_gpus()
    local_vram = local_gpus[0]["vram_total_mb"] if local_gpus else 0

    # Remote GPU VRAM from inference service /health (if it exposes it)
    remote_vram: Dict[str, int] = {}
    svc_url = urls.get("inference_service", "").rstrip("/")
    if svc_url:
        svc = _inference_service_status(svc_url)
        if svc.get("reachable") and "vram_total_mb" in svc:
            # Covers any model assigned to the inference service URL directly
            remote_vram[svc_url] = svc["vram_total_mb"]
            # The inference service host may also run Ollama on the same machine
            ollama_on_svc = urls.get("hyde", "").rstrip("/")
            if ollama_on_svc:
                remote_vram[ollama_on_svc] = svc["vram_total_mb"]

    all_models: dict = models_cfg.get("models") or {}

    # Group models by assigned_to host
    by_host: Dict[str, List[dict]] = {}
    for name, cfg in all_models.items():
        host = (cfg.get("assigned_to") or "").rstrip("/")
        if not host:
            continue
        by_host.setdefault(host, []).append({"name": name, **cfg})

    results = []
    for host, models in by_host.items():
        # Determine GPU VRAM for this host
        is_local = "localhost" in host or "127.0.0.1" in host
        gpu_vram = local_vram if is_local else remote_vram.get(host, 0)

        # Group by swap_group -- only the largest in a group loads at once
        swap_groups: Dict[str, List[dict]] = {}
        ungrouped: List[dict] = []
        for m in models:
            sg = m.get("swap_group")
            if sg:
                swap_groups.setdefault(sg, []).append(m)
            else:
                ungrouped.append(m)

        peak_needed = 0
        group_summaries = []

        for sg, members in swap_groups.items():
            biggest = max(members, key=lambda x: x.get("loaded_mb", 0))
            peak_mb = biggest.get("loaded_mb", 0)
            peak_needed += peak_mb
            group_summaries.append({
                "swap_group": sg,
                "peak_model": biggest["name"],
                "peak_mb": peak_mb,
                "members": [m["name"] for m in members],
            })

        for m in ungrouped:
            peak_needed += m.get("loaded_mb", 0)

        overflow_mb = max(0, peak_needed - gpu_vram) if gpu_vram > 0 else 0
        warnings = []
        for m in models:
            cpu_overflow = m.get("loaded_mb", 0) - m.get("vram_mb", m.get("loaded_mb", 0))
            if cpu_overflow > 0:
                warnings.append(
                    f"{m['name']}: {cpu_overflow:,} MB measured as CPU overflow "
                    f"(run --discover to update)"
                )
        if overflow_mb > 0:
            warnings.append(
                f"Peak load {peak_needed:,} MB exceeds GPU VRAM {gpu_vram:,} MB "
                f"-- {overflow_mb:,} MB will spill to CPU RAM"
            )

        results.append({
            "host": host,
            "gpu_vram_mb": gpu_vram,
            "peak_needed_mb": peak_needed,
            "overflow_mb": overflow_mb,
            "swap_groups": group_summaries,
            "ungrouped": [m["name"] for m in ungrouped],
            "models": models,
            "warnings": warnings,
            "ok": overflow_mb == 0 and not any("CPU overflow" in w for w in warnings),
        })

    return results


# ---------------------------------------------------------------------------
# Hardware-aware model assignment
# ---------------------------------------------------------------------------

# Ollama models at or below this VRAM footprint are preferred candidates for
# remote offload (they fit comfortably on a mid-range card alongside the
# HuggingFace models already resident there).
# Override via `gpu.small_model_vram_mb` in configs/runtime.yaml.
def _get_small_model_vram_mb() -> int:
    """Read gpu.small_model_vram_mb from runtime.yaml, falling back to 3000."""
    try:
        rt = _load_yaml(_DEFAULT_RUNTIME)
        return int((rt.get("gpu") or {}).get("small_model_vram_mb", 3000))
    except Exception:
        return 3000


_SMALL_MODEL_VRAM_MB = _get_small_model_vram_mb()


def _role_to_runtime_paths(role: str, model_type: str) -> List[List[str]]:
    """Map a model's role + type to the runtime.yaml nested key paths it controls."""
    if model_type == "huggingface":
        return [["inference_service", "url"]]
    role_l = role.lower()
    paths: List[List[str]] = []
    if "embedding" in role_l:
        paths.append(["embedding", "ollama_base_url"])
    if any(k in role_l for k in ("llm", "answer generation", "answer gen")):
        paths.append(["llm", "base_url"])
    if any(k in role_l for k in ("hyde", "search-rewrite")):
        paths.append(["hyde", "base_url"])
    return paths


def plan_assignment(
    local_gpus: List[Dict],
    ollama_nodes: List[Dict],
    inference_svc: Dict,
    models_cfg: dict,
    local_ollama_url: str = "http://localhost:11434",
    small_model_vram_mb: int = _SMALL_MODEL_VRAM_MB,
) -> Tuple[Dict[str, str], List[str]]:
    """Determine which Ollama / inference-service URL each model should run on.

    Strategy
    --------
    1. HuggingFace models  → remote inference service if reachable, else warn.
    2. Ollama models sorted smallest-first:
       a. If the model fits within a reachable remote Ollama node's remaining
          VRAM budget, assign it there (offloads small helpers such as HyDE).
       b. Otherwise assign to local Ollama.

    Returns
    -------
    assignment : {model_key: host_url}
    warnings   : list of human-readable warning strings
    """
    all_models: dict = models_cfg.get("models") or {}
    assignment: Dict[str, str] = {}
    warnings: List[str] = []

    svc_url = inference_svc.get("url", "").rstrip("/") if inference_svc.get("reachable") else ""
    svc_vram_total: int = inference_svc.get("vram_total_mb", 0) if inference_svc.get("reachable") else 0

    # Reachable remote Ollama nodes (anything that is not localhost)
    remote_ollama: List[Dict] = [
        n for n in ollama_nodes
        if n.get("reachable")
        and "localhost" not in n["url"]
        and "127.0.0.1" not in n["url"]
    ]

    def _host_ip(u: str) -> str:
        return u.replace("http://", "").replace("https://", "").split(":")[0]

    # Estimate VRAM budget per remote Ollama node.
    # If the inference service runs on the same machine (matching IP) use its
    # reported VRAM; otherwise fall back to a conservative 6 GB estimate.
    remote_vram_budget: Dict[str, int] = {}
    for node in remote_ollama:
        node_url = node["url"].rstrip("/")
        budget = 0
        if svc_vram_total and svc_url:
            if _host_ip(svc_url) == _host_ip(node_url):
                budget = svc_vram_total
        if budget == 0:
            budget = 6144  # conservative fallback for an unknown remote GPU
        remote_vram_budget[node_url] = budget

    # ── 1. HuggingFace models ───────────────────────────────────────────────
    for key, cfg in all_models.items():
        if cfg.get("type") != "huggingface":
            continue
        if svc_url:
            assignment[key] = svc_url
        else:
            warnings.append(
                f"{key}: type=huggingface but no inference service is reachable. "
                "Start inference_service/server.py on a remote node and set its URL "
                "in runtime.yaml, or these models will not be available."
            )

    # ── 2. Ollama models (smallest VRAM first → maximise remote fit) ────────
    ollama_items = sorted(
        [(k, v) for k, v in all_models.items() if v.get("type") == "ollama"],
        key=lambda kv: kv[1].get("loaded_mb", 0),
    )

    for key, cfg in ollama_items:
        loaded_mb = cfg.get("loaded_mb", 0)
        vram_mb   = cfg.get("vram_mb", loaded_mb)
        placed    = False

        if loaded_mb and loaded_mb <= small_model_vram_mb and remote_ollama:
            for node in remote_ollama:
                node_url = node["url"].rstrip("/")
                budget   = remote_vram_budget.get(node_url, 0)
                if budget >= vram_mb:
                    assignment[key] = node_url
                    remote_vram_budget[node_url] = budget - vram_mb
                    placed = True
                    break

        if not placed:
            assignment[key] = local_ollama_url
            if loaded_mb == 0:
                warnings.append(
                    f"{key}: loaded_mb not measured yet — assigned to local by default. "
                    "Run --discover first for accurate placement."
                )

    return assignment, warnings


def apply_assignment_to_runtime(
    assignment: Dict[str, str],
    models_cfg: dict,
    runtime: dict,
) -> dict:
    """Return a deep copy of *runtime* with URLs updated to match *assignment*.

    Only keys with a known role mapping are touched; everything else is
    preserved verbatim.
    """
    import copy
    runtime_out = copy.deepcopy(runtime)
    all_models: dict = models_cfg.get("models") or {}

    for model_key, host_url in assignment.items():
        cfg        = all_models.get(model_key, {})
        role       = cfg.get("role", "")
        model_type = cfg.get("type", "")
        for path in _role_to_runtime_paths(role, model_type):
            node = runtime_out
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = host_url

    return runtime_out


def _print_plan(
    assignment: Dict[str, str],
    models_cfg: dict,
    warnings: List[str],
) -> None:
    """Pretty-print the proposed model assignment."""
    all_models: dict = models_cfg.get("models") or {}
    print("\n=== Proposed Model Assignment ===\n")
    print(f"  {'Model':<38} {'Host':<35} {'VRAM':>8}  Role")
    print(f"  {'-'*38} {'-'*35} {'-'*8}  {'-'*28}")
    any_changed = False
    for key in sorted(assignment):
        host     = assignment[key]
        cfg      = all_models.get(key, {})
        vram_mb  = cfg.get("vram_mb", cfg.get("loaded_mb", 0))
        role     = cfg.get("role", "")
        current  = (cfg.get("assigned_to") or "").rstrip("/")
        changed  = current and current != host
        marker   = " *" if changed else "  "
        vram_str = f"{vram_mb:,} MB" if vram_mb else "?"
        if changed:
            any_changed = True
        print(f"  {key:<38} {host:<35} {vram_str:>8}{marker}  {role}")

    if any_changed:
        print("\n  (* = differs from current models.yaml assignment)")
    if warnings:
        print("\n  Warnings:")
        for w in warnings:
            print(f"    [!] {w}")


# ---------------------------------------------------------------------------
# Discovery mode
# ---------------------------------------------------------------------------

def discover(models_cfg: dict, runtime: dict, models_path: Path) -> None:
    """
    Load each configured Ollama model one at a time, measure VRAM via /api/ps,
    update models.yaml with real numbers, then unload.

    HuggingFace models (type: huggingface) are skipped -- their VRAM is read
    from the inference service /health endpoint instead.
    """
    print("\n[discover] This will load and unload each Ollama model to measure VRAM.")
    print("[discover] Running models will be displaced temporarily.")
    ans = input("[discover] Proceed? [y/N] ").strip().lower()
    if ans != "y":
        print("Aborted.")
        return

    urls = _runtime_urls(runtime)
    all_models: dict = models_cfg.setdefault("models", {})
    changed = False

    for model_key, cfg in all_models.items():
        if cfg.get("type") != "ollama":
            print(f"\n  {model_key}: skipping (type={cfg.get('type', '?')} -- not an Ollama model)")
            continue

        base_url = (cfg.get("assigned_to") or "").rstrip("/")
        if not base_url:
            print(f"\n  {model_key}: skipping (no assigned_to URL)")
            continue

        # Check host reachable
        probe = _http_get(f"{base_url}/api/tags")
        if probe is None:
            print(f"\n  {model_key}: skipping ({base_url} unreachable)")
            continue

        # The Ollama model name is the key itself or the first word before any description
        ollama_name = model_key   # e.g. "rag-llm", "qwen3-embedding:latest", "hyde-model:latest"
        print(f"\n  {model_key}  [{base_url}]")

        if not _ollama_load(base_url, ollama_name):
            print(f"    Could not load {ollama_name} -- skipping")
            continue

        loaded_mb, vram_mb = _ollama_measure(base_url, ollama_name)
        cpu_mb = max(0, loaded_mb - vram_mb)

        if loaded_mb == 0:
            print(f"    Could not read /api/ps for {ollama_name} -- skipping")
        else:
            print(f"    loaded_mb : {loaded_mb:,}  ({loaded_mb/1024:.2f} GB total)")
            print(f"    vram_mb   : {vram_mb:,}  ({vram_mb/1024:.2f} GB in VRAM)")
            if cpu_mb > 0:
                print(f"    cpu_overflow_mb: {cpu_mb:,}  *** OVERFLOWS TO CPU RAM ***")
            cfg["loaded_mb"] = loaded_mb
            cfg["vram_mb"]   = vram_mb
            changed = True

        print(f"    Unloading...", end=" ", flush=True)
        _ollama_unload(base_url, ollama_name)
        # Wait for model to fully disappear from /api/ps before loading the next
        for _ in range(15):
            ps = _http_get(f"{base_url}/api/ps")
            loaded_names = [(m.get("name") or "").lower() for m in (ps or {}).get("models", [])]
            if not any(ollama_name.split(":")[0].lower() in n for n in loaded_names):
                break
            time.sleep(1)
        print("done")

    # HuggingFace: read from inference service /health
    svc_url = (urls.get("inference_service") or "").rstrip("/")
    if svc_url:
        print(f"\n  [inference service]  {svc_url}")
        svc = _inference_service_status(svc_url)
        if svc.get("reachable") and "vram_allocated_mb" in svc:
            for model_key, cfg in all_models.items():
                if cfg.get("type") == "huggingface" and (cfg.get("assigned_to") or "").rstrip("/") == svc_url:
                    vram_alloc = svc["vram_allocated_mb"]
                    print(f"    {model_key}: vram_allocated_mb={vram_alloc}")
                    cfg["vram_mb"]   = vram_alloc
                    cfg["loaded_mb"] = vram_alloc
                    changed = True
        else:
            print("    unreachable or no VRAM info in /health")

    if changed:
        _save_yaml(models_path, models_cfg)
        print(f"\n[discover] Saved measurements to {models_path}")
    else:
        print("\n[discover] No measurements recorded.")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _bar(used: int, total: int, width: int = 20) -> str:
    if total == 0:
        return "?" * width
    filled = int(width * used / total)
    return "#" * filled + "." * (width - filled)


def _print_status(runtime: dict, models_path: Path) -> None:
    gpus = _local_gpus()
    urls = _runtime_urls(runtime)

    print("\n=== Local GPU ===")
    if gpus:
        for g in gpus:
            used = g["vram_total_mb"] - g["vram_free_mb"]
            pct  = round(100 * used / g["vram_total_mb"], 1) if g["vram_total_mb"] else 0
            print(f"  {g['name']}")
            print(f"  VRAM: {used:,} / {g['vram_total_mb']:,} MB  ({pct}%)  [{_bar(used, g['vram_total_mb'])}]")
    else:
        print("  nvidia-smi not found or no GPU detected")

    # Ollama nodes
    seen: set = set()
    nodes = []
    for key in ("llm", "embedding", "hyde"):
        url = (urls.get(key) or "").rstrip("/")
        if url and url not in seen:
            seen.add(url)
            nodes.append(_ollama_status(url))

    print(f"\n=== Ollama Nodes ({len(nodes)}) ===")
    for node in nodes:
        label = "OK" if node["reachable"] else "UNREACHABLE"
        print(f"\n  {node['url']}  [{label}]")
        if node["reachable"]:
            if node["loaded_models"]:
                print("  Currently loaded:")
                for m in node["loaded_models"]:
                    cpu = m["cpu_overflow_mb"]
                    flag = f"  *** {cpu:,} MB on CPU RAM" if cpu > 0 else ""
                    print(f"    {m['name']}  -  {m['vram_mb']:,} MB VRAM  /  {m['loaded_mb']:,} MB total{flag}")
            else:
                print("  Currently loaded: (none)")
            print(f"  Available: {len(node['available_models'])} model(s)")
            for m in sorted(node["available_models"], key=lambda x: x["name"]):
                print(f"    {m['name']}  ({m['size_mb']:,} MB on disk)")

    # Inference service
    svc_url = (urls.get("inference_service") or "").rstrip("/")
    if svc_url:
        svc = _inference_service_status(svc_url)
        label = "OK" if svc.get("reachable") else "UNREACHABLE"
        print(f"\n=== Inference Service ===")
        print(f"  {svc_url}  [{label}]")
        if svc.get("reachable"):
            for k, v in svc.items():
                if k not in ("url", "reachable"):
                    print(f"    {k}: {v}")

    # Feasibility
    if models_path.exists():
        models_cfg = _load_yaml(models_path)
        results = _feasibility(models_cfg, runtime)
        print(f"\n=== VRAM Feasibility ===")
        for r in results:
            host = r["host"]
            vram = r["gpu_vram_mb"]
            peak = r["peak_needed_mb"]
            vram_str = f"{vram:,} MB" if vram else "unknown"
            status = "OK" if r["ok"] else "WARNING"
            print(f"\n  {host}  [{status}]")
            print(f"  GPU VRAM    : {vram_str}")
            print(f"  Peak needed : {peak:,} MB  ({peak/1024:.1f} GB)")
            if r["swap_groups"]:
                print(f"  Swap groups (mutually exclusive):")
                for sg in r["swap_groups"]:
                    print(f"    [{sg['swap_group']}] peak={sg['peak_model']} ({sg['peak_mb']:,} MB)  "
                          f"members: {', '.join(sg['members'])}")
            if r["ungrouped"]:
                print(f"  Always resident: {', '.join(r['ungrouped'])}")
            for w in r["warnings"]:
                print(f"  [!] {w}")
    else:
        print(f"\n[info] No {models_path.name} found -- run --discover to measure VRAM costs.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="GPU / inference-node status and VRAM feasibility checker")
    parser.add_argument("--config",   default=str(_DEFAULT_RUNTIME),  help="Path to runtime.yaml")
    parser.add_argument("--models",   default=str(_DEFAULT_MODELS),   help="Path to models.yaml")
    parser.add_argument("--discover", action="store_true",
                        help="Load each Ollama model, measure VRAM, save to models.yaml")
    parser.add_argument("--plan",     action="store_true",
                        help="Detect available hardware and print the recommended model assignment")
    parser.add_argument("--apply",    action="store_true",
                        help="Like --plan but also write updated runtime.yaml and models.yaml")
    parser.add_argument("--json",     action="store_true",
                        help="Output machine-readable JSON (status mode only)")
    args = parser.parse_args()

    runtime_path = Path(args.config)
    models_path  = Path(args.models)

    if not runtime_path.exists():
        print(f"Config not found: {runtime_path}", file=sys.stderr)
        sys.exit(1)

    runtime = _load_yaml(runtime_path)

    if args.discover:
        if not models_path.exists():
            print(f"[warn] {models_path} not found -- create it first (see configs/models.yaml example)")
            sys.exit(1)
        models_cfg = _load_yaml(models_path)
        discover(models_cfg, runtime, models_path)
    elif args.plan or args.apply:
        if not models_path.exists():
            print(f"[error] {models_path} not found -- run --discover first to populate it.", file=sys.stderr)
            sys.exit(1)
        models_cfg = _load_yaml(models_path)
        urls       = _runtime_urls(runtime)

        # Probe all unique Ollama nodes referenced in runtime.yaml
        seen_urls: set = set()
        ollama_nodes: List[Dict] = []
        for key in ("llm", "embedding", "hyde"):
            url = (urls.get(key) or "").rstrip("/")
            if url and url not in seen_urls:
                seen_urls.add(url)
                print(f"  Probing Ollama node {url} ...", end=" ", flush=True)
                status = _ollama_status(url)
                ollama_nodes.append(status)
                print("ok" if status["reachable"] else "unreachable")

        svc_raw = (urls.get("inference_service") or "").rstrip("/")
        inference_svc: Dict = {"reachable": False}
        if svc_raw:
            print(f"  Probing inference service {svc_raw} ...", end=" ", flush=True)
            inference_svc = _inference_service_status(svc_raw)
            inference_svc["url"] = svc_raw
            print("ok" if inference_svc.get("reachable") else "unreachable")

        local_gpus = _local_gpus()
        local_url  = (urls.get("llm") or "http://localhost:11434").rstrip("/")
        # Use the localhost Ollama URL as the local anchor
        for node in ollama_nodes:
            if "localhost" in node["url"] or "127.0.0.1" in node["url"]:
                local_url = node["url"].rstrip("/")
                break

        assignment, warnings = plan_assignment(
            local_gpus, ollama_nodes, inference_svc, models_cfg,
            local_ollama_url=local_url,
        )
        _print_plan(assignment, models_cfg, warnings)

        if args.apply:
            # Update assigned_to in models.yaml
            all_models = models_cfg.setdefault("models", {})
            for key, host in assignment.items():
                if key in all_models:
                    all_models[key]["assigned_to"] = host
            _save_yaml(models_path, models_cfg)
            print(f"\n  [apply] Saved updated assignments to {models_path}")

            # Update runtime.yaml URLs
            updated_runtime = apply_assignment_to_runtime(assignment, models_cfg, runtime)
            _save_yaml(runtime_path, updated_runtime)
            print(f"  [apply] Saved updated URLs to {runtime_path}")
        else:
            print("\n  Run with --apply to write these changes to disk.")
    elif args.json:
        gpus  = _local_gpus()
        urls  = _runtime_urls(runtime)
        nodes = {}
        for key in ("llm", "embedding", "hyde"):
            url = (urls.get(key) or "").rstrip("/")
            if url and url not in nodes:
                nodes[url] = _ollama_status(url)
        svc = _inference_service_status(urls.get("inference_service", "")) if urls.get("inference_service") else {}
        models_cfg = _load_yaml(models_path) if models_path.exists() else {}
        feasibility = _feasibility(models_cfg, runtime) if models_cfg else []
        print(json.dumps({
            "gpus": gpus,
            "ollama_nodes": list(nodes.values()),
            "inference_service": svc,
            "feasibility": feasibility,
        }, indent=2))
    else:
        _print_status(runtime, models_path)


if __name__ == "__main__":
    main()

