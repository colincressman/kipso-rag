"""Grid-search retrieval parameter evaluation.

Sweeps combinations of (alpha_vector, alpha_lexical, cross_encoder_weight,
diversity_penalty, top_k) over grid_search_eval.json and records pass/fail
per question per config. Useful for overnight calibration runs.

Usage:
    python scripts/eval/run_grid_search.py
    python scripts/eval/run_grid_search.py --output data/diagnostics/grid_run1.csv
    python scripts/eval/run_grid_search.py --quick             # 16-combo mini-grid
    python scripts/eval/run_grid_search.py --ids gs_c01 gs_c11 # subset of questions
    python scripts/eval/run_grid_search.py --no-llm            # retrieval-only (fast)
    python scripts/eval/run_grid_search.py --resume data/diagnostics/grid_run1.json
"""
from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Windows console Unicode fix
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters, retrieve_as_dict
from retrieval.context_pack import build_context_pack
from retrieval.router import route_query
from llm.answer import answer_query_with_retrieval
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
)

GRID_EVAL_PATH = PROJECT_ROOT / "data" / "qa" / "grid_search_eval.json"

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grids
# ─────────────────────────────────────────────────────────────────────────────

# Full overnight grid: ~72 configs × 20 questions = ~1440 LLM calls
DEFAULT_GRID: Dict[str, List[Any]] = {
    "top_k":                [8, 10, 12],
    "alpha_vector":         [0.60, 0.68, 0.76],
    "alpha_lexical":        [0.24, 0.36],
    "cross_encoder_weight": [0.25, 0.40],
    "diversity_penalty":    [0.08, 0.18],
}

# Quick grid: 16 configs (single axis sweep from baseline)
QUICK_GRID: Dict[str, List[Any]] = {
    "top_k":                [8, 12],
    "alpha_vector":         [0.62, 0.72],
    "alpha_lexical":        [0.28, 0.38],
    "cross_encoder_weight": [0.25, 0.40],
    "diversity_penalty":    [0.08, 0.20],
}

# Tight grid: 32 configs zoomed in on winner (k=12, av=0.68, dp=0.08).
# al=0.24 is the overnight winner — included as the control anchor.
# cew=0.25–0.35 resolves the gs_c02 regression question.
TIGHT_GRID: Dict[str, List[Any]] = {
    "top_k":                [12, 14],
    "alpha_vector":         [0.68],
    "alpha_lexical":        [0.18, 0.22, 0.24, 0.28],
    "cross_encoder_weight": [0.20, 0.25, 0.30, 0.35],
    "diversity_penalty":    [0.08],
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with run_golden_eval
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_for_fact_check(text: str) -> str:
    import re
    text = re.sub(r'(\d)\s*%', r'\1 percent', text)
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    return text.lower()


def _check_facts(answer_text: str, expected_facts: List[str]) -> Dict[str, bool]:
    normalized = _normalize_for_fact_check(answer_text)
    results: Dict[str, bool] = {}
    for fact in expected_facts:
        alternatives = [_normalize_for_fact_check(a) for a in fact.split("|")]
        results[fact] = any(alt in normalized for alt in alternatives)
    return results


def _color(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

def _ok(t: str) -> str:    return _color(t, "32")
def _fail(t: str) -> str:  return _color(t, "31")
def _dim(t: str) -> str:   return _color(t, "2")

# ─────────────────────────────────────────────────────────────────────────────
# Config / grid helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_configs(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Cartesian product of all grid axis values."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    return [dict(zip(keys, c)) for c in combos]


def _config_key(cfg: Dict[str, Any]) -> str:
    """Compact string key for a config, used as CSV/JSON identifier."""
    parts = []
    for k in sorted(cfg):
        abbr = {
            "top_k": "k",
            "alpha_vector": "av",
            "alpha_lexical": "al",
            "cross_encoder_weight": "cew",
            "diversity_penalty": "dp",
        }.get(k, k)
        parts.append(f"{abbr}={cfg[k]}")
    return "|".join(parts)

# ─────────────────────────────────────────────────────────────────────────────
# Per-question runner with explicit parameter overrides
# ─────────────────────────────────────────────────────────────────────────────

def run_question_with_config(
    q: Dict[str, Any],
    cfg: Dict[str, Any],
    *,
    db_dsn: str = str(DEFAULT_DB_DSN),
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_model: str = DEFAULT_EMBED_MODEL_NAME,
    llm_enabled: bool = True,
) -> Dict[str, Any]:
    """Run one question with explicit retrieval parameter overrides.

    Mirrors the rag_retrieve() + llm_answer() pipeline from api.py, but
    replaces strategy-provided alpha/k values with the supplied ``cfg`` values
    so we can sweep them independently.
    """
    question = q["question"]
    expected_facts = q.get("expected_facts", [])
    must_not_contain = q.get("must_not_contain", [])

    top_k    = int(cfg.get("top_k", 8))
    av       = float(cfg.get("alpha_vector", 0.68))
    al       = float(cfg.get("alpha_lexical", 0.32))
    cew      = float(cfg.get("cross_encoder_weight", 0.35))
    dp       = float(cfg.get("diversity_penalty", 0.12))

    t0 = time.perf_counter()

    # ── Routing ───────────────────────────────────────────────────────────────
    routed = route_query(question, db_dsn=db_dsn)
    effective_query = routed.effective_query

    effective_source_type = routed.source_type_filter
    effective_collection  = routed.collection_id
    effective_doc_ids     = routed.doc_ids or None

    filters = RetrievalFilters(
        doc_ids=effective_doc_ids,
        source_type=effective_source_type,
        collection_id=effective_collection,
    )

    # ── Retrieval with overridden parameters ──────────────────────────────────
    # We bypass api.rag_retrieve() here so we can inject our own alpha/k values
    # instead of taking them from the router's strategy object.
    candidate_k = max(top_k * 4, 32)
    retrieve_kwargs: Dict[str, Any] = {
        "db_dsn":                db_dsn,
        "filters":               filters,
        "embed_backend":         embed_backend,
        "embed_model_name":      embed_model,
        "top_k":                 top_k,
        "rerank_candidate_k":    candidate_k,
        "rerank_alpha_vector":   av,
        "rerank_alpha_lexical":  al,
        "cross_encoder_weight":  cew,
        "rerank_diversity_penalty": dp,
        "intent":                routed.intent,
        "needs_web":             False,  # grid eval is corpus-only
    }

    result = retrieve_as_dict(effective_query, **retrieve_kwargs)

    # ── Context pack ──────────────────────────────────────────────────────────
    context_pack = build_context_pack(result, routed, max_chunks=top_k)
    result["context_pack"] = context_pack

    retrieval_elapsed = time.perf_counter() - t0

    # ── LLM answer ────────────────────────────────────────────────────────────
    answer_text = ""
    answer_mode = "skipped"
    llm_elapsed = 0.0

    if llm_enabled:
        t1 = time.perf_counter()
        answer = answer_query_with_retrieval(question, result)
        llm_elapsed = time.perf_counter() - t1
        answer_text = str(answer.get("answer") or answer.get("refusal") or "")
        answer_mode = str(answer.get("mode") or "")

    total_elapsed = time.perf_counter() - t0

    # ── Fact checking ─────────────────────────────────────────────────────────
    fact_results = _check_facts(answer_text, expected_facts)
    must_not_results = {p: p.lower() in answer_text.lower() for p in must_not_contain}

    facts_passed = sum(1 for v in fact_results.values() if v)
    facts_total  = len(fact_results)
    must_not_violations = sum(1 for v in must_not_results.values() if v)
    passed = facts_total > 0 and facts_passed == facts_total and must_not_violations == 0

    # Collect retrieved chunk sources for diagnosing why a config succeeded/failed
    hits = result.get("hits") or []
    top_chunk_ids   = [h.get("chunk_id", "") for h in hits[:5]]
    top_chunk_texts = [str(h.get("text", ""))[:120] for h in hits[:3]]

    return {
        "qid":                q["id"],
        "question":           question[:80],
        "category":           q.get("category", ""),
        "stress_axis":        q.get("stress_axis", ""),
        "config_key":         _config_key(cfg),
        "top_k":              top_k,
        "alpha_vector":       av,
        "alpha_lexical":      al,
        "cross_encoder_weight": cew,
        "diversity_penalty":  dp,
        "passed":             passed,
        "facts_found":        facts_passed,
        "facts_total":        facts_total,
        "must_not_violations": must_not_violations,
        "answer_mode":        answer_mode,
        "num_hits":           len(hits),
        "top_chunk_ids":      top_chunk_ids,
        "top_chunk_texts":    top_chunk_texts,
        "answer_text":        answer_text[:600],
        "fact_results":       fact_results,
        "retrieval_elapsed":  round(retrieval_elapsed, 2),
        "llm_elapsed":        round(llm_elapsed, 2),
        "total_elapsed":      round(total_elapsed, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Summary / reporting
# ─────────────────────────────────────────────────────────────────────────────

def _summarise(all_results: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Aggregate per-question results into per-config summary rows."""
    from collections import defaultdict

    # stress_axis → list of qids
    axis_qids: Dict[str, List[str]] = defaultdict(list)
    for q in questions:
        axis_qids[q.get("stress_axis", "other")].append(q["id"])

    # group by config_key
    by_cfg: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_results:
        by_cfg[r["config_key"]].append(r)

    summaries = []
    for cfg_key, rows in by_cfg.items():
        total   = len(rows)
        n_pass  = sum(1 for r in rows if r["passed"])
        first   = rows[0]
        row: Dict[str, Any] = {
            "config_key":         cfg_key,
            "top_k":              first["top_k"],
            "alpha_vector":       first["alpha_vector"],
            "alpha_lexical":      first["alpha_lexical"],
            "cross_encoder_weight": first["cross_encoder_weight"],
            "diversity_penalty":  first["diversity_penalty"],
            "pass_rate":          round(n_pass / total, 3) if total else 0,
            "n_pass":             n_pass,
            "n_total":            total,
        }
        # per-axis pass rates
        for axis, qids in axis_qids.items():
            axis_rows = [r for r in rows if r["qid"] in qids]
            n_axis = len(axis_rows)
            n_axis_pass = sum(1 for r in axis_rows if r["passed"])
            safe_axis = axis.replace(" ", "_")
            row[f"axis_{safe_axis}_pass"] = n_axis_pass
            row[f"axis_{safe_axis}_total"] = n_axis
            row[f"axis_{safe_axis}_rate"] = round(n_axis_pass / n_axis, 3) if n_axis else 0
        summaries.append(row)

    summaries.sort(key=lambda r: (-r["pass_rate"], -r["top_k"]))
    return summaries


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  → Written: {path}")


def _print_top_configs(summaries: List[Dict[str, Any]], n: int = 10) -> None:
    print(f"\n{'─'*74}")
    print(f"  TOP {min(n, len(summaries))} CONFIGS BY PASS RATE")
    print(f"{'─'*74}")
    print(f"  {'Rank':<5} {'Pass':>6} {'k':>4} {'av':>5} {'al':>5} {'cew':>5} {'dp':>5}  config")
    print(f"  {'─'*5} {'─'*6} {'─'*4} {'─'*5} {'─'*5} {'─'*5} {'─'*5}  {'─'*30}")
    for rank, s in enumerate(summaries[:n], 1):
        pct = f"{s['pass_rate']*100:.0f}% ({s['n_pass']}/{s['n_total']})"
        print(f"  {rank:<5} {pct:>10}  {s['top_k']:>3}  {s['alpha_vector']:>5}  "
              f"{s['alpha_lexical']:>5}  {s['cross_encoder_weight']:>5}  {s['diversity_penalty']:>5}  "
              f"{s['config_key'][:45]}")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search retrieval parameter evaluation.")
    parser.add_argument("--ids",      nargs="*",  help="Run only specific question IDs.")
    parser.add_argument("--output",   type=str,   default=None, help="Base path for CSV outputs (auto-named if omitted).")
    parser.add_argument("--quick",    action="store_true",      help="Use the small 16-combo quick grid.")
    parser.add_argument("--tight",    action="store_true",      help="Use the 24-combo tight grid around the overnight winner (k=12, av=0.68, dp=0.08).")
    parser.add_argument("--no-llm",   action="store_true",      help="Skip LLM answer; only measure retrieval hit patterns.")
    parser.add_argument("--resume",   type=str,   default=None, help="Resume from partial JSON results file.")
    parser.add_argument("--dataset",  type=str,   default=None, help="Path to eval JSON (default: data/qa/grid_search_eval.json).")
    parser.add_argument("--db-dsn",   type=str,   default=str(DEFAULT_DB_DSN))
    parser.add_argument("--embed-backend", type=str, default=DEFAULT_EMBED_BACKEND)
    parser.add_argument("--embed-model",   type=str, default=DEFAULT_EMBED_MODEL_NAME)
    # Parameter overrides (replace grid axis with a single value)
    parser.add_argument("--top-k",              type=int,   default=None, help="Pin top_k to a single value (removes it from the sweep).")
    parser.add_argument("--alpha-vector",        type=float, default=None)
    parser.add_argument("--alpha-lexical",       type=float, default=None)
    parser.add_argument("--cross-encoder-weight",type=float, default=None)
    parser.add_argument("--diversity-penalty",   type=float, default=None)
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else GRID_EVAL_PATH
    data = json.loads(dataset_path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if not questions:
        print("No questions matched — check --ids filter.")
        sys.exit(1)

    # Build grid (apply single-value overrides to collapse axes)
    grid = (TIGHT_GRID if args.tight else QUICK_GRID if args.quick else DEFAULT_GRID).copy()
    for param, arg_val in [
        ("top_k",                args.top_k),
        ("alpha_vector",          args.alpha_vector),
        ("alpha_lexical",         args.alpha_lexical),
        ("cross_encoder_weight",  args.cross_encoder_weight),
        ("diversity_penalty",     args.diversity_penalty),
    ]:
        if arg_val is not None:
            grid[param] = [arg_val]

    configs = _build_configs(grid)
    n_configs   = len(configs)
    n_questions = len(questions)
    n_total     = n_configs * n_questions
    llm_tag = "retrieval-only" if args.no_llm else "with LLM"
    print(f"\n{'═'*74}")
    print(f"  GRID SEARCH EVAL — {dataset_path.name}")
    print(f"  {n_configs} configs × {n_questions} questions = {n_total} calls  [{llm_tag}]")
    print(f"  Grid axes: {', '.join(f'{k}={v}' for k,v in grid.items())}")
    print(f"{'═'*74}\n")

    # ── Resume: load already-completed (config_key, qid) pairs ───────────────
    completed_keys: Set[tuple] = set()
    all_results: List[Dict[str, Any]] = []
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            prev = json.loads(resume_path.read_text(encoding="utf-8"))
            all_results = prev.get("results", [])
            for r in all_results:
                completed_keys.add((r["config_key"], r["qid"]))
            print(f"  Resuming — loaded {len(all_results)} previous results\n")

    # ── Output paths ─────────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    diag_dir = PROJECT_ROOT / "data" / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    if args.output:
        base = Path(args.output).with_suffix("")  # strip extension if given
    else:
        base = diag_dir / f"grid_search_{ts}"

    detail_csv  = base.with_suffix(".detail.csv")
    summary_csv = base.with_suffix(".summary.csv")
    raw_json    = base.with_suffix(".json")

    # ── Main sweep ────────────────────────────────────────────────────────────
    run_start = time.perf_counter()
    n_done = 0

    for cfg_idx, cfg in enumerate(configs, 1):
        cfg_key = _config_key(cfg)
        cfg_results: List[Dict[str, Any]] = []
        cfg_pass = 0

        for q in questions:
            if (cfg_key, q["id"]) in completed_keys:
                # Reuse previous result
                prev_r = next(
                    r for r in all_results
                    if r["config_key"] == cfg_key and r["qid"] == q["id"]
                )
                cfg_results.append(prev_r)
                if prev_r["passed"]:
                    cfg_pass += 1
                n_done += 1
                continue

            try:
                r = run_question_with_config(
                    q, cfg,
                    db_dsn=args.db_dsn,
                    embed_backend=args.embed_backend,
                    embed_model=args.embed_model,
                    llm_enabled=not args.no_llm,
                )
            except Exception as exc:
                print(f"    ERROR {q['id']}: {exc}", file=sys.stderr)
                r = {
                    "qid": q["id"], "question": q["question"][:80],
                    "category": q.get("category",""), "stress_axis": q.get("stress_axis",""),
                    "config_key": cfg_key,
                    "top_k": cfg.get("top_k"), "alpha_vector": cfg.get("alpha_vector"),
                    "alpha_lexical": cfg.get("alpha_lexical"),
                    "cross_encoder_weight": cfg.get("cross_encoder_weight"),
                    "diversity_penalty": cfg.get("diversity_penalty"),
                    "passed": False, "facts_found": 0, "facts_total": 0,
                    "must_not_violations": 0, "answer_mode": "error",
                    "num_hits": 0, "top_chunk_ids": [], "top_chunk_texts": [],
                    "answer_text": f"ERROR: {exc}",
                    "fact_results": {}, "retrieval_elapsed": 0, "llm_elapsed": 0,
                    "total_elapsed": 0,
                }

            cfg_results.append(r)
            all_results.append(r)
            if r["passed"]:
                cfg_pass += 1
            n_done += 1

        # Per-config progress line
        pct = cfg_pass / n_questions * 100 if n_questions else 0
        elapsed_total = time.perf_counter() - run_start
        rate = n_done / elapsed_total if elapsed_total > 0 else 0
        eta_s = (n_total - n_done) / rate if rate > 0 else 0
        eta_str = f"{eta_s/60:.1f}m" if eta_s > 0 else "?"
        marker = _ok("✓") if pct >= 70 else (_fail("✗") if pct < 50 else "~")
        print(
            f"  [{cfg_idx:>3}/{n_configs}] {marker} {pct:4.0f}%  "
            f"({cfg_pass}/{n_questions})  {_dim(cfg_key[:55])}  ETA {eta_str}"
        )

        # Incremental save after each config
        raw_json.write_text(
            json.dumps({"ts": ts, "grid": grid, "results": all_results}, indent=2),
            encoding="utf-8",
        )

    # ── Summaries ─────────────────────────────────────────────────────────────
    summaries = _summarise(all_results, questions)
    _write_csv(all_results, detail_csv)
    _write_csv(summaries, summary_csv)
    raw_json.write_text(
        json.dumps({"ts": ts, "grid": grid, "results": all_results, "summaries": summaries}, indent=2),
        encoding="utf-8",
    )

    total_elapsed = time.perf_counter() - run_start
    print(f"\n{'═'*74}")
    print(f"  Finished {n_done} calls in {total_elapsed/60:.1f} minutes")
    _print_top_configs(summaries, n=10)
    print(f"\n  Detail CSV : {detail_csv}")
    print(f"  Summary CSV: {summary_csv}")
    print(f"  Raw JSON   : {raw_json}")
    print(f"{'═'*74}\n")


if __name__ == "__main__":
    main()
