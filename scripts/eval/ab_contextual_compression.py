"""A/B evaluation: contextual compression enabled vs disabled.

Runs every question in golden_eval_v2.json through both conditions and
produces a side-by-side comparison of pass rate and latency.

Usage::

    python scripts/eval/ab_contextual_compression.py
    python scripts/eval/ab_contextual_compression.py --verbose
    python scripts/eval/ab_contextual_compression.py --ids gold_c01 gold_c08
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.eval.run_golden_eval import _load_questions, run_question
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
)

GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "qa" / "golden_eval_v2.json"
DIAG_DIR = PROJECT_ROOT / "data" / "diagnostics"


# ── colour helpers ──────────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

_ok   = lambda t: _c(t, "32")
_fail = lambda t: _c(t, "31")
_warn = lambda t: _c(t, "33")
_dim  = lambda t: _c(t, "2")


# ── run one condition ────────────────────────────────────────────────────────

def _run_condition(
    questions: List[Dict[str, Any]],
    *,
    compress: bool,
    top_k: int,
    db_dsn: str,
    embed_backend: str,
    embed_model: str,
    verbose: bool,
) -> tuple[List[Dict[str, Any]], float]:
    """Run all questions with compression toggled to *compress*.

    Returns (results_list, total_elapsed_seconds).
    """
    import llm.answer_context as _ctx_mod

    _ctx_mod.DEFAULT_CONTEXTUAL_COMPRESSION_ENABLED = compress

    label = "compression=ON " if compress else "compression=OFF"
    print(f"\n{'─'*70}")
    print(f"  Condition: {label}")
    print(f"{'─'*70}")

    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for q in questions:
        r = run_question(
            q,
            top_k=top_k,
            db_dsn=db_dsn,
            embed_backend=embed_backend,
            embed_model=embed_model,
            verbose=verbose,
        )
        results.append(r)
        status = _ok("PASS") if r["passed"] else _fail("FAIL")
        facts = f"{r['facts_found']}/{r['facts_total']}"
        print(f"  [{status}] {r['id']:12s} facts={facts}  {r['elapsed_seconds']:.1f}s")
        if verbose:
            print(f"           {r['answer_text'][:200].strip()!r}")

    total_elapsed = time.perf_counter() - t0
    passed = sum(1 for r in results if r["passed"])
    pct = 100 * passed / len(results) if results else 0
    summary = f"PASSED {passed}/{len(results)} ({pct:.0f}%)  — {total_elapsed:.1f}s total"
    print(f"\n  {_ok(summary) if passed == len(results) else _warn(summary)}")

    return results, total_elapsed


# ── comparison report ────────────────────────────────────────────────────────

def _report(
    off_results: List[Dict[str, Any]],
    on_results: List[Dict[str, Any]],
    off_elapsed: float,
    on_elapsed: float,
) -> Dict[str, Any]:
    off_map = {r["id"]: r for r in off_results}
    on_map  = {r["id"]: r for r in on_results}

    off_pass = sum(1 for r in off_results if r["passed"])
    on_pass  = sum(1 for r in on_results  if r["passed"])
    n = len(off_results)

    gained = [qid for qid, r in on_map.items()  if r["passed"] and not off_map[qid]["passed"]]
    lost   = [qid for qid, r in off_map.items() if r["passed"] and not on_map[qid]["passed"]]

    latency_delta = on_elapsed - off_elapsed
    latency_pct   = 100 * latency_delta / off_elapsed if off_elapsed else 0

    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  compression=OFF  {off_pass}/{n} passed  {off_elapsed:.1f}s")
    print(f"  compression=ON   {on_pass}/{n} passed  {on_elapsed:.1f}s")
    delta_str = f"{on_pass - off_pass:+d} questions"
    print(f"\n  Pass rate delta : {delta_str}")
    sign = "+" if latency_delta >= 0 else ""
    print(f"  Latency delta   : {sign}{latency_delta:.1f}s  ({sign}{latency_pct:.1f}%)")

    if gained:
        print(f"\n  Questions gained by compression ON  : {', '.join(gained)}")
    if lost:
        print(f"  Questions lost  by compression ON  : {', '.join(lost)}")
    if not gained and not lost:
        print("\n  No questions changed pass/fail status.")

    verdict: str
    if on_pass > off_pass:
        verdict = "ENABLE — compression improved pass rate with no regression"
    elif on_pass == off_pass and latency_delta < 5.0:
        verdict = "NEUTRAL — same pass rate, acceptable latency increase"
    elif on_pass == off_pass and latency_delta >= 5.0:
        verdict = "SKIP — same pass rate but adds significant latency"
    else:
        verdict = "DISABLE — compression hurt pass rate"

    print(f"\n  Recommendation  : {_ok(verdict) if 'ENABLE' in verdict or 'NEUTRAL' in verdict else _warn(verdict)}")
    print()

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_questions": n,
        "off": {"passed": off_pass, "elapsed_seconds": off_elapsed},
        "on":  {"passed": on_pass,  "elapsed_seconds": on_elapsed},
        "gained": gained,
        "lost":   lost,
        "latency_delta_seconds": latency_delta,
        "recommendation": verdict,
        "per_question": {
            qid: {
                "off_passed": off_map[qid]["passed"],
                "on_passed":  on_map[qid]["passed"],
                "off_elapsed": off_map[qid]["elapsed_seconds"],
                "on_elapsed":  on_map[qid]["elapsed_seconds"],
            }
            for qid in off_map
        },
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="A/B eval for contextual compression.")
    parser.add_argument("--ids", nargs="*", help="Limit to specific question IDs.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--db-dsn", type=str, default=str(DEFAULT_DB_DSN))
    parser.add_argument("--embed-backend", type=str, default=DEFAULT_EMBED_BACKEND)
    parser.add_argument("--embed-model",   type=str, default=DEFAULT_EMBED_MODEL_NAME)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON results to this path (auto-saved if omitted).")
    args = parser.parse_args()

    dataset_path = Path(args.dataset) if args.dataset else GOLDEN_EVAL_PATH
    questions = _load_questions(args.ids, dataset_path=dataset_path)
    if not questions:
        print("No questions matched.")
        sys.exit(1)

    print(f"\nContextual Compression A/B Evaluation")
    print(f"{len(questions)} question(s)  top_k={args.top_k}  [{dataset_path.stem}]")
    print(f"DB: {args.db_dsn}")

    kw = dict(
        top_k=args.top_k,
        db_dsn=args.db_dsn,
        embed_backend=args.embed_backend,
        embed_model=args.embed_model,
        verbose=args.verbose,
    )

    off_results, off_elapsed = _run_condition(questions, compress=False, **kw)
    on_results,  on_elapsed  = _run_condition(questions, compress=True,  **kw)

    # Always restore to disabled after the run
    import llm.answer_context as _ctx_mod
    _ctx_mod.DEFAULT_CONTEXTUAL_COMPRESSION_ENABLED = False

    report = _report(off_results, on_results, off_elapsed, on_elapsed)

    # Save
    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        DIAG_DIR.mkdir(parents=True, exist_ok=True)
        out_path = DIAG_DIR / f"ab_compression_{ts}.json"

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Results saved → {out_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
