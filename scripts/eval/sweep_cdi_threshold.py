"""Sweep cdi_threshold against the golden eval to find the optimal value.

Iterates over [0.10, 0.15, 0.20, 0.25], monkey-patches
``retrieval.router._CDI_THRESHOLD`` for each run, and prints a side-by-side
pass/fail table so you can pick the best value to put in scoring.yaml.

Usage:
    python scripts/eval/sweep_cdi_threshold.py
    python scripts/eval/sweep_cdi_threshold.py --values 0.10 0.15 0.20 0.25
    python scripts/eval/sweep_cdi_threshold.py --no-llm   # retrieval-only (fast)
    python scripts/eval/sweep_cdi_threshold.py --ids gold_i01 gold_i02  # subset
    python scripts/eval/sweep_cdi_threshold.py --output data/diagnostics/cdi_sweep.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import retrieval.router early so we can patch _CDI_THRESHOLD at module level
import retrieval.router as _router_mod  # noqa: E402

from api import llm_answer, rag_retrieve  # noqa: E402
from utils.runtime_defaults import DEFAULT_DB_DSN, DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME  # noqa: E402

GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "qa" / "golden_eval_v2.json"
DEFAULT_CDI_VALUES = [0.10, 0.15, 0.20, 0.25]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_questions(ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    data = json.loads(GOLDEN_EVAL_PATH.read_text(encoding="utf-8"))
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def _normalize(text: str) -> str:
    import re
    text = re.sub(r"(\d)\s*%", r"\1 percent", text)
    return text.lower()


def _check_facts(answer_text: str, expected_facts: List[str]) -> Dict[str, bool]:
    normalized = _normalize(answer_text)
    return {
        fact: any(_normalize(alt) in normalized for alt in fact.split("|"))
        for fact in expected_facts
    }


def _run_question(
    q: Dict[str, Any],
    *,
    db_dsn: str,
    embed_backend: str,
    embed_model: str,
    top_k: int,
    llm_enabled: bool,
) -> Dict[str, Any]:
    question = q["question"]
    expected_facts = q.get("expected_facts", [])

    t0 = time.perf_counter()
    result = rag_retrieve(
        question,
        db_dsn=db_dsn,
        embed_backend=embed_backend,
        embed_model_name=embed_model,
        top_k=top_k,
    )

    answer_text = ""
    answer_mode = "skipped"
    if llm_enabled:
        answer = llm_answer(question, result)
        answer_text = str(answer.get("answer") or answer.get("refusal") or "")
        answer_mode = str(answer.get("mode") or "")

    elapsed = round(time.perf_counter() - t0, 1)
    fact_results = _check_facts(answer_text, expected_facts)
    facts_passed = sum(1 for v in fact_results.values() if v)
    passed = len(fact_results) > 0 and facts_passed == len(fact_results)

    return {
        "qid": q["id"],
        "category": q.get("category", ""),
        "passed": passed,
        "facts_passed": facts_passed,
        "facts_total": len(fact_results),
        "answer_mode": answer_mode,
        "elapsed": elapsed,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main sweep
# ──────────────────────────────────────────────────────────────────────────────

def run_sweep(
    cdi_values: List[float],
    questions: List[Dict[str, Any]],
    *,
    db_dsn: str,
    embed_backend: str,
    embed_model: str,
    top_k: int,
    llm_enabled: bool,
) -> Dict[float, List[Dict[str, Any]]]:
    results: Dict[float, List[Dict[str, Any]]] = {}

    for cdi in cdi_values:
        _router_mod._CDI_THRESHOLD = cdi
        print(f"\n{'='*70}")
        print(f"  cdi_threshold = {cdi}")
        print(f"{'='*70}")

        run_results = []
        for q in questions:
            r = _run_question(
                q,
                db_dsn=db_dsn,
                embed_backend=embed_backend,
                embed_model=embed_model,
                top_k=top_k,
                llm_enabled=llm_enabled,
            )
            status = "PASS" if r["passed"] else "FAIL"
            facts = f"{r['facts_passed']}/{r['facts_total']}"
            print(f"  [{status}] {r['qid']:<14} facts={facts:<5} mode={r['answer_mode']:<20} {r['elapsed']}s")
            run_results.append(r)

        total = len(run_results)
        passed = sum(1 for r in run_results if r["passed"])
        print(f"\n  Score: {passed}/{total} ({100*passed//total}%)")
        results[cdi] = run_results

    return results


def _print_summary(results: Dict[float, List[Dict[str, Any]]]) -> None:
    cdi_values = sorted(results)
    all_qids = [r["qid"] for r in next(iter(results.values()))]

    # Header
    col = 18
    header = f"{'Question':<16}" + "".join(f"  cdi={v:<6}" for v in cdi_values)
    print(f"\n{'─'*len(header)}")
    print("SUMMARY")
    print(f"{'─'*len(header)}")
    print(header)
    print("─" * len(header))

    for qid in all_qids:
        row = f"{qid:<16}"
        for cdi in cdi_values:
            r = next(x for x in results[cdi] if x["qid"] == qid)
            cell = "PASS" if r["passed"] else "FAIL"
            row += f"  {cell:<10}"
        print(row)

    print("─" * len(header))
    totals_row = f"{'TOTAL':<16}"
    for cdi in cdi_values:
        passed = sum(1 for r in results[cdi] if r["passed"])
        total = len(results[cdi])
        totals_row += f"  {passed}/{total:<8}"
    print(totals_row)
    print("─" * len(header))

    # Recommendation
    best_cdi = max(cdi_values, key=lambda c: sum(1 for r in results[c] if r["passed"]))
    best_score = sum(1 for r in results[best_cdi] if r["passed"])
    print(f"\nBest cdi_threshold = {best_cdi}  ({best_score}/{len(results[best_cdi])} pass)")
    print(f"  → set in configs/scoring.yaml: routing.cdi_threshold: {best_cdi}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep cdi_threshold over golden eval")
    parser.add_argument(
        "--values", type=float, nargs="+", default=DEFAULT_CDI_VALUES,
        help="CDI threshold values to test (default: 0.10 0.15 0.20 0.25)",
    )
    parser.add_argument(
        "--ids", type=str, nargs="+", default=None,
        help="Subset of question IDs to evaluate",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="top_k for retrieval (default: 8)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Skip LLM answer step (routing/retrieval check only)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Write full results to this JSON file",
    )
    args = parser.parse_args()

    db_dsn = str(DEFAULT_DB_DSN)
    embed_backend = DEFAULT_EMBED_BACKEND
    embed_model = DEFAULT_EMBED_MODEL_NAME

    questions = _load_questions(args.ids)
    print(f"CDI threshold sweep — {len(questions)} question(s), {len(args.values)} threshold value(s)")
    print(f"LLM: {'disabled (--no-llm)' if args.no_llm else 'enabled'}")

    results = run_sweep(
        args.values,
        questions,
        db_dsn=db_dsn,
        embed_backend=embed_backend,
        embed_model=embed_model,
        top_k=args.top_k,
        llm_enabled=not args.no_llm,
    )

    _print_summary(results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cdi_values": args.values,
            "questions": [q["id"] for q in questions],
            "results": {str(cdi): runs for cdi, runs in results.items()},
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
