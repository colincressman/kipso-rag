"""Golden evaluation dataset runner.

Runs each question in data/qa/golden_eval.json through the full RAG pipeline
and checks whether expected facts appear in the answer. Reports pass/fail per
question with the actual answer text for manual review.

Usage:
    python scripts/run_golden_eval.py
    python scripts/run_golden_eval.py --output data/diagnostics/golden_eval_results.json
    python scripts/run_golden_eval.py --ids gold_01 gold_04  # run specific questions
    python scripts/run_golden_eval.py --top-k 8
"""
from __future__ import annotations

import argparse
import json
import sys
import io
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure Unicode box-drawing characters print correctly on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from api import llm_answer, rag_retrieve
from utils.runtime_defaults import DEFAULT_DB_DSN, DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME

GOLDEN_EVAL_PATH = PROJECT_ROOT / "data" / "qa" / "golden_eval_v2.json"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_questions(ids: Optional[List[str]] = None, dataset_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = dataset_path or GOLDEN_EVAL_PATH
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def _normalize_for_fact_check(text: str) -> str:
    """Normalize answer text before substring matching.
    Converts '42%' → '42 percent' so expected_facts written with 'percent'
    match answers that use the '%' symbol (and vice versa).
    """
    import re
    text = re.sub(r'(\d)\s*%', r'\1 percent', text)
    # Normalize spaced number-commas: "7 , 930" → "7,930"
    text = re.sub(r'(\d)\s*,\s*(\d)', r'\1,\2', text)
    return text.lower()


def _check_facts(answer_text: str, expected_facts: List[str]) -> Dict[str, bool]:
    """Return a mapping of each expected fact to True/False (case-insensitive substring).

    A fact string may contain ``|``-separated alternatives (e.g. ``"lookahead|future information"``),
    in which case the fact passes if ANY alternative is found in the answer.
    """
    normalized = _normalize_for_fact_check(answer_text)
    results: Dict[str, bool] = {}
    for fact in expected_facts:
        alternatives = [_normalize_for_fact_check(a) for a in fact.split("|")]
        results[fact] = any(alt in normalized for alt in alternatives)
    return results


def _color(text: str, code: str) -> str:
    """ANSI color for terminal output (skipped if not a TTY)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _ok(text: str) -> str:
    return _color(text, "32")


def _fail(text: str) -> str:
    return _color(text, "31")


def _warn(text: str) -> str:
    return _color(text, "33")


# ──────────────────────────────────────────────────────────────────────────────
# Per-question runner
# ──────────────────────────────────────────────────────────────────────────────

def run_question(
    q: Dict[str, Any],
    *,
    top_k: int = 8,
    db_dsn: str = str(DEFAULT_DB_DSN),
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_model: str = DEFAULT_EMBED_MODEL_NAME,
    verbose: bool = False,
) -> Dict[str, Any]:
    question = q["question"]
    expected_facts = q.get("expected_facts", [])
    must_not_contain = q.get("must_not_contain", [])
    collection = q.get("collection")  # optional collection scope (e.g. "Notes")

    t0 = time.perf_counter()

    # Route + Retrieve (full api pipeline: router → HyDE → cosine scan → rerank → context pack)
    retrieved = rag_retrieve(
        question,
        top_k=top_k,
        db_dsn=db_dsn,
        embed_backend=embed_backend,
        embed_model_name=embed_model,
        collection=collection,
    )

    # Answer
    answer = llm_answer(question, retrieved)

    elapsed = time.perf_counter() - t0

    answer_text = str(answer.get("answer") or answer.get("refusal") or "")

    # Check facts
    fact_results = _check_facts(answer_text, expected_facts)
    must_not_results = {phrase: phrase.lower() in answer_text.lower() for phrase in must_not_contain}

    facts_passed = sum(1 for v in fact_results.values() if v)
    facts_total = len(fact_results)
    must_not_violations = sum(1 for v in must_not_results.values() if v)

    passed = facts_total > 0 and facts_passed == facts_total and must_not_violations == 0

    internet_triggered = bool((retrieved.get("internet_fallback") or {}).get("triggered"))
    pipeline_mode_expected = q.get("pipeline_mode_expected", "corpus")
    # For internet questions, flag if the pipeline didn't actually trigger web search
    mode_matched = (
        internet_triggered if pipeline_mode_expected == "internet"
        else True  # corpus/general: we don't auto-fail on mode mismatch
    )

    result = {
        "id": q["id"],
        "question": question,
        "category": q.get("category", ""),
        "source_books": q.get("source_books", []),
        "pipeline_mode_expected": pipeline_mode_expected,
        "passed": passed,
        "mode_matched": mode_matched,
        "facts_found": facts_passed,
        "facts_total": facts_total,
        "fact_results": fact_results,
        "must_not_violations": must_not_violations,
        "must_not_results": must_not_results if must_not_contain else {},
        "answer_text": answer_text,
        "answer_mode": answer.get("mode"),
        "internet_triggered": internet_triggered,
        "num_hits": len(retrieved.get("hits") or []),
        "num_context_chunks": len((retrieved.get("context_pack") or {}).get("selected_chunks") or []),
        "elapsed_seconds": round(elapsed, 2),
        "quality_hints": q.get("quality_hints", []),
        "human_review_required": q.get("human_review_required", False),
    }

    if verbose:
        status = _ok("PASS") if passed else _fail("FAIL")
        print(f"\n  [{status}] {q['id']} — {question[:80]}")
        print(f"         Facts: {facts_passed}/{facts_total}")
        for fact, found in fact_results.items():
            sym = _ok("✓") if found else _fail("✗")
            print(f"           {sym}  {fact!r}")
        if must_not_violations:
            for phrase, found in must_not_results.items():
                if found:
                    print(_warn(f"         MUST_NOT violated: {phrase!r}"))
        if not mode_matched:
            print(_warn(f"         ⚠ internet_triggered={internet_triggered} but expected mode={pipeline_mode_expected}"))
        print(f"         Mode: {result['answer_mode']}  | Hits: {result['num_hits']}  | Internet: {internet_triggered} | {elapsed:.1f}s")
        print(f"         Answer (first 600 chars):")
        print(f"           {answer_text[:600]!r}")
        if result['human_review_required'] and result['quality_hints']:
            print(_warn(f"         Human review hints:"))
            for hint in result['quality_hints']:
                print(f"           • {hint}")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run golden evaluation dataset through the RAG pipeline.")
    parser.add_argument("--ids", nargs="*", help="Run only specific question IDs (e.g. gold_01 gold_04).")
    parser.add_argument("--top-k", type=int, default=8, help="Number of chunks to retrieve per question.")
    parser.add_argument("--output", type=str, default=None, help="Write JSON results to this path (auto-saved to data/diagnostics/ if omitted).")
    parser.add_argument("--dataset", type=str, default=None, help="Path to eval JSON file (default: data/qa/golden_eval.json). Use 'v2' as shorthand for golden_eval_v2.json.")
    parser.add_argument("--db-dsn", type=str, default=str(DEFAULT_DB_DSN))
    parser.add_argument("--embed-backend", type=str, default=DEFAULT_EMBED_BACKEND)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each question's answer and fact results.")
    args = parser.parse_args()

    if args.dataset:
        dataset_path = Path(args.dataset)
    else:
        dataset_path = GOLDEN_EVAL_PATH
    dataset_name = dataset_path.stem

    questions = _load_questions(args.ids, dataset_path=dataset_path)
    if not questions:
        print("No questions matched — check --ids filter or golden_eval.json.")
        sys.exit(1)

    print(f"\nGolden Evaluation — {len(questions)} question(s)  [top_k={args.top_k}]  [{dataset_name}]")
    print(f"DB: {args.db_dsn}")
    print("=" * 70)

    results: List[Dict[str, Any]] = []
    passed_count = 0
    total_start = time.perf_counter()

    for q in questions:
        result = run_question(
            q,
            top_k=args.top_k,
            db_dsn=args.db_dsn,
            embed_backend=args.embed_backend,
            embed_model=args.embed_model,
            verbose=args.verbose,
        )
        results.append(result)
        if result["passed"]:
            passed_count += 1

        # Always print a compact status line even without --verbose
        if not args.verbose:
            status = _ok("PASS") if result["passed"] else _fail("FAIL")
            facts = f"{result['facts_found']}/{result['facts_total']}"
            mode_warn = _warn(" ⚠NO-WEB") if not result["mode_matched"] else ""
            human_flag = _warn(" [HUMAN]") if result["human_review_required"] else ""
            print(f"  [{status}] {result['id']:12s} facts={facts}  mode={result['answer_mode']:20s}  {result['elapsed_seconds']:.1f}s{mode_warn}{human_flag}")
            print(f"           {result['answer_text'][:300].strip()!r}")
            print()

    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 70)
    total = len(results)
    pct = 100 * passed_count / total if total else 0
    summary_str = f"PASSED {passed_count}/{total} ({pct:.0f}%)"
    print(_ok(summary_str) if passed_count == total else _warn(summary_str))
    print(f"Total time: {total_elapsed:.1f}s")

    # Per-mode breakdown
    for mode_label in ("corpus", "internet", "general"):
        mode_results = [r for r in results if r.get("pipeline_mode_expected") == mode_label]
        if mode_results:
            mp = sum(1 for r in mode_results if r["passed"])
            print(f"  {mode_label:10s}: {mp}/{len(mode_results)} passed", end="")
            no_web = [r for r in mode_results if not r["mode_matched"]]
            if no_web:
                print(_warn(f"  ({len(no_web)} internet question(s) did not trigger web search)"), end="")
            print()
    print()

    # Failures detail
    failures = [r for r in results if not r["passed"]]
    if failures:
        print("Failures:")
        for r in failures:
            missing = [f for f, v in r["fact_results"].items() if not v]
            print(f"  {r['id']}: missing facts: {missing}")
            if r["must_not_violations"]:
                violated = [f for f, v in r["must_not_results"].items() if v]
                print(f"         must_not violations: {violated}")
        print()

    # Human review section
    review_needed = [r for r in results if r.get("human_review_required")]
    if review_needed:
        print("─" * 70)
        print(_warn(f"Human Review Required ({len(review_needed)} questions):"))
        for r in review_needed:
            status = _ok("PASS") if r["passed"] else _fail("FAIL")
            print(f"  [{status}] {r['id']} ({r.get('pipeline_mode_expected','?')})")
            print(f"         Answer: {r['answer_text'][:400].strip()!r}")
            for hint in r.get("quality_hints", []):
                print(f"         • {hint}")
            print()


    # Save output — always auto-save with timestamp; --output overrides the path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    auto_path = PROJECT_ROOT / "data" / "diagnostics" / f"{dataset_name}_{timestamp}.json"
    out_path = Path(args.output) if args.output else auto_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed_count,
        "total": total,
        "pct": round(pct, 1),
        "elapsed_seconds": round(total_elapsed, 2),
        "top_k": args.top_k,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results written to: {out_path}")

    sys.exit(0 if passed_count == total else 1)


if __name__ == "__main__":
    main()
