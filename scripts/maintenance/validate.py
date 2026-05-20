"""
validate.py — Master regression and improvement tracking script.

Answers two questions after every code change:
  A) Did we break anything?   — unit tests, DB health, acceptance battery
  B) Did we improve anything? — pass-rates and latency vs. a saved baseline

Modes
-----
  Full (default):
      Unit tests + DB health + retrieval harness + acceptance battery (LLM required)

  Fast (--fast):
      Unit tests + DB health + retrieval harness only.
      Uses the hashing embed backend — no Ollama required.
      Run this for quick sanity checks after small changes.

  Save baseline (--save-baseline):
      Runs the full suite, then saves results as the new comparison baseline.
      Use this after a confirmed improvement to lock in the new bar.

Usage
-----
    python scripts/validate.py
    python scripts/validate.py --fast
    python scripts/validate.py --save-baseline
    python scripts/validate.py --only core_ss_01,internet_01
    python scripts/validate.py --embed-backend hashing --fast
"""

from __future__ import annotations

import argparse
import json
import psycopg
from psycopg.rows import dict_row
import subprocess
import sys
import time

# Ensure UTF-8 output on Windows regardless of the active console code page.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Force line-buffered output so progress is visible as it happens when piped.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
)

BASELINE_PATH = PROJECT_ROOT / "data" / "diagnostics" / "validation_baseline.json"
REPORT_DIR = PROJECT_ROOT / "data" / "diagnostics"

# ── ANSI colours (disabled on Windows if not supported) ───────────────────────
_COLORS = sys.platform != "win32" or "ANSICON" in __import__("os").environ
_GREEN  = "\033[92m" if _COLORS else ""
_RED    = "\033[91m" if _COLORS else ""
_YELLOW = "\033[93m" if _COLORS else ""
_BLUE   = "\033[94m" if _COLORS else ""
_BOLD   = "\033[1m"  if _COLORS else ""
_RESET  = "\033[0m"  if _COLORS else ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _header(text: str) -> None:
    width = 72
    print(f"\n{_BOLD}{'═' * width}{_RESET}")
    print(f"{_BOLD}  {text}{_RESET}")
    print(f"{_BOLD}{'═' * width}{_RESET}")


def _result_line(label: str, passed: bool, detail: str = "") -> None:
    icon = f"{_GREEN}✓{_RESET}" if passed else f"{_RED}✗{_RESET}"
    suffix = f"  {_YELLOW}{detail}{_RESET}" if detail else ""
    print(f"  {icon}  {label}{suffix}")


def _delta_str(old: float, new: float, higher_is_better: bool = True) -> str:
    diff = new - old
    if abs(diff) < 0.001:
        return "~"
    arrow = "▲" if diff > 0 else "▼"
    color = _GREEN if (diff > 0) == higher_is_better else _RED
    return f"{color}{arrow}{abs(diff):.3f}{_RESET}"


# ══════════════════════════════════════════════════════════════════════════════
# Section 1 — Unit tests (pytest)
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests() -> Dict[str, Any]:
    """Run pytest and return a summary dict."""
    _header("1 / Unit Tests  (pytest)")

    cmd = [
        sys.executable, "-m", "pytest",
        str(PROJECT_ROOT / "tests"),
        "--tb=short", "-q",
        "--no-header",
    ]
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.perf_counter() - start
    output = proc.stdout + proc.stderr

    passed = failed = errors = 0
    for line in output.splitlines():
        # pytest summary line: "5 passed, 2 failed, 1 error in 12.34s"
        if " passed" in line or " failed" in line or " error" in line:
            import re
            m_pass  = re.search(r"(\d+) passed",  line)
            m_fail  = re.search(r"(\d+) failed",  line)
            m_error = re.search(r"(\d+) error",   line)
            if m_pass:  passed  = int(m_pass.group(1))
            if m_fail:  failed  = int(m_fail.group(1))
            if m_error: errors  = int(m_error.group(1))

    total = passed + failed + errors
    suite_passed = failed == 0 and errors == 0

    # Print a condensed view of failures/errors (not the whole output)
    if not suite_passed:
        for line in output.splitlines():
            if line.strip():
                print(f"  {line}")
    else:
        print(f"  All {passed} tests passed in {elapsed:.1f}s")

    _result_line(
        f"Unit tests  ({passed}/{total} passed)",
        suite_passed,
        f"{'failed: ' + str(failed) if failed else ''}{'errors: ' + str(errors) if errors else ''}",
    )

    return {
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "total": total,
        "suite_passed": suite_passed,
        "elapsed_seconds": round(elapsed, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 2 — DB health
# ══════════════════════════════════════════════════════════════════════════════

def run_db_health(db_path: str) -> Dict[str, Any]:
    """Check document/chunk counts and embedding coverage."""
    _header("2 / DB Health")

    p = db_path
    conn = psycopg.connect(p, row_factory=dict_row)
    try:
        doc_count   = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunk_count = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded    = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
        zero_text   = conn.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE text IS NULL OR trim(text) = ''"
        ).fetchone()["n"]

        # Per-source-type breakdown
        type_rows = conn.execute(
            "SELECT source_type, COUNT(*) AS n FROM chunks GROUP BY source_type ORDER BY COUNT(*) DESC"
        ).fetchall()
        by_source_type = {(row["source_type"] or "unknown"): row["n"] for row in type_rows}

        # Per-role breakdown
        role_rows = conn.execute(
            "SELECT structural_role, COUNT(*) AS n FROM chunks GROUP BY structural_role ORDER BY COUNT(*) DESC"
        ).fetchall()
        by_role = {(row["structural_role"] or "unknown"): row["n"] for row in role_rows}
    finally:
        conn.close()

    embed_pct = (embedded / chunk_count * 100) if chunk_count else 0.0
    healthy = doc_count > 0 and chunk_count > 0 and embed_pct >= 95.0 and zero_text == 0

    print(f"  Documents   : {doc_count}")
    print(f"  Chunks      : {chunk_count}")
    print(f"  Embedded    : {embedded}  ({embed_pct:.1f}%)")
    print(f"  Zero-text   : {zero_text}")
    print(f"  By source   : {by_source_type}")
    print(f"  By role     : {by_role}")

    _result_line(
        "DB health",
        healthy,
        "" if healthy else f"embed_pct={embed_pct:.1f}%  zero_text={zero_text}",
    )

    return {
        "healthy": healthy,
        "doc_count": doc_count,
        "chunk_count": chunk_count,
        "embedded_count": embedded,
        "embed_coverage_pct": round(embed_pct, 2),
        "zero_text_chunks": zero_text,
        "by_source_type": by_source_type,
        "by_structural_role": by_role,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 3 — Retrieval harness (no LLM required)
# ══════════════════════════════════════════════════════════════════════════════

def run_retrieval_harness(
    db_path: str,
    embed_backend: str,
    embed_model: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run the structured retrieval test cases from evaluate_retrieval_harness."""
    _header("3 / Retrieval Harness  (no LLM)")

    # Import here so failures in the import are surfaced clearly.
    from scripts.evaluate_retrieval_harness import DEFAULT_CASES, _evaluate_case

    results: List[Dict[str, Any]] = []
    start_total = time.perf_counter()

    for case in DEFAULT_CASES:
        t0 = time.perf_counter()
        try:
            result = _evaluate_case(
                case,
                db_dsn=db_path,
                top_k=top_k,
                embed_backend=embed_backend,
                embed_model_name=embed_model,
                source_type=None,
                structural_role=None,
            )
            elapsed = time.perf_counter() - t0
            passed = result.checks_passed
            _result_line(
                f"[{result.case_id}] {result.category}  — {case.query[:60]}",
                passed,
                ", ".join(result.failed_checks) if result.failed_checks else f"{elapsed:.2f}s",
            )
            results.append({
                "case_id": result.case_id,
                "category": result.category,
                "passed": passed,
                "failed_checks": list(result.failed_checks),
                "top1_source": result.top1_source,
                "top1_role": result.top1_role,
                "unique_sources_top_n": result.unique_sources_top_n,
                "elapsed_seconds": round(elapsed, 3),
            })
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            print(f"  {_RED}✗  [{case.case_id}] ERROR: {exc}{_RESET}")
            results.append({
                "case_id": case.case_id,
                "category": case.category,
                "passed": False,
                "failed_checks": [f"exception: {str(exc)[:120]}"],
                "elapsed_seconds": round(elapsed, 3),
            })

    total_elapsed = time.perf_counter() - start_total
    total   = len(results)
    passed  = sum(1 for r in results if r["passed"])
    pass_rate = round(passed / total, 4) if total else 0.0

    print(f"\n  Result: {passed}/{total} passed  ({pass_rate * 100:.1f}%)  in {total_elapsed:.1f}s")

    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        cat = by_category.setdefault(r["category"], {"count": 0, "passed": 0})
        cat["count"] += 1
        if r["passed"]:
            cat["passed"] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": pass_rate,
        "by_category": by_category,
        "cases": results,
        "elapsed_seconds": round(total_elapsed, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 4 — Acceptance battery (LLM required)
# ══════════════════════════════════════════════════════════════════════════════

def run_acceptance_battery(
    db_path: str,
    embed_backend: str,
    embed_model: str,
    llm_config: str,
    only_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run the full acceptance battery from run_acceptance_battery."""
    _header("4 / Acceptance Battery  (LLM + Ollama required)")

    from scripts.run_acceptance_battery import CASES, _run_case, _summarize

    selected = [c for c in CASES if not only_ids or c.id in only_ids]
    if not selected:
        print(f"  {_RED}No cases matched the --only filter.{_RESET}")
        return {"skipped": True}

    results: List[Dict[str, Any]] = []
    total_start = time.perf_counter()

    for idx, case in enumerate(selected, 1):
        print(f"  [{idx:02d}/{len(selected)}] {case.id}  {case.question[:65]}", flush=True)
        try:
            result = _run_case(
                case,
                db_dsn=db_path,
                embed_backend=embed_backend,
                embed_model=embed_model,
                llm_config=llm_config,
            )
            passed = result["checks"]["passed"]
            failed_checks = result["checks"]["failed_checks"]
            review_flags  = result["checks"]["review_flags"]
            _result_line(
                f"  {case.id} — {case.category}",
                passed,
                ", ".join(failed_checks + [f"⚑ {f}" for f in review_flags]) if not passed else "",
            )
            results.append(result)
        except Exception as exc:
            print(f"  {_RED}✗  {case.id} — ERROR: {exc}{_RESET}")
            results.append({
                "id": case.id,
                "category": case.category,
                "question": case.question,
                "checks": {"passed": False, "failed_checks": [f"exception: {str(exc)[:120]}"], "review_flags": [], "metrics": {}},
                "total_seconds": 0,
            })

    total_elapsed = time.perf_counter() - total_start
    summary = _summarize(results)

    pass_rate = summary["pass_rate"]
    print(f"\n  Result: {summary['passed_cases']}/{summary['total_cases']} passed"
          f"  ({pass_rate * 100:.1f}%)  in {total_elapsed:.1f}s")
    if summary.get("review_flagged_cases", 0):
        print(f"  {_YELLOW}Review flags: {summary['review_flagged_cases']} cases need manual inspection{_RESET}")

    for cat, stats in summary.get("by_category", {}).items():
        c, p = stats["count"], stats["passed"]
        mark = _GREEN if p == c else (_YELLOW if p > 0 else _RED)
        print(f"    {cat:35s}: {mark}{p}/{c}{_RESET}")

    return {
        "summary": summary,
        "elapsed_seconds": round(total_elapsed, 2),
        "cases": [
            {
                "id": r["id"],
                "category": r["category"],
                "passed": r["checks"]["passed"],
                "failed_checks": r["checks"]["failed_checks"],
                "review_flags": r["checks"].get("review_flags", []),
                "metrics": r["checks"].get("metrics", {}),
                "answer_mode": (r.get("answer") or {}).get("mode"),
                "answer_seconds": (r.get("answer") or {}).get("seconds"),
                "total_seconds": r.get("total_seconds"),
            }
            for r in results
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# Section 5 — Baseline comparison
# ══════════════════════════════════════════════════════════════════════════════

def compare_to_baseline(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Load the baseline and print a delta table. Returns delta dict or None."""
    if not BASELINE_PATH.exists():
        print(f"\n  {_YELLOW}No baseline found at {BASELINE_PATH}.{_RESET}")
        print(f"  Run with --save-baseline after a confirmed good run to set one.")
        return None

    try:
        baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"\n  {_RED}Could not load baseline: {exc}{_RESET}")
        return None

    _header("5 / Baseline Comparison")

    metrics: List[Tuple[str, float, float, bool]] = []  # (label, old, new, higher_is_better)

    # Unit tests
    old_ut = baseline.get("unit_tests", {})
    new_ut = current.get("unit_tests", {})
    if old_ut and new_ut:
        metrics.append(("unit tests passed",
                         old_ut.get("passed", 0), new_ut.get("passed", 0), True))
        metrics.append(("unit tests failed",
                         old_ut.get("failed", 0), new_ut.get("failed", 0), False))

    # Retrieval harness
    old_rh = baseline.get("retrieval_harness", {})
    new_rh = current.get("retrieval_harness", {})
    if old_rh and new_rh:
        metrics.append(("retrieval pass_rate",
                         old_rh.get("pass_rate", 0), new_rh.get("pass_rate", 0), True))

    # Acceptance battery
    old_ab = (baseline.get("acceptance_battery") or {}).get("summary", {})
    new_ab = (current.get("acceptance_battery") or {}).get("summary", {})
    if old_ab and new_ab:
        metrics.append(("acceptance pass_rate",
                         old_ab.get("pass_rate", 0), new_ab.get("pass_rate", 0), True))
        for cat in set(list(old_ab.get("by_category", {}).keys()) + list(new_ab.get("by_category", {}).keys())):
            old_cat = old_ab.get("by_category", {}).get(cat, {})
            new_cat = new_ab.get("by_category", {}).get(cat, {})
            if old_cat and new_cat and old_cat["count"] == new_cat["count"]:
                old_r = old_cat["passed"] / old_cat["count"]
                new_r = new_cat["passed"] / new_cat["count"]
                metrics.append((f"  {cat}", old_r, new_r, True))

    regressions = improvements = unchanged = 0
    for label, old_v, new_v, hib in metrics:
        delta = new_v - old_v
        if abs(delta) < 0.001:
            status = f"{_BLUE}~{_RESET}"
            unchanged += 1
        elif (delta > 0) == hib:
            status = f"{_GREEN}▲ IMPROVEMENT{_RESET}"
            improvements += 1
        else:
            status = f"{_RED}▼ REGRESSION{_RESET}"
            regressions += 1
        ds = _delta_str(old_v, new_v, hib)
        print(f"  {label:45s}  {old_v:.3f} → {new_v:.3f}  {ds}  {status}")

    print()
    if regressions:
        print(f"  {_RED}{_BOLD}{regressions} REGRESSION(s) detected.{_RESET}")
    if improvements:
        print(f"  {_GREEN}{improvements} improvement(s).{_RESET}")
    if not regressions and not improvements:
        print(f"  {_BLUE}No significant changes vs. baseline.{_RESET}")

    return {
        "regressions": regressions,
        "improvements": improvements,
        "unchanged": unchanged,
        "baseline_run_at": baseline.get("run_at"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the RAG pipeline — regression and improvement tracking."
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Skip LLM-dependent stages (acceptance battery). Uses hashing embed backend.",
    )
    parser.add_argument(
        "--save-baseline", action="store_true",
        help="Save the results of this run as the new comparison baseline.",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="Comma-separated acceptance battery case IDs to run (e.g. core_ss_01,internet_01).",
    )
    parser.add_argument("--db",           type=str, default=DEFAULT_DB_DSN)
    parser.add_argument("--embed-backend", type=str, default=None,
                        help="Embedding backend. Defaults to 'hashing' in --fast mode, else config default.")
    parser.add_argument("--embed-model",  type=str, default=DEFAULT_EMBED_MODEL_NAME)
    parser.add_argument("--llm-config",   type=str, default="configs/llm.yaml")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    embed_backend = args.embed_backend
    if embed_backend is None:
        embed_backend = "hashing" if args.fast else DEFAULT_EMBED_BACKEND

    only_ids = [s.strip() for s in args.only.split(",") if s.strip()] if args.only else None

    print(f"{_BOLD}RAG Validation Suite{_RESET}  —  {_now()}")
    if args.fast:
        print(f"  {_YELLOW}Fast mode: skipping acceptance battery (no LLM calls).{_RESET}")
    print(f"  DB:            {args.db}")
    print(f"  Embed backend: {embed_backend}")
    if not args.fast:
        print(f"  Embed model:   {args.embed_model}")

    report: Dict[str, Any] = {"run_at": _now(), "fast_mode": args.fast}

    # ── 1. Unit tests ──────────────────────────────────────────────────────────
    report["unit_tests"] = run_unit_tests()

    # ── 2. DB health ───────────────────────────────────────────────────────────
    report["db_health"] = run_db_health(args.db)

    # ── 3. Retrieval harness ───────────────────────────────────────────────────
    if report["db_health"]["healthy"]:
        report["retrieval_harness"] = run_retrieval_harness(
            db_dsn=args.db,
            embed_backend=embed_backend,
            embed_model=args.embed_model,
        )
    else:
        _header("3 / Retrieval Harness")
        print(f"  {_YELLOW}Skipped — DB is not healthy.{_RESET}")
        report["retrieval_harness"] = {"skipped": True, "reason": "db_not_healthy"}

    # ── 4. Acceptance battery (full mode only) ─────────────────────────────────
    if not args.fast:
        if report["db_health"]["healthy"]:
            report["acceptance_battery"] = run_acceptance_battery(
                db_dsn=args.db,
                embed_backend=embed_backend,
                embed_model=args.embed_model,
                llm_config=args.llm_config,
                only_ids=only_ids,
            )
        else:
            _header("4 / Acceptance Battery")
            print(f"  {_YELLOW}Skipped — DB is not healthy.{_RESET}")
            report["acceptance_battery"] = {"skipped": True, "reason": "db_not_healthy"}
    else:
        report["acceptance_battery"] = {"skipped": True, "reason": "fast_mode"}

    # ── 5. Baseline comparison ─────────────────────────────────────────────────
    delta = compare_to_baseline(report)
    if delta:
        report["baseline_delta"] = delta

    # ── Save timestamped report ────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    report_path = REPORT_DIR / f"validation_{ts}.json"
    _write_json(report_path, report)
    print(f"\n  Report saved: {report_path}")

    if args.save_baseline:
        _write_json(BASELINE_PATH, report)
        print(f"  {_GREEN}Baseline updated: {BASELINE_PATH}{_RESET}")

    # ── Final summary ──────────────────────────────────────────────────────────
    _header("Summary")

    unit_ok   = report["unit_tests"]["suite_passed"]
    db_ok     = report["db_health"]["healthy"]
    rh        = report.get("retrieval_harness") or {}
    rh_ok     = not rh.get("skipped") and rh.get("pass_rate", 0) >= 0.8
    ab        = (report.get("acceptance_battery") or {}).get("summary") or {}
    ab_ok     = not (report.get("acceptance_battery") or {}).get("skipped") and ab.get("pass_rate", 0) >= 0.7

    _result_line("Unit tests",           unit_ok)
    _result_line("DB health",            db_ok)
    if not rh.get("skipped"):
        _result_line(
            f"Retrieval harness  ({rh.get('passed', '?')}/{rh.get('total', '?')} @ {rh.get('pass_rate', 0)*100:.1f}%)",
            rh_ok,
        )
    if not (report.get("acceptance_battery") or {}).get("skipped"):
        _result_line(
            f"Acceptance battery ({ab.get('passed_cases', '?')}/{ab.get('total_cases', '?')} @ {ab.get('pass_rate', 0)*100:.1f}%)",
            ab_ok,
        )

    regressions = (delta or {}).get("regressions", 0)
    if regressions:
        print(f"\n  {_RED}{_BOLD}⚠  {regressions} REGRESSION(s) vs. baseline. Review before merging changes.{_RESET}")

    all_ok = unit_ok and db_ok and (rh.get("skipped") or rh_ok)
    if not args.fast:
        all_ok = all_ok and (ab.get("skipped") or ab_ok)

    overall = f"{_GREEN}PASS{_RESET}" if all_ok else f"{_RED}FAIL{_RESET}"
    print(f"\n  Overall: {_BOLD}{overall}{_RESET}\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
