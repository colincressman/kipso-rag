#!/usr/bin/env python3
"""
RAG Diagnostics Script
======================

Runs the full test suite and collects RAG-specific statistics/diagnostics,
then writes a Markdown report to data/diagnostics/.

Usage:
    python scripts/rag_diagnostics.py
    python scripts/rag_diagnostics.py --backend ollama --model qwen3-embedding
    python scripts/rag_diagnostics.py --skip-tests
    python scripts/rag_diagnostics.py --out-dir /path/to/reports

RAG metrics collected:
  - Test suite pass/fail/skip summary, broken down by file
  - Corpus health: embedding coverage, zero-length chunks, orphaned chunks
  - Chunk statistics: count by structural role/source type, size distribution
  - Retrieval benchmark: top score, score gap, latency, role accuracy
  - Grounding evaluation: pass rate by confidence band (requires QA export)
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from utils.runtime_defaults import DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DB = "postgresql://postgres:postgres@localhost/rag"
DEFAULT_OUT = PROJECT_ROOT / "data" / "diagnostics"
DEFAULT_QA = PROJECT_ROOT / "data" / "qa" / "question_battery_answers.json"

# ---------------------------------------------------------------------------
# Benchmark query set — covers query categories the retrieval stack handles
# ---------------------------------------------------------------------------
BENCHMARK_QUERIES: List[Tuple[str, str]] = [
    ("metadata",    "What is the ISBN of the book?"),
    ("metadata",    "Who are the authors?"),
    ("metadata",    "What year was the book published?"),
    ("metadata",    "Who published this book?"),
    ("fact",        "What is arbitrage?"),
    ("fact",        "What is CAPM?"),
    ("fact",        "Explain market-making mechanics."),
    ("fact",        "What is company valuation?"),
    ("overview",    "What topics are covered in this book?"),
    ("overview",    "What chapters does this book have?"),
    ("formula",     "What is the CAPM formula?"),
    ("off_topic",   "What does the book say about cryptocurrency trading?"),
    ("off_topic",   "Explain neural networks in hedge funds."),
]


# ===========================================================================
# Helpers
# ===========================================================================

def _percentile(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    idx = p / 100.0 * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def _distribution(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p25": 0.0, "median": 0.0,
                "mean": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0, "stdev": 0.0}
    s = sorted(values)
    n = len(s)
    mean = sum(s) / n
    variance = sum((x - mean) ** 2 for x in s) / n
    return {
        "count":  n,
        "min":    round(s[0], 4),
        "p25":    round(_percentile(s, 25), 4),
        "median": round(_percentile(s, 50), 4),
        "mean":   round(mean, 4),
        "p75":    round(_percentile(s, 75), 4),
        "p95":    round(_percentile(s, 95), 4),
        "max":    round(s[-1], 4),
        "stdev":  round(math.sqrt(variance), 4),
    }


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render a GitHub-flavoured Markdown table."""
    if not rows:
        return ""
    col_w = [
        max(len(h), max(len(str(r[i])) for r in rows))
        for i, h in enumerate(headers)
    ]
    header_row = "| " + " | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)) + " |"
    sep_row    = "| " + " | ".join("-" * w for w in col_w) + " |"
    data_rows  = "\n".join(
        "| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(headers))) + " |"
        for r in rows
    )
    return header_row + "\n" + sep_row + "\n" + data_rows


# ===========================================================================
# 1. Test suite runner
# ===========================================================================

def run_tests(extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run pytest -v and parse per-test and summary results."""
    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--no-header", "-q"]
    if extra_args:
        cmd.extend(extra_args)

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = round(time.perf_counter() - t0, 2)

    output = proc.stdout + proc.stderr

    passed = failed = skipped = errors = 0
    tests: List[Dict[str, str]] = []
    failures: List[str] = []

    # Parse per-test lines: "tests/foo.py::test_bar PASSED" or "FAILED" or "SKIPPED"
    for line in output.splitlines():
        m = re.match(r"^(tests[/\\]\S+)\s+(PASSED|FAILED|SKIPPED|ERROR)\s*(.*)$", line)
        if m:
            name, status, note = m.group(1), m.group(2), m.group(3).strip()
            # Strip pytest progress percentage from note
            note = re.sub(r"\[\s*\d+%\].*", "", note).strip()
            tests.append({"name": name, "status": status, "note": note})
            if status == "PASSED":
                passed += 1
            elif status == "FAILED":
                failed += 1
                failures.append(name)
            elif status == "SKIPPED":
                skipped += 1
            elif status == "ERROR":
                errors += 1

    # Fallback: parse summary line when -v is suppressed or test lines were missed
    if not tests:
        m2 = re.search(
            r"(\d+) passed(?:.*?(\d+) failed)?(?:.*?(\d+) skipped)?(?:.*?(\d+) error)?",
            output,
        )
        if m2:
            passed  = int(m2.group(1) or 0)
            failed  = int(m2.group(2) or 0)
            skipped = int(m2.group(3) or 0)
            errors  = int(m2.group(4) or 0)

    return {
        "passed":          passed,
        "failed":          failed,
        "skipped":         skipped,
        "errors":          errors,
        "elapsed_seconds": elapsed,
        "failures":        failures,
        "tests":           tests,
        "return_code":     proc.returncode,
        "raw_tail":        output[-3000:],  # last 3 KB for failure context
    }


# ===========================================================================
# 2. Database / corpus statistics
# ===========================================================================

def collect_db_stats(db_path: str) -> Dict[str, Any]:
    """Query the PostgreSQL DB and collect corpus health + shape statistics."""
    try:
        conn = psycopg.connect(db_path, row_factory=dict_row)
    except Exception as exc:
        return {"error": f"Cannot connect to DB: {exc}"}

    stats: Dict[str, Any] = {}

    # --- Documents ---
    doc_rows = conn.execute(
        "SELECT source_type, COUNT(*) AS cnt FROM documents GROUP BY source_type"
    ).fetchall()
    stats["documents_by_source_type"] = {r["source_type"]: r["cnt"] for r in doc_rows}
    stats["document_total"] = sum(stats["documents_by_source_type"].values())

    # --- Chunks ---
    stats["chunk_total"] = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]

    role_rows = conn.execute(
        "SELECT structural_role, COUNT(*) AS cnt FROM chunks GROUP BY structural_role"
    ).fetchall()
    stats["chunks_by_structural_role"] = {r["structural_role"]: r["cnt"] for r in role_rows}

    src_rows = conn.execute(
        "SELECT source_type, COUNT(*) AS cnt FROM chunks GROUP BY source_type"
    ).fetchall()
    stats["chunks_by_source_type"] = {r["source_type"]: r["cnt"] for r in src_rows}

    # --- Embedding coverage ---
    embedded = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
    ).fetchone()["n"]
    missing = stats["chunk_total"] - embedded
    coverage = round(100.0 * embedded / max(1, stats["chunk_total"]), 2)
    stats["embedding_coverage"] = {"embedded": embedded, "missing": missing, "coverage_pct": coverage}

    # --- Chunk size distribution ---
    tok_vals = [
        float(r["token_count_est"])
        for r in conn.execute(
            "SELECT token_count_est FROM chunks WHERE token_count_est IS NOT NULL AND token_count_est > 0"
        ).fetchall()
    ]
    stats["chunk_token_distribution"] = _distribution(tok_vals)

    # --- Health: zero-length and orphaned chunks ---
    stats["zero_length_chunks"] = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE text IS NULL OR LENGTH(TRIM(text)) = 0"
    ).fetchone()["n"]
    stats["orphaned_chunks"] = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)"
    ).fetchone()["n"]

    # --- Artifacts ---
    try:
        stats["artifact_total"] = conn.execute("SELECT COUNT(*) AS n FROM artifacts").fetchone()["n"]
    except Exception:
        stats["artifact_total"] = None

    # --- Embedding dimension from pg catalog ---
    try:
        row = conn.execute(
            "SELECT atttypmod FROM pg_attribute JOIN pg_class ON attrelid=pg_class.oid "
            "WHERE relname='chunks' AND attname='embedding'"
        ).fetchone()
        stats["embedding_dimension"] = row["atttypmod"] if row else None
    except Exception:
        stats["embedding_dimension"] = None

    conn.close()
    return stats


# ===========================================================================
# 3. Retrieval benchmark
# ===========================================================================

def run_retrieval_benchmark(
    db_path: str,
    *,
    embed_backend: str = "ollama",
    embed_model_name: str = "qwen3-embedding",
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Run the benchmark query set and collect retrieval quality indicators.
    """
    try:
        from retrieval.query import retrieve_as_dict
    except ImportError as exc:
        return {"error": f"Cannot import retrieval.query: {exc}"}

    all_top_scores: List[float] = []
    all_gaps: List[float] = []
    all_latencies: List[float] = []
    top_role_hits: Counter = Counter()
    by_category: Dict[str, List[float]] = defaultdict(list)
    role_correct = role_total = 0

    per_query: List[Dict[str, Any]] = []

    for category, query in BENCHMARK_QUERIES:
        t0 = time.perf_counter()
        try:
            result = retrieve_as_dict(
                query,
                db_dsn=str(db_path),
                top_k=top_k,
                embed_backend=embed_backend,
                embed_model_name=embed_model_name,
            )
        except Exception as exc:
            per_query.append({"category": category, "query": query, "error": str(exc)})
            continue

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        all_latencies.append(latency_ms)

        hits = result.get("hits", [])
        if hits:
            top = hits[0]
            top_score = float(top.get("score", 0.0))
            top_role  = str(top.get("structural_role", "unknown"))
            gap = top_score - float(hits[1].get("score", 0.0)) if len(hits) > 1 else 1.0
            snippet = (top.get("text") or "")[:80].replace("\n", " ")
        else:
            top_score, top_role, gap, snippet = 0.0, "none", 0.0, ""

        all_top_scores.append(top_score)
        all_gaps.append(gap)
        by_category[category].append(top_score)
        top_role_hits[top_role] += 1

        # Role accuracy: metadata queries should surface role=='metadata'
        if category == "metadata":
            role_total += 1
            if top_role == "metadata":
                role_correct += 1

        per_query.append({
            "category":       category,
            "query":          query,
            "top_score":      round(top_score, 4),
            "score_gap":      round(gap, 4),
            "top_role":       top_role,
            "hit_count":      len(hits),
            "latency_ms":     latency_ms,
            "top_snippet":    snippet,
        })

    metadata_role_accuracy = (
        round(100 * role_correct / role_total, 1) if role_total else None
    )

    return {
        "queries_run":              len([q for q in per_query if "error" not in q]),
        "top_score_distribution":   _distribution(all_top_scores),
        "score_gap_distribution":   _distribution(all_gaps),
        "latency_ms_distribution":  _distribution(all_latencies),
        "top_score_by_category":    {k: _distribution(v) for k, v in by_category.items()},
        "top_role_distribution":    dict(top_role_hits),
        "metadata_role_accuracy_pct": metadata_role_accuracy,
        "per_query":                per_query,
    }


# ===========================================================================
# 4. Grounding evaluation wrapper
# ===========================================================================

def run_grounding_eval(qa_path: Path, db_path: str) -> Optional[Dict[str, Any]]:
    """Wrap evaluate_grounding logic to produce a summary."""
    if not qa_path.exists():
        return None

    # Import evaluate_grounding from the scripts directory
    scripts_dir = str(PROJECT_ROOT / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from evaluate_grounding import evaluate_grounding
        return evaluate_grounding(
            qa_path=qa_path,
            db_dsn=db_path,
            medium_threshold=0.55,
            high_threshold=0.70,
            sentence_overlap_threshold=0.35,
        )
    except Exception as exc:
        return {"error": str(exc)}


# ===========================================================================
# 5. Markdown report builder
# ===========================================================================

_METRIC_LEGEND = """
### Metric Legend

| Metric | What it measures |
|--------|-----------------|
| **top_score** | Cosine similarity + structural-role adjustment for the best-matching chunk. Higher is better. Scores >0.60 are strong; <0.30 suggest poor coverage or embedding mismatch. |
| **score_gap** | Rank-1 score minus Rank-2 score. A larger gap means the retriever is *certain* about the top result. Gaps <0.03 indicate low confidence. |
| **metadata_role_accuracy** | For queries known to need publication info (ISBN, author, year), the % of times the rank-1 hit has `structural_role=metadata`. Should approach 100%. |
| **embedding_coverage** | % of chunks with a stored embedding vector. Anything <100% means some chunks are invisible to vector search. |
| **chunk_token_distribution** | Size of chunks in estimated tokens. Ideal range: 100–400 tokens. Very small (<50) chunks lack context; very large (>800) reduce precision. |
| **grounding pass rate** | % of LLM answers that contain no unsupported years, no malformed citations, and ≤1 weakly-supported sentence. Requires a QA export file. |
""".strip()


def build_markdown_report(
    *,
    timestamp: str,
    embed_backend: str,
    db_path: str,
    test_results: Dict[str, Any],
    db_stats: Dict[str, Any],
    benchmark: Dict[str, Any],
    grounding: Optional[Dict[str, Any]],
) -> str:
    lines: List[str] = []

    def h(level: int, text: str) -> None:
        lines.append(f"\n{'#' * level} {text}\n")

    def row(*cells: str) -> None:
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("# RAG Diagnostics Report")
    lines.append(f"\n**Generated:** {timestamp}  ")
    lines.append(f"**Embed backend:** `{embed_backend}`  ")
    lines.append(f"**DB:** `{db_path}`\n")

    # ── 1. Test Suite ───────────────────────────────────────────────────────
    h(2, "1. Test Suite")

    if "error" in test_results:
        lines.append(f"> ⚠️ Error running tests: {test_results['error']}\n")
    else:
        tr = test_results
        ok = tr["failed"] == 0 and tr["errors"] == 0
        verdict = "✅ ALL PASSING" if ok else "❌ FAILURES PRESENT"
        lines.append(
            f"{verdict} — "
            f"**{tr['passed']} passed** | {tr['failed']} failed | "
            f"{tr['skipped']} skipped | {tr['errors']} errors | "
            f"elapsed: {tr['elapsed_seconds']}s\n"
        )

        if tr["failures"]:
            h(3, "Failing Tests")
            for name in tr["failures"]:
                lines.append(f"- `{name}`")
            lines.append("")

        if tr["tests"]:
            h(3, "Results by File")
            by_file: Dict[str, Dict[str, int]] = defaultdict(lambda: {"p": 0, "f": 0, "s": 0, "e": 0})
            for t in tr["tests"]:
                file_key = t["name"].split("::")[0]
                if t["status"] == "PASSED":
                    by_file[file_key]["p"] += 1
                elif t["status"] == "FAILED":
                    by_file[file_key]["f"] += 1
                elif t["status"] == "SKIPPED":
                    by_file[file_key]["s"] += 1
                else:
                    by_file[file_key]["e"] += 1

            table_rows = [
                [f, str(v["p"]), str(v["f"]), str(v["s"]), str(v["e"])]
                for f, v in sorted(by_file.items())
            ]
            lines.append(_md_table(["Test File", "Passed", "Failed", "Skipped", "Error"], table_rows))
            lines.append("")

        if not ok and tr["raw_tail"]:
            h(3, "Output Tail (last 50 lines)")
            lines.append("```")
            tail_lines = tr["raw_tail"].splitlines()[-50:]
            lines.extend(tail_lines)
            lines.append("```\n")

    # ── 2. Database / Corpus Stats ──────────────────────────────────────────
    h(2, "2. Database & Corpus Statistics")

    if "error" in db_stats:
        lines.append(f"> ⚠️ {db_stats['error']}\n")
    else:
        lines.append(
            f"**DB size:** {db_stats['db_file_size_mb']} MB  \n"
            f"**Documents:** {db_stats['document_total']}  \n"
            f"**Chunks total:** {db_stats['chunk_total']}  \n"
            f"**Embedding dimension:** {db_stats.get('embedding_dimension', 'N/A')}  \n"
        )

        h(3, "Documents by Source Type")
        doc_rows = [[k, str(v)] for k, v in sorted(db_stats["documents_by_source_type"].items())]
        if doc_rows:
            lines.append(_md_table(["Source Type", "Count"], doc_rows))
            lines.append("")

        h(3, "Chunks by Structural Role")
        role_rows = sorted(db_stats["chunks_by_structural_role"].items(), key=lambda x: -x[1])
        if role_rows:
            lines.append(_md_table(["Role", "Count"], [[k, str(v)] for k, v in role_rows]))
            lines.append("")

        h(3, "Chunks by Source Type")
        src_rows = sorted(db_stats["chunks_by_source_type"].items(), key=lambda x: -x[1])
        if src_rows:
            lines.append(_md_table(["Source Type", "Count"], [[k, str(v)] for k, v in src_rows]))
            lines.append("")

        h(3, "Embedding Coverage")
        ec = db_stats["embedding_coverage"]
        cov_icon = "✅" if ec["coverage_pct"] >= 95 else "⚠️"
        lines.append(
            f"{cov_icon} **{ec['coverage_pct']}%** embedded "
            f"({ec['embedded']} / {ec['embedded'] + ec['missing']} chunks)  \n"
            f"Missing embeddings: {ec['missing']}\n"
        )

        h(3, "Chunk Size Distribution (estimated tokens)")
        ctd = db_stats["chunk_token_distribution"]
        if ctd["count"] > 0:
            size_rows = [[k, str(v)] for k, v in ctd.items()]
            lines.append(_md_table(["Statistic", "Value"], size_rows))
            lines.append("")
            p25, p75 = ctd["p25"], ctd["p75"]
            if ctd["p95"] > 800:
                lines.append("> ⚠️ p95 token count is high — consider smaller chunk sizes for better precision.\n")
            elif ctd["median"] < 80:
                lines.append("> ⚠️ Median chunk size is very small — chunks may lack sufficient context.\n")
            else:
                lines.append("> ✅ Chunk sizes look healthy.\n")

        h(3, "Health Flags")
        flags = []
        if db_stats["zero_length_chunks"] > 0:
            flags.append(f"⚠️ {db_stats['zero_length_chunks']} zero-length chunks")
        if db_stats["orphaned_chunks"] > 0:
            flags.append(f"⚠️ {db_stats['orphaned_chunks']} orphaned chunks (doc_id not in documents table)")
        if not flags:
            flags.append("✅ No data integrity issues")
        for f in flags:
            lines.append(f"- {f}")
        lines.append("")

    # ── 3. Retrieval Benchmark ───────────────────────────────────────────────
    h(2, "3. Retrieval Benchmark")

    if not benchmark:
        lines.append("> Benchmark was skipped.\n")
    elif "error" in benchmark:
        lines.append(f"> ⚠️ {benchmark['error']}\n")
    else:
        lines.append(f"**Queries run:** {benchmark['queries_run']}  \n")

        ara = benchmark.get("metadata_role_accuracy_pct")
        if ara is not None:
            ara_icon = "✅" if ara >= 75 else "⚠️"
            lines.append(f"**Metadata role accuracy:** {ara_icon} {ara}%  \n")

        h(3, "Top Score Distribution (all queries)")
        tsd = benchmark["top_score_distribution"]
        if tsd["count"] > 0:
            lines.append(_md_table(["Statistic", "Value"], [[k, str(v)] for k, v in tsd.items()]))
            lines.append("")
            mean_score = tsd["mean"]
            if mean_score >= 0.50:
                lines.append("> ✅ Mean top score is strong — retrieval is finding relevant chunks.\n")
            elif mean_score >= 0.30:
                lines.append("> ⚠️ Mean top score is moderate. Consider re-ingesting with a better embedding model.\n")
            else:
                lines.append("> ❌ Mean top score is low. Embedding model may be mismatched with query style.\n")

        h(3, "Score Gap Distribution (rank-1 minus rank-2)")
        sgd = benchmark["score_gap_distribution"]
        if sgd["count"] > 0:
            lines.append(_md_table(["Statistic", "Value"], [[k, str(v)] for k, v in sgd.items()]))
            lines.append("")
            if sgd["median"] < 0.03:
                lines.append("> ⚠️ Low median score gap — retriever is often uncertain. Low-confidence answers may be frequent.\n")

        h(3, "Query Latency (ms)")
        lat = benchmark["latency_ms_distribution"]
        if lat["count"] > 0:
            lines.append(_md_table(["Statistic", "Value"], [[k, str(v)] for k, v in lat.items()]))
            lines.append("")

        h(3, "Top Score by Query Category")
        tsbc = benchmark.get("top_score_by_category", {})
        if tsbc:
            cat_rows = [
                [cat, str(s["count"]), str(s["mean"]), str(s["min"]), str(s["max"]), str(s["stdev"])]
                for cat, s in sorted(tsbc.items())
            ]
            lines.append(_md_table(["Category", "N", "Mean", "Min", "Max", "StDev"], cat_rows))
            lines.append("")

        h(3, "Top Hit Structural Role Distribution")
        trd = benchmark.get("top_role_distribution", {})
        if trd:
            role_total = sum(trd.values())
            role_rows = sorted(trd.items(), key=lambda x: -x[1])
            role_table = [
                [role, str(cnt), f"{100*cnt/role_total:.1f}%"]
                for role, cnt in role_rows
            ]
            lines.append(_md_table(["Role", "Times #1 Hit", "% of Queries"], role_table))
            lines.append("")

        h(3, "Per-Query Results")
        pq = benchmark["per_query"]
        if pq:
            pq_rows = []
            for q in pq:
                if "error" in q:
                    pq_rows.append([q["category"], q["query"][:45], "ERROR", "-", "-", q["error"][:35]])
                else:
                    pq_rows.append([
                        q["category"],
                        q["query"][:45],
                        str(q["top_score"]),
                        str(q["score_gap"]),
                        q["top_role"],
                        f"{q['latency_ms']:.0f}ms",
                    ])
            lines.append(_md_table(
                ["Category", "Query", "Top Score", "Gap", "Top Role", "Latency"],
                pq_rows,
            ))
            lines.append("")

    # ── 4. Grounding Evaluation ──────────────────────────────────────────────
    h(2, "4. Grounding Evaluation")
    lines.append(
        "_Grounding eval requires a QA export file generated by "
        "`python scripts/export_question_answers.py` (needs Ollama LLM running)._\n"
    )

    if grounding is None:
        lines.append("> No QA export found — grounding eval skipped.\n")
    elif "error" in grounding:
        lines.append(f"> ⚠️ {grounding['error']}\n")
    else:
        summary = grounding.get("summary_by_band", {})
        if summary:
            band_rows = [
                [
                    band,
                    str(s["count"]),
                    f"{s['pass_rate'] * 100:.1f}%",
                    str(s.get("avg_weak_sentence_count", 0)),
                    str(s.get("avg_unsupported_year_count", 0)),
                    str(s.get("avg_unsupported_entity_count", 0)),
                ]
                for band, s in sorted(summary.items())
            ]
            lines.append(_md_table(
                ["Band", "Count", "Pass Rate", "Avg Weak Sentences", "Avg Unsupported Years", "Avg Unsupported Entities"],
                band_rows,
            ))
            lines.append("")

            total     = sum(s["count"] for s in summary.values())
            total_p   = sum(s["pass_count"] for s in summary.values())
            if total:
                pct = 100 * total_p / total
                icon = "✅" if pct >= 85 else "⚠️" if pct >= 60 else "❌"
                lines.append(f"{icon} **Overall pass rate: {total_p}/{total} ({pct:.1f}%)**\n")

    # ── 5. Health Summary ────────────────────────────────────────────────────
    h(2, "5. Overall Health Summary")

    issues: List[str] = []
    ok_items: List[str] = []

    if "error" not in test_results:
        if test_results.get("failed", 0) == 0 and test_results.get("errors", 0) == 0:
            ok_items.append(f"Tests: {test_results['passed']} passing, {test_results['skipped']} skipped")
        else:
            issues.append(
                f"❌ Tests: {test_results['failed']} failing, {test_results['errors']} errors"
            )

    if "error" not in db_stats:
        ec = db_stats["embedding_coverage"]
        if ec["coverage_pct"] < 95:
            issues.append(f"⚠️ Embedding coverage {ec['coverage_pct']}% ({ec['missing']} chunks missing)")
        else:
            ok_items.append(f"Embedding coverage: {ec['coverage_pct']}%")

        if db_stats["zero_length_chunks"] > 0:
            issues.append(f"⚠️ {db_stats['zero_length_chunks']} zero-length chunks")
        if db_stats["orphaned_chunks"] > 0:
            issues.append(f"⚠️ {db_stats['orphaned_chunks']} orphaned chunks")

    if "error" not in benchmark and benchmark:
        ara = benchmark.get("metadata_role_accuracy_pct")
        if ara is not None and ara < 75:
            issues.append(f"⚠️ Metadata role accuracy {ara}% — structural scoring may need tuning")
        mean_top = benchmark.get("top_score_distribution", {}).get("mean", 0)
        if mean_top < 0.30:
            issues.append(f"❌ Mean retrieval score {mean_top:.3f} — possible embedding mismatch")
        elif mean_top >= 0.50:
            ok_items.append(f"Mean retrieval score: {mean_top:.3f}")

    if grounding and "error" not in grounding:
        summary = grounding.get("summary_by_band", {})
        total   = sum(s["count"] for s in summary.values())
        total_p = sum(s["pass_count"] for s in summary.values())
        if total:
            pct = 100 * total_p / total
            if pct < 60:
                issues.append(f"❌ Grounding pass rate {pct:.1f}%")
            elif pct < 85:
                issues.append(f"⚠️ Grounding pass rate {pct:.1f}%")
            else:
                ok_items.append(f"Grounding pass rate: {pct:.1f}%")

    if issues:
        lines.append("**Issues detected:**\n")
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")

    if ok_items:
        lines.append("**Healthy:**\n")
        for item in ok_items:
            lines.append(f"- ✅ {item}")
        lines.append("")

    if not issues and not ok_items:
        lines.append("_No data to summarise — run without `--skip-*` flags._\n")

    # ── 6. Metric Legend ─────────────────────────────────────────────────────
    h(2, "6. Metric Legend")
    lines.append(_METRIC_LEGEND)
    lines.append("")

    return "\n".join(lines)


# ===========================================================================
# CLI entry point
# ===========================================================================

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="RAG diagnostics: run tests + collect stats + write Markdown report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--db",           default=str(DEFAULT_DB),  help="PostgreSQL DSN")
    parser.add_argument("--out-dir",      default=str(DEFAULT_OUT), help="Directory to write report into")
    parser.add_argument("--backend",      default=DEFAULT_EMBED_BACKEND,          help="Embed backend for benchmark (ollama/sentence-transformers)")
    parser.add_argument("--model",        default=DEFAULT_EMBED_MODEL_NAME,  help="Embedding model name")
    parser.add_argument("--qa-path",      default=str(DEFAULT_QA),  help="QA export JSON for grounding eval")
    parser.add_argument("--skip-tests",     action="store_true", help="Skip pytest run")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip retrieval benchmark")
    parser.add_argument("--skip-grounding", action="store_true", help="Skip grounding evaluation")
    args = parser.parse_args()

    db_path  = args.db
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    now      = datetime.now(timezone.utc)
    ts       = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    file_ts  = now.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"diagnostics_{file_ts}.md"

    width = 62
    print(f"\n{'=' * width}")
    print(f"  RAG Diagnostics  —  {ts}")
    print(f"{'=' * width}")

    # 1. Tests
    if args.skip_tests:
        print("\n[SKIP] Test suite")
        test_results: Dict[str, Any] = {
            "passed": 0, "failed": 0, "skipped": 0, "errors": 0,
            "elapsed_seconds": 0, "failures": [], "tests": [], "raw_tail": "",
        }
    else:
        print("\n[1/4] Running test suite ...")
        test_results = run_tests()
        ok = test_results["failed"] == 0 and test_results["errors"] == 0
        verdict = "✅ PASS" if ok else "❌ FAIL"
        print(
            f"      {verdict}  "
            f"{test_results['passed']}p / {test_results['failed']}f / "
            f"{test_results['skipped']}s  ({test_results['elapsed_seconds']}s)"
        )

    # 2. DB stats
    print("\n[2/4] Collecting DB stats ...")
    db_stats = collect_db_stats(db_path)
    if "error" not in db_stats:
        ec = db_stats["embedding_coverage"]
        print(
            f"      docs={db_stats['document_total']}  "
            f"chunks={db_stats['chunk_total']}  "
            f"embedded={ec['coverage_pct']}%"
        )
    else:
        print(f"      ⚠️  {db_stats['error']}")

    # 3. Retrieval benchmark
    if args.skip_benchmark:
        print("\n[SKIP] Retrieval benchmark")
        benchmark: Dict[str, Any] = {}
    else:
        print(f"\n[3/4] Running retrieval benchmark "
              f"({len(BENCHMARK_QUERIES)} queries, backend={args.backend}) ...")
        benchmark = run_retrieval_benchmark(
            db_path,
            embed_backend=args.backend,
            embed_model_name=args.model,
        )
        if "error" not in benchmark:
            tsd = benchmark["top_score_distribution"]
            lat = benchmark["latency_ms_distribution"]
            ara = benchmark.get("metadata_role_accuracy_pct")
            print(
                f"      mean_top={tsd.get('mean', 0)}  "
                f"min={tsd.get('min', 0)}  max={tsd.get('max', 0)}  "
                f"avg_latency={lat.get('mean', 0):.1f}ms  "
                f"meta_role_acc={ara}%"
            )
        else:
            print(f"      ⚠️  {benchmark['error']}")

    # 4. Grounding eval
    qa_path = Path(args.qa_path)
    if args.skip_grounding:
        print("\n[SKIP] Grounding evaluation")
        grounding: Optional[Dict[str, Any]] = None
    elif not qa_path.exists():
        print(f"\n[4/4] Grounding eval skipped (no QA file at {qa_path})")
        grounding = None
    else:
        print(f"\n[4/4] Running grounding evaluation from {qa_path.name} ...")
        grounding = run_grounding_eval(qa_path, db_path)
        if grounding and "error" not in grounding:
            summary = grounding.get("summary_by_band", {})
            total   = sum(s["count"] for s in summary.values())
            total_p = sum(s["pass_count"] for s in summary.values())
            pct = 100 * total_p / total if total else 0
            print(f"      pass_rate={total_p}/{total} ({pct:.1f}%)")
        elif grounding:
            print(f"      ⚠️  {grounding.get('error', 'unknown error')}")

    # Build report
    print(f"\nBuilding report ...")
    report = build_markdown_report(
        timestamp=ts,
        embed_backend=args.backend,
        db_path=db_path,
        test_results=test_results,
        db_stats=db_stats,
        benchmark=benchmark,
        grounding=grounding,
    )
    out_path.write_text(report, encoding="utf-8")

    # Always overwrite the "latest" convenience copy
    latest_path = out_dir / "diagnostics_latest.md"
    latest_path.write_text(report, encoding="utf-8")

    print(f"\n{'=' * width}")
    print(f"  Report written:")
    print(f"    {out_path}")
    print(f"    {latest_path}  (symlink copy)")
    print(f"{'=' * width}\n")


if __name__ == "__main__":
    main()
