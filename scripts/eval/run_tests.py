"""
Encoder comparison battery for the RAG pipeline.

Design goals
------------
1. Ask fair questions that match the system's actual capabilities.
2. Avoid known trap / trick questions that are mainly stress tests.
3. Run every question twice:
   - bi_encoder baseline (current reranker only)
   - cross_encoder reranker enabled
4. Save output incrementally after every run so the JSON file updates live.

Usage
-----
    python scripts/run_tests.py
    python scripts/run_tests.py --only ss_01,section_lookup
    python scripts/run_tests.py --out data/diagnostics/encoder_comparison_latest.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
CLI = str(PROJECT_ROOT / "scripts" / "query_cli.py")
DEFAULT_OUT = PROJECT_ROOT / "data" / "diagnostics" / "encoder_comparison_latest.json"

ENCODER_MODES = [
    {
        "id": "bi_encoder",
        "label": "Bi-encoder baseline",
        "cli_flag": "--no-cross-encoder",
    },
    {
        "id": "cross_encoder",
        "label": "Cross-encoder reranker",
        "cli_flag": "--cross-encoder",
    },
]

# Fair capability coverage only: no adversarial or known-unanswerable prompts.
QUESTIONS = [
    {
        "id": "ss_01",
        "category": "single_source",
        "question": "What is the Capital Asset Pricing Model (CAPM) and what does each term in the formula represent?",
        "source_hint": "What Hedge Funds Really Do",
        "top_k": 6,
    },
    {
        "id": "ss_02",
        "category": "single_source",
        "question": "Explain the bias-variance tradeoff in machine learning.",
        "source_hint": "ISLP",
        "top_k": 6,
    },
    {
        "id": "ss_03",
        "category": "single_source",
        "question": "What is a Markov Decision Process and what are its key components?",
        "source_hint": "RLbook2020",
        "top_k": 6,
    },
    {
        "id": "ss_04",
        "category": "single_source",
        "question": "How does backpropagation work in training a neural network?",
        "source_hint": "Deep Learning: Foundations and Concepts",
        "top_k": 6,
    },
    {
        "id": "ss_05",
        "category": "single_source",
        "question": "What is the Sharpe ratio and how is it used to evaluate a portfolio?",
        "source_hint": "What Hedge Funds Really Do",
        "top_k": 6,
    },
    {
        "id": "sl_01",
        "category": "section_lookup",
        "question": "Where is market making discussed?",
        "source_hint": "Course notes / market mechanics material",
        "top_k": 5,
    },
    {
        "id": "sl_02",
        "category": "section_lookup",
        "question": "Where is Q-learning discussed?",
        "source_hint": "RLbook2020 / related notes",
        "top_k": 5,
    },
    {
        "id": "ms_01",
        "category": "multi_source",
        "question": "What techniques can be used to prevent overfitting in machine learning models?",
        "source_hint": "ISLP + Mitchell + Probabilistic ML",
        "top_k": 8,
    },
    {
        "id": "ms_02",
        "category": "multi_source",
        "question": "What are the key differences between supervised learning and reinforcement learning?",
        "source_hint": "ISLP + RLbook2020",
        "top_k": 8,
    },
    {
        "id": "ms_03",
        "category": "multi_source",
        "question": "How are hedge funds using artificial intelligence and big data in their investment strategies?",
        "source_hint": "What Hedge Funds + AI and Big Data",
        "top_k": 8,
    },
    {
        "id": "gk_01",
        "category": "general_knowledge",
        "question": "What is the Pythagorean theorem?",
        "source_hint": "Prompt behavior only; simple general-knowledge check",
        "top_k": 5,
    },
]

_REFUSAL_PHRASES = (
    "cannot answer",
    "not covered",
    "not found in",
    "does not appear",
    "no answer fabricated",
    "i cannot",
    "unable to answer",
    "not in the provided",
    "not present in",
    "no relevant",
)

_GENERAL_KNOWLEDGE_SIGNAL = "⚠️ GENERAL KNOWLEDGE"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_snapshot(out_path: Path, payload: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(out_path)


def _evaluate(category: str, result: Dict[str, Any]) -> tuple[bool, str]:
    mode = str(result.get("mode") or "")
    answer = str(result.get("llm_answer") or "").strip()
    answer_lower = answer.lower()
    route_intent = str(result.get("route_intent") or "")
    sources = result.get("sources_used") or []
    error = result.get("error")
    internet_triggered = bool(result.get("internet_triggered"))

    if error:
        return False, f"pipeline error: {str(error)[:100]}"

    if category == "single_source":
        if mode == "no_coverage":
            return False, "no_coverage on supported question"
        if not answer:
            return False, "empty answer"
        if any(p in answer_lower for p in _REFUSAL_PHRASES):
            return False, "refusal phrase in answer"
        return True, "answered from retrieved context"

    if category == "section_lookup":
        if route_intent != "section_lookup":
            return False, f"routed as {route_intent or 'unknown'}"
        if not answer:
            return False, "empty answer"
        return True, "section lookup routed correctly"

    if category == "multi_source":
        if mode == "no_coverage":
            return False, "no_coverage on supported multi-source question"
        unique_sources = len(set(sources))
        if unique_sources < 2:
            return False, f"only {unique_sources} source(s) used"
        if not answer:
            return False, "empty answer"
        return True, f"used {unique_sources} sources"

    if category == "general_knowledge":
        if not internet_triggered:
            return True, "internet not triggered; not graded as failure"
        if _GENERAL_KNOWLEDGE_SIGNAL not in answer:
            return False, "missing general knowledge warning"
        return True, "warning present as expected"

    return False, f"unknown category {category}"


def _extract_result(q: Dict[str, Any], encoder: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    answer = data.get("answer") or data.get("llm_answer") or ""
    mode = data.get("mode") or data.get("answer_mode") or ""
    routing = data.get("routing") or {}
    confidence = data.get("confidence") or {}
    internet_fb = data.get("internet_fallback") or {}
    context_pack = data.get("context_pack") or {}
    selected_chunks = context_pack.get("selected_chunks") or []
    top_chunk = selected_chunks[0] if selected_chunks else {}
    top_meta = top_chunk.get("metadata") or {}

    sources_used = list(dict.fromkeys(
        c.get("document_path") or c.get("source_name") or ""
        for c in selected_chunks
        if (c.get("document_path") or c.get("source_name"))
    ))
    source_types = list(dict.fromkeys(
        c.get("source_type") or ""
        for c in selected_chunks
        if c.get("source_type")
    ))

    route_intent = (
        data.get("route_intent")
        or routing.get("intent")
        or data.get("intent")
        or ""
    )

    result = {
        "id": q["id"],
        "category": q["category"],
        "question": q["question"],
        "source_hint": q.get("source_hint", ""),
        "encoder_id": encoder["id"],
        "encoder_label": encoder["label"],
        "llm_answer": answer,
        "mode": mode,
        "route_intent": route_intent,
        "internet_triggered": bool(internet_fb.get("triggered")),
        "sources_used": sources_used,
        "source_types": source_types,
        "selected_chunk_count": len(selected_chunks),
        "top_chunk": {
            "chunk_id": top_chunk.get("chunk_id"),
            "source_name": top_chunk.get("source_name"),
            "document_title": top_chunk.get("document_title"),
            "section_header": top_chunk.get("section_header"),
            "path_text": top_chunk.get("path_text"),
            "score": top_chunk.get("score"),
        },
        "query_expansion": top_meta.get("query_expansion"),
        "cross_encoder": top_meta.get("cross_encoder"),
        "score_gap_to_second": top_meta.get("score_gap_to_second"),
        "low_confidence": top_meta.get("low_confidence"),
        "confidence_band": routing.get("confidence_band") or confidence.get("band"),
        "confidence_rule": confidence.get("rule"),
        "top_score": confidence.get("top_score") or top_chunk.get("score"),
    }
    passed, reason = _evaluate(q["category"], result)
    result["passed"] = passed
    result["eval_reason"] = reason
    return result


def _run_variant(q: Dict[str, Any], encoder: Dict[str, Any]) -> Dict[str, Any]:
    cmd = [
        PYTHON,
        CLI,
        q["question"],
        "--top-k",
        str(q["top_k"]),
        "--answer",
        encoder["cli_flag"],
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            cwd=str(PROJECT_ROOT),
        )
        raw = (proc.stdout or "").strip()
        if not raw:
            return {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "encoder_id": encoder["id"],
                "encoder_label": encoder["label"],
                "error": f"empty output (stderr: {(proc.stderr or '')[:500]})",
                "passed": False,
                "eval_reason": "pipeline error: empty output",
            }
        data = json.loads(raw)
        result = _extract_result(q, encoder, data)
        if proc.stderr:
            result["stderr_tail"] = proc.stderr[-1200:]
        return result
    except subprocess.TimeoutExpired:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "encoder_id": encoder["id"],
            "encoder_label": encoder["label"],
            "error": "timeout",
            "passed": False,
            "eval_reason": "pipeline error: timeout",
        }
    except json.JSONDecodeError as exc:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "encoder_id": encoder["id"],
            "encoder_label": encoder["label"],
            "error": f"json_parse: {exc}",
            "passed": False,
            "eval_reason": f"pipeline error: json_parse {exc}",
        }
    except Exception as exc:
        return {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "encoder_id": encoder["id"],
            "encoder_label": encoder["label"],
            "error": str(exc),
            "passed": False,
            "eval_reason": f"pipeline error: {str(exc)[:100]}",
        }


def _build_summary(results: List[Dict[str, Any]], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    encoder_stats: Dict[str, Dict[str, Any]] = {}
    for encoder in ENCODER_MODES:
        subset = [r for r in results if r.get("encoder_id") == encoder["id"]]
        passed = sum(1 for r in subset if r.get("passed"))
        encoder_stats[encoder["id"]] = {
            "label": encoder["label"],
            "runs": len(subset),
            "passed": passed,
            "failed": len(subset) - passed,
            "pass_rate": round((passed / len(subset)) * 100, 1) if subset else 0.0,
            "avg_top_score": round(
                sum(float(r.get("top_score") or 0.0) for r in subset) / len(subset),
                4,
            ) if subset else 0.0,
        }

    question_pairs: List[Dict[str, Any]] = []
    for q in questions:
        pair = {
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "runs": [r for r in results if r.get("id") == q["id"]],
        }
        question_pairs.append(pair)

    return {
        "completed_runs": len(results),
        "expected_runs": len(questions) * len(ENCODER_MODES),
        "encoder_stats": encoder_stats,
        "question_pairs": question_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live-updating encoder comparison battery")
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated question ids or categories, e.g. ss_01,section_lookup",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Live-updating output JSON path",
    )
    args = parser.parse_args()

    filters = {f.strip() for f in args.only.split(",") if f.strip()}
    questions = [
        q for q in QUESTIONS
        if not filters or q["id"] in filters or q["category"] in filters
    ]

    out_path = Path(args.out)
    payload: Dict[str, Any] = {
        "run_started_at": _now(),
        "updated_at": _now(),
        "question_count": len(questions),
        "encoder_modes": ENCODER_MODES,
        "results": [],
        "summary": _build_summary([], questions),
    }
    _write_snapshot(out_path, payload)

    print(f"\nRunning {len(questions)} questions x {len(ENCODER_MODES)} encoders = {len(questions) * len(ENCODER_MODES)} runs")
    print(f"Live file: {out_path}\n")

    results: List[Dict[str, Any]] = []
    total_runs = len(questions) * len(ENCODER_MODES)
    run_idx = 0

    for q_idx, q in enumerate(questions, 1):
        print(f"[{q_idx}/{len(questions)}] {q['id']} | {q['question']}")
        for encoder in ENCODER_MODES:
            run_idx += 1
            print(f"  [{run_idx}/{total_runs}] {encoder['label']} ...")
            result = _run_variant(q, encoder)
            results.append(result)

            icon = "PASS" if result.get("passed") else "FAIL"
            print(f"    {icon} | {result.get('eval_reason', '')}")

            payload = {
                "run_started_at": payload["run_started_at"],
                "updated_at": _now(),
                "question_count": len(questions),
                "encoder_modes": ENCODER_MODES,
                "results": results,
                "summary": _build_summary(results, questions),
            }
            _write_snapshot(out_path, payload)

    summary = payload["summary"]
    print("\nEncoder summary:")
    for encoder in ENCODER_MODES:
        stats = summary["encoder_stats"][encoder["id"]]
        print(
            f"  {encoder['label']}: {stats['passed']}/{stats['runs']} passed "
            f"({stats['pass_rate']}%), avg_top_score={stats['avg_top_score']}"
        )

    print(f"\nSaved live results: {out_path}")


if __name__ == "__main__":
    main()
