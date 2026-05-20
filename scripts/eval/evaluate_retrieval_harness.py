from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters, retrieve_as_dict
from utils.runtime_defaults import DEFAULT_DB_DSN, DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME


@dataclass
class RetrievalCase:
    case_id: str
    category: str
    query: str
    notes: str = ""
    expected_source_contains: List[str] = field(default_factory=list)
    min_unique_sources_top_n: int = 1
    avoid_top1_table_data: bool = False


@dataclass
class CaseResult:
    case_id: str
    category: str
    query: str
    notes: str
    top_k: int
    expected_source_contains: List[str]
    min_unique_sources_top_n: int
    avoid_top1_table_data: bool
    unique_sources_top_n: int
    top1_source: str
    top1_role: str
    checks_passed: bool
    failed_checks: List[str]
    hits: List[Dict[str, Any]]


DEFAULT_CASES: List[RetrievalCase] = [
    RetrievalCase(
        case_id="S1",
        category="single-source",
        query="In ISLP 4.3.1, what logistic function is used for probability?",
        notes="Should strongly favor ISLP source.",
        expected_source_contains=["ISLP_website.pdf"],
    ),
    RetrievalCase(
        case_id="S2",
        category="single-source",
        query="What does section 10.2.3 Maximum likelihood estimation say about logistic regression?",
        notes="Should favor PML intro source.",
        expected_source_contains=["Probabilistic Machine Learning_ An Introduction"],
    ),
    RetrievalCase(
        case_id="S3",
        category="single-source",
        query="In 5.4.3 logistic regression, how does parameter count scale with feature dimension M?",
        notes="Should favor Deep Learning Foundations and Concepts source.",
        expected_source_contains=["Deep Learning_ Foundations and Concepts"],
    ),
    RetrievalCase(
        case_id="M1",
        category="multi-source",
        query="Explain logistic regression using perspectives from multiple books.",
        notes="Expect at least two distinct sources in top hits.",
        min_unique_sources_top_n=2,
    ),
    RetrievalCase(
        case_id="M2",
        category="multi-source",
        query="Compare binary logistic regression definitions across sources.",
        notes="Expect source diversity in top hits.",
        min_unique_sources_top_n=2,
    ),
    RetrievalCase(
        case_id="M3",
        category="multi-source",
        query="How is maximum likelihood used for logistic regression across different texts?",
        notes="Expect source diversity in top hits.",
        min_unique_sources_top_n=2,
    ),
    RetrievalCase(
        case_id="N1",
        category="noisy-source",
        query="In ISLP Figure 4.2, why does linear regression give invalid probabilities while logistic regression does not?",
        notes="Checks if noisy parsed regions are not dominating top-1 role.",
        expected_source_contains=["ISLP_website.pdf"],
        avoid_top1_table_data=True,
    ),
    RetrievalCase(
        case_id="N2",
        category="noisy-source",
        query="In ISLP logistic regression section, what happens for low and high balances?",
        notes="Should retrieve substantive prose from noisy-source document.",
        expected_source_contains=["ISLP_website.pdf"],
        avoid_top1_table_data=True,
    ),
    RetrievalCase(
        case_id="N3",
        category="noisy-source",
        query="What does 10.2.2 Nonlinear classifiers discuss near logistic regression?",
        notes="Potentially table-heavy neighborhood; inspect role mix.",
        expected_source_contains=["Probabilistic Machine Learning_ An Introduction"],
    ),
    RetrievalCase(
        case_id="C1",
        category="conflict",
        query="Is logistic regression a regression model or a classification model?",
        notes="Expect multiple sources; inspect consistency/conflicts manually.",
        min_unique_sources_top_n=2,
    ),
    RetrievalCase(
        case_id="C2",
        category="conflict",
        query="What default-probability threshold is discussed in ISLP, and are alternatives suggested?",
        notes="Can surface potentially conflicting threshold mentions.",
        expected_source_contains=["ISLP_website.pdf"],
    ),
    RetrievalCase(
        case_id="C3",
        category="conflict",
        query="Do sources describe logistic regression with labels {0,1} or {-1,+1}?",
        notes="Expect multi-source retrieval for label-convention differences.",
        min_unique_sources_top_n=2,
    ),
]


def _build_hit_row(hit: Dict[str, Any]) -> Dict[str, Any]:
    md = hit.get("metadata") or {}
    return {
        "chunk_id": hit.get("chunk_id"),
        "score": hit.get("score"),
        "source_name": hit.get("source_name") or md.get("source_name"),
        "document_title": hit.get("document_title") or md.get("document_title"),
        "document_path": hit.get("document_path") or md.get("document_path"),
        "collection_id": hit.get("collection_id") or md.get("collection_id"),
        "section_header": hit.get("section_header") or md.get("section_header") or hit.get("title"),
        "page_number": hit.get("page_number") or md.get("page_number") or hit.get("page_start"),
        "source_type": hit.get("source_type") or md.get("source_type"),
        "structural_role": hit.get("structural_role") or md.get("structural_role"),
        "path_text": hit.get("path_text"),
    }


def _evaluate_case(
    case: RetrievalCase,
    *,
    db_path: str,
    top_k: int,
    embed_backend: str,
    embed_model_name: str,
    source_type: Optional[str],
    structural_role: Optional[str],
) -> CaseResult:
    retrieved = retrieve_as_dict(
        case.query,
        db_dsn=db_path,
        top_k=top_k,
        filters=RetrievalFilters(
            source_type=source_type,
            structural_role=structural_role,
        ),
        embed_backend=embed_backend,
        embed_model_name=embed_model_name,
    )

    hits = list(retrieved.get("hits") or [])
    top_hits = hits[:top_k]
    source_names = [str((_h.get("source_name") or ((_h.get("metadata") or {}).get("source_name")) or "")).strip() for _h in top_hits]
    unique_sources = len({s for s in source_names if s})

    failed: List[str] = []

    if case.expected_source_contains:
        joined = "\n".join(source_names).lower()
        if not any(token.lower() in joined for token in case.expected_source_contains):
            failed.append("expected_source_not_found_in_top_hits")

    if unique_sources < int(case.min_unique_sources_top_n):
        failed.append("insufficient_source_diversity")

    top1 = top_hits[0] if top_hits else {}
    top1_role = str(top1.get("structural_role") or ((top1.get("metadata") or {}).get("structural_role")) or "")
    if case.avoid_top1_table_data and top1_role == "table_data":
        failed.append("top1_is_table_data")

    rows = [_build_hit_row(h) for h in top_hits]

    return CaseResult(
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        notes=case.notes,
        top_k=top_k,
        expected_source_contains=list(case.expected_source_contains),
        min_unique_sources_top_n=int(case.min_unique_sources_top_n),
        avoid_top1_table_data=bool(case.avoid_top1_table_data),
        unique_sources_top_n=unique_sources,
        top1_source=source_names[0] if source_names else "",
        top1_role=top1_role,
        checks_passed=len(failed) == 0,
        failed_checks=failed,
        hits=rows,
    )


def _summary(results: List[CaseResult]) -> Dict[str, Any]:
    by_category: Dict[str, Dict[str, int]] = {}
    for r in results:
        slot = by_category.setdefault(r.category, {"count": 0, "passed": 0})
        slot["count"] += 1
        if r.checks_passed:
            slot["passed"] += 1

    total = len(results)
    passed = sum(1 for r in results if r.checks_passed)
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate": round((passed / total), 4) if total else 0.0,
        "by_category": by_category,
    }


def _to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# RAG Retrieval Harness Report")
    lines.append("")
    lines.append(f"Generated (UTC): {payload['generated_at_utc']}")
    lines.append("")

    s = payload["summary"]
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total cases: **{s['total_cases']}**")
    lines.append(f"- Passed: **{s['passed_cases']}**")
    lines.append(f"- Failed: **{s['failed_cases']}**")
    lines.append(f"- Pass rate: **{s['pass_rate']:.2%}**")
    lines.append("")

    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Cases | Passed |")
    lines.append("|---|---:|---:|")
    for cat, row in sorted(s["by_category"].items()):
        lines.append(f"| {cat} | {row['count']} | {row['passed']} |")
    lines.append("")

    lines.append("## Case Details")
    lines.append("")
    for item in payload["cases"]:
        lines.append(f"### {item['case_id']} · {item['category']}")
        lines.append("")
        lines.append(f"- Query: {item['query']}")
        if item.get("notes"):
            lines.append(f"- Notes: {item['notes']}")
        lines.append(f"- Checks passed: **{item['checks_passed']}**")
        if item.get("failed_checks"):
            lines.append(f"- Failed checks: {', '.join(item['failed_checks'])}")
        lines.append(f"- Unique sources in top-{item['top_k']}: {item['unique_sources_top_n']}")
        lines.append("")
        lines.append("Top chunks:")
        for idx, hit in enumerate(item["hits"], start=1):
            score = hit.get("score")
            score_str = f"{float(score):.4f}" if isinstance(score, (float, int)) else "n/a"
            lines.append(
                f"{idx}. score={score_str} | source={hit.get('source_name') or 'n/a'} | "
                f"title={hit.get('document_title') or 'n/a'} | page={hit.get('page_number') or 'n/a'} | "
                f"role={hit.get('structural_role') or 'n/a'} | chunk={hit.get('chunk_id') or 'n/a'}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_harness(
    *,
    db_path: str,
    top_k: int,
    embed_backend: str,
    embed_model_name: str,
    source_type: Optional[str],
    structural_role: Optional[str],
) -> Dict[str, Any]:
    results: List[CaseResult] = []
    for case in DEFAULT_CASES:
        results.append(
            _evaluate_case(
                case,
                db_path=db_path,
                top_k=top_k,
                embed_backend=embed_backend,
                embed_model_name=embed_model_name,
                source_type=source_type,
                structural_role=structural_role,
            )
        )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "settings": {
            "db_path": db_path,
            "top_k": top_k,
            "embed_backend": embed_backend,
            "embed_model_name": embed_model_name,
            "source_type": source_type,
            "structural_role": structural_role,
            "case_count": len(DEFAULT_CASES),
        },
        "summary": _summary(results),
        "cases": [asdict(r) for r in results],
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lightweight RAG retrieval evaluation harness (single-source, multi-source, noisy-source, conflict)."
    )
    parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--embed-backend", type=str, default=DEFAULT_EMBED_BACKEND)
    parser.add_argument("--embed-model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
    parser.add_argument("--source-type", type=str, default=None)
    parser.add_argument("--structural-role", type=str, default=None)
    parser.add_argument("--out-json", type=str, default="data/qa/retrieval_harness_report.json")
    parser.add_argument("--out-md", type=str, default="data/qa/retrieval_harness_report.md")
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    payload = run_harness(
        db_path=args.db,
        top_k=args.top_k,
        embed_backend=args.embed_backend,
        embed_model_name=args.embed_model,
        source_type=args.source_type,
        structural_role=args.structural_role,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_to_markdown(payload), encoding="utf-8")

    print(str(out_json))
    print(str(out_md))
    print(f"pass_rate={payload['summary']['pass_rate']:.2%}")

    if args.fail_on_mismatch and payload["summary"]["failed_cases"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
