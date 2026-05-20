"""Precision/recall evaluation harness for the RAG retrieval pipeline.

Metrics computed per case:
  recall@k   — fraction of expected chunk IDs found in the top-k results
  precision@k — fraction of top-k results that are expected chunk IDs
  MRR        — reciprocal rank of the first expected chunk in the result list

Usage
-----
# 1. Seed a dataset from current top-k results (populates expected_chunk_ids
#    with whatever the system currently returns — treat as a snapshot, not
#    ground truth until you validate them).
python scripts/eval/recall_mrr_harness.py --build-dataset --out data/qa/retrieval_recall_dataset.json

# 2. Evaluate against a saved dataset:
python scripts/eval/recall_mrr_harness.py --dataset data/qa/retrieval_recall_dataset.json

# 3. Evaluate only specific cases:
python scripts/eval/recall_mrr_harness.py --dataset data/qa/retrieval_recall_dataset.json --ids S1 S2

# 4. Save a JSON report:
python scripts/eval/recall_mrr_harness.py --dataset data/qa/retrieval_recall_dataset.json --out report.json

Dataset schema
--------------
{
  "version": "1.0",
  "cases": [
    {
      "case_id": "R1",
      "query": "...",
      "notes": "...",
      "expected_chunk_ids": ["abc123", "def456"],
      "top_k": 10
    }
  ]
}

If "expected_chunk_ids" is empty or absent the case is skipped for metric
computation but printed (useful for exploratory queries).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters, retrieve_as_dict
from retrieval.router import route_query
from utils.runtime_defaults import DEFAULT_DB_DSN, DEFAULT_EMBED_BACKEND, DEFAULT_EMBED_MODEL_NAME

_HISTORY_DIR = PROJECT_ROOT / "data" / "diagnostics" / "eval_history"

# ── Default dataset of queries ────────────────────────────────────────────────
# expected_chunk_ids is populated by --build-dataset and validated by humans.
_DEFAULT_CASES: List[Dict[str, Any]] = [
    {
        "case_id": "R1",
        "query": "In ISLP, what is the bias-variance tradeoff and how does model flexibility affect test MSE?",
        "notes": "Chapter 2 ISLP. Should retrieve bias/variance/flexibility chunks.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R2",
        "query": "According to ISLP, what is recursive binary splitting in the context of decision trees?",
        "notes": "Chapter 8 ISLP. Should retrieve tree-splitting chunks.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R3",
        "query": "What does beta measure in CAPM and what does beta greater than 1 indicate?",
        "notes": "CS7646 course notes. CAPM/beta definition.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R4",
        "query": "What is dropout regularization and how does it prevent overfitting in neural networks?",
        "notes": "Deep Learning book. Dropout technique.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R5",
        "query": "In ISLP logistic regression section, what happens for low and high balance values?",
        "notes": "ISLP logistic regression chapter.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R6",
        "query": "Explain the backpropagation algorithm for training neural networks.",
        "notes": "Any deep learning source. Core NN training algorithm.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R7",
        "query": "What is the kernel trick in support vector machines?",
        "notes": "Any ML source covering SVM kernels.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
    {
        "case_id": "R8",
        "query": "What is cross-validation and how does k-fold cross-validation work?",
        "notes": "ISLP or any ML source. Standard CV explanation.",
        "expected_chunk_ids": [],
        "top_k": 10,
    },
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RecallCase:
    case_id: str
    query: str
    notes: str = ""
    expected_chunk_ids: List[str] = field(default_factory=list)
    top_k: int = 10
    intent: str = ""  # optional ground-truth intent label for per-intent breakdown
    is_negative: bool = False  # True → no corpus answer expected (internet-fallback query)
    split: str = "dev"  # "dev" | "test" — used with --split filter
    difficulty: str = ""  # "easy" | "medium" | "hard"
    chunk_relevance: Dict[str, int] = field(default_factory=dict)  # chunk_id → 0|1|2 graded score


@dataclass
class CaseMetrics:
    case_id: str
    query: str
    notes: str
    top_k: int
    intent: str
    expected_count: int
    retrieved_count: int
    hits_in_results: int
    recall_at_k: Optional[float]      # None when no expected IDs
    recall_at_1: Optional[float]      # R@1 — was the top result relevant?
    recall_at_3: Optional[float]      # R@3
    recall_at_5: Optional[float]      # R@5
    precision_at_k: Optional[float]   # None when no expected IDs
    mrr: Optional[float]              # None when no expected IDs
    ndcg_at_k: Optional[float]        # None when no expected IDs
    map_at_k: Optional[float]         # None when no expected IDs
    first_hit_rank: Optional[int]     # 1-indexed; None if not found
    retrieved_ids: List[str]          # top-k chunk IDs actually retrieved
    is_negative: bool = False         # True → corpus_negative case; excluded from aggregate
    split: str = "dev"
    difficulty: str = ""
    graded_ndcg_at_k: Optional[float] = None  # graded NDCG when chunk_relevance is present


# ── Metric helpers ─────────────────────────────────────────────────────

import math as _math


def _ndcg(retrieved: List[str], relevant: set, k: int) -> float:
    """Binary NDCG@k.  DCG / IDCG where each relevant item has gain=1."""
    dcg = sum(
        1.0 / _math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
        if cid in relevant
    )
    # Ideal: all relevant items appear at ranks 1..min(|relevant|, k)
    n_ideal = min(len(relevant), k)
    idcg = sum(1.0 / _math.log2(i + 1) for i in range(1, n_ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _ndcg_graded(retrieved: List[str], relevance: Dict[str, int], k: int) -> float:
    """Graded NDCG@k using per-chunk relevance scores (0=irrelevant, 1=partial, 2=full)."""
    dcg = sum(
        relevance.get(cid, 0) / _math.log2(rank + 1)
        for rank, cid in enumerate(retrieved[:k], start=1)
    )
    ideal_gains = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum(g / _math.log2(i + 1) for i, g in enumerate(ideal_gains, start=1))
    return dcg / idcg if idcg > 0 else 0.0


def _average_precision(retrieved: List[str], relevant: set, k: int) -> float:
    """Average Precision (AP) for one query."""
    if not relevant:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, cid in enumerate(retrieved[:k], start=1):
        if cid in relevant:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(relevant)


# ── Core computation ──────────────────────────────────────────────────────────

def _compute_metrics(
    case: RecallCase,
    retrieved_ids: List[str],
) -> CaseMetrics:
    expected = set(case.expected_chunk_ids)
    retrieved = retrieved_ids[: case.top_k]
    n = len(retrieved)

    if not expected:
        return CaseMetrics(
            case_id=case.case_id,
            query=case.query,
            notes=case.notes,
            top_k=case.top_k,
            intent=case.intent,
            expected_count=0,
            retrieved_count=n,
            hits_in_results=0,
            recall_at_k=None,
            recall_at_1=None,
            recall_at_3=None,
            recall_at_5=None,
            precision_at_k=None,
            mrr=None,
            ndcg_at_k=None,
            map_at_k=None,
            first_hit_rank=None,
            retrieved_ids=retrieved,
            is_negative=case.is_negative,
            split=case.split,
            difficulty=case.difficulty,
            graded_ndcg_at_k=None,
        )

    hits = [cid for cid in retrieved if cid in expected]
    hits_in_top_k = len(hits)

    recall = hits_in_top_k / len(expected) if expected else None
    precision = hits_in_top_k / n if n else 0.0
    recall1 = len([c for c in retrieved[:1] if c in expected]) / len(expected) if expected else None
    recall3 = len([c for c in retrieved[:3] if c in expected]) / len(expected) if expected else None
    recall5 = len([c for c in retrieved[:5] if c in expected]) / len(expected) if expected else None

    first_hit_rank: Optional[int] = None
    for rank, cid in enumerate(retrieved, start=1):
        if cid in expected:
            first_hit_rank = rank
            break
    mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0
    ndcg = _ndcg(retrieved, expected, case.top_k)
    ap = _average_precision(retrieved, expected, case.top_k)
    graded_ndcg: Optional[float] = None
    if case.chunk_relevance:
        graded_ndcg = round(_ndcg_graded(retrieved, case.chunk_relevance, case.top_k), 4)

    return CaseMetrics(
        case_id=case.case_id,
        query=case.query,
        notes=case.notes,
        top_k=case.top_k,
        intent=case.intent,
        expected_count=len(expected),
        retrieved_count=n,
        hits_in_results=hits_in_top_k,
        recall_at_k=round(recall, 4) if recall is not None else None,
        recall_at_1=round(recall1, 4) if recall1 is not None else None,
        recall_at_3=round(recall3, 4) if recall3 is not None else None,
        recall_at_5=round(recall5, 4) if recall5 is not None else None,
        precision_at_k=round(precision, 4),
        mrr=round(mrr, 4),
        ndcg_at_k=round(ndcg, 4),
        map_at_k=round(ap, 4),
        first_hit_rank=first_hit_rank,
        retrieved_ids=retrieved,
        is_negative=case.is_negative,
        split=case.split,
        difficulty=case.difficulty,
        graded_ndcg_at_k=graded_ndcg,
    )


def _run_retrieval(
    case: RecallCase,
    *,
    db_dsn: str,
    embed_backend: str,
    embed_model_name: str,
) -> List[str]:
    """Call the retrieval pipeline and return ordered chunk IDs.

    Routes the query first (so book-scope, collection-scope, and intent
    signals are applied) then calls ``retrieve_as_dict`` with the resulting
    doc_id filter and rewritten query — matching what the production API does.
    """
    routed = route_query(case.query, db_dsn=db_dsn)
    result = retrieve_as_dict(
        routed.effective_query,
        db_dsn=db_dsn,
        top_k=case.top_k,
        filters=RetrievalFilters(doc_ids=routed.doc_ids or None),
        embed_backend=embed_backend,
        embed_model_name=embed_model_name,
        internet_fallback_enabled=False,
    )
    hits: List[Dict[str, Any]] = result.get("hits") or []
    return [str(h.get("chunk_id", "")) for h in hits if h.get("chunk_id")]


# ── Aggregate summary ─────────────────────────────────────────────────────────

def _aggregate(metrics_list: List[CaseMetrics]) -> Dict[str, Any]:
    # Exclude corpus_negative cases from aggregate metrics (they have no expected IDs).
    evaluable = [m for m in metrics_list if m.recall_at_k is not None and not m.is_negative]
    n = len(evaluable)
    if n == 0:
        return {
            "evaluable_cases": 0,
            "skipped_no_expected": sum(1 for m in metrics_list if not m.is_negative),
            "negative_cases": sum(1 for m in metrics_list if m.is_negative),
            "mean_recall_at_k": None,
            "mean_precision_at_k": None,
            "mean_mrr": None,
        }
    mean_recall = sum(m.recall_at_k for m in evaluable) / n  # type: ignore[arg-type]
    mean_r1 = sum(m.recall_at_1 for m in evaluable if m.recall_at_1 is not None) / n
    mean_r3 = sum(m.recall_at_3 for m in evaluable if m.recall_at_3 is not None) / n
    mean_r5 = sum(m.recall_at_5 for m in evaluable if m.recall_at_5 is not None) / n
    mean_precision = sum(m.precision_at_k for m in evaluable) / n  # type: ignore[arg-type]
    mean_mrr = sum(m.mrr for m in evaluable) / n  # type: ignore[arg-type]
    mean_ndcg = sum(m.ndcg_at_k for m in evaluable) / n  # type: ignore[arg-type]
    mean_map = sum(m.map_at_k for m in evaluable) / n  # type: ignore[arg-type]

    # Per-intent breakdown
    intent_groups: Dict[str, List[CaseMetrics]] = {}
    for m in evaluable:
        key = m.intent or "(unknown)"
        intent_groups.setdefault(key, []).append(m)
    per_intent: Dict[str, Any] = {}
    for intent, group in sorted(intent_groups.items()):
        per_intent[intent] = {
            "n": len(group),
            "mean_recall_at_k": round(sum(m.recall_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
            "mean_mrr": round(sum(m.mrr for m in group) / len(group), 4),  # type: ignore[arg-type]
            "mean_ndcg_at_k": round(sum(m.ndcg_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
            "map": round(sum(m.map_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
        }

    # Per-difficulty breakdown
    difficulty_groups: Dict[str, List[CaseMetrics]] = {}
    for m in evaluable:
        key = m.difficulty or "(unset)"
        difficulty_groups.setdefault(key, []).append(m)
    per_difficulty: Dict[str, Any] = {}
    for diff, group in sorted(difficulty_groups.items()):
        per_difficulty[diff] = {
            "n": len(group),
            "mean_recall_at_k": round(sum(m.recall_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
            "mean_mrr": round(sum(m.mrr for m in group) / len(group), 4),  # type: ignore[arg-type]
            "mean_ndcg_at_k": round(sum(m.ndcg_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
            "map": round(sum(m.map_at_k for m in group) / len(group), 4),  # type: ignore[arg-type]
        }

    # Graded NDCG average (only when chunk_relevance data is present)
    graded_ndcg_vals = [m.graded_ndcg_at_k for m in evaluable if m.graded_ndcg_at_k is not None]
    mean_graded_ndcg: Optional[float] = (
        round(sum(graded_ndcg_vals) / len(graded_ndcg_vals), 4) if graded_ndcg_vals else None
    )

    n_negative = sum(1 for m in metrics_list if m.is_negative)
    n_skipped = sum(1 for m in metrics_list if m.recall_at_k is None and not m.is_negative)
    return {
        "evaluable_cases": n,
        "skipped_no_expected": n_skipped,
        "negative_cases": n_negative,
        "mean_recall_at_1": round(mean_r1, 4),
        "mean_recall_at_3": round(mean_r3, 4),
        "mean_recall_at_5": round(mean_r5, 4),
        "mean_recall_at_k": round(mean_recall, 4),
        "mean_precision_at_k": round(mean_precision, 4),
        "mean_mrr": round(mean_mrr, 4),
        "mean_ndcg_at_k": round(mean_ndcg, 4),
        "mean_graded_ndcg_at_k": mean_graded_ndcg,
        "map": round(mean_map, 4),
        "per_intent": per_intent,
        "per_difficulty": per_difficulty,
    }


# ── CLI output ────────────────────────────────────────────────────────────────

def _print_results(metrics_list: List[CaseMetrics], agg: Dict[str, Any]) -> None:
    print()
    header = f"{'ID':<6} {'R@k':>6} {'P@k':>6} {'MRR':>6} {'NDCG':>6} {'AP':>6} {'Rank':>5}  Query"
    print(header)
    print("-" * len(header))
    for m in metrics_list:
        if m.is_negative:
            # Negative cases: show as NEG row with no metric values.
            q = m.query[:45] + ("…" if len(m.query) > 45 else "")
            print(f"{m.case_id:<6} {'[NEG]':>6} {'':>6} {'':>6} {'':>6} {'':>6} {'':>5}  {q}")
            continue
        r = f"{m.recall_at_k:.3f}" if m.recall_at_k is not None else "  N/A"
        p = f"{m.precision_at_k:.3f}" if m.precision_at_k is not None else "  N/A"
        mrr = f"{m.mrr:.3f}" if m.mrr is not None else "  N/A"
        ndcg = f"{m.ndcg_at_k:.3f}" if m.ndcg_at_k is not None else "  N/A"
        ap = f"{m.map_at_k:.3f}" if m.map_at_k is not None else "  N/A"
        rank = str(m.first_hit_rank) if m.first_hit_rank else "-"
        q = m.query[:45] + ("…" if len(m.query) > 45 else "")
        print(f"{m.case_id:<6} {r:>6} {p:>6} {mrr:>6} {ndcg:>6} {ap:>6} {rank:>5}  {q}")

    print()
    n = agg["evaluable_cases"]
    skip = agg["skipped_no_expected"]
    neg = agg.get("negative_cases", 0)
    parts = [f"{n} evaluated"]
    if skip:
        parts.append(f"{skip} skipped (no expected IDs)")
    if neg:
        parts.append(f"{neg} negative (corpus_negative)")
    print("  ".join(parts))
    if n:
        k_label = f"@{metrics_list[0].top_k}" if metrics_list else "@k"
        print(f"  R@1            : {agg.get('mean_recall_at_1', 0):.4f}")
        print(f"  R@3            : {agg.get('mean_recall_at_3', 0):.4f}")
        print(f"  R@5            : {agg.get('mean_recall_at_5', 0):.4f}")
        print(f"  R{k_label:<13}: {agg['mean_recall_at_k']:.4f}")
        print(f"  Mean precision{k_label}: {agg['mean_precision_at_k']:.4f}")
        print(f"  Mean MRR       : {agg['mean_mrr']:.4f}")
        print(f"  Mean NDCG{k_label}  : {agg['mean_ndcg_at_k']:.4f}")
        if agg.get("mean_graded_ndcg_at_k") is not None:
            print(f"  Graded NDCG{k_label}: {agg['mean_graded_ndcg_at_k']:.4f}")
        print(f"  MAP            : {agg['map']:.4f}")
        if agg.get("per_difficulty") and len(agg["per_difficulty"]) > 1:
            print()
            print("  Per-difficulty breakdown:")
            pd_header = f"    {'Difficulty':<10} {'N':>3} {'R@k':>6} {'MRR':>6} {'NDCG':>6} {'MAP':>6}"
            print(pd_header)
            print("    " + "-" * (len(pd_header) - 4))
            for diff, stats in agg["per_difficulty"].items():
                print(
                    f"    {diff:<10} {stats['n']:>3} "
                    f"{stats['mean_recall_at_k']:>6.3f} {stats['mean_mrr']:>6.3f} "
                    f"{stats['mean_ndcg_at_k']:>6.3f} {stats['map']:>6.3f}"
                )
        if agg.get("per_intent") and len(agg["per_intent"]) > 1:
            print()
            print("  Per-intent breakdown:")
            pi_header = f"    {'Intent':<22} {'N':>3} {'R@k':>6} {'MRR':>6} {'NDCG':>6} {'MAP':>6}"
            print(pi_header)
            print("    " + "-" * (len(pi_header) - 4))
            for intent, stats in agg["per_intent"].items():
                print(
                    f"    {intent:<22} {stats['n']:>3} "
                    f"{stats['mean_recall_at_k']:>6.3f} {stats['mean_mrr']:>6.3f} "
                    f"{stats['mean_ndcg_at_k']:>6.3f} {stats['map']:>6.3f}"
                )
    print()


# ── Regression history helpers ─────────────────────────────────────────

def _save_history(agg: Dict[str, Any], metrics_list: List[CaseMetrics]) -> Path:
    """Write a timestamped JSON snapshot to the eval history directory."""
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = _HISTORY_DIR / f"run_{ts}.json"
    snap = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "summary": {k: v for k, v in agg.items() if k != "per_intent"},
        "per_intent": agg.get("per_intent", {}),
        "cases": [{"case_id": m.case_id, "recall_at_k": m.recall_at_k,
                   "mrr": m.mrr, "ndcg_at_k": m.ndcg_at_k, "map_at_k": m.map_at_k}
                  for m in metrics_list],
    }
    path.write_text(json.dumps(snap, indent=2), encoding="utf-8")
    return path


def _load_last_history() -> Optional[Dict[str, Any]]:
    """Load the most recent run snapshot, or None if none exists."""
    if not _HISTORY_DIR.exists():
        return None
    snapshots = sorted(_HISTORY_DIR.glob("run_*.json"))
    if not snapshots:
        return None
    return json.loads(snapshots[-1].read_text(encoding="utf-8"))


_METRIC_LABELS = {
    "mean_recall_at_k": "R@k",
    "mean_mrr": "MRR",
    "mean_ndcg_at_k": "NDCG@k",
    "map": "MAP",
}


def _print_regression_delta(agg: Dict[str, Any], prev: Dict[str, Any]) -> None:
    """Print a delta table comparing current run to the previous snapshot."""
    print()
    print("─" * 52)
    print("  Regression delta vs last run")
    print("─" * 52)
    prev_summary = prev.get("summary", {})
    any_regression = False
    for key, label in _METRIC_LABELS.items():
        cur = agg.get(key)
        old = prev_summary.get(key)
        if cur is None or old is None:
            continue
        delta = cur - old
        flag = ""
        if delta < -0.005:
            flag = "  ⚠️  REGRESSION"
            any_regression = True
        elif delta > 0.005:
            flag = "  ⬆  improved"
        sign = "+" if delta >= 0 else "-"
        print(f"  {label:<10}  {old:.4f}  →  {cur:.4f}  ({sign}{abs(delta):.4f}){flag}")
    print("─" * 52)
    if any_regression:
        print("  ⚠️  One or more metrics regressed vs last run.")
    else:
        print("  ✅  No regressions detected.")
    print()


# ── Chunk dedup / overlap quality check ──────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity between two text strings."""
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def _dedup_check(
    cases: List[RecallCase],
    db_dsn: str,
    threshold: float = 0.85,
) -> None:
    """Scan expected_chunk_ids for near-duplicate chunk pairs.

    Loads chunk text from the DB, then computes pairwise token-level Jaccard
    similarity.  Pairs exceeding *threshold* are flagged — they should be
    merged or one dropped from the expected set.
    """
    all_ids: List[str] = []
    for case in cases:
        all_ids.extend(case.expected_chunk_ids)
    unique_ids = list(dict.fromkeys(all_ids))  # preserve order, deduplicate
    if not unique_ids:
        print("  dedup-check: no expected chunk IDs to scan.")
        return

    # Load chunk text from DB
    try:
        import psycopg2  # type: ignore
    except ImportError:
        print("  dedup-check: psycopg2 not available — skipping.")
        return

    texts: Dict[str, str] = {}
    try:
        conn = psycopg2.connect(db_dsn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
                (unique_ids,),
            )
            for row in cur.fetchall():
                texts[str(row[0])] = row[1] or ""
        conn.close()
    except Exception as exc:
        print(f"  dedup-check: DB error — {exc}")
        return

    if not texts:
        print("  dedup-check: no chunk text loaded from DB.")
        return

    loaded = [cid for cid in unique_ids if cid in texts]
    flagged: List[tuple] = []
    for i, a in enumerate(loaded):
        for b in loaded[i + 1:]:
            sim = _jaccard(texts[a], texts[b])
            if sim >= threshold:
                flagged.append((a, b, sim))

    print()
    print("─" * 60)
    print(f"  Chunk dedup quality check  (Jaccard threshold ≥ {threshold})")
    print("─" * 60)
    print(f"  Scanned {len(loaded)} unique expected chunk IDs")
    if flagged:
        print(f"  ⚠️  {len(flagged)} near-duplicate pair(s) found:")
        for a, b, sim in sorted(flagged, key=lambda x: -x[2]):
            print(f"    {a}  ↔  {b}  (Jaccard={sim:.3f})")
    else:
        print("  ✅  No near-duplicate expected chunks detected.")
    print()


# ── Build-dataset mode ────────────────────────────────────────────────────────

def _build_dataset(
    cases: List[RecallCase],
    *,
    db_dsn: str,
    embed_backend: str,
    embed_model_name: str,
    out_path: Path,
    existing_cases: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Run retrieval for each case and save the top-k chunk IDs as expected IDs.

    This creates an initial snapshot.  Validate and trim the expected_chunk_ids
    list by hand before using the dataset for regression testing.

    When ``existing_cases`` is provided (a list of all case dicts from the
    dataset file), the seeded results are *merged* back into it so that
    existing curated cases are preserved.  Cases not in ``cases`` keep their
    current ``expected_chunk_ids``.
    """
    print(f"Building dataset from {len(cases)} cases …")
    seeded_by_id: Dict[str, Dict[str, Any]] = {}
    for case in cases:
        print(f"  {case.case_id}: {case.query[:60]}")
        if case.is_negative:
            # Negative cases: skip retrieval seeding; expected remains empty.
            print(f"    (negative case — skipping retrieval seed)")
            seeded_by_id[case.case_id] = {
                "case_id": case.case_id,
                "query": case.query,
                "notes": case.notes,
                "expected_chunk_ids": [],
                "top_k": case.top_k,
                "intent": case.intent,
                "is_negative": True,
            }
            continue
        ids = _run_retrieval(
            case,
            db_dsn=db_dsn,
            embed_backend=embed_backend,
            embed_model_name=embed_model_name,
        )
        seeded_by_id[case.case_id] = {
            "case_id": case.case_id,
            "query": case.query,
            "notes": case.notes,
            "expected_chunk_ids": ids,
            "top_k": case.top_k,
            "intent": case.intent,
        }

    if existing_cases is not None:
        # Merge: update seeded cases in the existing list (preserving order).
        existing_by_id = {c["case_id"]: c for c in existing_cases}
        existing_by_id.update(seeded_by_id)
        # Preserve original order, then append any new cases.
        seen: set = set()
        merged: List[Dict[str, Any]] = []
        for orig in existing_cases:
            cid = orig["case_id"]
            merged.append(existing_by_id[cid])
            seen.add(cid)
        for cid, entry in seeded_by_id.items():
            if cid not in seen:
                merged.append(entry)
        built = merged
    else:
        built = list(seeded_by_id.values())

    dataset = {
        "version": "1.0",
        "description": (
            "Snapshot of top-k chunk IDs per query.  "
            "Validate expected_chunk_ids manually before using for regression."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases": built,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"Dataset written to: {out_path} ({len(built)} total cases, {len(seeded_by_id)} seeded)")
    print("IMPORTANT: Review expected_chunk_ids before using as ground truth.")


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recall@k / Precision@k / MRR harness")
    p.add_argument("--dataset", type=Path, help="Path to JSON dataset file")
    p.add_argument("--ids", nargs="+", help="Restrict to specific case IDs")
    p.add_argument("--top-k", type=int, default=None, help="Override top_k for all cases")
    p.add_argument("--db-dsn", default=DEFAULT_DB_DSN)
    p.add_argument("--embed-backend", default=DEFAULT_EMBED_BACKEND)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL_NAME)
    p.add_argument("--build-dataset", action="store_true",
                   help="Run retrieval and write top-k IDs as expected IDs to --out")
    p.add_argument("--out", type=Path, help="Write JSON report (eval) or dataset (--build-dataset)")
    p.add_argument("--verbose", action="store_true", help="Print retrieved IDs per case")
    p.add_argument("--compare-last", action="store_true",
                   help="Compare this run against the last saved history snapshot")
    p.add_argument("--save-history", action="store_true",
                   help="Save this run as a timestamped snapshot in data/diagnostics/eval_history/")
    p.add_argument("--dedup-check", action="store_true",
                   help="After eval, scan expected_chunk_ids for near-duplicate chunks in the DB")
    p.add_argument("--dedup-threshold", type=float, default=0.85,
                   help="Jaccard similarity threshold for --dedup-check (default: 0.85)")
    p.add_argument("--bench", type=int, default=None, metavar="N",
                   help="Run each query N times and report per-query p50/p95 latency (ms)")
    p.add_argument("--split", choices=["dev", "test", "all"], default="all",
                   help="Filter cases by split label: dev, test, or all (default: all)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Load cases
    if args.dataset and args.dataset.exists():
        raw = json.loads(args.dataset.read_text(encoding="utf-8"))
        case_dicts = raw.get("cases", [])
    else:
        if args.dataset:
            print(f"Dataset file not found: {args.dataset} — using built-in cases")
        case_dicts = _DEFAULT_CASES

    cases = [
        RecallCase(
            case_id=c["case_id"],
            query=c["query"],
            notes=c.get("notes", ""),
            expected_chunk_ids=c.get("expected_chunk_ids") or [],
            top_k=args.top_k or c.get("top_k", 10),
            intent=c.get("intent", ""),
            is_negative=c.get("is_negative", False),
            split=c.get("split", "dev"),
            difficulty=c.get("difficulty", ""),
            chunk_relevance=c.get("chunk_relevance") or {},
        )
        for c in case_dicts
    ]

    if args.ids:
        cases = [c for c in cases if c.case_id in args.ids]
        if not cases:
            print(f"No cases matched IDs: {args.ids}")
            sys.exit(1)

    if args.split != "all":
        cases = [c for c in cases if c.split == args.split]
        if not cases:
            print(f"No cases found with split={args.split!r}")
            sys.exit(1)

    if args.build_dataset:
        out = args.out or Path("data/qa/retrieval_recall_dataset.json")
        # When --ids narrows the build but --dataset points to a full dataset,
        # merge seeded results back into the existing dataset so curated cases
        # in other cases are not lost.
        existing_cases: Optional[List[Dict[str, Any]]] = None
        if args.ids and args.dataset and args.dataset.exists():
            existing_cases = json.loads(args.dataset.read_text(encoding="utf-8")).get("cases", [])
        _build_dataset(
            cases,
            db_dsn=args.db_dsn,
            embed_backend=args.embed_backend,
            embed_model_name=args.embed_model,
            out_path=out,
            existing_cases=existing_cases,
        )
        return

    # Evaluate
    metrics_list: List[CaseMetrics] = []
    for case in cases:
        print(f"  {case.case_id}: {case.query[:60]}")
        ids = _run_retrieval(
            case,
            db_dsn=args.db_dsn,
            embed_backend=args.embed_backend,
            embed_model_name=args.embed_model,
        )
        m = _compute_metrics(case, ids)
        metrics_list.append(m)
        if args.verbose:
            print(f"    retrieved: {ids[:5]} …")

    agg = _aggregate(metrics_list)
    _print_results(metrics_list, agg)

    if args.save_history or args.compare_last:
        hist_path = _save_history(agg, metrics_list)
        print(f"  History snapshot saved: {hist_path.relative_to(PROJECT_ROOT)}")

    if args.compare_last:
        # Load the snapshot written *before* this run (i.e., second-to-last)
        snapshots = sorted(_HISTORY_DIR.glob("run_*.json"))
        if len(snapshots) >= 2:
            prev = json.loads(snapshots[-2].read_text(encoding="utf-8"))
            _print_regression_delta(agg, prev)
        else:
            print("  (No previous run to compare against — this is the first snapshot.)")

    if args.out:
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": agg,
            "cases": [asdict(m) for m in metrics_list],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Report written to: {args.out}")

    if args.dedup_check:
        _dedup_check(cases, db_dsn=args.db_dsn, threshold=args.dedup_threshold)

    if args.bench:
        _run_bench(
            cases,
            n=args.bench,
            db_dsn=args.db_dsn,
            embed_backend=args.embed_backend,
            embed_model_name=args.embed_model,
        )


def _run_bench(
    cases: List["RecallCase"],
    n: int,
    *,
    db_dsn: str,
    embed_backend: str,
    embed_model_name: str,
) -> None:
    """Run each case N times, record latency, and print p50/p95 summary."""
    print(f"\n── Latency benchmark ({n} runs per query) ────────────────")
    header = f"  {'ID':<6} {'p50 ms':>8} {'p95 ms':>8}  Query"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_latencies: List[float] = []
    for case in cases:
        latencies: List[float] = []
        for _ in range(n):
            t0 = time.perf_counter()
            _run_retrieval(
                case,
                db_dsn=db_dsn,
                embed_backend=embed_backend,
                embed_model_name=embed_model_name,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
        p50 = statistics.median(latencies)
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies)
        q = case.query[:45] + ("\u2026" if len(case.query) > 45 else "")
        print(f"  {case.case_id:<6} {p50:>8.1f} {p95:>8.1f}  {q}")
        all_latencies.extend(latencies)

    if all_latencies:
        overall_p50 = statistics.median(all_latencies)
        overall_p95 = statistics.quantiles(all_latencies, n=20)[18] if len(all_latencies) >= 20 else max(all_latencies)
        print()
        print(f"  Overall p50: {overall_p50:.1f} ms")
        print(f"  Overall p95: {overall_p95:.1f} ms")
    print()


if __name__ == "__main__":
    main()
