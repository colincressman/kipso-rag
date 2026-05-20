"""Interactive CLI for annotating chunk relevance in the retrieval eval dataset.

For each query case (or a filtered subset), fetches the expected chunk texts
from the database and prompts the annotator for a 0/1/2 relevance rating.
Ratings are written back to the dataset JSON under ``chunk_relevance``.

Relevance scale
---------------
  0 = not relevant  — chunk does not address the query
  1 = partial       — chunk is related but does not directly answer
  2 = fully relevant — chunk directly answers the query

Usage
-----
# Annotate all unannotated cases (those with no chunk_relevance entries):
python scripts/eval/annotate.py --dataset data/qa/retrieval_recall_dataset.json

# Annotate specific cases:
python scripts/eval/annotate.py --dataset data/qa/retrieval_recall_dataset.json --ids R1 R2

# Re-annotate even if already rated (review / correction mode):
python scripts/eval/annotate.py --dataset data/qa/retrieval_recall_dataset.json --reannotate

# Annotate only dev-split cases:
python scripts/eval/annotate.py --dataset data/qa/retrieval_recall_dataset.json --split dev
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.runtime_defaults import DEFAULT_DB_DSN

_DATASET_DEFAULT = PROJECT_ROOT / "data" / "qa" / "retrieval_recall_dataset.json"
_LINE_WIDTH = 80


# ── DB helpers ────────────────────────────────────────────────────────────────

def _fetch_chunk_texts(chunk_ids: List[str], db_dsn: str) -> Dict[str, str]:
    """Return {chunk_id: text} for the given chunk IDs using psycopg (v3)."""
    if not chunk_ids:
        return {}
    try:
        import psycopg  # type: ignore
    except ImportError:
        print("  [annotate] psycopg not available — cannot fetch chunk text.")
        return {}

    texts: Dict[str, str] = {}
    try:
        with psycopg.connect(db_dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, text FROM chunks WHERE chunk_id = ANY(%s)",
                    (chunk_ids,),
                )
                for row in cur.fetchall():
                    texts[str(row[0])] = row[1] or ""
    except Exception as exc:
        print(f"  [annotate] DB error: {exc}")
    return texts


# ── Display helpers ───────────────────────────────────────────────────────────

def _divider(char: str = "─", width: int = _LINE_WIDTH) -> None:
    print(char * width)


def _wrap(text: str, indent: str = "  ") -> None:
    for line in textwrap.wrap(text, width=_LINE_WIDTH - len(indent)):
        print(indent + line)


def _prompt_rating(chunk_id: str, text: str, current: Optional[int]) -> Optional[int]:
    """Show chunk text and prompt for 0/1/2 rating.

    Returns the rating (int 0–2), or None if the annotator skips ("s") or
    quits ("q").  Raises SystemExit on "q".
    """
    _divider()
    short_id = chunk_id[-40:] if len(chunk_id) > 40 else chunk_id
    print(f"  Chunk: …{short_id}")
    if current is not None:
        print(f"  Current rating: {current}")
    _divider("-")
    preview = text[:600].strip()
    if len(text) > 600:
        preview += " …"
    _wrap(preview)
    _divider("-")
    while True:
        try:
            raw = input("  Rate [0=not relevant / 1=partial / 2=full / s=skip / q=quit]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if raw == "q":
            raise SystemExit(0)
        if raw == "s":
            return None
        if raw in ("0", "1", "2"):
            return int(raw)
        print("  Invalid input — enter 0, 1, 2, s, or q.")


# ── Core annotation loop ──────────────────────────────────────────────────────

def _annotate_case(
    case: Dict[str, Any],
    *,
    db_dsn: str,
    reannotate: bool,
) -> bool:
    """Annotate one case.  Returns True if any changes were made."""
    case_id = case["case_id"]
    query = case.get("query", "")
    expected_ids: List[str] = case.get("expected_chunk_ids") or []
    chunk_relevance: Dict[str, int] = case.get("chunk_relevance") or {}

    if not expected_ids:
        print(f"\n  [{case_id}] No expected_chunk_ids — skipping.")
        return False

    # Determine which chunks need annotation
    if reannotate:
        to_annotate = expected_ids
    else:
        to_annotate = [cid for cid in expected_ids if cid not in chunk_relevance]

    if not to_annotate:
        print(f"\n  [{case_id}] All chunks already rated — skipping (use --reannotate to review).")
        return False

    print(f"\n{'=' * _LINE_WIDTH}")
    print(f"  Case: {case_id}  |  {len(to_annotate)} chunk(s) to rate")
    print(f"  Difficulty: {case.get('difficulty', '?')}  |  Split: {case.get('split', '?')}")
    _divider()
    print(f"  Query: {query}")
    notes = case.get("notes", "")
    if notes:
        print(f"  Notes: {notes}")
    print()

    # Fetch chunk texts from DB
    texts = _fetch_chunk_texts(to_annotate, db_dsn)
    if not texts:
        print("  [annotate] Could not fetch chunk texts — skipping case.")
        return False

    changed = False
    for chunk_id in to_annotate:
        text = texts.get(chunk_id, "")
        if not text:
            print(f"  [{chunk_id}] text not found in DB — skipping.")
            continue
        current = chunk_relevance.get(chunk_id)
        rating = _prompt_rating(chunk_id, text, current)
        if rating is not None:
            chunk_relevance[chunk_id] = rating
            changed = True
            print(f"  → Saved {chunk_id}: {rating}")

    case["chunk_relevance"] = chunk_relevance
    return changed


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive relevance annotation for the retrieval eval dataset",
    )
    p.add_argument(
        "--dataset", type=Path, default=_DATASET_DEFAULT,
        help=f"Path to dataset JSON (default: {_DATASET_DEFAULT.relative_to(PROJECT_ROOT)})",
    )
    p.add_argument("--ids", nargs="+", metavar="ID", help="Annotate only specific case IDs")
    p.add_argument(
        "--split", choices=["dev", "test", "all"], default="all",
        help="Filter cases by split label (default: all)",
    )
    p.add_argument(
        "--reannotate", action="store_true",
        help="Re-prompt for all chunks even if already rated",
    )
    p.add_argument("--db-dsn", default=DEFAULT_DB_DSN)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.dataset.exists():
        print(f"Dataset not found: {args.dataset}")
        sys.exit(1)

    data = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases: List[Dict[str, Any]] = data.get("cases", [])

    # Apply filters
    if args.ids:
        id_set = set(args.ids)
        cases_to_run = [c for c in cases if c["case_id"] in id_set]
        if not cases_to_run:
            print(f"No cases matched IDs: {args.ids}")
            sys.exit(1)
    elif args.split != "all":
        cases_to_run = [c for c in cases if c.get("split") == args.split]
        if not cases_to_run:
            print(f"No cases with split={args.split!r}")
            sys.exit(1)
    else:
        cases_to_run = cases

    print(f"Annotation session: {len(cases_to_run)} case(s) from {args.dataset.name}")
    print("Scale: 0=not relevant  1=partial  2=fully relevant")
    print("Commands: s=skip chunk  q=quit and save")

    total_changed = 0
    try:
        for case in cases_to_run:
            changed = _annotate_case(case, db_dsn=args.db_dsn, reannotate=args.reannotate)
            if changed:
                total_changed += 1
                # Save incrementally after each case so progress is not lost
                args.dataset.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
                )
    except SystemExit:
        pass  # quit command — fall through to final save

    # Final save
    args.dataset.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. {total_changed} case(s) updated — saved to {args.dataset}")


if __name__ == "__main__":
    main()
