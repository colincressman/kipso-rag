from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg
from psycopg.rows import dict_row

STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "if",
    "then",
    "else",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "into",
    "over",
    "under",
    "than",
    "can",
    "could",
    "should",
    "would",
    "may",
    "might",
    "very",
    "more",
    "most",
    "less",
    "least",
    "not",
    "no",
    "yes",
    "do",
    "does",
    "did",
    "done",
    "have",
    "has",
    "had",
}


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _norm_tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[A-Za-z]{3,}", text.lower()) if t not in STOPWORDS]


def _confidence_band(top_score: float, medium_threshold: float, high_threshold: float) -> str:
    if top_score >= high_threshold:
        return "high"
    if top_score >= medium_threshold:
        return "medium"
    return "low"


def _fetch_cited_text(conn, citations: List[str]) -> str:
    parts: List[str] = []
    for cid in citations:
        row = conn.execute("select text from chunks where chunk_id = %s", (cid,)).fetchone()
        if row and row["text"]:
            parts.append(str(row["text"]))
    return "\n".join(parts)


def _unsupported_counts(answer: str, cited_text: str, sentence_overlap_threshold: float) -> Dict[str, int]:
    source_lower = cited_text.lower()
    weak_sentence_count = 0

    for sent in _split_sentences(answer):
        tokens = _norm_tokens(sent)
        if len(tokens) < 6:
            continue
        overlap = sum(1 for token in tokens if token in source_lower)
        ratio = overlap / len(tokens)
        if ratio < sentence_overlap_threshold:
            weak_sentence_count += 1

    answer_lower = answer.lower()
    unsupported_year_count = 0
    for year in re.findall(r"\b(19\d{2}|20\d{2}|\d{4}s)\b", answer):
        if year.lower() not in source_lower:
            unsupported_year_count += 1

    answer_entities = {
        token
        for token in re.findall(r"\b[A-Z][A-Za-z]{2,}\b", answer)
        if token.lower() not in {"citations", "chapter", "part"}
    }
    unsupported_entity_count = 0
    for entity in answer_entities:
        if entity.lower() not in source_lower:
            unsupported_entity_count += 1

    malformed_citation_count = len(
        re.findall(r"\bc[0-9a-f]{20,}\b|\bc\d{7,}\b|\[ch[a-z0-9_-]{6,}\]", answer, flags=re.IGNORECASE)
    )

    return {
        "weak_sentence_count": weak_sentence_count,
        "unsupported_year_count": unsupported_year_count,
        "unsupported_entity_count": unsupported_entity_count,
        "malformed_citation_count": malformed_citation_count,
    }


def evaluate_grounding(
    *,
    qa_path: Path,
    db_dsn: str,
    medium_threshold: float,
    high_threshold: float,
    sentence_overlap_threshold: float,
) -> Dict[str, object]:
    data = json.loads(qa_path.read_text(encoding="utf-8"))
    conn = psycopg.connect(db_dsn, row_factory=dict_row)

    per_item: List[Dict[str, object]] = []
    band_totals: Dict[str, Counter] = defaultdict(Counter)

    _SAFE_REFUSAL_MODES = {"no_coverage", "formula_not_found"}

    for item in data.get("items", []):
        top_hits = item.get("top_hits") or []
        top_score = float((top_hits[0].get("score") if top_hits else 0.0) or 0.0)
        mode = str(item.get("mode") or "")
        is_safe_refusal = mode in _SAFE_REFUSAL_MODES

        # Safe refusals are honest, grounded non-answers — they are always compliant.
        # Skip the overlap checks entirely; count them in their own summary band.
        if is_safe_refusal:
            band = "refusal"
            counts = {
                "weak_sentence_count": 0,
                "unsupported_year_count": 0,
                "unsupported_entity_count": 0,
                "malformed_citation_count": 0,
            }
            passed = True
        else:
            band = _confidence_band(top_score, medium_threshold, high_threshold)
            answer = str(item.get("answer") or "")
            cited_text = _fetch_cited_text(conn, list(item.get("citations") or []))
            counts = _unsupported_counts(answer, cited_text, sentence_overlap_threshold)
            passed = (
                counts["unsupported_year_count"] == 0
                and counts["malformed_citation_count"] == 0
                and counts["weak_sentence_count"] <= 1
            )

        row = {
            "index": item.get("index"),
            "question": item.get("question"),
            "mode": mode,
            "band": band,
            "top_score": top_score,
            "safe_refusal": is_safe_refusal,
            "pass": passed,
            **counts,
        }
        per_item.append(row)

        band_totals[band]["count"] += 1
        band_totals[band]["pass_count"] += 1 if passed else 0
        for key, value in counts.items():
            band_totals[band][key] += value

    conn.close()

    summary = {}
    for band, totals in band_totals.items():
        count = totals["count"] or 1
        summary[band] = {
            "count": totals["count"],
            "pass_count": totals["pass_count"],
            "pass_rate": round(float(totals["pass_count"]) / float(count), 4),
            "avg_weak_sentence_count": round(float(totals["weak_sentence_count"]) / float(count), 3),
            "avg_unsupported_year_count": round(float(totals["unsupported_year_count"]) / float(count), 3),
            "avg_unsupported_entity_count": round(float(totals["unsupported_entity_count"]) / float(count), 3),
            "avg_malformed_citation_count": round(float(totals["malformed_citation_count"]) / float(count), 3),
        }

    return {
        "qa_path": str(qa_path),
        "db_dsn": db_dsn,
        "thresholds": {
            "medium_confidence_score": medium_threshold,
            "high_confidence_score": high_threshold,
            "sentence_overlap_threshold": sentence_overlap_threshold,
        },
        "summary_by_band": summary,
        "items": per_item,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate grounding quality from exported QA JSON.")
    parser.add_argument("--qa", type=str, default="data/qa/question_battery_answers.json")
    parser.add_argument("--db", type=str, default="postgresql://postgres:postgres@localhost/rag")
    parser.add_argument("--medium-threshold", type=float, default=0.55)
    parser.add_argument("--high-threshold", type=float, default=0.70)
    parser.add_argument("--sentence-overlap-threshold", type=float, default=0.35)
    parser.add_argument("--out", type=str, default="data/qa/grounding_eval.json")
    args = parser.parse_args()

    result = evaluate_grounding(
        qa_path=Path(args.qa),
        db_dsn=args.db,
        medium_threshold=args.medium_threshold,
        high_threshold=args.high_threshold,
        sentence_overlap_threshold=args.sentence_overlap_threshold,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
