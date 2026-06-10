"""Inspect PDF pages using native PyMuPDF/PDFPlumber signals and compare to heuristics.

This is a diagnostic script for deciding whether we can rely more heavily on
document-native structure before falling back to artifact heuristics.

Example
-------
python scripts/ops/inspect_pdf_page_types.py --pdf path/to/file.pdf
python scripts/ops/inspect_pdf_page_types.py --pdf path/to/file.pdf --json-out page_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import fitz  # type: ignore
import pdfplumber  # type: ignore

from pipeline.extract.page_classifier import classify_all_pages_with_details
from pipeline.ingest_v3 import (
    _looks_like_appendix_heading_page,
    _looks_like_form_page,
    _looks_like_table_of_contents_page,
)


def _page_widgets(page: "fitz.Page") -> List[Any]:
    try:
        widgets = page.widgets()
        if widgets is None:
            return []
        return list(widgets)
    except Exception:
        return []


def _first_nonempty_lines(text: str, limit: int = 6) -> List[str]:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    return lines[:limit]


def _native_page_guess(
    *,
    widget_count: int,
    table_count: int,
    table_area_ratio: float,
    image_count: int,
    text_len: int,
    toc_like: bool,
    appendix_heading_like: bool,
) -> str:
    """Prefer native structure first, then a couple of low-risk text-only suppressions."""
    if toc_like:
        return "toc_text"
    if appendix_heading_like:
        return "appendix_heading"
    if widget_count > 0:
        return "form_widget"
    if table_count > 0 and table_area_ratio >= 0.15:
        return "table_heavy"
    if image_count > 0 and text_len < 200:
        return "image_sparse_text"
    return "standard"


def _current_artifact_guess(
    *,
    classified_page_type: str,
    text: str,
    image_count: int,
    text_len: int,
    toc_like: bool,
    appendix_heading_like: bool,
) -> str:
    """Mirror the current high-level artifact routing decision as closely as possible."""
    if toc_like or appendix_heading_like:
        return "skip"
    image_dominant = image_count > 0 and text_len < 350
    if classified_page_type == "table_heavy" and not image_dominant:
        return "table"
    if _looks_like_form_page(text):
        return "form"
    if image_count > 0 and text_len < 200:
        return "image"
    return "skip"


def inspect_pdf(pdf_path: Path) -> Dict[str, Any]:
    page_analysis = classify_all_pages_with_details(pdf_path)
    page_reports: List[Dict[str, Any]] = []

    with fitz.open(str(pdf_path)) as doc:
        with pdfplumber.open(str(pdf_path)) as plumber_doc:
            for page_idx in range(len(doc)):
                fitz_page = doc[page_idx]
                plumber_page = plumber_doc.pages[page_idx]
                text = fitz_page.get_text("text") or ""
                clean_text = " ".join(text.split())
                text_len = len(clean_text)
                image_count = len(fitz_page.get_images(full=True) or [])
                widgets = _page_widgets(fitz_page)

                try:
                    tables = list(plumber_page.find_tables() or [])
                except Exception:
                    tables = []

                pw = float(plumber_page.width or 1.0)
                ph = float(plumber_page.height or 1.0)
                page_area = max(pw * ph, 1.0)
                table_area = 0.0
                for table in tables:
                    bbox = getattr(table, "bbox", None)
                    if bbox is None:
                        continue
                    table_area += abs(float(bbox[2]) - float(bbox[0])) * abs(float(bbox[3]) - float(bbox[1]))
                table_area_ratio = table_area / page_area

                toc_like = _looks_like_table_of_contents_page(text)
                appendix_heading_like = _looks_like_appendix_heading_page(text)
                form_like = _looks_like_form_page(text)
                classified = str((page_analysis.get(page_idx) or {}).get("page_type") or "standard")

                native_guess = _native_page_guess(
                    widget_count=len(widgets),
                    table_count=len(tables),
                    table_area_ratio=table_area_ratio,
                    image_count=image_count,
                    text_len=text_len,
                    toc_like=toc_like,
                    appendix_heading_like=appendix_heading_like,
                )
                current_guess = _current_artifact_guess(
                    classified_page_type=classified,
                    text=text,
                    image_count=image_count,
                    text_len=text_len,
                    toc_like=toc_like,
                    appendix_heading_like=appendix_heading_like,
                )

                page_reports.append({
                    "page_index": page_idx,
                    "page_number": page_idx + 1,
                    "classified_page_type": classified,
                    "native_guess": native_guess,
                    "current_artifact_guess": current_guess,
                    "widget_count": len(widgets),
                    "table_count": len(tables),
                    "table_area_ratio": round(table_area_ratio, 4),
                    "image_count": image_count,
                    "text_length": text_len,
                    "toc_like": toc_like,
                    "appendix_heading_like": appendix_heading_like,
                    "form_like_heuristic": form_like,
                    "first_lines": _first_nonempty_lines(text),
                })

    summary = {
        "pdf": str(pdf_path),
        "page_count": len(page_reports),
        "native_guess_counts": _count_values(page_reports, "native_guess"),
        "current_artifact_guess_counts": _count_values(page_reports, "current_artifact_guess"),
        "toc_like_pages": [row["page_number"] for row in page_reports if row["toc_like"]],
        "appendix_heading_pages": [row["page_number"] for row in page_reports if row["appendix_heading_like"]],
        "widget_form_pages": [row["page_number"] for row in page_reports if row["widget_count"] > 0],
    }
    return {"summary": summary, "pages": page_reports}


def _count_values(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _print_report(report: Dict[str, Any]) -> None:
    summary = report["summary"]
    print(f"PDF: {summary['pdf']}")
    print(f"Pages: {summary['page_count']}")
    print()
    print("Native guess counts:")
    for key, value in sorted(summary["native_guess_counts"].items()):
        print(f"  - {key}: {value}")
    print("Current artifact guess counts:")
    for key, value in sorted(summary["current_artifact_guess_counts"].items()):
        print(f"  - {key}: {value}")
    if summary["toc_like_pages"]:
        print(f"TOC-like pages: {summary['toc_like_pages']}")
    if summary["appendix_heading_pages"]:
        print(f"Appendix heading pages: {summary['appendix_heading_pages']}")
    if summary["widget_form_pages"]:
        print(f"Widget/form pages: {summary['widget_form_pages']}")
    print()

    print("Per-page detail:")
    for row in report["pages"]:
        first_line = row["first_lines"][0] if row["first_lines"] else ""
        print(
            f"  p.{row['page_number']:>3} | class={row['classified_page_type']:<11} "
            f"| native={row['native_guess']:<17} | current={row['current_artifact_guess']:<8} "
            f"| widgets={row['widget_count']:<2} tables={row['table_count']:<2} "
            f"| table_area={row['table_area_ratio']:<6} | images={row['image_count']:<2} "
            f"| text={row['text_length']:<4} | {first_line[:90]}"
        )


def main() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(description="Inspect PDF pages using native PDF signals.")
    parser.add_argument("--pdf", required=True, help="Path to the PDF file to inspect.")
    parser.add_argument("--json-out", help="Optional path to write the full JSON report.")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    report = inspect_pdf(pdf_path)
    _print_report(report)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"Wrote JSON report to: {out_path}")


if __name__ == "__main__":
    main()
