"""
PDFPlumber table extractor for table-heavy pages in pipeline2.

Extracts all tables on a page and converts them to markdown tables.
No ML model is required — PDFPlumber is already a project dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)


def _cell(value: Optional[object]) -> str:
    """Normalise a table cell value to a clean string."""
    return " ".join(str(value or "").replace("\t", " ").split()).strip()


def _table_to_markdown(rows: List[List]) -> str:
    """Convert a 2-D list of cell values to a markdown table string."""
    if not rows:
        return ""

    width = max((len(r) for r in rows), default=0)
    if width == 0:
        return ""

    # Pad rows to uniform width
    padded = [list(r) + [""] * (width - len(r)) for r in rows]

    header  = padded[0]
    divider = ["---"] * width
    lines   = [
        "| " + " | ".join(_cell(c) for c in header)  + " |",
        "| " + " | ".join(divider)                    + " |",
    ]
    for row in padded[1:]:
        lines.append("| " + " | ".join(_cell(c) for c in row) + " |")
    return "\n".join(lines)


def extract_with_pdfplumber(pdf_path: Path, page_nums: Iterable[int]) -> Dict[int, str]:
    """
    Extract tables from *page_nums* of *pdf_path* using PDFPlumber.

    Parameters
    ----------
    pdf_path  : Path
        Path to the PDF file.
    page_nums : iterable of int
        0-based page indices to process.

    Returns
    -------
    dict
        Mapping of page index → markdown string containing one or more
        markdown tables.  Pages where extraction fails or yields no tables
        are stored as empty strings (Marker's output is kept as fallback).
    """
    page_set = set(page_nums)
    if not page_set:
        return {}

    results: Dict[int, str] = {}

    try:
        with pdfplumber.open(str(pdf_path)) as doc:
            for pn in sorted(page_set):
                if pn >= len(doc.pages):
                    logger.warning(
                        "pdfplumber: page %d out of range for %s", pn, pdf_path.name
                    )
                    results[pn] = ""
                    continue
                try:
                    tables   = doc.pages[pn].extract_tables()
                    md_parts = [_table_to_markdown(t) for t in (tables or []) if t]
                    md_parts = [p for p in md_parts if p]
                    results[pn] = "\n\n".join(md_parts)
                    logger.debug(
                        "pdfplumber page %d: %d table(s)", pn, len(md_parts)
                    )
                except Exception as exc:
                    logger.warning(
                        "pdfplumber failed on page %d of %s: %s — keeping Marker output",
                        pn, pdf_path.name, exc,
                    )
                    results[pn] = ""
    except Exception as exc:
        logger.error("pdfplumber extraction failed for %s: %s", pdf_path, exc)

    return results
