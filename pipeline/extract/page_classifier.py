"""
Page-type classifier for pipeline2.

Classifies each PDF page into one of four types:
  - "standard"    : mostly prose → PyMuPDF fast extraction
  - "math_heavy"  : dense math Unicode/PUA glyphs OR vector-drawn symbols → Marker
  - "table_heavy" : significant table area → PDFPlumber
  - "code_heavy"  : monospace-dominant text → Marker

Detection strategies
--------------------
1. Unicode / PUA scan  : catches PDFs that encode math in the text layer using
   non-standard code-points (Greek, letterlike, operator blocks, PUA).
2. Vector drawing scan : catches PDFs where math symbols are drawn as vector
   paths (fill/stroke shapes) rather than encoded as characters.  These pages
   produce completely empty or garbage text from PyMuPDF.

Heuristics use PyMuPDF for character-level font inspection, drawing
inspection, and PDFPlumber for table-area detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import fitz  # PyMuPDF
import pdfplumber

from utils.config import load_yaml_config

logger = logging.getLogger(__name__)

# ── Thresholds — loaded from configs/parsing.yaml page_classifier: section ───
_pc_cfg: dict = (load_yaml_config("configs/parsing.yaml", default={}) or {}).get(
    "page_classifier", {}
)
_TABLE_AREA_RATIO          = float(_pc_cfg.get("table_area_ratio",          0.15))
_MATH_CHAR_RATIO           = float(_pc_cfg.get("math_char_ratio",            0.05))
_MONO_CHAR_RATIO           = float(_pc_cfg.get("mono_char_ratio",            0.40))
_DRAWING_SMALL_AREA_PT     = float(_pc_cfg.get("drawing_small_area_pt",     400))
_DRAWING_DENSITY_RATIO     = float(_pc_cfg.get("drawing_density_ratio",     0.08))
_DRAWING_ABSOLUTE_MIN      = int(  _pc_cfg.get("drawing_absolute_min",       15))
_DRAWING_SPARSE_TEXT_CHARS = int(  _pc_cfg.get("drawing_sparse_text_chars",  80))
_DRAWING_SPARSE_MIN_PATHS  = int(  _pc_cfg.get("drawing_sparse_min_paths",   20))

# ── Unicode ranges that strongly indicate mathematical content ───────────────
_MATH_RANGES: list[tuple[int, int]] = [
    (0x0391, 0x03FF),  # Greek & Coptic letters
    (0x2100, 0x214F),  # Letterlike symbols (ℝ, ℤ, ∞ …)
    (0x2200, 0x22FF),  # Mathematical operators
    (0x2700, 0x27BF),  # Dingbats (some math arrows)
    (0x27C0, 0x27EF),  # Misc mathematical symbols A
    (0x2980, 0x29FF),  # Misc mathematical symbols B
    (0x2A00, 0x2AFF),  # Supplemental mathematical operators
    (0x2070, 0x209F),  # Superscripts and subscripts
    (0xE000, 0xF8FF),  # Private-Use Area (custom math glyphs common in PDFs)
]

# Font name substrings that identify monospace/code fonts
_MONO_FONT_MARKERS = (
    "mono", "courier", "consolas", "typewriter",
    "inconsolata", "code", "fixed", "terminal",
)


def _is_math_char(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in _MATH_RANGES)


def _vector_drawing_math_risk(fitz_page: "fitz.Page", total_text_chars: int) -> bool:
    """
    Return True if the page likely contains vector-drawn math symbols.

    Vector math is invisible to text extraction — symbols are drawn as paths
    (bezier curves, lines, fills) rather than encoded as text glyphs.
    We detect this by counting small drawing paths on pages that have text
    (ruling out fully-scanned image pages, which have zero drawings too).

    Parameters
    ----------
    fitz_page        : already-opened fitz page
    total_text_chars : total characters extracted from this page's text layer
    """
    try:
        drawings = fitz_page.get_drawings()
    except Exception:
        return False

    if not drawings:
        return False

    # Count paths that are symbol-sized (small bbox area)
    small_paths = 0
    for d in drawings:
        rect = d.get("rect")
        if rect is None:
            continue
        w = abs(rect[2] - rect[0])
        h = abs(rect[3] - rect[1])
        if (w * h) < _DRAWING_SMALL_AREA_PT and (w > 0 or h > 0):
            small_paths += 1

    # Near-empty text + many drawings → whole page is vector math
    if total_text_chars < _DRAWING_SPARSE_TEXT_CHARS and small_paths >= _DRAWING_SPARSE_MIN_PATHS:
        return True

    # Dense small drawings relative to text → inline vector math scattered through prose
    if (small_paths >= _DRAWING_ABSOLUTE_MIN
            and total_text_chars > 0
            and (small_paths / total_text_chars) > _DRAWING_DENSITY_RATIO):
        return True

    return False


def _classify_single_page(
    fitz_page: "fitz.Page",
    plumber_page: "pdfplumber.page.Page",
) -> str:
    # ── Table check (PDFPlumber) ─────────────────────────────────────────────
    try:
        tables = plumber_page.find_tables()
        if tables:
            pw = plumber_page.width  or 1.0
            ph = plumber_page.height or 1.0
            page_area = pw * ph
            table_area = sum(
                abs(t.bbox[2] - t.bbox[0]) * abs(t.bbox[3] - t.bbox[1])
                for t in tables
                if t.bbox is not None
            )
            if page_area > 0 and (table_area / page_area) > _TABLE_AREA_RATIO:
                return "table_heavy"
    except Exception as exc:
        logger.debug("PDFPlumber table check failed: %s", exc)

    # ── Text / font analysis (PyMuPDF rawdict) ───────────────────────────────
    try:
        raw = fitz_page.get_text("rawdict")
    except Exception as exc:
        logger.debug("PyMuPDF rawdict failed: %s", exc)
        return "standard"

    total_chars = 0
    math_chars  = 0
    mono_chars  = 0

    for block in raw.get("blocks", []):
        if block.get("type") != 0:   # type 0 = text block
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text: str = span.get("text", "")
                font: str = span.get("font", "").lower()
                n = len(text)
                total_chars += n

                # Math signal: count Unicode math characters
                math_chars += sum(1 for ch in text if _is_math_char(ord(ch)))

                # Code signal: monospace font families
                if any(m in font for m in _MONO_FONT_MARKERS):
                    mono_chars += n

    # ── Vector drawing math check ────────────────────────────────────────────
    if _vector_drawing_math_risk(fitz_page, total_chars):
        return "math_heavy"

    if total_chars == 0:
        return "standard"

    if math_chars / total_chars > _MATH_CHAR_RATIO:
        return "math_heavy"

    if mono_chars / total_chars > _MONO_CHAR_RATIO:
        return "code_heavy"

    return "standard"


def _classify_single_page_with_details(
    fitz_page: "fitz.Page",
    plumber_page: "pdfplumber.page.Page",
) -> Dict[str, Any]:
    table_bboxes: list[list[float]] = []
    page_type = "standard"

    # Table check (PDFPlumber)
    try:
        tables = plumber_page.find_tables()
        if tables:
            pw = plumber_page.width or 1.0
            ph = plumber_page.height or 1.0
            page_area = pw * ph
            table_area = 0.0
            for t in tables:
                bbox = getattr(t, "bbox", None)
                if bbox is None:
                    continue
                table_bboxes.append([float(v) for v in bbox])
                table_area += abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])
            if page_area > 0 and (table_area / page_area) > _TABLE_AREA_RATIO:
                page_type = "table_heavy"
    except Exception as exc:
        logger.debug("PDFPlumber table check failed: %s", exc)

    if page_type == "table_heavy":
        return {"page_type": page_type, "table_bboxes": table_bboxes}

    # Text / font analysis (PyMuPDF rawdict)
    try:
        raw = fitz_page.get_text("rawdict")
    except Exception as exc:
        logger.debug("PyMuPDF rawdict failed: %s", exc)
        return {"page_type": "standard", "table_bboxes": table_bboxes}

    total_chars = 0
    math_chars = 0
    mono_chars = 0

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text: str = span.get("text", "")
                font: str = span.get("font", "").lower()
                n = len(text)
                total_chars += n
                math_chars += sum(1 for ch in text if _is_math_char(ord(ch)))
                if any(m in font for m in _MONO_FONT_MARKERS):
                    mono_chars += n

    if _vector_drawing_math_risk(fitz_page, total_chars):
        page_type = "math_heavy"
    elif total_chars == 0:
        page_type = "standard"
    elif math_chars / total_chars > _MATH_CHAR_RATIO:
        page_type = "math_heavy"
    elif mono_chars / total_chars > _MONO_CHAR_RATIO:
        page_type = "code_heavy"

    return {"page_type": page_type, "table_bboxes": table_bboxes}


def classify_all_pages(pdf_path: Path) -> Dict[int, str]:
    """
    Open *pdf_path* and classify every page.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.

    Returns
    -------
    dict
        Mapping of 0-based page index → page type string
        (``"standard"``, ``"math_heavy"``, ``"table_heavy"``, or ``"code_heavy"``).
        Returns an empty dict if the file cannot be opened.
    """
    details = classify_all_pages_with_details(pdf_path)
    return {
        page_num: str(meta.get("page_type") or "standard")
        for page_num, meta in details.items()
    }


def classify_all_pages_with_details(pdf_path: Path) -> Dict[int, Dict[str, Any]]:
    """
    Return detailed page classification metadata keyed by 0-based page index.

    Each value includes:
    - page_type: standard | math_heavy | table_heavy | code_heavy
    - table_bboxes: list of [x0, top, x1, bottom] table boxes detected by pdfplumber
    """
    result: Dict[int, Dict[str, Any]] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            with pdfplumber.open(str(pdf_path)) as plumber_doc:
                for page_num in range(len(doc)):
                    try:
                        fitz_page = doc[page_num]
                        plumber_page = plumber_doc.pages[page_num]
                        result[page_num] = _classify_single_page_with_details(fitz_page, plumber_page)
                    except Exception as exc:
                        logger.warning(
                            "page %d classification failed, defaulting to standard: %s",
                            page_num, exc,
                        )
                        result[page_num] = {"page_type": "standard", "table_bboxes": []}
    except Exception as exc:
        logger.error("classify_all_pages_with_details failed for %s: %s", pdf_path, exc)
    return result
