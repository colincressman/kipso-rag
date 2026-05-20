"""pipeline2 extraction layer.

Public API
----------
    from pipeline.extract import extract_to_markdown
    markdown: str = extract_to_markdown(Path("paper.pdf"))
"""

from __future__ import annotations

import logging
from pathlib import Path

import fitz  # PyMuPDF — CPU-only, always available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_pymupdf_all_pages(pdf_path: Path) -> dict[int, str]:
    """
    Extract plain text from every page with PyMuPDF.

    Uses ``get_text("markdown")`` which preserves basic structure (headings,
    bold, etc.) where the font metadata allows.  Falls back to plain text for
    pages where the markdown mode fails.

    Returns a dict ``{0-based page index: text}``.  Pages that produce no text
    (fully vector / image) map to empty strings — the caller treats those as
    math-risk candidates.
    """
    result: dict[int, str] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc):
                try:
                    text = page.get_text("markdown")
                except Exception:
                    try:
                        text = page.get_text()
                    except Exception:
                        text = ""
                result[i] = text
    except Exception as exc:
        logger.error("pymupdf fast-pass failed for %s: %s", pdf_path, exc)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_to_markdown(pdf_path: Path) -> str:
    """
    Extract *pdf_path* to a single markdown string using per-page dispatch.

    Dispatch strategy (PyMuPDF-first, surgical Marker)
    ---------------------------------------------------
    1. Fast-extract ALL pages with PyMuPDF (CPU-only, sub-second per book).
    2. Classify all pages (math/code/table/body) using per-page heuristics.
    3. Run Marker only on math-risk + near-empty pages via ``page_range``.
    4. Run PDFPlumber on table pages.
    5. Assemble in page order and return.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file to process.

    Returns
    -------
    str
        Full document markdown.  Returns an empty string if the PDF cannot
        be processed at all.

    Raises
    ------
    FileNotFoundError
        If *pdf_path* does not exist.
    """
    # Heavy imports are lazy so that ``import pipeline.extract`` succeeds
    # in environments where torch/marker/surya are not installed.
    from pipeline.extract.page_classifier  import classify_all_pages
    from pipeline.extract.marker_extractor import extract_with_marker
    from pipeline.extract.table_extractor  import extract_with_pdfplumber
    from pipeline.normalize.to_markdown2   import assemble
    import pipeline.extract.model_manager  as _model_manager

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # ── Stage 1: fast PyMuPDF extraction on all pages (CPU, sub-second) ─────
    logger.info("pipeline/extract: fast PyMuPDF pass — %s", pdf_path.name)
    pymupdf_pages = _extract_pymupdf_all_pages(pdf_path)

    if not pymupdf_pages:
        logger.error("pipeline/extract: PyMuPDF returned nothing for %s", pdf_path.name)
        return ""

    # ── Stage 2: classify pages ──────────────────────────────────────────────
    logger.info("pipeline/extract: classifying pages — %s", pdf_path.name)
    page_types = classify_all_pages(pdf_path)

    if not page_types:
        logger.warning(
            "pipeline/extract: no pages found in %s — falling back to PyMuPDF only",
            pdf_path.name,
        )
        return assemble(pymupdf_pages)

    math_risk_pages = sorted(
        pn for pn, t in page_types.items()
        if t in ("math_heavy", "code_heavy")
    )
    table_pages = sorted(
        pn for pn, t in page_types.items()
        if t == "table_heavy"
    )

    # Pages where PyMuPDF extracted almost nothing are very likely fully
    # vector-drawn or image pages — also send those to Marker.
    empty_pages = sorted(
        pn for pn, text in pymupdf_pages.items()
        if len(text.strip()) < 30
    )
    marker_pages = sorted(set(math_risk_pages) | set(empty_pages))

    n_standard = len(page_types) - len(math_risk_pages) - len(table_pages)
    logger.info(
        "pipeline/extract: %s — %d pages | standard=%d  math/code=%d  table=%d"
        "  near-empty=%d  → Marker: %d",
        pdf_path.name,
        len(page_types),
        n_standard,
        len(math_risk_pages),
        len(table_pages),
        len(empty_pages),
        len(marker_pages),
    )

    # ── Stage 3: start with PyMuPDF output for all pages ────────────────────
    pages: dict[int, str] = dict(pymupdf_pages)

    # ── Stage 4: Marker only on math-risk + near-empty pages ────────────────
    if marker_pages:
        logger.info(
            "pipeline/extract: running Marker on %d page(s) of %s",
            len(marker_pages), pdf_path.name,
        )
        marker_results = extract_with_marker(pdf_path, page_range=marker_pages)
        if not marker_results:
            logger.warning(
                "pipeline/extract: Marker returned nothing for %s — keeping PyMuPDF text",
                pdf_path.name,
            )
        else:
            for pn, md in marker_results.items():
                if md.strip():
                    pages[pn] = md
    else:
        logger.info("pipeline/extract: no math-risk pages detected — Marker skipped")

    # ── Stage 5: PDFPlumber overrides for table pages ────────────────────────
    if table_pages:
        logger.info(
            "pipeline/extract: running PDFPlumber on %d table page(s)", len(table_pages)
        )
        tbl_results = extract_with_pdfplumber(pdf_path, table_pages)
        for pn, tbl_md in tbl_results.items():
            if tbl_md.strip():
                pages[pn] = tbl_md

    # ── Stage 6: assemble and unload GPU models ───────────────────────────────
    _model_manager.unload_all()
    return assemble(pages)
