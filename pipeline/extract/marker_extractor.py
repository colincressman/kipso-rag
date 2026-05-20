"""
Marker extractor for standard pages in pipeline2.

Marker handles complex PDF layouts, multi-column text, and broken font
encodings far better than raw PyMuPDF text extraction.

Selective page extraction
-------------------------
Pass ``page_range`` to limit Marker to specific pages (0-based indices).
This is the critical optimization for math-heavy PDFs: instead of running
full-document OCR, only math-risk pages are sent to Marker.  Clean pages
are handled by the fast PyMuPDF path in the caller.

Page splitting
--------------
Marker v1 returns one large markdown string for the whole document.  We split
on form-feed characters (``\\f``) which Marker emits as page separators.

When ``page_range`` is provided Marker only emits pages in that range, but
the form-feed separators are still relative to the *requested* pages (i.e.
the Nth ``\\f``-separated block corresponds to ``page_range[N]``).

Mutual exclusion
----------------
This module calls ``model_manager.get_marker()``, which will unload pix2tex
first if it happens to be resident.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pipeline.extract import model_manager

logger = logging.getLogger(__name__)


def extract_with_marker(
    pdf_path: Path,
    page_range: Optional[List[int]] = None,
) -> Dict[int, str]:
    """
    Run Marker on *pdf_path* and return a per-page markdown dict.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file.
    page_range : list[int] | None
        0-based page indices to process.  If ``None``, the whole document is
        processed (legacy behaviour).  Pass a sorted list of page numbers to
        restrict Marker to only those pages — this is the primary performance
        optimisation for large math-heavy PDFs.

    Returns
    -------
    dict
        Mapping of 0-based page index → markdown string.
        Keys correspond to the original page numbers in the PDF.
        On failure, returns an empty dict (callers fall back gracefully).
    """
    models = model_manager.get_marker()

    if page_range is not None:
        logger.info(
            "marker: extracting %d page(s) from %s: %s",
            len(page_range), pdf_path.name,
            page_range[:10],  # show up to 10 for log readability
        )
    else:
        logger.info("marker: extracting full document — %s", pdf_path.name)

    config: dict = {}
    if page_range is not None:
        config["page_range"] = page_range

    try:
        # marker >= 1.0: PdfConverter(artifact_dict, config=...)(filepath)
        try:
            from marker.converters.pdf import PdfConverter  # type: ignore[import]
            converter = PdfConverter(artifact_dict=models, config=config)
            rendered = converter(str(pdf_path))
            full_text = rendered.markdown
        except (ImportError, AttributeError):
            # marker < 1.0 fallback (no page_range support in old API)
            from marker.convert import convert_single_pdf  # type: ignore[import]
            full_text, _images, _out_meta = convert_single_pdf(
                str(pdf_path), models, max_pages=None, langs=None, batch_multiplier=1,
            )
    except Exception as exc:
        logger.error("marker extraction failed for %s: %s", pdf_path, exc)
        return {}

    if not full_text:
        logger.warning("marker returned empty output for %s", pdf_path.name)
        return {}

    # Split on form-feed page separators emitted by Marker.
    if "\f" in full_text:
        parts = full_text.split("\f")
        if page_range is not None:
            # Marker emits one block per *requested* page in the order they
            # were requested.  Map each split block back to its real page index.
            result: Dict[int, str] = {}
            for block_idx, text_block in enumerate(parts):
                if block_idx < len(page_range):
                    result[page_range[block_idx]] = text_block.strip()
            return result
        return {i: p.strip() for i, p in enumerate(parts)}

    logger.debug(
        "marker: no form-feed separators found in %s — storing as single block",
        pdf_path.name,
    )
    # No separators: single-page result or entire doc as one block
    if page_range is not None and len(page_range) == 1:
        return {page_range[0]: full_text.strip()}
    return {0: full_text.strip()}
