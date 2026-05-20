"""
pix2tex extractor for math-heavy *and* code-heavy pages in pipeline2.

Code-heavy pages are routed here because we have no dedicated code model.
pix2tex (LatexOCR) produces LaTeX from a rendered page image and is more
useful than PyMuPDF's garbled text on symbol-dense pages regardless of
whether those symbols are equations or code identifiers.

Rendering
---------
Each page is rendered to a PIL image at 300 DPI using PyMuPDF before being
passed to pix2tex.  No text extraction is attempted — the pixel data is all
that matters.

Mutual exclusion
----------------
``model_manager.get_pix2tex()`` unloads Marker first if it is resident.

Installation
------------
pix2tex is not installed by default::

    pip install pix2tex[gui]

The first call to ``get_pix2tex()`` will raise ``ImportError`` with a helpful
message if the package is missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable

import fitz  # PyMuPDF — used for page rendering only
from PIL import Image

from pipeline.extract import model_manager

logger = logging.getLogger(__name__)

_DPI   = 300
_SCALE = _DPI / 72.0   # fitz natively works at 72 DPI


def _render_page(doc: "fitz.Document", page_num: int) -> Image.Image:
    """Render *page_num* of *doc* to a RGB PIL image at ``_DPI`` resolution."""
    page = doc[page_num]
    mat  = fitz.Matrix(_SCALE, _SCALE)
    pix  = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def extract_with_pix2tex(pdf_path: Path, page_nums: Iterable[int]) -> Dict[int, str]:
    """
    Run pix2tex on the specified pages of *pdf_path*.

    Both ``math_heavy`` and ``code_heavy`` pages are dispatched here.

    Parameters
    ----------
    pdf_path  : Path
        Path to the PDF file.
    page_nums : iterable of int
        0-based page indices to process.

    Returns
    -------
    dict
        Mapping of page index → LaTeX string wrapped in a ``$$`` block.
        Pages where pix2tex fails are stored as empty strings (Marker's
        output for those pages remains as the fallback in the caller).
    """
    page_nums = sorted(set(page_nums))
    if not page_nums:
        return {}

    # model_manager unloads Marker before loading pix2tex
    model = model_manager.get_pix2tex()
    results: Dict[int, str] = {}

    try:
        with fitz.open(str(pdf_path)) as doc:
            for pn in page_nums:
                if pn >= len(doc):
                    logger.warning("pix2tex: page %d out of range for %s", pn, pdf_path.name)
                    results[pn] = ""
                    continue
                try:
                    img   = _render_page(doc, pn)
                    latex = model(img)
                    results[pn] = f"$$\n{latex.strip()}\n$$"
                    logger.debug("pix2tex page %d: %d chars of LaTeX", pn, len(latex))
                except Exception as exc:
                    logger.warning(
                        "pix2tex failed on page %d of %s: %s — keeping Marker output",
                        pn, pdf_path.name, exc,
                    )
                    results[pn] = ""   # empty → caller keeps Marker's text
    except Exception as exc:
        logger.error("pix2tex extraction failed for %s: %s", pdf_path, exc)

    return results
