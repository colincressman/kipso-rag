"""
Code-heavy page extractor stub for pipeline2.

Code-heavy pages are currently routed to
``pipeline2.extract.math_extractor.extract_with_pix2tex`` — the same path
as math-heavy pages.  This is intentional: we have no dedicated code model,
and pix2tex (a LaTeX OCR model) is more useful than PyMuPDF's garbled output
on pages dominated by monospace or symbol-heavy text.

This module exists as a placeholder for a future dedicated code extractor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable


def extract_code_page(pdf_path: Path, page_nums: Iterable[int]) -> Dict[int, str]:
    """
    Placeholder for a dedicated code-page extractor.

    Not called at runtime — ``code_heavy`` pages are dispatched to
    ``math_extractor.extract_with_pix2tex`` in ``ingest2.py``.
    """
    pass
