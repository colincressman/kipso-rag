"""
Assembles per-page extraction results into a single markdown document.

Input  : dict {page_index: markdown_string} from any combination of
         marker_extractor / math_extractor / table_extractor.
Output : single markdown string with pages joined by double-newlines.
"""

from __future__ import annotations

from typing import Dict


def assemble(pages: Dict[int, str]) -> str:
    """
    Concatenate per-page markdown strings in ascending page order.

    Empty pages (blank or whitespace-only) are skipped.

    Parameters
    ----------
    pages : dict
        Mapping of 0-based page index → markdown string.

    Returns
    -------
    str
        Full document markdown, or empty string if all pages are empty.
    """
    parts = [
        pages[i].strip()
        for i in sorted(pages)
        if pages[i].strip()
    ]
    return "\n\n".join(parts)
