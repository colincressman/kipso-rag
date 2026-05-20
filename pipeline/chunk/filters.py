"""
Chunk quality filters — bibliography detection and low-value chunk rejection.
"""

from __future__ import annotations

import re

_BIBLIO_TITLE_RE = re.compile(
    r'bibliograph|reference[s]?\s*$|works\s+cited|further\s+reading',
    re.IGNORECASE,
)

_ALPHA_TOKEN_RE = re.compile(r'[a-zA-Z]{3,}')


def _section_is_bibliography(title: str, path_text: str) -> bool:
    """Return True when a section's heading indicates a bibliography/references section."""
    return bool(
        _BIBLIO_TITLE_RE.search(title or "")
        or _BIBLIO_TITLE_RE.search(path_text or "")
    )


def _chunk_is_low_value(text: str) -> bool:
    """Return True for chunks that are pure OCR noise or math symbol fragments.

    Criteria: fewer than 8 words total, or fewer than 3 real alphabetic words
    (length ≥ 3). This catches fragments like 'K K', 'O(k), O(kN).', '√n ε δ'.
    """
    words = text.split()
    if len(words) < 4:
        return True
    real_words = _ALPHA_TOKEN_RE.findall(text)
    return len(real_words) < 3
