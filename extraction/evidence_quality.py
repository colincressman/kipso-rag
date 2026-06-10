"""Heuristics for filtering low-information extracted evidence."""

from __future__ import annotations

import re
from typing import Literal


EvidenceClass = Literal[
    "substantive",
    "reference_only",
    "heading_only",
    "appendix_admin",
]


_APPENDIX_PATTERNS = (
    "appendix",
    "table of contents",
    "revision history",
    "good faith efforts",
    "gfe outreach",
)

_REFERENCE_PATTERNS = (
    "refer to",
    "see appendix",
    "see form",
    "refer to dmi",
    "refer to gmi",
)


def strip_html(text: str) -> str:
    """Remove simple HTML tags and normalize whitespace."""
    clean = re.sub(r"<[^>]+>", "", str(text or ""))
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def strip_leading_reference_noise(text: str) -> str:
    """Remove repeated admin/reference wrappers without deleting real content."""
    clean = strip_html(text)
    patterns = (
        r"^\*{0,2}\s*\(?refer to\s+(?:dmi|gmi)\s*50\s*form[^)]*\)?\s*",
        r"^\*{0,2}\s*\[\s*table\s*\]\s*",
        r"^\*{0,2}\s*agenda page \d+\s*",
    )
    changed = True
    while changed:
        changed = False
        for pat in patterns:
            updated = re.sub(pat, "", clean, flags=re.IGNORECASE).strip()
            if updated != clean:
                clean = updated
                changed = True
    return clean


def normalize_evidence_text(text: str) -> str:
    """Normalize evidence text for duplicate detection and heuristics."""
    return strip_leading_reference_noise(text).casefold()


def is_appendix_or_toc_chunk(text: str) -> bool:
    """Return True for chunk text that is clearly appendix/TOC/admin content.

    This is intentionally stricter than ``classify_evidence_text()`` because it
    runs earlier in the extraction pipeline and should only remove chunks that
    are very likely to pollute scan/synthesis passes.
    """
    clean = strip_leading_reference_noise(text)
    norm = clean.casefold()
    if not norm:
        return True

    early_window = norm[:280]
    if "table of contents" in early_window:
        return True
    if "revision history" in early_window:
        return True
    if "appendix" in early_window and len(clean) < 1200:
        return True
    if ("good faith efforts" in early_window or "gfe outreach" in early_window) and len(clean) < 260:
        return True
    if ("refer to dmi" in early_window or "refer to gmi" in early_window) and len(clean) < 260:
        return True
    if re.match(r"^(contents|table of contents)\b", norm):
        return True
    return False


def classify_evidence_text(text: str) -> EvidenceClass:
    """Classify evidence text as substantive or low-information."""
    clean = strip_leading_reference_noise(text)
    norm = clean.casefold()
    if not norm:
        return "heading_only"

    if any(pat in norm for pat in _APPENDIX_PATTERNS):
        if len(norm) < 180:
            return "appendix_admin"

    if any(pat in norm for pat in _REFERENCE_PATTERNS):
        if len(norm) < 220:
            return "reference_only"

    if re.match(r"^\d+(?:\.\d+){1,4}\s*[A-Za-z\"].{0,90}$", clean):
        return "heading_only"

    if len(clean) < 90:
        if re.match(r"^\*{0,2}\d+(?:\.\d+){1,4}", clean):
            return "heading_only"
        if re.match(r"^[A-Z0-9][A-Za-z0-9/&\"().:\- ]{2,80}$", clean) and "." not in clean:
            return "heading_only"

    return "substantive"
