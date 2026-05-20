"""Shared text-processing utilities."""

from __future__ import annotations

import re
from typing import List

# Matches a 4-digit year in the range 1900–2099.
# Used for temporal-query detection in retrieval and LLM layers.
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Search/BM25 tokenizer: letter-start, length >= 2.
# Used for query tokenization, BM25 scoring, and lexical overlap.
# Digit-start tokens (e.g. "24V", "3D") and single chars are excluded for
# consistency across all scoring modules.
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-\.]+")

# Near-duplicate tokenizer: any alphanum run, length >= 1.
# Used for near-duplicate detection in context packing, where single-char
# tokens (e.g. "A", "B", "2") are meaningful differentiators in short texts.
_WORD_RE = re.compile(r"[A-Za-z0-9_\-\.]+")


def tokenize(text: str) -> List[str]:
    """Return search tokens from *text* (letter-start, length >= 2).

    Lowercase the input first when case-insensitive matching is needed::

        tokenize("The 24V pump uses Hydraulic pressure.".lower())
        # ['the', 'pump', 'uses', 'hydraulic', 'pressure']
    """
    return _TOKEN_RE.findall(text or "")


def tokenize_all(text: str) -> List[str]:
    """Return all word tokens from *text* including single chars and digit-start.

    Use for near-duplicate detection where short distinguishing tokens matter::

        tokenize_all("Topic details A".lower())  # ['Topic', 'details', 'A']
    """
    return _WORD_RE.findall(text or "")


def slugify_path(name: str) -> str:
    """Convert a name to a safe filesystem directory/file slug (underscores).

    Example::

        slugify_path("Controls & I/O Scope")  # "controls_io_scope"
    """
    slug = name.lower()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug or "branch"


def slugify_anchor(text: str) -> str:
    """Convert heading text to a GitHub-style markdown anchor slug (dashes).

    Example::

        slugify_anchor("Controls & I/O Scope")  # "controls--io-scope"
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower())
    return re.sub(r"[^a-z0-9-]", "", slug).strip("-")


def normalize_tokens(text: str, *, max_tokens: int = 120) -> List[str]:
    """Return a truncated list of all-alphanum tokens for near-duplicate detection.

    Uses the broader ``tokenize_all`` tokenizer so single-char and digit-start
    tokens (meaningful in short texts) are preserved.
    """
    toks = tokenize_all((text or "").lower())
    return toks[:max_tokens]


def jaccard(a, b) -> float:
    """Jaccard similarity coefficient between two token iterables."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa.intersection(sb)) / max(1, len(sa.union(sb)))
