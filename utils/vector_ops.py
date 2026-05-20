"""Vector math utilities shared across retrieval modules.

Canonical implementations — import from here, do not redefine locally.
"""

from __future__ import annotations

import math
from typing import Sequence


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors.

    Returns -1.0 when either vector is empty, lengths differ, or either has
    zero magnitude.  This sentinel (-1.0) is below any real cosine value and
    safe to filter on::

        if cosine(qvec, chunk_vec) <= -1.0:
            continue  # skip unscored / incomparable chunks
    """
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)
