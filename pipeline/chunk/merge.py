"""
Merge undersized chunks with their neighbours.

A chunk is considered "small" if its ``token_count_est`` is below
``min_tokens``.  Small chunks are appended to the *preceding* chunk
(if one exists in the same section) or to the *following* chunk.
Merging never crosses section boundaries (``section_id`` must match).

Input / Output: list of chunk dicts as produced by chunker.py.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _tokens(chunk: Dict[str, Any]) -> int:
    return int(chunk.get("token_count_est") or 0)


def merge_small_chunks(
    chunks: List[Dict[str, Any]],
    *,
    min_tokens: int = 40,
    max_tokens_after_merge: int = 520,
) -> List[Dict[str, Any]]:
    """
    Merge chunks below *min_tokens* into an adjacent same-section neighbour.

    Parameters
    ----------
    chunks : list
        Chunk dicts with at minimum ``section_id``, ``text``, and
        ``token_count_est`` keys.
    min_tokens : int
        Chunks with fewer tokens than this are candidates for merging.
    max_tokens_after_merge : int
        A merge is skipped if the combined token count would exceed this.

    Returns
    -------
    list
        New list of chunk dicts.  The merged chunk keeps the ``chunk_id``
        and ``page_start`` of the earlier chunk; ``page_end`` is taken from
        the later chunk.  ``token_count_est`` and ``word_count`` are summed.
    """
    if not chunks:
        return []

    result: List[Dict[str, Any]] = []

    for chunk in chunks:
        if (
            result
            and _tokens(chunk) < min_tokens
            and result[-1].get("section_id") == chunk.get("section_id")
            and _tokens(result[-1]) + _tokens(chunk) <= max_tokens_after_merge
        ):
            # Merge into the previous chunk
            prev = result[-1]
            prev["text"] = (prev["text"].rstrip() + "\n\n" + chunk["text"].lstrip()).strip()
            prev["token_count_est"] = _tokens(prev) + _tokens(chunk)
            prev["word_count"]      = int(prev.get("word_count") or 0) + int(chunk.get("word_count") or 0)
            prev["page_end"]        = chunk.get("page_end", prev.get("page_end"))
        else:
            result.append(dict(chunk))

    # Second pass: try to push any still-small leading chunks forward into
    # the next chunk in the same section (covers the case where there is no
    # preceding chunk to absorb into).
    final: List[Dict[str, Any]] = []
    i = 0
    while i < len(result):
        cur = result[i]
        if (
            _tokens(cur) < min_tokens
            and i + 1 < len(result)
            and result[i + 1].get("section_id") == cur.get("section_id")
            and _tokens(cur) + _tokens(result[i + 1]) <= max_tokens_after_merge
        ):
            nxt = dict(result[i + 1])
            nxt["text"]           = (cur["text"].rstrip() + "\n\n" + nxt["text"].lstrip()).strip()
            nxt["token_count_est"]= _tokens(cur) + _tokens(nxt)
            nxt["word_count"]     = int(cur.get("word_count") or 0) + int(nxt.get("word_count") or 0)
            nxt["chunk_id"]       = cur["chunk_id"]   # keep earlier id
            nxt["page_start"]     = cur.get("page_start", nxt.get("page_start"))
            result[i + 1] = nxt
            i += 1  # skip cur, nxt already updated in result
        else:
            final.append(cur)
            i += 1

    return final
