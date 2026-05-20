"""Near-duplicate detection for extraction output.

Two types of deduplication:
  1. Within-branch: identical or near-identical items from the same branch
     (e.g. the same requirement extracted twice from overlapping chunks).
  2. Cross-branch: items that appear in multiple branches.  The winner is
     the item from the highest-priority source; ties go to the earlier branch.

Both use Jaccard similarity on word tokens — no extra LLM call needed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

from extraction.branch_config import BranchResult, ExtractionItem


# ---------------------------------------------------------------------------
# Token normalization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> Set[str]:
    """Lowercase word tokens from text, ignoring punctuation."""
    return set(_TOKEN_RE.findall(text.lower()))


def jaccard(a: str, b: str) -> float:
    """Jaccard similarity between the word-token sets of two strings."""
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    intersection = len(ta & tb)
    union = len(ta | tb)
    return intersection / union


# ---------------------------------------------------------------------------
# Within-branch deduplication
# ---------------------------------------------------------------------------

def dedup_items(
    items: List[ExtractionItem],
    threshold: float = 0.70,
) -> List[ExtractionItem]:
    """Remove near-duplicate items within a single branch result.

    When two items exceed the Jaccard threshold, the one from the higher-priority
    source (priority_weight) is kept.  Ties go to the earlier item (preserve order).

    Parameters
    ----------
    items : extraction items from a single branch.
    threshold : Jaccard similarity above which two items are considered duplicates.

    Returns
    -------
    Deduplicated list preserving original order of kept items.
    """
    kept: List[ExtractionItem] = []
    for candidate in items:
        is_dup = False
        for i, existing in enumerate(kept):
            sim = jaccard(candidate.text, existing.text)
            if sim >= threshold:
                # Keep the higher-priority source; replace if candidate wins
                if candidate.priority_weight > existing.priority_weight:
                    kept[i] = candidate
                is_dup = True
                break
        if not is_dup:
            kept.append(candidate)
    return kept


# ---------------------------------------------------------------------------
# Cross-branch duplicate detection
# ---------------------------------------------------------------------------

def find_cross_branch_duplicates(
    branch_results: Dict[str, List[ExtractionItem]],
    threshold: float = 0.80,
) -> Dict[str, List[str]]:
    """Find items that appear in more than one branch.

    Parameters
    ----------
    branch_results : {branch_name: [ExtractionItem, ...]}
    threshold : Jaccard similarity threshold for cross-branch match.

    Returns
    -------
    Dict mapping "branch_name::item_index" → [other_branch_names...]
    where each value lists the branches the item also appears in.
    """
    # Build flat index: (branch_name, idx, text)
    index: List[Tuple[str, int, str]] = []
    for branch_name, items in branch_results.items():
        for idx, item in enumerate(items):
            index.append((branch_name, idx, item.text))

    cross_refs: Dict[str, List[str]] = {}
    n = len(index)
    for i in range(n):
        bn_i, idx_i, text_i = index[i]
        key_i = f"{bn_i}::{idx_i}"
        for j in range(i + 1, n):
            bn_j, idx_j, text_j = index[j]
            if bn_i == bn_j:
                continue  # same branch handled by dedup_items
            sim = jaccard(text_i, text_j)
            if sim >= threshold:
                cross_refs.setdefault(key_i, []).append(bn_j)
                cross_refs.setdefault(f"{bn_j}::{idx_j}", []).append(bn_i)

    return cross_refs


# ---------------------------------------------------------------------------
# Cross-branch deduplication with priority winner
# ---------------------------------------------------------------------------

def dedup_cross_branch(
    branch_results: List[BranchResult],
    threshold: float = 0.80,
) -> Tuple[List[BranchResult], List[Dict[str, Any]]]:
    """Remove cross-branch near-duplicates, keeping the highest-priority source.

    When the same content appears in multiple branches (e.g. the same chunk
    selected by both "Controls" and "Safety" branches), the item is retained
    only in the branch whose source has the highest ``priority_weight``.
    Ties go to the earlier branch (preserving branch order).

    Parameters
    ----------
    branch_results : ordered list of BranchResult from all branches.
    threshold      : Jaccard similarity above which items are considered
                     cross-branch duplicates.

    Returns
    -------
    (deduped_results, overrides_log)

    Each ``overrides_log`` entry is a dict with keys:
      winner_branch, loser_branch, winner_text, loser_text,
      similarity, winner_priority
    """
    # Build flat index: (branch_idx, item_idx, item)
    flat: List[Tuple[int, int, ExtractionItem]] = []
    for b_idx, br in enumerate(branch_results):
        for i_idx, item in enumerate(br.items):
            flat.append((b_idx, i_idx, item))

    n = len(flat)
    removed: Set[Tuple[int, int]] = set()
    overrides_log: List[Dict[str, Any]] = []

    for i in range(n):
        b_i, i_i, item_i = flat[i]
        if (b_i, i_i) in removed:
            continue
        for j in range(i + 1, n):
            b_j, i_j, item_j = flat[j]
            if b_j == b_i:
                continue  # same branch — handled by within-branch dedup
            if (b_j, i_j) in removed:
                continue
            sim = jaccard(item_i.text, item_j.text)
            if sim < threshold:
                continue
            # Higher priority_weight wins; ties → earlier branch (i)
            if item_j.priority_weight > item_i.priority_weight:
                removed.add((b_i, i_i))
                overrides_log.append({
                    "winner_branch": branch_results[b_j].branch_name,
                    "loser_branch": branch_results[b_i].branch_name,
                    "winner_text": item_j.text[:120],
                    "loser_text": item_i.text[:120],
                    "similarity": round(sim, 3),
                    "winner_priority": item_j.priority_weight,
                })
                break  # item_i is removed; stop comparing it
            else:
                removed.add((b_j, i_j))
                overrides_log.append({
                    "winner_branch": branch_results[b_i].branch_name,
                    "loser_branch": branch_results[b_j].branch_name,
                    "winner_text": item_i.text[:120],
                    "loser_text": item_j.text[:120],
                    "similarity": round(sim, 3),
                    "winner_priority": item_i.priority_weight,
                })

    # Rebuild BranchResults without removed items
    new_results: List[BranchResult] = []
    for b_idx, br in enumerate(branch_results):
        kept = [
            item
            for i_idx, item in enumerate(br.items)
            if (b_idx, i_idx) not in removed
        ]
        new_results.append(BranchResult(
            branch_name=br.branch_name,
            output_heading=br.output_heading,
            items=kept,
            stats=br.stats,
            status=br.status,
        ))

    return new_results, overrides_log
