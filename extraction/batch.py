"""Intelligent batching for the extraction scan pass.

Implements the ID-assignment → batching → rehydration pipeline described in
Section 4 of LargeDocIngest.md.

Public API
----------
assign_ids_and_batch(pool, batch_max_chars, ...) -> (id_map, batches)
    Split a list of scored chunks into scan-pass batches, each with
    short numeric IDs replacing full chunk text.

rehydrate(scan_ids, id_map) -> List[dict]
    Given the short IDs returned by the scan LLM, look up the full chunk dicts.

split_for_synthesis(chunks, synthesis_max_chars) -> List[List[dict]]
    Split the rehydrated chunk list into synthesis sub-batches that fit in
    the synthesis context window.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterator, List, Tuple


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

# One entry in the chunk pool (as produced by retrieval/query.py or
# as raw SQLite rows post-scored).
ChunkEntry = Dict[str, Any]
# Required keys: "chunk_id", "text"
# Optional:      "score", "source_filename", "page_start"

# Short ID → full chunk dict
IdMap = Dict[int, ChunkEntry]

# One scan batch: list of (short_id, preview_text) pairs
ScanBatch = List[Tuple[int, str]]


# ---------------------------------------------------------------------------
# assign_ids_and_batch
# ---------------------------------------------------------------------------

def assign_ids_and_batch(
    pool: List[ChunkEntry],
    batch_max_chars: int = 28000,
    preview_chars: int = 80,
    id_prefix: str = "",
) -> Tuple[IdMap, List[ScanBatch]]:
    """Assign short IDs and split pool into scan-pass batches.

    Parameters
    ----------
    pool : list of chunk dicts, sorted by relevance score (descending).
        Each dict must have "chunk_id" and "text".
    batch_max_chars : maximum cumulative character count of preview text per batch.
        The scan pass receives `preview_chars` chars per chunk as a directory.
    preview_chars : how many chars of each chunk text to include in the scan directory.
    id_prefix : string prefix for IDs (e.g. "B1-").  Leave blank for plain integers.

    Returns
    -------
    id_map : {short_id: chunk_dict}
    batches : list of ScanBatch, each = [(short_id, preview_text), ...]
    """
    id_map: IdMap = {}
    batches: List[ScanBatch] = []
    current_batch: ScanBatch = []
    current_chars = 0

    for i, chunk in enumerate(pool, start=1):
        short_id = i  # plain integer; prefix applied in prompt formatting
        id_map[short_id] = chunk

        preview = _make_preview(chunk.get("text", ""), preview_chars)
        entry_chars = len(preview) + 10  # ~10 chars for "[N] " formatting overhead

        if current_batch and current_chars + entry_chars > batch_max_chars:
            batches.append(current_batch)
            current_batch = []
            current_chars = 0

        current_batch.append((short_id, preview))
        current_chars += entry_chars

    if current_batch:
        batches.append(current_batch)

    return id_map, batches


def _strip_leading_artifacts(text: str) -> str:
    """Strip leading PDF-extraction artifacts from chunk text.

    PDF chunkers often leave stray characters at the start of a chunk when
    a split falls mid-equation, mid-section-label, or mid-word.  Examples:
      'N  Batch gradient descent…'   → single-letter section label
      '( x , y ):  Chain rule…'      → equation fragment with trailing colon
      'h 1 ,    h 1 ,  minimize…'    → math variable cluster

    Strategy: if the first 60 characters contain a run of 2+ spaces and
    everything before that run contains no word of 5+ letters, strip that
    leading junk and repeat (up to 3 passes).
    """
    for _ in range(3):
        m = re.search(r'\s{2,}', text[:60])
        if not m:
            break
        candidate = text[:m.end()]
        if re.search(r'[a-zA-Z]{5,}', candidate):
            break  # real word present — stop stripping
        text = text[m.end():]
    return text


def _make_preview(text: str, max_chars: int) -> str:
    """Return up to *max_chars* of text, stripped to a single line.

    Pass ``max_chars=0`` (or any value <= 0) to return the full line with no
    truncation.  Useful when batching controls the budget instead of preview size.

    Also removes leading PDF-extraction artifacts (stray letters/equations
    that appear when a chunk split landed mid-sentence or mid-label).
    """
    line = text.replace("\n", " ").replace("\r", " ").strip()
    line = _strip_leading_artifacts(line)
    if max_chars <= 0 or len(line) <= max_chars:
        return line
    return line[:max_chars] + "…"


# ---------------------------------------------------------------------------
# rehydrate
# ---------------------------------------------------------------------------

def rehydrate(scan_ids: List[int], id_map: IdMap) -> List[ChunkEntry]:
    """Look up full chunk dicts from scan-pass short IDs.

    Unknown IDs are silently ignored (the LLM sometimes hallucinates IDs).
    Preserves the order in which IDs were returned (usually relevance order).
    Deduplicates: each chunk_id appears at most once.
    """
    seen_chunk_ids: set[str] = set()
    result: List[ChunkEntry] = []
    for sid in scan_ids:
        chunk = id_map.get(sid)
        if chunk is None:
            continue
        cid = chunk.get("chunk_id", "")
        if cid in seen_chunk_ids:
            continue
        seen_chunk_ids.add(cid)
        result.append(chunk)
    return result


# ---------------------------------------------------------------------------
# split_for_synthesis
# ---------------------------------------------------------------------------

def split_for_synthesis(
    chunks: List[ChunkEntry],
    synthesis_max_chars: int = 24000,
) -> List[List[ChunkEntry]]:
    """Split the rehydrated chunk list into synthesis sub-batches.

    Each sub-batch fits within synthesis_max_chars of full text.
    If a single chunk exceeds synthesis_max_chars it is still included
    alone in its own sub-batch (truncation happens in the prompt builder).
    """
    batches: List[List[ChunkEntry]] = []
    current: List[ChunkEntry] = []
    current_chars = 0

    for chunk in chunks:
        text_len = len(chunk.get("text", ""))
        if current and current_chars + text_len > synthesis_max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(chunk)
        current_chars += text_len

    if current:
        batches.append(current)

    return batches if batches else [[]]


# ---------------------------------------------------------------------------
# format_scan_batch_text
# ---------------------------------------------------------------------------

def format_scan_batch_text(batch: ScanBatch, id_prefix: str = "") -> str:
    """Render a scan batch as the chunk directory shown to the LLM.

    Example output:
        [1] The SCADA system shall provide remote monitoring…
        [2] PLC cabinets shall be rated NEMA 4X…
        [3] EtherNet/IP shall be used for control network…
    """
    lines = []
    for short_id, preview in batch:
        lines.append(f"[{id_prefix}{short_id}] {preview}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# parse_scan_response
# ---------------------------------------------------------------------------

def parse_scan_response(response: str) -> List[int]:
    """Extract short IDs from a scan-pass LLM response.

    Strategy: prefer explicit [N] bracketed format (which matches the prompt
    instruction).  Only fall back to bare integers if no bracketed IDs are
    found — this avoids accidentally capturing page numbers, years, or other
    stray integers that appear in free-text responses.

    Handles formats:
      [1], [4], [7]
      [1] [4] [7]
      • [1]\\n• [4]
      NONE  → returns []
    """
    if response.strip().upper() == "NONE":
        return []
    # Prefer explicit [N] format
    bracketed = re.findall(r'\[(\d+)\]', response)
    if bracketed:
        ids = [int(m) for m in bracketed]
    else:
        # Fallback: bare integers only (last resort — prone to false positives)
        ids = [int(m) for m in re.findall(r'\b(\d+)\b', response) if m]
    # Deduplicate while preserving order
    seen: set[int] = set()
    result: List[int] = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# assign_ids_for_synthesis / format_synthesis_batch_text
# ---------------------------------------------------------------------------

# One synthesis subbatch entry: (short_id, full chunk dict)
SynthBatch = List[Tuple[int, ChunkEntry]]


def assign_ids_for_synthesis(
    chunks: List[ChunkEntry],
    synthesis_max_chars: int = 10000,
) -> Tuple[Dict[int, ChunkEntry], List[SynthBatch]]:
    """Assign short IDs to rehydrated chunks and split into subbatches.

    Each subbatch fits within synthesis_max_chars of full chunk text.
    Returns (id_map, subbatches).
    """
    id_map: Dict[int, ChunkEntry] = {i: c for i, c in enumerate(chunks, start=1)}
    batches: List[SynthBatch] = []
    current: SynthBatch = []
    current_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        text_len = len(chunk.get("text", ""))
        if current and current_chars + text_len > synthesis_max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append((i, chunk))
        current_chars += text_len

    if current:
        batches.append(current)

    return id_map, batches if batches else []


def format_synthesis_batch_text(batch: SynthBatch) -> str:
    """Render a synthesis subbatch as labeled full-text chunks for the LLM.

    Example output::

        [1] Backpropagation computes gradients by applying the chain rule...

        [2] The vanishing gradient problem occurs when...
    """
    parts = []
    for short_id, chunk in batch:
        text = _strip_leading_artifacts((chunk.get("text") or "").strip())
        parts.append(f"[{short_id}] {text}")
    return "\n\n".join(parts)
