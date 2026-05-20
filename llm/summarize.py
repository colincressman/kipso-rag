"""Document-level summary generation — library API.

This module exposes ``generate_doc_summary()`` for use as an ingest-time hook.
It contains the same logic as ``scripts/generate_summaries.py`` but uses
``logging`` instead of ``print()`` so it runs silently inside the pipeline.

``scripts/generate_summaries.py`` imports the core helpers from here so the
implementation lives in one place.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from db.client import fetch_all_chunks_ordered, upsert_summary_chunk
from llm.generation import ollama_chat
from pipeline.embed.embedder import create_embedder
from utils.config import load_yaml_config
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)

log = logging.getLogger(__name__)

# ── Thresholds — loaded from configs/llm.yaml summarize: section ─────────────

_sum_cfg: dict = (load_yaml_config("configs/llm.yaml", default={}) or {}).get("summarize", {})

SINGLE_PASS_THRESHOLD: int   = int(_sum_cfg.get("single_pass_threshold", 60_000))
SEGMENT_CHARS: int            = int(_sum_cfg.get("segment_chars", 15_000))
MAX_SEGMENTS: int             = int(_sum_cfg.get("max_segments", 30))
MAX_CHARS_PER_SEGMENT_PROMPT: int = int(_sum_cfg.get("max_chars_per_segment_prompt", 60_000))
MAX_CHARS_PER_REDUCE_PROMPT: int  = int(_sum_cfg.get("max_chars_per_reduce_prompt", 24_000))
SUMMARY_TEMPERATURE: float    = float(_sum_cfg.get("temperature", 0.3))
SUMMARY_TIMEOUT: float        = float(_sum_cfg.get("timeout_seconds", 240.0))

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_SINGLE = (
    "You are a precise document summarizer. "
    "Write a clear, flowing 3-4 paragraph summary of the document in your own words. "
    "Cover: (1) what the document is about and its main purpose, "
    "(2) the key topics, methods, or arguments it covers, "
    "(3) who the intended audience is or what they will learn. "
    "Do NOT copy bullet points or lists verbatim. "
    "Do NOT include citation tags or source references. "
    "Write in plain prose — clear, direct, and informative."
)

_SYSTEM_SEGMENT = (
    "You are summarizing one section of a larger document. "
    "Write 1-2 concise paragraphs covering the main topics and ideas in this section. "
    "Do NOT copy bullet points verbatim — synthesize into prose. "
    "Be specific: mention key concepts, methods, algorithms, or arguments introduced. "
    "Do not add fluff or meta-commentary like 'this section covers…'."
)

_SYSTEM_REDUCE = (
    "You are combining section-by-section summaries into a cohesive document summary. "
    "Write a clear, flowing 4-5 paragraph summary of the whole document. "
    "Cover: (1) what the document is about and its main purpose, "
    "(2) the full range of topics it covers — not just the beginning, "
    "(3) key methods, frameworks, or arguments, "
    "(4) who the intended audience is and what they will gain. "
    "Write in plain prose. Do NOT use bullet points. "
    "Do NOT include citation tags."
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _llm(system: str, user: str) -> str:
    return ollama_chat(
        model=DEFAULT_LLM_MODEL,
        system_prompt=system,
        user_prompt=user,
        base_url=DEFAULT_LLM_BASE_URL,
        timeout_seconds=SUMMARY_TIMEOUT,
        temperature=SUMMARY_TEMPERATURE,
        think=False,
    )


def split_into_segments(chunks: List[Dict[str, Any]], segment_chars: int, max_segments: int) -> List[List[Dict[str, Any]]]:
    """Divide chunks into at most *max_segments* groups of roughly equal char count."""
    total_chars = sum(len(c["text"]) for c in chunks)
    n_segs = min(max_segments, max(1, round(total_chars / segment_chars)))
    target = total_chars / n_segs

    segments: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    current_chars = 0
    for chunk in chunks:
        current.append(chunk)
        current_chars += len(chunk["text"])
        if current_chars >= target and len(segments) < n_segs - 1:
            segments.append(current)
            current = []
            current_chars = 0
    if current:
        segments.append(current)
    return segments


def segment_prompt(seg_idx: int, n_segs: int, title: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the LLM prompt for one segment.

    If the full segment text exceeds MAX_CHARS_PER_SEGMENT_PROMPT, chunks are
    evenly sampled across the segment so the LLM sees the full breadth rather
    than just the leading portion before the context window truncates.
    """
    total_chars = sum(len(c["text"]) for c in chunks)
    if total_chars > MAX_CHARS_PER_SEGMENT_PROMPT and len(chunks) > 1:
        # How many chunks fit within the cap (approx, using average chunk size)
        avg = total_chars / len(chunks)
        n_sample = max(1, int(MAX_CHARS_PER_SEGMENT_PROMPT / avg))
        # Evenly spaced indices across the segment
        step = len(chunks) / n_sample
        sampled = [chunks[int(i * step)] for i in range(n_sample)]
    else:
        sampled = chunks

    lines = [f"Document: {title}", f"Section {seg_idx + 1} of {n_segs} (sampled {len(sampled)} of {len(chunks)} chunks)\n"]
    for c in sampled:
        role = c.get("structural_role", "body")
        path = c.get("path_text") or ""
        header = f"[{role}] {path}".strip("[] ").strip() or role
        lines.append(f"--- {header} ---")
        lines.append(c["text"])
        lines.append("")
    lines.append(f"Summarize section {seg_idx + 1} of {n_segs} of '{title}':")
    return "\n".join(lines)


def single_pass_prompt(title: str, chunks: List[Dict[str, Any]]) -> str:
    lines = [f"Document: {title}\n"]
    for c in chunks:
        role = c.get("structural_role", "body")
        path = c.get("path_text") or ""
        header = f"[{role}] {path}".strip("[] ").strip() or role
        lines.append(f"--- {header} ---")
        lines.append(c["text"])
        lines.append("")
    lines.append(f"Write a 3-4 paragraph summary of '{title}':")
    return "\n".join(lines)


def reduce_prompt(title: str, seg_summaries: List[str]) -> str:
    lines = [f"Document: {title}", f"({len(seg_summaries)} section summaries below)\n"]
    for i, s in enumerate(seg_summaries, 1):
        lines.append(f"=== Section {i} ===")
        lines.append(s.strip())
        lines.append("")
    lines.append(f"Now write a cohesive 4-5 paragraph summary of the full document '{title}':")
    return "\n".join(lines)


def _pyramid_reduce(title: str, summaries: List[str]) -> str:
    """Reduce a list of segment summaries to a single document summary.

    If the summaries are too large to fit in one reduce call, they are grouped
    into batches and reduced layer by layer (pyramid / hierarchical reduce)
    until the intermediate results are small enough for a single final call.
    """
    layer = 0
    while True:
        total_chars = sum(len(s) for s in summaries)
        if total_chars <= MAX_CHARS_PER_REDUCE_PROMPT:
            return _llm(_SYSTEM_REDUCE, reduce_prompt(title, summaries))
        # Too large — group into batches and reduce each batch first.
        layer += 1
        avg = total_chars / len(summaries)
        batch_size = max(2, int(MAX_CHARS_PER_REDUCE_PROMPT / avg))
        batches = [summaries[i:i + batch_size] for i in range(0, len(summaries), batch_size)]
        log.info(
            "_pyramid_reduce: layer %d — %d summaries → %d batches of ~%d",
            layer, len(summaries), len(batches), batch_size,
        )
        summaries = [_llm(_SYSTEM_REDUCE, reduce_prompt(title, batch)) for batch in batches]


# ── Public API ────────────────────────────────────────────────────────────────

def generate_page_range_summary(
    doc_id: str,
    page_start: int,
    page_end: int,
    db_dsn: str = DEFAULT_DB_DSN,
    *,
    document_title: str = "",
    collection_id: str = "",
    source_name: str = "",
    document_path: str = "",
    source_type: str = "pdf_book",
) -> bool:
    """Generate and store a summary for pages *page_start*–*page_end* of *doc_id*."""
    from db.client import fetch_chunks_in_page_range, upsert_page_range_summary_chunk  # noqa: PLC0415

    title = document_title or doc_id
    range_label = f"Pages {page_start}\u2013{page_end}"
    display_title = f"{title} ({range_label})"

    chunks = fetch_chunks_in_page_range(db_dsn, doc_id, page_start, page_end)
    if not chunks:
        log.warning("generate_page_range_summary: no chunks for doc=%r pages %d-%d", doc_id, page_start, page_end)
        return False

    total_chars = sum(len(c["text"]) for c in chunks)
    log.info("generate_page_range_summary: doc=%r range=%d-%d chunks=%d chars=%d",
             doc_id, page_start, page_end, len(chunks), total_chars)

    t0 = time.perf_counter()
    if total_chars <= SINGLE_PASS_THRESHOLD:
        log.info("generate_page_range_summary: strategy=single-pass")
        summary = _llm(_SYSTEM_SINGLE, single_pass_prompt(display_title, chunks))
    else:
        segments = split_into_segments(chunks, SEGMENT_CHARS, MAX_SEGMENTS)
        log.info("generate_page_range_summary: strategy=map-reduce segments=%d", len(segments))
        seg_summaries = [_llm(_SYSTEM_SEGMENT, segment_prompt(i, len(segments), display_title, seg))
                         for i, seg in enumerate(segments)]
        summary = _pyramid_reduce(display_title, seg_summaries)

    elapsed = time.perf_counter() - t0
    if not summary or len(summary) < 50:
        log.warning("generate_page_range_summary: LLM returned empty/short response in %.1fs", elapsed)
        return False

    embedder = create_embedder(backend=DEFAULT_EMBED_BACKEND, model_name=DEFAULT_EMBED_MODEL_NAME)
    [embedding] = embedder.embed_texts([summary])

    upsert_page_range_summary_chunk(
        db_dsn,
        doc_id=doc_id,
        page_start=page_start,
        page_end=page_end,
        summary_text=summary,
        embedding=embedding,
        collection_id=collection_id,
        document_title=title,
        source_name=source_name or title,
        document_path=document_path,
        source_type=source_type,
    )
    log.info("generate_page_range_summary: stored in %.1fs", elapsed)
    return True


def generate_doc_summary(
    doc_id: str,
    db_dsn: str = DEFAULT_DB_DSN,
    *,
    document_title: str = "",
    collection_id: str = "",
    source_name: str = "",
    document_path: str = "",
    source_type: str = "pdf_book",
) -> bool:
    """Generate and store a summary chunk for *doc_id*.

    Uses ``logging`` instead of ``print()`` — safe to call from within the
    ingest pipeline without polluting stdout.

    Returns ``True`` on success, ``False`` if no chunks were found or the LLM
    returned an empty/too-short response.
    """
    title = document_title or doc_id

    all_chunks = fetch_all_chunks_ordered(db_dsn, doc_id)
    if not all_chunks:
        log.warning("generate_doc_summary: no chunks found for doc_id=%r — skipping", doc_id)
        return False

    total_chars = sum(len(c["text"]) for c in all_chunks)
    log.info("generate_doc_summary: doc_id=%r  chunks=%d  chars=%d", doc_id, len(all_chunks), total_chars)

    t0 = time.perf_counter()

    if total_chars <= SINGLE_PASS_THRESHOLD:
        log.info("generate_doc_summary: strategy=single-pass")
        summary = _llm(_SYSTEM_SINGLE, single_pass_prompt(title, all_chunks))
    else:
        segments = split_into_segments(all_chunks, SEGMENT_CHARS, MAX_SEGMENTS)
        log.info("generate_doc_summary: strategy=map-reduce  segments=%d", len(segments))
        seg_summaries: List[str] = []
        for i, seg in enumerate(segments):
            log.info("generate_doc_summary: map segment %d/%d (%d chunks)", i + 1, len(segments), len(seg))
            seg_summaries.append(_llm(_SYSTEM_SEGMENT, segment_prompt(i, len(segments), title, seg)))
        log.info("generate_doc_summary: reduce stage")
        summary = _pyramid_reduce(title, seg_summaries)

    elapsed = time.perf_counter() - t0

    if not summary or len(summary.strip()) < 50:
        log.warning("generate_doc_summary: LLM returned empty/too-short summary for doc_id=%r", doc_id)
        return False

    log.info("generate_doc_summary: summary=%d chars in %.1fs — embedding…", len(summary), elapsed)

    embedder = create_embedder(backend=DEFAULT_EMBED_BACKEND, model_name=DEFAULT_EMBED_MODEL_NAME)
    [embedding] = embedder.embed_texts([summary])

    upsert_summary_chunk(
        db_dsn,
        doc_id=doc_id,
        summary_text=summary,
        embedding=embedding,
        collection_id=collection_id,
        document_title=title,
        source_name=source_name or title,
        document_path=document_path,
        source_type=source_type,
    )
    log.info("generate_doc_summary: stored summary chunk for doc_id=%r", doc_id)
    return True
