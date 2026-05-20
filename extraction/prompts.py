"""Extraction prompt templates.

All LLM prompts for the extraction pipeline live here as plain functions
that return (system_prompt, user_prompt) tuples.

No LLM calls are made in this module — it is pure string formatting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from extraction.branch_config import BranchConfig
from extraction.batch import (
    ScanBatch,
    ChunkEntry,
    SynthBatch,
    format_scan_batch_text,
    format_synthesis_batch_text,
)


# ---------------------------------------------------------------------------
# Scan pass
# ---------------------------------------------------------------------------

def build_scan_prompts(
    batch: ScanBatch,
    branch: BranchConfig,
    id_prefix: str = "",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for a scan-pass batch.

    The scan pass asks the LLM to identify which chunk IDs are relevant to
    the branch topic.  It should NOT extract content — just return IDs.
    """
    system_prompt = (
        "You are a document analysis assistant. Your only job is to identify which of "
        "the provided document chunks are relevant to the user's stated topic.\n"
        "Do NOT summarize or extract content. Return ONLY a comma-separated list of "
        "chunk IDs that are relevant, in the format: [1], [4], [7]\n"
        "If no chunks are relevant, return exactly: NONE\n"
        "Do not select chunks that are clearly bibliography entries, reference lists, "
        "or book index pages — these contain no substantive content for extraction."
    )

    if branch.mode == "keyword":
        kw_list = ", ".join(branch.keywords) if branch.keywords else "(no keywords provided)"
        keyword_instruction = f"Look specifically for content mentioning: {kw_list}"
        topic_line = f"Topic: {branch.name}"
    else:
        keyword_instruction = "Look for content semantically related to the topic description below."
        topic_line = f"Topic description: {branch.topic_description or branch.name}"

    chunk_directory = format_scan_batch_text(batch, id_prefix=id_prefix)

    context_line = f"\nAdditional context: {branch.prompt_context}" if branch.prompt_context else ""

    user_prompt = (
        f"{topic_line}\n"
        f"{keyword_instruction}{context_line}\n\n"
        f"Chunks:\n{chunk_directory}\n\n"
        "Which chunk IDs contain information relevant to the topic above?\n"
        "Return format: [1], [4], [7]  — or NONE if none are relevant."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Synthesis pass
# ---------------------------------------------------------------------------


def build_synthesis_prompts(
    batch: List[Tuple[int, ChunkEntry]],
    branch: BranchConfig,
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for a synthesis-pass subbatch.

    The synthesis pass selects which chunk IDs should be included verbatim
    as extracted items. It does NOT generate any text — the LLM only returns
    IDs, and the caller rehydrates those IDs to produce verbatim chunk text.
    """
    system_prompt = (
        "You are a document extraction assistant. Your only job is to decide "
        "which of the provided document chunks should be included as extracted "
        "items for the given topic.\n"
        "Do NOT summarize, paraphrase, or write any content. "
        "Return ONLY a comma-separated list of chunk IDs in the format: [1], [4], [7]\n"
        "If no chunks should be included, return exactly: NONE"
    )

    chunk_text = format_synthesis_batch_text(batch)

    if branch.mode == "keyword":
        kw_list = ", ".join(branch.keywords) if branch.keywords else ""
        focus_line = f"\nFocus on content mentioning: {kw_list}" if kw_list else ""
        topic_line = f"Topic: {branch.name}"
    else:
        topic_line = f"Topic description: {branch.topic_description or branch.name}"
        focus_line = ""

    context_line = f"\nAdditional context: {branch.prompt_context}" if branch.prompt_context else ""

    user_prompt = (
        f"{topic_line}{focus_line}{context_line}\n\n"
        f"Chunks:\n{chunk_text}\n\n"
        "Which chunk IDs should be included as extracted items for this topic?\n"
        "Return format: [1], [4], [7]  — or NONE if none should be included."
    )

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# Branch suggestion
# ---------------------------------------------------------------------------

def build_suggest_prompts(
    sample_text: str,
    doc_type: str = "",
    title: str = "",
) -> Tuple[str, str]:
    """Return (system_prompt, user_prompt) for the branch suggestion call."""
    system_prompt = (
        "You are a document analysis assistant. Analyze the provided document sample "
        "and suggest extraction branches that would help a user understand the scope "
        "and requirements of this document.\n\n"
        "Return a JSON array only — no explanation, no markdown fences. "
        "Each element must have exactly these keys: "
        '"name", "description", "mode" ("keyword" or "semantic"), '
        '"sample_keywords" (array, keyword mode only — can be empty for semantic).'
    )

    title_line = f"Document title: {title}\n" if title else ""
    type_line = f"Document type: {doc_type}\n" if doc_type else ""

    user_prompt = (
        f"{title_line}"
        f"{type_line}\n"
        "Here is a representative sample of the document:\n"
        "---\n"
        f"{sample_text}\n"
        "---\n\n"
        "Suggest 3–6 extraction branches. For each branch provide:\n"
        "- name: short branch name\n"
        "- description: one sentence describing what it captures\n"
        "- mode: 'keyword' if specific technical terms are likely, 'semantic' otherwise\n"
        "- sample_keywords: (keyword mode) 3–5 example keywords; empty array for semantic\n\n"
        "Return as a JSON array."
    )

    return system_prompt, user_prompt
