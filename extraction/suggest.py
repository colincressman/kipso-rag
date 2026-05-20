"""Branch auto-suggestion from a document sample.

suggest_branches(document_path, db_dsn, llm_fn, doc_type, title)
    -> List[BranchConfig]

Samples representative chunks from the document, calls the LLM once,
and returns a list of pre-filled BranchConfig objects for the user to
review/edit before running extraction.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from extraction.branch_config import BranchConfig
from extraction.prompts import build_suggest_prompts


# ---------------------------------------------------------------------------
# Sample chunks from an already-ingested document
# ---------------------------------------------------------------------------

def _sample_chunks(
    db_dsn: str,
    collection_id: Optional[str],
    max_chunks: int = 30,
    strategy: str = "uniform",
) -> List[str]:
    """Return a list of chunk texts sampled from the corpus."""
    from db.client import _connect, init_db

    init_db(db_dsn)
    conn = _connect(db_dsn)
    try:
        if collection_id:
            rows = conn.execute(
                """
                SELECT text, structural_role
                FROM chunks
                WHERE collection_id = %s
                  AND structural_role NOT IN ('document_summary', 'toc')
                ORDER BY doc_id, COALESCE(page_start, 0)
                """,
                (collection_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT text, structural_role
                FROM chunks
                WHERE structural_role NOT IN ('document_summary', 'toc')
                ORDER BY doc_id, COALESCE(page_start, 0)
                """,
            ).fetchall()
    finally:
        conn.close()

    texts = [r["text"] for r in rows if r["text"]]

    if not texts:
        return []

    if strategy in ("uniform", "every_nth") or len(texts) <= max_chunks:
        step = max(1, len(texts) // max_chunks)
        return [texts[i] for i in range(0, len(texts), step)][:max_chunks]
    elif strategy == "top_body":
        # Just take the first max_chunks — they're already in page order
        return texts[:max_chunks]
    else:
        step = max(1, len(texts) // max_chunks)
        return [texts[i] for i in range(0, len(texts), step)][:max_chunks]


def _sample_from_file(
    document_path: str,
    max_chars: int = 8000,
) -> str:
    """Fallback: read raw text from a file if not yet ingested."""
    try:
        from pipeline.extract import extract_to_markdown
        text = extract_to_markdown(Path(document_path))
        return text[:max_chars]
    except Exception:
        pass
    try:
        return Path(document_path).read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Parse LLM response into BranchConfig list
# ---------------------------------------------------------------------------

def _parse_suggest_response(raw: str) -> List[Dict[str, Any]]:
    """Extract JSON array from LLM response (handles markdown fences)."""
    # Strip optional ```json ... ``` wrapper
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    # Try to find a [...] array anywhere in the response
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass
    return []


def _suggestions_to_branches(suggestions: List[Dict[str, Any]]) -> List[BranchConfig]:
    """Convert raw suggestion dicts to BranchConfig objects."""
    branches: List[BranchConfig] = []
    for s in suggestions:
        name = s.get("name", "").strip()
        if not name:
            continue
        mode = "semantic" if s.get("mode", "keyword") == "semantic" else "keyword"
        desc = s.get("description", "").strip()
        kws = [k.strip() for k in s.get("sample_keywords", []) if k.strip()]
        branch = BranchConfig(
            name=name,
            mode=mode,
            keywords=kws if mode == "keyword" else [],
            topic_description=desc if mode == "semantic" else "",
            output_heading=name,
        )
        branches.append(branch)
    return branches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def suggest_branches(
    llm_fn: Callable,
    *,
    document_path: Optional[str] = None,
    db_dsn: Optional[str] = None,
    collection_id: Optional[str] = None,
    doc_type: str = "",
    title: str = "",
    max_chunks: int = 30,
    sample_strategy: str = "uniform",
) -> List[BranchConfig]:
    """Auto-suggest extraction branches from a document or collection."""
    from utils.runtime_defaults import DEFAULT_DB_DSN

    actual_db_dsn = db_dsn or DEFAULT_DB_DSN

    # Get sample text — try DB first, then fall back to raw file
    sample_parts: List[str] = []
    if actual_db_dsn and collection_id:
        texts = _sample_chunks(
            actual_db_dsn,
            collection_id=collection_id,
            max_chunks=max_chunks,
            strategy=sample_strategy,
        )
        sample_parts = texts

    # Fall back to raw file if DB had no chunks (not yet ingested)
    if not sample_parts and document_path:
        # Resolve relative paths (strip leading /)
        resolved_path = document_path
        from pathlib import Path as _Path
        if not _Path(document_path).exists():
            stripped = document_path.lstrip("/\\")
            if _Path(stripped).exists():
                resolved_path = stripped
        raw = _sample_from_file(resolved_path)
        sample_parts = [raw] if raw else []

    if not sample_parts:
        return []

    sample_text = "\n\n---\n\n".join(t[:300] for t in sample_parts[:max_chunks])

    # Build and call LLM
    system, user = build_suggest_prompts(sample_text, doc_type=doc_type, title=title)
    try:
        response = llm_fn(
            system_prompt=system,
            user_prompt=user,
            temperature=0.2,
            timeout_seconds=60.0,
        ) or ""
    except Exception:
        return []

    suggestions = _parse_suggest_response(response)
    return _suggestions_to_branches(suggestions)
