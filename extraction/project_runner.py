"""Project-level extraction orchestrator.

run_project(project_config, emit) -> ProjectRunResult

Orchestrates the full extraction pipeline:
  1. Ingest any un-ingested documents into the RAG collection.
  2. Build CorpusHandle (shared across all branches).
  3. Run each enabled branch via branch_runner.run_branch().
  4. Build and save the markdown report.
  5. Optionally clean up the temporary collection.

The `emit` callback receives progress strings (used by SSE in server.py).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from extraction.branch_config import (
    BranchResult,
    SecondPassResult,
    ProjectConfig,
    SecondPassConfig,
    build_priority_map,
)
from extraction.cancel import CancelCheck, ExtractionCancelled, raise_if_cancelled
from extraction.branch_runner import CorpusHandle, ExtractionConfig, run_branch
from extraction.batch import format_scan_batch_text, parse_scan_response
from extraction.evidence_quality import strip_html
from extraction.report_builder import assemble_report, save_report, save_report_variant
from utils.text_utils import slugify_path as _slugify


# ---------------------------------------------------------------------------
# ProjectRunResult
# ---------------------------------------------------------------------------

@dataclass
class ProjectRunResult:
    project_slug: str
    branch_results: List[BranchResult] = field(default_factory=list)
    second_pass_results: List[SecondPassResult] = field(default_factory=list)
    report_path: Optional[str] = None
    appendix_path: Optional[str] = None
    report_markdown: str = ""
    elapsed_seconds: float = 0.0
    checkpoint_path: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_CHECKPOINT_DIR = "data/extraction_checkpoints"


def _checkpoint_path(slug: str, run_id: str) -> Path:
    return Path(_CHECKPOINT_DIR) / f"{slug}_{run_id}.jsonl"


def _write_checkpoint(path: Path, record_type: str, data: Dict[str, Any]) -> None:
    """Append one record to the run's checkpoint JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"type": record_type, "data": data}) + "\n")


def _vdump(vdir: Path, filename: str, content: str) -> None:
    """Write plain text verbose output to disk."""
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / filename).write_text(content, encoding="utf-8")


def _vdump_json(vdir: Path, filename: str, data: Any) -> None:
    """Write JSON verbose output to disk."""
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / filename).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_checkpoint_branch_results(path: Path) -> List[BranchResult]:
    """Reconstruct ordered BranchResult list (with items) from checkpoint file."""
    if not path.exists():
        return []
    results: List[BranchResult] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
            if record.get("type") == "branch_result":
                results.append(BranchResult.from_dict(record["data"]))
        except Exception:
            pass
    return results


def _seed_checkpoint_with_branch_results(path: Path, branch_results: List[BranchResult]) -> None:
    """Write branch results into a fresh checkpoint file for report-only reruns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    for result in branch_results:
        _write_checkpoint(path, "branch_result", {
            "branch_name": result.branch_name,
            "output_heading": result.output_heading,
            "status": result.status,
            "stats": asdict(result.stats),
            "items": [item.to_dict() for item in result.items],
            "evidence_chunks": list(result.evidence_chunks or []),
        })


# ---------------------------------------------------------------------------
# Flag library helpers  (re-exported from extraction.flag_library)
# ---------------------------------------------------------------------------

from extraction.flag_library import (  # noqa: E402, F401
    _DEFAULT_FLAG_LIBRARY_PATH,
    load_flag_library,
    save_flag,
    list_projects,
)


# ---------------------------------------------------------------------------
# Ingest helpers
# ---------------------------------------------------------------------------

def _is_ingested(db_dsn: str, source_path: str, collection_id: str) -> bool:
    """Delegate to db.client.is_document_ingested."""
    from db.client import is_document_ingested
    return is_document_ingested(db_dsn, source_path, collection_id)


def _resolve_source_path(path: str) -> Optional[Path]:
    """Resolve a document path to an existing file.

    Tries the path as-is first, then relative to the project root (CWD),
    stripping a leading '/' so that paths like '/data/raw/doc.pdf' work
    when the server is run from the project root.
    """
    p = Path(path)
    if p.exists():
        return p
    # Strip leading slash/backslash and try relative to CWD
    stripped = Path(path.lstrip("/\\"))
    if stripped.exists():
        return stripped
    return None


def _ingest_source(
    source_path: str,
    collection_id: str,
    db_dsn: str,
    emit: Callable[[str], None],
) -> Optional[str]:
    """Ingest one document into the RAG collection."""
    from pipeline.ingest_v3 import ingest_file
    ext = Path(source_path).suffix.lower()
    source_type = "pdf_book" if ext == ".pdf" else "notes"
    emit(f"  -> Ingesting {Path(source_path).name}...")
    result = ingest_file(
        source_path,
        db_dsn=db_dsn,
        collection_id=collection_id,
        source_type=source_type,
        run_post_ingest_hooks=False,
    )
    emit(f"  OK {Path(source_path).name} ingested")
    return str(result.get("doc_id") or "").strip() or None


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm_fn() -> Callable:
    """Return a callable wrapping llm.generation.ollama_chat."""
    from llm.generation import ollama_chat
    from utils.runtime_defaults import (
        DEFAULT_LLM_MODEL,
        DEFAULT_LLM_BASE_URL,
    )

    def _llm(
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        timeout_seconds: float = 120.0,
        num_predict: int = -1,
        **_kwargs: Any,
    ) -> str:
        return ollama_chat(
            model=DEFAULT_LLM_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            base_url=DEFAULT_LLM_BASE_URL,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            num_predict=num_predict,
            think=False,
            keep_alive=300,
            manage_vram=False,
        ) or ""

    return _llm


def _unload_main_llm() -> None:
    """Unload the primary Ollama LLM after final report assembly."""
    try:
        from llm.generation import unload_llm_model
        from utils.runtime_defaults import DEFAULT_LLM_BASE_URL, DEFAULT_LLM_MODEL
        unload_llm_model(model=DEFAULT_LLM_MODEL, base_url=DEFAULT_LLM_BASE_URL)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CorpusHandle factory
# ---------------------------------------------------------------------------

def _build_corpus_handle(
    db_dsn: str,
    collection_id: str,
) -> CorpusHandle:
    """Load all chunk rows for the collection into a CorpusHandle."""
    from db.client import _connect, init_db
    from utils.runtime_defaults import (
        DEFAULT_EMBED_BACKEND,
        DEFAULT_EMBED_MODEL_NAME,
        DEFAULT_EMBED_DIMENSION,
        DEFAULT_OLLAMA_BASE_URL,
    )

    init_db(db_dsn)
    conn = _connect(db_dsn)
    try:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.text, c.doc_id,
                   COALESCE(c.page_start, 0) AS page_start,
                   d.filename AS source_filename
            FROM chunks c
            JOIN documents d ON d.doc_id = c.doc_id
            WHERE c.collection_id = %s
              AND c.structural_role NOT IN ('document_summary', 'bibliography')
            ORDER BY c.doc_id, COALESCE(c.page_start, 0)
            """,
            (collection_id,),
        ).fetchall()
    finally:
        conn.close()

    chunk_list = [dict(r) for r in rows]

    return CorpusHandle(
        db_dsn=db_dsn,
        collection_id=collection_id,
        chunk_rows=chunk_list,
        embed_backend=DEFAULT_EMBED_BACKEND,
        embed_model_name=DEFAULT_EMBED_MODEL_NAME,
        embed_dimension=DEFAULT_EMBED_DIMENSION,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
    )


def _format_item_for_pass(item: Any) -> str:
    """Format one ExtractionItem as a bullet for post-pass LLM input.

    Includes source attribution (page + filename) so the LLM can reference
    provenance in summaries and consolidated outputs.
    """
    clean_text = _prepare_evidence_text_for_pass(str(item.text or ""))
    parts: List[str] = []
    if getattr(item, "source_chunk_id", ""):
        parts.append(f"[chunk:{item.source_chunk_id}]")
    parts.append(f"[{item.branch_name}]")
    parts.append(clean_text)
    cite_parts = []
    if getattr(item, "source_page", 0):
        cite_parts.append(f"p.{item.source_page}")
    if getattr(item, "source_filename", ""):
        cite_parts.append(item.source_filename)
    if cite_parts:
        parts.append(f"({', '.join(cite_parts)})")
    return " ".join(parts)


def _format_evidence_chunk_for_pass(branch_name: str, chunk: Dict[str, Any]) -> str:
    """Format one saved branch evidence chunk for second-pass input."""
    text = _prepare_evidence_text_for_pass(chunk.get("text", ""))
    parts: List[str] = []
    if chunk.get("chunk_id"):
        parts.append(f"[chunk:{chunk.get('chunk_id')}]")
    parts.append(f"[{branch_name}]")
    parts.append(text)
    cite_parts = []
    if int(chunk.get("page_start", 0) or 0):
        cite_parts.append(f"p.{int(chunk.get('page_start', 0) or 0)}")
    if chunk.get("source_filename"):
        cite_parts.append(str(chunk.get("source_filename")))
    if cite_parts:
        parts.append(f"({', '.join(cite_parts)})")
    return " ".join(part for part in parts if part).strip()


def _normalized_evidence_text(text: str) -> str:
    """Normalize evidence text for duplicate detection."""
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _prepare_evidence_text_for_pass(text: str) -> str:
    """Clean raw evidence for downstream prompts, including compacting markdown tables."""
    text = _compact_markdown_tables(text)
    text = re.sub(r"<[^>]+>", "", str(text or ""))
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in cleaned_lines if line).strip()


def _compact_markdown_tables(text: str) -> str:
    """Convert markdown table blocks into compact prose-like rows for cheaper LLM use."""
    raw_lines = [str(line).rstrip() for line in str(text or "").splitlines()]
    if not raw_lines:
        return ""

    output: List[str] = []
    idx = 0
    while idx < len(raw_lines):
        line = raw_lines[idx].strip()
        if line.count("|") >= 2:
            table_lines: List[str] = []
            while idx < len(raw_lines):
                candidate = raw_lines[idx].strip()
                if candidate.count("|") < 2:
                    break
                table_lines.append(candidate)
                idx += 1
            rendered = _render_markdown_table_block(table_lines)
            if rendered:
                output.append(rendered)
            continue
        if line:
            output.append(line)
        idx += 1

    return "\n".join(output)


def _render_markdown_table_block(lines: List[str]) -> str:
    """Render a markdown table block into compact line-based text."""
    parsed_rows = [_split_markdown_table_row(line) for line in lines if line.strip()]
    parsed_rows = [row for row in parsed_rows if row]
    if not parsed_rows:
        return ""

    if len(parsed_rows) >= 2 and _looks_like_markdown_divider_row(parsed_rows[1]):
        header = parsed_rows[0]
        data_rows = parsed_rows[2:]
    else:
        header = [f"Column {i + 1}" for i in range(len(parsed_rows[0]))]
        data_rows = parsed_rows

    header = [_clean_table_cell_text(cell) for cell in header]
    data_rows = [
        [_clean_table_cell_text(cell) for cell in row]
        for row in data_rows
    ]
    data_rows = [row for row in data_rows if any(cell for cell in row)]
    if not data_rows:
        return ""

    rendered_rows: List[str] = []
    max_rows = 12
    for row in data_rows[:max_rows]:
        pairs: List[str] = []
        for col_idx, cell in enumerate(row):
            if not cell:
                continue
            label = header[col_idx] if col_idx < len(header) and header[col_idx] else f"Column {col_idx + 1}"
            pairs.append(f"{label}: {cell}")
        if pairs:
            rendered_rows.append("; ".join(pairs))

    if not rendered_rows:
        return ""

    suffix = ""
    if len(data_rows) > max_rows:
        suffix = f"\nTable continues with {len(data_rows) - max_rows} more row(s)."
    return "Table rows:\n" + "\n".join(f"- {row}" for row in rendered_rows) + suffix


def _split_markdown_table_row(line: str) -> List[str]:
    """Split one markdown table row into cells."""
    stripped = line.strip()
    stripped = re.sub(r"\|{3,}", "|", stripped)
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _looks_like_markdown_divider_row(cells: List[str]) -> bool:
    """Return True when a parsed row is just markdown table separators."""
    if not cells:
        return False
    cleaned = [cell.replace(":", "").replace("-", "").strip() for cell in cells]
    return all(not cell for cell in cleaned)


def _clean_table_cell_text(text: str) -> str:
    """Normalize a table cell while dropping obvious markdown noise."""
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    clean = clean.strip("|")
    if re.fullmatch(r"[:\-\s]+", clean):
        return ""
    return clean


def _is_low_information_cross_reference(text: str) -> bool:
    """Detect repetitive appendix/form cross-reference bullets that add little standalone value."""
    norm = _normalized_evidence_text(text)
    if not norm:
        return True
    patterns = (
        "refer to",
        "see appendix",
        "see form",
        "good faith efforts",
        "gfe outreach",
    )
    return any(pat in norm for pat in patterns) and len(norm) < 180


def _render_pass_line_for_llm(raw_line: str) -> str:
    """Render a structured evidence line for LLM prompts without exposing chunk IDs."""
    parsed = _parse_formatted_pass_item(raw_line)
    if not parsed:
        return _clean_report_text(raw_line.strip(), strip_page_footers=True)
    clean_text = _clean_report_text(parsed["text"], strip_page_footers=True)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    if len(clean_text) > 650:
        clean_text = clean_text[:647].rstrip(" ,;:-") + "..."
    parts = [f"[{parsed['branch_name']}]", clean_text]
    if parsed["citation_suffix"]:
        parts.append(parsed["citation_suffix"])
    return " ".join(part for part in parts if part).strip()


def _clean_report_text(
    text: str,
    *,
    strip_page_footers: bool = False,
    preserve_newlines: bool = False,
) -> str:
    """Clean obvious extraction artifacts for user-facing report content."""
    cleaned = str(text or "")
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€“": "-",
        "â€”": "-",
        "â€¢": "-",
        "Ã¢â‚¬Â¦": "...",
        "ï‚·": "-",
        "ï¬": "fi",
        "ï¬ƒ": "ffi",
        "Â§": "§",
    }
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    cleaned = re.sub(r"(?im)^\s*Outreach\)\*\*\s*\[ TABLE \]\s*", "", cleaned)
    cleaned = re.sub(r"\[\s*TABLE\s*\]", "", cleaned, flags=re.IGNORECASE)
    if strip_page_footers:
        cleaned = re.sub(r"\bPage\s+\d+\s+of\s+\d+\b", "", cleaned, flags=re.IGNORECASE)
    if preserve_newlines:
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()
    else:
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _looks_truncated_output(text: str) -> bool:
    """Best-effort detection for outputs that were cut off by context or token limits."""
    value = str(text or "").rstrip()
    if not value:
        return False
    if value.endswith(("...", "…")):
        return True
    if re.search(r"\b(Source:|Figure|Appendix|Step\s+\d+|The result of these steps should look as shown in)\s*$", value):
        return True
    last_line = value.splitlines()[-1].strip()
    if not last_line:
        return False
    if last_line in {"*", "-", "+", "•"}:
        return True
    if re.fullmatch(r"#{1,6}\s+.+", last_line):
        return True
    if re.fullmatch(r"\*\*[^*]+\*\*", last_line):
        return True
    if last_line[-1] in ".!?)]\"'`":
        return False
    return bool(re.search(r"[A-Za-z0-9]$", last_line))


def _repair_truncated_section(
    *,
    heading: str,
    draft_text: str,
    source_context: str,
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
    retry_label: str,
    cancel_check: CancelCheck = None,
) -> str:
    """Rewrite a truncated section so it finishes cleanly without inventing new facts."""
    trimmed_context = _clean_report_text(source_context, preserve_newlines=True)
    if len(trimmed_context) > 2600:
        trimmed_context = trimmed_context[:2597].rstrip(" ,;:-") + "..."
    repaired = _run_post_pass_llm_with_retry(
        llm_fn,
        system_prompt="You repair truncated report sections using only the provided source material. Do not invent facts.",
        user_prompt=(
            f"Section heading: {heading}\n\n"
            f"Source material:\n{trimmed_context}\n\n"
            f"Truncated draft:\n{draft_text}\n\n"
            "Rewrite this section from scratch so it is concise, complete, and ends cleanly. "
            "Preserve grounded facts only, remove dangling bullets/headings, and do not mention truncation."
        ),
        temperature=min(float(getattr(pp, "temperature", 0.0) or 0.0), 0.2),
        timeout_seconds=pp.timeout_seconds,
        num_predict=max(int(getattr(pp, "num_predict", 0) or 0), 768),
        retries=1,
        emit=emit,
        retry_label=retry_label,
        cancel_check=cancel_check,
    ).strip()
    return repaired or draft_text


def _compress_summary_source_blocks(
    blocks: List[str],
    *,
    max_blocks: int = 8,
    max_chars_per_block: int = 1600,
) -> List[str]:
    """Trim later-pass source blocks so they leave room for output tokens."""
    compressed: List[str] = []
    for block in blocks:
        clean = _clean_report_text(str(block or ""))
        if not clean:
            continue
        if len(clean) > max_chars_per_block:
            clean = clean[: max_chars_per_block - 3].rstrip(" ,;:-") + "..."
        compressed.append(clean)
        if len(compressed) >= max_blocks:
            break
    return compressed


def _build_evidence_appendix_report(
    project: ProjectConfig,
    *,
    second_pass_results: List[SecondPassResult],
) -> str:
    """Build a separate appendix/debug report containing the organized raw evidence."""
    organizer_blocks: List[str] = []
    for res in second_pass_results:
        if res.status != "ok" or res.artifact_type != "category_organizer_v1" or not res.artifact_data:
            continue
        rendered = _render_category_organizer_markdown(res.artifact_data).strip()
        if rendered:
            organizer_blocks.append(f"## {res.output_heading}\n\n{rendered}")
    if not organizer_blocks:
        return ""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return "\n".join(
        [
            f"# Extraction Evidence Appendix: {project.name}",
            f"Generated: {now}",
            "",
            "This appendix contains organized raw evidence retained for QA and traceability.",
            "It is intentionally noisier than the main report.",
            "",
            "---",
            "",
            "\n\n---\n\n".join(organizer_blocks),
            "",
        ]
    ).strip() + "\n"


def _dedupe_second_pass_items_text(items_text: str) -> str:
    """Collapse repeated evidence lines so appendices/cross-references do not dominate post passes."""
    seen: set[tuple[str, str]] = set()
    deduped_lines: List[str] = []
    low_info_lines: List[str] = []
    for raw_line in items_text.splitlines():
        parsed = _parse_formatted_pass_item(raw_line)
        if not parsed:
            continue
        key = (parsed["branch_name"], _normalized_evidence_text(parsed["text"]))
        if key in seen:
            continue
        seen.add(key)
        if _is_low_information_cross_reference(parsed["text"]):
            low_info_lines.append(parsed["formatted_line"])
        else:
            deduped_lines.append(parsed["formatted_line"])
    return "\n".join(deduped_lines + low_info_lines)


def _split_items_text(items_text: str, max_chars: int) -> List[str]:
    """Split items_text into batches of at most max_chars, breaking on line boundaries."""
    if max_chars <= 0 or len(items_text) <= max_chars:
        return [items_text]
    lines = items_text.splitlines(keepends=True)
    batches: List[str] = []
    current_parts: List[str] = []
    current_len = 0
    for line in lines:
        if current_parts and current_len + len(line) > max_chars:
            batches.append("".join(current_parts))
            current_parts = []
            current_len = 0
        current_parts.append(line)
        current_len += len(line)
    if current_parts:
        batches.append("".join(current_parts))
    return batches or [items_text]


def _split_numbered_lines(
    numbered_lines: List[tuple[int, str]],
    max_chars: int,
) -> List[List[tuple[int, str]]]:
    """Split numbered evidence lines into directory batches by character budget."""
    if max_chars <= 0 or not numbered_lines:
        return [numbered_lines] if numbered_lines else []

    batches: List[List[tuple[int, str]]] = []
    current: List[tuple[int, str]] = []
    current_len = 0

    for item_id, line in numbered_lines:
        rendered = f"[{item_id}] {line}"
        rendered_len = len(rendered) + 1
        if current and current_len + rendered_len > max_chars:
            batches.append(current)
            current = []
            current_len = 0
        current.append((item_id, line))
        current_len += rendered_len

    if current:
        batches.append(current)
    return batches


def _render_post_pass_user_prompt(template: str, branch_names: str, items_text: str) -> str:
    """Fill the standard placeholders used by post-pass prompts."""
    return template.replace("{branch_names}", branch_names).replace("{items_text}", items_text)

def _parse_formatted_pass_item(line: str) -> Optional[Dict[str, Any]]:
    """Parse one formatted evidence line into structured fields."""
    match = re.match(r"^\[chunk:([^\]]+)\]\s+\[([^\]]+)\]\s+(.*)$", line.strip())
    if not match:
        return None
    chunk_id, branch_name, remainder = match.groups()
    citation_suffix = ""
    text = remainder.strip()
    cite_match = re.search(r"\s+(\([^()]+\))$", text)
    if cite_match:
        citation_suffix = cite_match.group(1)
        text = text[: cite_match.start()].rstrip()
    return {
        "chunk_id": chunk_id.strip(),
        "branch_name": branch_name.strip(),
        "text": text.strip(),
        "citation_suffix": citation_suffix.strip(),
        "formatted_line": line.strip(),
    }

def _artifact_result(
    response_text: str,
    *,
    artifact_type: str = "",
    artifact_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Bundle text and optional machine-readable artifact payload."""
    return {
        "response_text": response_text,
        "artifact_type": artifact_type,
        "artifact_data": artifact_data,
    }


def _merge_category_artifact(artifact: Dict[str, Any]) -> Dict[str, Any]:
    """Deterministically collapse duplicate category entries into one header per name."""
    merged_order: List[str] = []
    merged_map: Dict[str, Dict[str, Any]] = {}
    merged_warnings: List[str] = list(artifact.get("routing_warnings", []) or [])

    for raw_category in artifact.get("categories", []) or []:
        name = str(raw_category.get("name") or "Uncategorized").strip() or "Uncategorized"
        key = name.casefold()
        if key not in merged_map:
            merged_map[key] = {
                "name": name,
                "item_count": 0,
                "selected_ids": [],
                "chunk_ids": [],
                "evidence_lines": [],
            }
            merged_order.append(key)
        dest = merged_map[key]

        for field in ("selected_ids", "chunk_ids", "evidence_lines"):
            for value in raw_category.get(field, []) or []:
                if value not in dest[field]:
                    dest[field].append(value)

        dest["item_count"] = len(dest["evidence_lines"])

    merged_categories = [merged_map[key] for key in merged_order if merged_map[key]["evidence_lines"]]
    return {
        "_artifact_type": "category_organizer_v1",
        "category_signature": str(artifact.get("category_signature") or _category_signature([cat["name"] for cat in merged_categories])),
        "routing_warnings": merged_warnings,
        "categories": merged_categories,
    }


def _organize_items_text_by_category(
    items_text: str,
    *,
    preferred_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Deterministically bucket extracted evidence by branch/category."""
    order: List[str] = []
    categories: Dict[str, Dict[str, Any]] = {}

    for name in preferred_categories or []:
        clean = re.sub(r"\s+", " ", str(name or "")).strip()
        if not clean or clean in categories:
            continue
        categories[clean] = {
            "name": clean,
            "item_count": 0,
            "chunk_ids": [],
            "evidence_lines": [],
        }
        order.append(clean)

    for raw_line in items_text.splitlines():
        parsed = _parse_formatted_pass_item(raw_line)
        if not parsed:
            continue
        name = parsed["branch_name"] or "Uncategorized"
        if name not in categories:
            categories[name] = {
                "name": name,
                "item_count": 0,
                "chunk_ids": [],
                "evidence_lines": [],
            }
            order.append(name)
        entry = categories[name]
        line = parsed["formatted_line"]
        if line not in entry["evidence_lines"]:
            entry["evidence_lines"].append(line)
        chunk_id = parsed["chunk_id"]
        if chunk_id and chunk_id not in entry["chunk_ids"]:
            entry["chunk_ids"].append(chunk_id)
        entry["item_count"] = len(entry["evidence_lines"])

    return _merge_category_artifact({
        "_artifact_type": "category_organizer_v1",
        "category_signature": _category_signature(order),
        "categories": [categories[name] for name in order if categories[name]["evidence_lines"]],
    })


def _organize_items_text_into_report_categories(
    items_text: str,
    *,
    report_categories: List[str],
    llm_fn: Callable,
    pp: Any,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Route extracted items into report-facing categories chosen by the user."""
    clean_categories = _normalize_category_names(report_categories)
    if not clean_categories:
        return _organize_items_text_by_category(items_text)

    numbered_lines: List[tuple[int, str]] = []
    numeric_to_line: Dict[int, str] = {}
    numeric_to_chunk: Dict[int, str] = {}
    next_id = 1
    for raw_line in items_text.splitlines():
        parsed = _parse_formatted_pass_item(raw_line)
        if not parsed:
            continue
        line = parsed["formatted_line"]
        numbered_lines.append((next_id, line))
        numeric_to_line[next_id] = line
        numeric_to_chunk[next_id] = parsed["chunk_id"]
        next_id += 1

    if not numbered_lines:
        return {
            "_artifact_type": "category_organizer_v1",
            "category_signature": _category_signature(clean_categories),
            "categories": [],
        }

    if len(clean_categories) == 1:
        only_category = clean_categories[0]
        evidence_lines = [line for _nid, line in numbered_lines]
        chunk_ids = [
            numeric_to_chunk[nid]
            for nid, _line in numbered_lines
            if numeric_to_chunk.get(nid)
        ]
        return _merge_category_artifact({
            "_artifact_type": "category_organizer_v1",
            "category_signature": _category_signature(clean_categories),
            "routing_warnings": ["single report category detected; skipped LLM routing"],
            "categories": [{
                "name": only_category,
                "item_count": len(evidence_lines),
                "selected_ids": [nid for nid, _line in numbered_lines],
                "chunk_ids": chunk_ids,
                "evidence_lines": evidence_lines,
            }],
        })

    directory_batches = _split_numbered_lines(numbered_lines, max(1200, int(getattr(pp, "max_chars_per_batch", 0) or 0)))
    category_state: Dict[str, Dict[str, Any]] = {
        category_name: {
            "name": category_name,
            "item_count": 0,
            "selected_ids": [],
            "chunk_ids": [],
            "evidence_lines": [],
        }
        for category_name in clean_categories
    }
    remaining_numbered_lines: List[tuple[int, str]] = []
    category_lookup = {name.casefold(): name for name in clean_categories}
    for nid, line in numbered_lines:
        parsed = _parse_formatted_pass_item(line)
        branch_name = str((parsed or {}).get("branch_name") or "").strip()
        matched_category = category_lookup.get(branch_name.casefold()) if branch_name else None
        if not matched_category:
            remaining_numbered_lines.append((nid, line))
            continue
        state = category_state[matched_category]
        state["selected_ids"].append(nid)
        state["evidence_lines"].append(line)
        chunk_id = numeric_to_chunk.get(nid)
        if chunk_id and chunk_id not in state["chunk_ids"]:
            state["chunk_ids"].append(chunk_id)

    if remaining_numbered_lines and len(remaining_numbered_lines) != len(numbered_lines):
        emit(
            f"    category routing reused exact branch/category matches for "
            f"{len(numbered_lines) - len(remaining_numbered_lines)} item(s)"
        )

    if not remaining_numbered_lines:
        return _merge_category_artifact({
            "_artifact_type": "category_organizer_v1",
            "category_signature": _category_signature(clean_categories),
            "routing_warnings": ["all evidence routed by exact branch/category match"],
            "categories": list(category_state.values()),
        })

    directory_batches = _split_numbered_lines(
        remaining_numbered_lines,
        max(1200, int(getattr(pp, "max_chars_per_batch", 0) or 0)),
    )
    routing_warnings: List[str] = []
    guidance = getattr(pp, "instructions", "").strip()
    for batch_idx, directory_batch in enumerate(directory_batches, start=1):
        raise_if_cancelled(cancel_check)
        if len(directory_batches) > 1:
            emit(f"    category routing batch {batch_idx}/{len(directory_batches)}")
        scan_directory = format_scan_batch_text(directory_batch)
        categories_block = "\n".join(f"- {name}" for name in clean_categories)
        prompt = (
            "Assign each evidence ID to at most one best-fit report category.\n"
            "Return one line for every category using exactly this format:\n"
            "- Category Name: [1] [4] [7]\n"
            "If a category has no matches, return:\n"
            "- Category Name: NONE\n"
            "Do not include explanations. Do not omit categories.\n\n"
            f"{f'Guidance: {guidance}\\n\\n' if guidance else ''}"
            "Report categories:\n"
            f"{categories_block}\n\n"
            "Evidence directory:\n"
            f"{scan_directory}\n"
        )
        try:
            response = _run_post_pass_llm_with_retry(
                llm_fn,
                system_prompt="You assign extracted evidence into user-defined report categories. Each evidence ID may belong to at most one category. Return only category-to-ID lines.",
                user_prompt=prompt,
                temperature=0.0,
                timeout_seconds=pp.timeout_seconds,
                num_predict=min(pp.num_predict if pp.num_predict > 0 else 512, 1024),
                retries=2,
                emit=emit,
                retry_label=f"category routing batch {batch_idx}/{len(directory_batches)}",
                cancel_check=cancel_check,
            )
        except ExtractionCancelled:
            raise
        except Exception as exc:
            warning = f"routing batch {batch_idx}/{len(directory_batches)} failed: {exc}"
            routing_warnings.append(warning)
            emit(f"      warning: {warning}; continuing")
            continue

        assigned_ids: set[int] = set()
        response_lines = [line.strip() for line in str(response or "").splitlines() if line.strip()]
        for category_name in clean_categories:
            state = category_state[category_name]
            matched_line = next(
                (
                    line for line in response_lines
                    if re.match(rf"^-?\s*{re.escape(category_name)}\s*:", line, flags=re.IGNORECASE)
                ),
                "",
            )
            if not matched_line:
                continue
            if re.search(r"\bNONE\b", matched_line, flags=re.IGNORECASE):
                continue
            for nid in parse_scan_response(matched_line):
                if nid not in numeric_to_line or nid in assigned_ids:
                    continue
                assigned_ids.add(nid)
                if nid not in state["selected_ids"]:
                    state["selected_ids"].append(nid)
                line = numeric_to_line[nid]
                if line not in state["evidence_lines"]:
                    state["evidence_lines"].append(line)
                chunk_id = numeric_to_chunk.get(nid)
                if chunk_id and chunk_id not in state["chunk_ids"]:
                    state["chunk_ids"].append(chunk_id)

    categories: List[Dict[str, Any]] = []
    for category_name in clean_categories:
        state = category_state[category_name]
        state["item_count"] = len(state["evidence_lines"])
        categories.append(state)

    return _merge_category_artifact({
        "_artifact_type": "category_organizer_v1",
        "category_signature": _category_signature(clean_categories),
        "routing_warnings": routing_warnings,
        "categories": [category for category in categories if category["evidence_lines"]],
    })


def _render_category_organizer_markdown(
    artifact: Dict[str, Any],
    *,
    cancel_check: CancelCheck = None,
) -> str:
    """Render an organized-by-category artifact as readable markdown."""
    sections: List[str] = []
    for category in artifact.get("categories", []):
        raise_if_cancelled(cancel_check)
        name = str(category.get("name") or "Uncategorized").strip()
        evidence_lines = [str(line).strip() for line in category.get("evidence_lines", []) if str(line).strip()]
        if not evidence_lines:
            continue
        body = "\n".join(f"- {_render_pass_line_for_llm(line)}" for line in evidence_lines)
        sections.append(f"### {name}\n{body}")
    return "\n\n".join(sections)


def _normalize_category_names(names: List[str]) -> List[str]:
    """Normalize category names for reuse/signature comparisons."""
    normalized: List[str] = []
    seen: set[str] = set()
    for raw in names or []:
        clean = re.sub(r"\s+", " ", str(raw or "")).strip()
        if not clean:
            continue
        key = clean.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(clean)
    return normalized


def _category_signature(names: List[str]) -> str:
    """Return a deterministic signature for a category list."""
    normalized = _normalize_category_names(names)
    return "||".join(name.casefold() for name in normalized)


def _collect_source_pass_results(
    pp: Any,
    results_by_name: Dict[str, SecondPassResult],
) -> List[SecondPassResult]:
    """Resolve source pass names to completed post-pass results."""
    source_names = list(getattr(pp, "source_passes", []) or [])
    collected: List[SecondPassResult] = []
    for pass_name in source_names:
        prior = results_by_name.get(pass_name)
        if prior is None or prior.status != "ok":
            raise ValueError(f"Required source pass '{pass_name}' has no successful output available.")
        collected.append(prior)
    return collected


def _resolve_linked_report_categories(pp: Any, configured_passes: List[Any]) -> List[str]:
    """Resolve effective report categories, following any explicit reuse link."""
    linked_name = str(getattr(pp, "reuse_categories_from_pass", "") or "").strip()
    if linked_name:
        for other in configured_passes:
            if other is pp:
                continue
            if str(getattr(other, "name", "") or "").strip() == linked_name:
                return _normalize_category_names(list(getattr(other, "report_categories", []) or []))
    return _normalize_category_names(list(getattr(pp, "report_categories", []) or []))


def _resolve_category_artifact(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Get an organizer artifact from prior passes or raw branch evidence."""
    report_categories = _normalize_category_names(list(getattr(pp, "report_categories", []) or []))
    wanted_signature = _category_signature(report_categories)
    source_results = _collect_source_pass_results(pp, results_by_name) if getattr(pp, "input_source", "") == "selected_pass_outputs" else []
    for prior in source_results:
        if prior.artifact_type == "category_organizer_v1" and prior.artifact_data:
            prior_sig = str(prior.artifact_data.get("category_signature") or "")
            if not wanted_signature or prior_sig == wanted_signature:
                return _merge_category_artifact(prior.artifact_data)
        if prior.artifact_type == "category_summaries_v1" and prior.artifact_data:
            organizer_artifact = prior.artifact_data.get("organizer_artifact")
            if isinstance(organizer_artifact, dict):
                prior_sig = str(organizer_artifact.get("category_signature") or "")
                if not wanted_signature or prior_sig == wanted_signature:
                    return _merge_category_artifact(organizer_artifact)

    for prior in results_by_name.values():
        if prior.status != "ok" or not prior.artifact_data:
            continue
        if prior.artifact_type == "category_organizer_v1":
            prior_sig = str(prior.artifact_data.get("category_signature") or "")
            if wanted_signature and prior_sig == wanted_signature:
                return _merge_category_artifact(prior.artifact_data)
        elif prior.artifact_type == "category_summaries_v1":
            organizer_artifact = prior.artifact_data.get("organizer_artifact")
            if isinstance(organizer_artifact, dict):
                prior_sig = str(organizer_artifact.get("category_signature") or "")
                if wanted_signature and prior_sig == wanted_signature:
                    return _merge_category_artifact(organizer_artifact)

    merged_text = "\n".join(items_text for _branch_names, items_text in per_branch_inputs if items_text.strip())
    if report_categories:
        return _organize_items_text_into_report_categories(
            merged_text,
            report_categories=report_categories,
            llm_fn=llm_fn,
            pp=pp,
            emit=emit,
            cancel_check=cancel_check,
        )
    preferred = list(getattr(pp, "source_branches", []) or [])
    return _organize_items_text_by_category(merged_text, preferred_categories=preferred)


def _render_category_summary_prompt(
    pp: Any,
    *,
    category_name: str,
    category_evidence: str,
) -> str:
    """Build the category-summary user prompt."""
    template = (pp.user_prompt_template or "").strip()
    if template:
        return (
            template
            .replace("{category_name}", category_name)
            .replace("{category_evidence}", category_evidence)
            .replace("{items_text}", category_evidence)
            .replace("{branch_names}", category_name)
        )
    return (
        f"Category: {category_name}\n\n"
        f"Evidence:\n{category_evidence}\n\n"
        "Write a grounded summary for this category using only the evidence above. "
        "Preserve human-readable source references where they are useful."
    )


def _run_organize_by_category_pass(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Organize branch evidence into report-facing category buckets."""
    artifact = _resolve_category_artifact(
        pp,
        per_branch_inputs=per_branch_inputs,
        results_by_name=results_by_name,
        llm_fn=llm_fn,
        emit=emit,
        cancel_check=cancel_check,
    )
    return _artifact_result(
        _render_category_organizer_markdown(artifact, cancel_check=cancel_check),
        artifact_type="category_organizer_v1",
        artifact_data=artifact,
    )


def _run_summarize_by_category_pass(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Summarize each category from organized evidence, auto-organizing if needed."""
    artifact = _resolve_category_artifact(
        pp,
        per_branch_inputs=per_branch_inputs,
        results_by_name=results_by_name,
        llm_fn=llm_fn,
        emit=emit,
        cancel_check=cancel_check,
    )
    rendered_sections: List[str] = []
    category_payloads: List[Dict[str, Any]] = []
    summary_warnings: List[str] = []

    for category in artifact.get("categories", []):
        raise_if_cancelled(cancel_check)
        name = str(category.get("name") or "Uncategorized").strip()
        evidence_lines = [str(line).strip() for line in category.get("evidence_lines", []) if str(line).strip()]
        if not evidence_lines:
            continue
        evidence_text = "\n".join(f"- {_render_pass_line_for_llm(line)}" for line in evidence_lines)
        evidence_batches = _split_items_text(evidence_text, pp.max_chars_per_batch)
        partials: List[str] = []
        for b_idx, batch_text in enumerate(evidence_batches, start=1):
            raise_if_cancelled(cancel_check)
            if len(evidence_batches) > 1:
                emit(f"    {name} summary batch {b_idx}/{len(evidence_batches)}...")
            try:
                partial = _run_post_pass_llm_with_retry(
                    llm_fn,
                    system_prompt=(
                        pp.system_prompt
                        or "You summarize one evidence category using only the provided evidence."
                    ),
                    user_prompt=_render_category_summary_prompt(
                        pp,
                        category_name=name,
                        category_evidence=batch_text,
                    ),
                    temperature=pp.temperature,
                    timeout_seconds=pp.timeout_seconds,
                    num_predict=pp.num_predict,
                    retries=2,
                    emit=emit,
                    retry_label=f"{name} summary batch {b_idx}/{len(evidence_batches)}",
                    cancel_check=cancel_check,
                ).strip()
            except ExtractionCancelled:
                raise
            except Exception as exc:
                warning = f"{name} summary batch {b_idx}/{len(evidence_batches)} failed: {exc}"
                summary_warnings.append(warning)
                emit(f"      warning: {warning}; continuing")
                continue
            if partial:
                partials.append(partial)
        summary_text = _reduce_post_pass_outputs(partials, pp, llm_fn, emit, cancel_check=cancel_check).strip() if len(partials) > 1 else (partials[0] if partials else "")
        if not summary_text:
            summary_text = "Not found in extracted text."
        if _looks_truncated_output(summary_text):
            emit(f"      truncation detected in final summary for {name}; repairing...")
            summary_text = _repair_truncated_section(
                heading=name,
                draft_text=summary_text,
                source_context=evidence_text,
                pp=pp,
                llm_fn=llm_fn,
                emit=emit,
                retry_label=f"final summary repair {name}",
                cancel_check=cancel_check,
            )
        if not summary_text.lstrip().startswith("#"):
            summary_text = f"### {name}\n{summary_text}"
        rendered_sections.append(summary_text)
        category_payloads.append({
            "name": name,
            "item_count": int(category.get("item_count") or len(evidence_lines)),
            "summary_markdown": summary_text,
            "chunk_ids": list(category.get("chunk_ids", [])),
        })

    artifact_out = {
        "_artifact_type": "category_summaries_v1",
        "categories": category_payloads,
    }
    return _artifact_result(
        "\n\n".join(rendered_sections),
        artifact_type="category_summaries_v1",
        artifact_data={
            **artifact_out,
            "category_signature": str(artifact.get("category_signature") or ""),
            "organizer_artifact": artifact,
            "summary_warnings": summary_warnings,
        },
    )


def _run_executive_summary_pass(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Create a top-level executive summary from category summaries or branch-level fallbacks."""
    source_results = _collect_source_pass_results(pp, results_by_name) if getattr(pp, "input_source", "") == "selected_pass_outputs" else []
    summary_blocks: List[str] = []
    for prior in source_results:
        if prior.artifact_type == "category_summaries_v1" and prior.artifact_data:
            summary_blocks.extend(
                str(cat.get("summary_markdown") or "").strip()
                for cat in prior.artifact_data.get("categories", [])
                if str(cat.get("summary_markdown") or "").strip()
            )
        elif prior.response_text.strip():
            summary_blocks.append(prior.response_text.strip())

    if not summary_blocks:
        for branch_name, items_text in per_branch_inputs:
            if not items_text.strip():
                continue
            branch_summary = _summarize_blocks_with_reduction(
                [items_text],
                heading=branch_name or "Branch Summary",
                system_prompt="You summarize one branch of extracted evidence into a concise grounded summary.",
                prompt_builder=lambda batch_text, branch_name=branch_name: (
                    f"Branch: {branch_name}\n\n"
                    f"Evidence:\n{batch_text}\n\n"
                    "Write a concise grounded summary of this branch using only the evidence above."
                ),
                pp=pp,
                llm_fn=llm_fn,
                emit=emit,
                batch_label=f"{branch_name} executive-source",
                cancel_check=cancel_check,
            )
            if branch_summary.strip():
                summary_blocks.append(branch_summary)

    summary_blocks = _compress_summary_source_blocks(summary_blocks, max_blocks=8, max_chars_per_block=1400)

    summary_text = _summarize_blocks_with_reduction(
        summary_blocks,
        heading=pp.effective_heading() or "Executive Summary",
        system_prompt=pp.system_prompt or "You write a concise executive summary from category summaries.",
        prompt_builder=lambda batch_text: (
            (pp.user_prompt_template or "").replace("{items_text}", batch_text).replace("{branch_names}", "category summaries")
            if (pp.user_prompt_template or "").strip()
            else (
                "Category summaries for executive summary:\n\n"
                f"{batch_text}\n\n"
                "Write a concise executive summary that introduces the report. "
                "Stay grounded in the supplied material only."
            )
        ),
        pp=pp,
        llm_fn=llm_fn,
        emit=emit,
        batch_label="executive summary",
        cancel_check=cancel_check,
    )
    artifact = {
        "_artifact_type": "executive_summary_v1",
        "summary_markdown": summary_text,
    }
    return _artifact_result(
        summary_text,
        artifact_type="executive_summary_v1",
        artifact_data=artifact,
    )


def _run_key_findings_pass(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Produce a concise key-findings block from reduced evidence selected by ID-only passes."""
    source_results = _collect_source_pass_results(pp, results_by_name) if getattr(pp, "input_source", "") == "selected_pass_outputs" else []
    summary_blocks = _collect_category_summary_blocks(source_results)
    grouped_blocks = _collect_grouped_organizer_blocks(source_results)
    selected_evidence_lines: List[str] = []
    source_blocks = list(summary_blocks or grouped_blocks)
    if not source_blocks:
        evidence_lines = _collect_formatted_evidence_lines(per_branch_inputs, source_results)
        selected_evidence_lines = _select_key_finding_evidence_lines(
            evidence_lines,
            pp=pp,
            llm_fn=llm_fn,
            emit=emit,
            cancel_check=cancel_check,
        )
        source_blocks = [
            "\n".join(f"- {line}" for line in selected_evidence_lines)
        ] if selected_evidence_lines else []
    if not source_blocks:
        source_blocks = [
            prior.response_text.strip() for prior in source_results if prior.response_text.strip()
        ]
    if not source_blocks:
        source_blocks = [
            items_text for _branch_names, items_text in per_branch_inputs if items_text.strip()
        ]
    source_blocks = _compress_summary_source_blocks(source_blocks, max_blocks=10, max_chars_per_block=1200)

    finding_text = _summarize_blocks_with_reduction(
        source_blocks,
        heading=pp.effective_heading() or "Key Findings",
        system_prompt="You write concise grounded key findings from extracted evidence.",
        prompt_builder=lambda batch_text: (
            f"Source material:\n\n{batch_text}\n\n"
            "Write a short 'Key Findings' section using only the source material above. "
            "Prefer 3 to 7 bullets. Stay grounded."
            + (f"\n\nAdditional instructions:\n{pp.instructions.strip()}" if getattr(pp, "instructions", "").strip() else "")
        ),
        pp=pp,
        llm_fn=llm_fn,
        emit=emit,
        batch_label="key findings",
        cancel_check=cancel_check,
    )
    artifact = {
        "_artifact_type": "key_findings_v1",
        "summary_markdown": finding_text,
        "selected_evidence_lines": selected_evidence_lines,
    }
    return _artifact_result(finding_text, artifact_type="key_findings_v1", artifact_data=artifact)


def _run_next_actions_pass(
    pp: Any,
    *,
    per_branch_inputs: List[tuple[str, str]],
    results_by_name: Dict[str, SecondPassResult],
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> Dict[str, Any]:
    """Produce a grounded next-actions block from reduced evidence selected by ID-only passes."""
    source_results = _collect_source_pass_results(pp, results_by_name) if getattr(pp, "input_source", "") == "selected_pass_outputs" else []
    summary_blocks = _collect_category_summary_blocks(source_results)
    grouped_blocks = _collect_grouped_organizer_blocks(source_results)
    selected_evidence_lines: List[str] = []
    source_blocks = list(summary_blocks or grouped_blocks)
    if not source_blocks:
        evidence_lines = _collect_formatted_evidence_lines(per_branch_inputs, source_results)
        selected_evidence_lines = _select_next_action_evidence_lines(
            evidence_lines,
            pp=pp,
            llm_fn=llm_fn,
            emit=emit,
            cancel_check=cancel_check,
        )
        source_blocks = [
            "\n".join(f"- {line}" for line in selected_evidence_lines)
        ] if selected_evidence_lines else []
    if not source_blocks:
        source_blocks = [
            prior.response_text.strip() for prior in source_results if prior.response_text.strip()
        ]
    if not source_blocks:
        source_blocks = [
            items_text for _branch_names, items_text in per_branch_inputs if items_text.strip()
        ]
    source_blocks = _compress_summary_source_blocks(source_blocks, max_blocks=10, max_chars_per_block=1200)

    action_text = _summarize_blocks_with_reduction(
        source_blocks,
        heading=pp.effective_heading() or "Next Actions",
        system_prompt="You write concise grounded next actions from extracted evidence.",
        prompt_builder=lambda batch_text: (
            f"Source material:\n\n{batch_text}\n\n"
            "Write a short 'Next Actions' section using only the source material above. "
            "Focus on practical follow-up actions that are directly supported by the evidence."
            + (f"\n\nAdditional instructions:\n{pp.instructions.strip()}" if getattr(pp, "instructions", "").strip() else "")
        ),
        pp=pp,
        llm_fn=llm_fn,
        emit=emit,
        batch_label="next actions",
        cancel_check=cancel_check,
    )
    artifact = {
        "_artifact_type": "next_actions_v1",
        "summary_markdown": action_text,
        "selected_evidence_lines": selected_evidence_lines,
    }
    return _artifact_result(action_text, artifact_type="next_actions_v1", artifact_data=artifact)


def _default_second_pass_title(pass_type: str) -> str:
    titles = {
        "report_plan": "Report Plan",
        "organize_by_category": "Organized Evidence by Category",
        "summarize_by_category": "Category Summaries",
        "executive_summary": "Executive Summary",
        "key_findings": "Key Findings",
        "next_actions": "Next Actions",
        "assemble_report": "Assembled Report",
    }
    return titles.get(pass_type, pass_type.replace("_", " ").title())


def _second_pass_slot(pass_type: str) -> str:
    if pass_type == "executive_summary":
        return "intro"
    if pass_type in {"key_findings", "summarize_by_category", "next_actions"}:
        return "body"
    if pass_type == "organize_by_category":
        return "appendix"
    return "body"


def _make_report_plan_result(project: ProjectConfig, second_passes: List[Any]) -> SecondPassResult:
    """Build a deterministic skeleton report plan from the selected second passes."""
    sections: List[Dict[str, Any]] = []
    for sp in second_passes:
        if getattr(sp, "pass_type", "") == "assemble_report":
            continue
        sections.append({
            "pass_name": sp.name,
            "pass_type": sp.pass_type,
            "heading": sp.effective_heading(),
            "slot": _second_pass_slot(sp.pass_type),
        })
    artifact = {
        "_artifact_type": "report_plan_v1",
        "report_title": f"{project.name} Report",
        "sections": sections,
    }
    plan_md = "\n".join(
        ["## Report Plan", ""]
        + [f"- {entry['slot']}: {entry['heading']}" for entry in sections]
    )
    return SecondPassResult(
        pass_name="report_plan",
        output_heading="Report Plan",
        response_text=plan_md,
        artifact_type="report_plan_v1",
        artifact_data=artifact,
        status="ok",
        elapsed_seconds=0.0,
    )


def _run_assemble_report_pass(
    pp: Any,
    *,
    results_by_name: Dict[str, SecondPassResult],
) -> Dict[str, Any]:
    """Assemble a final report from prior artifacts in a deterministic layout."""
    source_results = _collect_source_pass_results(pp, results_by_name)
    plan_artifact = None
    for prior in results_by_name.values():
        if prior.artifact_type == "report_plan_v1" and prior.artifact_data:
            plan_artifact = prior.artifact_data
            break
    executive_blocks: List[str] = []
    summary_blocks: List[str] = []
    findings_blocks: List[str] = []
    actions_blocks: List[str] = []
    organized_blocks: List[str] = []
    other_blocks: List[str] = []
    block_by_pass_name: Dict[str, str] = {}

    for prior in source_results:
        block = ""
        if prior.artifact_type == "executive_summary_v1" and prior.artifact_data:
            block = str(prior.artifact_data.get("summary_markdown") or "").strip()
            if block:
                executive_blocks.append(block)
        elif prior.artifact_type == "category_summaries_v1" and prior.artifact_data:
            rendered = "\n\n".join(
                str(cat.get("summary_markdown") or "").strip()
                for cat in prior.artifact_data.get("categories", [])
                if str(cat.get("summary_markdown") or "").strip()
            )
            if rendered:
                summary_blocks.append(rendered)
                block = rendered
        elif prior.artifact_type == "key_findings_v1" and prior.artifact_data:
            block = str(prior.artifact_data.get("summary_markdown") or "").strip()
            if block:
                findings_blocks.append(block)
        elif prior.artifact_type == "next_actions_v1" and prior.artifact_data:
            block = str(prior.artifact_data.get("summary_markdown") or "").strip()
            if block:
                actions_blocks.append(block)
        elif prior.artifact_type == "category_organizer_v1" and prior.artifact_data:
            rendered = _render_category_organizer_markdown(prior.artifact_data).strip()
            if rendered:
                block = f"## {prior.output_heading}\n\n{rendered}"
                organized_blocks.append(block)
        elif prior.response_text.strip():
            block = f"## {prior.output_heading}\n\n{prior.response_text.strip()}"
            other_blocks.append(block)
        if block:
            block_by_pass_name[prior.pass_name] = block

    assembled_blocks: List[str] = []
    if plan_artifact:
        title = str(plan_artifact.get("report_title") or "").strip()
        if title:
            assembled_blocks.append(f"# {title}")
        for entry in plan_artifact.get("sections", []):
            if str(entry.get("slot") or "") == "appendix":
                continue
            block = block_by_pass_name.get(str(entry.get("pass_name") or ""))
            if block and block.strip():
                assembled_blocks.append(block)
    else:
        assembled_blocks = executive_blocks + findings_blocks + summary_blocks + actions_blocks + other_blocks

    assembled = "\n\n".join(block for block in assembled_blocks if block.strip()).strip() or "Not found in extracted text."
    artifact = {
        "_artifact_type": "assembled_report_v1",
        "block_count": len([b for b in assembled_blocks if b.strip()]),
        "markdown": assembled,
    }
    return _artifact_result(
        assembled,
        artifact_type="assembled_report_v1",
        artifact_data=artifact,
    )


def _run_post_pass_llm(
    llm_fn: Callable,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_seconds: float,
    num_predict: int,
) -> str:
    """Run one post-pass LLM call and normalize the response."""
    return llm_fn(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        num_predict=num_predict,
    ) or ""


def _run_post_pass_llm_with_retry(
    llm_fn: Callable,
    *,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_seconds: float,
    num_predict: int,
    retries: int,
    emit: Optional[Callable[[str], None]] = None,
    retry_label: str = "LLM call",
    cancel_check: CancelCheck = None,
) -> str:
    """Run one post-pass LLM call with local retries."""
    last_exc: Optional[Exception] = None
    total_attempts = max(1, int(retries) + 1)
    for attempt in range(1, total_attempts + 1):
        raise_if_cancelled(cancel_check)
        try:
            return _run_post_pass_llm(
                llm_fn,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                num_predict=num_predict,
            )
        except ExtractionCancelled:
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < total_attempts and emit:
                emit(f"        retrying {retry_label} ({attempt}/{total_attempts - 1} retries used)")
    if last_exc is not None:
        raise last_exc
    return ""


def _reduce_post_pass_outputs(
    partial_outputs: List[str],
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
    cancel_check: CancelCheck = None,
) -> str:
    """Merge map-step partial outputs into one cohesive final section."""
    if len(partial_outputs) <= 1:
        return partial_outputs[0] if partial_outputs else ""
    current_outputs = [str(txt).strip() for txt in partial_outputs if str(txt).strip()]
    round_idx = 1
    while len(current_outputs) > 1:
        raise_if_cancelled(cancel_check)
        emit(f"    reduce step round {round_idx} ({len(current_outputs)} partials)...")
        rendered_blocks = [
            f"Partial output {idx + 1}:\n{txt}"
            for idx, txt in enumerate(current_outputs)
        ]
        reduce_batches = _split_items_text(
            "\n\n---\n\n".join(rendered_blocks),
            max(1200, int(getattr(pp, "max_chars_per_batch", 0) or 0)),
        )
        next_outputs: List[str] = []
        for batch_idx, batch_text in enumerate(reduce_batches, start=1):
            raise_if_cancelled(cancel_check)
            reduce_system = (
                pp.system_prompt.strip()
                or "You are consolidating partial report sections into one final polished section."
            )
            reduce_user = (
                "The following partial outputs were produced from the same instruction over different batches.\n\n"
                f"{batch_text}\n\n"
                "Combine them into one coherent final section. Deduplicate overlap, preserve the intended structure, "
                "and do not mention batching, partials, or intermediate steps. Preserve human-readable source references when present."
            )
            reduced = _run_post_pass_llm_with_retry(
                llm_fn,
                system_prompt=reduce_system,
                user_prompt=reduce_user,
                temperature=pp.temperature,
                timeout_seconds=pp.timeout_seconds,
                num_predict=pp.num_predict,
                retries=2,
                emit=emit,
                retry_label=f"reduce step round {round_idx} batch {batch_idx}/{len(reduce_batches)}",
                cancel_check=cancel_check,
            ).strip()
            if reduced and _looks_truncated_output(reduced):
                emit(
                    f"      truncation detected in reduce step round {round_idx} batch "
                    f"{batch_idx}/{len(reduce_batches)}; retrying with higher output allowance..."
                )
                continued = _run_post_pass_llm_with_retry(
                    llm_fn,
                    system_prompt=reduce_system,
                    user_prompt=(
                        f"{reduce_user}\n\n"
                        "Your previous answer was cut off mid-sentence. Rewrite the section from scratch "
                        "more concisely so it fully completes. Do not leave any sentence unfinished."
                    ),
                    temperature=pp.temperature,
                    timeout_seconds=pp.timeout_seconds,
                    num_predict=max(int(pp.num_predict or 0), 768) if int(pp.num_predict or 0) > 0 else 768,
                    retries=1,
                    emit=emit,
                    retry_label=f"truncation recovery reduce round {round_idx} batch {batch_idx}/{len(reduce_batches)}",
                    cancel_check=cancel_check,
                ).strip()
                if continued:
                    reduced = continued
            if reduced:
                next_outputs.append(reduced)
        if not next_outputs:
            break
        current_outputs = next_outputs
        round_idx += 1

    return current_outputs[0] if current_outputs else ""


def _summarize_blocks_with_reduction(
    blocks: List[str],
    *,
    heading: str,
    system_prompt: str,
    prompt_builder: Callable[[str], str],
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
    batch_label: str,
    cancel_check: CancelCheck = None,
) -> str:
    """Summarize batched source blocks, then reduce them into one final section."""
    clean_blocks = [str(block).strip() for block in blocks if str(block).strip()]
    if not clean_blocks:
        return ""

    source_text = "\n\n".join(clean_blocks)
    source_batches = _split_items_text(source_text, pp.max_chars_per_batch)
    partials: List[str] = []
    summary_warnings: List[str] = []
    for b_idx, batch_text in enumerate(source_batches, start=1):
        raise_if_cancelled(cancel_check)
        if len(source_batches) > 1:
            emit(f"    {batch_label} batch {b_idx}/{len(source_batches)}...")
        try:
            partial = _run_post_pass_llm_with_retry(
                llm_fn,
                system_prompt=system_prompt,
                user_prompt=prompt_builder(batch_text),
                temperature=pp.temperature,
                timeout_seconds=pp.timeout_seconds,
                num_predict=pp.num_predict,
                retries=2,
                emit=emit,
                retry_label=f"{batch_label} batch {b_idx}/{len(source_batches)}",
                cancel_check=cancel_check,
            ).strip()
        except Exception as exc:
            warning = f"{batch_label} batch {b_idx}/{len(source_batches)} failed: {exc}"
            summary_warnings.append(warning)
            emit(f"      warning: {warning}; continuing")
            continue
        if partial and _looks_truncated_output(partial):
            emit(
                f"      truncation detected in {batch_label} batch {b_idx}/{len(source_batches)}; "
                "retrying with reduced context..."
            )
            smaller_batches = _split_items_text(batch_text, max(1500, int(len(batch_text) * 0.55)))
            retry_partials: List[str] = []
            for s_idx, smaller_text in enumerate(smaller_batches, start=1):
                retry_partial = _run_post_pass_llm_with_retry(
                    llm_fn,
                    system_prompt=system_prompt,
                    user_prompt=prompt_builder(smaller_text) + "\n\nBe concise and make sure the section finishes cleanly.",
                    temperature=pp.temperature,
                    timeout_seconds=pp.timeout_seconds,
                    num_predict=max(int(pp.num_predict or 0), 768) if int(pp.num_predict or 0) > 0 else 768,
                    retries=1,
                    emit=emit,
                    retry_label=f"truncation recovery {batch_label} batch {b_idx}.{s_idx}/{len(smaller_batches)}",
                    cancel_check=cancel_check,
                ).strip()
                if retry_partial:
                    retry_partials.append(retry_partial)
            if retry_partials:
                partial = (
                    _reduce_post_pass_outputs(retry_partials, pp, llm_fn, emit, cancel_check=cancel_check).strip()
                    if len(retry_partials) > 1
                    else retry_partials[0].strip()
                )
        if partial:
            partials.append(partial)

    summary_text = _reduce_post_pass_outputs(partials, pp, llm_fn, emit, cancel_check=cancel_check).strip() if len(partials) > 1 else (partials[0] if partials else "")
    if not summary_text:
        summary_text = "Not found in extracted text."
    if _looks_truncated_output(summary_text):
        emit(f"      truncation detected in final {batch_label}; repairing...")
        summary_text = _repair_truncated_section(
            heading=heading,
            draft_text=summary_text,
            source_context=source_text,
            pp=pp,
            llm_fn=llm_fn,
            emit=emit,
            retry_label=f"final {batch_label} repair",
            cancel_check=cancel_check,
        )
    summary_text = _clean_report_text(summary_text, preserve_newlines=True)
    if not summary_text.lstrip().startswith("#"):
        summary_text = f"### {heading}\n{summary_text}"
    return summary_text


def _collect_formatted_evidence_lines(
    per_branch_inputs: List[tuple[str, str]],
    source_results: List[SecondPassResult],
) -> List[str]:
    """Collect unique formatted evidence lines from organizer artifacts or raw branch inputs."""
    evidence_lines: List[str] = []

    for prior in source_results:
        if prior.artifact_type == "category_organizer_v1" and prior.artifact_data:
            for category in prior.artifact_data.get("categories", []):
                for line in category.get("evidence_lines", []) or []:
                    clean = str(line).strip()
                    if clean and clean not in evidence_lines:
                        evidence_lines.append(clean)

    if evidence_lines:
        return evidence_lines

    for _branch_name, items_text in per_branch_inputs:
        for raw_line in items_text.splitlines():
            clean = str(raw_line).strip()
            if clean and clean not in evidence_lines:
                evidence_lines.append(clean)
    return evidence_lines


def _collect_grouped_organizer_blocks(source_results: List[SecondPassResult]) -> List[str]:
    """Collect grouped organizer evidence as category-labeled source blocks."""
    blocks: List[str] = []
    for prior in source_results:
        if prior.artifact_type != "category_organizer_v1" or not prior.artifact_data:
            continue
        for category in prior.artifact_data.get("categories", []):
            name = str(category.get("name") or "Uncategorized").strip() or "Uncategorized"
            evidence_lines = [
                str(line).strip()
                for line in category.get("evidence_lines", []) or []
                if str(line).strip()
            ]
            if not evidence_lines:
                continue
            blocks.append(
                f"Category: {name}\n\n" +
                "\n".join(f"- {_render_pass_line_for_llm(line)}" for line in evidence_lines)
            )
    return blocks


def _collect_category_summary_blocks(source_results: List[SecondPassResult]) -> List[str]:
    """Collect per-category summary blocks when summaries already exist."""
    blocks: List[str] = []
    for prior in source_results:
        if prior.artifact_type != "category_summaries_v1" or not prior.artifact_data:
            continue
        for category in prior.artifact_data.get("categories", []):
            summary_text = str(category.get("summary_markdown") or "").strip()
            if summary_text:
                blocks.append(summary_text)
    return blocks


def _select_key_finding_evidence_lines(
    evidence_lines: List[str],
    *,
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
    target_count: int = 12,
    per_batch_cap: int = 5,
    cancel_check: CancelCheck = None,
) -> List[str]:
    """Reduce a large evidence set to a smaller key-findings evidence set using ID-only selection rounds."""
    current_lines = [line for line in evidence_lines if str(line).strip()]
    round_idx = 1

    while len(current_lines) > target_count:
        raise_if_cancelled(cancel_check)
        numbered_lines = [(idx, line) for idx, line in enumerate(current_lines, start=1)]
        batches = _split_numbered_lines(numbered_lines, max(1200, int(getattr(pp, "max_chars_per_batch", 0) or 0)))
        next_lines: List[str] = []
        seen_lines: set[str] = set()

        for batch_idx, batch in enumerate(batches, start=1):
            raise_if_cancelled(cancel_check)
            emit(f"    key findings selection round {round_idx} batch {batch_idx}/{len(batches)}...")
            directory = format_scan_batch_text(batch)
            prompt = (
                f"Select up to {per_batch_cap} evidence IDs that best support the most important report-level key findings.\n"
                "Return only bracketed numeric IDs like [1], [4], [7]. Return NONE if nothing fits.\n"
                "Do not explain your answer.\n\n"
                f"{f'Guidance: {getattr(pp, 'instructions', '').strip()}\\n' if getattr(pp, 'instructions', '').strip() else ''}"
                "Evidence directory:\n"
                f"{directory}\n"
            )
            try:
                response = _run_post_pass_llm_with_retry(
                    llm_fn,
                    system_prompt="You select the most important evidence for a concise Key Findings section. Return only bracketed numeric IDs or NONE.",
                    user_prompt=prompt,
                    temperature=0.0,
                    timeout_seconds=pp.timeout_seconds,
                    num_predict=min(pp.num_predict if pp.num_predict > 0 else 256, 512),
                    retries=2,
                    emit=emit,
                    retry_label=f"key findings selection round {round_idx} batch {batch_idx}/{len(batches)}",
                    cancel_check=cancel_check,
                )
            except ExtractionCancelled:
                raise
            except Exception as exc:
                emit(f"      warning: key findings selection round {round_idx} batch {batch_idx}/{len(batches)} failed: {exc}; continuing")
                continue
            for nid in parse_scan_response(response):
                if 1 <= nid <= len(current_lines):
                    line = current_lines[nid - 1]
                    if line not in seen_lines:
                        seen_lines.add(line)
                        next_lines.append(line)

        if not next_lines:
            current_lines = current_lines[:target_count]
            break
        if len(next_lines) >= len(current_lines):
            current_lines = next_lines[:target_count]
            break
        current_lines = next_lines
        round_idx += 1

    return current_lines


def _select_next_action_evidence_lines(
    evidence_lines: List[str],
    *,
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
    target_count: int = 12,
    per_batch_cap: int = 5,
    cancel_check: CancelCheck = None,
) -> List[str]:
    """Reduce a large evidence set to action-worthy evidence using ID-only selection rounds."""
    current_lines = [line for line in evidence_lines if str(line).strip()]
    round_idx = 1

    while len(current_lines) > target_count:
        raise_if_cancelled(cancel_check)
        numbered_lines = [(idx, line) for idx, line in enumerate(current_lines, start=1)]
        batches = _split_numbered_lines(numbered_lines, max(1200, int(getattr(pp, "max_chars_per_batch", 0) or 0)))
        next_lines: List[str] = []
        seen_lines: set[str] = set()

        for batch_idx, batch in enumerate(batches, start=1):
            raise_if_cancelled(cancel_check)
            emit(f"    next actions selection round {round_idx} batch {batch_idx}/{len(batches)}...")
            directory = format_scan_batch_text(batch)
            prompt = (
                f"Select up to {per_batch_cap} evidence IDs that most strongly justify practical follow-up actions, clarifications, or internal review items.\n"
                "Return only bracketed numeric IDs like [1], [4], [7]. Return NONE if nothing fits.\n"
                "Do not explain your answer.\n\n"
                f"{f'Guidance: {getattr(pp, 'instructions', '').strip()}\\n' if getattr(pp, 'instructions', '').strip() else ''}"
                "Evidence directory:\n"
                f"{directory}\n"
            )
            try:
                response = _run_post_pass_llm_with_retry(
                    llm_fn,
                    system_prompt="You select the most action-relevant evidence for a concise Next Actions section. Return only bracketed numeric IDs or NONE.",
                    user_prompt=prompt,
                    temperature=0.0,
                    timeout_seconds=pp.timeout_seconds,
                    num_predict=min(pp.num_predict if pp.num_predict > 0 else 256, 512),
                    retries=2,
                    emit=emit,
                    retry_label=f"next actions selection round {round_idx} batch {batch_idx}/{len(batches)}",
                    cancel_check=cancel_check,
                )
            except ExtractionCancelled:
                raise
            except Exception as exc:
                emit(f"      warning: next actions selection round {round_idx} batch {batch_idx}/{len(batches)} failed: {exc}; continuing")
                continue
            for nid in parse_scan_response(response):
                if 1 <= nid <= len(current_lines):
                    line = current_lines[nid - 1]
                    if line not in seen_lines:
                        seen_lines.add(line)
                        next_lines.append(line)

        if not next_lines:
            current_lines = current_lines[:target_count]
            break
        if len(next_lines) >= len(current_lines):
            current_lines = next_lines[:target_count]
            break
        current_lines = next_lines
        round_idx += 1

    return current_lines


# ---------------------------------------------------------------------------
# Post-branch pass runner
# ---------------------------------------------------------------------------

def _run_post_passes(
    project: ProjectConfig,
    ckpt_path: Path,
    llm_fn: Callable,
    emit: Callable[[str], None],
    verbose_dir: Optional[Path] = None,
    cancel_check: CancelCheck = None,
) -> List[SecondPassResult]:
    """Run the simplified serial second-pass stage after branch extraction."""
    configured = [sp for sp in getattr(project, "second_passes", []) if getattr(sp, "enabled", True)]
    if not configured:
        return []

    branch_results = _load_checkpoint_branch_results(ckpt_path)
    branch_items_by_name: Dict[str, List[Any]] = {
        br.branch_name: br.items for br in branch_results
    }
    branch_evidence_by_name: Dict[str, List[Dict[str, Any]]] = {
        br.branch_name: list(br.evidence_chunks or []) for br in branch_results
    }

    passes = list(configured)
    if not any(getattr(sp, "pass_type", "") == "assemble_report" for sp in passes):
        passes.append(SecondPassConfig(name="Assemble Report", pass_type="assemble_report", title="Assembled Report"))

    results: List[SecondPassResult] = []
    results_by_name: Dict[str, SecondPassResult] = {}

    plan_result = _make_report_plan_result(project, passes)
    _write_checkpoint(ckpt_path, "second_pass_result", plan_result.to_dict())
    results.append(plan_result)
    results_by_name[plan_result.pass_name] = plan_result
    emit(f"\nSecond-pass plan: {len(passes)} step(s)")

    for i, sp in enumerate(passes, start=1):
        raise_if_cancelled(cancel_check)
        emit(f"\nSecond pass {i}/{len(passes)}: {sp.effective_heading()}")
        t0 = time.perf_counter()
        try:
            sp.report_categories = _resolve_linked_report_categories(sp, passes)
            source_names = list(getattr(sp, "source_branches", []) or list(branch_items_by_name.keys()))
            per_branch_inputs = [
                (
                    bn,
                    _dedupe_second_pass_items_text(
                        "\n".join(
                            _format_evidence_chunk_for_pass(bn, chunk)
                            for chunk in branch_evidence_by_name.get(bn, [])
                        )
                        if branch_evidence_by_name.get(bn, [])
                        else "\n".join(_format_item_for_pass(item) for item in branch_items_by_name.get(bn, []))
                    ),
                )
                for bn in source_names
            ]
            if verbose_dir:
                pass_dir = verbose_dir / f"second_pass_{i:02d}_{_slugify(sp.name)}"
                _vdump_json(pass_dir, "00_input_summary.json", {
                    "pass_name": sp.name,
                    "pass_type": sp.pass_type,
                    "source_branches": source_names,
                    "branch_input_counts": {
                        branch_name: len([ln for ln in items_text.splitlines() if ln.strip()])
                        for branch_name, items_text in per_branch_inputs
                    },
                })
                for branch_name, items_text in per_branch_inputs:
                    _vdump(pass_dir, f"01_input_{_slugify(branch_name)}.txt", items_text)

            artifact_result: Dict[str, Any]
            if sp.pass_type == "organize_by_category":
                setattr(sp, "input_source", "selected_branch_items" if getattr(sp, "source_branches", []) else "all_branch_items")
                artifact_result = _run_organize_by_category_pass(
                    sp,
                    per_branch_inputs=per_branch_inputs,
                    results_by_name=results_by_name,
                    llm_fn=llm_fn,
                    emit=emit,
                    cancel_check=cancel_check,
                )
            elif sp.pass_type == "summarize_by_category":
                organizer_names = [
                    name for name, prior in results_by_name.items()
                    if prior.artifact_type == "category_organizer_v1"
                ]
                if organizer_names:
                    setattr(sp, "input_source", "selected_pass_outputs")
                    setattr(sp, "source_passes", organizer_names)
                else:
                    setattr(sp, "input_source", "selected_branch_items" if getattr(sp, "source_branches", []) else "all_branch_items")
                artifact_result = _run_summarize_by_category_pass(
                    sp,
                    per_branch_inputs=per_branch_inputs,
                    results_by_name=results_by_name,
                    llm_fn=llm_fn,
                    emit=emit,
                    cancel_check=cancel_check,
                )
            elif sp.pass_type == "executive_summary":
                summary_names = [
                    name for name, prior in results_by_name.items()
                    if prior.artifact_type == "category_summaries_v1"
                ]
                if summary_names:
                    setattr(sp, "input_source", "selected_pass_outputs")
                    setattr(sp, "source_passes", summary_names)
                artifact_result = _run_executive_summary_pass(
                    sp,
                    per_branch_inputs=per_branch_inputs,
                    results_by_name=results_by_name,
                    llm_fn=llm_fn,
                    emit=emit,
                    cancel_check=cancel_check,
                )
            elif sp.pass_type == "key_findings":
                organizer_names = [
                    name for name, prior in results_by_name.items()
                    if prior.artifact_type == "category_organizer_v1"
                ]
                if organizer_names:
                    setattr(sp, "input_source", "selected_pass_outputs")
                    setattr(sp, "source_passes", organizer_names)
                artifact_result = _run_key_findings_pass(
                    sp,
                    per_branch_inputs=per_branch_inputs,
                    results_by_name=results_by_name,
                    llm_fn=llm_fn,
                    emit=emit,
                    cancel_check=cancel_check,
                )
            elif sp.pass_type == "next_actions":
                organizer_names = [
                    name for name, prior in results_by_name.items()
                    if prior.artifact_type == "category_organizer_v1"
                ]
                if organizer_names:
                    setattr(sp, "input_source", "selected_pass_outputs")
                    setattr(sp, "source_passes", organizer_names)
                artifact_result = _run_next_actions_pass(
                    sp,
                    per_branch_inputs=per_branch_inputs,
                    results_by_name=results_by_name,
                    llm_fn=llm_fn,
                    emit=emit,
                    cancel_check=cancel_check,
                )
            elif sp.pass_type == "assemble_report":
                selected_pass_names = [
                    prior.pass_name for prior in results
                    if prior.pass_name != "report_plan" and prior.status == "ok"
                ]
                setattr(sp, "input_source", "selected_pass_outputs")
                setattr(sp, "source_passes", selected_pass_names)
                artifact_result = _run_assemble_report_pass(sp, results_by_name=results_by_name)
            else:
                raise ValueError(f"Unsupported second pass type: {sp.pass_type}")

            elapsed = round(time.perf_counter() - t0, 2)
            res = SecondPassResult(
                pass_name=sp.name,
                output_heading=sp.effective_heading(),
                response_text=artifact_result["response_text"],
                artifact_type=artifact_result["artifact_type"],
                artifact_data=artifact_result["artifact_data"],
                status="ok",
                elapsed_seconds=elapsed,
            )
            emit(f"  OK Second pass complete ({elapsed}s)")
        except ExtractionCancelled:
            raise
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            emit(f"  FAIL Second pass failed: {exc}")
            res = SecondPassResult(
                pass_name=sp.name,
                output_heading=sp.effective_heading(),
                status="error",
                error=str(exc),
                elapsed_seconds=elapsed,
            )

        _write_checkpoint(ckpt_path, "second_pass_result", res.to_dict())
        results.append(res)
        results_by_name[res.pass_name] = res

    return results


# ---------------------------------------------------------------------------
# run_project - public entry point
# ---------------------------------------------------------------------------

def run_project(
    project: ProjectConfig,
    *,
    db_dsn: Optional[str] = None,
    emit: Optional[Callable[[str], None]] = None,
    config_path: str = "configs/extraction.yaml",
    verbose: Optional[bool] = None,
    cancel_check: CancelCheck = None,
) -> ProjectRunResult:
    """Execute a full extraction project."""
    from utils.runtime_defaults import DEFAULT_DB_DSN

    if emit is None:
        emit = lambda _: None  # noqa: E731

    actual_db_dsn = db_dsn or DEFAULT_DB_DSN
    config = ExtractionConfig.from_yaml(config_path)
    if verbose is not None:
        config.verbose = verbose
    llm_fn = _make_llm_fn()
    t0 = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    ckpt_path = _checkpoint_path(project.slug, run_id)

    # â”€â”€ Verbose run directory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    verbose_run_dir: Optional[Path] = None
    if config.verbose:
        verbose_run_dir = Path(config.verbose_dir) / f"{project.slug}_{run_id}"
        verbose_run_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Starting extraction project: {project.name}")
    if verbose_run_dir:
        emit(f"Verbose mode ON -> {verbose_run_dir}")
    emit(f"Checkpoint: {ckpt_path}")

    # â”€â”€ 1. Ingest documents â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    collection_id = project.effective_collection_id()
    emit(f"Collection: {collection_id}")

    ingested_doc_ids: List[str] = []
    for src in project.document_sources:
        raise_if_cancelled(cancel_check)
        resolved = _resolve_source_path(src.path)
        if resolved is None:
            emit(f"  FAIL File not found: {src.path}")
            continue
        src.path = str(resolved)  # normalise to resolved path
        if _is_ingested(actual_db_dsn, src.path, collection_id):
            emit(f"  OK Already ingested: {src.effective_label()}")
        else:
            doc_id = _ingest_source(src.path, collection_id, actual_db_dsn, emit)
            if doc_id:
                ingested_doc_ids.append(doc_id)

    if ingested_doc_ids:
        emit("Finalizing ingested document metadata...")
        try:
            from pipeline.ingest_v3 import _post_ingest_hooks  # noqa: PLC0415

            for doc_id in ingested_doc_ids:
                _post_ingest_hooks(actual_db_dsn, doc_id)
            emit(f"  OK Finalized {len(ingested_doc_ids)} document(s)")
        except Exception as exc:
            emit(f"  WARN Post-ingest refresh failed: {exc}")

    # â”€â”€ 2. Build priority map and propagate to branches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    priority_map = build_priority_map(project.document_sources)
    for branch in project.branches:
        # Project-level weights are the defaults; branch-level overrides win
        merged = {**priority_map, **branch.source_priority}
        branch.source_priority = merged

    # â”€â”€ 3. Build corpus handle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    emit("Building corpus index...")
    corpus = _build_corpus_handle(actual_db_dsn, collection_id)
    total_chunks = len(corpus.chunk_rows)
    emit(f"  -> {total_chunks} chunks loaded for collection '{collection_id}'")

    if total_chunks == 0:
        return ProjectRunResult(
            project_slug=project.slug,
            error="No chunks found in collection - check that documents were ingested successfully.",
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )

    # â”€â”€ 4. Run branches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    branch_results: List[BranchResult] = []
    enabled_branches = [b for b in project.branches if b.enabled]
    emit(f"Running {len(enabled_branches)} branch(es)...")

    # Warm the LLM once so scan/synthesis/HyDE don't stall on cold model load
    emit("Warming LLM...")
    try:
        raise_if_cancelled(cancel_check)
        llm_fn(system_prompt="ping", user_prompt="ping", temperature=0.0, timeout_seconds=180.0)
        emit("  OK LLM ready")
    except ExtractionCancelled:
        raise
    except Exception as _warm_exc:
        emit(f"  WARN LLM warmup failed (will retry per-branch): {_warm_exc}")

    for i, branch in enumerate(enabled_branches, start=1):
        raise_if_cancelled(cancel_check)
        emit(f"\nBranch {i}/{len(enabled_branches)}: {branch.name}")
        branch_vdir = (verbose_run_dir / _slugify(branch.name)) if verbose_run_dir else None
        result = run_branch(
            branch, corpus, config, llm_fn, emit=emit, verbose_dir=branch_vdir, cancel_check=cancel_check
        )
        # Checkpoint to disk immediately and free items from RAM
        _write_checkpoint(ckpt_path, "branch_result", {
            "branch_name": result.branch_name,
            "output_heading": result.output_heading,
            "status": result.status,
            "stats": asdict(result.stats),
            "items": [item.to_dict() for item in result.items],
            "evidence_chunks": list(result.evidence_chunks or []),
        })
        emit(f"  -> Saved {len(result.items)} item(s) to checkpoint")
        result.items.clear()
        branch_results.append(result)

    # â”€â”€ 5. Post-branch passes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    # â”€â”€ 6. Build report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    emit("\nBuilding report...")
    raise_if_cancelled(cancel_check)
    second_pass_results = _run_post_passes(project, ckpt_path, llm_fn, emit, verbose_run_dir, cancel_check=cancel_check)
    # Reload full items from checkpoint (items were cleared after each branch to save RAM)
    branch_results = _load_checkpoint_branch_results(ckpt_path)
    base_report_md = assemble_report(
        project,
        branch_results,
        include_page_citations=True,
        include_branch_stats=True,
    )
    assembled_result = next(
        (
            res for res in reversed(second_pass_results)
            if res.status == "ok" and res.artifact_type == "assembled_report_v1" and res.artifact_data
        ),
        None,
    )
    report_md = (
        str(assembled_result.artifact_data.get("markdown") or "").strip()
        if assembled_result
        and assembled_result.artifact_data
        and int(assembled_result.artifact_data.get("block_count") or 0) > 0
        else base_report_md
    ) or base_report_md
    report_md = _clean_report_text(report_md, preserve_newlines=True)
    report_path = save_report(report_md, project)
    appendix_md = _build_evidence_appendix_report(project, second_pass_results=second_pass_results)
    appendix_path: Optional[Path] = None
    if appendix_md:
        appendix_path = save_report_variant(appendix_md, project, suffix="evidence_appendix")
        emit(f"Evidence appendix saved to: {appendix_path}")
    _unload_main_llm()
    emit(f"Report saved to: {report_path}")

    # â”€â”€ 7. Optionally clean up collection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if not project.keep_collection_after_run:
        emit("Removing temporary collection...")
        try:
            from db.client import delete_collection
            delete_collection(actual_db_dsn, collection_id, clear_chunks=True)
            emit("  OK Collection removed")
        except Exception as exc:
            emit(f"  WARN Could not remove collection: {exc}")

    elapsed = round(time.perf_counter() - t0, 2)
    total_items = sum(len(br.items) for br in branch_results)
    emit(f"\nOK Extraction complete - {total_items} item(s) across {len(branch_results)} branch(es) in {elapsed}s")

    return ProjectRunResult(
        project_slug=project.slug,
        branch_results=branch_results,
        second_pass_results=second_pass_results,
        report_path=str(report_path),
        appendix_path=str(appendix_path) if appendix_path else None,
        report_markdown=report_md,
        elapsed_seconds=elapsed,
        checkpoint_path=str(ckpt_path),
    )


def rerun_reports_from_checkpoint(
    project: ProjectConfig,
    *,
    checkpoint_path: str,
    emit: Optional[Callable[[str], None]] = None,
    cancel_check: CancelCheck = None,
) -> ProjectRunResult:
    """Re-run the serial second-pass/report stage from a prior branch checkpoint."""
    if emit is None:
        emit = lambda _: None  # noqa: E731

    source_ckpt = Path(checkpoint_path)
    if not source_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    t0 = time.perf_counter()
    llm_fn = _make_llm_fn()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rerun_ckpt = _checkpoint_path(project.slug, f"{run_id}_reports")

    emit(f"Starting report-only rerun: {project.name}")
    emit(f"Source checkpoint: {source_ckpt}")
    emit(f"Checkpoint: {rerun_ckpt}")

    branch_results = _load_checkpoint_branch_results(source_ckpt)
    if not branch_results:
        raise ValueError("No branch results were found in the source checkpoint.")

    _seed_checkpoint_with_branch_results(rerun_ckpt, branch_results)
    emit(f"Loaded {len(branch_results)} branch result(s) from checkpoint")
    raise_if_cancelled(cancel_check)
    second_pass_results = _run_post_passes(project, rerun_ckpt, llm_fn, emit, cancel_check=cancel_check)


    emit("\nBuilding report...")
    base_report_md = assemble_report(
        project,
        branch_results,
        include_page_citations=True,
        include_branch_stats=True,
    )
    assembled_result = next(
        (
            res for res in reversed(second_pass_results)
            if res.status == "ok" and res.artifact_type == "assembled_report_v1" and res.artifact_data
        ),
        None,
    )
    report_md = (
        str(assembled_result.artifact_data.get("markdown") or "").strip()
        if assembled_result
        and assembled_result.artifact_data
        and int(assembled_result.artifact_data.get("block_count") or 0) > 0
        else base_report_md
    ) or base_report_md
    report_md = _clean_report_text(report_md, preserve_newlines=True)
    report_path = save_report(report_md, project)
    appendix_md = _build_evidence_appendix_report(project, second_pass_results=second_pass_results)
    appendix_path: Optional[Path] = None
    if appendix_md:
        appendix_path = save_report_variant(appendix_md, project, suffix="evidence_appendix")
        emit(f"Evidence appendix saved to: {appendix_path}")
    _unload_main_llm()
    emit(f"Report saved to: {report_path}")

    elapsed = round(time.perf_counter() - t0, 2)
    emit(f"\nOK Report-only rerun complete in {elapsed}s")

    return ProjectRunResult(
        project_slug=project.slug,
        branch_results=branch_results,
        second_pass_results=second_pass_results,
        report_path=str(report_path),
        appendix_path=str(appendix_path) if appendix_path else None,
        report_markdown=report_md,
        elapsed_seconds=elapsed,
        checkpoint_path=str(rerun_ckpt),
    )

