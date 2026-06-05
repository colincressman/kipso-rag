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
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from extraction.branch_config import (
    BranchResult,
    PostBranchPassResult,
    ProjectConfig,
    build_priority_map,
)
from extraction.branch_runner import CorpusHandle, ExtractionConfig, run_branch
from extraction.report_builder import assemble_report, save_report
from utils.text_utils import slugify_path as _slugify


# ---------------------------------------------------------------------------
# ProjectRunResult
# ---------------------------------------------------------------------------

@dataclass
class ProjectRunResult:
    project_slug: str
    branch_results: List[BranchResult] = field(default_factory=list)
    post_pass_results: List[PostBranchPassResult] = field(default_factory=list)
    report_path: Optional[str] = None
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
) -> None:
    """Ingest one document into the RAG collection."""
    from pipeline.ingest_v3 import ingest_file
    ext = Path(source_path).suffix.lower()
    source_type = "pdf_book" if ext == ".pdf" else "notes"
    emit(f"  → Ingesting {Path(source_path).name}…")
    ingest_file(source_path, db_dsn=db_dsn, collection_id=collection_id, source_type=source_type)
    emit(f"  ✓ {Path(source_path).name} ingested")


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
        ) or ""

    return _llm


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
    parts = [f"[{item.branch_name}]", item.text]
    cite_parts = []
    if getattr(item, "source_page", 0):
        cite_parts.append(f"p.{item.source_page}")
    if getattr(item, "source_filename", ""):
        cite_parts.append(item.source_filename)
    if cite_parts:
        parts.append(f"({', '.join(cite_parts)})")
    return " ".join(parts)


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


def _render_post_pass_user_prompt(template: str, branch_names: str, items_text: str) -> str:
    """Fill the standard placeholders used by post-pass prompts."""
    return template.replace("{branch_names}", branch_names).replace("{items_text}", items_text)


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


def _reduce_post_pass_outputs(
    partial_outputs: List[str],
    pp: Any,
    llm_fn: Callable,
    emit: Callable[[str], None],
) -> str:
    """Merge map-step partial outputs into one cohesive final section."""
    if len(partial_outputs) <= 1:
        return partial_outputs[0] if partial_outputs else ""

    emit("    reduce step…")
    partials_text = "\n\n---\n\n".join(
        f"Partial output {idx + 1}:\n{txt.strip()}" for idx, txt in enumerate(partial_outputs)
    )
    reduce_system = (
        pp.system_prompt.strip()
        or "You are consolidating partial report sections into one final polished section."
    )
    reduce_user = (
        "The following partial outputs were produced from the same instruction over different batches.\n\n"
        f"{partials_text}\n\n"
        "Combine them into one coherent final section. Deduplicate overlap, preserve the intended structure, "
        "and do not mention batching, partials, or intermediate steps."
    )
    return _run_post_pass_llm(
        llm_fn,
        system_prompt=reduce_system,
        user_prompt=reduce_user,
        temperature=pp.temperature,
        timeout_seconds=pp.timeout_seconds,
        num_predict=pp.num_predict,
    )


# ---------------------------------------------------------------------------
# Post-branch pass runner
# ---------------------------------------------------------------------------

def _run_post_passes(
    project: ProjectConfig,
    ckpt_path: Path,
    llm_fn: Callable,
    emit: Callable[[str], None],
) -> List[PostBranchPassResult]:
    """Run each enabled post-branch pass in series, checkpointing each result."""
    passes = [p for p in project.post_branch_passes if p.enabled]
    if not passes:
        return []

    # Load all branch items written so far
    branch_results = _load_checkpoint_branch_results(ckpt_path)
    branch_items_by_name: Dict[str, List[Any]] = {
        br.branch_name: br.items for br in branch_results
    }

    results: List[PostBranchPassResult] = []
    prev_response: Optional[str] = None  # output of the most recently completed pass
    for i, pp in enumerate(passes, start=1):
        emit(f"\nPost-pass {i}/{len(passes)}: {pp.name}")
        t0 = time.perf_counter()
        try:
            input_source = getattr(pp, "input_source", "selected_branch_items" if pp.source_branches else "all_branch_items")
            pass_mode = getattr(pp, "pass_mode", "per_branch" if getattr(pp, "per_branch", False) else "single")
            if pass_mode == "chain":
                input_source = "previous_pass_output"

            if input_source == "previous_pass_output":
                if not prev_response or not results:
                    raise ValueError("This pass requires a previous pass output, but no previous pass has completed yet.")
                branch_names_str = f"[output of: {results[-1].pass_name}]"
                per_branch_inputs = [(branch_names_str, prev_response)]
                input_desc = "previous pass output"
            else:
                if input_source == "selected_branch_items":
                    source_names = list(pp.source_branches or [])
                    if not source_names:
                        raise ValueError("This pass is set to selected branch items, but no source branches were provided.")
                else:
                    source_names = list(branch_items_by_name.keys())

                if pass_mode == "per_branch":
                    # One LLM call per branch
                    per_branch_inputs = [
                        (bn, "\n".join(_format_item_for_pass(item) for item in branch_items_by_name.get(bn, [])))
                        for bn in source_names
                    ]
                    input_desc = f"{len(source_names)} branches"
                else:
                    all_items: List[Any] = []
                    for bn in source_names:
                        all_items.extend(branch_items_by_name.get(bn, []))
                    per_branch_inputs = [
                        (", ".join(source_names), "\n".join(_format_item_for_pass(item) for item in all_items))
                    ]
                    input_desc = f"{len(all_items)} items"

            branch_responses: List[str] = []
            for branch_name_str, branch_items_text in per_branch_inputs:
                if pass_mode == "per_branch" and len(per_branch_inputs) > 1:
                    emit(f"  → {branch_name_str}…")
                batches = _split_items_text(branch_items_text, pp.max_chars_per_batch)
                batch_responses: List[str] = []
                for b_idx, batch_text in enumerate(batches, start=1):
                    if len(batches) > 1:
                        emit(f"    batch {b_idx}/{len(batches)}…")
                    user_prompt = _render_post_pass_user_prompt(pp.user_prompt_template, branch_name_str, batch_text)
                    batch_resp = _run_post_pass_llm(
                        llm_fn,
                        system_prompt=pp.system_prompt,
                        user_prompt=user_prompt,
                        temperature=pp.temperature,
                        timeout_seconds=pp.timeout_seconds,
                        num_predict=pp.num_predict,
                    )
                    batch_responses.append(batch_resp)
                if pass_mode == "map_reduce":
                    branch_responses.append(_reduce_post_pass_outputs(batch_responses, pp, llm_fn, emit))
                else:
                    branch_responses.append("\n\n".join(batch_responses))

            response = "\n\n---\n\n".join(branch_responses)

            elapsed = round(time.perf_counter() - t0, 2)
            res = PostBranchPassResult(
                pass_name=pp.name,
                output_heading=pp.effective_heading(),
                response_text=response,
                status="ok",
                elapsed_seconds=elapsed,
            )
            emit(f"  ✓ Post-pass complete ({elapsed}s, {input_desc} in)")
        except Exception as exc:
            elapsed = round(time.perf_counter() - t0, 2)
            emit(f"  ✗ Post-pass failed: {exc}")
            res = PostBranchPassResult(
                pass_name=pp.name,
                output_heading=pp.effective_heading(),
                status="error",
                error=str(exc),
                elapsed_seconds=elapsed,
            )

        _write_checkpoint(ckpt_path, "post_pass_result", res.to_dict())
        results.append(res)
        if res.status == "ok" and res.response_text:
            prev_response = res.response_text

    return results


# ---------------------------------------------------------------------------
# run_project — public entry point
# ---------------------------------------------------------------------------

def run_project(
    project: ProjectConfig,
    *,
    db_dsn: Optional[str] = None,
    emit: Optional[Callable[[str], None]] = None,
    config_path: str = "configs/extraction.yaml",
    verbose: Optional[bool] = None,
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

    # ── Verbose run directory ─────────────────────────────────────────────
    verbose_run_dir: Optional[Path] = None
    if config.verbose:
        verbose_run_dir = Path(config.verbose_dir) / f"{project.slug}_{run_id}"
        verbose_run_dir.mkdir(parents=True, exist_ok=True)

    emit(f"Starting extraction project: {project.name}")
    if verbose_run_dir:
        emit(f"Verbose mode ON → {verbose_run_dir}")
    emit(f"Checkpoint: {ckpt_path}")

    # ── 1. Ingest documents ──────────────────────────────────────────────
    collection_id = project.effective_collection_id()
    emit(f"Collection: {collection_id}")

    for src in project.document_sources:
        resolved = _resolve_source_path(src.path)
        if resolved is None:
            emit(f"  ✗ File not found: {src.path}")
            continue
        src.path = str(resolved)  # normalise to resolved path
        if _is_ingested(actual_db_dsn, src.path, collection_id):
            emit(f"  ✓ Already ingested: {src.effective_label()}")
        else:
            _ingest_source(src.path, collection_id, actual_db_dsn, emit)

    # ── 2. Build priority map and propagate to branches ──────────────────
    priority_map = build_priority_map(project.document_sources)
    for branch in project.branches:
        # Project-level weights are the defaults; branch-level overrides win
        merged = {**priority_map, **branch.source_priority}
        branch.source_priority = merged

    # ── 3. Build corpus handle ───────────────────────────────────────────
    emit("Building corpus index…")
    corpus = _build_corpus_handle(actual_db_dsn, collection_id)
    total_chunks = len(corpus.chunk_rows)
    emit(f"  → {total_chunks} chunks loaded for collection '{collection_id}'")

    if total_chunks == 0:
        return ProjectRunResult(
            project_slug=project.slug,
            error="No chunks found in collection — check that documents were ingested successfully.",
            elapsed_seconds=round(time.perf_counter() - t0, 2),
        )

    # ── 4. Run branches ──────────────────────────────────────────────────
    branch_results: List[BranchResult] = []
    enabled_branches = [b for b in project.branches if b.enabled]
    emit(f"Running {len(enabled_branches)} branch(es)…")

    # Warm the LLM once so scan/synthesis/HyDE don't stall on cold model load
    emit("Warming LLM…")
    try:
        llm_fn(system_prompt="ping", user_prompt="ping", temperature=0.0, timeout_seconds=180.0)
        emit("  ✓ LLM ready")
    except Exception as _warm_exc:
        emit(f"  ⚠ LLM warmup failed (will retry per-branch): {_warm_exc}")

    for i, branch in enumerate(enabled_branches, start=1):
        emit(f"\nBranch {i}/{len(enabled_branches)}: {branch.name}")
        branch_vdir = (verbose_run_dir / _slugify(branch.name)) if verbose_run_dir else None
        result = run_branch(branch, corpus, config, llm_fn, emit=emit, verbose_dir=branch_vdir)
        # Checkpoint to disk immediately and free items from RAM
        _write_checkpoint(ckpt_path, "branch_result", {
            "branch_name": result.branch_name,
            "output_heading": result.output_heading,
            "status": result.status,
            "stats": asdict(result.stats),
            "items": [item.to_dict() for item in result.items],
        })
        emit(f"  → Saved {len(result.items)} item(s) to checkpoint")
        result.items.clear()
        branch_results.append(result)

    # ── 5. Post-branch passes ──────────────────────────────────────────
    post_pass_results = _run_post_passes(project, ckpt_path, llm_fn, emit)

    # ── 6. Build report ───────────────────────────────────────────────────
    emit("\nBuilding report…")
    # Reload full items from checkpoint (items were cleared after each branch to save RAM)
    branch_results = _load_checkpoint_branch_results(ckpt_path)
    report_md = assemble_report(
        project,
        branch_results,
        post_pass_results=post_pass_results,
        include_page_citations=True,
        include_branch_stats=True,
    )
    report_path = save_report(report_md, project)
    emit(f"Report saved to: {report_path}")

    # ── 7. Optionally clean up collection ────────────────────────────────
    if not project.keep_collection_after_run:
        emit("Removing temporary collection…")
        try:
            from db.client import delete_collection
            delete_collection(actual_db_dsn, collection_id, clear_chunks=True)
            emit("  ✓ Collection removed")
        except Exception as exc:
            emit(f"  ⚠ Could not remove collection: {exc}")

    elapsed = round(time.perf_counter() - t0, 2)
    total_items = sum(len(br.items) for br in branch_results)
    emit(f"\n✓ Extraction complete — {total_items} item(s) across {len(branch_results)} branch(es) in {elapsed}s")

    return ProjectRunResult(
        project_slug=project.slug,
        branch_results=branch_results,
        post_pass_results=post_pass_results,
        report_path=str(report_path),
        report_markdown=report_md,
        elapsed_seconds=elapsed,
        checkpoint_path=str(ckpt_path),
    )
