"""Single-branch extraction runner.

run_branch(branch, corpus, config, llm_fn, emit) -> BranchResult

Executes all 5 stages described in Section 5 of LargeDocIngest.md:
  1. Retrieval - dense + BM25 + keyword filtering
  2. Batching  - assign short IDs, split into scan batches
  3. Scan pass - LLM identifies relevant chunk IDs
  4. Full-text selection - LLM selects which rehydrated chunks become items
  5. Deduplication - Jaccard near-duplicate removal

The caller owns the LLM callable and the progress emitter, keeping this
module free of I/O side effects other than the LLM calls.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from extraction.batch import (
    assign_ids_and_batch,
    assign_ids_for_synthesis,
    parse_scan_response,
    rehydrate,
)
from extraction.cancel import CancelCheck, ExtractionCancelled, raise_if_cancelled
from extraction.branch_config import (
    BranchConfig,
    BranchResult,
    BranchStats,
    ExtractionItem,
    resolve_priority,
)
from extraction.dedup import dedup_items
from extraction.evidence_quality import (
    classify_evidence_text,
    is_appendix_or_toc_chunk,
    strip_leading_reference_noise,
    strip_html,
)
from extraction.prompts import build_scan_prompts, build_synthesis_prompts
from extraction.source_kinds import infer_source_kind


# ---------------------------------------------------------------------------
# CorpusHandle — thin wrapper around a collection's chunks
# ---------------------------------------------------------------------------

@dataclass
class CorpusHandle:
    """Pre-loaded corpus for a project's collection.

    Built once by project_runner.py and shared across all branches to avoid
    repeated database reads.
    """

    db_dsn: str
    collection_id: str
    chunk_rows: List[Dict[str, Any]]
    """All raw chunk dicts from PostgreSQL for this collection, in page order."""

    embed_backend: str = "ollama"
    embed_model_name: str = "qwen3-embedding:latest"
    embed_dimension: int = 1024
    ollama_base_url: str = "http://localhost:11434"


# ---------------------------------------------------------------------------
# ExtractionConfig — thin view into configs/extraction.yaml
# ---------------------------------------------------------------------------

@dataclass
class ExtractionConfig:
    batch_max_chars: int = 10000
    scan_chunk_preview_chars: int = 80
    max_candidate_chunks: int = 500
    synthesis_max_chars: int = 10000
    synthesis_overlap: bool = True
    id_prefix: str = ""
    top_k_per_branch: int = 500
    keyword_score_boost: float = 0.05
    use_hyde_in_semantic_mode: bool = True
    scan_pass_temperature: float = 0.05
    synthesis_pass_temperature: float = 0.15
    scan_pass_timeout_seconds: float = 60.0
    synthesis_pass_timeout_seconds: float = 120.0
    max_retries: int = 2
    item_jaccard_threshold: float = 0.85
    use_cross_encoder: bool = False
    """Enable cross-encoder reranking during branch retrieval.  Off by default —
    cross-encoder over 500 candidates is very slow."""
    verbose: bool = False
    verbose_dir: str = "data/extraction_verbose"

    @classmethod
    def from_yaml(cls, path: str = "configs/extraction.yaml") -> "ExtractionConfig":
        try:
            import yaml
            with open(path, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh) or {}
        except Exception:
            return cls()
        cfg = cls()
        batching = raw.get("batching", {})
        cfg.batch_max_chars = batching.get("batch_max_chars", cfg.batch_max_chars)
        cfg.scan_chunk_preview_chars = batching.get("scan_chunk_preview_chars", cfg.scan_chunk_preview_chars)
        cfg.max_candidate_chunks = batching.get("max_candidate_chunks", cfg.max_candidate_chunks)
        cfg.synthesis_max_chars = batching.get("synthesis_max_chars", cfg.synthesis_max_chars)
        cfg.synthesis_overlap = batching.get("synthesis_overlap", cfg.synthesis_overlap)
        cfg.id_prefix = batching.get("id_prefix", cfg.id_prefix)
        retrieval = raw.get("retrieval", {})
        cfg.top_k_per_branch = retrieval.get("top_k_per_branch", cfg.top_k_per_branch)
        cfg.keyword_score_boost = retrieval.get("keyword_score_boost", cfg.keyword_score_boost)
        cfg.use_hyde_in_semantic_mode = retrieval.get("use_hyde_in_semantic_mode", cfg.use_hyde_in_semantic_mode)
        llm = raw.get("llm", {})
        cfg.scan_pass_temperature = llm.get("scan_pass_temperature", cfg.scan_pass_temperature)
        cfg.synthesis_pass_temperature = llm.get("synthesis_pass_temperature", cfg.synthesis_pass_temperature)
        cfg.scan_pass_timeout_seconds = llm.get("scan_pass_timeout_seconds", cfg.scan_pass_timeout_seconds)
        cfg.synthesis_pass_timeout_seconds = llm.get("synthesis_pass_timeout_seconds", cfg.synthesis_pass_timeout_seconds)
        cfg.max_retries = llm.get("max_retries", cfg.max_retries)
        cfg.item_jaccard_threshold = raw.get("deduplication", {}).get("item_jaccard_threshold", cfg.item_jaccard_threshold)
        cfg.use_cross_encoder = raw.get("retrieval", {}).get("use_cross_encoder", cfg.use_cross_encoder)
        verbose_cfg = raw.get("verbose", {})
        cfg.verbose = verbose_cfg.get("enabled", cfg.verbose)
        cfg.verbose_dir = verbose_cfg.get("output_dir", cfg.verbose_dir)
        return cfg


# ---------------------------------------------------------------------------
# Keyword regex helpers
# ---------------------------------------------------------------------------

def compile_keyword_regexes(keywords: List[str], keywords_are_regex: bool = False) -> Optional[re.Pattern]:
    """Compile keyword list to a single OR regex.  Returns None for empty list.

    If *keywords_are_regex* is True, entries are used verbatim as regex patterns.
    Otherwise they are regex-escaped so that special chars (e.g. in ``C++``,
    ``1.5x``) are treated as literal text.
    """
    if not keywords:
        return None
    if keywords_are_regex:
        parts = keywords
    else:
        parts = [re.escape(kw) for kw in keywords]
    try:
        return re.compile("|".join(parts), re.IGNORECASE)
    except re.error:
        # Fall back to fully-escaped version if any raw pattern is invalid
        return re.compile("|".join(re.escape(kw) for kw in keywords), re.IGNORECASE)


# ---------------------------------------------------------------------------
# Verbose dump helpers
# ---------------------------------------------------------------------------

def _vdump(vdir: Path, filename: str, content: str) -> None:
    """Write text content to vdir/filename, creating the directory as needed."""
    vdir.mkdir(parents=True, exist_ok=True)
    (vdir / filename).write_text(content, encoding="utf-8")


def _vdump_json(vdir: Path, filename: str, data: Any) -> None:
    """Write data as indented JSON to vdir/filename."""
    _vdump(vdir, filename, json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _fmt_prompt(system: str, user: str) -> str:
    """Format system+user prompts into one readable text block."""
    return f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}\n"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _retrieve_candidates(
    branch: BranchConfig,
    corpus: CorpusHandle,
    config: ExtractionConfig,
    emit: Callable[[str], None],
) -> List[Dict[str, Any]]:
    """Dense + BM25 retrieval over the corpus for one branch.

    Returns a list of chunk dicts enriched with "score", "source_filename",
    sorted by descending score, capped at config.max_candidate_chunks.
    """
    from retrieval.query import retrieve, RetrievalFilters
    from utils.runtime_defaults import DEFAULT_DB_DSN

    emit(f"Retrieving chunks for branch '{branch.name}'…")

    query_text = (
        " ".join(branch.keywords) if branch.mode == "keyword" and branch.keywords
        else branch.topic_description or branch.name
    )

    use_hyde = branch.mode == "semantic" and config.use_hyde_in_semantic_mode

    filters = RetrievalFilters(collection_id=corpus.collection_id)

    result = retrieve(
        query_text,
        db_dsn=corpus.db_dsn,
        top_k=config.top_k_per_branch,
        filters=filters,
        rerank_enabled=True,
        cross_encoder_enabled=config.use_cross_encoder,
        internet_fallback_enabled=False,
        hyde_enabled=use_hyde,
        embed_backend=corpus.embed_backend,
        embed_model_name=corpus.embed_model_name,
        embed_dimension=corpus.embed_dimension,
        ollama_base_url=corpus.ollama_base_url,
        rerank_diversity_penalty=0.0,  # skip O(n²) MMR — extraction uses all candidates anyway
        progress_fn=emit,
    )

    # Convert RetrievedChunk objects to plain dicts
    chunks: List[Dict[str, Any]] = []
    for hit in result.hits:
        d = {
            "chunk_id": hit.chunk_id,
            "text": hit.text,
            "score": hit.score,
            "source_filename": getattr(hit, "filename", "") or getattr(hit, "source_name", ""),
            "page_start": getattr(hit, "page_start", 0) or 0,
            "doc_id": getattr(hit, "doc_id", ""),
        }
        chunks.append(d)

    return chunks


def _apply_keyword_filter_and_boost(
    chunks: List[Dict[str, Any]],
    branch: BranchConfig,
    config: ExtractionConfig,
) -> List[Dict[str, Any]]:
    """Hard-filter (keyword mode) + score boost + source priority multiplier."""
    if branch.mode == "keyword" and branch.keywords:
        pattern = compile_keyword_regexes(branch.keywords, branch.keywords_are_regex)
        if pattern:
            chunks = [c for c in chunks if pattern.search(c.get("text", ""))]
            # Boost: +0.05 per keyword match, capped at 5
            compiled_kws = [compile_keyword_regexes([kw], branch.keywords_are_regex) for kw in branch.keywords]
            for chunk in chunks:
                text = chunk.get("text", "")
                match_count = sum(1 for kp in compiled_kws if kp and kp.search(text))
                chunk["score"] = chunk.get("score", 0.0) + config.keyword_score_boost * min(match_count, 5)

    # Source priority multiplier
    if branch.source_priority:
        for chunk in chunks:
            filename = chunk.get("source_filename", "")
            weight = resolve_priority(filename, branch.source_priority)
            chunk["score"] = chunk.get("score", 0.0) * weight
            chunk["priority_weight"] = weight
    else:
        for chunk in chunks:
            chunk.setdefault("priority_weight", 1.0)

    # Drop obviously non-substantive appendix / TOC / admin chunks early so
    # they do not dominate scan and synthesis passes. Keep the original pool as
    # a fallback if everything matched the filter.
    original_chunks = list(chunks)
    filtered_chunks = [
        chunk for chunk in chunks
        if not is_appendix_or_toc_chunk(chunk.get("text", ""))
    ]
    if filtered_chunks:
        chunks = filtered_chunks

    # Sort descending and cap
    chunks.sort(key=lambda c: c.get("score", 0.0), reverse=True)
    if not chunks:
        chunks = original_chunks
    return chunks[:config.max_candidate_chunks]


# ---------------------------------------------------------------------------
# LLM helpers (scan + synthesis)
# ---------------------------------------------------------------------------

def _call_llm_with_retry(
    llm_fn: Callable,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    timeout_seconds: float,
    max_retries: int,
    cancel_check: CancelCheck = None,
) -> str:
    """Call the LLM, retrying up to max_retries on exception."""
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        raise_if_cancelled(cancel_check)
        try:
            return llm_fn(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
            ) or ""
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(1)
    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts") from last_exc


def _scan_batch(
    batch,
    branch: BranchConfig,
    config: ExtractionConfig,
    llm_fn: Callable,
    cancel_check: CancelCheck = None,
) -> Tuple[List[int], str, str, str]:
    """Run one scan-pass batch.

    Returns
    -------
    (selected_ids, system_prompt, user_prompt, raw_response)
    """
    system, user = build_scan_prompts(batch, branch, id_prefix=config.id_prefix)
    response = _call_llm_with_retry(
        llm_fn, system, user,
        temperature=config.scan_pass_temperature,
        timeout_seconds=config.scan_pass_timeout_seconds,
        max_retries=config.max_retries,
        cancel_check=cancel_check,
    )
    return parse_scan_response(response), system, user, response


def _synthesis_batch(
    batch: List[Tuple[int, Dict[str, Any]]],
    branch: BranchConfig,
    config: ExtractionConfig,
    llm_fn: Callable,
    cancel_check: CancelCheck = None,
) -> Tuple[List[int], str, str, str]:
    """Run one synthesis pass.

    Returns
    -------
    (selected_ids, system_prompt, user_prompt, raw_response)
    """
    system, user = build_synthesis_prompts(batch, branch)
    response = _call_llm_with_retry(
        llm_fn, system, user,
        temperature=config.synthesis_pass_temperature,
        timeout_seconds=config.synthesis_pass_timeout_seconds,
        max_retries=config.max_retries,
        cancel_check=cancel_check,
    )
    return parse_scan_response(response), system, user, response


# ---------------------------------------------------------------------------
# run_branch — public entry point
# ---------------------------------------------------------------------------

def run_branch(
    branch: BranchConfig,
    corpus: CorpusHandle,
    config: ExtractionConfig,
    llm_fn: Callable,
    emit: Optional[Callable[[str], None]] = None,
    verbose_dir: Optional[Path] = None,
    cancel_check: CancelCheck = None,
) -> BranchResult:
    """Execute one extraction branch end-to-end.

    Parameters
    ----------
    branch      : the branch configuration to run.
    corpus      : pre-loaded corpus handle with collection context.
    config      : extraction config from extraction.yaml.
    llm_fn      : callable(system_prompt, user_prompt, temperature, timeout_seconds) → str.
                  Typically wraps llm.generation.ollama_chat().
    emit        : optional callable(str) for progress messages (e.g. SSE sender).
                  If None, messages are silently discarded.
    verbose_dir : if set, every pipeline step is saved to disk under this directory.
                  Created automatically if it doesn't exist.

    Returns
    -------
    BranchResult with all extracted items and run statistics.
    """
    if emit is None:
        emit = lambda _: None  # noqa: E731

    if not branch.enabled:
        return BranchResult(
            branch_name=branch.name,
            output_heading=branch.effective_heading(),
            status="empty",
            stats=BranchStats(error="Branch disabled"),
        )

    t0 = time.perf_counter()
    stats = BranchStats()
    vdir = verbose_dir  # shorthand; None means verbose is off

    try:
        raise_if_cancelled(cancel_check)
        # ── Stage 1: Retrieval ───────────────────────────────────────────
        raw_chunks = _retrieve_candidates(branch, corpus, config, emit)
        stats.chunks_retrieved = len(raw_chunks)

        pool = _apply_keyword_filter_and_boost(raw_chunks, branch, config)
        stats.chunks_after_filter = len(pool)
        emit(f"  → {stats.chunks_after_filter} chunks after filter (from {stats.chunks_retrieved} retrieved)")

        if vdir:
            _vdump_json(vdir, "01_pool.json", [
                {k: v for k, v in c.items() if k != "embedding"} for c in pool
            ])
            emit(f"  [verbose] 01_pool.json ({len(pool)} chunks)")

        if not pool:
            emit(f"  → No candidates found for branch '{branch.name}'")
            return BranchResult(
                branch_name=branch.name,
                output_heading=branch.effective_heading(),
                status="empty",
                stats=stats,
            )

        # ── Stage 2: Batching ────────────────────────────────────────────
        id_map, scan_batches = assign_ids_and_batch(
            pool,
            batch_max_chars=config.batch_max_chars,
            preview_chars=config.scan_chunk_preview_chars,
            id_prefix=config.id_prefix,
        )
        stats.batches_scanned = len(scan_batches)
        emit(f"  → Scan: {len(pool)} chunks across {len(scan_batches)} batch(es)")

        if vdir:
            _vdump_json(vdir, "02_id_map.json", {
                str(sid): {k: v for k, v in chunk.items() if k != "embedding"}
                for sid, chunk in id_map.items()
            })
            emit(f"  [verbose] 02_id_map.json ({len(id_map)} entries)")

        # ── Stage 3: Scan pass ───────────────────────────────────────────
        all_selected_ids: List[int] = []
        for batch_idx, batch in enumerate(scan_batches, start=1):
            raise_if_cancelled(cancel_check)
            emit(f"  → Scan batch {batch_idx}/{len(scan_batches)}…")
            selected, sys_p, usr_p, raw_resp = _scan_batch(
                batch, branch, config, llm_fn, cancel_check=cancel_check
            )
            all_selected_ids.extend(selected)
            if vdir:
                tag = f"03_scan_{batch_idx:02d}"
                _vdump(vdir, f"{tag}_prompt.txt", _fmt_prompt(sys_p, usr_p))
                _vdump(vdir, f"{tag}_response.txt", raw_resp)
                _vdump(vdir, f"{tag}_ids.txt", f"Parsed IDs: {selected}\n")
                emit(f"  [verbose] {tag}_*.txt (prompt / response / ids)")
        stats.ids_selected_by_scan = len(set(all_selected_ids))
        emit(f"  → Scan selected {stats.ids_selected_by_scan} chunk(s)")

        rehydrated = rehydrate(all_selected_ids, id_map)
        if vdir:
            _vdump_json(vdir, "04_rehydrated.json", [
                {k: v for k, v in c.items() if k != "embedding"} for c in rehydrated
            ])
            emit(f"  [verbose] 04_rehydrated.json ({len(rehydrated)} chunks)")

        if not rehydrated:
            emit(f"  → No chunks selected by scan pass for branch '{branch.name}'")
            return BranchResult(
                branch_name=branch.name,
                output_heading=branch.effective_heading(),
                status="empty",
                stats=stats,
            )

        # ── Stage 4: Full-text selection ────────────────────────────────
        synth_id_map, synthesis_batches = assign_ids_for_synthesis(
            rehydrated, config.synthesis_max_chars
        )
        selected_synth_ids: List[int] = []
        for sb_idx, sub_batch in enumerate(synthesis_batches, start=1):
            raise_if_cancelled(cancel_check)
            if len(synthesis_batches) > 1:
                emit(f"  → Synthesis sub-batch {sb_idx}/{len(synthesis_batches)}…")
            else:
                emit(f"  → Synthesis pass ({len(rehydrated)} chunk(s))…")
            ids, sys_p, usr_p, raw_resp = _synthesis_batch(
                sub_batch, branch, config, llm_fn, cancel_check=cancel_check
            )
            selected_synth_ids.extend(ids)
            if vdir:
                tag = f"05_synth_{sb_idx:02d}"
                _vdump(vdir, f"{tag}_prompt.txt", _fmt_prompt(sys_p, usr_p))
                _vdump(vdir, f"{tag}_response.txt", raw_resp)
                _vdump(vdir, f"{tag}_ids.txt", f"Parsed IDs: {ids}\n")
                emit(f"  [verbose] {tag}_*.txt (prompt / response / ids)")

        # Deduplicate while preserving selection order
        seen_syn: set[int] = set()
        final_chunks: List[Dict[str, Any]] = []
        for sid in selected_synth_ids:
            if sid not in seen_syn:
                seen_syn.add(sid)
                chunk = synth_id_map.get(sid)
                if chunk:
                    final_chunks.append(chunk)

        stats.items_before_dedup = len(final_chunks)

        if vdir:
            _vdump_json(vdir, "06_final_chunks.json", [
                {k: v for k, v in c.items() if k != "embedding"} for c in final_chunks
            ])
            emit(f"  [verbose] 06_final_chunks.json ({len(final_chunks)} chunks)")

        # Build ExtractionItem objects directly from source chunk text.
        items: List[ExtractionItem] = []
        for chunk in final_chunks:
            priority = float(chunk.get("priority_weight", 1.0) or 1.0)
            clean_text = strip_leading_reference_noise(chunk.get("text", ""))
            items.append(ExtractionItem(
                text=clean_text,
                branch_name=branch.name,
                source_chunk_id=chunk.get("chunk_id", ""),
                source_filename=chunk.get("source_filename", ""),
                source_page=int(chunk.get("page_start", 0) or 0),
                source_kind=infer_source_kind(
                    text=clean_text,
                    filename=str(chunk.get("source_filename", "") or ""),
                ),
                priority_weight=priority,
                addendum_override=priority > 1.0,
            ))

        substantive_items = [
            item for item in items
            if classify_evidence_text(item.text) == "substantive"
        ]
        if substantive_items:
            filtered_out = len(items) - len(substantive_items)
            if filtered_out:
                emit(f"  -> Filtered {filtered_out} low-information item(s)")
            items = substantive_items

        # ── Stage 5: Deduplication ───────────────────────────────────────
        items = dedup_items(items, threshold=config.item_jaccard_threshold)
        # Enforce max_items cap
        items = items[:branch.max_items]
        kept_chunk_ids = {item.source_chunk_id for item in items if item.source_chunk_id}
        evidence_chunks = [
            {
                "chunk_id": chunk.get("chunk_id", ""),
                "text": strip_leading_reference_noise(chunk.get("text", "")),
                "source_filename": chunk.get("source_filename", ""),
                "page_start": int(chunk.get("page_start", 0) or 0),
                "source_kind": infer_source_kind(
                    text=strip_leading_reference_noise(chunk.get("text", "")),
                    filename=str(chunk.get("source_filename", "") or ""),
                ),
                "priority_weight": float(chunk.get("priority_weight", 1.0) or 1.0),
            }
            for chunk in final_chunks
            if not kept_chunk_ids or chunk.get("chunk_id", "") in kept_chunk_ids
        ]
        stats.items_after_dedup = len(items)
        stats.elapsed_seconds = round(time.perf_counter() - t0, 2)

        if vdir:
            _vdump_json(vdir, "07_items_final.json", [i.to_dict() for i in items])
            emit(f"  [verbose] 07_items_final.json ({len(items)} items after dedup)")

        emit(f"  → Branch '{branch.name}': {stats.items_after_dedup} item(s) extracted ({stats.elapsed_seconds}s)")

        return BranchResult(
            branch_name=branch.name,
            output_heading=branch.effective_heading(),
            items=items,
            evidence_chunks=evidence_chunks,
            stats=stats,
            status="ok" if items else "empty",
        )

    except ExtractionCancelled:
        raise
    except Exception as exc:
        stats.error = str(exc)
        stats.elapsed_seconds = round(time.perf_counter() - t0, 2)
        emit(f"  ✗ Branch '{branch.name}' failed: {exc}")
        return BranchResult(
            branch_name=branch.name,
            output_heading=branch.effective_heading(),
            status="error",
            stats=stats,
        )
