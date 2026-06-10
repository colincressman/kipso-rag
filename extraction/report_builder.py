"""Report builder — assembles branch results into a markdown report.

assemble_report(project, branch_results, cross_refs, config) -> str

Produces the markdown document described in Section 7 of LargeDocIngest.md.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from extraction.branch_config import BranchResult, ExtractionItem, ProjectConfig
from extraction.dedup import dedup_cross_branch
from utils.text_utils import slugify_anchor as _slugify_anchor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _slugify(text: str) -> str:
    """Alias kept for internal use — delegates to utils.text_utils.slugify_anchor."""
    return _slugify_anchor(text)


def _make_toc(
    branch_results: List[BranchResult],
) -> str:
    """Build a markdown Table of Contents for all report sections."""
    lines = ["## Table of Contents", ""]
    n = 1
    for br in branch_results:
        anchor = _slugify(br.output_heading)
        count = len(br.items)
        suffix = f" ({count} items)" if count else " _(empty)_"
        lines.append(f"{n}. [{br.output_heading}](#{anchor}){suffix}")
        n += 1
    lines += ["", "---", ""]
    return "\n".join(lines)


def _format_item(
    item: ExtractionItem,
    idx: int,
    cross_refs: Dict[str, List[str]],
    branch_idx_map: Dict[str, int],
    addendum_marker: str = "⚑",
    cross_branch_marker: str = "⚠",
    include_page_citations: bool = True,
    output_format: str = "bullets",
) -> str:
    """Render one ExtractionItem as a markdown list entry."""
    # Citation suffix
    cite = ""
    if include_page_citations and item.source_filename:
        if item.source_page:
            cite = f" _(p. {item.source_page}, {item.source_filename})_"
        else:
            cite = f" _({item.source_filename})_"

    # Addendum override marker
    addon = f" {addendum_marker}" if item.addendum_override else ""

    # Cross-branch marker
    key = f"{item.branch_name}::{branch_idx_map.get(item.text, idx)}"
    cross_branches = cross_refs.get(key, [])
    cross = f" {cross_branch_marker} _also in: {', '.join(cross_branches)}_" if cross_branches else ""

    text = item.text.rstrip(".")

    if output_format == "numbered":
        return f"{idx + 1}. {text}{cite}{addon}{cross}"
    elif output_format == "table":
        # Table format — caller must build the table header separately
        return f"| {text} | {item.source_filename} | {item.source_page or ''} |{addon}"
    else:
        return f"• {text}{cite}{addon}{cross}"


# ---------------------------------------------------------------------------
# assemble_report
# ---------------------------------------------------------------------------

def assemble_report(
    project: ProjectConfig,
    branch_results: List[BranchResult],
    *,
    addendum_marker: str = "⚑",
    cross_branch_marker: str = "⚠",
    include_page_citations: bool = True,
    include_branch_stats: bool = True,
) -> str:
    """Assemble a full markdown extraction report.

    Parameters
    ----------
    project         : the ProjectConfig that was run.
    branch_results  : list of BranchResult in branch order.
    addendum_marker : marker appended to items from high-priority sources.
    cross_branch_marker : marker for items that appear in multiple branches.
    include_page_citations : append "(p. X, filename.pdf)" to items.
    include_branch_stats   : include the summary table.

    Returns
    -------
    Markdown string ready to write to a .md file.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc_names = [s.effective_label() for s in project.document_sources]
    doc_list = ", ".join(doc_names) if doc_names else "(no documents listed)"

    lines: List[str] = []

    # Header
    lines += [
        f"# Extraction Report: {project.name}",
        f"Generated: {now}",
        f"Documents: {doc_list}",
        f"Branches run: {len(branch_results)}",
        "",
        "---",
        "",
    ]

    # Table of Contents
    lines.append(_make_toc(branch_results))

    # Summary table
    if include_branch_stats:
        lines += [
            "## Summary",
            "",
            "| Branch | Items | Mode | Status |",
            "|--------|-------|------|--------|",
        ]
        for br in branch_results:
            mode = "keyword"
            for b in project.branches:
                if b.name == br.branch_name:
                    mode = b.mode
                    break
            status_icon = "✅" if br.status == "ok" else ("⚠️" if br.status == "empty" else "❌")
            count = len(br.items)
            lines.append(f"| {br.output_heading} | {count} | {mode} | {status_icon} |")
        lines += ["", "---", ""]

    # Cross-branch deduplication (with priority winner selection)
    overrides_log: List[Dict[str, Any]] = []
    if getattr(project, "cross_branch_dedup", True):
        branch_results, overrides_log = dedup_cross_branch(branch_results)

    # Build cross_refs for annotation (empty after dedup — kept for compat)
    cross_refs: Dict[str, List[str]] = {}

    # Per-branch sections
    for br in branch_results:
        lines += [
            f"## {br.output_heading}",
        ]

        # Find branch config for mode/format
        branch_cfg = next((b for b in project.branches if b.name == br.branch_name), None)
        mode_label = branch_cfg.mode if branch_cfg else "?"
        fmt = branch_cfg.output_format if branch_cfg else "bullets"
        source_names = ", ".join(doc_names) if doc_names else ""

        subtitle_parts = []
        if source_names:
            subtitle_parts.append(f"Source: {source_names}")
        subtitle_parts.append(f"Mode: {mode_label}")
        subtitle_parts.append(f"{len(br.items)} items")
        if br.stats.error:
            subtitle_parts.append(f"⚠ Error: {br.stats.error}")

        lines.append(f"_{' · '.join(subtitle_parts)}_")
        lines.append("")

        if not br.items:
            lines.append("_No items extracted for this branch._")
        else:
            # Build index for cross-ref lookup (item text → its position in branch)
            branch_idx_map: Dict[str, int] = {item.text: i for i, item in enumerate(br.items)}

            if fmt == "table":
                lines += [
                    "| Item | Source | Page |",
                    "|------|--------|------|",
                ]
                for i, item in enumerate(br.items):
                    lines.append(_format_item(
                        item, i, cross_refs, branch_idx_map,
                        addendum_marker=addendum_marker,
                        cross_branch_marker=cross_branch_marker,
                        include_page_citations=False,
                        output_format="table",
                    ))
            else:
                for i, item in enumerate(br.items):
                    lines.append(_format_item(
                        item, i, cross_refs, branch_idx_map,
                        addendum_marker=addendum_marker,
                        cross_branch_marker=cross_branch_marker,
                        include_page_citations=include_page_citations,
                        output_format=fmt,
                    ))

        lines += ["", "---", ""]

    # Overrides appendix
    if overrides_log:
        lines += [
            "## Source Overrides",
            "",
            f"_{len(overrides_log)} cross-branch duplicate(s) resolved by source priority._",
            "",
            "| Winner branch | Loser branch | Similarity | Priority |",
            "|---|---|---|---|",
        ]
        for rec in overrides_log:
            sim_pct = f"{rec['similarity']*100:.0f}%"
            lines.append(
                f"| {rec['winner_branch']} | {rec['loser_branch']} "
                f"| {sim_pct} | {rec['winner_priority']:.2f} |"
            )
        lines += ["", "---", ""]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# save_report
# ---------------------------------------------------------------------------

def save_report(markdown: str, project: ProjectConfig) -> Path:
    """Write the report to disk.  Returns the Path of the written file."""
    out_dir = Path(project.report_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{project.slug}_{timestamp}.md"
    path = out_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return path


def save_report_variant(markdown: str, project: ProjectConfig, *, suffix: str) -> Path:
    """Write a variant report beside the main report using a deterministic suffix."""
    out_dir = Path(project.report_output_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{project.slug}_{timestamp}_{suffix}.md"
    path = out_dir / filename
    path.write_text(markdown, encoding="utf-8")
    return path
