"""Data models for branched extraction.

BranchConfig  — one extraction topic defined by the user
DocumentSource — one input document with its role/weight
ProjectConfig  — top-level run config (all documents + all branches)
ExtractionItem — a single extracted bullet from a branch run
BranchResult   — output of running one branch
BranchStats    — run statistics for a branch
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Role → weight defaults
# ---------------------------------------------------------------------------

ROLE_DEFAULT_WEIGHTS: Dict[str, float] = {
    "primary":   1.0,
    "addendum":  1.5,
    "reference": 0.6,
    "custom":    1.0,
}


# ---------------------------------------------------------------------------
# BranchConfig
# ---------------------------------------------------------------------------

@dataclass
class BranchConfig:
    """One extraction branch: a named topic to pull from the document corpus."""

    name: str
    """Human-readable branch name; used as the section heading unless output_heading is set."""

    mode: Literal["keyword", "semantic"] = "keyword"
    """
    keyword  — user-supplied terms hard-filter candidate chunks and appear in the prompt.
    semantic — topic_description drives HyDE + dense retrieval with no hard filter.
    """

    # keyword mode
    keywords: List[str] = field(default_factory=list)
    """Exact phrases / regex patterns (keyword mode). Used for filtering + prompt injection."""

    # semantic mode
    topic_description: str = ""
    """Natural-language description of the topic (semantic mode). Used as retrieval query + prompt."""

    # shared
    output_heading: str = ""
    """Section heading in the report. Defaults to `name` if blank."""

    output_format: Literal["bullets", "numbered", "table"] = "bullets"

    prompt_context: Optional[str] = None
    """Additional context injected into the synthesis prompt for this branch.
    Use to guide selection — e.g. 'Focus on formal definitions only' or
    'Prioritize content from chapters 5–10'. Does not replace the prompt structure."""

    source_priority: Dict[str, float] = field(default_factory=dict)
    """filename → weight multiplier.  Populated automatically from ProjectConfig or set manually."""

    max_items: int = 200
    """Hard cap on items extracted per branch."""

    enabled: bool = True
    """Set False to skip this branch without deleting it."""

    keywords_are_regex: bool = False
    """When True, all entries in `keywords` are treated as raw regex patterns.
    When False (default), keywords are plain strings and are regex-escaped before
    compiling. Use True only if you intentionally wrote regex syntax (e.g. r"\\bPLC\\b")."""

    # flag-library metadata (not used during extraction)
    origin: str = ""
    """Optional provenance note shown in the flag library UI."""

    def effective_heading(self) -> str:
        return self.output_heading or self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BranchConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - valid_keys
        if unknown:
            logger.warning("BranchConfig: unknown keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# DocumentSource
# ---------------------------------------------------------------------------

@dataclass
class DocumentSource:
    """One input document and its authority role within the project corpus."""

    path: str
    """Absolute or workspace-relative path to the file."""

    role: Literal["primary", "addendum", "reference", "custom"] = "primary"

    weight: float = 1.0
    """Score multiplier for chunks from this document. Auto-set from role unless role=custom."""

    label: str = ""
    """Display name shown in the report. Defaults to filename if blank."""

    def effective_label(self) -> str:
        return self.label or Path(self.path).name

    def effective_weight(self) -> float:
        if self.role == "custom":
            return self.weight
        return ROLE_DEFAULT_WEIGHTS[self.role]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DocumentSource":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - valid_keys
        if unknown:
            logger.warning("DocumentSource: unknown keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# PostBranchPass / PostBranchPassResult
# ---------------------------------------------------------------------------

@dataclass
class PostBranchPass:
    """An LLM call that runs after all branches complete, in series.

    user_prompt_template supports two placeholders:
      {items_text}    — bullet-formatted items from source_branches (or all branches)
      {branch_names}  — comma-separated names of the source branches
    """

    name: str
    enabled: bool = True
    output_heading: str = ""
    input_source: Literal["all_branch_items", "selected_branch_items", "previous_pass_output"] = "all_branch_items"
    """What this pass consumes as input."""

    pass_mode: Literal["single", "per_branch", "map_reduce", "chain"] = "single"
    """How this pass executes over the chosen input."""

    source_branches: List[str] = field(default_factory=list)
    """Branch names to draw items from when input_source='selected_branch_items'."""
    system_prompt: str = ""
    user_prompt_template: str = ""
    temperature: float = 0.1
    timeout_seconds: float = 180.0
    max_chars_per_batch: int = 10000
    """Maximum characters of items_text sent in a single LLM call.
    If the input exceeds this, it is split into batches and responses are
    concatenated, or reduced when pass_mode='map_reduce'. 0 = no limit."""

    num_predict: int = -1
    """Maximum tokens the LLM may generate in a single call. -1 = model default.
    Set to 4096 or higher for synthesis passes that need long outputs."""

    per_branch: bool = False
    """When True, call the LLM once per source branch and concatenate the results.
    Each call receives only that branch's items and the {branch_names} placeholder
    resolves to that single branch name.  Use for per-branch summaries."""

    def effective_heading(self) -> str:
        return self.output_heading or self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PostBranchPass":
        raw = dict(d)
        if "input_source" not in raw:
            source_branches = raw.get("source_branches") or []
            raw["input_source"] = "selected_branch_items" if source_branches else "all_branch_items"
        if "pass_mode" not in raw:
            raw["pass_mode"] = "per_branch" if raw.get("per_branch") else "single"
        if raw.get("pass_mode") == "chain":
            raw["input_source"] = "previous_pass_output"

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw) - valid_keys - {"per_branch"}
        if unknown:
            logger.warning("PostBranchPass: unknown keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in raw.items() if k in valid_keys})


@dataclass
class PostBranchPassResult:
    """Output of one post-branch LLM pass."""

    pass_name: str
    output_heading: str
    response_text: str = ""
    status: Literal["ok", "error"] = "ok"
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PostBranchPassResult":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# ProjectConfig
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Top-level config for a single extraction run. Persisted to flag_library/projects/."""

    slug: str
    """URL-safe identifier, e.g. 'springfield_wwtp_2026'."""

    name: str
    """Human display name, e.g. 'City of Springfield WWTP — April 2026 RFP'."""

    document_sources: List[DocumentSource] = field(default_factory=list)
    """Ordered list of input documents with roles and weights."""

    branches: List[BranchConfig] = field(default_factory=list)
    """Ordered list of branches to run."""

    collection_id: Optional[str] = None
    """RAG collection for ingested chunks. Auto-generated from slug if None."""

    keep_collection_after_run: bool = True
    """Keep ingested chunks in SQLite after extraction finishes."""

    report_output_path: str = "data/reports"
    """Directory where the output .md report is written."""

    cross_branch_dedup: bool = False
    """When True, items that appear in multiple branches are deduplicated across
    branches after all branch runs complete — only the copy from the highest-
    priority source is kept.  Set False (default) if you want the same
    content to appear under each relevant section heading."""

    created_at: str = ""
    updated_at: str = ""
    post_branch_passes: List[PostBranchPass] = field(default_factory=list)
    """Ordered list of LLM passes to run after all branches complete, in series."""

    def effective_collection_id(self) -> str:
        return self.collection_id or f"extraction_{self.slug}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectConfig":
        sources = [DocumentSource.from_dict(s) for s in d.pop("document_sources", [])]
        branches = [BranchConfig.from_dict(b) for b in d.pop("branches", [])]
        post_passes = [PostBranchPass.from_dict(p) for p in d.pop("post_branch_passes", [])]
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - valid_keys
        if unknown:
            logger.warning("ProjectConfig: unknown keys ignored: %s", sorted(unknown))
        obj = cls(**{k: v for k, v in d.items() if k in valid_keys})
        obj.document_sources = sources
        obj.branches = branches
        obj.post_branch_passes = post_passes
        return obj

    def save(self, projects_dir: str = "data/flag_library/projects") -> None:
        p = Path(projects_dir)
        p.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = self.updated_at
        (p / f"{self.slug}.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, slug: str, projects_dir: str = "data/flag_library/projects") -> "ProjectConfig":
        path = Path(projects_dir) / f"{slug}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# ExtractionItem
# ---------------------------------------------------------------------------

@dataclass
class ExtractionItem:
    """One extracted bullet or row from a branch synthesis pass."""

    text: str
    """The extracted item text (as written in the document)."""

    branch_name: str

    source_chunk_id: str = ""
    """chunk_id in SQLite (for traceability)."""

    source_filename: str = ""
    """Original document filename."""

    source_page: int = 0
    """Page number from chunk metadata (0 = unknown)."""

    confidence: float = 0.0
    """Synthesis pass confidence estimate (0–1).  May be 0 if not available."""

    priority_weight: float = 1.0
    """Source document priority weight at time of extraction."""

    addendum_override: bool = False
    """True if this item came from an addendum/high-priority source (marks ⚑ in report)."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExtractionItem":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ---------------------------------------------------------------------------
# BranchStats / BranchResult
# ---------------------------------------------------------------------------

@dataclass
class BranchStats:
    chunks_retrieved: int = 0
    chunks_after_filter: int = 0
    batches_scanned: int = 0
    ids_selected_by_scan: int = 0
    items_before_dedup: int = 0
    items_after_dedup: int = 0
    elapsed_seconds: float = 0.0
    error: Optional[str] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BranchStats":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class BranchResult:
    branch_name: str
    output_heading: str
    items: List[ExtractionItem] = field(default_factory=list)
    stats: BranchStats = field(default_factory=BranchStats)
    status: Literal["ok", "empty", "error"] = "ok"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BranchResult":
        items = [ExtractionItem.from_dict(i) for i in d.get("items", [])]
        stats = BranchStats.from_dict(d.get("stats", {}))
        return cls(
            branch_name=d["branch_name"],
            output_heading=d["output_heading"],
            items=items,
            stats=stats,
            status=d.get("status", "ok"),
        )


# ---------------------------------------------------------------------------
# Project-level helpers
# ---------------------------------------------------------------------------

def build_priority_map(sources: List[DocumentSource]) -> Dict[str, float]:
    """filename → effective weight for all project sources."""
    return {Path(s.path).name: s.effective_weight() for s in sources}


def resolve_priority(filename: str, source_priority: Dict[str, float]) -> float:
    """Return the weight for a chunk based on its source filename (case-insensitive substring match)."""
    fn_lower = filename.lower()
    for pattern, weight in source_priority.items():
        if pattern.lower() in fn_lower:
            return weight
    return 1.0
