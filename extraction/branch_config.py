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
# SecondPassConfig / SecondPassResult
# ---------------------------------------------------------------------------

SECOND_PASS_TYPES = Literal[
    "organize_by_category",
    "summarize_by_category",
    "executive_summary",
    "key_findings",
    "next_actions",
    "assemble_report",
]


@dataclass
class SecondPassConfig:
    """A simple serial post-LLM pass that runs after all branch calls finish."""

    name: str
    pass_type: SECOND_PASS_TYPES
    enabled: bool = True
    title: str = ""
    source_branches: List[str] = field(default_factory=list)
    report_categories: List[str] = field(default_factory=list)
    reuse_categories_from_pass: str = ""
    instructions: str = ""
    system_prompt: str = ""
    user_prompt_template: str = ""
    max_chars_per_batch: int = 12000
    temperature: float = 0.1
    timeout_seconds: float = 180.0
    num_predict: int = -1

    def effective_heading(self) -> str:
        return self.title or self.name

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SecondPassConfig":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(d) - valid_keys
        if unknown:
            logger.warning("SecondPassConfig: unknown keys ignored: %s", sorted(unknown))
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


@dataclass
class SecondPassResult:
    """Output of one serial second-pass step."""

    pass_name: str
    output_heading: str
    response_text: str = ""
    artifact_type: str = ""
    artifact_data: Optional[Dict[str, Any]] = None
    status: Literal["ok", "error"] = "ok"
    error: Optional[str] = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SecondPassResult":
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

    report_output_path: str = "data/extraction_reports"
    """Directory where the output .md report is written."""

    cross_branch_dedup: bool = False
    """When True, items that appear in multiple branches are deduplicated across
    branches after all branch runs complete — only the copy from the highest-
    priority source is kept.  Set False (default) if you want the same
    content to appear under each relevant section heading."""

    created_at: str = ""
    updated_at: str = ""
    second_passes: List[SecondPassConfig] = field(default_factory=list)
    """Optional serial post-LLM passes that run after branch extraction."""

    def effective_collection_id(self) -> str:
        return self.collection_id or f"extraction_{self.slug}"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProjectConfig":
        raw = dict(d)
        sources = [DocumentSource.from_dict(s) for s in raw.pop("document_sources", [])]
        branches = [BranchConfig.from_dict(b) for b in raw.pop("branches", [])]
        second_passes = [SecondPassConfig.from_dict(p) for p in raw.pop("second_passes", [])]
        raw.pop("guided_reports", None)
        raw.pop("post_branch_passes", None)
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = set(raw) - valid_keys
        if unknown:
            logger.warning("ProjectConfig: unknown keys ignored: %s", sorted(unknown))
        obj = cls(**{k: v for k, v in raw.items() if k in valid_keys})
        obj.document_sources = sources
        obj.branches = branches
        obj.second_passes = second_passes
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

    source_kind: str = "unknown"
    """Broad source context label such as rfq, scada_standard, workshop_note, or evaluation_memo."""

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
    evidence_chunks: List[Dict[str, Any]] = field(default_factory=list)
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
            evidence_chunks=list(d.get("evidence_chunks", []) or []),
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
