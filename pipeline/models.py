"""
Shared data-contract dataclasses for the ingest pipeline.

These were previously defined in pipeline/ingest.py.  They are kept here
so that db/client.py (persist_pipeline_outputs/upsert_document) and any
tests that construct fake IngestedDocument objects continue to work after
pipeline/ingest.py is removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class IngestedBlock:
    """Single semantic block on a page (heading, body, list_item, table, …)."""
    block_type:    str
    text:          str
    page_num:      int
    bbox:          Dict[str, float]
    font_size:     float        = 0.0
    is_bold:       bool         = False
    confidence:    float        = 1.0
    heading_level: Optional[int] = None
    extra:         Dict[str, Any] = field(default_factory=dict)


@dataclass
class IngestedPage:
    """All extracted content for a single PDF page."""
    page_num:    int
    width:       float
    height:      float
    raw_text:    str
    blocks:      List[IngestedBlock]
    tables:      List[List[List[str]]]
    image_count: int = 0

    def blocks_of_type(self, *types: str) -> List[IngestedBlock]:
        return [b for b in self.blocks if b.block_type in types]

    @property
    def headings(self) -> List[IngestedBlock]:
        return self.blocks_of_type("heading", "title")

    @property
    def body_text(self) -> str:
        return "\n".join(b.text for b in self.blocks_of_type("body", "list_item"))


@dataclass
class IngestedDocument:
    """
    Fully ingested document — the shared contract used by db/client.py and tests.
    """
    doc_id:      str
    source_path: str
    filename:    str
    num_pages:   int
    metadata:    Dict[str, Any]
    pages:       List[IngestedPage]
    ingested_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def all_blocks(self) -> Iterator[IngestedBlock]:
        for page in self.pages:
            yield from page.blocks

    def all_blocks_of_type(self, *types: str) -> List[IngestedBlock]:
        return [b for b in self.all_blocks() if b.block_type in types]

    @property
    def full_text(self) -> str:
        return "\n---PAGE BREAK---\n".join(p.raw_text for p in self.pages)
