"""
Structure enrichment utilities.

Turns parsed markdown sections into enrichment records ready for chunking,
embedding, and retrieval metadata filtering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from pipeline.structure.md_parser import ParsedMarkdownDocument, Section, parse_markdown, parse_markdown_file


# ── Structural role signal sets ────────────────────────────────────────────────
# These are generic structural/navigational signals, NOT corpus-specific terms.

_METADATA_CONTENT_SIGNALS: frozenset = frozenset({
	"isbn", "copyright", "all rights reserved", "first published",
	"library of congress", "cataloging-in-publication",
	"printed in the united states",
})
_METADATA_TITLE_SIGNALS: frozenset = frozenset({"copyright", "credits"})
_FRONTMATTER_TITLE_SIGNALS: frozenset = frozenset({
	"preface", "foreword", "abstract", "acknowledgment", "acknowledgments",
	"acknowledgements", "table of contents", "contents", "introduction",
	"about the author", "about the authors",
})
_INDEX_TITLE_SIGNALS: frozenset = frozenset({
	"index", "bibliography", "references", "further reading",
	"glossary", "appendix",
})


def _assign_structural_role(section: Section) -> str:
	"""Derive a structural role label from section position and content signals.

	Roles:
	  metadata        - copyright / ISBN / publishing boilerplate (early pages)
	  index_noise     - back-matter index, bibliography, glossary
	  heading_overview - table of contents
	  frontmatter     - preface, foreword, abstract, introduction (early pages)
	  table_data      - section whose primary content is a table
	  body            - everything else (default)
	"""
	title_l = (section.title or "").lower().strip()
	path_l = " > ".join(section.path).lower() if section.path else ""
	content_l = (section.content or "").lower()
	page = section.page_start if section.page_start is not None else 9999

	# Metadata: copyright/publishing boilerplate on early pages
	if page <= 8 and any(sig in content_l for sig in _METADATA_CONTENT_SIGNALS):
		return "metadata"
	if any(sig in title_l for sig in _METADATA_TITLE_SIGNALS):
		return "metadata"
	if "copyright" in path_l and page <= 12:
		return "metadata"

	# Index / back-matter noise
	if title_l in _INDEX_TITLE_SIGNALS:
		return "index_noise"
	if page >= 130 and any(sig in title_l for sig in _INDEX_TITLE_SIGNALS):
		return "index_noise"

	# Table of contents
	if "table of contents" in title_l or title_l == "contents":
		return "heading_overview"

	# Front matter: early navigational or descriptive sections
	if page <= 30 and (
		any(sig in title_l for sig in _FRONTMATTER_TITLE_SIGNALS)
		or any(sig in path_l for sig in _FRONTMATTER_TITLE_SIGNALS)
	):
		return "frontmatter"

	# Tabular content
	if section.has_table:
		return "table_data"

	return "body"


@dataclass
class EnrichedSection:
	section_id: str
	title: str
	level: int
	parent_id: Optional[str]
	path: List[str]
	path_text: str
	content: str
	page_start: Optional[int]
	page_end: Optional[int]
	has_table: bool
	char_count: int
	word_count: int
	quality_flags: List[str]
	structural_role: str = "body"

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


def _quality_flags(section: Section) -> List[str]:
	flags: List[str] = []
	content = section.content.strip()

	if not content:
		flags.append("empty_content")
	if len(content) < 80:
		flags.append("short_content")
	if "�" in content:
		flags.append("encoding_artifact")
	if "  " in content:
		flags.append("double_spaces")
	if section.page_start is None:
		flags.append("missing_page")

	return flags


def enrich_document(parsed: ParsedMarkdownDocument) -> Dict[str, Any]:
	enriched_sections: List[EnrichedSection] = []

	for s in parsed.sections:
		content = s.content.strip()
		words = [w for w in content.split() if w]
		enriched_sections.append(
			EnrichedSection(
				section_id=s.section_id,
				title=s.title,
				level=s.level,
				parent_id=s.parent_id,
				path=s.path,
				path_text=" > ".join(s.path),
				content=content,
				page_start=s.page_start,
				page_end=s.page_end,
				has_table=s.has_table,
				char_count=len(content),
				word_count=len(words),
				quality_flags=_quality_flags(s),
				structural_role=_assign_structural_role(s),
			)
		)

	return {
		"source_path": parsed.source_path,
		"metadata": parsed.metadata,
		"preamble": parsed.preamble,
		"sections": [s.to_dict() for s in enriched_sections],
		"stats": {
			"section_count": len(enriched_sections),
			"sections_with_tables": sum(1 for s in enriched_sections if s.has_table),
			"empty_sections": sum(1 for s in enriched_sections if "empty_content" in s.quality_flags),
		},
	}


def enrich_markdown(markdown: str, source_path: str = "") -> Dict[str, Any]:
	parsed = parse_markdown(markdown, source_path=source_path)
	return enrich_document(parsed)


def enrich_markdown_file(markdown_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
	import json

	parsed = parse_markdown_file(markdown_path)
	enriched = enrich_document(parsed)

	if output_path:
		out = Path(output_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		out.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")

	return enriched

