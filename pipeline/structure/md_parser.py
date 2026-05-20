"""
Markdown structure parser.

Parses normalized markdown (from pipeline.normalize) into a hierarchical,
serializable section model for downstream chunking/embedding.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
PAGE_MARKER_RE = re.compile(r"^<!--\s*page:(\d+)\s*-->\s*$")


@dataclass
class Section:
	section_id: str
	title: str
	level: int
	page_start: Optional[int] = None
	page_end: Optional[int] = None
	content: str = ""
	path: List[str] = field(default_factory=list)
	parent_id: Optional[str] = None
	has_table: bool = False

	def to_dict(self) -> Dict[str, Any]:
		return asdict(self)


@dataclass
class ParsedMarkdownDocument:
	source_path: str
	metadata: Dict[str, Any] = field(default_factory=dict)
	preamble: str = ""
	sections: List[Section] = field(default_factory=list)

	def to_dict(self) -> Dict[str, Any]:
		return {
			"source_path": self.source_path,
			"metadata": self.metadata,
			"preamble": self.preamble,
			"sections": [s.to_dict() for s in self.sections],
		}


def _parse_frontmatter(lines: List[str]) -> Tuple[Dict[str, Any], int]:
	"""Return (metadata, start_index_after_frontmatter)."""
	if not lines or lines[0].strip() != "---":
		return {}, 0

	meta: Dict[str, Any] = {}
	i = 1
	while i < len(lines):
		line = lines[i].strip()
		if line == "---":
			return meta, i + 1
		if ":" in line:
			key, value = line.split(":", 1)
			meta[key.strip()] = value.strip()
		i += 1
	return meta, 0


def _next_section_id(index: int, title: str) -> str:
	slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
	return f"s{index:04d}-{slug or 'section'}"


def _finalize_section(section: Section, current_page: Optional[int]) -> None:
	if section.page_start is None:
		section.page_start = current_page
	if section.page_end is None:
		section.page_end = current_page
	section.content = section.content.strip()
	section.has_table = "|" in section.content and "---" in section.content


def parse_markdown(markdown: str, source_path: str = "") -> ParsedMarkdownDocument:
	lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
	metadata, i = _parse_frontmatter(lines)
	doc = ParsedMarkdownDocument(source_path=source_path, metadata=metadata)

	stack: List[Section] = []
	current: Optional[Section] = None
	preamble_lines: List[str] = []
	current_page: Optional[int] = None
	section_idx = 1

	while i < len(lines):
		raw = lines[i]
		line = raw.rstrip()

		page_match = PAGE_MARKER_RE.match(line.strip())
		if page_match:
			current_page = int(page_match.group(1))
			if current is not None:
				current.page_end = current_page
			i += 1
			continue

		heading_match = HEADING_RE.match(line)
		if heading_match:
			level = len(heading_match.group(1))
			title = heading_match.group(2).strip()

			if current is not None:
				_finalize_section(current, current_page)
				doc.sections.append(current)

			while stack and stack[-1].level >= level:
				stack.pop()

			parent = stack[-1] if stack else None
			path = [*parent.path, title] if parent else [title]
			current = Section(
				section_id=_next_section_id(section_idx, title),
				title=title,
				level=level,
				page_start=current_page,
				page_end=current_page,
				path=path,
				parent_id=parent.section_id if parent else None,
			)
			section_idx += 1
			stack.append(current)
			i += 1
			continue

		target = current
		if target is None:
			preamble_lines.append(line)
		else:
			if line.strip():
				if target.content:
					target.content += "\n"
				target.content += line
			else:
				if target.content and not target.content.endswith("\n"):
					target.content += "\n"
		i += 1

	if current is not None:
		_finalize_section(current, current_page)
		doc.sections.append(current)

	doc.preamble = "\n".join(preamble_lines).strip()
	return doc


def parse_markdown_file(markdown_path: str) -> ParsedMarkdownDocument:
	path = Path(markdown_path)
	text = path.read_text(encoding="utf-8")
	return parse_markdown(text, source_path=str(path))

