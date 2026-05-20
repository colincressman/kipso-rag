"""
Markdown cleaning utilities.

Designed for markdown emitted by pipeline.normalize.to_markdown.
"""

from __future__ import annotations

import re
from pathlib import Path


_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_SOFT_HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z])[-\u00AD]\s*\n\s*([a-z])")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")
_REPLACEMENT_RUN_RE = re.compile(r"\uFFFD{2,}")
_NOISY_LEADER_RE = re.compile(r"([\w\)\]])[\uFFFD\.·•]{6,}\s*(\d{1,4})$")


def _normalise_bullets(text: str) -> str:
	bullet_chars = ["•", "◦", "▪", "►", "▶", "‣", "●", "○", "�"]
	for ch in bullet_chars:
		text = re.sub(rf"^\s*{re.escape(ch)}\s+", "- ", text, flags=re.MULTILINE)
	return text


def _collapse_inline_spacing(text: str) -> str:
	lines = []
	for line in text.splitlines():
		if line.startswith("```"):
			lines.append(line)
			continue
		lines.append(_MULTI_SPACE_RE.sub(" ", line).rstrip())
	return "\n".join(lines)


def _remove_empty_comments(text: str) -> str:
	# Keep page markers, drop accidental empty comments.
	return re.sub(r"^<!--\s*-->\s*$\n?", "", text, flags=re.MULTILINE)


def _clean_common_artifacts(text: str) -> str:
	"""
	Clean common extraction artifacts conservatively.

	- Removes non-printing control chars (including BS / \x08).
	- Collapses long runs of replacement characters (�).
	- Normalises table-of-contents leader noise at end of lines.
	"""
	text = _CONTROL_CHAR_RE.sub("", text)
	text = _NOISY_LEADER_RE.sub(r"\1 ... \2", text)
	text = _REPLACEMENT_RUN_RE.sub(" ", text)
	return text


def clean_markdown(markdown: str) -> str:
	"""
	Clean markdown for chunking / embedding.

	Steps:
	  1) normalise newline style
	  2) join hyphenated word breaks across lines
	  3) normalise bullet symbols
	  4) collapse excessive inline spaces
	  5) collapse multiple blank lines
	"""
	text = markdown.replace("\r\n", "\n").replace("\r", "\n")
	text = _clean_common_artifacts(text)
	text = _SOFT_HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)
	text = _normalise_bullets(text)
	text = _collapse_inline_spacing(text)
	text = _remove_empty_comments(text)
	text = _MULTI_BLANK_RE.sub("\n\n", text)
	return text.strip() + "\n"


def clean_markdown_file(input_path: str, output_path: str | None = None) -> Path:
	"""Clean markdown file and write result (in place if output_path omitted)."""
	src = Path(input_path)
	out = Path(output_path) if output_path else src
	cleaned = clean_markdown(src.read_text(encoding="utf-8"))
	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(cleaned, encoding="utf-8")
	return out
