"""
Text-splitting strategies for chunk generation.

All functions are pure (no I/O, no side effects).
"""

from __future__ import annotations

import re
from typing import List

from utils.runtime_defaults import (
	DEFAULT_CHARS_PER_PAGE,
	DEFAULT_CHUNK_MAX_CHARS,
	DEFAULT_CHUNK_OVERLAP_CHARS,
)

# Sentence boundary: split after . ! ? followed by whitespace.
# Lookbehind avoids splitting on abbreviations like "e.g. " less reliably
# but is good enough for prose and lecture notes.
_SENTENCE_RE = re.compile(r'(?<=[.!?])[ \t]+')


def estimate_tokens(text: str) -> int:
	"""Rough token estimate without external tokenizer dependency."""
	if not text.strip():
		return 0
	words = len(text.split())
	word_based = max(1, int(words * 1.3))
	# Floor by char count: handles no-space content (math, table pipes, etc.)
	# Average ~4 chars per token for English; use 3 to be conservative.
	char_based = max(1, len(text) // 3)
	return max(word_based, char_based)


def _split_by_chars_with_overlap(
	text: str,
	max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
	overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> List[str]:
	"""Split text by character boundaries with overlap (fallback for very large chunks)."""
	if len(text) <= max_chars:
		return [text]

	segments = []
	start = 0
	step = max_chars - overlap_chars

	while start < len(text):
		end = min(len(text), start + max_chars)
		segments.append(text[start:end])
		if end >= len(text):
			break
		start += step

	return segments


def _split_by_simulated_pages(
	text: str,
	page_start: int,
	page_end: int,
	title: str = "",
) -> List[tuple[str, int, int]]:
	"""
	Split a multi-page section into simulated page chunks.
	Returns list of (text_chunk, sim_page_start, sim_page_end) tuples.

	Estimates ~3500 chars per page and distributes the section accordingly.
	"""
	CHARS_PER_PAGE = DEFAULT_CHARS_PER_PAGE
	total_chars = len(text)

	# If section already fits in ~1 page worth, no need to split further
	if total_chars <= CHARS_PER_PAGE * 1.5:
		return [(text, page_start, page_end)]

	# Calculate how many simulated pages this text should span
	simulated_pages = max(1, int(total_chars / CHARS_PER_PAGE))
	chunk_size = max(2000, int(total_chars / simulated_pages))

	result = []
	start = 0
	sim_page = page_start

	while start < total_chars:
		end = min(total_chars, start + chunk_size)
		chunk_text = text[start:end]
		result.append((chunk_text, sim_page, sim_page))
		sim_page += 1
		start = end

	return result


def _split_by_paragraphs(
	text: str,
	max_tokens: int,
	max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> List[str]:
	"""
	Split section content into chunks that never break a paragraph mid-sentence.

	Tier 1 — pack whole paragraphs (blank-line separated) until token budget.
	Tier 2 — if a single paragraph exceeds the budget, split on sentence boundaries.
	Tier 3 — if a single sentence still exceeds the budget, fall back to word-window split.

	No overlap is added.  Clean boundaries mean no duplicated text in the index
	and no mid-thought cuts seen by the LLM.
	"""
	paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
	if not paragraphs:
		return []

	max_words_fallback = max(20, int(max_tokens / 1.3))
	chunks: List[str] = []
	current_parts: List[str] = []
	current_tokens = 0

	def _flush() -> None:
		if current_parts:
			chunks.append("\n\n".join(current_parts))

	for para in paragraphs:
		para_tokens = estimate_tokens(para)

		if para_tokens > max_tokens:
			# Flush current accumulation before handling the big paragraph
			_flush()
			current_parts.clear()
			current_tokens = 0

			# Tier 2: split on sentence boundaries
			sentences = [s.strip() for s in _SENTENCE_RE.split(para) if s.strip()]
			sent_parts: List[str] = []
			sent_tokens = 0
			for sent in sentences:
				s_toks = estimate_tokens(sent)
				if s_toks > max_tokens:
					# Tier 3: single sentence too long — word-window, no overlap
					if sent_parts:
						chunks.append(" ".join(sent_parts))
						sent_parts = []
						sent_tokens = 0
					chunks.extend(_split_words_with_overlap(sent, max_words=max_words_fallback, overlap_words=0))
				elif sent_tokens + s_toks > max_tokens and sent_parts:
					chunks.append(" ".join(sent_parts))
					sent_parts = [sent]
					sent_tokens = s_toks
				else:
					sent_parts.append(sent)
					sent_tokens += s_toks
			if sent_parts:
				chunks.append(" ".join(sent_parts))

		elif current_tokens + para_tokens > max_tokens and current_parts:
			# This paragraph would overflow the current chunk — flush and start fresh
			_flush()
			current_parts = [para]
			current_tokens = para_tokens

		else:
			current_parts.append(para)
			current_tokens += para_tokens

	_flush()

	# Safety: enforce hard char limit without splitting mid-word
	final: List[str] = []
	for chunk in chunks:
		if len(chunk) > max_chars:
			final.extend(_split_by_chars_with_overlap(chunk, max_chars=max_chars, overlap_chars=0))
		else:
			final.append(chunk)

	return final


def _split_words_with_overlap(text: str, max_words: int, overlap_words: int) -> List[str]:
	words = text.split()
	if not words:
		return []

	if len(words) <= max_words:
		joined = " ".join(words)
		# Even if few words, content may have no-space runs (tables, formulas) making it huge
		if len(joined) > DEFAULT_CHUNK_MAX_CHARS:
			return _split_by_chars_with_overlap(
				joined,
				max_chars=DEFAULT_CHUNK_MAX_CHARS,
				overlap_chars=DEFAULT_CHUNK_OVERLAP_CHARS,
			)
		return [joined]

	segments: List[str] = []
	start = 0
	step = max(1, max_words - overlap_words)

	while start < len(words):
		end = min(len(words), start + max_words)
		segments.append(" ".join(words[start:end]))
		if end >= len(words):
			break
		start += step

	# Safeguard: if any segment exceeds 4k chars, split it further by character boundaries
	final_segments = []
	for segment in segments:
		if len(segment) > DEFAULT_CHUNK_MAX_CHARS:
			final_segments.extend(
				_split_by_chars_with_overlap(
					segment,
					max_chars=DEFAULT_CHUNK_MAX_CHARS,
					overlap_chars=DEFAULT_CHUNK_OVERLAP_CHARS,
				)
			)
		else:
			final_segments.append(segment)

	return final_segments


def _split_table_block(
	text: str,
	max_tokens: int,
	max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
) -> List[str]:
	"""Split a block that contains a table.

	Unlike _split_by_paragraphs, this never splits on sentence boundaries —
	only on blank lines (paragraph breaks) — so table rows are never cut mid-row.
	A hard char cap still applies to avoid gigantic chunks.

	Each returned segment will be prefixed with ``[ TABLE ]`` by the caller.
	"""
	if not text.strip():
		return []

	# If it fits in one chunk, no splitting needed.
	if estimate_tokens(text) <= max_tokens and len(text) <= max_chars:
		return [text]

	# Split at paragraph (blank-line) boundaries only.
	paras = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
	if not paras:
		return [text[:max_chars]]

	chunks: List[str] = []
	current_parts: List[str] = []
	current_tokens = 0

	def _flush() -> None:
		if current_parts:
			chunks.append("\n\n".join(current_parts))

	for para in paras:
		para_tokens = estimate_tokens(para)
		if current_tokens + para_tokens > max_tokens and current_parts:
			_flush()
			current_parts = [para]
			current_tokens = para_tokens
		else:
			current_parts.append(para)
			current_tokens += para_tokens

	_flush()

	# Hard char cap on any individual chunk.
	final: List[str] = []
	for chunk in chunks:
		if len(chunk) > max_chars:
			final.extend(_split_by_chars_with_overlap(chunk, max_chars=max_chars, overlap_chars=0))
		else:
			final.append(chunk)

	return final or [text[:max_chars]]

