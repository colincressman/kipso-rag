"""Citation selection, normalization, and inline enforcement."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from utils.text_utils import tokenize

CITATION_RE = re.compile(r"\[[^\[\]]+\]")
CHUNK_SUFFIX_RE = re.compile(r"-c(?P<idx>\d{6})$")
SHORT_CIT_RE = re.compile(r"\bc(?P<idx>\d{1,6})\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
NOISY_CITATION_PATTERNS = [
	re.compile(r"\[chunk_id\]", re.IGNORECASE),
	re.compile(r"\[chunk_[^\]]+\]", re.IGNORECASE),
	re.compile(r"\[CHUNK\s+[^\]]+\]", re.IGNORECASE),
	re.compile(r"\[chunk_id\s+[^\]]+\]", re.IGNORECASE),
	re.compile(r"\(chunk[^)]*\)", re.IGNORECASE),
	re.compile(r"chunk_[0-9a-f\-]+", re.IGNORECASE),
	re.compile(r"chunk\s+[0-9a-f]{24,}[-a-z0-9]*", re.IGNORECASE),
	re.compile(r"\b(?:CH|c)[0-9a-f]{12,}(?:-c\d{1,6})?\b", re.IGNORECASE),
	re.compile(r"\bc\d{7,}\b", re.IGNORECASE),
]


def short_citation(chunk_id: str) -> str:
	m = CHUNK_SUFFIX_RE.search(chunk_id or "")
	if m:
		return f"c{m.group('idx')}"
	m2 = SHORT_CIT_RE.search(chunk_id or "")
	if m2:
		return f"c{int(m2.group('idx')):06d}"
	if chunk_id:
		return chunk_id[-12:]
	return "unknown"


def select_citations(hits: List[Dict[str, Any]], prompt_cfg: Dict[str, Any]) -> List[str]:
	if not hits:
		return []
	min_c = int(prompt_cfg.get("min_citations", 2))
	max_c = int(prompt_cfg.get("max_citations", 3))
	window = float(prompt_cfg.get("citation_score_window", 0.06))
	top_score = float(hits[0].get("score") or 0.0)
	selected = [
		h.get("chunk_id")
		for h in hits
		if h.get("chunk_id") and (top_score - float(h.get("score") or 0.0) <= window)
	]

	if len(selected) < min_c:
		for h in hits:
			cid = h.get("chunk_id")
			if cid and cid not in selected:
				selected.append(cid)
			if len(selected) >= min_c:
				break

	return selected[:max_c]


def normalize_answer_citations(answer: str, citation_ids: List[str]) -> str:
	text = answer or ""
	for pattern in NOISY_CITATION_PATTERNS:
		text = pattern.sub("", text)
	text = re.sub(r"\[\s*\]", "", text)
	text = re.sub(r"\[(?!c\d{6}\])[^\[\]]{1,140}\]", "", text, flags=re.IGNORECASE)
	text = re.sub(r"\s+([,.;:!?])", r"\1", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = re.sub(r"\n{3,}", "\n\n", text).strip()

	if not citation_ids:
		return text

	short_tags = [f"[{short_citation(cid)}]" for cid in citation_ids]
	return f"{text}\n\nCitations: {', '.join(short_tags)}"


def strip_trailing_citations_block(answer: str) -> str:
	text = (answer or "").strip()
	return re.split(r"\n\s*Citations\s*:\s*", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def is_factual_sentence(sentence: str) -> bool:
	tokens = tokenize((sentence or "").lower())
	if len(tokens) >= 5:
		return True
	return bool(re.search(r"\d", sentence or ""))


def ensure_inline_sentence_citations(answer: str, citation_ids: List[str]) -> tuple[str, int]:
	"""Attach at least one inline citation to each factual sentence if missing."""
	if not answer or not citation_ids:
		return answer, 0

	body = strip_trailing_citations_block(answer)
	if not body:
		return answer, 0

	sentences = [s.strip() for s in SENTENCE_RE.split(body) if s.strip()]
	if not sentences:
		return answer, 0

	default_tag = f"[{short_citation(citation_ids[0])}]"
	updated: List[str] = []
	added = 0
	for sentence in sentences:
		if is_factual_sentence(sentence) and not CITATION_RE.search(sentence):
			updated.append(f"{sentence} {default_tag}")
			added += 1
		else:
			updated.append(sentence)

	return " ".join(updated), added


def collect_source_text_for_validation(hits: List[Dict[str, Any]], citation_ids: List[str]) -> str:
	if not hits:
		return ""

	hit_map = {str(h.get("chunk_id")): h for h in hits if h.get("chunk_id")}
	ordered = [cid for cid in citation_ids if cid in hit_map] or [str(h.get("chunk_id")) for h in hits[:3] if h.get("chunk_id")]

	parts: List[str] = []
	for cid in ordered:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = str(h.get("text") or "").strip()
		if text:
			parts.append(text)
		for n in ((h.get("metadata") or {}).get("neighbors") or []):
			ntext = str(n.get("text") or "").strip()
			if ntext:
				parts.append(ntext)

	return "\n".join(parts).lower()
