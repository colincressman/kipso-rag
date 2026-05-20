"""Evidence, section-summary and section-locator extractors."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from llm.citations import SENTENCE_RE, short_citation
from llm.extraction_helpers import query_keywords, query_terms
from llm.grounding import ANSWER_STOPWORDS, _COVERAGE_EXTRA_STOPWORDS, term_present_in_text
from utils.text_utils import tokenize


def extract_section_summary_answer(
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
	confidence_band: str,
) -> str:
	"""Return a concise extractive summary for section-focused summary requests."""
	if not hits:
		return ""

	kw = query_keywords(query)
	kw = {
		t for t in kw
		if t not in {"chapter", "section", "part", "introduction", "summarize", "summary", "overview", "explain", "describe"}
	}

	hit_map = {str(h.get("chunk_id")): h for h in hits if h.get("chunk_id")}
	ordered_ids = [cid for cid in citation_ids if cid in hit_map] or [str(h.get("chunk_id")) for h in hits if h.get("chunk_id")]

	candidates: List[tuple[float, str, str]] = []
	for cid in ordered_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = " ".join(str(h.get("text") or "").split())
		if not text:
			continue
		for sent in SENTENCE_RE.split(text):
			s = sent.strip()
			if len(s) < 45 or len(s) > 320:
				continue
			s_l = s.lower()
			score = 0.0
			if kw:
				score += float(sum(1 for t in kw if term_present_in_text(t, s_l)))
			if any(term in s_l for term in ("introduction", "chapter", "section", "book", "topics", "covers")):
				score += 0.5
			if len(s.split()) >= 12:
				score += 0.2
			if score > 0.0:
				candidates.append((score, s, str(cid)))

	if not candidates:
		return ""

	candidates.sort(key=lambda x: x[0], reverse=True)
	max_sents = 2 if confidence_band in {"medium", "high"} else 1
	selected: List[str] = []
	seen = set()
	for _, sent, cid in candidates:
		k = sent.lower()
		if k in seen:
			continue
		seen.add(k)
		selected.append(f"{sent} [{short_citation(cid)}]")
		if len(selected) >= max_sents:
			break

	if not selected:
		return ""
	if confidence_band == "low":
		return "Low confidence: " + " ".join(selected)
	return " ".join(selected)


def extract_section_locator_answer(
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
	confidence_band: str,
) -> str:
	if not hits:
		return ""

	kw = query_keywords(query)
	if not kw:
		kw = query_terms(query)
	kw = {t for t in kw if t not in {"chapter", "section", "part", "where", "what", "does", "say", "covers", "covered", "discusses", "discussed"}}
	q_lower = (query or "").lower()

	hit_map = {str(h.get("chunk_id")): h for h in hits}
	ordered_ids = [cid for cid in citation_ids if cid in hit_map] or [str(h.get("chunk_id")) for h in hits if h.get("chunk_id")]

	candidates: List[tuple[float, str, str, Optional[int], str]] = []
	for cid in ordered_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		title = str(h.get("title") or "").strip()
		path = str(h.get("path_text") or "").strip()
		page_start = h.get("page_start")
		path_segments = [seg.strip() for seg in path.split(">") if seg.strip()]
		header_blob = f"{title} {path}".lower()
		if "introduction" in q_lower and "introduction" not in header_blob:
			continue
		overlap = sum(1 for t in kw if t in header_blob)
		if overlap == 0 and kw:
			continue
		score = float(overlap)
		if "chapter" in path.lower():
			score += 0.4
		if "section" in path.lower() or "section" in q_lower:
			score += 0.2
		if "introduction" in q_lower and "introduction" in header_blob:
			score += 0.8
			if isinstance(page_start, int):
				if page_start <= 30:
					score += 0.6
				elif page_start >= 80:
					score -= 0.4
		if isinstance(page_start, int):
			if page_start <= 40:
				score += 0.2
			elif page_start >= 120:
				score -= 0.35
		path_l = path.lower()
		if any(noisy in path_l for noisy in ("keywords", "index", "references", "bibliography")):
			score -= 0.8
		if title:
			score += 0.2
		if re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+", title) and "chapter" not in path.lower():
			score -= 0.5
		label = title or (path_segments[-1] if path_segments else "relevant section")
		if "introduction" in q_lower:
			for seg in path_segments:
				if "introduction" in seg.lower():
					label = seg
					break
		if "chapter" in q_lower:
			chapter_match = re.search(r"\bchapter\s+\d+[^>\n]*", path, flags=re.IGNORECASE)
			if chapter_match:
				label = chapter_match.group(0).strip()
		candidates.append((score, label, str(cid), page_start, path))

	if not candidates:
		return ""

	candidates.sort(key=lambda x: x[0], reverse=True)
	max_items = 1 if confidence_band == "low" else 2
	lines: List[str] = []
	seen = set()
	for _, label, cid, page_start, path in candidates:
		norm = label.lower()
		if norm in seen:
			continue
		seen.add(norm)
		where = f" (page {page_start})" if isinstance(page_start, int) and page_start > 0 else ""
		lines.append(f"{label}{where} [{short_citation(cid)}]")
		if len(lines) >= max_items:
			break

	if not lines:
		return ""

	if len(lines) == 1:
		return f"The most relevant section is {lines[0]}."
	return "Relevant sections include: " + "; ".join(lines) + "."


def extractive_evidence_facts(
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
	max_facts: int = 6,
	confidence_band: str = "medium",
) -> str:
	if not hits or not citation_ids:
		return ""

	if confidence_band == "high":
		all_stops = ANSWER_STOPWORDS | _COVERAGE_EXTRA_STOPWORDS
		content_terms = sorted(
			{t for t in tokenize(query.lower()) if t not in all_stops and len(t) > 2},
			key=len,
			reverse=True,
		)[:6]

		def _extract_passage(hit_text: str, hit_cid: str) -> str:
			t_lower = hit_text.lower()
			found: List[tuple] = []
			for term in content_terms:
				idx = t_lower.find(term)
				if idx >= 0:
					found.append((len(term), idx, term))
			if not found:
				return ""
			found.sort(key=lambda x: (-x[0], -x[1]))
			_, best_idx, _ = found[0]
			start = max(0, best_idx - 80)
			while start > 0 and hit_text[start] not in " .!?":
				start -= 1
			start = start + 1 if start > 0 else 0
			end = min(len(hit_text), best_idx + 300)
			while end < len(hit_text) and hit_text[end] not in ".!?\n":
				end += 1
			end = min(end + 1, len(hit_text))
			passage = hit_text[start:end].strip()
			if len(passage) > 480:
				passage = passage[:480].rsplit(" ", 1)[0] + "..."
			sc = short_citation(hit_cid) if hit_cid else "unknown"
			return f"- {passage} [{sc}]" if len(passage) >= 30 else ""

		for term in content_terms:
			for hit in hits[:3]:
				raw = " ".join(str(hit.get("text") or "").split())
				if term not in raw.lower():
					continue
				passage_str = _extract_passage(raw, str(hit.get("chunk_id") or ""))
				if passage_str:
					return passage_str

		raw = " ".join(str(hits[0].get("text") or "").split())
		fb_cid = str(hits[0].get("chunk_id") or "")
		fb_sc = short_citation(fb_cid) if fb_cid else "unknown"
		for sent in SENTENCE_RE.split(raw) or [raw]:
			s = sent.strip()
			if len(s) >= 40:
				return f"- {s} [{fb_sc}]"
		if raw.strip():
			return f"- {raw.strip()[:400]} [{fb_sc}]"
		return ""

	keywords = query_keywords(query)
	hit_map = {str(h.get("chunk_id")): h for h in hits}
	facts: List[str] = []

	for cid in citation_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = str(h.get("text") or "").strip()
		if not text:
			continue
		short_c = short_citation(cid)
		sentences = SENTENCE_RE.split(" ".join(text.split()))
		for sentence in sentences:
			s = sentence.strip()
			if len(s) < 45 or len(s) > 320:
				continue
			if keywords and not any(k in s.lower() for k in keywords):
				continue
			facts.append(f"- {s} [{short_c}]")
			if len(facts) >= max_facts:
				return "\n".join(facts)

	if facts:
		return "\n".join(facts)

	for cid in citation_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = str(h.get("text") or "").strip()
		if not text:
			continue
		fallback_sentence = SENTENCE_RE.split(" ".join(text.split()))[0].strip()
		if fallback_sentence:
			facts.append(f"- {fallback_sentence} [{short_citation(cid)}]")
		if len(facts) >= max_facts:
			break

	return "\n".join(facts)


def path_title_match_override(query: str, top_hit: Dict[str, Any], min_matches: int) -> bool:
	terms = query_terms(query)
	if not terms:
		return False
	title = str(top_hit.get("title") or "")
	path = str(top_hit.get("path_text") or "")
	header_terms = set(tokenize(f"{title} {path}".lower()))
	return len(terms.intersection(header_terms)) >= max(1, int(min_matches))
