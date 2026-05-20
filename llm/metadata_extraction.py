"""Metadata field extraction — author, title, ISBN, publisher, year."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from llm.citations import SENTENCE_RE, short_citation
from llm.extraction_helpers import query_keywords, query_terms
from utils.text_utils import YEAR_RE


def clean_publisher_name(raw: str) -> str:
	text = " ".join((raw or "").split()).strip(" .,")
	text = re.split(r"\s+\d{1,5}\s+", text, maxsplit=1)[0]
	text = re.split(r"\s+www\.", text, maxsplit=1, flags=re.IGNORECASE)[0]
	text = re.split(r"\s+https?://", text, maxsplit=1, flags=re.IGNORECASE)[0]
	name_match = re.search(
		r"([A-Z][A-Za-z&\.,\- ]{2,80}?(?:LLC|Ltd\.?|Inc\.?|Press|Press, LLC))\b",
		text,
	)
	if name_match:
		return name_match.group(1).strip(" .,")
	return text.strip(" .,")


def extract_title_candidate(blob: str) -> str:
	"""Best-effort book/document title extraction from a metadata text blob."""
	text = " ".join((blob or "").split())
	if not text:
		return ""

	m = re.search(
		r"\b([A-Z][A-Za-z0-9'':,\- ]{5,140}?)\s+by\s+[A-Z][A-Za-z\.\- ]{3,120}\b",
		text,
	)
	if m:
		candidate = " ".join(m.group(1).split()).strip(" .,")
		if len(candidate.split()) >= 2:
			return candidate

	m = re.search(r"\b([A-Z][A-Za-z0-9'':,\- ]{8,140})\s+[\(\|\-]\s*(?:\d{4}|isbn|copyright)", text, flags=re.IGNORECASE)
	if m:
		candidate = " ".join(m.group(1).split()).strip(" .,")
		if len(candidate.split()) >= 2:
			return candidate

	return ""


def is_section_summary_query(query: str) -> bool:
	q = (query or "").lower()
	return bool(re.search(r"\b(summarize|summary|overview|explain|describe|what\s+does)\b", q))


def extract_metadata_field_answer(
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
) -> str:
	if not hits:
		return ""

	q = (query or "").lower()
	hit_map = {str(h.get("chunk_id")): h for h in hits}
	ordered_ids = [cid for cid in citation_ids if cid in hit_map] or [str(h.get("chunk_id")) for h in hits if h.get("chunk_id")]

	search_rows: List[tuple[str, str]] = []
	for cid in ordered_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = " ".join(str(h.get("text") or "").split())
		title = " ".join(str(h.get("title") or "").split())
		path = " ".join(str(h.get("path_text") or "").split())
		blob = " ".join(part for part in [title, path, text] if part)
		text = blob
		if text:
			search_rows.append((str(cid), text))

	is_isbn_query = "isbn" in q
	is_publisher_query = (
		"publisher" in q
		or bool(re.search(r"\bwho\s+published\b", q))
		or ("published" in q and "who" in q)
	)
	is_year_query = (
		"year" in q
		or "publication year" in q
		or bool(re.search(r"\bwhen\s+was\b", q))
	)
	is_author_query = (
		bool(re.search(r"\bauthor\b|\bauthors\b", q))
		or bool(re.search(r"\bwho\s+wrote\b", q))
		or bool(re.search(r"\bwho\s+are\s+the\s+authors\b", q))
	)
	is_title_query = (
		"title" in q
		or bool(re.search(r"\bwhat\s+is\s+the\s+book\s+title\b", q))
	)

	if is_author_query:
		author_pair_re = re.compile(
			r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)\s+and\s+"
			r"([A-Z][a-z]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-z]+)+)\b",
			flags=re.IGNORECASE,
		)
		for cid in ordered_ids:
			h = hit_map.get(str(cid))
			if not h:
				continue
			blobs = [
				str(h.get("title") or ""),
				str(h.get("path_text") or ""),
				str(h.get("text") or ""),
			]
			for blob in blobs:
				m = author_pair_re.search(blob)
				if not m:
					continue
				author_a = " ".join(m.group(1).split())
				author_b = " ".join(m.group(2).split())
				return f"The authors are {author_a} and {author_b} [{short_citation(cid)}]."

		for cid in ordered_ids:
			h = hit_map.get(str(cid))
			if not h:
				continue
			path_text = str(h.get("path_text") or "")
			root = path_text.split(">")[0].strip() if path_text else ""
			m = re.search(
				r"^([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)\s+([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)$",
				root,
			)
			if m:
				author_a = " ".join(m.group(1).split())
				author_b = " ".join(m.group(2).split())
				return f"The authors are {author_a} and {author_b} [{short_citation(cid)}]."
		return "Low confidence: I could not find the author names in the retrieved context."

	if is_title_query:
		for cid, text in search_rows:
			title = extract_title_candidate(text)
			if title:
				return f"The book title is {title} [{short_citation(cid)}]."
		return "Low confidence: I could not find the book title in the retrieved context."

	if is_isbn_query:
		for cid, text in search_rows:
			m = re.search(r"\bISBN(?:-1[03])?\s*[:\-]?\s*([0-9Xx][0-9Xx\-]{9,20})\b", text, flags=re.IGNORECASE)
			if m:
				return f"The ISBN is {m.group(1)} [{short_citation(cid)}]."
		return "Low confidence: I could not find an ISBN in the retrieved context."

	if is_publisher_query:
		for cid, text in search_rows:
			m = re.search(r"First\s+published\s+in\s+\d{4}\s+by\s+([^\.\n]+)", text, flags=re.IGNORECASE)
			if m:
				publisher = clean_publisher_name(m.group(1))
				return f"The book was published by {publisher} [{short_citation(cid)}]."
		for cid, text in search_rows:
			m = re.search(r"Copyright\s*[^\n]{0,40}?\b([A-Z][A-Za-z&\.,\- ]{2,80})\b,\s*\d{4}", text, flags=re.IGNORECASE)
			if m:
				publisher = clean_publisher_name(m.group(1))
				return f"The book was published by {publisher} [{short_citation(cid)}]."
		return "Low confidence: I could not find publisher information in the retrieved context."

	if is_year_query:
		for cid, text in search_rows:
			m = re.search(r"First\s+published\s+in\s+(19\d{2}|20\d{2})", text, flags=re.IGNORECASE)
			if m:
				return f"The book was first published in {m.group(1)} [{short_citation(cid)}]."
		for cid, text in search_rows:
			m = re.search(r"Copyright\s*[^\n]{0,120}?\b(19\d{2}|20\d{2})\b", text, flags=re.IGNORECASE)
			if m:
				return f"The publication year is {m.group(1)} [{short_citation(cid)}]."
		return "Low confidence: I could not find a publication year in the retrieved context."

	return ""


def extractive_factoid_answer(
	query: str,
	hits: List[Dict[str, Any]],
	citation_ids: List[str],
	confidence_band: str,
) -> str:
	if not hits:
		return ""

	kw = query_keywords(query)
	kw = {t for t in kw if t not in {"meaning", "explain", "describe", "overview", "summary", "detail", "details"}}
	if not kw:
		kw = query_keywords(query)
	kw_lower = {t.lower() for t in kw}
	query_has_digits = bool(re.search(r"\d", query or ""))
	is_isbn_query = "isbn" in kw_lower
	is_publisher_query = "publisher" in kw_lower
	is_year_query = bool(kw_lower.intersection({"published", "publication", "year", "when"}))
	hit_map = {str(h.get("chunk_id")): h for h in hits}
	ordered_ids = [cid for cid in citation_ids if cid in hit_map] or [str(h.get("chunk_id")) for h in hits if h.get("chunk_id")]

	scored: List[tuple[float, str, str]] = []
	for cid in ordered_ids:
		h = hit_map.get(str(cid))
		if not h:
			continue
		text = " ".join(str(h.get("text") or "").split())
		if not text:
			continue
		for sent in SENTENCE_RE.split(text):
			s = sent.strip()
			if len(s) < 30 or len(s) > 320:
				continue
			s_lower = s.lower()
			if is_isbn_query and "isbn" not in s_lower:
				continue
			if is_publisher_query and not any(k in s_lower for k in ("published by", "publisher")):
				continue
			if is_year_query:
				has_year = bool(re.search(r"\b(19\d{2}|20\d{2})\b", s_lower))
				has_pub_term = any(k in s_lower for k in ("first published", "published", "copyright", "edition"))
				if not (has_year and has_pub_term):
					continue
			overlap = sum(1 for t in kw if t in s_lower)
			if kw and overlap == 0:
				continue
			score = float(overlap)
			if query_has_digits and re.search(r"\d", s):
				score += 1.0
			if any(k in s_lower for k in ("isbn", "first published", "copyright", "publisher")):
				score += 1.0
			scored.append((score, s, str(cid)))

	if not scored:
		if is_isbn_query:
			return "Low confidence: I could not find an ISBN in the retrieved context."
		if is_publisher_query:
			return "Low confidence: I could not find publisher information in the retrieved context."
		if is_year_query:
			return "Low confidence: I could not find a publication year in the retrieved context."
		return ""

	scored.sort(key=lambda x: x[0], reverse=True)
	max_sents = 1 if confidence_band == "low" else 2
	chosen: List[str] = []
	used = set()
	for _, sent, cid in scored:
		key = sent.lower()
		if key in used:
			continue
		used.add(key)
		chosen.append(f"{sent} [{short_citation(cid)}]")
		if len(chosen) >= max_sents:
			break

	if not chosen:
		return ""

	body = " ".join(chosen)
	if confidence_band == "low":
		return f"Low confidence: {body}"
	return body
