from __future__ import annotations

import hashlib
import html
import json
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from utils.runtime_defaults import DEFAULT_INTERNET_MAX_ARTICLE_CHARS
from utils.vector_ops import cosine as _cosine
from retrieval.web_search import (
	fetch_page,
	parse_bing_rss_results as _parse_bing_rss_results,
	search_web,
)
from retrieval.hyde import _SEARCH_REWRITE_SYSTEM
from utils.text_utils import tokenize
from llm.grounding import ANSWER_STOPWORDS, _COVERAGE_EXTRA_STOPWORDS

_SEARCH_STOPWORDS = ANSWER_STOPWORDS | _COVERAGE_EXTRA_STOPWORDS


TAG_RE = re.compile(r"<[^>]+>")
SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")
from utils.text_utils import YEAR_RE
NAV_ELEMENT_RE = re.compile(
	r'<(?:nav|header|footer|aside)(?:\s[^>]*)?>.*?</(?:nav|header|footer|aside)>',
	re.IGNORECASE | re.DOTALL,
)
MAIN_CONTENT_RE = re.compile(
	r'<(?:main|article)(?:\s[^>]*)?>(.+?)</(?:main|article)>',
	re.IGNORECASE | re.DOTALL,
)

FACT_QUERY_OPENERS = {"who", "when", "where", "which", "whom"}
# Openers that indicate an information request (not ML-corpus questions)
INFORMATIONAL_OPENERS = frozenset({"tell", "explain", "describe", "show", "give", "find", "list"})
# Terms that signal the user wants real-world / current-events information.
# Keep narrow: avoid 'update/recent/current' which appear in ML/math questions.
NEWS_INTENT_TERMS = frozenset({
	"news", "politics", "political", "election", "elections", "happening",
})
# Single tokens that alone make a query time-sensitive regardless of opener.
# Keep this set narrow to avoid false-triggering on ML corpus questions.
RECENCY_KEYWORDS = frozenset({
	"today", "yesterday", "tonight", "currently", "nowadays",
})
# Verbs that signal a product/software release query — combined with a proper
# noun these reliably indicate "what has [Company] shipped" queries that need
# internet regardless of temporal wording.  Keep narrow: only past-form release
# verbs that appear in product catalog questions, not ML methodology texts.
PRODUCT_RELEASE_VERBS = frozenset({
	"released", "release",
	"launched", "launch",
	"unveiled", "unveil",
	"announced", "announce",
})
# Biographical relationship/attribute terms that signal a person-fact query.
# Triggers fact mode for "who is X's [term]" and "what is X's [term]" even
# when no recency/news signal is present. Deliberately narrow to avoid
# false-triggering on ML corpus questions.
BIOGRAPHICAL_TERMS = frozenset({
	"son", "daughter", "wife", "husband", "father", "mother",
	"brother", "sister", "spouse", "partner", "child", "children",
	"youngest", "oldest", "eldest", "parent", "sibling",
	"born", "died", "age", "birthday", "birthplace", "nationality",
	"married", "divorced", "family",
})
QUERY_STOPWORDS = {
	"the", "a", "an", "of", "and", "for", "to", "in", "on", "at", "by", "with",
	"is", "are", "was", "were", "be", "been", "what", "who", "when", "where", "which", "whom",
	# Adjectives that confuse search engines when treated as standalone keywords
	"current",
	"did", "does", "do", "it", "this", "that", "as",
	# Question openers
	"how",
	# Imperative command verbs — these are query openers, not content signals
	"tell", "explain", "describe", "show", "give", "find", "list",
	# Generic prepositions / discourse words that appear on almost every page
	"about", "me", "us",
}
LOW_VALUE_DOMAIN_HINTS = {
	"merriam-webster.com", "dictionary.com", "dictionary.cambridge.org", "vocabulary.com", "thesaurus.com",
	"collinsdictionary.com", "thefreedictionary.com", "macmillandictionary.com", "lexico.com",
}
LOW_VALUE_PATH_HINTS = {"/dictionary", "/define", "/meaning", "/thesaurus"}
FACT_PREFERRED_DOMAIN_HINTS = {
	"openai.com", "wikipedia.org", "britannica.com", "techcrunch.com", "time.com",
	"mashable.com", "arstechnica.com", "ibm.com", "mindstudio.ai", "fifa.com",
	"sportingnews.com", "apnews.com", "reuters.com",
}
FACT_DEMOTED_DOMAIN_HINTS = {"github.com", "chat4us"}

# Keep a large, coherent full-article window for internet hits so answering can
# rely on complete sentences/paragraphs instead of tiny fragments.
INTERNET_FULL_ARTICLE_MAX_CHARS = DEFAULT_INTERNET_MAX_ARTICLE_CHARS
LOW_VALUE_TEXT_HINTS = {
	"table of contents",
	"quick summary",
	"ask anything",
	"top questions",
	"external websites",
	"submit feedback",
	"edit links",
	"related changes",
	"permanent link",
	"page information",
	"printable version",
	"wikidata item",
}


@dataclass
class WebChunk:
	chunk_id: str
	url: str
	title: str
	text: str
	score: float
	vector_score: float
	lexical_jaccard: float
	search_provider: str
	search_url: str

	def to_hit_dict(self) -> Dict[str, Any]:
		section_header = self.title or self.url
		return {
			"chunk_id": self.chunk_id,
			"doc_id": self.url,
			"collection_id": "internet",
			"source_name": self.url,
			"document_title": self.title or self.url,
			"document_path": self.url,
			"section_id": None,
			"title": self.title,
			"path_text": self.url,
			"page_number": None,
			"section_header": section_header,
			"page_start": None,
			"page_end": None,
			"text": self.text,
			"score": float(self.score),
			"source_type": "internet",
			"structural_role": "web_fallback",
			"metadata": {
				"source_type": "internet",
				"structural_role": "web_fallback",
				"collection_id": "internet",
				"source_name": self.url,
				"document_title": self.title or self.url,
				"document_path": self.url,
				"section_header": section_header,
				"token_count_est": len(tokenize(self.text)),
				"vector_score": float(self.vector_score),
				"lexical_jaccard": float(self.lexical_jaccard),
				"search_provider": self.search_provider,
				"search_url": self.search_url,
			},
		}



def _token_set(text: str) -> set[str]:
	return set(tokenize((text or "").lower()))


def _query_terms(text: str) -> List[str]:
	terms: List[str] = []
	for tok in tokenize((text or "").lower()):
		if tok in QUERY_STOPWORDS:
			continue
		if len(tok) <= 2 and not tok.isdigit():
			continue
		terms.append(tok)
	return terms





def _llm_rewrite_search_query(
	query: str,
	hyde_passage: str,
	*,
	model: str,
	base_url: str,
	timeout: float = 6.0,
) -> str:
	"""Ask the LLM (already running for HyDE) to rewrite the query as a web search string.

	VRAM management: the Ollama embedding model (qwen3-embedding, ~4.6 GB) is
	still loaded at this point. We must evict it before loading the LLM, then
	reload it afterwards so internet chunk scoring can proceed.

	Returns the rewritten query on success, or the original query on any failure
	so retrieval is never blocked.
	"""
	if not hyde_passage or not hyde_passage.strip() or not model or not base_url:
		return query

	# ── 1. Unload the embedding model to free VRAM ────────────────────────────
	try:
		_evict_url = f"{base_url.rstrip('/')}/api/embed"
		from utils.runtime_defaults import DEFAULT_EMBED_MODEL_NAME
		_evict_payload = json.dumps({
			"model": DEFAULT_EMBED_MODEL_NAME,
			"input": [],
			"keep_alive": 0,
		}).encode()
		_evict_req = urllib.request.Request(
			_evict_url, data=_evict_payload,
			headers={"Content-Type": "application/json"}, method="POST",
		)
		with urllib.request.urlopen(_evict_req, timeout=10) as r:
			r.read()
	except Exception:
		pass  # if eviction fails, attempt the LLM call anyway

	# ── 2. LLM rewrite call ───────────────────────────────────────────────────
	user_msg = f'Question: {query}\nPassage: {hyde_passage.strip()}'
	payload = json.dumps({
		"model": model,
		"stream": False,
		"keep_alive": 0,
		"options": {"temperature": 0.0, "num_predict": 20},
		"messages": [
			{"role": "system", "content": _SEARCH_REWRITE_SYSTEM},
			{"role": "user", "content": user_msg},
		],
	}).encode()
	rewritten = query
	try:
		req = urllib.request.Request(
			f"{base_url.rstrip('/')}/api/chat",
			data=payload,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			body = json.loads(resp.read())
		result = ((body.get("message") or {}).get("content") or "").strip().strip('"\'')
		if result and len(result) <= 120:
			rewritten = result
	except Exception:
		pass

	# ── 3. Reload the embedding model so chunk scoring can proceed ────────────
	try:
		from utils.runtime_defaults import DEFAULT_EMBED_MODEL_NAME
		_reload_payload = json.dumps({
			"model": DEFAULT_EMBED_MODEL_NAME,
			"input": ["warmup"],
			"keep_alive": 300,
		}).encode()
		_reload_req = urllib.request.Request(
			f"{base_url.rstrip('/')}/api/embed",
			data=_reload_payload,
			headers={"Content-Type": "application/json"},
			method="POST",
		)
		with urllib.request.urlopen(_reload_req, timeout=60) as r:
			r.read()
	except Exception:
		pass  # embedder will cold-load on first chunk scoring call if this fails

	return rewritten


def _is_external_fact_query(query: str) -> bool:
	q = (query or "").strip()
	if not q:
		return False
	tokens = tokenize(q)
	if not tokens:
		return False
	opener = tokens[0].lower()
	has_year = bool(YEAR_RE.search(q))
	has_entity_hint = any(tok.isupper() and len(tok) >= 2 for tok in tokens)
	tokens_lower = {t.lower() for t in tokens}
	has_recency_word = bool(tokens_lower & RECENCY_KEYWORDS)
	has_news_intent = bool(tokens_lower & NEWS_INTENT_TERMS)
	has_biographical_term = bool(tokens_lower & BIOGRAPHICAL_TERMS)
	# Detect proper-noun entities: title-case tokens after the first position
	has_proper_noun = any(
		tok[0].isupper() and not tok.isupper()
		for tok in tokens[1:]
		if len(tok) >= 2
	)
	if opener in FACT_QUERY_OPENERS:
		# who/when/where/which/whom + year, entity, recency word, or biographical term
		return bool(has_year or has_entity_hint or has_proper_noun or has_recency_word or has_biographical_term)
	if opener == "what":
		# "what" is common for ML questions; trigger on recency/news as before,
		# but ALSO on biographical terms ("what is X's youngest son's name")
		# or a year reference ("what happened at X 2026?" = current-events),
		# or a product-release query ("what has [Company] released/launched").
		has_product_release = bool(
			has_proper_noun and (tokens_lower & PRODUCT_RELEASE_VERBS)
		)
		return bool(
			has_recency_word or has_news_intent or has_biographical_term
			or has_year or has_product_release
		)
	if opener in INFORMATIONAL_OPENERS:
		# "tell me about X today" / "explain the news" — require recency or news intent
		# to avoid false-triggering on "explain backpropagation"
		return bool(has_recency_word or has_news_intent)
	# Any query containing explicit news/current-events intent, regardless of opener
	if has_news_intent:
		return True
	return False


def _match_count(query_terms: Sequence[str], text: str) -> int:
	blob = (text or "").lower()
	return sum(1 for t in query_terms if t in blob)


def _is_low_value_result(url: str, title: str) -> bool:
	parsed = urllib.parse.urlparse(url)
	netloc = (parsed.netloc or "").lower()
	path = (parsed.path or "").lower()
	title_l = (title or "").lower()
	if any(hint in netloc for hint in LOW_VALUE_DOMAIN_HINTS):
		return True
	if any(h in path for h in LOW_VALUE_PATH_HINTS):
		return True
	if "definition" in title_l or "meaning" in title_l:
		return True
	return False


def _fact_result_rank(query: str, row: Dict[str, str]) -> tuple[int, int, int]:
	url = str(row.get("url") or "")
	title = str(row.get("title") or "")
	snippet = str(row.get("snippet") or "")
	parsed = urllib.parse.urlparse(url)
	netloc = (parsed.netloc or "").lower()
	match_count = _match_count(_query_terms(query), f"{title} {snippet} {url}")
	preferred = int(any(hint in netloc for hint in FACT_PREFERRED_DOMAIN_HINTS))
	demoted = int(any(hint in netloc for hint in FACT_DEMOTED_DOMAIN_HINTS))
	return (preferred - demoted, match_count, len(snippet))


def _filter_search_results(query: str, rows: Sequence[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
	if not rows:
		return [], []

	query_terms = _query_terms(query)
	is_fact = _is_external_fact_query(query)
	filtered: List[Dict[str, str]] = []
	rejected: List[Dict[str, str]] = []

	for row in rows:
		url = str(row.get("url") or "").strip()
		title = str(row.get("title") or "").strip()
		snippet = str(row.get("snippet") or "").strip()
		if not url:
			continue

		combined = f"{title} {snippet} {url}"
		match_count = _match_count(query_terms, combined)
		# Always reject dictionary/definition sites unless they closely match the query
		if _is_low_value_result(url, title) and match_count < 2:
			rejected.append({**row, "reject_reason": "low_value_definition_like", "query_match_count": match_count})
			continue

		if is_fact and match_count == 0:
			rejected.append({**row, "reject_reason": "no_fact_term_overlap", "query_match_count": match_count})
			continue

		filtered.append(row)

	if not filtered:
		# Don't fall back to dictionary/definition noise — return empty rather than junk
		non_junk = [
			r for r in rows
			if not (
				_is_low_value_result(str(r.get("url") or ""), str(r.get("title") or ""))
				and _match_count(_query_terms(query), f"{r.get('title', '')} {r.get('snippet', '')} {r.get('url', '')}") < 2
			)
		]
		return non_junk, rejected
	if is_fact:
		filtered.sort(key=lambda row: _fact_result_rank(query, row), reverse=True)
	return filtered, rejected


def _extract_visible_text(raw_html: str) -> str:
	clean = SCRIPT_STYLE_RE.sub(" ", raw_html or "")
	# Prefer semantic content containers (<main>, <article>) if present
	main_match = MAIN_CONTENT_RE.search(clean)
	if main_match:
		clean = main_match.group(1)
	else:
		# Strip structural navigation noise when no semantic container found
		clean = NAV_ELEMENT_RE.sub(" ", clean)
	clean = TAG_RE.sub(" ", clean)
	clean = html.unescape(clean)
	clean = WHITESPACE_RE.sub(" ", clean)
	return clean.strip()


def _is_low_quality_chunk(text: str) -> bool:
	"""True if the chunk looks like navigation / ToC / language-list noise rather than prose."""
	if not text or len(text) < 80:
		return False
	text_l = text.lower()
	hint_hits = sum(1 for hint in LOW_VALUE_TEXT_HINTS if hint in text_l)
	if hint_hits >= 2:
		return True
	# High ratio of non-ASCII bytes = Unicode language names, encoded noise, etc.
	non_ascii = sum(1 for c in text if ord(c) > 127)
	if non_ascii / len(text) > 0.12:
		return True
	# Long chunk with almost no sentence boundaries = list / nav structure
	sentences = [s for s in text.split(". ") if len(s.strip()) > 20]
	if len(text) > 400 and len(sentences) < 2:
		return True
	return False


def _chunk_text(text: str, max_chars: int = 900, overlap: int = 160) -> List[str]:
	if not text:
		return []
	clean = WHITESPACE_RE.sub(" ", text).strip()
	if len(clean) <= max_chars:
		return [clean]

	chunks: List[str] = []
	start = 0
	n = len(clean)
	while start < n:
		# Avoid chunks starting mid-word when overlap shifts the window.
		if start > 0 and start < n and clean[start - 1].isalnum() and clean[start].isalnum():
			while start < n and clean[start].isalnum():
				start += 1
			while start < n and clean[start].isspace():
				start += 1
			if start >= n:
				break
		end = min(n, start + max_chars)
		window = clean[start:end]
		if end < n:
			cut = window.rfind(". ")
			if cut > 320:
				end = start + cut + 1
				window = clean[start:end]
		chunk = window.strip()
		if chunk:
			chunks.append(chunk)
		if end >= n:
			break
		start = max(end - overlap, start + 1)
	return chunks


# ---------------------------------------------------------------------------
# Web search result cache — 24-hour SQLite TTL cache.
# Caches raw chunk_rows (fetched text before embedding/scoring) so repeat
# queries skip all network I/O.  Scoring always runs fresh on the caller's
# query vector.
# ---------------------------------------------------------------------------

_WEB_CACHE_DB = Path("data/cache/web_search.db")
_WEB_CACHE_TTL = 24 * 3600  # seconds


def _web_cache_get(search_query: str) -> Optional[List[Dict[str, str]]]:
	try:
		_WEB_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
		conn = sqlite3.connect(str(_WEB_CACHE_DB))
		conn.execute(
			"CREATE TABLE IF NOT EXISTS web_cache "
			"(key TEXT PRIMARY KEY, rows TEXT NOT NULL, ts REAL NOT NULL)"
		)
		key = hashlib.sha256(search_query.lower().strip().encode()).hexdigest()
		row = conn.execute(
			"SELECT rows, ts FROM web_cache WHERE key = ?", (key,)
		).fetchone()
		conn.close()
		if row and (time.time() - row[1]) < _WEB_CACHE_TTL:
			return json.loads(row[0])
	except Exception:
		pass
	return None


def _web_cache_set(search_query: str, chunk_rows: List[Dict[str, str]]) -> None:
	try:
		_WEB_CACHE_DB.parent.mkdir(parents=True, exist_ok=True)
		conn = sqlite3.connect(str(_WEB_CACHE_DB))
		conn.execute(
			"CREATE TABLE IF NOT EXISTS web_cache "
			"(key TEXT PRIMARY KEY, rows TEXT NOT NULL, ts REAL NOT NULL)"
		)
		key = hashlib.sha256(search_query.lower().strip().encode()).hexdigest()
		conn.execute(
			"INSERT OR REPLACE INTO web_cache (key, rows, ts) VALUES (?, ?, ?)",
			(key, json.dumps(chunk_rows), time.time()),
		)
		conn.commit()
		conn.close()
	except Exception:
		pass


def retrieve_internet_chunks(
	*,
	query: str,
	query_vector: Sequence[float],
	embedder: Any,
	max_results: int,
	max_chunks: int,
	timeout_seconds: float,
	score_weight: float,
	hyde_passage: str = "",
	hyde_search_query: str = "",
	llm_model: str = "",
	llm_base_url: str = "",
	) -> Dict[str, Any]:
	"""Search the web, fetch pages, chunk content, embed and score candidate chunks."""
	# Use the pre-baked search query from HyDE (written while the model was
	# still warm) if available.  Fall back to an on-demand LLM rewrite only if
	# that wasn't provided and we have a passage + model config.  If neither,
	# use the original query unchanged.
	#
	# Always apply stopword stripping as the final step — even if HyDE or an
	# LLM rewrite provided a query, the rewrite may have failed and returned
	# the original question verbatim (e.g. "Who is the current CEO of OpenAI…").
	# Without stripping, generic words like "current" cause providers to return
	# results for the "Current" banking brand rather than OpenAI content.
	_raw_candidate: str
	if hyde_search_query and hyde_search_query.strip().lower() != query.strip().lower():
		_raw_candidate = hyde_search_query
	elif hyde_passage and llm_model and llm_base_url:
		_raw_candidate = _llm_rewrite_search_query(
			query, hyde_passage, model=llm_model, base_url=llm_base_url,
		)
	else:
		_raw_candidate = query

	# Strip conversational scaffolding so filler words don't pollute the
	# provider query (e.g. "current" → Bing returns Current banking app).
	stripped_terms = [
		tok for tok in tokenize(_raw_candidate.lower())
		if tok not in _SEARCH_STOPWORDS and tok not in QUERY_STOPWORDS and len(tok) >= 2
	]
	search_query = " ".join(stripped_terms) if stripped_terms else _raw_candidate

	# --- Cache check: skip all network I/O if we have a fresh hit ---------------
	cached_rows = _web_cache_get(search_query)
	if cached_rows is not None:
		chunk_rows: List[Dict[str, str]] = cached_rows
		fetched_urls: List[str] = list(dict.fromkeys(str(r.get("url", "")) for r in chunk_rows))
		search_trace: Dict[str, Any] = {
			"original_query": query,
			"search_query_used": search_query,
			"hyde_distilled": search_query != query,
			"cache_hit": True,
			"search_results_raw": [],
			"search_results_rejected": [],
			"search_results": [],
		}
		raw_search_results: List[Dict[str, str]] = []
		search_results: List[Dict[str, str]] = []
		rejected_results: List[Dict[str, str]] = []
	else:
		# --- Live network path -------------------------------------------------
		search_trace = search_web(search_query, max_results=max_results, timeout=timeout_seconds)
		search_trace["original_query"] = query
		search_trace["search_query_used"] = search_query
		search_trace["hyde_distilled"] = search_query != query
		search_trace["cache_hit"] = False
		raw_search_results = list(search_trace.get("search_results") or [])
		search_results, rejected_results = _filter_search_results(query, raw_search_results)
		if not search_results:
			return {
				"hits": [],
				"trace": {
					**search_trace,
					"search_results_raw": raw_search_results,
					"search_results_rejected": rejected_results,
					"search_results": search_results,
					"fetched_urls": [],
					"selected_urls": [],
				},
			}

		chunk_rows = []
		fetched_urls = []
		# Cap per URL so every search result gets a chance; final top-k is cut after scoring
		per_url_cap = max(3, int(max_chunks))
		# Cap page-fetching to max_results URLs — search may now return more candidates
		# from multiple providers; we only fetch the top max_results to stay within budget.
		for r in search_results[:max_results]:
			url = str(r.get("url") or "")
			title = str(r.get("title") or "")
			snippet = str(r.get("snippet") or "")
			provider = str(r.get("provider") or "")
			search_url = str(r.get("search_url") or "")
			if not url:
				continue

			url_chunk_count = 0
			if len(snippet) >= 80 and not _is_low_quality_chunk(snippet):
				chunk_rows.append(
					{
						"chunk_id": f"internet-{abs(hash(url + '-snippet')) % 1_000_000_000}-c000001",
						"url": url,
						"title": title,
						"text": snippet,
						"search_provider": provider,
						"search_url": search_url,
					}
				)
				url_chunk_count += 1

			fetched_urls.append(url)
			text = fetch_page(url, timeout=timeout_seconds, max_chars=INTERNET_FULL_ARTICLE_MAX_CHARS)
			if len(text) < 240:
				continue

			for idx, chunk in enumerate(_chunk_text(text), start=1):
				if _is_low_quality_chunk(chunk):
					continue
				chunk_rows.append(
					{
						"chunk_id": f"internet-{abs(hash(url)) % 1_000_000_000}-c{idx:06d}",
						"url": url,
						"title": title,
						"text": chunk,
						"search_provider": provider,
						"search_url": search_url,
					}
				)
				url_chunk_count += 1
				if url_chunk_count >= per_url_cap:
					break

		if chunk_rows:  # don't cache empty results — a failed fetch should not poison subsequent calls
			_web_cache_set(search_query, chunk_rows)

	if not chunk_rows:
		return {
			"hits": [],
			"trace": {
				**search_trace,
				"search_results_raw": raw_search_results,
				"search_results_rejected": rejected_results,
				"search_results": search_results,
				"fetched_urls": fetched_urls,
				"selected_urls": [],
			},
		}

	texts = [r["text"] for r in chunk_rows]
	vectors = embedder.embed_texts(texts)
	qset = _token_set(query)

	scored: List[WebChunk] = []
	for row, vec in zip(chunk_rows, vectors):
		vec_score = _cosine(query_vector, vec)
		if vec_score <= -1:
			continue
		cset = _token_set(row["text"])
		union = len(qset.union(cset)) or 1
		lex = len(qset.intersection(cset)) / union
		combined = score_weight * float(vec_score) + (1.0 - score_weight) * float(lex)
		scored.append(
			WebChunk(
				chunk_id=row["chunk_id"],
				url=row["url"],
				title=row["title"],
				text=row["text"],
				score=float(combined),
				vector_score=float(vec_score),
				lexical_jaccard=float(lex),
				search_provider=str(row.get("search_provider") or ""),
				search_url=str(row.get("search_url") or ""),
			)
		)

	scored.sort(key=lambda x: x.score, reverse=True)
	selected_hits = [w.to_hit_dict() for w in scored[: max(1, int(max_chunks))]]
	selected_urls = list(dict.fromkeys(str(hit.get("document_path") or "") for hit in selected_hits if hit.get("document_path")))
	return {
		"hits": selected_hits,
		"trace": {
			**search_trace,
			"search_results_raw": raw_search_results,
			"search_results_rejected": rejected_results,
			"search_results": search_results,
			"fetched_urls": list(dict.fromkeys(fetched_urls)),
			"selected_urls": selected_urls,
		},
	}
