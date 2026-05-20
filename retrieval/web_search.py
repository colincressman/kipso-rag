"""
Robust multi-provider web search engine for the RAG internet fallback.

Provider cascade (tried in order):
  1. WikipediaProvider  — official Wikipedia OpenSearch + REST summary API (no scraping)
  2. BingRSSProvider    — Bing RSS feed (soft-block prone)
  3. DuckDuckGoProvider — scrapes duckduckgo.com/html
  4. BingHTMLProvider   — scrapes bing.com/search HTML (currently dead — Bing returns JS shell)

HTTP:             httpx (preferred: redirects, gzip, connection pooling) → urllib fallback
Content extract:  trafilatura (preferred: ignores nav/ads/footers) → regex fallback
Cache:            SQLite TTL (24 h search / 1 h page) at data/cache/web_search.sqlite
"""
from __future__ import annotations

import hashlib
import html
import json
import random
import re
import sqlite3
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── optional fast HTTP client ─────────────────────────────────────────────────
try:
	import httpx as _httpx  # type: ignore[import]
	_HTTPX_AVAILABLE = True
except ImportError:
	_HTTPX_AVAILABLE = False

# ── optional content extractor ───────────────────────────────────────────────
try:
	import trafilatura as _trafilatura  # type: ignore[import]
	_TRAFILATURA_AVAILABLE = True
except ImportError:
	_TRAFILATURA_AVAILABLE = False

# ── regex constants ───────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.DOTALL)
_NAV_RE = re.compile(
	r"<(?:nav|header|footer|aside)(?:\s[^>]*)?>.*?</(?:nav|header|footer|aside)>",
	re.I | re.DOTALL,
)
_MAIN_RE = re.compile(
	r"<(?:main|article)(?:\s[^>]*)?>(.+?)</(?:main|article)>",
	re.I | re.DOTALL,
)
_WS_RE = re.compile(r"\s+")

from utils.text_utils import tokenize

# Bing HTML result patterns
_BING_H2_RE = re.compile(
	r'<h2[^>]*>\s*<a[^>]+href="(?P<href>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>',
	re.I | re.DOTALL,
)
_BING_SNIPPET_RE = re.compile(
	r'<p[^>]*class="[^"]*b_lineclamp[^"]*"[^>]*>(?P<s>.*?)</p>'
	r'|<p>(?P<s2>[^<]{20,})</p>',
	re.I | re.DOTALL,
)

# DuckDuckGo HTML result patterns
_DDG_RESULT_RE = re.compile(
	r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
	re.I | re.DOTALL,
)
_DDG_SNIPPET_RE = re.compile(
	r'<a[^>]+class="[^"]*result__snippet[^"]*"[^>]*>(?P<s>.*?)</a>',
	re.I | re.DOTALL,
)
_DDG_FALLBACK_RE = re.compile(
	r'<a[^>]+href="(?P<href>https?://[^"]+)"[^>]*>(?P<title>.*?)</a>',
	re.I | re.DOTALL,
)

# Bing RSS patterns
_RSS_ITEM_RE = re.compile(r"<item>(?P<body>.*?)</item>", re.I | re.DOTALL)
_RSS_TAG_RE = re.compile(
	r"<(?P<tag>title|link|description)>(?P<val>.*?)</(?P=tag)>",
	re.I | re.DOTALL,
)

# DDG redirect unwrap
_UDDG_RE = re.compile(r"uddg=([^&]+)")

_BLOCKED_NETLOCS = frozenset({
	"www.bing.com", "bing.com", "r.bing.com", "th.bing.com",
	"go.microsoft.com", "microsoft.com",
})
_STOPWORDS = frozenset({
	"the", "a", "an", "of", "and", "for", "to", "in", "on", "at", "by",
	"with", "is", "are", "was", "were", "be", "been", "what", "who", "when",
	"where", "which", "whom", "did", "does", "do", "it", "this", "that",
	"as", "tell", "explain", "describe", "show", "give", "find", "list",
	"about", "me", "us",
})

_DEFAULT_CACHE_TTL = 86_400		# 24 h for search results
_PAGE_CACHE_TTL = 3_600			# 1 h for fetched page text
_PAGE_FETCH_HARD_CAP = 60_000	# max chars stored per page in cache

# Rotate across realistic browser fingerprints to reduce soft-block frequency.
# Each entry is a (User-Agent, Accept, Accept-Language) tuple.
_UA_POOL: List[Dict[str, str]] = [
	{
		# Chrome 124 — Windows
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.9",
	},
	{
		# Firefox 125 — Windows
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) "
			"Gecko/20100101 Firefox/125.0"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.5",
	},
	{
		# Edge 124 — Windows
		"User-Agent": (
			"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.8",
	},
	{
		# Chrome 124 — macOS
		"User-Agent": (
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
			"AppleWebKit/537.36 (KHTML, like Gecko) "
			"Chrome/124.0.0.0 Safari/537.36"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.9",
	},
	{
		# Firefox 125 — macOS
		"User-Agent": (
			"Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) "
			"Gecko/20100101 Firefox/125.0"
		),
		"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
		"Accept-Language": "en-US,en;q=0.5",
	},
]

# Legacy alias — used by urllib fallback path and Wikipedia provider headers.
# Points to the first (Chrome/Windows) entry for stability where rotation
# would cause issues (Wikipedia's documented API UA requirements).
_BROWSER_HEADERS: Dict[str, str] = _UA_POOL[0]


def _random_browser_headers() -> Dict[str, str]:
	"""Return a randomly chosen browser fingerprint from the UA pool."""
	return random.choice(_UA_POOL)


# Jitter delay between page fetches to reduce rate-limit risk.
_FETCH_JITTER_MIN = 0.2   # seconds
_FETCH_JITTER_MAX = 0.9   # seconds


def _fetch_jitter() -> None:
	"""Sleep for a short random interval between outgoing HTTP requests."""
	time.sleep(random.uniform(_FETCH_JITTER_MIN, _FETCH_JITTER_MAX))


# ── data types ────────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
	url: str
	title: str
	snippet: str
	provider: str
	search_url: str


# ── SQLite TTL cache ──────────────────────────────────────────────────────────
class WebSearchCache:
	"""Persistent TTL cache backed by SQLite.

	Caches search results (24 h) and fetched page text (1 h) to avoid
	redundant network calls for the same queries.
	"""

	def __init__(
		self,
		db_path: Path,
		ttl_search: float = _DEFAULT_CACHE_TTL,
		ttl_page: float = _PAGE_CACHE_TTL,
	) -> None:
		db_path.parent.mkdir(parents=True, exist_ok=True)
		self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
		self._ttl_search = ttl_search
		self._ttl_page = ttl_page
		self._conn.execute(
			"CREATE TABLE IF NOT EXISTS cache "
			"(key TEXT PRIMARY KEY, value TEXT, expires_at REAL NOT NULL)"
		)
		self._conn.execute(
			"CREATE INDEX IF NOT EXISTS idx_expires ON cache(expires_at)"
		)
		self._conn.commit()

	def _key(self, *parts: str) -> str:
		return hashlib.sha256("|".join(parts).encode()).hexdigest()

	def get(self, *parts: str) -> Optional[Any]:
		key = self._key(*parts)
		row = self._conn.execute(
			"SELECT value, expires_at FROM cache WHERE key = ?", (key,)
		).fetchone()
		if row is None:
			return None
		value, expires_at = row
		if time.time() > expires_at:
			self._conn.execute("DELETE FROM cache WHERE key = ?", (key,))
			self._conn.commit()
			return None
		return json.loads(value)

	def set(self, *parts: str, value: Any, ttl: Optional[float] = None) -> None:
		key = self._key(*parts)
		effective_ttl = ttl if ttl is not None else self._ttl_search
		self._conn.execute(
			"INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
			(key, json.dumps(value), time.time() + effective_ttl),
		)
		self._conn.commit()

	def purge_expired(self) -> None:
		self._conn.execute("DELETE FROM cache WHERE expires_at < ?", (time.time(),))
		self._conn.commit()


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def _urllib_get(url: str, extra_headers: Optional[Dict[str, str]], timeout: float) -> str:
	headers = {**_random_browser_headers(), "Accept-Encoding": "identity", **(extra_headers or {})}
	req = urllib.request.Request(url, headers=headers)
	with urllib.request.urlopen(req, timeout=timeout) as resp:
		ctype = str(resp.headers.get("Content-Type") or "")
		if not any(t in ctype for t in ("text/html", "application/xhtml+xml", "application/xml", "text/xml")):
			return ""
		charset = resp.headers.get_content_charset() or "utf-8"
		return resp.read(3_000_000).decode(charset, errors="replace")


def _httpx_get(url: str, extra_headers: Optional[Dict[str, str]], timeout: float) -> str:
	headers = {**_random_browser_headers(), **(extra_headers or {})}
	with _httpx.Client(
		follow_redirects=True,
		timeout=_httpx.Timeout(connect=5.0, read=timeout, write=5.0, pool=2.0),
		headers=headers,
		limits=_httpx.Limits(max_connections=4, max_keepalive_connections=2),
	) as client:
		resp = client.get(url)
		resp.raise_for_status()
		return resp.text


def _http_get(url: str, extra_headers: Optional[Dict[str, str]] = None, timeout: float = 10.0) -> str:
	"""Fetch a URL and return raw HTML/text. Prefers httpx when available.

	Falls back to urllib on 4xx responses (e.g. Wikipedia returns 403 to
	httpx's default headers but accepts urllib's User-Agent).
	"""
	if _HTTPX_AVAILABLE:
		try:
			return _httpx_get(url, extra_headers, timeout)
		except Exception as exc:
			# On client errors (4xx) retry with urllib which uses a different
			# TLS fingerprint and Accept-Encoding that some sites prefer.
			status = getattr(getattr(exc, "response", None), "status_code", None)
			if status is not None and 400 <= status < 500:
				return _urllib_get(url, extra_headers, timeout)
			raise
	return _urllib_get(url, extra_headers, timeout)


# ── content extraction ────────────────────────────────────────────────────────
def _regex_extract(raw_html: str) -> str:
	"""Extract visible text via regex. Prefers <main>/<article> over nav/footer."""
	clean = _SCRIPT_STYLE_RE.sub(" ", raw_html or "")
	m = _MAIN_RE.search(clean)
	clean = m.group(1) if m else _NAV_RE.sub(" ", clean)
	clean = _TAG_RE.sub(" ", clean)
	clean = html.unescape(clean)
	return _WS_RE.sub(" ", clean).strip()


def extract_page_text(raw_html: str, url: str = "") -> str:
	"""Extract main article text from raw HTML.

	Uses trafilatura when available (far better at filtering nav/ads/footers).
	Falls back to regex if trafilatura is absent or returns an empty result.
	"""
	if _TRAFILATURA_AVAILABLE and raw_html:
		text = _trafilatura.extract(
			raw_html,
			include_comments=False,
			include_tables=True,
			no_fallback=False,
			url=url or None,
		)
		if text and len(text) >= 150:
			return text
	return _regex_extract(raw_html)


# ── URL normalisation ─────────────────────────────────────────────────────────
def _normalize_url(href: str) -> str:
	"""Normalise a raw href to an absolute HTTP(S) URL. Returns '' if invalid."""
	href = html.unescape(href or "").strip()
	if not href:
		return ""
	if href.startswith("//"):
		href = "https:" + href
	# Unwrap DDG /l/?uddg=... redirect
	if "/l/" in href and "uddg=" in href:
		m = _UDDG_RE.search(href)
		if m:
			href = urllib.parse.unquote(m.group(1))
	parsed = urllib.parse.urlparse(href)
	if parsed.scheme not in {"http", "https"}:
		return ""
	netloc = (parsed.netloc or "").lower()
	if "duckduckgo.com" in netloc or "bing.com" in netloc:
		return ""
	return href


# ── query helpers ─────────────────────────────────────────────────────────────
def _query_terms(query: str) -> List[str]:
	terms = []
	for tok in tokenize((query or "").lower()):
		if tok in _STOPWORDS or (len(tok) <= 2 and not tok.isdigit()):
			continue
		terms.append(tok)
	return terms


def _is_relevant(result: SearchResult, query_terms: List[str]) -> bool:
	if not query_terms:
		return True
	blob = f"{result.title} {result.snippet} {result.url}".lower()
	return any(t in blob for t in query_terms)


def _is_cjk_dominated(results: List[SearchResult]) -> bool:
	"""Return True if the result batch is dominated by CJK text (geo-routing error).

	When Bing ignores locale params it serves results from its regional index.
	CJK (Chinese/Japanese/Korean) characters in the titles/snippets of most
	results is a reliable signal that the response is from the wrong locale.
	"""
	if not results:
		return False
	total_chars = 0
	cjk_chars = 0
	for r in results:
		blob = f"{r.title} {r.snippet}"
		total_chars += len(blob)
		cjk_chars += sum(
			1 for c in blob
			if ("\u4e00" <= c <= "\u9fff")  # CJK Unified Ideographs
			or ("\u3040" <= c <= "\u30ff")  # Hiragana / Katakana
			or ("\uac00" <= c <= "\ud7af")  # Hangul
		)
	if total_chars == 0:
		return False
	return (cjk_chars / total_chars) > 0.15

# ── providers ─────────────────────────────────────────────────────────────────
class BingHTMLProvider:
	"""Scrape Bing HTML results — more reliable than Bing RSS."""

	name = "bing_html"

	def search_url(self, query: str) -> str:
		return (
			f"https://www.bing.com/search"
			f"?q={urllib.parse.quote_plus(query)}&setlang=en&mkt=en-US"
		)

	def search(self, query: str, max_results: int = 5, timeout: float = 10.0) -> List[SearchResult]:
		url = self.search_url(query)
		try:
			html_doc = _http_get(url, timeout=timeout)
		except Exception:
			return []

		snippets = [
			_regex_extract(m.group("s") or m.group("s2") or "")
			for m in _BING_SNIPPET_RE.finditer(html_doc)
		]
		results: List[SearchResult] = []
		snippet_idx = 0
		for m in _BING_H2_RE.finditer(html_doc):
			href = _normalize_url(m.group("href") or "")
			if not href:
				continue
			parsed = urllib.parse.urlparse(href)
			netloc = (parsed.netloc or "").lower()
			if netloc in _BLOCKED_NETLOCS or netloc.endswith(".bing.com") or netloc.endswith(".microsoft.com"):
				continue
			if any(r.url == href for r in results):
				continue
			title = _regex_extract(m.group("title") or "")
			snippet = snippets[snippet_idx] if snippet_idx < len(snippets) else ""
			snippet_idx += 1
			results.append(
				SearchResult(url=href, title=title, snippet=snippet, provider=self.name, search_url=url)
			)
			if len(results) >= max_results:
				break
		return results


class DuckDuckGoProvider:
	"""Scrape DuckDuckGo HTML results."""

	name = "duckduckgo"

	def search_url(self, query: str) -> str:
		return f"https://duckduckgo.com/html/?q={urllib.parse.quote_plus(query)}"

	def search(self, query: str, max_results: int = 5, timeout: float = 10.0) -> List[SearchResult]:
		url = self.search_url(query)
		try:
			html_doc = _http_get(
				url,
				extra_headers={"Referer": "https://duckduckgo.com/"},
				timeout=timeout,
			)
		except Exception:
			return []

		snippets = [
			_regex_extract(m.group("s") or "")
			for m in _DDG_SNIPPET_RE.finditer(html_doc)
		]
		results: List[SearchResult] = []
		snippet_idx = 0
		for m in _DDG_RESULT_RE.finditer(html_doc):
			href = _normalize_url(m.group("href") or "")
			if not href:
				continue
			if any(r.url == href for r in results):
				continue
			title = _regex_extract(m.group("title") or "")
			snippet = snippets[snippet_idx] if snippet_idx < len(snippets) else ""
			snippet_idx += 1
			results.append(
				SearchResult(url=href, title=title, snippet=snippet, provider=self.name, search_url=url)
			)
			if len(results) >= max_results:
				break

		# Fallback: generic link extraction when CSS selectors yield nothing
		if not results:
			for m in _DDG_FALLBACK_RE.finditer(html_doc):
				href = _normalize_url(m.group("href") or "")
				if not href:
					continue
				if any(r.url == href for r in results):
					continue
				title = _regex_extract(m.group("title") or "")
				results.append(
					SearchResult(url=href, title=title, snippet="", provider=self.name, search_url=url)
				)
				if len(results) >= max_results:
					break
		return results


class WikipediaProvider:
	"""Wikipedia full-text search + REST summary API.

	Uses only official, publicly documented Wikipedia APIs — no scraping.
	Reference: https://www.mediawiki.org/wiki/API:Search
	           https://en.wikipedia.org/api/rest_v1/#/Page_content/get_page_summary__title_

	Flow:
	  1. MediaWiki search API (action=query&list=search) → article titles
	  2. REST summary API per title → plain-text extract (lead paragraph)
	  Results are returned as SearchResult objects with the extract as snippet.
	"""

	name = "wikipedia"
	_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
	_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
	_HEADERS = {
		"User-Agent": "RAG-research-assistant/1.0 (personal project; no commercial use)",
		"Accept": "application/json",
	}

	def _json_get(self, url: str, timeout: float) -> Any:
		"""Minimal urllib fetch for Wikipedia JSON APIs — avoids browser-header conflicts."""
		req = urllib.request.Request(url, headers=self._HEADERS)
		with urllib.request.urlopen(req, timeout=timeout) as resp:
			return json.loads(resp.read())

	def search_url(self, query: str) -> str:
		params = urllib.parse.urlencode({
			"action": "query",
			"list": "search",
			"srsearch": query,
			"srlimit": 5,
			"format": "json",
		})
		return f"{self._SEARCH_URL}?{params}"

	def search(self, query: str, max_results: int = 5, timeout: float = 10.0) -> List[SearchResult]:
		s_url = self.search_url(query)
		try:
			data = self._json_get(s_url, timeout)
		except Exception:
			return []

		search_hits = (data.get("query") or {}).get("search") or []
		results: List[SearchResult] = []
		for hit in search_hits:
			if len(results) >= max_results:
				break
			title = hit.get("title") or ""
			if not title:
				continue
			wiki_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'), safe='')}"
			# Fetch plain-text extract from REST summary API
			snippet = _TAG_RE.sub("", html.unescape(hit.get("snippet") or ""))
			try:
				encoded_title = urllib.parse.quote(title.replace(" ", "_"), safe="")
				summary_url = self._SUMMARY_URL.format(title=encoded_title)
				summary_data = self._json_get(summary_url, timeout)
				extract = summary_data.get("extract") or ""
				if extract and len(extract) > len(snippet):
					snippet = extract
			except Exception:
				pass  # fall back to search snippet
			results.append(SearchResult(
				url=wiki_url,
				title=title,
				snippet=snippet,
				provider=self.name,
				search_url=s_url,
			))
		return results


class BingRSSProvider:
	"""Bing RSS feed — fast but prone to soft-blocking. Used as last resort."""

	name = "bing_rss"

	def search_url(self, query: str) -> str:
		return (
			f"https://www.bing.com/search"
			f"?format=rss&q={urllib.parse.quote_plus(query)}&mkt=en-US&setlang=en&cc=US"
		)

	def search(self, query: str, max_results: int = 5, timeout: float = 10.0) -> List[SearchResult]:
		url = self.search_url(query)
		try:
			xml_doc = _http_get(url, timeout=timeout)
		except Exception:
			return []
		return [
			SearchResult(
				url=r["url"], title=r["title"], snippet=r["snippet"],
				provider=self.name, search_url=url,
			)
			for r in parse_bing_rss_results(xml_doc, max_results=max_results)
		]


# ── search engine ─────────────────────────────────────────────────────────────
class WebSearchEngine:
	"""Multi-provider search with caching and per-provider soft-block detection.

	Providers are tried in cascade order. If a provider returns results with
	zero query-term overlap (soft-block / rate-limit) it is skipped and the
	next provider tried. Results are merged and deduplicated by URL.
	"""

	def __init__(
		self,
		providers: Optional[List[Any]] = None,
		cache: Optional[WebSearchCache] = None,
	) -> None:
		self._providers: List[Any] = providers if providers is not None else [
			WikipediaProvider(),
			BingRSSProvider(),
			DuckDuckGoProvider(),
			BingHTMLProvider(),
		]
		self._cache = cache

	def search(
		self,
		query: str,
		max_results: int = 5,
		timeout: float = 10.0,
	) -> Dict[str, Any]:
		"""Search using the provider cascade.

		Returns a dict compatible with the shape expected by internet_fallback.py:
		  {"search_query": str, "search_attempts": list, "search_results": list}
		where each search_results entry has: url, title, snippet, provider, search_url.
		"""
		query_terms = _query_terms(query)
		seen_urls: set = set()
		all_results: List[Dict[str, Any]] = []
		attempts: List[Dict[str, Any]] = []

		for provider in self._providers:
			s_url = provider.search_url(query)
			attempt: Dict[str, Any] = {
				"provider": provider.name,
				"search_url": s_url,
				"result_count": 0,
			}

			# Cache lookup
			rows: Optional[List[SearchResult]] = None
			if self._cache is not None:
				cached = self._cache.get("search", provider.name, query)
				if cached is not None:
					rows = [SearchResult(**r) for r in cached]
					attempt["cached"] = True

			if rows is None:
				_fetch_jitter()  # polite delay before each provider request
				try:
					rows = provider.search(query, max_results=max_results, timeout=timeout)
				except Exception as exc:
					rows = []
					attempt["error"] = type(exc).__name__
				# Cache successful non-empty results
				if self._cache is not None and rows:
					self._cache.set(
						"search", provider.name, query,
						value=[vars(r) for r in rows],
						ttl=_DEFAULT_CACHE_TTL,
					)

			# Soft-block detection 1: CJK geo-routing (wrong regional index)
			if rows and _is_cjk_dominated(rows):
				attempt["error"] = attempt.get("error") or "soft_blocked_cjk_geo_routing"
				attempts.append(attempt)
				continue

			# Soft-block detection 2: skip provider if no query-term overlap in results
			if rows and query_terms:
				relevant = [r for r in rows if _is_relevant(r, query_terms)]
				if not relevant:
					attempt["error"] = attempt.get("error") or "soft_blocked_irrelevant_results"
					attempts.append(attempt)
					continue
				rows = relevant

			for r in rows:
				if not r.url or r.url in seen_urls:
					continue
				seen_urls.add(r.url)
				all_results.append({
					"provider": r.provider,
					"search_url": r.search_url,
					"url": r.url,
					"title": r.title,
					"snippet": r.snippet,
				})
				attempt["result_count"] = attempt.get("result_count", 0) + 1
			attempts.append(attempt)

		# Return all unique results across every provider (hard cap to avoid runaway
		# growth; scoring + filtering in retrieve_internet_chunks picks the best).
		all_results = all_results[:max_results * 3]

		return {
			"search_query": query,
			"search_attempts": attempts,
			"search_results": all_results,
		}


# ── page fetcher ──────────────────────────────────────────────────────────────
class PageFetcher:
	"""Fetch a URL and return clean extracted text, with optional caching."""

	def __init__(self, cache: Optional[WebSearchCache] = None) -> None:
		self._cache = cache

	def fetch(self, url: str, timeout: float = 10.0, max_chars: int = 12_000) -> str:
		# Cache stores up to _PAGE_FETCH_HARD_CAP chars; caller chooses own cap
		if self._cache is not None:
			cached = self._cache.get("page", url)
			if cached is not None:
				return str(cached)[:max_chars]

		_fetch_jitter()  # polite delay before outbound request
		try:
			raw_html = _http_get(url, timeout=timeout)
		except Exception:
			return ""

		if not raw_html:
			return ""

		text = extract_page_text(raw_html, url=url)
		if not text:
			return ""

		to_cache = text[:_PAGE_FETCH_HARD_CAP]
		if self._cache is not None:
			self._cache.set("page", url, value=to_cache, ttl=_PAGE_CACHE_TTL)

		return to_cache[:max_chars]


# ── standalone helper (also used by BingRSSProvider) ─────────────────────────
def parse_bing_rss_results(xml_doc: str, max_results: int = 10) -> List[Dict[str, str]]:
	"""Parse a Bing RSS XML document into a list of {url, title, snippet} dicts."""
	results: List[Dict[str, str]] = []
	for m in _RSS_ITEM_RE.finditer(xml_doc):
		body = m.group("body") or ""
		vals: Dict[str, str] = {"title": "", "link": "", "description": ""}
		for tm in _RSS_TAG_RE.finditer(body):
			tag = tm.group("tag").lower()
			val = html.unescape(tm.group("val") or "")
			vals[tag] = _regex_extract(val)
		href = vals.get("link", "").strip()
		parsed = urllib.parse.urlparse(href)
		if parsed.scheme not in {"http", "https"} or "bing.com" in (parsed.netloc or ""):
			continue
		if any(r["url"] == href for r in results):
			continue
		results.append({
			"url": href,
			"title": vals.get("title", ""),
			"snippet": vals.get("description", ""),
		})
		if len(results) >= max(1, max_results):
			break
	return results


# ── module-level singletons ───────────────────────────────────────────────────
_engine: Optional[WebSearchEngine] = None
_fetcher: Optional[PageFetcher] = None


def _make_cache() -> Optional[WebSearchCache]:
	try:
		return WebSearchCache(db_path=Path("data") / "cache" / "web_search.sqlite")
	except Exception:
		return None


def _get_engine() -> WebSearchEngine:
	global _engine
	if _engine is None:
		_engine = WebSearchEngine(cache=_make_cache())
	return _engine


def _get_fetcher() -> PageFetcher:
	global _fetcher
	if _fetcher is None:
		_fetcher = PageFetcher(cache=getattr(_get_engine(), "_cache", None))
	return _fetcher


# ── public API ────────────────────────────────────────────────────────────────
def search_web(query: str, max_results: int = 5, timeout: float = 10.0) -> Dict[str, Any]:
	"""Search the web using the provider cascade.

	Returns {"search_query": str, "search_attempts": list, "search_results": list}.
	Drop-in replacement for the old _search_web() in internet_fallback.py.
	"""
	return _get_engine().search(query, max_results=max_results, timeout=timeout)


def fetch_page(url: str, timeout: float = 10.0, max_chars: int = 12_000) -> str:
	"""Fetch a URL and return clean extracted text, capped at max_chars.

	Drop-in replacement for the old _fetch_url() + _extract_visible_text() pair
	in internet_fallback.py.
	"""
	return _get_fetcher().fetch(url, timeout=timeout, max_chars=max_chars)
