"""Unit tests for retrieval.web_search."""
from __future__ import annotations

import sys
import urllib.parse
from pathlib import Path
from unittest.mock import MagicMock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.web_search import (
	BingHTMLProvider,
	BingRSSProvider,
	DuckDuckGoProvider,
	PageFetcher,
	SearchResult,
	WebSearchCache,
	WebSearchEngine,
	_is_relevant,
	_normalize_url,
	_query_terms,
	_regex_extract,
	extract_page_text,
	fetch_page,
	parse_bing_rss_results,
	search_web,
)


# ── WebSearchCache ─────────────────────────────────────────────────────────────
class TestWebSearchCache:
	def test_miss_returns_none(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		assert cache.get("search", "bing_html", "test query") is None

	def test_set_and_get_round_trip(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		cache.set("search", "bing_html", "query", value=[{"url": "https://example.com"}])
		result = cache.get("search", "bing_html", "query")
		assert result == [{"url": "https://example.com"}]

	def test_expired_entry_returns_none(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		cache.set("search", "test", "q", value={"x": 1}, ttl=-1)
		assert cache.get("search", "test", "q") is None

	def test_purge_removes_expired(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		cache.set("search", "test", "q", value={"x": 1}, ttl=-1)
		cache.purge_expired()
		assert cache.get("search", "test", "q") is None

	def test_different_keys_stored_independently(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		cache.set("search", "bing", "q1", value="a")
		cache.set("search", "bing", "q2", value="b")
		assert cache.get("search", "bing", "q1") == "a"
		assert cache.get("search", "bing", "q2") == "b"

	def test_overwrite_updates_value(self, tmp_path) -> None:
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		cache.set("k", value="v1")
		cache.set("k", value="v2")
		assert cache.get("k") == "v2"


# ── _normalize_url ─────────────────────────────────────────────────────────────
class TestNormalizeUrl:
	def test_plain_https_url_unchanged(self) -> None:
		assert _normalize_url("https://example.com/page") == "https://example.com/page"

	def test_scheme_relative_gets_https(self) -> None:
		assert _normalize_url("//example.com/page") == "https://example.com/page"

	def test_bing_url_rejected(self) -> None:
		assert _normalize_url("https://www.bing.com/search?q=test") == ""

	def test_duckduckgo_url_rejected(self) -> None:
		assert _normalize_url("https://duckduckgo.com/html/?q=test") == ""

	def test_ddg_redirect_unwrapped(self) -> None:
		target = "https://example.com/target"
		encoded = urllib.parse.quote(target, safe="")
		href = f"https://duckduckgo.com/l/?uddg={encoded}&rut=abc"
		assert _normalize_url(href) == target

	def test_javascript_scheme_rejected(self) -> None:
		assert _normalize_url("javascript:void(0)") == ""

	def test_ftp_scheme_rejected(self) -> None:
		assert _normalize_url("ftp://example.com/file") == ""

	def test_empty_returns_empty(self) -> None:
		assert _normalize_url("") == ""

	def test_html_entity_decoded(self) -> None:
		assert _normalize_url("https://example.com/page?a=1&amp;b=2") == "https://example.com/page?a=1&b=2"


# ── _query_terms ──────────────────────────────────────────────────────────────
class TestQueryTerms:
	def test_extracts_content_words(self) -> None:
		terms = _query_terms("What is backpropagation?")
		assert "backpropagation" in terms
		assert "what" not in terms
		assert "is" not in terms

	def test_empty_query_returns_empty(self) -> None:
		assert _query_terms("") == []

	def test_stopwords_removed(self) -> None:
		terms = _query_terms("tell me about the learning rate")
		assert "tell" not in terms
		assert "me" not in terms
		assert "learning" in terms
		assert "rate" in terms

	def test_short_tokens_excluded(self) -> None:
		# Single-letter tokens below the length threshold are skipped
		terms = _query_terms("a b c deep learning")
		assert "a" not in terms
		assert "b" not in terms
		assert "deep" in terms
		assert "learning" in terms


# ── _regex_extract ────────────────────────────────────────────────────────────
class TestRegexExtract:
	def test_strips_html_tags(self) -> None:
		result = _regex_extract("<p>Hello <b>world</b></p>")
		assert "Hello" in result
		assert "world" in result
		assert "<" not in result

	def test_prefers_article_content(self) -> None:
		raw = (
			"<html><body>"
			"<nav>Nav junk Language: Español Français</nav>"
			"<article><p>Main article text about science.</p></article>"
			"<footer>Copyright 2024</footer>"
			"</body></html>"
		)
		result = _regex_extract(raw)
		assert "Main article text" in result
		assert "Nav junk" not in result

	def test_strips_script_and_style(self) -> None:
		raw = "<html><body><script>alert('x')</script><p>Clean text.</p></body></html>"
		result = _regex_extract(raw)
		assert "alert" not in result
		assert "Clean text" in result

	def test_html_entities_decoded(self) -> None:
		result = _regex_extract("<p>Caf&eacute; &amp; more</p>")
		assert "Café" in result or "Caf" in result
		assert "&eacute;" not in result


# ── extract_page_text ─────────────────────────────────────────────────────────
class TestExtractPageText:
	def test_returns_nonempty_for_valid_html(self) -> None:
		raw = "<html><body><p>Hello world, this is a valid page with content.</p></body></html>"
		text = extract_page_text(raw)
		assert len(text) > 0

	def test_empty_html_returns_empty_or_short(self) -> None:
		text = extract_page_text("")
		assert text == "" or len(text) < 20

	def test_article_content_preferred_over_nav(self) -> None:
		raw = (
			"<html><body>"
			"<nav>Home About Contact Language Español Français Deutsch</nav>"
			"<article><p>Argentina won the 2022 FIFA World Cup final against France.</p></article>"
			"<footer>Copyright 2022</footer>"
			"</body></html>"
		)
		text = extract_page_text(raw)
		assert "Argentina" in text


# ── parse_bing_rss_results ────────────────────────────────────────────────────
class TestParseBingRssResults:
	def test_parses_single_item(self) -> None:
		xml = """<rss><channel>
		<item>
		  <title>Test Title</title>
		  <link>https://example.com/page</link>
		  <description>Test description text.</description>
		</item>
		</channel></rss>"""
		results = parse_bing_rss_results(xml, max_results=5)
		assert len(results) == 1
		assert results[0]["url"] == "https://example.com/page"
		assert results[0]["title"] == "Test Title"
		assert "description" in results[0]["snippet"].lower()

	def test_skips_bing_internal_links(self) -> None:
		xml = """<rss><channel>
		<item>
		  <title>Internal</title>
		  <link>https://www.bing.com/internal</link>
		  <description>Internal</description>
		</item>
		</channel></rss>"""
		results = parse_bing_rss_results(xml, max_results=5)
		assert len(results) == 0

	def test_deduplicates_urls(self) -> None:
		xml = """<rss><channel>
		<item>
		  <title>Title A</title>
		  <link>https://example.com/page</link>
		  <description>A</description>
		</item>
		<item>
		  <title>Title B</title>
		  <link>https://example.com/page</link>
		  <description>B</description>
		</item>
		</channel></rss>"""
		results = parse_bing_rss_results(xml, max_results=5)
		assert len(results) == 1

	def test_respects_max_results(self) -> None:
		items = "\n".join(
			f"""<item>
			<title>T{i}</title>
			<link>https://example.com/page{i}</link>
			<description>D{i}</description>
			</item>"""
			for i in range(10)
		)
		xml = f"<rss><channel>{items}</channel></rss>"
		results = parse_bing_rss_results(xml, max_results=3)
		assert len(results) == 3


# ── BingRSSProvider ───────────────────────────────────────────────────────────
class TestBingRSSProvider:
	def test_search_url_contains_query_and_rss_format(self) -> None:
		p = BingRSSProvider()
		url = p.search_url("FIFA World Cup 2022")
		assert "FIFA" in url or "FIFA+World" in url
		assert "format=rss" in url

	def test_search_returns_empty_on_http_error(self, monkeypatch) -> None:
		def _raise(*a, **kw):
			raise Exception("timeout")
		monkeypatch.setattr("retrieval.web_search._http_get", _raise)
		p = BingRSSProvider()
		assert p.search("query", max_results=5, timeout=2.0) == []

	def test_search_parses_rss_response(self, monkeypatch) -> None:
		xml = """<rss><channel>
		<item>
		  <title>Result</title>
		  <link>https://example.com/result</link>
		  <description>Good snippet about the query</description>
		</item>
		</channel></rss>"""
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: xml)
		p = BingRSSProvider()
		results = p.search("query", max_results=5, timeout=2.0)
		assert len(results) == 1
		assert results[0].url == "https://example.com/result"
		assert results[0].provider == "bing_rss"


# ── BingHTMLProvider ──────────────────────────────────────────────────────────
class TestBingHTMLProvider:
	def test_search_url_contains_query(self) -> None:
		p = BingHTMLProvider()
		url = p.search_url("neural networks")
		assert "neural" in url or "neural+networks" in url

	def test_search_parses_h2_links(self, monkeypatch) -> None:
		html_doc = (
			"<html><body>"
			'<h2 class="b_algo"><a href="https://wikipedia.org/wiki/Neural_network">Neural network - Wikipedia</a></h2>'
			'<p class="b_lineclamp1">A neural network is a computational model.</p>'
			"</body></html>"
		)
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: html_doc)
		p = BingHTMLProvider()
		results = p.search("neural networks", max_results=5, timeout=2.0)
		assert len(results) == 1
		assert "wikipedia.org" in results[0].url
		assert results[0].provider == "bing_html"

	def test_search_skips_blocked_domains(self, monkeypatch) -> None:
		html_doc = (
			"<html><body>"
			'<h2><a href="https://www.microsoft.com/page">Microsoft page</a></h2>'
			'<h2><a href="https://go.microsoft.com/fwlink?id=1">MS Link</a></h2>'
			"</body></html>"
		)
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: html_doc)
		p = BingHTMLProvider()
		results = p.search("query", max_results=5, timeout=2.0)
		assert len(results) == 0

	def test_search_deduplicates_urls(self, monkeypatch) -> None:
		html_doc = (
			"<html><body>"
			'<h2><a href="https://example.com/page">Title A</a></h2>'
			'<h2><a href="https://example.com/page">Title B</a></h2>'
			"</body></html>"
		)
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: html_doc)
		p = BingHTMLProvider()
		results = p.search("query", max_results=5, timeout=2.0)
		assert len(results) == 1

	def test_search_returns_empty_on_http_error(self, monkeypatch) -> None:
		def _raise(*a, **kw):
			raise Exception("connection refused")
		monkeypatch.setattr("retrieval.web_search._http_get", _raise)
		p = BingHTMLProvider()
		assert p.search("q", max_results=5, timeout=2.0) == []


# ── DuckDuckGoProvider ────────────────────────────────────────────────────────
class TestDuckDuckGoProvider:
	def test_search_url_contains_query(self) -> None:
		p = DuckDuckGoProvider()
		url = p.search_url("backpropagation")
		assert "backpropagation" in url
		assert "duckduckgo.com" in url

	def test_search_parses_result_a_links(self, monkeypatch) -> None:
		html_doc = (
			"<html><body>"
			'<a class="result__a" href="https://example.com/result">Result Title</a>'
			'<a class="result__snippet">A good snippet about the result.</a>'
			"</body></html>"
		)
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: html_doc)
		p = DuckDuckGoProvider()
		results = p.search("query result", max_results=5, timeout=2.0)
		assert len(results) == 1
		assert results[0].url == "https://example.com/result"
		assert results[0].provider == "duckduckgo"

	def test_search_fallback_when_no_result_a(self, monkeypatch) -> None:
		html_doc = (
			"<html><body>"
			'<a href="https://example.com/fallback">Fallback Title</a>'
			"</body></html>"
		)
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: html_doc)
		p = DuckDuckGoProvider()
		results = p.search("fallback query", max_results=5, timeout=2.0)
		assert any(r.url == "https://example.com/fallback" for r in results)

	def test_search_returns_empty_on_http_error(self, monkeypatch) -> None:
		def _raise(*a, **kw):
			raise Exception("timeout")
		monkeypatch.setattr("retrieval.web_search._http_get", _raise)
		p = DuckDuckGoProvider()
		assert p.search("q", max_results=5, timeout=2.0) == []


# ── WebSearchEngine ───────────────────────────────────────────────────────────
class TestWebSearchEngine:
	def _make_provider(self, name: str, results: list, search_url: str = "") -> MagicMock:
		provider = MagicMock()
		provider.name = name
		provider.search_url.return_value = search_url or f"https://{name}.example.com/search?q=test"
		provider.search.return_value = results
		return provider

	def test_returns_results_from_first_provider(self) -> None:
		r = SearchResult(
			url="https://example.com", title="Title matching test", snippet="test content",
			provider="p1", search_url="http://s",
		)
		p1 = self._make_provider("p1", [r])
		engine = WebSearchEngine(providers=[p1], cache=None)
		result = engine.search("test content here", max_results=5)
		assert len(result["search_results"]) == 1
		assert result["search_results"][0]["url"] == "https://example.com"
		assert result["search_query"] == "test content here"

	def test_falls_through_on_soft_block(self) -> None:
		# p1 returns irrelevant results (no query-term overlap) → soft-block skip
		r1 = SearchResult(url="https://p1.com", title="Unrelated stuff", snippet="nothing matches", provider="p1", search_url="")
		r2 = SearchResult(url="https://p2.com", title="testquery result found", snippet="relevant content", provider="p2", search_url="")
		p1 = self._make_provider("p1", [r1])
		p2 = self._make_provider("p2", [r2])
		engine = WebSearchEngine(providers=[p1, p2], cache=None)
		result = engine.search("testquery", max_results=5)
		providers_used = {r["provider"] for r in result["search_results"]}
		assert "p2" in providers_used
		soft_blocked = [a for a in result["search_attempts"] if a.get("error") == "soft_blocked_irrelevant_results"]
		assert any(a["provider"] == "p1" for a in soft_blocked)

	def test_deduplicates_across_providers(self) -> None:
		url = "https://shared.com/page"
		r1 = SearchResult(url=url, title="T1 shared content", snippet="", provider="p1", search_url="")
		r2 = SearchResult(url=url, title="T2 shared content", snippet="", provider="p2", search_url="")
		p1 = self._make_provider("p1", [r1])
		p2 = self._make_provider("p2", [r2])
		engine = WebSearchEngine(providers=[p1, p2], cache=None)
		result = engine.search("shared content page", max_results=10)
		urls = [r["url"] for r in result["search_results"]]
		assert urls.count(url) == 1

	def test_caches_results_and_avoids_second_call(self, tmp_path) -> None:
		r = SearchResult(
			url="https://example.com", title="Title matching query words",
			snippet="query words result", provider="p1", search_url="",
		)
		call_count = {"n": 0}

		def fake_search(*a, **kw):
			call_count["n"] += 1
			return [r]

		p1 = MagicMock()
		p1.name = "p1"
		p1.search_url.return_value = "https://p1.example.com/search?q=test"
		p1.search.side_effect = fake_search
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		engine = WebSearchEngine(providers=[p1], cache=cache)
		engine.search("query words result", max_results=5)
		engine.search("query words result", max_results=5)
		assert call_count["n"] == 1

	def test_returns_empty_when_all_providers_fail(self) -> None:
		p1 = self._make_provider("p1", [])
		engine = WebSearchEngine(providers=[p1], cache=None)
		result = engine.search("anything at all")
		assert result["search_results"] == []
		assert result["search_query"] == "anything at all"

	def test_attempts_list_records_all_providers(self) -> None:
		p1 = self._make_provider("p1", [])
		p2 = self._make_provider("p2", [])
		engine = WebSearchEngine(providers=[p1, p2], cache=None)
		result = engine.search("some query")
		provider_names = {a["provider"] for a in result["search_attempts"]}
		assert "p1" in provider_names
		assert "p2" in provider_names

	def test_all_providers_tried_results_capped(self) -> None:
		# Engine now tries every provider and returns all unique results up to
		# max_results * 3; it no longer stops mid-cascade on the first provider.
		results = [
			SearchResult(url=f"https://example{i}.com", title=f"Result {i} query", snippet="query", provider="p1", search_url="")
			for i in range(10)
		]
		p1 = self._make_provider("p1", results)
		engine = WebSearchEngine(providers=[p1], cache=None)
		result = engine.search("query result", max_results=3)
		# cap is max_results * 3 = 9; provider returns 10, so we get 9
		assert len(result["search_results"]) == 9


# ── PageFetcher ───────────────────────────────────────────────────────────────
class TestPageFetcher:
	def test_returns_text_on_success(self, monkeypatch) -> None:
		monkeypatch.setattr(
			"retrieval.web_search._http_get",
			lambda *a, **kw: "<html><article><p>Hello world content here is long enough.</p></article></html>",
		)
		fetcher = PageFetcher(cache=None)
		text = fetcher.fetch("https://example.com", timeout=5.0)
		assert "Hello" in text or "world" in text or "content" in text

	def test_returns_empty_on_http_error(self, monkeypatch) -> None:
		def _raise(*a, **kw):
			raise Exception("connection refused")
		monkeypatch.setattr("retrieval.web_search._http_get", _raise)
		fetcher = PageFetcher(cache=None)
		assert fetcher.fetch("https://example.com") == ""

	def test_respects_max_chars(self, monkeypatch) -> None:
		long_text = "word " * 5000
		monkeypatch.setattr(
			"retrieval.web_search._http_get",
			lambda *a, **kw: f"<html><article><p>{long_text}</p></article></html>",
		)
		fetcher = PageFetcher(cache=None)
		text = fetcher.fetch("https://example.com", max_chars=500)
		assert len(text) <= 500

	def test_caches_and_avoids_refetch(self, tmp_path, monkeypatch) -> None:
		call_count = {"n": 0}

		def fake_get(*a, **kw):
			call_count["n"] += 1
			return "<html><article><p>Cached content here is long enough to be useful prose text for the test.</p></article></html>"

		monkeypatch.setattr("retrieval.web_search._http_get", fake_get)
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		fetcher = PageFetcher(cache=cache)
		fetcher.fetch("https://example.com")
		fetcher.fetch("https://example.com")
		assert call_count["n"] == 1

	def test_empty_html_returns_empty(self, monkeypatch) -> None:
		monkeypatch.setattr("retrieval.web_search._http_get", lambda *a, **kw: "")
		fetcher = PageFetcher(cache=None)
		assert fetcher.fetch("https://example.com") == ""

	def test_different_urls_fetched_independently(self, tmp_path, monkeypatch) -> None:
		responses = {
			"https://a.com": "<html><article><p>Content from site A, long enough to cache.</p></article></html>",
			"https://b.com": "<html><article><p>Content from site B, also long enough to cache.</p></article></html>",
		}
		monkeypatch.setattr("retrieval.web_search._http_get", lambda url, **kw: responses.get(url, ""))
		cache = WebSearchCache(db_path=tmp_path / "cache.sqlite")
		fetcher = PageFetcher(cache=cache)
		text_a = fetcher.fetch("https://a.com")
		text_b = fetcher.fetch("https://b.com")
		assert text_a != text_b


# ── _is_relevant ──────────────────────────────────────────────────────────────
class TestIsRelevant:
	def test_relevant_when_term_in_title(self) -> None:
		r = SearchResult(url="https://x.com", title="backpropagation explained", snippet="", provider="p", search_url="")
		assert _is_relevant(r, ["backpropagation"]) is True

	def test_irrelevant_when_no_term_overlap(self) -> None:
		r = SearchResult(url="https://x.com", title="cooking recipes", snippet="pasta sauce", provider="p", search_url="")
		assert _is_relevant(r, ["backpropagation"]) is False

	def test_always_relevant_when_no_query_terms(self) -> None:
		r = SearchResult(url="https://x.com", title="anything", snippet="", provider="p", search_url="")
		assert _is_relevant(r, []) is True

	def test_relevant_when_term_in_url(self) -> None:
		r = SearchResult(url="https://example.com/backpropagation", title="Deep Learning", snippet="", provider="p", search_url="")
		assert _is_relevant(r, ["backpropagation"]) is True
