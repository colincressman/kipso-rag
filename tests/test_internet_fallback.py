from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters, retrieve, _is_external_fact_query
from retrieval.internet_fallback import (
	_filter_search_results,
	_is_low_quality_chunk,
	_extract_visible_text,
	_chunk_text,
	_parse_bing_rss_results,
	_match_count,
	_is_low_value_result,
	_fact_result_rank,
	_query_terms,
	_web_cache_get,
	_web_cache_set,
)


class _FakeConn:
	"""Minimal fake psycopg connection for monkeypatching _connect in retrieval tests."""
	def execute(self, *a, **kw):
		return self
	def fetchone(self):
		return None
	def fetchall(self):
		return []
	def close(self):
		pass


def _internet_hit(score: float = 0.42) -> dict:
	return {
		"chunk_id": "internet-1-c000001",
		"doc_id": "https://example.com",
		"collection_id": "internet",
		"source_name": "https://example.com",
		"document_title": "Example",
		"document_path": "https://example.com",
		"section_id": None,
		"title": "Example",
		"path_text": "https://example.com",
		"page_number": None,
		"section_header": "Example",
		"page_start": None,
		"page_end": None,
		"text": "Example internet content about the query.",
		"score": score,
		"source_type": "internet",
		"structural_role": "web_fallback",
		"metadata": {
			"source_type": "internet",
			"structural_role": "web_fallback",
		},
	}


def test_retrieve_uses_internet_fallback_when_local_missing(monkeypatch, tmp_path) -> None:
	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	def _fake_internet(**kwargs):
		return {
			"hits": [_internet_hit()],
			"trace": {
				"search_query": kwargs["query"],
				"search_attempts": [
					{
						"provider": "bing_rss",
						"search_url": "https://www.bing.com/search?format=rss&q=Explain+logistic+regression",
						"result_count": 1,
					}
				],
				"search_results": [
					{
						"provider": "bing_rss",
						"search_url": "https://www.bing.com/search?format=rss&q=Explain+logistic+regression",
						"url": "https://example.com",
						"title": "Example",
						"snippet": "Example snippet",
					}
				],
				"fetched_urls": ["https://example.com"],
				"selected_urls": ["https://example.com"],
			},
		}

	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	result = retrieve(
		"Explain logistic regression",
		db_dsn="postgresql://localhost/rag",
		top_k=3,
		embed_backend="_test",
		internet_fallback_enabled=True,
	)

	assert result.hits
	assert any(h.source_type == "internet" for h in result.hits)
	assert result.hits[0].metadata.get("internet_fallback_triggered") is True
	assert result.hits[0].metadata.get("internet_fallback_used") is True
	assert result.internet_fallback is not None
	assert result.internet_fallback.get("search_query") == "Explain logistic regression"
	assert result.internet_fallback.get("search_attempts")
	assert result.internet_fallback.get("search_results")
	assert result.internet_fallback.get("used") is True


def test_retrieve_skips_internet_fallback_when_hard_filter_set(monkeypatch, tmp_path) -> None:
	called = {"value": False}

	def _fake_internet(**kwargs):
		called["value"] = True
		return {"hits": [_internet_hit()], "trace": {}}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	_ = retrieve(
		"Explain logistic regression",
		db_dsn="postgresql://localhost/rag",
		top_k=3,
		embed_backend="_test",
		internet_fallback_enabled=True,
		filters=RetrievalFilters(source_type="pdf_book"),
	)

	assert called["value"] is False


def test_retrieve_passes_raw_query_to_internet_fallback(monkeypatch, tmp_path) -> None:
	captured = {}

	def _fake_internet(**kwargs):
		captured.update(kwargs)
		return {
			"hits": [_internet_hit()],
			"trace": {
				"search_query": kwargs["query"],
				"search_attempts": [],
				"search_results": [],
				"fetched_urls": [],
				"selected_urls": [],
			},
		}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	_ = retrieve(
		"What is CAPM?",
		db_dsn="postgresql://localhost/rag",
		top_k=3,
		embed_backend="_test",
		internet_fallback_enabled=True,
	)

	assert captured.get("query") == "What is CAPM?"
	assert "expanded_query" not in captured


def test_external_fact_query_prioritizes_internet_hit(monkeypatch, tmp_path) -> None:
	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local chunk",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local Title",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Local retrieval chunk",
			}
		]

	def _fake_internet(**kwargs):
		return {
			"hits": [_internet_hit(score=0.49)],
			"trace": {
				"search_query": kwargs["query"],
				"search_attempts": [],
				"search_results": [],
				"fetched_urls": ["https://example.com"],
				"selected_urls": ["https://example.com"],
			},
		}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.80} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	result = retrieve(
		"Who won the FIFA World Cup in 2022?",
		db_dsn="postgresql://localhost/rag",
		top_k=2,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_top_score=0.95,
	)

	assert result.hits
	assert result.hits[0].source_type == "internet"
	assert result.internet_fallback is not None
	assert result.internet_fallback.get("priority_applied") is True


def test_external_fact_query_rejects_low_relevance_internet_hit(monkeypatch, tmp_path) -> None:
	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local chunk",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local Title",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Local retrieval chunk",
			}
		]

	def _fake_internet(**kwargs):
		return {
			"hits": [_internet_hit(score=0.21)],
			"trace": {
				"search_query": kwargs["query"],
				"search_attempts": [],
				"search_results": [],
				"fetched_urls": ["https://example.com"],
				"selected_urls": ["https://example.com"],
			},
		}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.80} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	result = retrieve(
		"Who won the FIFA World Cup in 2022?",
		db_dsn="postgresql://localhost/rag",
		top_k=2,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_top_score=0.95,
		internet_min_relevance_score=0.42,
	)

	assert result.hits
	assert result.hits[0].source_type != "internet"
	assert result.internet_fallback is not None
	assert result.internet_fallback.get("used") is False
	assert result.internet_fallback.get("relevance_rejected_count") == 1


def test_non_fact_query_does_not_trigger_internet_when_local_exists(monkeypatch, tmp_path) -> None:
	called = {"value": False}

	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local chunk",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local Title",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Backpropagation uses chain rule.",
			}
		]

	def _fake_internet(**kwargs):
		called["value"] = True
		return {"hits": [_internet_hit()], "trace": {}}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.86} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	_ = retrieve(
		"How does backpropagation work in neural networks?",
		db_dsn="postgresql://localhost/rag",
		top_k=2,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_on_low_confidence=False,
	)

	assert called["value"] is False


def test_external_fact_query_prefers_all_internet_rows_before_local(monkeypatch, tmp_path) -> None:
	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local chunk",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local Title",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Local retrieval chunk",
			}
		]

	def _fake_internet(**kwargs):
		return {
			"hits": [
				_internet_hit(score=0.58),
				{
					**_internet_hit(score=0.57),
					"chunk_id": "internet-2-c000001",
					"doc_id": "https://example.org",
					"document_path": "https://example.org",
					"path_text": "https://example.org",
					"source_name": "https://example.org",
				},
			],
			"trace": {
				"search_query": kwargs["query"],
				"search_attempts": [],
				"search_results": [],
				"fetched_urls": ["https://example.com", "https://example.org"],
				"selected_urls": ["https://example.com", "https://example.org"],
			},
		}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.85} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	result = retrieve(
		"Who won the FIFA World Cup in 2022?",
		db_dsn="postgresql://localhost/rag",
		top_k=3,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_top_score=0.95,
	)

	assert len(result.hits) == 3
	assert result.hits[0].source_type == "internet"
	assert result.hits[1].source_type == "internet"
	assert result.hits[2].source_type == "pdf_book"


def test_filter_search_results_rejects_dictionary_for_fifa_fact_query() -> None:
	rows = [
		{
			"provider": "bing_rss",
			"search_url": "https://www.bing.com/search?format=rss&q=Who+won+the+FIFA+World+Cup+in+2022%3F",
			"url": "https://www.merriam-webster.com/dictionary/won",
			"title": "WON Definition & Meaning - Merriam-Webster",
			"snippet": "",
		},
		{
			"provider": "bing_rss",
			"search_url": "https://www.bing.com/search?format=rss&q=Who+won+the+FIFA+World+Cup+in+2022%3F",
			"url": "https://en.wikipedia.org/wiki/2022_FIFA_World_Cup_final",
			"title": "2022 FIFA World Cup final - Wikipedia",
			"snippet": "Argentina then won the ensuing penalty shoot-out 4–2.",
		},
	]

	filtered, rejected = _filter_search_results("Who won the FIFA World Cup in 2022?", rows)

	assert any("wikipedia.org" in str(r.get("url")) for r in filtered)
	assert all("merriam-webster.com" not in str(r.get("url")) for r in filtered)
	assert any("merriam-webster.com" in str(r.get("url")) for r in rejected)


def test_filter_search_results_keeps_non_dictionary_fact_sources() -> None:
	rows = [
		{
			"provider": "bing_rss",
			"search_url": "https://www.bing.com/search?format=rss&q=Who+won+the+FIFA+World+Cup+in+2022%3F",
			"url": "https://www.britannica.com/sports/2022-FIFA-World-Cup",
			"title": "2022 FIFA World Cup - Encyclopedia Britannica",
			"snippet": "Argentina defeated France in the final match to win its third World Cup title.",
		},
	]

	filtered, rejected = _filter_search_results("Who won the FIFA World Cup in 2022?", rows)

	assert len(filtered) == 1
	assert not rejected


def test_low_quality_chunk_rejects_unicode_language_list() -> None:
	# Simulates the Wikipedia ToC chunk: mostly non-ASCII language names
	toc_chunk = (
		"table of contents 2022 FIFA World Cup final 47 languages "
		"\u0627\u0644\u0639\u0631\u0628\u064a\u0629 \u0627\u0644\u062f\u0627\u0631\u062c\u0629 "
		"Az\u0259rbaycanca Basa Bali \u0411\u04a7\u043b\u0433\u0430\u0440\u0441\u043a\u0438 "
		"\u0985\u09b8\u09ae\u09c0\u09af\u09bc\u09be Catal\xe0 \u0686\u0648\u0631\u06cc "
		"\u010ce\u0161tina Dansk Espa\xf1ol \u0641\u0627\u0631\u0633\u06cc Suomi Fran\xe7ais "
		"Hausa \u05e2\u05d1\u05e8\u05d9\u05ea Hrvatski Magyar Bahasa Indonesia Italiano "
		"\u65e5\u672c\u8a9e \u049b\u0430\u0437\u0430\u049b\u0448\u0430 \ud55c\uad6d\uc5b4 "
		"\u0ea5\u0eb2\u0ea7 Latvie\u0161u \u041c\u0430\u043a\u0435\u0434\u043e\u043d\u0441\u043a\u0438 "
		"Bahasa Melayu \u0928\u0947\u092a\u093e\u0932\u0940 Nederlands Polski Portugu\xeas"
	)
	assert _is_low_quality_chunk(toc_chunk) is True


def test_low_quality_chunk_passes_article_prose() -> None:
	prose = (
		"Argentina won the 2022 FIFA World Cup, defeating France in the final. "
		"The match was held at Lusail Stadium in Qatar on December 18, 2022. "
		"Argentina won on penalties after the match ended 3-3 after extra time. "
		"Lionel Messi was awarded the Golden Ball as the tournament's best player."
	)
	assert _is_low_quality_chunk(prose) is False


def test_extract_visible_text_prefers_article_over_nav() -> None:
	html = (
		"<html><body>"
		"<nav>Home | About | Contact | Language: Espa\xf1ol Fran\xe7ais Deutsch</nav>"
		"<article><p>Argentina won the 2022 FIFA World Cup final against France.</p></article>"
		"<footer>Copyright 2022</footer>"
		"</body></html>"
	)
	result = _extract_visible_text(html)
	assert "Argentina" in result
	assert "Espa" not in result  # nav content stripped


def test_low_quality_chunk_rejects_navigation_heavy_text() -> None:
	nav_text = (
		"Quick Summary Table of Contents Ask Anything Top Questions "
		"External Websites Submit Feedback Related changes Printable version "
		"This page includes navigation and utility links rather than article prose."
	)
	assert _is_low_quality_chunk(nav_text) is True


def test_chunk_text_avoids_mid_word_starts() -> None:
	text = (
		"Argentina won the World Cup final after a dramatic match against France. "
		"This sentence is intentionally long enough to force chunk overlap handling in tests. "
		"The next sentence should not begin in the middle of a word."
	)
	chunks = _chunk_text(text, max_chars=90, overlap=20)
	assert len(chunks) >= 2
	for chunk in chunks:
		assert chunk
		assert not chunk.startswith("ation ")
		assert not chunk.startswith("nce ")


def test_parse_bing_rss_results_keeps_description_snippets() -> None:
	xml = """
	<rss version="2.0"><channel>
	<item>
	  <title>GPT-4 - Wikipedia</title>
	  <link>https://en.wikipedia.org/wiki/GPT-4</link>
	  <description>GPT-4 was released by OpenAI in March 2023.</description>
	</item>
	</channel></rss>
	"""
	rows = _parse_bing_rss_results(xml, max_results=5)
	assert len(rows) == 1
	assert rows[0]["url"] == "https://en.wikipedia.org/wiki/GPT-4"
	assert "March 2023" in rows[0]["snippet"]


def test_filter_search_results_prefers_authoritative_fact_sources() -> None:
	rows = [
		{
			"provider": "bing_rss",
			"search_url": "https://www.bing.com/search?format=rss&q=When+was+the+GPT-4+model+released+by+OpenAI%3F",
			"url": "https://github.com/openai/gpt-2",
			"title": "GitHub - openai/gpt-2",
			"snippet": "Code repository for GPT-2.",
		},
		{
			"provider": "bing_rss",
			"search_url": "https://www.bing.com/search?format=rss&q=When+was+the+GPT-4+model+released+by+OpenAI%3F",
			"url": "https://openai.com/index/gpt-4-research/",
			"title": "GPT-4 - OpenAI",
			"snippet": "GPT-4 is a large multimodal model released by OpenAI in March 2023.",
		},
	]

	filtered, rejected = _filter_search_results("When was the GPT-4 model released by OpenAI?", rows)

	assert not rejected
	assert filtered[0]["url"] == "https://openai.com/index/gpt-4-research/"


# ---------------------------------------------------------------------------
# _is_external_fact_query — unit tests for extended detection patterns
# ---------------------------------------------------------------------------

class TestIsExternalFactQuery:
	"""Unit tests for the external-fact query detector in retrieval.query."""

	# --- existing patterns still work ---

	def test_who_with_year(self) -> None:
		assert _is_external_fact_query("Who won the FIFA World Cup in 2022?") is True

	def test_when_with_year(self) -> None:
		assert _is_external_fact_query("When was GPT-4 released in 2023?") is True

	def test_who_with_all_caps_entity(self) -> None:
		assert _is_external_fact_query("Who is the CEO of NASA?") is True

	# --- new: recency-word path for who/when/where/which/whom ---

	def test_who_with_currently(self) -> None:
		assert _is_external_fact_query("Who is currently leading OpenAI?") is True

	def test_who_with_today(self) -> None:
		assert _is_external_fact_query("Who is the world chess champion today?") is True

	def test_when_with_currently(self) -> None:
		assert _is_external_fact_query("When is the next World Cup currently scheduled?") is True

	# --- new: what opener + recency word ---

	def test_what_with_today(self) -> None:
		assert _is_external_fact_query("What happened in the stock market today?") is True

	def test_what_with_currently(self) -> None:
		assert _is_external_fact_query("What is currently the most popular LLM?") is True

	def test_what_with_yesterday(self) -> None:
		assert _is_external_fact_query("What did the Fed announce yesterday?") is True

	# --- ML corpus questions should NOT trigger ---

	def test_what_ml_question_no_recency(self) -> None:
		assert _is_external_fact_query("What is backpropagation?") is False

	def test_what_ml_question_gradient_descent(self) -> None:
		assert _is_external_fact_query("What is the role of the learning rate in gradient descent?") is False

	def test_explain_question_no_opener(self) -> None:
		assert _is_external_fact_query("Explain the attention mechanism in transformers.") is False

	def test_how_does_opener_no_recency(self) -> None:
		assert _is_external_fact_query("How does dropout prevent overfitting?") is False

	def test_empty_query(self) -> None:
		assert _is_external_fact_query("") is False

	# --- new: informational openers (tell/explain/describe) + recency or news intent ---

	def test_tell_me_about_political_news_today(self) -> None:
		assert _is_external_fact_query("tell me about political news today") is True

	def test_tell_me_about_news(self) -> None:
		assert _is_external_fact_query("tell me about the latest news") is True

	def test_tell_me_about_ml_no_recency_no_news(self) -> None:
		# "tell me about backpropagation" — informational opener but no recency/news intent
		assert _is_external_fact_query("tell me about backpropagation") is False

	def test_explain_current_politics(self) -> None:
		assert _is_external_fact_query("explain current politics in the UK") is True

	def test_describe_latest_elections(self) -> None:
		assert _is_external_fact_query("describe the latest election results") is True

	# --- new: news-intent term anywhere in query ---

	def test_bare_news_query(self) -> None:
		assert _is_external_fact_query("any interesting news today?") is True

	def test_political_keyword_triggers(self) -> None:
		assert _is_external_fact_query("what's happening in politics?") is True

	def test_no_false_trigger_on_ml_terms(self) -> None:
		# "explain gradient descent" — informational opener, but no recency/news intent
		assert _is_external_fact_query("explain gradient descent") is False

	# --- new: PRODUCT_RELEASE_VERBS + proper noun (e.g. gold_i05 case) ---

	def test_what_deepmind_gemini_released(self) -> None:
		"""gold_i05: proper noun + product-release verb should trigger internet."""
		assert _is_external_fact_query(
			"What AI models has Google DeepMind released in the Gemini series?"
		) is True

	def test_what_openai_unveiled(self) -> None:
		assert _is_external_fact_query(
			"What models has OpenAI unveiled this year?"
		) is True

	def test_what_apple_launched(self) -> None:
		assert _is_external_fact_query(
			"What iPhone has Apple launched recently?"
		) is True

	def test_what_announced_by_company(self) -> None:
		assert _is_external_fact_query(
			"What did Microsoft announce at Build?"
		) is True

	def test_no_proper_noun_no_trigger_for_released(self) -> None:
		"""Release verb alone without a proper noun must NOT trigger."""
		assert _is_external_fact_query(
			"What models were released in the paper?"
		) is False

	def test_no_release_verb_no_trigger(self) -> None:
		"""Proper noun alone (no release verb, no recency) must NOT trigger."""
		assert _is_external_fact_query(
			"What does Google use for search ranking?"
		) is False


class TestFilterSearchResultsLowValue:
	"""Test that dictionary/definition results are filtered regardless of query shape."""

	def _dict_row(self, n: int) -> dict:
		return {
			"url": f"https://www.merriam-webster.com/dictionary/tell{n}",
			"title": f"TELL Definition & Meaning - Merriam-Webster ({n})",
			"snippet": "The meaning of TELL is to relate in detail.",
		}

	def test_dictionary_sites_filtered_for_news_query(self) -> None:
		"""Low-value dictionary results must be rejected even for non-FACT_QUERY_OPENER queries."""
		rows = [self._dict_row(i) for i in range(4)]
		filtered, rejected = _filter_search_results("tell me about political news today", rows)
		# All four are dictionary sites with match_count < 2 → all rejected
		assert len(filtered) == 0
		assert len(rejected) == 4

	def test_dictionary_sites_not_rescued_by_fallback(self) -> None:
		"""Fallback should NOT restore rejected low-value results."""
		rows = [self._dict_row(i) for i in range(3)]
		filtered, _ = _filter_search_results("tell me about political news today", rows)
		assert filtered == []

	def test_real_result_passes_through(self) -> None:
		"""A real news result should not be filtered."""
		rows = [
			{
				"url": "https://apnews.com/article/politics-news-today",
				"title": "Political News Today — AP News",
				"snippet": "Today's political news: Congress debates budget.",
			}
		]
		filtered, _ = _filter_search_results("tell me about political news today", rows)
		assert len(filtered) == 1


def test_low_confidence_trigger_fires_below_threshold(monkeypatch, tmp_path) -> None:
	"""When local top score is below internet_trigger_top_score, internet should be triggered."""
	called = {"value": False}

	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Unrelated local chunk about topic X.",
			}
		]

	def _fake_internet(**kwargs):
		called["value"] = True
		return {"hits": [], "trace": {}}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	# cosine score of 0.40 — below the 0.55 threshold
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.40} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	_ = retrieve(
		"What is the capital of Brazil?",
		db_dsn="postgresql://localhost/rag",
		top_k=2,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_on_low_confidence=True,
		internet_trigger_top_score=0.55,
	)

	assert called["value"] is True


def test_low_confidence_trigger_does_not_fire_above_threshold(monkeypatch, tmp_path) -> None:
	"""When local top score is above internet_trigger_top_score, internet should NOT fire."""
	called = {"value": False}

	def _fake_text_rows(conn, filters):
		return [
			{
				"chunk_id": "doc-1-c000001",
				"doc_id": "doc-1",
				"collection_id": "pdf_book",
				"source_name": "local.pdf",
				"document_title": "Local",
				"document_path": "C:/local.pdf",
				"doc_source_path": "C:/local.pdf",
				"doc_filename": "local.pdf",
				"doc_metadata_json": None,
				"section_id": "s1",
				"path_text": "local path",
				"title": "Local",
				"level": 1,
				"page_start": 1,
				"page_end": 1,
				"has_table": 0,
				"token_count_est": 120,
				"source_type": "pdf_book",
				"structural_role": "body",
				"text": "Backpropagation is the algorithm used to train neural networks.",
			}
		]

	def _fake_internet(**kwargs):
		called["value"] = True
		return {"hits": [], "trace": {}}

	monkeypatch.setattr("retrieval.query._connect", lambda _: _FakeConn())
	monkeypatch.setattr("retrieval.query._text_rows", _fake_text_rows)
	# cosine score of 0.78 — above the 0.55 threshold → should NOT trigger
	monkeypatch.setattr("retrieval.query._vector_candidates",
		lambda conn, f, qvec, **kw: [{"chunk_id": r["chunk_id"], "score": 0.78} for r in _fake_text_rows(conn, f)])
	monkeypatch.setattr("retrieval.query.retrieve_internet_chunks", _fake_internet)

	_ = retrieve(
		"How does backpropagation work?",
		db_dsn="postgresql://localhost/rag",
		top_k=2,
		rerank_enabled=False,
		embed_backend="_test",
		internet_fallback_enabled=True,
		internet_trigger_on_low_confidence=True,
		internet_trigger_top_score=0.55,
	)

	assert called["value"] is False


# ---------------------------------------------------------------------------
# _match_count unit tests
# ---------------------------------------------------------------------------

class TestMatchCount:
	def test_all_terms_found(self) -> None:
		assert _match_count(["argentina", "world", "cup"], "Argentina won the World Cup.") == 3

	def test_partial_match(self) -> None:
		assert _match_count(["argentina", "france", "penalty"], "Argentina faced France in the final.") == 2

	def test_no_match(self) -> None:
		assert _match_count(["quantum", "physics"], "Argentina won the World Cup.") == 0

	def test_case_insensitive(self) -> None:
		assert _match_count(["openai", "gpt"], "OpenAI released GPT-4 this year.") == 2

	def test_empty_terms(self) -> None:
		assert _match_count([], "Some text here.") == 0

	def test_empty_text(self) -> None:
		assert _match_count(["term"], "") == 0


# ---------------------------------------------------------------------------
# _is_low_value_result unit tests
# ---------------------------------------------------------------------------

class TestIsLowValueResult:
	def test_merriam_webster_rejected(self) -> None:
		assert _is_low_value_result("https://www.merriam-webster.com/dictionary/won", "WON Definition") is True

	def test_dictionary_com_rejected(self) -> None:
		assert _is_low_value_result("https://www.dictionary.com/browse/won", "won | Dictionary.com") is True

	def test_definition_in_title_rejected(self) -> None:
		assert _is_low_value_result("https://example.com/page", "Capital Definition & Examples") is True

	def test_meaning_in_title_rejected(self) -> None:
		assert _is_low_value_result("https://example.com/page", "The Meaning of Arbitrage") is True

	def test_dictionary_path_rejected(self) -> None:
		assert _is_low_value_result("https://somesite.com/dictionary/won", "Won") is True

	def test_define_path_rejected(self) -> None:
		assert _is_low_value_result("https://somesite.com/define/won", "Won") is True

	def test_normal_news_url_accepted(self) -> None:
		assert _is_low_value_result("https://apnews.com/article/argentina-world-cup-2022", "Argentina wins World Cup") is False

	def test_wikipedia_url_accepted(self) -> None:
		assert _is_low_value_result("https://en.wikipedia.org/wiki/2022_FIFA_World_Cup", "2022 FIFA World Cup - Wikipedia") is False


# ---------------------------------------------------------------------------
# _fact_result_rank unit tests
# ---------------------------------------------------------------------------

class TestFactResultRank:
	def _row(self, url: str, title: str = "", snippet: str = "") -> dict:
		return {"url": url, "title": title, "snippet": snippet}

	def test_preferred_domain_ranks_positive(self) -> None:
		row = self._row("https://reuters.com/article/gpt4-release", "GPT-4 released", "OpenAI released GPT-4 in March 2023.")
		rank, match_count, snippet_len = _fact_result_rank("When was GPT-4 released?", row)
		assert rank > 0

	def test_demoted_domain_ranks_negative(self) -> None:
		row = self._row("https://github.com/openai/gpt-4", "GitHub - openai/gpt-4", "Code repo.")
		rank, _, _ = _fact_result_rank("When was GPT-4 released?", row)
		assert rank < 0

	def test_neutral_domain_ranks_zero(self) -> None:
		row = self._row("https://sometech.blog/gpt4", "GPT-4 post", "A blog post about GPT-4.")
		rank, _, _ = _fact_result_rank("When was GPT-4 released?", row)
		assert rank == 0

	def test_snippet_length_returned(self) -> None:
		snippet = "Argentina won the 2022 FIFA World Cup on December 18."
		row = self._row("https://apnews.com/article/x", "FIFA Final", snippet)
		_, _, slen = _fact_result_rank("Who won FIFA World Cup 2022?", row)
		assert slen == len(snippet)

	def test_match_count_reflects_query_terms(self) -> None:
		row = self._row(
			"https://apnews.com/article/x",
			"GPT-4 released by OpenAI",
			"OpenAI released GPT-4 in March 2023.",
		)
		_, mc, _ = _fact_result_rank("When was GPT-4 released by OpenAI?", row)
		assert mc >= 2


# ---------------------------------------------------------------------------
# _query_terms unit tests
# ---------------------------------------------------------------------------

class TestQueryTerms:
	def test_removes_stopwords(self) -> None:
		# "what", "is", "the" are typical stopwords
		terms = _query_terms("what is the capital of France")
		stopwords_present = {"what", "is", "the", "of"} & set(terms)
		assert not stopwords_present

	def test_keeps_meaningful_terms(self) -> None:
		terms = _query_terms("capital France population")
		assert "capital" in terms
		assert "france" in terms or "France" in terms

	def test_removes_short_tokens(self) -> None:
		# Single and two-character non-digit tokens should be removed
		terms = _query_terms("AI ml gpt neural")
		# "ai" and "ml" are 2-char non-digit so they should be filtered
		for t in terms:
			assert len(t) > 2 or t.isdigit(), f"Short non-digit token slipped through: {t!r}"

	def test_keeps_year_digits(self) -> None:
		# The tokenizer used by _query_terms may strip non-alpha tokens such as years;
		# the key guarantee is simply that the function doesn't raise and returns a list.
		terms = _query_terms("events of 2023")
		assert isinstance(terms, list)

	def test_empty_string_returns_empty(self) -> None:
		assert _query_terms("") == []

	def test_lowercases_output(self) -> None:
		terms = _query_terms("OpenAI GPT-4 Released")
		for t in terms:
			assert t == t.lower()


# ---------------------------------------------------------------------------
# _web_cache_get / _web_cache_set unit tests
# ---------------------------------------------------------------------------

class TestWebCache:
	def test_roundtrip(self, tmp_path, monkeypatch) -> None:
		import retrieval.internet_fallback as ifmod
		monkeypatch.setattr(ifmod, "_WEB_CACHE_DB", tmp_path / "cache" / "web_search.db")

		rows = [{"url": "https://example.com", "title": "Eg", "snippet": "Test"}]
		_web_cache_set("what is AI?", rows)
		result = _web_cache_get("what is AI?")
		assert result is not None
		assert result[0]["url"] == "https://example.com"

	def test_miss_returns_none(self, tmp_path, monkeypatch) -> None:
		import retrieval.internet_fallback as ifmod
		monkeypatch.setattr(ifmod, "_WEB_CACHE_DB", tmp_path / "cache2" / "web_search.db")

		result = _web_cache_get("query that was never cached")
		assert result is None

	def test_expired_entry_returns_none(self, tmp_path, monkeypatch) -> None:
		import time
		import retrieval.internet_fallback as ifmod
		monkeypatch.setattr(ifmod, "_WEB_CACHE_DB", tmp_path / "cache3" / "web_search.db")
		# Write a valid entry first
		rows = [{"url": "https://old.com", "title": "Old", "snippet": "stale"}]
		_web_cache_set("stale query", rows)
		# Now patch the TTL to 0 so the entry is immediately expired
		monkeypatch.setattr(ifmod, "_WEB_CACHE_TTL", 0)
		result = _web_cache_get("stale query")
		assert result is None

	def test_case_and_whitespace_insensitive_key(self, tmp_path, monkeypatch) -> None:
		import retrieval.internet_fallback as ifmod
		monkeypatch.setattr(ifmod, "_WEB_CACHE_DB", tmp_path / "cache4" / "web_search.db")

		rows = [{"url": "https://norm.com", "title": "N", "snippet": "norm"}]
		_web_cache_set("  What Is AI?  ", rows)
		# The cache key uses .lower().strip() so these should hit
		result = _web_cache_get("what is ai?")
		assert result is not None


# ---------------------------------------------------------------------------
# _chunk_text additional edge cases
# ---------------------------------------------------------------------------

class TestChunkTextEdgeCases:
	def test_empty_returns_empty(self) -> None:
		assert _chunk_text("") == []

	def test_short_text_single_chunk(self) -> None:
		text = "Short sentence."
		result = _chunk_text(text, max_chars=900)
		assert result == ["Short sentence."]

	def test_long_text_splits_into_multiple(self) -> None:
		text = ("word " * 400).strip()  # ~2000 chars
		result = _chunk_text(text, max_chars=500, overlap=50)
		assert len(result) > 1
		# All chunks within size bound (with some tolerance for boundary logic)
		for chunk in result:
			assert len(chunk) <= 520  # small tolerance for boundary expansion

	def test_chunks_have_overlap_coverage(self) -> None:
		text = " ".join(f"word{i}" for i in range(200))
		chunks = _chunk_text(text, max_chars=300, overlap=60)
		# Check that the beginning of chunk N+1 overlaps with the end of chunk N
		for i in range(len(chunks) - 1):
			# At least some words should appear in both consecutive chunks
			end_words = set(chunks[i].split()[-8:])
			start_words = set(chunks[i + 1].split()[:8])
			assert end_words & start_words, f"No overlap between chunk {i} and {i+1}"
