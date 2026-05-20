"""BM25 index management for the retrieval layer.

Builds, persists, and caches a BM25Okapi index over corpus chunks.
The index is keyed on a cheap DB fingerprint (row count + last chunk_id)
and stored to disk so it survives process restarts without a rebuild.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg

from retrieval.acronym_expand import _tokenize_words
from retrieval.db_search import _connect, _text_rows
from retrieval.models import RetrievalFilters
from utils.runtime_defaults import DEFAULT_DB_DSN

_log = logging.getLogger(__name__)

# In-process LRU: maps cache_key -> {"bm25": BM25Okapi, "chunk_ids": [...]}
# Invalidated when the DB's chunk count or last chunk_id changes.
_bm25_index_cache: Dict[str, Any] = {}

# Set by warm_bm25_index() once the index is loaded.  Queries that arrive
# before the warmup thread finishes will still work (they trigger an inline
# build) but a warning is emitted so slow first-queries are observable.
_bm25_ready = threading.Event()

# Set at the very start of warm_bm25_index() so we can distinguish
# "warmup started but not finished" (warn) from "warmup never called" (silent).
_bm25_warmup_started = threading.Event()

# Prevents the warmup thread and an early query thread from building the
# index simultaneously (benign data race, but wasteful).
_bm25_build_lock = threading.Lock()

_BM25_INDEX_DIR = Path(__file__).resolve().parents[1] / "data" / "index"


def _bm25_cache_key(db_dsn: str, conn: psycopg.Connection) -> str:
	"""Cheap fingerprint: row count + last chunk_id (no full table scan needed)."""
	row = conn.execute(
		"SELECT COUNT(*) AS n, MAX(chunk_id) AS last_id FROM chunks"
	).fetchone()
	n = row["n"] if row else 0
	last = (row["last_id"] or "") if row else ""
	raw = f"{db_dsn}:{n}:{last}"
	return hashlib.md5(raw.encode()).hexdigest()


def _load_or_build_bm25(
	rows: List[dict],
	db_dsn: str,
	conn: psycopg.Connection,
) -> tuple:
	"""Return (chunk_ids, BM25Okapi) from cache or build + persist."""
	try:
		from rank_bm25 import BM25Okapi  # type: ignore
	except ImportError:
		return [r["chunk_id"] for r in rows], None

	cache_key = _bm25_cache_key(db_dsn, conn)

	# 1. In-process cache hit (fastest path — same process, same corpus state)
	if cache_key in _bm25_index_cache:
		entry = _bm25_index_cache[cache_key]
		return entry["chunk_ids"], entry["bm25"]

	# 2. On-disk cache hit
	_BM25_INDEX_DIR.mkdir(parents=True, exist_ok=True)
	pkl_path = _BM25_INDEX_DIR / f"bm25_{cache_key}.pkl"
	if pkl_path.exists():
		try:
			with pkl_path.open("rb") as fh:
				entry = pickle.load(fh)
			_bm25_index_cache[cache_key] = entry
			return entry["chunk_ids"], entry["bm25"]
		except Exception:
			pkl_path.unlink(missing_ok=True)  # corrupt file — rebuild

	# 3. Build from rows — hold the lock for the entire build so a concurrent
	#    query/warmup thread does not duplicate the work.
	with _bm25_build_lock:
		# Re-check cache after acquiring lock — another thread may have just built it.
		if cache_key in _bm25_index_cache:
			entry = _bm25_index_cache[cache_key]
			return entry["chunk_ids"], entry["bm25"]

		chunk_ids: List[str] = []
		corpus: List[List[str]] = []
		for r in rows:
			# Repeat title 5× so title-match chunks aren't swamped by long prose chapters.
			title_boost = " ".join([str(r["title"] or "")] * 5)
			blob = " ".join(filter(None, [
				str(r["text"] or ""),
				title_boost,
				str(r["path_text"] or ""),
			]))
			corpus.append(_tokenize_words(blob.lower()))
			chunk_ids.append(r["chunk_id"])

		bm25 = BM25Okapi(corpus)
		entry = {"chunk_ids": chunk_ids, "bm25": bm25}

		# Persist to disk (best-effort — never block retrieval)
		try:
			with pkl_path.open("wb") as fh:
				pickle.dump(entry, fh, protocol=pickle.HIGHEST_PROTOCOL)
			# Evict stale pkl files for this db_dsn (different cache keys)
			for old in _BM25_INDEX_DIR.glob("bm25_*.pkl"):
				if old != pkl_path:
					try:
						old.unlink()
					except Exception:
						pass
		except Exception:
			pass

		_bm25_index_cache[cache_key] = entry
		return chunk_ids, bm25


def _bm25_scores(
	rows: List[dict],
	query_tokens: List[str],
	db_dsn: str = "",
	conn: Optional[psycopg.Connection] = None,
) -> Tuple[List[str], List[float]]:
	"""
	Compute BM25Okapi scores for all candidate rows.

	Uses a persistent on-disk index (data/index/bm25_*.pkl) that is rebuilt
	only when the corpus changes.  Falls back gracefully to zero scores if
	``rank_bm25`` is not installed.
	"""
	if not rows or not query_tokens:
		return [r["chunk_id"] for r in rows], [0.0] * len(rows)

	if _bm25_warmup_started.is_set() and not _bm25_ready.is_set():
		_log.warning(
			"BM25 index not yet ready — warmup thread still running. "
			"This query will build the index inline (may be slow)."
		)

	if db_dsn and conn is not None:
		chunk_ids, bm25 = _load_or_build_bm25(rows, db_dsn, conn)
	else:
		# Fallback: build in-memory (old path — used when called without conn)
		try:
			from rank_bm25 import BM25Okapi  # type: ignore
		except ImportError:
			return [r["chunk_id"] for r in rows], [0.0] * len(rows)
		chunk_ids = []
		corpus: List[List[str]] = []
		for r in rows:
			title_boost = " ".join([str(r["title"] or "")] * 5)
			blob = " ".join(filter(None, [
				str(r["text"] or ""),
				title_boost,
				str(r["path_text"] or ""),
			]))
			corpus.append(_tokenize_words(blob.lower()))
			chunk_ids.append(r["chunk_id"])
		bm25 = BM25Okapi(corpus)

	if bm25 is None:
		return chunk_ids, [0.0] * len(chunk_ids)

	scores: List[float] = bm25.get_scores(query_tokens).tolist()
	return chunk_ids, scores


def warm_bm25_index(db_dsn: str = DEFAULT_DB_DSN) -> int:
	"""
	Pre-load (or build) the BM25 index for *db_dsn* into the in-process cache.

	Call at server startup in a background thread so the first query does not
	pay the BM25 build cost.  Returns the number of chunks indexed (0 on error
	or when BM25 is disabled / rank_bm25 not installed).
	"""
	try:
		_bm25_warmup_started.set()
		conn = _connect(db_dsn)
		try:
			rows = _text_rows(conn, RetrievalFilters())
			if not rows:
				return 0
			chunk_ids, _bm25 = _load_or_build_bm25(rows, db_dsn, conn)
			n = len(chunk_ids)
			_bm25_ready.set()
			_log.info("BM25 index warmed: %d chunks", n)
			return n
		finally:
			conn.close()
	except Exception as exc:
		_log.warning("warm_bm25_index: %s", exc)
		return 0
