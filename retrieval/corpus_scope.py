"""
Corpus scope manifest — tracks what topics are covered by ingested documents.

The manifest is a small JSON file that records:
  - A flat set of topic keywords (deduplicated, lowercased)
  - The document titles and their per-document top keywords
  - Counts used for condensing when the manifest grows large

This lets plan.py do a zero-cost relevance check before deciding whether
to route a fact_lookup query to RAG or to the web.

Lifecycle
---------
- Built from scratch by ``rebuild()`` — scans the DB for all chunks.
- Updated incrementally by ``add_document()`` — called after ingest.
- Auto-condensed when topic count exceeds MAX_TOPICS: rare/low-count topics
  are dropped so the manifest stays small and fast.
- Queried by ``scope_score(query)`` — returns 0.0–1.0 overlap fraction.
  A score near zero means the corpus almost certainly cannot answer the query.
"""

from __future__ import annotations

import json
import logging
import math
import re
import psycopg
from psycopg.rows import dict_row
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MANIFEST_PATH = Path(__file__).parent.parent / "data" / "corpus_manifest.json"

# When topic count exceeds this, drop the least-frequent topics on the next save.
MAX_TOPICS = 600

# Minimum document-frequency for a topic to survive condensing.
MIN_DOC_FREQ = 2

# Top-N keywords extracted per document during manifest build/update.
KEYWORDS_PER_DOC = 40

# Threshold below which a fact_lookup is considered off-topic for the corpus.
# Requires ~20% of meaningful query tokens to match a corpus topic.
# A single incidental word match (e.g. "tree" in a biology corpus) scores
# ~0.11 for a 9-token query and is correctly rejected.
SCOPE_THRESHOLD = 0.20   # callers may override

# Stop-words to exclude from topic extraction (very common English + math).
_STOPWORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "up", "about", "into", "through", "during",
    "this", "that", "these", "those", "it", "its", "we", "you", "they",
    "he", "she", "i", "my", "your", "their", "our", "which", "who", "what",
    "how", "when", "where", "why", "not", "no", "so", "as", "than", "then",
    "each", "all", "any", "both", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "also", "very", "just", "s", "t",
    # common in academic/ML text but not useful as scope markers
    "figure", "table", "chapter", "section", "page", "appendix",
    "however", "therefore", "thus", "hence", "since", "while",
    "let", "note", "see", "use", "using", "used", "given",
    "can", "also", "well", "one", "two", "three", "first", "second",
    "new", "different", "important", "many", "often", "between",
    # generic English words that appear in any text — not domain signals
    "there", "over", "because", "like", "before", "less", "across",
    "known", "called", "them", "take", "down", "work", "build", "shown",
    "follows", "based", "form", "true", "number", "case", "line",
    "order", "small", "large", "high", "step", "output", "input",
    "approach", "consider", "discuss", "follows", "example", "examples",
    "book", "course", "document", "present", "strong", "early",
    "instead", "because", "multiple", "across", "better", "forward",
    "increases", "decreases", "because", "often", "research",
}

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_\-]*[a-zA-Z0-9]|[a-zA-Z]{3,}")


# ── Manifest data class ────────────────────────────────────────────────────────

class CorpusManifest:
    """
    In-memory representation of the corpus topic manifest.

    Attributes
    ----------
    topics          : {word: doc_frequency}  — flat topic index
    documents       : {doc_id: {"title": str, "keywords": [str], "source_type": str}}
    last_updated    : ISO-8601 timestamp of last save
    doc_count       : total number of documents in the manifest
    """

    def __init__(self) -> None:
        self.topics: Counter[str] = Counter()
        self.documents: Dict[str, Dict] = {}
        self.last_updated: str = ""
        self.doc_count: int = 0

    # ── Serialisation ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "topics": dict(self.topics.most_common()),
            "documents": self.documents,
            "last_updated": self.last_updated,
            "doc_count": self.doc_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CorpusManifest":
        m = cls()
        m.topics = Counter(d.get("topics", {}))
        m.documents = d.get("documents", {})
        m.last_updated = d.get("last_updated", "")
        m.doc_count = d.get("doc_count", len(m.documents))
        return m

    def save(self, path: Path = MANIFEST_PATH) -> None:
        self._maybe_condense()
        self.last_updated = datetime.now(timezone.utc).isoformat()
        self.doc_count = len(self.documents)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("corpus_scope: manifest saved (%d topics, %d docs) → %s",
                    len(self.topics), self.doc_count, path)

    @classmethod
    def load(cls, path: Path = MANIFEST_PATH) -> Optional["CorpusManifest"]:
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("corpus_scope: failed to load manifest: %s", exc)
            return None

    # ── Condensing ─────────────────────────────────────────────────────────────

    def _maybe_condense(self) -> None:
        """Drop the rarest topics when the manifest grows too large."""
        if len(self.topics) <= MAX_TOPICS:
            return
        before = len(self.topics)
        # Keep topics with doc_frequency >= MIN_DOC_FREQ, then trim to MAX_TOPICS
        survivors = {k: v for k, v in self.topics.items() if v >= MIN_DOC_FREQ}
        if len(survivors) > MAX_TOPICS:
            # Further trim: keep the MAX_TOPICS most frequent
            survivors = dict(Counter(survivors).most_common(MAX_TOPICS))
        self.topics = Counter(survivors)
        logger.info("corpus_scope: condensed manifest %d → %d topics", before, len(self.topics))

    # ── Document add/remove ────────────────────────────────────────────────────

    def add_document(self, doc_id: str, title: str, keywords: List[str],
                     source_type: str = "pdf_book") -> None:
        """Add or replace a document entry and update the topic index."""
        if doc_id in self.documents:
            # Remove old keyword contributions first
            old_kws = self.documents[doc_id].get("keywords", [])
            for kw in old_kws:
                if kw in self.topics:
                    self.topics[kw] -= 1
                    if self.topics[kw] <= 0:
                        del self.topics[kw]

        self.documents[doc_id] = {
            "title": title,
            "keywords": keywords,
            "source_type": source_type,
        }
        for kw in keywords:
            self.topics[kw] += 1

    def remove_document(self, doc_id: str) -> None:
        """Remove a document and subtract its keyword contributions."""
        entry = self.documents.pop(doc_id, None)
        if entry:
            for kw in entry.get("keywords", []):
                if kw in self.topics:
                    self.topics[kw] -= 1
                    if self.topics[kw] <= 0:
                        del self.topics[kw]

    # ── Scope query ────────────────────────────────────────────────────────────

    def scope_score(self, query: str) -> Tuple[float, List[str]]:
        """
        Return (score, matched_topics) where score is the fraction of
        meaningful query tokens that appear in the corpus topic index.

        score == 0.0  → no overlap at all — corpus almost certainly can't answer
        score == 1.0  → every query token is a known corpus topic
        """
        tokens = _extract_keywords(query, top_n=None)
        if not tokens:
            return 0.0, []
        matched = [t for t in tokens if t in self.topics]
        score = len(matched) / len(tokens)
        return score, matched

    def topic_summary(self) -> str:
        """Return a short human-readable summary of the corpus scope."""
        if not self.documents:
            return "No documents ingested."
        titles = [v["title"] for v in self.documents.values() if v.get("title")]
        top_topics = [k for k, _ in self.topics.most_common(20)]
        parts = []
        if titles:
            parts.append(f"{len(titles)} document(s): " + "; ".join(titles[:8])
                         + ("…" if len(titles) > 8 else ""))
        if top_topics:
            parts.append("Top topics: " + ", ".join(top_topics))
        return " | ".join(parts)


# ── Keyword extraction ─────────────────────────────────────────────────────────

def _extract_keywords(text: str, top_n: Optional[int] = KEYWORDS_PER_DOC) -> List[str]:
    """
    Extract meaningful keywords from text using simple TF weighting.
    Returns lowercased tokens, filtered for stop-words and short tokens.
    """
    tokens = _TOKEN_RE.findall(text.lower())
    counts: Counter[str] = Counter()
    for tok in tokens:
        if len(tok) >= 4 and tok not in _STOPWORDS:
            counts[tok] += 1
    if top_n is None:
        return list(counts.keys())
    return [kw for kw, _ in counts.most_common(top_n)]


def extract_doc_keywords(chunks_text: List[str], top_n: int = KEYWORDS_PER_DOC) -> List[str]:
    """Extract keywords from a list of chunk texts for a single document."""
    combined = " ".join(chunks_text)
    return _extract_keywords(combined, top_n=top_n)


# ── DB-level rebuild ───────────────────────────────────────────────────────────

def rebuild_manifest(db_dsn: str, save_path: Path = MANIFEST_PATH) -> CorpusManifest:
    """
    Build a fresh manifest by scanning all chunks in the DB.
    Called once at startup or via the rebuild script.
    """
    manifest = CorpusManifest()
    try:
        con = psycopg.connect(db_dsn, row_factory=dict_row)
        cur = con.cursor()

        # Load document metadata
        docs = cur.execute(
            "SELECT doc_id, filename, metadata_json, source_type FROM documents"
        ).fetchall()

        for doc in docs:
            doc_id = doc["doc_id"]
            source_type = doc["source_type"] or "pdf_book"
            try:
                meta = json.loads(doc["metadata_json"] or "{}")
            except Exception:
                meta = {}
            title = meta.get("document_title") or meta.get("title") or doc["filename"]

            # Fetch chunk texts for this document
            rows = cur.execute(
                "SELECT text FROM chunks WHERE doc_id = %s AND structural_role NOT IN ('metadata', 'index_noise')",
                (doc_id,),
            ).fetchall()
            chunk_texts = [r["text"] for r in rows if r["text"]]
            keywords = extract_doc_keywords(chunk_texts)
            manifest.add_document(doc_id, title, keywords, source_type)

        con.close()
    except Exception as exc:
        logger.error("corpus_scope: rebuild failed: %s", exc)

    manifest.save(save_path)
    return manifest


def update_manifest_for_doc(
    db_dsn: str,
    doc_id: str,
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """
    Called after a single document is ingested.  Loads the manifest,
    updates the one document entry, and saves.  Thread-safe via a module lock.
    """
    with _update_lock:
        manifest = CorpusManifest.load(manifest_path) or CorpusManifest()
        try:
            con = psycopg.connect(db_dsn, row_factory=dict_row)
            cur = con.cursor()
            doc_row = cur.execute(
                "SELECT filename, metadata_json, source_type FROM documents WHERE doc_id = %s",
                (doc_id,),
            ).fetchone()
            if not doc_row:
                con.close()
                return
            source_type = doc_row["source_type"] or "pdf_book"
            try:
                meta = json.loads(doc_row["metadata_json"] or "{}")
            except Exception:
                meta = {}
            title = meta.get("document_title") or meta.get("title") or doc_row["filename"]
            rows = cur.execute(
                "SELECT text FROM chunks WHERE doc_id = %s AND structural_role NOT IN ('metadata', 'index_noise')",
                (doc_id,),
            ).fetchall()
            chunk_texts = [r["text"] for r in rows if r["text"]]
            con.close()
            keywords = extract_doc_keywords(chunk_texts)
            manifest.add_document(doc_id, title, keywords, source_type)
        except Exception as exc:
            logger.error("corpus_scope: update failed for doc %s: %s", doc_id, exc)
            return
        manifest.save(manifest_path)


_update_lock = threading.Lock()


# ── Singleton cached manifest ──────────────────────────────────────────────────

_cached_manifest: Optional[CorpusManifest] = None
_cache_lock = threading.Lock()


def get_manifest(manifest_path: Path = MANIFEST_PATH) -> Optional[CorpusManifest]:
    """Return the cached manifest, loading from disk on first call."""
    global _cached_manifest
    if _cached_manifest is not None:
        return _cached_manifest
    with _cache_lock:
        if _cached_manifest is None:
            _cached_manifest = CorpusManifest.load(manifest_path)
    return _cached_manifest


def invalidate_cache() -> None:
    """Force the next call to get_manifest() to reload from disk."""
    global _cached_manifest
    with _cache_lock:
        _cached_manifest = None


# ── Convenience: is query in scope? ───────────────────────────────────────────

def is_in_scope(query: str, threshold: float = SCOPE_THRESHOLD) -> Tuple[bool, float, List[str]]:
    """
    Returns (in_scope, score, matched_topics).

    ``in_scope`` is True when at least one meaningful query token matches
    a known corpus topic — i.e., the corpus *might* contain an answer.
    ``in_scope`` is False (score == 0.0) only when there is zero lexical
    overlap, which is the safe trigger for redirecting to web.
    """
    manifest = get_manifest()
    if manifest is None or not manifest.topics:
        # No manifest yet → assume everything is in scope (safe default)
        return True, 1.0, []
    score, matched = manifest.scope_score(query)
    return score > threshold, score, matched
