"""
Scope and source-type detection helpers extracted from router.py.

This module contains:
  - Explicit source-type detection  (classify_source_type)
  - Temporal / freshness regex patterns  (_STRONG_TEMPORAL_RE, _LIVE_LOOKUP_RE, …)
  - Book-scope detection  (detect_book_scope)
  - Collection-scope detection  (classify_collection_from_query)

All symbols are re-exported by ``retrieval.router`` so callers need not import
this module directly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Source-type routing ────────────────────────────────────────────────────────

_SOURCE_TYPE_NOTES = re.compile(
    r"\b(my notes?|in my notes?|from my notes?|what do(es)? my notes? (say|show|cover)|"
    r"my (markdown|md|text) (files?|notes?)|in my (markdown|md) files?|"
    r"my personal notes?)\b",
    re.IGNORECASE,
)

_SOURCE_TYPE_DOCX = re.compile(
    r"\b(my (word|docx|doc) (files?|documents?)|in my (word|docx|doc) (files?|documents?)|"
    r"from my (word|docx|doc) (files?|documents?)|"
    r"what do(es)? my (word|docx) (files?|documents?) (say|show|cover))\b",
    re.IGNORECASE,
)

_SOURCE_TYPE_PDF = re.compile(
    r"\b(my (pdf|book|books|textbook|textbooks)|in (the|my) (books?|pdfs?|textbooks?)|"
    r"from (the|my) (books?|pdfs?|textbooks?)|"
    r"what do(es)? (the|my) (books?|pdfs?|textbooks?) (say|show|cover))\b",
    re.IGNORECASE,
)

_SOURCE_TYPE_WEB = re.compile(
    r"\b(from the (web|internet|online)|search (the web|online|internet) for|"
    r"look (it )?up online|find online)\b",
    re.IGNORECASE,
)


def classify_source_type(query: str) -> Optional[str]:
    """
    Detect an explicit source-type preference in a query.

    Returns one of ``"notes"``, ``"docx"``, ``"pdf_book"``, ``"internet"``,
    or ``None`` when no preference is expressed.
    """
    q = (query or "").strip()
    if _SOURCE_TYPE_NOTES.search(q):
        return "notes"
    if _SOURCE_TYPE_DOCX.search(q):
        return "docx"
    if _SOURCE_TYPE_PDF.search(q):
        return "pdf_book"
    if _SOURCE_TYPE_WEB.search(q):
        return "internet"
    return None


# ── Temporal / freshness patterns ─────────────────────────────────────────────

_STRONG_TEMPORAL_RE = re.compile(
    r"""
    # Absolute "right now" keywords — no sentence framing makes these timeless.
    \b(today|tonight|yesterday|right\s+now|real[\s\-]?time|live|breaking|nowadays)\b
    |
    # Calendar-scoped recency anchors.  A static corpus (textbooks, documents)
    # never answers "who won X this year?" or "what happened last week?" —
    # those always require live data.  "recently" is included because in the
    # context of a question-answering system backed by static documents,
    # "what has X released recently?" always means "I need current information".
    \b(this\s+year|this\s+season|this\s+week|this\s+month
      |last\s+night|last\s+week|last\s+month
      |just\s+announced|just\s+released|newly\s+(?:announced|released|approved|discovered)
      |recently)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_LIVE_LOOKUP_RE = re.compile(
    r"""
    # "current" followed by a financial rate keyword (with up to 3 interleaved
    # qualifier words, e.g. "current federal funds *target* rate")
    \bcurrent\b(?:\s+\w+){0,4}\s+(?:federal\s+funds?|fed\s+funds?|inflation|prime|mortgage|overnight)(?:\s+\w+){0,3}\s+rate
    |
    # "current yield / current price / current spread" — financial spot values
    # not covered by the rate pattern above.
    \bcurrent\b(?:\s+\w+){0,3}\s+(?:yield|price|spread|quote|bid|ask)\b
    |
    # "latest/newest/most recent [stable] version/release"
    \b(?:latest|newest|most\s+recent)\s+(?:stable\s+)?(?:version|release)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CHAT_OPENER_RE = re.compile(
    r"^i\s+(?:was\s+hoping\s+to|am\s+hoping\s+to|would\s+like\s+to"
    r"|want\s+to|wanted\s+to|'d\s+like\s+to)\s+"
    r"(?:talk|chat|discuss)\s+(?:about|with)\b",
    re.IGNORECASE,
)

_NON_FACTUAL_INTENTS: frozenset = frozenset({
    "conversational",
    "user_profile",
    "summary",
    "conversational_meta",
    "implicit_followup",
})


# ── Book-scope detection ───────────────────────────────────────────────────────

_BOOK_SCOPE_RE = re.compile(
    r"""(?ix)
    ^(?:according\s+to|as\s+(?:described|stated|discussed|explained|defined)\s+in
       |based\s+on\s+(?:the\s+)?)
    \s+
    (.+?)                           # book mention (lazy)
    (?:\s+(?:textbook|book|text))?  # optional trailing word
    \s*[,;]                         # must end at comma or semicolon
    """,
)

_BOOK_SCOPE_ACRONYM_RE = re.compile(
    r"""(?x)
    ^[Ii]n\s+
    ([A-Z]{2,}(?:\s+[A-Z]{2,})*)   # one or more ALL-CAPS acronym words
    \b                              # word boundary — stop before lower-case topic
    """,
)

_BOOK_SCOPE_STOPWORDS: frozenset = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "with", "by",
    "on", "at", "from", "as", "is", "it", "its", "their", "this", "that",
    "which", "what", "how", "who", "s", "introduction", "advanced", "topics",
    "foundations", "concepts", "approach", "modern", "adaptive", "computation",
    "series",
})

_book_scope_cache: Optional[List[Tuple[str, frozenset]]] = None
_book_scope_cache_dsn: Optional[str] = None


def _camel_split(text: str) -> List[str]:
    """Split CamelCase into words: 'MachineLearning' → ['Machine', 'Learning']."""
    return re.findall(r"[A-Z][a-z]+|[A-Z]+(?=[A-Z][a-z])|[A-Z]+$|[a-z]+", text)


def _title_tokens(raw: str) -> frozenset:
    """Return a frozenset of distinctive lowercase tokens from a document name."""
    camel_parts = _camel_split(raw)
    plain_parts = re.findall(r"[a-zA-Z]+", raw)
    all_words = [w.lower() for w in camel_parts + plain_parts if len(w) > 1]
    return frozenset(w for w in all_words if w not in _BOOK_SCOPE_STOPWORDS)


def _load_book_scope_cache(db_dsn: str) -> List[Tuple[str, frozenset]]:
    """Load (doc_id, title_token_set) for every document from the DB (cached)."""
    global _book_scope_cache, _book_scope_cache_dsn
    if _book_scope_cache is not None and _book_scope_cache_dsn == db_dsn:
        return _book_scope_cache
    try:
        import psycopg as _psycopg  # noqa: PLC0415
        from psycopg.rows import dict_row as _dict_row  # noqa: PLC0415
        conn = _psycopg.connect(db_dsn, row_factory=_dict_row)
        rows = conn.execute(
            "SELECT doc_id, filename FROM documents"
        ).fetchall()
        conn.close()
        cache: List[Tuple[str, frozenset]] = []
        for r in rows:
            doc_id = str(r["doc_id"])
            name = str(r["filename"] or "")
            stem = Path(name).stem
            for sfx in (" - libgen.li", " - libgen.is", " - libgen.rs"):
                if stem.endswith(sfx):
                    stem = stem[: -len(sfx)]
            cache.append((doc_id, _title_tokens(stem)))
        _book_scope_cache = cache
        _book_scope_cache_dsn = db_dsn
    except Exception:
        _book_scope_cache = []
    return _book_scope_cache or []


def detect_book_scope(
    query: str,
    db_dsn: Optional[str],
) -> Tuple[Optional[List[str]], Optional[str]]:
    """Detect 'According to [Book],' scoping and return (doc_ids, rewritten_query).

    Returns ``(None, None)`` when no book scope is found or the DB is not
    available.
    """
    if not db_dsn or not query:
        return None, None

    q = query.strip()
    is_acronym_scope = False
    m = _BOOK_SCOPE_RE.match(q)
    if not m:
        m = _BOOK_SCOPE_ACRONYM_RE.match(q)
        if m:
            is_acronym_scope = True
        else:
            return None, None

    mention = m.group(1).strip()
    mention_clean = re.sub(r"'s\b", "", mention)
    mention_tokens = frozenset(
        w.lower() for w in re.findall(r"[a-zA-Z]+", mention_clean)
        if len(w) > 1 and w.lower() not in _BOOK_SCOPE_STOPWORDS
    )
    mention_is_single_proper = (
        len(mention_tokens) == 1
        and bool(re.fullmatch(r"[A-Z][a-zA-Z0-9]*", mention_clean.strip()))
    )
    min_tokens = 1 if (is_acronym_scope or mention_is_single_proper) else 2
    if len(mention_tokens) < min_tokens:
        return None, None

    cache = _load_book_scope_cache(db_dsn)
    if not cache:
        return None, None

    scored: List[Tuple[float, str]] = []
    for doc_id, title_toks in cache:
        if not title_toks:
            continue
        overlap = len(mention_tokens & title_toks)
        score = overlap / len(mention_tokens)
        if score >= 0.5:
            scored.append((score, doc_id))

    if not scored:
        return None, None

    best_score = max(s for s, _ in scored)
    matched_ids = [doc_id for score, doc_id in scored if score >= best_score - 0.10]

    rewritten = q[m.end():].strip()
    if rewritten:
        rewritten = rewritten[0].upper() + rewritten[1:]
    rewritten = rewritten or q

    return matched_ids, rewritten


# ── Collection-scope detection ─────────────────────────────────────────────────

_COLL_PAT_TRAILING = r"[,:\s]*"
_COLL_PAT_PREFIX   = r"(?:(?:in|from)\s+(?:the|my|our|this)?\s*)"
_coll_pat_cache: Dict[str, Tuple[re.Pattern, re.Pattern]] = {}

_SCOPE_PREFIX = re.compile(
    r"(?:(?:in|from)\s+(?:the|my|our|this)?\s*)",
    re.IGNORECASE,
)


def classify_collection_from_query(
    query: str,
    db_path: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect an explicit collection-scope phrase in a natural-language query.

    Returns ``(collection_id, rewritten_query)`` or ``(None, None)``.
    """
    if not db_path or not query:
        return None, None

    try:
        from db.client import list_collections as _list_collections  # noqa: PLC0415
        collections = _list_collections(db_path)
    except Exception:
        return None, None

    if not collections:
        return None, None

    q = query.strip()

    candidates: List[Tuple[str, str]] = []
    for col in collections:
        cid: str = col["collection_id"]
        name: str = (col.get("name") or "").strip()
        candidates.append((cid, cid))
        if name and name.lower() != cid.lower():
            candidates.append((cid, name))
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    for collection_id, label in candidates:
        esc = re.escape(label)

        if label not in _coll_pat_cache:
            _coll_pat_cache[label] = (
                re.compile(_COLL_PAT_PREFIX + esc + _COLL_PAT_TRAILING, re.IGNORECASE),
                re.compile(r"^" + esc + r"[,:\s]+", re.IGNORECASE),
            )
        pat_a, pat_b = _coll_pat_cache[label]

        m = pat_a.search(q)
        if m:
            stripped = (q[: m.start()] + q[m.end() :]).strip().lstrip(",: ")
            if stripped:
                stripped = stripped[0].upper() + stripped[1:]
            return collection_id, stripped or q

        m = pat_b.match(q)
        if m:
            stripped = q[m.end() :].strip()
            if stripped:
                stripped = stripped[0].upper() + stripped[1:]
            return collection_id, stripped or q

    return None, None
