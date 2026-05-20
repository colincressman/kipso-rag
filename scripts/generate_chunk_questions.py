"""Generate hypothetical questions for indexed chunks and store their embeddings.

For each chunk in the database, calls the LLM to produce 3 short questions
that the chunk would answer, then embeds those questions and stores them in
the ``chunk_questions`` table.  At query time, a second ANN pass over this
table can surface relevant chunks even when the user's phrasing differs from
the chunk text.

Usage::

    python scripts/generate_chunk_questions.py [--collection-id <id>]
        [--batch-size 20] [--limit 0] [--overwrite]

Options
-------
--collection-id     Restrict to chunks in this collection (default: all)
--batch-size        Chunks to process per DB batch (default: 20)
--limit             Stop after N chunks (0 = unlimited, default)
--overwrite         Re-generate questions even if already present
--dry-run           Log what would happen without writing to DB
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

# Ensure project root is on the path when running from scripts/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from db.client import upsert_chunk_questions, count_chunk_questions, _connect, init_db
from pipeline.embed.embedder import create_embedder
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_EMBED_DIMENSION,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_QUESTIONS_PER_CHUNK = 3

_SYSTEM_PROMPT = (
    "You are a question generator. "
    "Given a passage from a document, write exactly {n} distinct, concise questions "
    "that the passage would answer directly. "
    "Output ONLY a numbered list: '1. <question>\\n2. <question>\\n...' "
    "No preamble, no explanations."
).format(n=_QUESTIONS_PER_CHUNK)


def _generate_questions(
    chunk_text: str,
    *,
    model: str,
    base_url: str,
    timeout_seconds: float = 30.0,
) -> List[str]:
    """Ask the LLM to generate {n} questions for the chunk. Returns list of strings."""
    user_msg = f"Passage:\n{chunk_text[:2000]}"  # cap to avoid huge prompts
    payload = json.dumps({
        "model": model,
        "stream": False,
        "keep_alive": -1,  # keep model warm across batch
        "think": False,
        "options": {"temperature": 0.3, "num_predict": 256},
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    }).encode("utf-8")
    url = f"{base_url.rstrip('/')}/api/chat"
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    text = ((body.get("message") or {}).get("content") or "").strip()

    # Parse numbered list
    questions: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading "1.", "2.", "-", "*" etc.
        for prefix_len in range(4):
            if line[prefix_len:prefix_len + 1] in (".", ")", " ") and line[:prefix_len].isdigit():
                line = line[prefix_len + 1:].strip()
                break
        if line.startswith(("-", "*", "•")):
            line = line[1:].strip()
        if line and len(line) > 10:
            questions.append(line)
    return questions[:_QUESTIONS_PER_CHUNK]


def _fetch_chunks(
    conn: Any,
    *,
    collection_id: Optional[str],
    offset: int,
    batch_size: int,
    overwrite: bool,
) -> List[Dict[str, Any]]:
    """Fetch a batch of chunks, skipping ones that already have questions."""
    if overwrite:
        exists_filter = ""
    else:
        exists_filter = "AND NOT EXISTS (SELECT 1 FROM chunk_questions cq WHERE cq.chunk_id = c.chunk_id)"

    coll_filter = "AND c.collection_id = %(cid)s" if collection_id else ""
    sql = f"""
    SELECT c.chunk_id, c.text
    FROM chunks c
    WHERE c.embedding IS NOT NULL
    {coll_filter}
    {exists_filter}
    ORDER BY c.chunk_id
    LIMIT %(lim)s OFFSET %(off)s
    """
    params: Dict[str, Any] = {"lim": batch_size, "off": offset}
    if collection_id:
        params["cid"] = collection_id
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def run(
    *,
    db_dsn: str = DEFAULT_DB_DSN,
    collection_id: Optional[str] = None,
    batch_size: int = 20,
    limit: int = 0,
    overwrite: bool = False,
    dry_run: bool = False,
    llm_model: str = DEFAULT_LLM_MODEL,
    llm_base_url: str = DEFAULT_LLM_BASE_URL,
) -> None:
    init_db(db_dsn)
    embedder = create_embedder(
        backend=DEFAULT_EMBED_BACKEND,
        model_name=DEFAULT_EMBED_MODEL_NAME,
        dimension=DEFAULT_EMBED_DIMENSION,
        ollama_base_url=DEFAULT_OLLAMA_BASE_URL,
        ollama_timeout_seconds=DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    )

    conn = _connect(db_dsn)
    try:
        offset = 0
        total_processed = 0
        total_skipped = 0

        while True:
            rows = _fetch_chunks(
                conn,
                collection_id=collection_id,
                offset=offset,
                batch_size=batch_size,
                overwrite=overwrite,
            )
            if not rows:
                break

            for row in rows:
                if limit and total_processed >= limit:
                    logger.info("Reached limit of %d chunks.", limit)
                    return

                chunk_id = row["chunk_id"]
                chunk_text = row["text"] or ""
                if not chunk_text.strip():
                    total_skipped += 1
                    continue

                t0 = time.perf_counter()
                try:
                    questions = _generate_questions(
                        chunk_text,
                        model=llm_model,
                        base_url=llm_base_url,
                    )
                except Exception as exc:
                    logger.warning("LLM error for chunk %s: %s", chunk_id, exc)
                    total_skipped += 1
                    continue

                if not questions:
                    total_skipped += 1
                    continue

                # Embed questions
                try:
                    embeddings = embedder.embed_batch(questions)
                except Exception as exc:
                    logger.warning("Embed error for chunk %s: %s", chunk_id, exc)
                    total_skipped += 1
                    continue

                elapsed = time.perf_counter() - t0
                if dry_run:
                    logger.info("[DRY-RUN] chunk=%s  questions=%s  (%.2fs)",
                                chunk_id, questions, elapsed)
                else:
                    upsert_chunk_questions(db_dsn, chunk_id, questions, embeddings)
                    logger.info("chunk=%s  questions=%d  (%.2fs)",
                                chunk_id, len(questions), elapsed)

                total_processed += 1

            offset += len(rows)

    finally:
        conn.close()

    existing = count_chunk_questions(db_dsn) if not dry_run else "N/A"
    logger.info(
        "Done. processed=%d  skipped=%d  total_question_rows=%s",
        total_processed, total_skipped, existing,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection-id", default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--llm-base-url", default=DEFAULT_LLM_BASE_URL)
    parser.add_argument("--db-dsn", default=DEFAULT_DB_DSN)
    args = parser.parse_args()

    run(
        db_dsn=args.db_dsn,
        collection_id=args.collection_id,
        batch_size=args.batch_size,
        limit=args.limit,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
    )


if __name__ == "__main__":
    main()
