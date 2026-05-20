"""Detect, clean, and re-embed noisy chunk text rows in PostgreSQL.

Targets table-conversion artifacts such as long repeated pipe runs (e.g., "||||||||").
Only affected/changed chunks are updated; all metadata columns are preserved.

Usage:
    .venv/Scripts/python.exe scripts/repair_noisy_chunks.py --dry-run
    .venv/Scripts/python.exe scripts/repair_noisy_chunks.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import psycopg
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.chunk.strategies import estimate_tokens
from pipeline.embed.embedder import create_embedder
from utils.config import load_yaml_config
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_DIMENSION,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
)


PIPE_RUN_RE = re.compile(r"\|{3,}")
PIPE_HEAVY_LINE_RE = re.compile(r"^[\s\|\-:\+]+$")
WHITESPACE_RE = re.compile(r"[ \t]+")
# Markdown table separators: | --- | or | :--- | or | ---: | etc.
TABLE_SEP_RE = re.compile(r"\|\s*[-:]{3,}\s*\|")
# 4+ consecutive bare pipes on the same line (no newlines): | | | |
# Uses ' *' not '\s*' so it doesn't match newline-separated table row boundaries.
BARE_PIPES_RE = re.compile(r"( *\| *){4,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair noisy chunk text and re-embed affected rows")
    parser.add_argument("--db", default=DEFAULT_DB_DSN, help="PostgreSQL DSN")
    parser.add_argument(
        "--embedding-config",
        default="configs/runtime.yaml",
        help="Runtime config YAML (embedding section)",
    )
    parser.add_argument(
        "--audit-log",
        default=None,
        help="Optional audit JSONL output path (default: data/diagnostics/noise_repair_*.jsonl)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on number of flagged chunks to process",
    )
    parser.add_argument("--dry-run", action="store_true", help="Detect/log only; do not update DB")
    return parser.parse_args()


def is_noisy_text(text: str) -> bool:
    if not text:
        return False
    if PIPE_RUN_RE.search(text):
        return True
    # Corrupted table: separator lines combined with multiple empty cells
    # (e.g. "| | | | | --- | --- | | | |" from website-scraped PDFs).
    # Require BOTH patterns together to avoid flagging legitimate data tables.
    if TABLE_SEP_RE.search(text) and BARE_PIPES_RE.search(text):
        return True
    lines = text.splitlines()
    if not lines:
        return False
    pipe_heavy = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        pipe_count = stripped.count("|")
        if pipe_count >= 6 and pipe_count / max(1, len(stripped)) >= 0.4:
            pipe_heavy += 1
    return pipe_heavy >= 2


def clean_noisy_text(text: str) -> str:
    """Safely clean pipe/table noise while preserving readable prose."""
    if not text:
        return text

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line_out = line

        # Replace extreme pipe runs with a single space so surrounding prose remains.
        line_out = PIPE_RUN_RE.sub(" ", line_out)

        # Drop markdown table separator-only lines like: | --- | --- |
        candidate = line_out.strip()
        if candidate and "|" in candidate and PIPE_HEAVY_LINE_RE.fullmatch(candidate):
            continue

        # Normalize dense inline whitespace introduced by cleanup.
        line_out = WHITESPACE_RE.sub(" ", line_out).strip()
        cleaned_lines.append(line_out)

    cleaned = "\n".join(cleaned_lines)

    # Remove inline table separator sequences: | --- | --- | (missed by line filter
    # when embedded mid-line in prose).
    cleaned = TABLE_SEP_RE.sub(" ", cleaned)

    # Remove runs of 4+ bare consecutive pipes: | | | | (table row boundaries).
    cleaned = BARE_PIPES_RE.sub(" ", cleaned)

    # Collapse excessive blank lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def embedding_hash(vector_json: str | None) -> str | None:
    if not vector_json:
        return None
    try:
        payload = json.loads(vector_json)
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    digest = hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest[:16]


def load_embedding_settings(path: str) -> dict[str, Any]:
    cfg = load_yaml_config(path, default={})
    emb = cfg.get("embedding", {}) if isinstance(cfg, dict) else {}
    if not isinstance(emb, dict):
        emb = {}
    return {
        "backend": emb.get("backend", DEFAULT_EMBED_BACKEND),
        "dimension": int(emb.get("dimension", DEFAULT_EMBED_DIMENSION)),
        "model_name": emb.get("model_name", DEFAULT_EMBED_MODEL_NAME),
        "ollama_base_url": emb.get("ollama_base_url", DEFAULT_OLLAMA_BASE_URL),
        "ollama_timeout_seconds": float(emb.get("ollama_timeout_seconds", DEFAULT_OLLAMA_TIMEOUT_SECONDS)),
    }


def default_audit_path() -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("data/diagnostics") / f"noise_repair_{ts}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> int:
    args = parse_args()
    db_dsn = args.db

    audit_path = Path(args.audit_log) if args.audit_log else default_audit_path()
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    conn = psycopg.connect(db_dsn, row_factory=dict_row)

    try:
        rows = conn.execute(
            """
            SELECT chunk_id, doc_id, page_start, page_end, structural_role, text
            FROM chunks
            """
        ).fetchall()

        flagged: list[dict] = []
        for row in rows:
            text = str(row["text"] or "")
            if is_noisy_text(text):
                flagged.append(row)

        if args.max_chunks is not None:
            flagged = flagged[: max(0, int(args.max_chunks))]

        print(f"[repair-noise] scanned={len(rows)} flagged={len(flagged)}")

        # Always log flagged chunk IDs so review is possible even in dry-run.
        with open(audit_path, "w", encoding="utf-8") as logf:
            for row in flagged:
                original_text = str(row["text"] or "")
                cleaned_text = clean_noisy_text(original_text)
                changed = cleaned_text != original_text
                log_entry = {
                    "chunk_id": row["chunk_id"],
                    "doc_id": row["doc_id"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "structural_role": row["structural_role"],
                    "flagged": True,
                    "changed_text": changed,
                    "old_text_len": len(original_text),
                    "new_text_len": len(cleaned_text),
                    "old_text_preview": original_text[:240],
                    "new_text_preview": cleaned_text[:240],
                    "note": "embedding will be re-generated",
                }
                logf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        print(f"[repair-noise] audit log written: {audit_path}")

        candidates = []
        for row in flagged:
            original_text = str(row["text"] or "")
            cleaned_text = clean_noisy_text(original_text)
            if cleaned_text != original_text and cleaned_text.strip():
                candidates.append((row, cleaned_text))

        print(f"[repair-noise] text_changed={len(candidates)}")

        if args.dry_run:
            print("[repair-noise] dry-run enabled; no DB updates")
            return 0

        if not candidates:
            print("[repair-noise] no changes to apply")
            return 0

        emb_cfg = load_embedding_settings(args.embedding_config)
        embedder = create_embedder(
            backend=str(emb_cfg["backend"]),
            dimension=int(emb_cfg["dimension"]),
            model_name=str(emb_cfg["model_name"]),
            ollama_base_url=str(emb_cfg["ollama_base_url"]),
            ollama_timeout_seconds=float(emb_cfg["ollama_timeout_seconds"]),
        )

        texts = [cleaned_text for _, cleaned_text in candidates]
        vectors = embedder.embed_texts(texts)

        if len(vectors) != len(candidates):
            raise RuntimeError(
                f"Embedder returned {len(vectors)} vectors for {len(candidates)} updated chunks"
            )

        updates = []
        post_logs = []

        for (row, cleaned_text), vec in zip(candidates, vectors):
            updates.append(
                (
                    cleaned_text,
                    int(estimate_tokens(cleaned_text)),
                    vec,
                    row["chunk_id"],
                )
            )
            post_logs.append(
                {
                    "chunk_id": row["chunk_id"],
                    "new_embedding_dim": len(vec) if isinstance(vec, list) else None,
                }
            )

        conn.executemany(
            """
            UPDATE chunks
            SET text = %s, token_count_est = %s, embedding = %s
            WHERE chunk_id = %s
            """,
            updates,
        )
        conn.commit()

        # Append post-update embedding hashes.
        with open(audit_path, "a", encoding="utf-8") as logf:
            for entry in post_logs:
                logf.write(json.dumps({"post_update": entry}, ensure_ascii=False) + "\n")

        print(f"[repair-noise] updated_chunks={len(updates)}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
