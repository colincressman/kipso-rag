"""
pipeline/ingest_v3.py  —  Unified Ingest v3
============================================

Single entry-point for ingesting any supported file type into the RAG system.

Key design rules
----------------
* All pipeline stages always run — no stage is silently skipped because a
  directory was not passed in.  Default output dirs are resolved relative to
  the project root automatically.
* One public function: ``ingest_file(path, ...)``
* Handles: PDF (Marker extraction), DOCX, plain text, code, markdown.
* Writes: markdown → structured JSON → chunks JSON → embeddings → PostgreSQL.
* On completion: refreshes book registry + corpus manifest.
* Thread-safe: safe to call from a FastAPI background thread.

Usage
-----
::

    from pipeline.ingest_v3 import ingest_file

    result = ingest_file("data/raw/my_doc.pdf", db_dsn="postgresql://...")
    # {"kind": "pdf", "filename": "my_doc.pdf", "doc_id": "abc123", "chunks": 47}
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_DIMENSION,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_TIMEOUT_SECONDS,
    DEFAULT_BOOK_REGISTRY_PATH,
)

logger = logging.getLogger(__name__)

# ── Project root resolution ───────────────────────────────────────────────────

def _project_root() -> Path:
    """Return the project root regardless of whether running as EXE or script."""
    try:
        from utils.config import get_install_dir  # type: ignore
        return Path(get_install_dir())
    except Exception:
        return Path(__file__).resolve().parents[1]


def _data_dir(*parts: str) -> Path:
    p = _project_root() / "data" / Path(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── Constants ─────────────────────────────────────────────────────────────────

_TEXT_EXTENSIONS: frozenset[str] = frozenset({
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".cs", ".go", ".rb", ".php",
    ".sh", ".bash", ".zsh", ".yaml", ".yml", ".json", ".toml", ".ini",
    ".cfg", ".csv", ".html", ".htm", ".xml", ".css", ".scss", ".sql",
    ".r", ".m", ".tex", ".bib", ".docx",
})

_ILLEGAL_PATH_CHARS_RE = re.compile(r'[<>:"/\\|?*]+')


# ── Utilities ─────────────────────────────────────────────────────────────────

def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_stem(doc_id: str, filename: str, *, max_name_len: int = 72) -> str:
    base = Path(filename).stem
    base = _ILLEGAL_PATH_CHARS_RE.sub("_", base)
    base = re.sub(r"\s+", " ", base).strip(" .")
    if not base:
        base = "document"
    return f"{doc_id[:12]}_{base[:max_name_len]}"


def _source_type_for_ext(ext: str) -> str:
    code_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".cpp",
                 ".h", ".hpp", ".cs", ".go", ".rb", ".php", ".sh", ".bash",
                 ".zsh", ".sql", ".r", ".m"}
    if ext in code_exts:
        return "code"
    if ext in {".md", ".rst"}:
        return "notes"
    if ext == ".docx":
        return "docx"
    return "text"


# ── Text extraction  (re-exported from pipeline.extract.text_extractor) ─────

from pipeline.extract.text_extractor import (  # noqa: E402, F401
    _extract_docx_text,
    _read_text_file,
    _pdf_metadata,
)


def _unload_embed_model(model_name: str, base_url: str) -> None:
    """Ask Ollama to unload the embedding model from VRAM (best-effort)."""
    import urllib.request
    import urllib.error
    url  = base_url.rstrip("/") + "/api/embed"
    body = json.dumps({"model": model_name, "input": [], "keep_alive": "0"}).encode()
    req  = urllib.request.Request(url, data=body,
                                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass


# ── Post-ingest hooks ─────────────────────────────────────────────────────────

def _post_ingest_hooks(db_dsn: str, doc_id: str) -> None:
    """Refresh book registry + corpus manifest after a successful ingest."""
    try:
        from utils.book_registry import refresh_registry_from_db
        refresh_registry_from_db(db_dsn, DEFAULT_BOOK_REGISTRY_PATH)
    except Exception as exc:
        logger.debug("ingest_v3: book registry refresh failed (non-fatal): %s", exc)

    try:
        from retrieval.corpus_scope import update_manifest_for_doc, invalidate_cache
        update_manifest_for_doc(db_dsn, doc_id)
        invalidate_cache()
    except Exception as exc:
        logger.debug("ingest_v3: corpus manifest update failed (non-fatal): %s", exc)

    try:
        from retrieval.query import invalidate_bm25_cache  # type: ignore
        invalidate_bm25_cache()
    except Exception:
        pass


# ── Core pipeline stages ──────────────────────────────────────────────────────

def _run_structure(cleaned: str, markdown_path: Path, doc_id: str,
                   structured_dir: Path) -> Path:
    from pipeline.structure.enrich import enrich_markdown
    structured = enrich_markdown(cleaned, source_path=str(markdown_path))
    structured.setdefault("metadata", {})["doc_id"] = doc_id
    out = structured_dir / f"{markdown_path.stem}.structured.json"
    out.write_text(json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def _run_chunk(structured_path: Path, chunks_dir: Path) -> Path:
    from pipeline.chunk.assembly import chunk_structured_file
    from pipeline.chunk.merge import merge_small_chunks

    # If the structured file has no sections but has preamble content (e.g.
    # schedule docs, plain-text files with no headings), inject the preamble
    # as a single body section so the chunker has something to work with.
    structured = json.loads(structured_path.read_text(encoding="utf-8"))
    if not structured.get("sections") and structured.get("preamble", "").strip():
        structured["sections"] = [{
            "section_id": "s0000",
            "title": structured.get("metadata", {}).get("title", structured_path.stem),
            "level": 1,
            "path": [],
            "path_text": "",
            "content": structured["preamble"],
            "has_table": False,
            "structural_role": "body",
        }]
        structured_path.write_text(
            json.dumps(structured, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    raw_path   = chunks_dir / f"{structured_path.stem}.chunks.json"
    raw_chunks = chunk_structured_file(str(structured_path), output_path=str(raw_path))
    merged     = merge_small_chunks(raw_chunks)
    merged_path = chunks_dir / f"{structured_path.stem}.chunks.merged.json"
    merged_path.write_text(
        json.dumps({"source_structured_path": str(structured_path),
                    "chunk_count": len(merged), "chunks": merged},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return merged_path


def _run_embed(merged_path: Path, index_dir: Path, *,
               backend: str, dimension: int, model_name: str,
               ollama_base_url: str, ollama_timeout: float) -> Path:
    from pipeline.embed.index import build_embedding_index

    index_path = index_dir / f"{merged_path.stem}.index.json"
    build_embedding_index(
        str(merged_path),
        output_path=str(index_path),
        backend=backend,
        dimension=dimension,
        model_name=model_name,
        ollama_base_url=ollama_base_url,
        ollama_timeout_seconds=ollama_timeout,
    )
    if backend == "ollama":
        _unload_embed_model(model_name, ollama_base_url)
    return index_path


def _retire_stale_doc(db_dsn: str, path: Path, new_doc_id: str) -> None:
    """Delete any existing document record for *path* if its hash has changed.

    When a file is re-ingested after being modified the new SHA-256 hash
    produces a different doc_id.  The old document row (and its chunks, via
    ON DELETE CASCADE) must be deleted so stale content does not linger in the
    retrieval index.  If the file is unchanged (same hash → same doc_id) this
    is a no-op.
    """
    from db.client import get_doc_id_for_path, delete_document
    old_id = get_doc_id_for_path(db_dsn, str(path))
    if old_id and old_id != new_doc_id:
        logger.info(
            "ingest_v3: file changed — retiring stale doc %s (was %s, now %s)",
            path.name, old_id[:12], new_doc_id[:12],
        )
        delete_document(db_dsn, old_id)


def _persist(db_dsn: str, doc_id: str, path: Path, *,
             source_type: str, num_pages: int, metadata: Dict[str, Any],
             now_iso: str, markdown_path: Optional[Path],
             structured_path: Optional[Path], merged_path: Optional[Path],
             index_path: Optional[Path]) -> int:
    from db.client import upsert_document_record, upsert_artifact, upsert_chunks_from_index

    upsert_document_record(
        db_dsn,
        doc_id=doc_id,
        filename=path.name,
        source_path=str(path),
        source_type=source_type,
        num_pages=num_pages,
        metadata=metadata,
        ingested_at=now_iso,
    )
    for kind, apath in [("markdown", markdown_path), ("structured", structured_path),
                        ("chunks", merged_path), ("index", index_path)]:
        if apath:
            upsert_artifact(db_dsn, doc_id, kind, str(apath))

    if index_path:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        return upsert_chunks_from_index(db_dsn, payload, replace_doc_chunks=True)
    return 0


# ── ingest_text ───────────────────────────────────────────────────────────────

def _ingest_text(
    path: Path, *,
    db_dsn: str,
    source_type: Optional[str],
    collection_id: Optional[str],
    embed_backend: str,
    embed_dimension: int,
    embed_model_name: str,
    embed_ollama_base_url: str,
    embed_ollama_timeout: float,
) -> Dict[str, Any]:
    ext = path.suffix.lower()
    logger.info("ingest_v3: [text] %s", path.name)

    text = _read_text_file(path)
    if not text.strip():
        logger.warning("ingest_v3: empty text from %s — skipping", path.name)
        return {"kind": "text", "filename": path.name, "doc_id": "", "chunks": 0, "error": "empty"}

    doc_id      = _file_hash(path)
    _retire_stale_doc(db_dsn, path, doc_id)
    eff_type    = source_type or _source_type_for_ext(ext)
    stem        = _artifact_stem(doc_id, path.name)
    now_iso     = datetime.now(timezone.utc).isoformat()
    full_meta   = {
        "title": path.stem, "source_name": path.name,
        "document_title": path.stem, "document_path": str(path),
        "source_type": eff_type, "collection_id": collection_id, "ext": ext,
    }

    # ── Dirs (always set — pipeline always runs) ──────────────────────────────
    md_dir         = _data_dir("markdown")
    structured_dir = _data_dir("structured")
    chunks_dir     = _data_dir("chunks")
    index_dir      = _data_dir("index")

    # ── 1. Clean ──────────────────────────────────────────────────────────────
    from pipeline.normalize.clean_markdown import clean_markdown
    cleaned = clean_markdown(text)

    # ── 2. Save markdown ──────────────────────────────────────────────────────
    markdown_path = md_dir / f"{stem}.md"
    markdown_path.write_text(cleaned, encoding="utf-8")
    logger.debug("ingest_v3: saved markdown → %s", markdown_path)

    # ── 3. Structure ──────────────────────────────────────────────────────────
    structured_path = _run_structure(cleaned, markdown_path, doc_id, structured_dir)
    logger.debug("ingest_v3: saved structured → %s", structured_path)

    # ── 4. Chunk ──────────────────────────────────────────────────────────────
    merged_path = _run_chunk(structured_path, chunks_dir)
    chunk_count = json.loads(merged_path.read_text(encoding="utf-8")).get("chunk_count", 0)
    logger.debug("ingest_v3: %d chunks → %s", chunk_count, merged_path)

    # ── 5. Embed ──────────────────────────────────────────────────────────────
    index_path = _run_embed(
        merged_path, index_dir,
        backend=embed_backend, dimension=embed_dimension, model_name=embed_model_name,
        ollama_base_url=embed_ollama_base_url, ollama_timeout=embed_ollama_timeout,
    )
    logger.debug("ingest_v3: saved index → %s", index_path)

    # ── 6. Persist ────────────────────────────────────────────────────────────
    chunk_rows = _persist(
        db_dsn, doc_id, path,
        source_type=eff_type, num_pages=0, metadata=full_meta, now_iso=now_iso,
        markdown_path=markdown_path, structured_path=structured_path,
        merged_path=merged_path, index_path=index_path,
    )
    logger.info("ingest_v3: [text] %s → %d chunks in DB", path.name, chunk_rows)

    # ── 7. Post-ingest hooks ──────────────────────────────────────────────────
    _post_ingest_hooks(db_dsn, doc_id)

    return {"kind": "text", "filename": path.name, "doc_id": doc_id, "chunks": chunk_rows}


# ── pdf_ingest integration helpers ───────────────────────────────────────────

# ── ingest_pdf ────────────────────────────────────────────────────────────────

def _ingest_pdf(
    path: Path, *,
    db_dsn: str,
    source_type: Optional[str],
    collection_id: Optional[str],
    embed_backend: str,
    embed_dimension: int,
    embed_model_name: str,
    embed_ollama_base_url: str,
    embed_ollama_timeout: float,
) -> Dict[str, Any]:
    logger.info("ingest_v3: [pdf] %s", path.name)

    pdf_meta = _pdf_metadata(path)
    doc_id   = _file_hash(path)
    _retire_stale_doc(db_dsn, path, doc_id)
    eff_type = source_type or "pdf"
    stem     = _artifact_stem(doc_id, path.name)
    now_iso  = datetime.now(timezone.utc).isoformat()
    full_meta = {
        **pdf_meta,
        "collection_id": collection_id, "source_name": path.name,
        "document_title": pdf_meta["title"], "document_path": str(path),
        "source_type": eff_type,
    }

    md_dir         = _data_dir("markdown")
    structured_dir = _data_dir("structured")
    chunks_dir     = _data_dir("chunks")
    index_dir      = _data_dir("index")

    # ── 1. Unload embed model before Marker loads GPU models ──────────────────
    if embed_backend == "ollama":
        _unload_embed_model(embed_model_name, embed_ollama_base_url)

    # ── 2. Extract → markdown (PyMuPDF + Marker + PDFPlumber) ────────────────
    from pipeline.extract import extract_to_markdown
    markdown_text = extract_to_markdown(path)
    if not markdown_text.strip():
        logger.error("ingest_v3: extraction returned empty markdown for %s", path.name)
        return {"kind": "pdf", "filename": path.name, "doc_id": doc_id,
                "chunks": 0, "error": "empty extraction"}

    # ── 3. Clean ──────────────────────────────────────────────────────────────
    from pipeline.normalize.clean_markdown import clean_markdown
    cleaned = clean_markdown(markdown_text)

    # ── 4. Save markdown ──────────────────────────────────────────────────────
    markdown_path = md_dir / f"{stem}.md"
    markdown_path.write_text(cleaned, encoding="utf-8")
    logger.debug("ingest_v3: saved markdown → %s", markdown_path)

    # ── 5. Structure ──────────────────────────────────────────────────────────
    structured_path = _run_structure(cleaned, markdown_path, doc_id, structured_dir)
    logger.debug("ingest_v3: saved structured → %s", structured_path)

    # ── 6. Chunk ──────────────────────────────────────────────────────────────
    merged_path = _run_chunk(structured_path, chunks_dir)
    chunk_count = json.loads(merged_path.read_text(encoding="utf-8")).get("chunk_count", 0)
    logger.debug("ingest_v3: %d chunks → %s", chunk_count, merged_path)

    # ── 7. Embed ──────────────────────────────────────────────────────────────
    index_path = _run_embed(
        merged_path, index_dir,
        backend=embed_backend, dimension=embed_dimension, model_name=embed_model_name,
        ollama_base_url=embed_ollama_base_url, ollama_timeout=embed_ollama_timeout,
    )
    logger.debug("ingest_v3: saved index → %s", index_path)

    # ── 8. Persist ────────────────────────────────────────────────────────────
    chunk_rows = _persist(
        db_dsn, doc_id, path,
        source_type=eff_type, num_pages=pdf_meta["num_pages"], metadata=full_meta,
        now_iso=now_iso, markdown_path=markdown_path, structured_path=structured_path,
        merged_path=merged_path, index_path=index_path,
    )
    logger.info("ingest_v3: [pdf] %s → %d chunks in DB", path.name, chunk_rows)

    # ── 9. Post-ingest hooks ──────────────────────────────────────────────────
    _post_ingest_hooks(db_dsn, doc_id)

    return {
        "kind": "pdf", "filename": path.name,
        "doc_id": doc_id, "chunks": chunk_rows,
        "num_pages": pdf_meta["num_pages"],
    }


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_file(
    input_path: str,
    db_dsn: Optional[str] = None,
    source_type: Optional[str] = None,
    collection_id: Optional[str] = None,
    embed_backend: str = DEFAULT_EMBED_BACKEND,
    embed_dimension: int = DEFAULT_EMBED_DIMENSION,
    embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
    embed_ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
    embed_ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """
    Ingest any supported file through the full pipeline into PostgreSQL.

    Always runs: extract → clean → structure → chunk → embed → persist.
    Output artifacts are written to data/{markdown,structured,chunks,index}/.

    Returns
    -------
    dict with keys: kind, filename, doc_id, chunks, [num_pages], [error]
    """
    path    = Path(input_path).resolve()
    db      = db_dsn or DEFAULT_DB_DSN
    kwargs  = dict(
        db_dsn=db,
        source_type=source_type,
        collection_id=collection_id,
        embed_backend=embed_backend,
        embed_dimension=embed_dimension,
        embed_model_name=embed_model_name,
        embed_ollama_base_url=embed_ollama_base_url,
        embed_ollama_timeout=embed_ollama_timeout_seconds,
    )
    if path.suffix.lower() == ".pdf":
        return _ingest_pdf(path, **kwargs)
    return _ingest_text(path, **kwargs)
