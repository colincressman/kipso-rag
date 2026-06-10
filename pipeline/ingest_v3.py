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
from typing import Any, Callable, Dict, List, Optional

import fitz  # PyMuPDF

from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_DIMENSION,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
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

ProgressCallback = Callable[[str, str], None]


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


def _extract_pymupdf_pages(pdf_path: Path) -> Dict[int, str]:
    """Return a per-page PyMuPDF text map for lightweight artifact heuristics."""
    pages: Dict[int, str] = {}
    try:
        with fitz.open(str(pdf_path)) as doc:
            for i, page in enumerate(doc):
                try:
                    text = page.get_text("text")
                except Exception:
                    text = ""
                pages[i] = text or ""
    except Exception as exc:
        logger.warning("ingest_v3: could not read per-page text for %s: %s", pdf_path.name, exc)
    return pages


def _report_progress(progress_cb: Optional[ProgressCallback], stage: str, detail: str) -> None:
    if not progress_cb:
        return
    try:
        progress_cb(stage, detail)
    except Exception:
        logger.debug("ingest_v3: progress callback failed for stage=%s", stage, exc_info=True)


def _looks_like_form_page(text: str) -> bool:
    """Generic heuristic for form-like pages without hard-coding document-specific names."""
    clean = " ".join(str(text or "").split())
    if not clean:
        return False
    lowered = clean.casefold()
    if _looks_like_table_of_contents_page(text):
        return False
    text_len = len(clean)
    field_markers = sum(
        1 for token in (
            "name:", "date:", "signature", "address", "phone", "email",
            "vendor", "bidder", "proposal", "submit", "contact", "form",
        )
        if token in lowered
    )
    blank_markers = len(re.findall(r"_{3,}|\.{4,}|□|☐|☑", clean))
    colon_count = clean.count(":")

    # Long substantive pages can contain field-like labels without actually
    # being forms. Keep this detector biased toward sparse, structured pages.
    if text_len > 1200 and blank_markers < 2:
        return False

    if blank_markers >= 3:
        return True
    if field_markers >= 4 and colon_count >= 5 and text_len <= 900:
        return True
    if field_markers >= 3 and colon_count >= 6 and text_len <= 650:
        return True
    return False


def _looks_like_table_of_contents_page(text: str) -> bool:
    """Return True when a page looks like a table of contents / section directory."""
    raw = str(text or "")
    clean = " ".join(raw.split())
    if not clean:
        return False

    lowered = clean.casefold()
    if "table of contents" in lowered or lowered.startswith("contents "):
        return True

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) < 5:
        return False

    leader_lines = sum(
        1
        for line in lines
        if re.search(r"(?:\.{3,}|_{3,}|[-·•]{3,})\s*\d{1,4}\s*$", line)
    )
    numbered_entry_lines = sum(
        1
        for line in lines
        if re.search(r"^\s*\d+(?:\.\d+){0,3}\s+\S.*\d{1,4}\s*$", line)
    )
    short_lines = sum(1 for line in lines if len(line) <= 120)

    if leader_lines >= 3:
        return True
    if numbered_entry_lines >= 4 and short_lines >= max(4, len(lines) // 2):
        return True
    return False


def _looks_like_appendix_heading_page(text: str) -> bool:
    """Return True for sparse appendix divider/title pages that should stay as plain text."""
    raw = str(text or "")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return False
    if len(lines) > 4:
        return False

    clean = " ".join(lines)
    if len(clean) > 180:
        return False

    lowered = clean.casefold()
    if "table of contents" in lowered:
        return False

    appendix_heading = re.match(
        r"^\s*appendix\s+[a-z0-9]+(?:\s*[:\-]\s*|\s+).+",
        clean,
        flags=re.IGNORECASE,
    )
    exhibit_heading = re.match(
        r"^\s*(?:exhibit|attachment|schedule)\s+[a-z0-9]+(?:\s*[:\-]\s*|\s+).+",
        clean,
        flags=re.IGNORECASE,
    )
    return bool(appendix_heading or exhibit_heading)


def _page_widget_count(page: "fitz.Page") -> int:
    """Return the number of real PDF form widgets on a page."""
    try:
        widgets = page.widgets()
        if widgets is None:
            return 0
        return len(list(widgets))
    except Exception:
        return 0


def _normalize_form_signature_line(line: str) -> str:
    text = str(line or "").strip()
    if not text:
        return ""
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"^\(?page\s+\d+\s+of\s+\d+\)?\s*[-:]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bpage\s+\d+\s+of\s+\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\brev\.?/?effective\b.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\(\d+\)\s*", "", text)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", " ", text).strip().casefold()
    return text


def _extract_form_packet_signature(text: str) -> tuple[str, ...]:
    """Extract reusable top-of-page signatures for multi-page form packets."""
    raw_lines = [str(line or "").strip() for line in str(text or "").splitlines()]
    lines = [line for line in raw_lines if line]
    if not lines:
        return ()

    signatures: List[str] = []
    for line in lines[:8]:
        normalized = _normalize_form_signature_line(line)
        if not normalized or len(normalized) < 8:
            continue
        if re.fullmatch(r"[ivxlcdm]+", normalized):
            continue
        token_count = len(normalized.split())
        if ":" in normalized and token_count <= 4:
            continue
        if normalized.startswith(("•", "-", "*")):
            continue
        if "form" in normalized:
            signatures.append(normalized)
            continue
        if any(keyword in normalized for keyword in ("schedule", "guideline", "guidelines", "instruction", "instructions", "plan")):
            signatures.append(normalized)
            continue
        if len(signatures) < 2 and token_count >= 5:
            signatures.append(normalized)

    deduped: List[str] = []
    seen: set[str] = set()
    for value in signatures:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return tuple(deduped[:4])


def _is_form_packet_continuation_page(
    page_idx: int,
    *,
    form_signatures: Dict[int, tuple[str, ...]],
    widget_counts: Dict[int, int],
) -> bool:
    """Return True when a page clearly belongs to the same named form packet as a nearby widget page."""
    signature = tuple(form_signatures.get(page_idx) or ())
    if not signature:
        return False

    neighbor_indices = (page_idx - 2, page_idx - 1, page_idx + 1, page_idx + 2)
    for other_idx in neighbor_indices:
        if other_idx == page_idx:
            continue
        if widget_counts.get(other_idx, 0) <= 0:
            continue
        other_signature = tuple(form_signatures.get(other_idx) or ())
        if not other_signature:
            continue
        if set(signature) & set(other_signature):
            return True
    return False


def _collect_pdf_page_artifacts(
    pdf_path: Path,
    doc_id: str,
    stem: str,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    page_text: Optional[Dict[int, str]] = None,
    page_analysis: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Detect and render artifact-like PDF pages into data/artifacts/<stem>/."""
    if page_analysis is None:
        from pipeline.extract.page_classifier import classify_all_pages_with_details
        page_analysis = classify_all_pages_with_details(pdf_path)

    min_table_width_pt = 120.0
    min_table_height_pt = 72.0
    min_table_area_ratio = 0.025
    max_tables_per_page = 3
    giant_table_area_ratio = 0.75
    giant_table_width_ratio = 0.85
    giant_table_height_ratio = 0.85
    blur_caption_kinds = {"image", "form"}

    page_analysis = page_analysis or {}
    page_text = page_text or _extract_pymupdf_pages(pdf_path)
    page_types = {
        pn: str(meta.get("page_type") or "standard")
        for pn, meta in page_analysis.items()
    }
    form_signatures = {
        pn: _extract_form_packet_signature(text)
        for pn, text in page_text.items()
    }
    artifact_dir = _data_dir("artifacts") / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifacts: List[Dict[str, Any]] = []

    try:
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            widget_counts = {
                idx: _page_widget_count(doc[idx])
                for idx in range(total_pages)
            }
            for page_idx, page in enumerate(doc):
                _report_progress(
                    progress_cb,
                    "artifact_analysis",
                    f"Detecting artifact pages ({page_idx + 1}/{total_pages})",
                )
                text = page_text.get(page_idx, "")
                classified = str(page_types.get(page_idx, "standard") or "standard")
                image_count = len(page.get_images(full=True) or [])
                widget_count = int(widget_counts.get(page_idx, 0) or 0)
                text_len = len(text.strip())
                image_dominant = image_count > 0 and text_len < 350

                if _looks_like_table_of_contents_page(text) or _looks_like_appendix_heading_page(text):
                    continue

                artifact_kind: Optional[str] = None
                reasons: List[str] = []

                if widget_count > 0:
                    artifact_kind = "form"
                    reasons.append("native:pdf_form_widgets")
                elif _is_form_packet_continuation_page(
                    page_idx,
                    form_signatures=form_signatures,
                    widget_counts=widget_counts,
                ):
                    artifact_kind = "form"
                    reasons.append("native:form_packet_continuation")
                elif classified == "table_heavy" and not image_dominant:
                    artifact_kind = "table"
                    reasons.append("page_classifier:table_heavy")
                elif image_count > 0 and text_len < 200:
                    artifact_kind = "image"
                    reasons.append("heuristic:image_heavy_sparse_text")

                if artifact_kind is None:
                    continue

                if artifact_kind == "table":
                    table_regions: List[tuple[int, fitz.Rect, float]] = []
                    try:
                        page_rect = page.rect
                        page_area = max(page_rect.width * page_rect.height, 1.0)
                        raw_tables = list((page_analysis.get(page_idx) or {}).get("table_bboxes") or [])
                        giant_page_table = False
                        for table in raw_tables:
                            if not table or len(table) != 4:
                                continue
                            x0, top, x1, bottom = [float(v) for v in table]
                            rect = fitz.Rect(x0, top, x1, bottom) & page_rect
                            if rect.is_empty:
                                continue
                            area_ratio = (rect.width * rect.height) / page_area
                            width_ratio = rect.width / max(page_rect.width, 1.0)
                            height_ratio = rect.height / max(page_rect.height, 1.0)
                            if (
                                area_ratio >= giant_table_area_ratio
                                or (width_ratio >= giant_table_width_ratio and height_ratio >= giant_table_height_ratio)
                            ):
                                giant_page_table = True
                                break

                        if giant_page_table:
                            artifact_kind = "image"
                            reasons.append("table_override:page_spanning_table_bbox")
                            raise StopIteration

                        for idx, table in enumerate(raw_tables, start=1):
                            if not table or len(table) != 4:
                                continue
                            x0, top, x1, bottom = [float(v) for v in table]
                            rect = fitz.Rect(x0, top, x1, bottom)
                            rect = rect & page_rect
                            if rect.is_empty:
                                continue
                            area_ratio = (rect.width * rect.height) / page_area
                            if rect.width < min_table_width_pt:
                                continue
                            if rect.height < min_table_height_pt:
                                continue
                            if area_ratio < min_table_area_ratio:
                                continue
                            # Add a small pad so cropped tables keep border lines and labels.
                            rect = fitz.Rect(
                                max(page_rect.x0, rect.x0 - 8),
                                max(page_rect.y0, rect.y0 - 8),
                                min(page_rect.x1, rect.x1 + 8),
                                min(page_rect.y1, rect.y1 + 8),
                            )
                            table_regions.append((idx, rect, area_ratio))
                    except StopIteration:
                        table_regions = []
                    except Exception as exc:
                        logger.debug(
                            "ingest_v3: table region detection failed for %s page %d: %s",
                            pdf_path.name,
                            page_idx + 1,
                            exc,
                        )

                    if table_regions:
                        table_regions.sort(key=lambda item: item[2], reverse=True)
                        table_regions = table_regions[:max_tables_per_page]
                        zoom = 2.0
                        for table_idx, rect, area_ratio in table_regions:
                            pix = page.get_pixmap(
                                matrix=fitz.Matrix(zoom, zoom),
                                clip=rect,
                                alpha=False,
                            )
                            suffix = f"page_{page_idx + 1:04d}_table_{table_idx:02d}"
                            image_path = artifact_dir / f"{suffix}.png"
                            pix.save(str(image_path))
                            meta = {
                                "doc_id": doc_id,
                                "page_index": page_idx,
                                "page_number": page_idx + 1,
                                "artifact_kind": "table",
                                "artifact_index": table_idx,
                                "table_count_on_page": len(table_regions),
                                "table_area_ratio": area_ratio,
                                "classified_page_type": classified,
                                "image_count": image_count,
                                "widget_count": widget_count,
                                "form_packet_signature": form_signatures.get(page_idx) or "",
                                "text_length": text_len,
                                "reasons": reasons,
                                "text_preview": text[:600],
                                "bbox": [rect.x0, rect.y0, rect.x1, rect.y1],
                                "image_path": str(image_path),
                            }
                            meta_path = artifact_dir / f"{suffix}.json"
                            meta_path.write_text(
                                json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                            artifacts.append({
                                "artifact_type": f"page_table_{page_idx + 1:04d}_{table_idx:02d}",
                                "artifact_path": str(image_path),
                                "metadata_path": str(meta_path),
                                "page_number": page_idx + 1,
                                "artifact_kind": "table",
                            })
                        continue

                zoom = 2.0
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                image_path = artifact_dir / f"page_{page_idx + 1:04d}_{artifact_kind}.png"
                pix.save(str(image_path))

                meta = {
                    "doc_id": doc_id,
                    "page_index": page_idx,
                    "page_number": page_idx + 1,
                    "artifact_kind": artifact_kind,
                    "classified_page_type": classified,
                    "image_count": image_count,
                    "widget_count": widget_count,
                    "form_packet_signature": form_signatures.get(page_idx) or "",
                    "text_length": text_len,
                    "reasons": reasons,
                    "text_preview": text[:600],
                    "image_path": str(image_path),
                }
                meta_path = artifact_dir / f"page_{page_idx + 1:04d}_{artifact_kind}.json"
                meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

                if artifact_kind in blur_caption_kinds:
                    try:
                        from llm.multimodal import describe_document_artifact_structured
                        _report_progress(
                            progress_cb,
                            "artifact_captioning",
                            f"Generating artifact caption for page {page_idx + 1}",
                        )

                        artifact_info = describe_document_artifact_structured(
                            image_path=image_path,
                            artifact_kind=artifact_kind,
                            model=DEFAULT_LLM_MODEL,
                            base_url=DEFAULT_LLM_BASE_URL,
                            timeout_seconds=float(DEFAULT_LLM_TIMEOUT_SECONDS),
                            temperature=0.1,
                            keep_alive=300,
                        )
                        replacement_text = str(artifact_info.get("replacement_text") or "").strip()
                        summary = str(artifact_info.get("summary") or "").strip()
                        artifact_blurb = replacement_text or summary
                        if artifact_blurb:
                            meta["artifact_type_inferred"] = artifact_info.get("artifact_type") or artifact_kind
                            meta["artifact_summary"] = summary
                            meta["artifact_key_elements"] = artifact_info.get("key_elements") or []
                            meta["artifact_blurb"] = artifact_blurb
                            meta["replacement_text"] = artifact_blurb
                            meta["normalized_text"] = artifact_blurb
                            meta["artifact_blurb_model"] = DEFAULT_LLM_MODEL
                            meta["artifact_blurb_generated_at"] = datetime.now(timezone.utc).isoformat()
                            meta_path.write_text(
                                json.dumps(meta, indent=2, ensure_ascii=False),
                                encoding="utf-8",
                            )
                    except Exception as exc:
                        logger.debug(
                            "ingest_v3: artifact blurb generation failed for %s page %d: %s",
                            pdf_path.name,
                            page_idx + 1,
                            exc,
                        )

                artifacts.append({
                    "artifact_type": f"page_{artifact_kind}_{page_idx + 1:04d}",
                    "artifact_path": str(image_path),
                    "metadata_path": str(meta_path),
                    "page_number": page_idx + 1,
                    "artifact_kind": artifact_kind,
                    "artifact_type_inferred": meta.get("artifact_type_inferred"),
                    "artifact_summary": meta.get("artifact_summary"),
                    "artifact_blurb": meta.get("artifact_blurb"),
                    "replacement_text": meta.get("replacement_text"),
                    "normalized_text": meta.get("normalized_text"),
                })
    except Exception as exc:
        logger.warning("ingest_v3: artifact extraction failed for %s: %s", pdf_path.name, exc)

    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "doc_id": doc_id,
        "source_pdf": str(pdf_path),
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    artifacts.append({
        "artifact_type": "page_artifact_manifest",
        "artifact_path": str(manifest_path),
        "metadata_path": str(manifest_path),
        "page_number": None,
        "artifact_kind": "manifest",
    })
    return artifacts


def _apply_artifact_replacements_to_merged(
    merged_path: Path,
    page_artifacts: List[Dict[str, Any]],
) -> int:
    """Replace chunk text with blended artifact text for matching artifact-backed pages."""
    if not merged_path.exists() or not page_artifacts:
        return 0

    replacement_by_page: Dict[int, str] = {}
    for artifact in page_artifacts:
        page_number = artifact.get("page_number")
        replacement_text = str(artifact.get("replacement_text") or artifact.get("normalized_text") or "").strip()
        summary = str(artifact.get("artifact_summary") or "").strip()
        key_elements = artifact.get("artifact_key_elements") or []
        if not isinstance(key_elements, list):
            key_elements = []
        clean_key_elements = [str(x).strip() for x in key_elements if str(x).strip()]
        raw_text = str(artifact.get("text_preview") or "").strip()
        artifact_kind = str(artifact.get("artifact_kind") or "")
        if not isinstance(page_number, int):
            continue
        if not replacement_text:
            continue
        if artifact_kind not in {"image", "form"}:
            continue

        blended_parts: List[str] = []
        if replacement_text:
            blended_parts.append("[Artifact Summary]\n" + replacement_text)
        elif summary:
            blended_parts.append("[Artifact Summary]\n" + summary)
        if clean_key_elements:
            blended_parts.append("[Key Elements]\n" + ", ".join(clean_key_elements[:8]))
        if raw_text:
            trimmed_raw = raw_text[:1200].strip()
            blended_parts.append("[Raw Technical Text]\n" + trimmed_raw)
        blended_text = "\n\n".join(part for part in blended_parts if part.strip()).strip()
        if not blended_text:
            continue
        replacement_by_page[page_number] = blended_text

    if not replacement_by_page:
        return 0

    payload = json.loads(merged_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks")
    if not isinstance(chunks, list):
        return 0

    replaced = 0
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if not isinstance(page_start, int):
            continue
        if not isinstance(page_end, int):
            page_end = page_start
        matched_pages = [p for p in range(page_start, page_end + 1) if p in replacement_by_page]
        if not matched_pages:
            continue

        replacement_parts = [replacement_by_page[p] for p in matched_pages]
        replacement_text = "\n\n".join(dict.fromkeys(part for part in replacement_parts if part.strip()))
        if not replacement_text:
            continue
        chunk["text"] = replacement_text
        chunk["artifact_replaced"] = True
        chunk["artifact_pages"] = matched_pages
        replaced += 1

    payload["chunks"] = chunks
    payload["artifact_replaced_chunk_count"] = replaced
    merged_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return replaced


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
    progress_cb: Optional[ProgressCallback] = None,
    run_post_ingest_hooks: bool = True,
) -> Dict[str, Any]:
    ext = path.suffix.lower()
    logger.info("ingest_v3: [text] %s", path.name)

    _report_progress(progress_cb, "extracting", f"Reading {path.name}")
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
    _report_progress(progress_cb, "preparing", f"Preparing ingest for {path.name}")
    _report_progress(progress_cb, "cleaning", "Cleaning source text")

    # ── 1. Clean ──────────────────────────────────────────────────────────────
    from pipeline.normalize.clean_markdown import clean_markdown
    cleaned = clean_markdown(text)

    # ── 2. Save markdown ──────────────────────────────────────────────────────
    markdown_path = md_dir / f"{stem}.md"
    markdown_path.write_text(cleaned, encoding="utf-8")
    _report_progress(progress_cb, "writing_markdown", "Saving cleaned markdown")
    _report_progress(progress_cb, "structuring", "Structuring document")
    logger.debug("ingest_v3: saved markdown → %s", markdown_path)

    # ── 3. Structure ──────────────────────────────────────────────────────────
    structured_path = _run_structure(cleaned, markdown_path, doc_id, structured_dir)
    logger.debug("ingest_v3: saved structured → %s", structured_path)

    # ── 4. Chunk ──────────────────────────────────────────────────────────────
    merged_path = _run_chunk(structured_path, chunks_dir)
    chunk_count = json.loads(merged_path.read_text(encoding="utf-8")).get("chunk_count", 0)
    logger.debug("ingest_v3: %d chunks → %s", chunk_count, merged_path)

    # ── 5. Embed ──────────────────────────────────────────────────────────────
    _report_progress(progress_cb, "embedding", "Generating embeddings")
    index_path = _run_embed(
        merged_path, index_dir,
        backend=embed_backend, dimension=embed_dimension, model_name=embed_model_name,
        ollama_base_url=embed_ollama_base_url, ollama_timeout=embed_ollama_timeout,
    )
    logger.debug("ingest_v3: saved index → %s", index_path)

    # ── 6. Persist ────────────────────────────────────────────────────────────
    _report_progress(progress_cb, "persisting", "Saving document to database")
    chunk_rows = _persist(
        db_dsn, doc_id, path,
        source_type=eff_type, num_pages=0, metadata=full_meta, now_iso=now_iso,
        markdown_path=markdown_path, structured_path=structured_path,
        merged_path=merged_path, index_path=index_path,
    )
    _report_progress(progress_cb, "finalizing", "Running post-ingest hooks")
    logger.info("ingest_v3: [text] %s → %d chunks in DB", path.name, chunk_rows)

    # ── 7. Post-ingest hooks ──────────────────────────────────────────────────
    if run_post_ingest_hooks:
        _post_ingest_hooks(db_dsn, doc_id)
    _report_progress(progress_cb, "completed", f"Finished {path.name}")

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
    progress_cb: Optional[ProgressCallback] = None,
    run_post_ingest_hooks: bool = True,
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
    _report_progress(progress_cb, "extracting", "Extracting PDF text")

    # ── 2. Extract → markdown (PyMuPDF + Marker + PDFPlumber) ────────────────
    from pipeline.extract import extract_to_markdown_with_context
    markdown_text, extract_ctx = extract_to_markdown_with_context(path)
    if not markdown_text.strip():
        logger.error("ingest_v3: extraction returned empty markdown for %s", path.name)
        return {"kind": "pdf", "filename": path.name, "doc_id": doc_id,
                "chunks": 0, "error": "empty extraction"}

    # ── 3. Clean ──────────────────────────────────────────────────────────────
    from pipeline.normalize.clean_markdown import clean_markdown
    cleaned = clean_markdown(markdown_text)
    _report_progress(progress_cb, "cleaning", "Cleaning extracted markdown")

    # ── 4. Save markdown ──────────────────────────────────────────────────────
    markdown_path = md_dir / f"{stem}.md"
    markdown_path.write_text(cleaned, encoding="utf-8")
    _report_progress(progress_cb, "writing_markdown", "Saving cleaned markdown")
    logger.debug("ingest_v3: saved markdown → %s", markdown_path)

    # ── 5. Structure ──────────────────────────────────────────────────────────
    structured_path = _run_structure(cleaned, markdown_path, doc_id, structured_dir)
    _report_progress(progress_cb, "structuring", "Structuring document")
    _report_progress(progress_cb, "chunking", "Creating chunks")
    logger.debug("ingest_v3: saved structured → %s", structured_path)

    _report_progress(progress_cb, "artifact_analysis", "Detecting artifact pages")
    page_artifacts = _collect_pdf_page_artifacts(
        path,
        doc_id,
        stem,
        progress_cb=progress_cb,
        page_text=(extract_ctx or {}).get("page_text"),
        page_analysis=(extract_ctx or {}).get("page_analysis"),
    )
    if False and page_artifacts:
        _report_progress(progress_cb, "artifact_analysis", f"Saving {max(0, len(page_artifacts) - 1)} artifact(s)")
        for artifact in page_artifacts:
            upsert_artifact(
                db_dsn,
                doc_id,
                str(artifact.get("artifact_type") or ""),
                str(artifact.get("artifact_path") or ""),
            )
        logger.info("ingest_v3: [pdf] %s → %d page artifact(s)", path.name, max(0, len(page_artifacts) - 1))

    # ── 6. Chunk ──────────────────────────────────────────────────────────────
    merged_path = _run_chunk(structured_path, chunks_dir)
    artifact_replaced_chunks = _apply_artifact_replacements_to_merged(merged_path, page_artifacts)
    chunk_count = json.loads(merged_path.read_text(encoding="utf-8")).get("chunk_count", 0)
    logger.debug("ingest_v3: %d chunks → %s", chunk_count, merged_path)
    if artifact_replaced_chunks:
        logger.info(
            "ingest_v3: [pdf] %s → replaced %d chunk(s) with artifact normalized text",
            path.name,
            artifact_replaced_chunks,
        )

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
    if page_artifacts:
        from db.client import upsert_artifact
        _report_progress(progress_cb, "artifact_analysis", f"Saving {max(0, len(page_artifacts) - 1)} artifact(s)")
        for artifact in page_artifacts:
            upsert_artifact(
                db_dsn,
                doc_id,
                str(artifact.get("artifact_type") or ""),
                str(artifact.get("artifact_path") or ""),
            )
        logger.info("ingest_v3: [pdf] %s â†’ saved %d page artifact(s)", path.name, max(0, len(page_artifacts) - 1))
    _report_progress(progress_cb, "finalizing", "Running post-ingest hooks")
    logger.info("ingest_v3: [pdf] %s → %d chunks in DB", path.name, chunk_rows)

    # ── 9. Post-ingest hooks ──────────────────────────────────────────────────
    if run_post_ingest_hooks:
        _post_ingest_hooks(db_dsn, doc_id)
    _report_progress(progress_cb, "completed", f"Finished {path.name}")

    return {
        "kind": "pdf", "filename": path.name,
        "doc_id": doc_id, "chunks": chunk_rows,
        "num_pages": pdf_meta["num_pages"],
        "page_artifacts": len(page_artifacts),
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
    progress_cb: Optional[ProgressCallback] = None,
    run_post_ingest_hooks: bool = True,
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
        progress_cb=progress_cb,
        run_post_ingest_hooks=run_post_ingest_hooks,
    )
    if path.suffix.lower() == ".pdf":
        return _ingest_pdf(path, **kwargs)
    return _ingest_text(path, **kwargs)
