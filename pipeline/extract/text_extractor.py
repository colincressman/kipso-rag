"""Text extraction helpers for DOCX, plain text, and PDF metadata.

Extracted from pipeline/ingest_v3.py so they can be reused across ingest modules.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from html import unescape
from pathlib import Path
from typing import Any, Dict, List

_W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_HEADING_STYLE_MAP = {
    "heading1": "#", "heading2": "##", "heading3": "###",
    "heading4": "####", "heading5": "#####", "heading6": "######",
    "title": "#", "subtitle": "##",
}
_DOCX_SPACE_RE = re.compile(r"\s+")


def _extract_docx_text(path: Path) -> str:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            xml_bytes = archive.read("word/document.xml")
    except Exception:
        return ""
    try:
        root = ET.fromstring(xml_bytes.decode("utf-8", errors="ignore"))
    except ET.ParseError:
        xml = xml_bytes.decode("utf-8", errors="ignore")
        xml = xml.replace("</w:p>", "\n").replace("</w:tr>", "\n")
        text = re.sub(r"<[^>]+>", " ", xml)
        text = unescape(text)
        lines = [_DOCX_SPACE_RE.sub(" ", ln).strip() for ln in text.splitlines()]
        return "\n".join(ln for ln in lines if ln).strip()

    body = root.find(f".//{{{_W}}}body")
    if body is None:
        return ""
    output_lines: List[str] = []
    for para in body.iter(f"{{{_W}}}p"):
        text_parts = [node.text for node in para.iter(f"{{{_W}}}t") if node.text]
        para_text = unescape("".join(text_parts)).strip()
        if not para_text:
            continue
        style_elem = para.find(f".//{{{_W}}}pStyle")
        style_val = (style_elem.get(f"{{{_W}}}val") or "").lower() if style_elem is not None else ""
        prefix = _HEADING_STYLE_MAP.get(style_val, "")
        output_lines.append(f"{prefix} {para_text}" if prefix else para_text)
    return "\n\n".join(output_lines).strip()


def _read_text_file(path: Path) -> str:
    if path.suffix.lower() == ".docx":
        return _extract_docx_text(path)
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return path.read_text(encoding="latin-1", errors="replace").strip()


def _pdf_metadata(path: Path) -> Dict[str, Any]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        meta = doc.metadata or {}
        return {
            "title":     (meta.get("title") or path.stem).strip() or path.stem,
            "author":    meta.get("author"),
            "subject":   meta.get("subject"),
            "num_pages": doc.page_count,
        }
    except Exception:
        return {"title": path.stem, "author": None, "subject": None, "num_pages": 0}
