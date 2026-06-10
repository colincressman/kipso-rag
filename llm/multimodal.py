"""Small multimodal helpers for Ollama image-capable chat models."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

from llm.generation import ollama_chat
from utils.runtime_defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def ollama_chat_with_image(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_path: str | Path,
    base_url: str = DEFAULT_LLM_BASE_URL,
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    temperature: float = 0.1,
    keep_alive: int = 0,
) -> str:
    path = Path(image_path)
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"temperature": temperature},
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [_image_to_base64(path)],
            },
        ],
    }
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return ((body.get("message") or {}).get("content") or "").strip()


def describe_document_artifact(
    *,
    image_path: str | Path,
    artifact_kind: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str = DEFAULT_LLM_BASE_URL,
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    temperature: float = 0.1,
    keep_alive: int = 300,
) -> str:
    system_prompt = (
        "You are helping normalize document artifacts for retrieval. Focus on what "
        "the page actually is. Prefer concrete descriptions over buzzwords. If the "
        "artifact is a drawing, diagram, schematic, form, table, or image, say that directly."
    )
    user_prompt = (
        f"This document artifact was detected as '{artifact_kind}'. "
        "Write a concise 1-2 sentence blurb that could replace noisy OCR text for search and extraction. "
        "Describe what the artifact actually is and what information it contains. "
        "Do not mention chunk ids, OCR, or the detection pipeline."
    )
    return ollama_chat_with_image(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=image_path,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        keep_alive=keep_alive,
    )


def describe_document_artifact_structured(
    *,
    image_path: str | Path,
    artifact_kind: str,
    model: str = DEFAULT_LLM_MODEL,
    base_url: str = DEFAULT_LLM_BASE_URL,
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    temperature: float = 0.1,
    keep_alive: int = 300,
) -> Dict[str, Any]:
    system_prompt = (
        "You are helping normalize document artifacts for retrieval and extraction. "
        "Return only valid JSON. Be concrete and factual. Focus on what the artifact actually is "
        "and what information it contains. Prefer specific technical labels over generic ones."
    )
    user_prompt = (
        f"This document artifact was detected as '{artifact_kind}'. "
        "Return a JSON object with exactly these keys: "
        "artifact_type, summary, key_elements, replacement_text. "
        "Rules: "
        "artifact_type must be a specific short label like network diagram, electrical schematic, table of contents, meeting notes, checklist form, title block, or photo. Avoid generic labels like image unless nothing more specific is possible. "
        "summary must be 1-2 sentences that state what the artifact is and what information it captures. "
        "key_elements must be an array of 3 to 6 short strings naming the main visible components, entities, or topics shown on the page. Do not leave it empty. "
        "replacement_text must be 2 to 4 sentences of clean text that could replace noisy OCR for retrieval and extraction. It should be slightly richer than the summary and mention the main relationships, components, or sections shown. "
        "Do not mention OCR, chunk ids, or the detection pipeline. "
        "Return JSON only."
    )
    raw = ollama_chat_with_image(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=image_path,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
        temperature=temperature,
        keep_alive=keep_alive,
    )
    try:
        obj = json.loads(raw)
    except Exception:
        fallback = describe_document_artifact(
            image_path=image_path,
            artifact_kind=artifact_kind,
            model=model,
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            temperature=temperature,
            keep_alive=keep_alive,
        )
        try:
            structured_raw = ollama_chat(
                model=model,
                system_prompt=(
                    "Return only valid JSON. Convert the artifact description into a structured summary. "
                    "Use specific technical labels when possible."
                ),
                user_prompt=(
                    f"Detected artifact kind: {artifact_kind}\n\n"
                    f"Artifact description:\n{fallback}\n\n"
                    "Return a JSON object with exactly these keys: "
                    "artifact_type, summary, key_elements, replacement_text. "
                    "Rules: artifact_type should be specific when possible, key_elements must contain 3 to 6 short strings, "
                    "and replacement_text should be slightly richer than summary. Return JSON only."
                ),
                base_url=base_url,
                timeout_seconds=timeout_seconds,
                temperature=temperature,
                keep_alive=keep_alive,
            )
            obj = json.loads(structured_raw)
            raw = structured_raw
        except Exception:
            return {
                "artifact_type": artifact_kind,
                "summary": fallback,
                "key_elements": [],
                "replacement_text": fallback,
                "raw_response": raw,
            }

    artifact_type = str(obj.get("artifact_type") or artifact_kind).strip() or artifact_kind
    summary = str(obj.get("summary") or "").strip()
    replacement_text = str(obj.get("replacement_text") or summary or "").strip()
    key_elements = obj.get("key_elements")
    if not isinstance(key_elements, list):
        key_elements = []
    clean_key_elements = [str(x).strip() for x in key_elements if str(x).strip()]

    if artifact_type.casefold() == "image":
        lowered = f"{summary} {replacement_text}".casefold()
        if "network" in lowered and ("diagram" in lowered or "schematic" in lowered):
            artifact_type = "network diagram"
        elif "diagram" in lowered:
            artifact_type = "diagram"
        elif "schematic" in lowered:
            artifact_type = "schematic"
        elif "table of contents" in lowered:
            artifact_type = "table of contents"
        elif "meeting notes" in lowered:
            artifact_type = "meeting notes"

    if not clean_key_elements:
        fallback_parts = []
        lowered = f"{summary} {replacement_text}".casefold()
        if "plc" in lowered:
            fallback_parts.append("PLCs")
        if "switch" in lowered:
            fallback_parts.append("switches")
        if "fiber" in lowered:
            fallback_parts.append("fiber connections")
        if "control" in lowered and "room" in lowered:
            fallback_parts.append("control rooms")
        if "table of contents" in lowered:
            fallback_parts.extend(["section headings", "page numbers", "document structure"])
        clean_key_elements = fallback_parts[:6]

    if summary and replacement_text and replacement_text == summary:
        if clean_key_elements:
            replacement_text = (
                f"{summary} Key visible elements include {', '.join(clean_key_elements[:4])}."
            )

    return {
        "artifact_type": artifact_type,
        "summary": summary,
        "key_elements": clean_key_elements,
        "replacement_text": replacement_text or summary,
        "raw_response": raw,
    }
