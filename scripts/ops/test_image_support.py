"""Standalone multimodal probe for Ollama image-capable models.

This does not touch the main llm/generation.py wrapper yet. It sends a direct
`/api/chat` request with a single image attached to the user message so we can
validate image support before wiring it into ingestion or extraction.

Examples
--------

Default artifact image:
    python scripts/ops/test_image_support.py

Custom image:
    python scripts/ops/test_image_support.py --image path/to/page.png

Custom model + prompt:
    python scripts/ops/test_image_support.py --model llama3.2-vision \
        --prompt "Describe what this page is and whether it is a table, form, or diagram."
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm.multimodal import (
    describe_document_artifact,
    describe_document_artifact_structured,
    ollama_chat_with_image,
)
from utils.config import load_yaml_config
from utils.runtime_defaults import (
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_TIMEOUT_SECONDS,
)

DEFAULT_IMAGE = (
    PROJECT_ROOT
    / "data"
    / "artifacts"
    / "10ff5af60898_26-d-00017-rfqupost"
    / "page_0093_image.png"
)

DEFAULT_SYSTEM_PROMPT = (
    "You are helping classify document artifacts. Focus on what the page actually is, "
    "not on guessing project meaning. If it is a drawing, diagram, image, table, or form, "
    "say that directly."
)

DEFAULT_USER_PROMPT = (
    "Look at this document artifact image and answer in 3 short parts:\n"
    "1. What type of page is this?\n"
    "2. Is it primarily a table, form, diagram/drawing, or something else?\n"
    "3. Write a 1-2 sentence blurb that could replace noisy OCR text for retrieval."
)


def _configure_utf8_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def _load_llm_defaults() -> dict:
    cfg = load_yaml_config("configs/llm.yaml", default={})
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    return {
        "llm_model": llm_cfg.get("model", DEFAULT_LLM_MODEL),
        "llm_base_url": llm_cfg.get("base_url", DEFAULT_LLM_BASE_URL),
        "llm_timeout": llm_cfg.get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS),
    }


def main() -> None:
    _configure_utf8_stdio()
    defaults = _load_llm_defaults()

    parser = argparse.ArgumentParser(description="Probe Ollama image support with one artifact image.")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Path to the PNG/JPG artifact image.")
    parser.add_argument("--model", type=str, default=defaults["llm_model"], help="Ollama model name.")
    parser.add_argument("--base-url", type=str, default=defaults["llm_base_url"], help="Ollama base URL.")
    parser.add_argument("--timeout", type=float, default=float(defaults["llm_timeout"]), help="HTTP timeout seconds.")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", type=str, default=DEFAULT_USER_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--raw-json", action="store_true", help="Print the full response JSON instead of only the text answer.")
    parser.add_argument("--structured", action="store_true", help="Request the richer structured artifact summary format.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")

    try:
        if args.structured:
            obj = describe_document_artifact_structured(
                image_path=image_path,
                artifact_kind="image",
                model=args.model,
                base_url=args.base_url,
                timeout_seconds=float(args.timeout),
                temperature=float(args.temperature),
            )
            body = {"message": {"content": json.dumps(obj, ensure_ascii=False)}}
        elif args.system_prompt == DEFAULT_SYSTEM_PROMPT and args.prompt == DEFAULT_USER_PROMPT:
            content = describe_document_artifact(
                image_path=image_path,
                artifact_kind="image",
                model=args.model,
                base_url=args.base_url,
                timeout_seconds=float(args.timeout),
                temperature=float(args.temperature),
            )
            body = {"message": {"content": content}}
        else:
            content = ollama_chat_with_image(
                model=args.model,
                system_prompt=args.system_prompt,
                user_prompt=args.prompt,
                image_path=image_path,
                base_url=args.base_url,
                timeout_seconds=float(args.timeout),
                temperature=float(args.temperature),
                keep_alive=300,
            )
            body = {"message": {"content": content}}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"HTTPError {exc.code}", file=sys.stderr)
        print(detail, file=sys.stderr)
        raise SystemExit(1)
    except Exception as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    if args.raw_json:
        print(json.dumps(body, indent=2, ensure_ascii=False))
        return

    if args.structured:
        print(json.dumps(json.loads((body.get("message") or {}).get("content") or "{}"), indent=2, ensure_ascii=False))
        return

    content = ((body.get("message") or {}).get("content") or "").strip()
    print(f"Model: {args.model}")
    print(f"Image: {image_path}")
    print()
    print(content or "[no content returned]")


if __name__ == "__main__":
    main()
