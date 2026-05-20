"""
Embedding index builder.

Consumes chunk artifacts from pipeline/chunk and writes a JSON vector index
that retrieval can load directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pipeline.embed.embedder import create_embedder, embed_chunks
from utils.runtime_defaults import (
	DEFAULT_EMBED_BACKEND,
	DEFAULT_EMBED_DIMENSION,
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_OLLAMA_BASE_URL,
	DEFAULT_OLLAMA_TIMEOUT_SECONDS,
)


def _load_chunks_payload(chunks_json_path: str) -> Dict[str, Any]:
	path = Path(chunks_json_path)
	payload = json.loads(path.read_text(encoding="utf-8"))
	if "chunks" not in payload or not isinstance(payload["chunks"], list):
		raise ValueError(f"Invalid chunk payload (missing chunks list): {path}")
	return payload


def build_embedding_index(
	chunks_json_path: str,
	*,
	output_path: Optional[str] = None,
	backend: str = DEFAULT_EMBED_BACKEND,
	dimension: int = DEFAULT_EMBED_DIMENSION,
	model_name: str = DEFAULT_EMBED_MODEL_NAME,
	ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
	ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
	"""
	Build an embedding index from chunk JSON.

	Args:
		chunks_json_path: Path to *.chunks.json or *.chunks.merged.json
		output_path: Optional destination JSON path
		backend: embedding backend
		dimension: vector dimension (used by _test backend only)
		model_name: model name for ollama or sentence-transformers backend
	"""
	payload = _load_chunks_payload(chunks_json_path)
	chunks = payload["chunks"]

	embedder = create_embedder(
		backend=backend,
		dimension=dimension,
		model_name=model_name,
		ollama_base_url=ollama_base_url,
		ollama_timeout_seconds=ollama_timeout_seconds,
	)
	embedded = embed_chunks(chunks, embedder=embedder)

	index_payload = {
		"source_chunks_path": str(Path(chunks_json_path)),
		"backend": backend,
		"model_name": model_name if backend == "sentence-transformers" else None,
		"dimension": (len(embedded[0]["embedding"]) if embedded else dimension),
		"vector_count": len(embedded),
		"items": embedded,
	}

	if output_path:
		out = Path(output_path)
	else:
		src = Path(chunks_json_path)
		out = src.with_name(f"{src.stem}.index.json")

	out.parent.mkdir(parents=True, exist_ok=True)
	out.write_text(json.dumps(index_payload, indent=2, ensure_ascii=False), encoding="utf-8")
	return index_payload


def build_index_for_directory(
	chunks_dir: str,
	*,
	pattern: str = "*.chunks.merged.json",
	output_dir: Optional[str] = None,
	backend: str = DEFAULT_EMBED_BACKEND,
	dimension: int = DEFAULT_EMBED_DIMENSION,
	model_name: str = DEFAULT_EMBED_MODEL_NAME,
	ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
	ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
) -> List[str]:
	"""Build indexes for all chunk files in a directory."""
	src_dir = Path(chunks_dir)
	files = sorted(src_dir.glob(pattern))
	created: List[str] = []

	for f in files:
		if output_dir:
			out = Path(output_dir) / f"{f.stem}.index.json"
		else:
			out = f.with_name(f"{f.stem}.index.json")

		build_embedding_index(
			str(f),
			output_path=str(out),
			backend=backend,
			dimension=dimension,
			model_name=model_name,
			ollama_base_url=ollama_base_url,
			ollama_timeout_seconds=ollama_timeout_seconds,
		)
		created.append(str(out))

	return created

