"""
Embedding utilities for chunk text.

Production backends:
	- ollama (default): local embedding model served by Ollama API
	- sentence-transformers (optional): enabled if package is installed

Internal (test-only):
	- _test: deterministic hash vectors; not for production use
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from utils.runtime_defaults import (
	DEFAULT_EMBED_DIMENSION,
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_OLLAMA_BASE_URL,
	DEFAULT_OLLAMA_TIMEOUT_SECONDS,
)


from utils.text_utils import tokenize

# Module-level flag: dimension detected and persisted for this process.
_dim_detected: bool = False


def _l2_normalize(vec: List[float]) -> List[float]:
	norm = math.sqrt(sum(v * v for v in vec))
	if norm == 0:
		return vec
	return [v / norm for v in vec]


def _tokenize(text: str) -> List[str]:
	return tokenize(text.lower())


class Embedder(Protocol):
	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		...

	def embed_query(self, text: str) -> List[float]:
		...


@dataclass
class _TestEmbedder:
	"""
	Deterministic hash-based embedder.

	FOR TEST USE ONLY — not a production backend.
	Produces reproducible low-dimensional vectors without any external
	dependencies, enabling offline unit tests for retrieval logic.
	"""

	dimension: int = DEFAULT_EMBED_DIMENSION
	salt: str = "rag-hash-v1"

	def _hash_token(self, token: str) -> int:
		digest = hashlib.blake2b(
			f"{self.salt}:{token}".encode("utf-8"),
			digest_size=8,
		).digest()
		return int.from_bytes(digest, "big", signed=False)

	def _embed_one(self, text: str) -> List[float]:
		vec = [0.0] * self.dimension
		tokens = _tokenize(text)
		if not tokens:
			return vec

		for tok in tokens:
			h = self._hash_token(tok)
			idx = h % self.dimension
			sign = -1.0 if ((h >> 1) & 1) else 1.0
			vec[idx] += sign

		return _l2_normalize(vec)

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		return [self._embed_one(t) for t in texts]

	def embed_query(self, text: str) -> List[float]:
		return self._embed_one(text)


@dataclass
class SentenceTransformersEmbedder:
	"""Wrapper around sentence-transformers backend (if installed)."""

	model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

	def __post_init__(self) -> None:
		from sentence_transformers import SentenceTransformer  # type: ignore

		self._model = SentenceTransformer(self.model_name)

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		vectors = self._model.encode(texts)
		return [v.tolist() if hasattr(v, "tolist") else list(v) for v in vectors]

	def embed_query(self, text: str) -> List[float]:
		return self.embed_texts([text])[0]


@dataclass
class OllamaEmbedder:
	"""Embedding backend using a locally-running Ollama model."""

	model_name: str = DEFAULT_EMBED_MODEL_NAME
	base_url: str = DEFAULT_OLLAMA_BASE_URL
	timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS
	batch_size: int = 16
	max_retries: int = 4
	retry_backoff_seconds: float = 1.0

	def __hash__(self) -> int:
		return hash((self.model_name, self.base_url, self.timeout_seconds))

	def __eq__(self, other: object) -> bool:
		if not isinstance(other, OllamaEmbedder):
			return NotImplemented
		return (self.model_name, self.base_url, self.timeout_seconds) == (
			other.model_name, other.base_url, other.timeout_seconds
		)

	def _post_json(self, path: str, payload: dict) -> dict:
		url = self.base_url.rstrip("/") + path
		body = json.dumps(payload).encode("utf-8")
		req = urllib.request.Request(
			url,
			data=body,
			headers={"Content-Type": "application/json"},
			method="POST",
		)

		attempt = 0
		while True:
			try:
				with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
					data = resp.read().decode("utf-8")
					return json.loads(data)
			except urllib.error.HTTPError as exc:
				retriable = exc.code in {429, 500, 502, 503, 504}
				if retriable and attempt < self.max_retries:
					time.sleep(self.retry_backoff_seconds * (2 ** attempt))
					attempt += 1
					continue
				raise RuntimeError(
					f"Ollama request failed for {url} (HTTP {exc.code}). "
					f"Is Ollama installed/running and model '{self.model_name}' pulled?"
				) from exc
			except urllib.error.URLError as exc:
				if attempt < self.max_retries:
					time.sleep(self.retry_backoff_seconds * (2 ** attempt))
					attempt += 1
					continue
				raise RuntimeError(
					f"Ollama request failed for {url}. "
					f"Is Ollama installed/running and model '{self.model_name}' pulled?"
				) from exc

	def embed_texts(self, texts: List[str]) -> List[List[float]]:
		"""Embed texts in batches to avoid timeout on large requests."""
		if not texts:
			return []

		# Process in batches to avoid timeout on large document sets
		batch_size = max(1, int(self.batch_size))
		out: List[List[float]] = []
		
		for i in range(0, len(texts), batch_size):
			batch = texts[i:i + batch_size]
			try:
				data = self._post_json("/api/embed", {"model": self.model_name, "input": batch})
				embeddings = data.get("embeddings")
				if isinstance(embeddings, list) and embeddings:
					out.extend(embeddings)
					continue
			except (urllib.error.URLError, RuntimeError):
				# Fallback to micro-batches if a larger batch fails
				pass

			# Fallback: embed one item at a time via /api/embed (avoid /api/embeddings)
			for text in batch:
				data_single = self._post_json(
					"/api/embed",
					{"model": self.model_name, "input": [text]},
				)
				embeddings_single = data_single.get("embeddings")
				vec = embeddings_single[0] if isinstance(embeddings_single, list) and embeddings_single else None
				if not isinstance(vec, list):
					raise RuntimeError("Invalid Ollama embedding response payload.")
				out.append(vec)

		# Lazily detect and persist the embedding dimension on the first
		# successful call in this process.
		global _dim_detected
		if not _dim_detected and out:
			try:
				from utils.embed_meta import detect_and_save as _meta_save  # noqa: PLC0415
				_meta_save(self.model_name, len(out[0]))
				_dim_detected = True
			except Exception:  # noqa: BLE001
				pass
		return out

	@functools.lru_cache(maxsize=256)
	def _cached_embed_query(self, text: str) -> tuple:
		"""Single-text embed with LRU cache. Returns a tuple (hashable) for caching."""
		vec = self.embed_texts([text])[0]
		return tuple(vec)

	def embed_query(self, text: str) -> List[float]:
		return list(self._cached_embed_query(text))


def create_embedder(
	backend: str = "ollama",
	*,
	dimension: int = DEFAULT_EMBED_DIMENSION,
	model_name: str = DEFAULT_EMBED_MODEL_NAME,
	ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL,
	ollama_timeout_seconds: float = DEFAULT_OLLAMA_TIMEOUT_SECONDS,
) -> Embedder:
	"""
	Factory for embedding backends.

	Production backend values:
	  - "ollama" (default) — Ollama local API with qwen3-embedding
	  - "sentence-transformers" — optional, requires the package installed

	Internal/test only:
	  - "_test" — deterministic hash vectors, no external deps; do not use in production
	"""
	b = (backend or "ollama").lower().strip()
	if b == "sentence-transformers":
		return SentenceTransformersEmbedder(model_name=model_name)
	if b == "ollama":
		return OllamaEmbedder(
			model_name=model_name,
			base_url=ollama_base_url,
			timeout_seconds=ollama_timeout_seconds,
		)
	if b == "_test":
		return _TestEmbedder(dimension=dimension)
	raise ValueError(
		f"Unknown embedding backend: {backend!r}. "
		"Use 'ollama' (default), 'sentence-transformers', or '_test'."
	)


def embed_chunks(
	chunks: List[Dict[str, Any]],
	*,
	embedder: Embedder,
	text_field: str = "text",
) -> List[Dict[str, Any]]:
	"""Attach embedding vectors to chunk dicts (non-mutating)."""
	texts = [str(c.get(text_field, "")) for c in chunks]
	vectors = embedder.embed_texts(texts)

	out: List[Dict[str, Any]] = []
	for chunk, vec in zip(chunks, vectors):
		row = dict(chunk)
		row["embedding"] = vec
		row["embedding_dim"] = len(vec)
		out.append(row)
	return out
