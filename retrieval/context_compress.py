"""Contextual chunk compression via LLM.

For each top-ranked chunk, ask the LLM to extract only the sentences that
are directly relevant to the query.  This reduces prompt size and focuses
the LLM's attention on the most pertinent evidence.

Gate: enabled via ``contextual_compression.enabled: true`` in runtime.yaml.
Only applied to the top ``top_n`` chunks after final ranking.

Usage::

    from retrieval.context_compress import compress_chunks
    hits = compress_chunks(query, hits, model=..., base_url=..., top_n=3)
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
	"You are a precise information extractor. "
	"Given a query and a passage from a document, output only the sentences from the "
	"passage that directly help answer the query. "
	"Copy those sentences verbatim, in their original order, joined with a single space. "
	"If no sentence is relevant, output the first sentence of the passage unchanged. "
	"Do NOT add explanations, labels, or any text that did not appear in the passage."
)

# Minimum chunk length (chars) to bother compressing.
# Short chunks are already focused — compressing them risks losing context.
_MIN_CHARS_TO_COMPRESS = 400


def compress_chunk(
	query: str,
	chunk_text: str,
	*,
	model: str,
	base_url: str,
	timeout_seconds: float = 10.0,
) -> str:
	"""Return a compressed version of *chunk_text* focused on *query*.

	Returns the original text unchanged on any LLM failure so that retrieval
	is never blocked by a compression error.
	"""
	if len(chunk_text) < _MIN_CHARS_TO_COMPRESS:
		return chunk_text

	user_msg = f"Query: {query}\n\nPassage:\n{chunk_text}"
	payload = json.dumps({
		"model": model,
		"stream": False,
		"keep_alive": 0,
		"think": False,
		"options": {"temperature": 0.0, "num_predict": 512},
		"messages": [
			{"role": "system", "content": _SYSTEM_PROMPT},
			{"role": "user", "content": user_msg},
		],
	}).encode("utf-8")
	url = f"{base_url.rstrip('/')}/api/chat"
	req = urllib.request.Request(
		url,
		data=payload,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	try:
		with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
			body = json.loads(resp.read().decode("utf-8"))
		compressed = ((body.get("message") or {}).get("content") or "").strip()
		# Sanity check: if the LLM returned something much longer than the
		# original, or returned nothing useful, fall back to original.
		if not compressed or len(compressed) > len(chunk_text) * 1.1:
			return chunk_text
		return compressed
	except Exception as exc:
		logger.debug("compress_chunk failed (query=%r): %s", query[:60], exc)
		return chunk_text


def compress_chunks(
	query: str,
	hits: List[Dict[str, Any]],
	*,
	model: str,
	base_url: str,
	timeout_seconds: float = 10.0,
	top_n: int = 3,
) -> List[Dict[str, Any]]:
	"""Compress the top *top_n* chunks in *hits* in-place (returns modified list).

	Metadata key ``compressed_by_llm`` is set to ``True`` on compressed chunks.
	The original text is preserved under ``original_text`` for debugging.
	"""
	result: List[Dict[str, Any]] = list(hits)
	for i, hit in enumerate(result):
		if i >= top_n:
			break
		original = str(hit.get("text") or "")
		if len(original) < _MIN_CHARS_TO_COMPRESS:
			continue
		compressed = compress_chunk(
			query,
			original,
			model=model,
			base_url=base_url,
			timeout_seconds=timeout_seconds,
		)
		if compressed != original:
			hit = dict(hit)
			hit["text"] = compressed
			md = dict(hit.get("metadata") or {})
			md["compressed_by_llm"] = True
			md["original_text_len"] = len(original)
			md["compressed_text_len"] = len(compressed)
			hit["metadata"] = md
			result[i] = hit
	return result
