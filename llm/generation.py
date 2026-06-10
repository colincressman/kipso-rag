"""Core LLM generation — Ollama HTTP call, config loading, and fallback answers."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List

from utils.config import load_yaml_config
from utils.runtime_defaults import (
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_LLM_BASE_URL,
	DEFAULT_LLM_MODEL,
	DEFAULT_LLM_TEMPERATURE,
	DEFAULT_LLM_TIMEOUT_SECONDS,
)

# Module-level cache for load_llm_config() — keyed by resolved path string.
# Eliminates repeated YAML reads on every request in steady-state operation.
_llm_config_cache: dict[str, Any] = {}
def _reload_after_llm(base_url: str = DEFAULT_LLM_BASE_URL) -> None:
	"""
	Non-blocking. Delegates to VRAMManager.after_llm_complete() which reloads
	DeBERTa and pre-warms the embedder, respecting VRAM constraints.
	"""
	try:
		from utils.vram_manager import get_manager  # noqa: PLC0415
		get_manager().after_llm_complete(
			base_url=base_url,
			embed_model=DEFAULT_EMBED_MODEL_NAME,
		)
	except Exception:
		pass


def unload_llm_model(
	*,
	model: str = DEFAULT_LLM_MODEL,
	base_url: str = DEFAULT_LLM_BASE_URL,
) -> None:
	"""Explicitly evict the main Ollama LLM after a run finishes."""
	try:
		from utils.vram_manager import VRAMManager  # noqa: PLC0415
		VRAMManager.evict_ollama_model(model, base_url)
	except Exception:
		pass


def ollama_chat(
	*,
	model: str,
	system_prompt: str,
	user_prompt: str,
	base_url: str = DEFAULT_LLM_BASE_URL,
	timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
	temperature: float = DEFAULT_LLM_TEMPERATURE,
	think: bool = False,
	num_predict: int = -1,
	keep_alive: int = 0,
	manage_vram: bool = True,
) -> str:
	if manage_vram:
		# Free the intent classifier from VRAM before the LLM loads.
		try:
			from retrieval.intent_classifier import unload as _unload_classifier
			_unload_classifier()
		except Exception:
			pass
	url = f"{base_url.rstrip('/')}/api/chat"
	payload = {
		"model": model,
		"stream": False,
		"think": think,
		"keep_alive": keep_alive,
		"options": {
			"temperature": temperature,
			**(({"num_predict": num_predict}) if num_predict > 0 else {}),
		},
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
	}
	data = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
		body = json.loads(resp.read().decode("utf-8"))
	if manage_vram:
		_reload_after_llm(base_url=base_url)
	return ((body.get("message") or {}).get("content") or "").strip()


def ollama_stream(
	*,
	model: str,
	system_prompt: str,
	user_prompt: str,
	base_url: str = DEFAULT_LLM_BASE_URL,
	timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
	temperature: float = DEFAULT_LLM_TEMPERATURE,
	think: bool = False,
	keep_alive: int = 0,
	manage_vram: bool = True,
) -> Iterator[str]:
	"""Stream tokens from Ollama's /api/chat endpoint. Yields content chunks."""
	if manage_vram:
		# Free the intent classifier from VRAM before the LLM loads.
		try:
			from retrieval.intent_classifier import unload as _unload_classifier
			_unload_classifier()
		except Exception:
			pass
	url = f"{base_url.rstrip('/')}/api/chat"
	payload = {
		"model": model,
		"stream": True,
		"think": think,
		"keep_alive": keep_alive,
		"options": {"temperature": temperature},
		"messages": [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
	}
	data = json.dumps(payload).encode("utf-8")
	req = urllib.request.Request(
		url,
		data=data,
		headers={"Content-Type": "application/json"},
		method="POST",
	)
	try:
		with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
			for raw_line in resp:
				line = raw_line.strip()
				if not line:
					continue
				try:
					obj = json.loads(line.decode("utf-8"))
				except (json.JSONDecodeError, UnicodeDecodeError):
					continue
				content = (obj.get("message") or {}).get("content") or ""
				if content:
					yield content
				if obj.get("done"):
					if manage_vram:
						_reload_after_llm(base_url=base_url)
					break
	except Exception:
		return


def fallback_answer(query: str, hits: List[Dict[str, Any]]) -> str:
	if not hits:
		return "I could not find relevant context to answer this question."
	top = hits[0]
	chunk_id = top.get("chunk_id", "unknown")
	text = (top.get("text") or "").strip()
	if len(text) > 700:
		text = text[:700].rstrip() + " ..."
	return (
		"I could not reach the LLM endpoint, so here is the best matching context excerpt.\n\n"
		f"[{chunk_id}] {text}"
	)


def grounded_citation_fallback(hits: List[Dict[str, Any]], max_items: int = 3) -> str:
	if not hits:
		return "I could not find relevant context to answer this question."
	lines = [
		"I could not produce a fully cited synthesis, so here are the most relevant grounded excerpts:",
	]
	for h in hits[:max_items]:
		chunk_id = h.get("chunk_id", "unknown")
		text = (h.get("text") or "").strip()
		if len(text) > 320:
			text = text[:320].rstrip() + " ..."
		lines.append(f"- [{chunk_id}] {text}")
	return "\n".join(lines)


def load_llm_config(config_path: str | None = None) -> Dict[str, Any]:
	path = Path(config_path) if config_path else Path("configs/llm.yaml")
	cache_key = str(path.resolve())
	if cache_key in _llm_config_cache:
		return _llm_config_cache[cache_key]
	default_cfg: Dict[str, Any] = {
		"llm": {
			"model": DEFAULT_LLM_MODEL,
			"base_url": DEFAULT_LLM_BASE_URL,
			"timeout_seconds": 180.0,
			"temperature": 0.05,
		},
		"prompt": {
			"max_chunks": 6,
			"max_chars_per_chunk": 1600,
			"include_neighbor_context": True,
			"max_neighbors_per_chunk": 2,
			"max_chars_per_neighbor": 280,
			"require_inline_citations": True,
			"enforce_sentence_citations": True,
			"min_citations": 2,
			"max_citations": 3,
			"include_retrieval_score": True,
			"force_grounded_fallback_when_uncited": False,
			"citation_score_window": 0.06,
		},
		"decision": {
			"medium_confidence_score": 0.55,
			"high_confidence_score": 0.70,
			"borderline_confidence_score": 0.62,
			"max_ambiguous_gap": 0.03,
			"path_override_min_term_matches": 1,
			"always_use_llm": False,
			"allow_uncited_if_confident": True,
			"allow_low_confidence_answer": True,
			"enforce_entity_grounding": True,
			"entity_grounding_bands": ["high", "medium"],
		},
	}
	path = Path(config_path) if config_path else Path("configs/llm.yaml")
	loaded = load_yaml_config(path, default=default_cfg)

	if not isinstance(loaded.get("llm"), dict):
		loaded["llm"] = dict(default_cfg["llm"])
	if not isinstance(loaded.get("prompt"), dict):
		loaded["prompt"] = dict(default_cfg["prompt"])
	if not isinstance(loaded.get("decision"), dict):
		loaded["decision"] = dict(default_cfg["decision"])

	for section in ("llm", "prompt", "decision"):
		base = default_cfg[section]
		for key, value in base.items():
			loaded[section].setdefault(key, value)

	_llm_config_cache[cache_key] = loaded
	return loaded
