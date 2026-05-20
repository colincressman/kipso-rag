from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from retrieval.query import RetrievalFilters, retrieve_as_dict
from retrieval.context_pack import build_context_pack
from llm.answer import answer_query_with_retrieval, prepare_rag_answer, finalize_rag_answer, _RagGenCtx
from llm.generation import ollama_stream
from retrieval.router import route_query
from utils.config import load_yaml_config
from utils.runtime_defaults import (
	DEFAULT_DB_DSN,
	DEFAULT_EMBED_BACKEND,
	DEFAULT_EMBED_MODEL_NAME,
	DEFAULT_LLM_BASE_URL,
	DEFAULT_LLM_MODEL,
	DEFAULT_LLM_TIMEOUT_SECONDS,
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


def _load_cli_llm_defaults() -> dict:
	cfg = load_yaml_config("configs/llm.yaml", default={})
	llm_cfg = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
	return {
		"llm_model": llm_cfg.get("model", DEFAULT_LLM_MODEL),
		"llm_base_url": llm_cfg.get("base_url", DEFAULT_LLM_BASE_URL),
		"llm_timeout": llm_cfg.get("timeout_seconds", DEFAULT_LLM_TIMEOUT_SECONDS),
	}


def main() -> None:
	_configure_utf8_stdio()

	raw_argv = sys.argv[1:]
	user_set_top_k = any(
		tok == "--top-k" or tok.startswith("--top-k=")
		for tok in raw_argv
	)

	defaults = _load_cli_llm_defaults()
	parser = argparse.ArgumentParser(description="Query SQLite vector store")
	parser.add_argument("query", type=str, help="Natural language query")
	parser.add_argument("--db", type=str, default=DEFAULT_DB_DSN)
	parser.add_argument("--top-k", type=int, default=8)
	parser.add_argument("--doc-id", type=str, default=None)
	parser.add_argument("--path-prefix", type=str, default=None)
	parser.add_argument("--min-page", type=int, default=None)
	parser.add_argument("--max-page", type=int, default=None)
	parser.add_argument("--has-table", action="store_true")
	parser.add_argument("--source-type", type=str, default=None)
	parser.add_argument("--structural-role", type=str, default=None)
	parser.add_argument("--collection", type=str, default=None, help="Scope retrieval to a named collection")
	parser.add_argument("--backend", type=str, default=DEFAULT_EMBED_BACKEND)
	parser.add_argument("--model", type=str, default=DEFAULT_EMBED_MODEL_NAME)
	parser.add_argument("--cross-encoder", dest="cross_encoder", action="store_true", help="Enable second-stage cross-encoder reranking")
	parser.add_argument("--no-cross-encoder", dest="cross_encoder", action="store_false", help="Disable second-stage cross-encoder reranking")
	parser.set_defaults(cross_encoder=None)
	parser.add_argument("--cross-only", dest="cross_only", action="store_true", help="Use cross-encoder-only retrieval (no bi-encoder vector scoring)")
	parser.add_argument("--no-cross-only", dest="cross_only", action="store_false", help="Disable cross-only retrieval mode")
	parser.set_defaults(cross_only=None)
	parser.add_argument("--answer", action="store_true", help="Generate grounded final answer via llm/answer.py")
	parser.add_argument("--stream", action="store_true", help="Stream LLM tokens to stdout as they arrive (implies --answer)")
	parser.add_argument("--llm-model", type=str, default=defaults["llm_model"])
	parser.add_argument("--llm-base-url", type=str, default=defaults["llm_base_url"])
	parser.add_argument("--llm-timeout", type=float, default=float(defaults["llm_timeout"]))
	parser.add_argument("--llm-config", type=str, default="configs/llm.yaml")
	args = parser.parse_args()

	# Route first: resolves intent, strategy, source-type preference, and any
	# collection-scope phrase detected from the query text.
	routed = route_query(args.query, db_dsn=args.db)
	strategy = routed.strategy

	# Build filters once. Explicit flags take priority; fall back to values
	# auto-detected from the query text.
	effective_source_type = args.source_type or routed.source_type_filter
	effective_collection = args.collection or routed.collection_id
	filters = RetrievalFilters(
		doc_id=args.doc_id,
		path_prefix=args.path_prefix,
		min_page=args.min_page,
		max_page=args.max_page,
		has_table=True if args.has_table else None,
		source_type=effective_source_type,
		structural_role=args.structural_role,
		collection_id=effective_collection,
	)

	# Use the stripped query (scoping phrase removed) for embedding/HyDE.
	effective_query = routed.effective_query

	runtime_top_k = int(args.top_k) if user_set_top_k else max(int(args.top_k), int(strategy.top_k))
	runtime_candidate_k = max(runtime_top_k, int(strategy.candidate_k))

	hard_filters_set = any([
		args.doc_id,
		args.path_prefix,
		args.min_page is not None,
		args.max_page is not None,
		args.has_table,
	])

	if hard_filters_set:
		# Hard filters: honour user intent exactly, skip strategy overrides.
		retrieve_kwargs = {
			"db_path": args.db,
			"top_k": args.top_k,
			"filters": filters,
			"embed_backend": args.backend,
			"embed_model_name": args.model,
		}
		if args.cross_encoder is not None:
			retrieve_kwargs["cross_encoder_enabled"] = bool(args.cross_encoder)
		if args.cross_only is not None:
			retrieve_kwargs["cross_encoder_only"] = bool(args.cross_only)
		result = retrieve_as_dict(effective_query, **retrieve_kwargs)
	else:
		routed_kwargs = {
			"db_path": args.db,
			"top_k": runtime_top_k,
			"filters": filters,
			"rerank_candidate_k": runtime_candidate_k,
			"rerank_alpha_vector": float(strategy.alpha_vector),
			"rerank_alpha_lexical": float(strategy.alpha_lexical),
			"rerank_prefer_tables": bool(strategy.prefer_tables),
			"rerank_prefer_shorter": bool(strategy.prefer_shorter),
			"embed_backend": args.backend,
			"embed_model_name": args.model,
			"intent": routed.intent,
		}
		if args.cross_encoder is not None:
			routed_kwargs["cross_encoder_enabled"] = bool(args.cross_encoder)
		if args.cross_only is not None:
			routed_kwargs["cross_encoder_only"] = bool(args.cross_only)
		result = retrieve_as_dict(effective_query, **routed_kwargs)

	context_pack = build_context_pack(result, routed, max_chunks=runtime_top_k)
	result["context_pack"] = context_pack

	if args.answer or args.stream:
		answer_input = {
			**result,
			"hits": context_pack.get("selected_chunks", []),
		}
		if args.stream:
			ctx = prepare_rag_answer(
				effective_query,
				answer_input,
				intent=routed.intent,
				llm_model=args.llm_model,
				llm_base_url=args.llm_base_url,
				llm_timeout_seconds=args.llm_timeout,
				config_path=args.llm_config,
			)
			if not isinstance(ctx, _RagGenCtx):
				answer = ctx
				print(answer.get("answer", ""), flush=True)
			else:
				full_text = ""
				for token in ollama_stream(
					model=ctx.model,
					system_prompt=ctx.system_prompt,
					user_prompt=ctx.user_prompt,
					base_url=ctx.base_url,
					timeout_seconds=ctx.timeout_seconds,
					temperature=ctx.temperature,
				):
					full_text += token
					print(token, end="", flush=True)
				print()  # newline after streaming
				answer = finalize_rag_answer(ctx, full_text)
		else:
			answer = answer_query_with_retrieval(
				effective_query,
				answer_input,
				intent=routed.intent,
				llm_model=args.llm_model,
				llm_base_url=args.llm_base_url,
				llm_timeout_seconds=args.llm_timeout,
				config_path=args.llm_config,
			)
		answer["context_pack"] = context_pack
		if "internet_fallback" in result:
			answer["internet_fallback"] = result["internet_fallback"]
		if not args.stream:
			print(json.dumps(answer, indent=2, ensure_ascii=False))
	else:
		print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()

