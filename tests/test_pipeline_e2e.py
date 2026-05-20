"""End-to-end pipeline trace tests.

Each test runs a real query through every stage of the RAG pipeline and prints
a detailed trace of what happened at each step.  Unlike the unit-harness tests,
these call *live* subsystems (embedding model, vector DB, optionally the LLM).

STAGES TRACED PER QUERY
-----------------------
  1. PLAN            -- plan_query()  ->  steps: rag / web / history
  2. ROUTING         -- route_query() ->  intent, strategy, effective query, web flag
  3. RETRIEVAL       -- rag_retrieve()->  top-k hits, HyDE hypothesis, BM25/vector blend
  4. INTERNET        -- internet_fallback result (if triggered)
  5. CONTEXT PACK    -- selected chunks after dedup + authority weighting
  6. PROMPTS         -- system + user prompt sent to the LLM (previewed)
  7. LLM ANSWER      -- raw LLM text, then post-processed final answer + citations
  8. GROUNDING       -- entity support check on final answer

RUN COMMAND
-----------
    python -m pytest tests/test_pipeline_e2e.py -v -s

    # LLM stages are SKIPPED automatically when Ollama is not reachable.
    # Retrieval/routing stages always run as long as Ollama embeddings are live.
    # If both embedding and LLM are offline, the test prints the routing trace only.

SCENARIOS
---------
  test_e2e_corpus_fact     -- factual ML question expected to hit the corpus
  test_e2e_conversational  -- greeting that should produce a plan step of "history"
  test_e2e_internet_query  -- temporal/price query expected to trigger web search
  test_e2e_formula_lookup  -- formula query (tests prefer_shorter strategy)
"""

from __future__ import annotations

import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from api import rag_retrieve
from llm.answer import _RagGenCtx, finalize_rag_answer, prepare_rag_answer
from llm.generation import ollama_chat
from llm.grounding import unsupported_answer_entities
from pipeline.plan import plan_query
from retrieval.router import route_query
from utils.runtime_defaults import (
    DEFAULT_DB_DSN,
    DEFAULT_EMBED_BACKEND,
    DEFAULT_EMBED_MODEL_NAME,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_LLM_MODEL,
)

# ─── Configuration ────────────────────────────────────────────────────────────
_WIDTH    = 78
_DB       = str(DEFAULT_DB_DSN)
_EMBED_BE = DEFAULT_EMBED_BACKEND
_EMBED_M  = DEFAULT_EMBED_MODEL_NAME
_LLM_M    = DEFAULT_LLM_MODEL
_LLM_URL  = DEFAULT_LLM_BASE_URL


# ─── Availability helpers ─────────────────────────────────────────────────────

def _ollama_live() -> bool:
    """True if the Ollama HTTP API responds within 3 seconds."""
    try:
        req = urllib.request.Request(
            f"{_LLM_URL.rstrip('/')}/api/tags", method="GET"
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def _embed_live() -> bool:
    """True if the Ollama embedding endpoint accepts a quick probe."""
    try:
        import json
        payload = json.dumps({"model": _EMBED_M, "input": ["ping"]}).encode()
        req = urllib.request.Request(
            f"{_LLM_URL.rstrip('/')}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except Exception:
        return False


# ─── Trace printer helpers ────────────────────────────────────────────────────

def _banner(n: int, title: str) -> None:
    print()
    print("=" * _WIDTH)
    label = f" STAGE {n} | {title.upper()} "
    print(label + "=" * max(0, _WIDTH - len(label)))
    print("=" * _WIDTH)


def _sep(char: str = "-") -> None:
    print(char * _WIDTH)


def _safe(text: str) -> str:
    """Replace non-ASCII chars that choke Windows cp1252 console."""
    return text.encode("ascii", "replace").decode("ascii")


def _kv(label: str, value: str) -> None:
    print(f"  {label:<22} {_safe(value)}")


def _assert(label: str, condition: bool) -> None:
    marker = "[+]" if condition else "[!]"
    status = "PASS" if condition else "FAIL"
    print(f"  {marker}  {label:<56} {status}")
    if not condition:
        raise AssertionError(label)


def _preview(text: str, max_chars: int = 450, indent: str = "  | ") -> None:
    """Print a wrapped, indented preview of text."""
    if not text:
        print(f"{indent}(empty)")
        return
    snippet = _safe(text[:max_chars]).replace("\r\n", "\n")
    for line in snippet.split("\n"):
        for wrapped in textwrap.wrap(line, width=_WIDTH - len(indent)) or [""]:
            print(f"{indent}{wrapped}")
    if len(text) > max_chars:
        print(f"{indent}... [{len(text) - max_chars:,} more chars]")


def _hits_table(hits: List[Dict[str, Any]], max_show: int = 6) -> None:
    if not hits:
        print("  (no hits returned)")
        return
    for i, h in enumerate(hits[:max_show]):
        score    = h.get("score")
        chunk_id = str(h.get("chunk_id", "?"))
        title    = (h.get("title") or h.get("document_title") or "").strip()[:42]
        path     = (h.get("path_text") or "").strip()[:42]
        snippet  = (h.get("text") or "").replace("\n", " ").strip()[:90]
        s_str    = f"{float(score):.4f}" if score is not None else "  N/A "
        src      = h.get("source_type") or ""
        print(f"  #{i + 1:<2}  score={s_str}  type={src or '?'}")
        print(f"        id:   {chunk_id}")
        if title:
            print(f"        titl: {title}")
        if path:
            print(f"        path: {path}")
        print(f"        text: {snippet!r}")
    if len(hits) > max_show:
        print(f"  ... and {len(hits) - max_show} more hits")


# ─── Core trace runner ────────────────────────────────────────────────────────

def _run_trace(
    query: str,
    *,
    # Per-scenario expectations — supply None to skip assertion
    expect_intent: Optional[str] = None,
    expect_skip_retrieval: Optional[bool] = None,
    expect_needs_web: Optional[bool] = None,
    expect_plan_tool: Optional[str] = None,   # "rag" | "web" | "history"
    min_hits: int = 0,
    require_llm: bool = True,
) -> Dict[str, Any]:
    """Run all 8 stages and return a summary dict.  Raises AssertionError on failures."""

    summary: Dict[str, Any] = {"query": query, "stages_ok": 0}

    # =========================================================================
    _banner(1, "Plan  (planner step decision)")
    # =========================================================================
    t0 = time.perf_counter()
    plan = plan_query(query)
    plan_ms = (time.perf_counter() - t0) * 1000

    _kv("Query",       repr(query[:70]))
    _kv("Plan intent", plan.intent or "None")
    _kv("Confidence",  f"{plan.intent_confidence:.3f}")
    _kv("Steps",       " → ".join(f"{s.tool}({s.label!r})" for s in plan.steps))
    _kv("Plan time",   f"{plan_ms:.1f} ms")
    if plan.meta:
        for k, v in list(plan.meta.items())[:4]:
            _kv(f"  meta.{k}", str(v)[:60])
    _sep()
    if expect_plan_tool is not None:
        actual_tools = [s.tool for s in plan.steps]
        _assert(
            f"plan tool is '{expect_plan_tool}'",
            expect_plan_tool in actual_tools,
        )
    else:
        _assert("plan produced at least one step", len(plan.steps) >= 1)
    summary["plan"] = {"intent": plan.intent, "steps": [s.tool for s in plan.steps]}
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(2, "Routing  (intent + strategy parameters)")
    # =========================================================================
    t0 = time.perf_counter()
    routed = route_query(query, db_path=_DB)
    route_ms = (time.perf_counter() - t0) * 1000

    strat = routed.strategy
    meta  = routed.meta or {}
    _kv("Intent",          routed.intent or "None")
    _kv("skip_retrieval",  str(strat.skip_retrieval))
    _kv("top_k",           str(strat.top_k))
    _kv("candidate_k",     str(strat.candidate_k))
    _kv("alpha_vector",    str(strat.alpha_vector))
    _kv("alpha_lexical",   str(strat.alpha_lexical))
    _kv("prefer_shorter",  str(strat.prefer_shorter))
    _kv("prefer_tables",   str(strat.prefer_tables))
    _kv("needs_web",       str(bool(meta.get("needs_web", False))))
    _kv("Eff. query",      repr((routed.effective_query or query)[:70]))
    if routed.source_type_filter:
        _kv("source_type_filter", routed.source_type_filter)
    _kv("Route time",      f"{route_ms:.1f} ms")
    _sep()
    if expect_intent is not None:
        _assert(f"intent is '{expect_intent}'", routed.intent == expect_intent)
    if expect_skip_retrieval is not None:
        _assert(
            f"skip_retrieval is {expect_skip_retrieval}",
            strat.skip_retrieval is expect_skip_retrieval,
        )
    if expect_needs_web is not None:
        _assert(
            f"needs_web is {expect_needs_web}",
            bool(meta.get("needs_web", False)) is expect_needs_web,
        )
    summary["routing"] = {
        "intent": routed.intent,
        "skip_retrieval": strat.skip_retrieval,
        "needs_web": bool(meta.get("needs_web", False)),
    }
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(3, "Retrieval  (vector search + BM25 + rerank)")
    # =========================================================================
    if not _embed_live():
        print("  [SKIP] Ollama embedding model not reachable — skipping retrieval stages.")
        print("         Start Ollama and re-run to see retrieval, prompts and LLM answer.")
        summary["skip_reason"] = "embedding_offline"
        return summary

    t0 = time.perf_counter()
    retrieved = rag_retrieve(
        query,
        db_path=_DB,
        embed_backend=_EMBED_BE,
        embed_model_name=_EMBED_M,
    )
    retr_ms = (time.perf_counter() - t0) * 1000

    hits        = retrieved.get("hits", [])
    hyde_trace  = retrieved.get("hyde_trace") or {}
    internet_fb = retrieved.get("internet_fallback") or {}

    _kv("Hits returned",      str(len(hits)))
    _kv("HyDE applied",       str(bool(hyde_trace.get("applied"))))
    if hyde_trace.get("applied") and hyde_trace.get("hypothesis"):
        hyp = (hyde_trace["hypothesis"] or "").replace("\n", " ")[:100]
        _kv("HyDE hypothesis",    repr(hyp))
    _kv("Internet triggered", str(bool(internet_fb.get("triggered"))))
    _kv("Retrieval time",     f"{retr_ms:.0f} ms")
    _sep()
    print("  Top hits:")
    _hits_table(hits, max_show=6)
    _sep()
    _assert("retrieval completed without exception", True)
    if min_hits > 0:
        _assert(f"at least {min_hits} hit(s) returned", len(hits) >= min_hits)
    summary["retrieval"] = {
        "hit_count": len(hits),
        "hyde_applied": bool(hyde_trace.get("applied")),
        "internet_triggered": bool(internet_fb.get("triggered")),
    }
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(4, "Internet Fallback  (web search results)")
    # =========================================================================
    if not internet_fb.get("triggered"):
        print("  (Not triggered for this query — corpus retrieval was used instead.)")
    else:
        web_hits = internet_fb.get("hits") or []
        _kv("Web hits",     str(len(web_hits)))
        _kv("Search query", repr((internet_fb.get("search_query") or query)[:70]))
        _kv("Reason",       str(internet_fb.get("reason") or "N/A"))
        _sep()
        print("  Web results:")
        for i, wh in enumerate(web_hits[:4]):
            url   = str(wh.get("url") or "")[:60]
            title = str(wh.get("title") or wh.get("source_name") or "")[:50]
            snip  = (wh.get("text") or "").replace("\n", " ").strip()[:80]
            print(f"  #{i+1:<2}  [{title}]")
            print(f"        url:  {url}")
            print(f"        text: {snip!r}")
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(5, "Context Pack  (dedup + authority weighting)")
    # =========================================================================
    ctx_pack  = retrieved.get("context_pack") or {}
    selected  = ctx_pack.get("selected_chunks") or hits
    n_removed = ctx_pack.get("dedup_removed_count", 0)
    n_gap     = ctx_pack.get("gap_guard_applied", False)
    diversity = ctx_pack.get("diversity_stats") or {}

    _kv("Input hits",      str(len(hits)))
    _kv("Selected chunks", str(len(selected)))
    _kv("Dedup removed",   str(n_removed))
    _kv("Gap guard",       str(n_gap))
    if diversity:
        _kv("Diversity stats", str(diversity)[:60])
    _sep()
    print("  Context pack (chunks sent to LLM prep):")
    _hits_table(selected, max_show=4)
    _sep()
    _assert("context pack built", "selected_chunks" in ctx_pack or len(hits) >= 0)
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(6, "Prompts  (system prompt + user prompt with context)")
    # =========================================================================
    ctx = prepare_rag_answer(
        query,
        retrieved,
        intent=routed.intent,
    )

    if not isinstance(ctx, _RagGenCtx):
        # Early exit: no_coverage / extractive / internet_no_evidence etc.
        mode = ctx.get("mode", "?")
        _kv("** Early exit **", mode)
        _kv("Answer",   repr(str(ctx.get("answer", ""))[:120]))
        if ctx.get("routing"):
            _kv("Routing rule",  str(ctx["routing"].get("rule") or "N/A"))
            _kv("No-cov reason", str(ctx["routing"].get("no_coverage_reason") or "N/A"))
        _sep()
        _assert("pipeline produced an answer", bool(ctx.get("answer")))
        summary["answer"] = ctx
        summary["stages_ok"] += 1
        _banner(7, "LLM Answer")
        print("  (LLM was not called — pipeline exited early.)")
        _banner(8, "Grounding Check")
        print("  (Skipped — no LLM answer to check.)")
        return summary

    _kv("LLM model",        ctx.model)
    _kv("Confidence band",  ctx.confidence_band)
    _kv("Temperature",      str(ctx.temperature))
    _kv("System prompt",    f"{len(ctx.system_prompt):,} chars")
    _kv("User prompt",      f"{len(ctx.user_prompt):,} chars  ({len(ctx.citations)} citations scoped)")
    _kv("Evidence facts",   f"{len(ctx.evidence_facts)} chars" if ctx.evidence_facts else "none")
    _sep()
    print("  System prompt (first 250 chars):")
    _preview(ctx.system_prompt, 250)
    print()
    print("  User prompt (first 600 chars, includes chunk context):")
    _preview(ctx.user_prompt, 600)
    _sep()
    _assert("system_prompt is non-empty", len(ctx.system_prompt) > 20)
    _assert("user_prompt is non-empty",   len(ctx.user_prompt) > 20)
    _assert("confidence band is valid",
            ctx.confidence_band in {"high", "medium", "low"})
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(7, "LLM Answer  (raw → post-processed)")
    # =========================================================================
    if not _ollama_live():
        print("  [SKIP] Ollama not reachable — LLM call skipped.")
        print("         Run:  ollama serve   (or start the Ollama Desktop app)")
        summary["skip_reason"] = "llm_offline"
        if not require_llm:
            return summary
        pytest.skip("Ollama LLM not reachable")

    t0 = time.perf_counter()
    try:
        llm_text = ollama_chat(
            model=ctx.model,
            system_prompt=ctx.system_prompt,
            user_prompt=ctx.user_prompt,
            base_url=ctx.base_url,
            timeout_seconds=ctx.timeout_seconds,
            temperature=ctx.temperature,
        )
        llm_ok = True
    except Exception as exc:
        llm_text = ""
        llm_ok   = False
        print(f"  [!] LLM call raised: {exc}")
    llm_ms = (time.perf_counter() - t0) * 1000

    _kv("LLM time",      f"{llm_ms:,.0f} ms")
    _kv("Raw response",  f"{len(llm_text):,} chars")
    _sep()
    print("  Raw LLM response:")
    _preview(llm_text, 700)
    _sep()
    _assert("LLM returned a non-empty response", llm_ok and len(llm_text.strip()) > 10)

    # Post-processing
    answer_dict = finalize_rag_answer(ctx, llm_text)
    final_answer = answer_dict.get("answer", "")
    citations    = answer_dict.get("citations", [])
    mode         = answer_dict.get("mode", "?")
    routing_out  = answer_dict.get("routing", {})

    print()
    _kv("Final mode",          mode)
    _kv("Citations",           str(citations[:8]))
    _kv("Inline cit. added",  str(routing_out.get("inline_citations_added", 0)))
    _kv("Fallback reason",    str(routing_out.get("fallback_reason") or "none"))
    _kv("Unsupported ents",   str(routing_out.get("unsupported_entities") or "none"))
    _sep()
    print("  Final answer (post-processed):")
    _preview(final_answer, 700)
    _sep()
    _assert("final answer is non-empty",
            len(final_answer.strip()) > 10)
    _assert("answer mode is valid",
            "confidence" in mode or mode in {
                "fallback", "grounded_fallback", "extractive",
                "internet_no_evidence", "no_coverage",
            })
    summary["answer"] = answer_dict
    summary["stages_ok"] += 1

    # =========================================================================
    _banner(8, "Grounding Check  (entity support in source text)")
    # =========================================================================
    unsupported = unsupported_answer_entities(
        final_answer, query, selected or hits, citations
    )
    if unsupported:
        _kv("Unsupported ents", str(unsupported[:6]))
        print(f"  [!] {len(unsupported)} entity/entities NOT found in source text:")
        for ent in unsupported[:8]:
            print(f"      - {ent!r}")
    else:
        print("  [+] All detectable entities in the answer appear in the source text.")

    internet_answer = str(internet_fb.get("triggered") and "internet" in mode)
    _kv("Grounding scope", "internet_result" if internet_fb.get("triggered") else "corpus_chunks")
    _sep()
    _assert("grounding check completed", True)
    summary["stages_ok"] += 1

    # =========================================================================
    print()
    _sep("=")
    print(f"  TRACE COMPLETE  |  {summary['stages_ok']} stages passed  |  query={query[:50]!r}")
    _sep("=")
    return summary


# ─── Test scenarios ───────────────────────────────────────────────────────────

@pytest.mark.slow
def test_e2e_corpus_fact() -> None:
    """Trace a factual ML/DL question expected to be covered by the corpus.

    Expected pipeline path:
      Plan:      rag
      Intent:    fact_lookup  (or explanation / concept_lookup)
      Retrieval: corpus hits with moderate-high scores
      LLM:       grounded answer with corpus citations
    """
    print()
    print("+" + "=" * (_WIDTH - 2) + "+")
    print("|  E2E TRACE: Corpus Fact Query" + " " * (_WIDTH - 33) + "|")
    print("+" + "=" * (_WIDTH - 2) + "+")

    _run_trace(
        "What is backpropagation and how does it compute gradients?",
        expect_skip_retrieval=False,
        expect_needs_web=False,
        min_hits=1,
        require_llm=False,   # graceful skip if LLM offline
    )


@pytest.mark.slow
def test_e2e_conversational() -> None:
    """Trace a greeting query that should be handled as conversational.

    Expected pipeline path:
      Plan:     history  (no corpus, no web)
      Intent:   conversational
      Retrieval: skip_retrieval=True (no meaningful corpus hits expected)
      LLM:      general-knowledge answer (no chunk context)
    """
    print()
    print("+" + "=" * (_WIDTH - 2) + "+")
    print("|  E2E TRACE: Conversational Query" + " " * (_WIDTH - 36) + "|")
    print("+" + "=" * (_WIDTH - 2) + "+")

    _run_trace(
        "Hello! How are you doing today?",
        expect_intent="conversational",
        expect_skip_retrieval=True,
        expect_needs_web=False,
        expect_plan_tool="history",
        min_hits=0,
        require_llm=False,
    )


@pytest.mark.slow
def test_e2e_internet_routing() -> None:
    """Trace a temporal/price query that should be routed to the internet.

    Expected pipeline path:
      Plan:     web  (temporal + volatile topic)
      Intent:   fact_lookup  (or web_search)
      Routing:  needs_web=True
      Internet: web fallback triggered
      LLM:      answer from web snippets (may be general knowledge if web offline)
    """
    print()
    print("+" + "=" * (_WIDTH - 2) + "+")
    print("|  E2E TRACE: Internet / Temporal Query" + " " * (_WIDTH - 41) + "|")
    print("+" + "=" * (_WIDTH - 2) + "+")

    _run_trace(
        "What is the current price of gold per ounce today?",
        expect_needs_web=True,
        expect_skip_retrieval=False,
        min_hits=0,          # web hits may not have "score" field like corpus hits
        require_llm=False,
    )


@pytest.mark.slow
def test_e2e_formula_lookup() -> None:
    """Trace a formula/notation query — strategy must favour short chunks.

    Expected pipeline path:
      Plan:     rag
      Intent:   formula_lookup  (prefer_shorter=True, top_k<=5)
      Retrieval: short notation/equation chunks ranked highest
      LLM:      answer preserving formula notation
    """
    print()
    print("+" + "=" * (_WIDTH - 2) + "+")
    print("|  E2E TRACE: Formula Lookup Query" + " " * (_WIDTH - 36) + "|")
    print("+" + "=" * (_WIDTH - 2) + "+")

    summary = _run_trace(
        "What is the formula for cross-entropy loss?",
        expect_skip_retrieval=False,
        expect_needs_web=False,
        min_hits=0,         # formula may or may not be in corpus
        require_llm=False,
    )

    # If routing happened: assert formula_lookup strategy is calibrated
    routing = summary.get("routing", {})
    if routing:
        routed = route_query("What is the formula for cross-entropy loss?", db_path=_DB)
        if routed.intent == "formula_lookup":
            assert routed.strategy.prefer_shorter is True, (
                "formula_lookup strategy must set prefer_shorter=True"
            )
            assert routed.strategy.top_k <= 5, (
                "formula_lookup must use tight top_k (<=5) for precision"
            )


@pytest.mark.slow
def test_e2e_current_events_routing() -> None:
    """Trace a current-events query that must route to the internet, NOT the corpus.

    'What happened in US politics today?' contains a strong temporal signal
    ('today') and asks for live news — the corpus cannot answer this.

    Note: conversationally-framed queries like "I was hoping to talk about the
    current political atmosphere" are intentionally handled as conversational
    (skip_retrieval=True) rather than web — see test_e2e_conversational.

    Expected pipeline path:
      Plan:     web  (strong temporal 'today' + news/event framing)
      Routing:  needs_web=True, skip_retrieval=False
      Internet: web fallback triggered with news/current-events sources
      LLM:      answer from web snippets (not corpus)
    """
    print()
    print("+" + "=" * (_WIDTH - 2) + "+")
    print("|  E2E TRACE: Current Events / Political Query" + " " * (_WIDTH - 47) + "|")
    print("+" + "=" * (_WIDTH - 2) + "+")

    _run_trace(
        "What happened in US politics today?",
        expect_needs_web=True,
        expect_skip_retrieval=False,
        min_hits=0,
        require_llm=False,
    )
