"""Query planner: given a query and force-override flags, decide which data
sources to consult before generating the final answer.

No LLM call — the decision is deterministic from the intent classifier + rules.

Decision priority
-----------------
1. ``force_*`` flags — user explicitly requested a source.
2. ``conversational_meta`` intent → history only (never touch the corpus).
3. ``web_search`` intent → web only.
4. Everything else → RAG (corpus search).

When both force flags are set the order is rag → web so the LLM first has
corpus grounding then sees what the internet says.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict, List, Any, Optional


@dataclass
class PlanStep:
    tool: str    # "rag" | "web" | "history"
    label: str   # human-readable label shown in the UI
    query: str   # query string to pass to this tool


# Icons emitted in the ``plan`` SSE event so the UI can render them.
TOOL_ICON: Dict[str, str] = {
    "rag":     "📚",
    "web":     "🌐",
    "history": "💬",
    "clarify": "❓",
}


@dataclass
class QueryPlan:
    steps: List[PlanStep]
    intent: str
    intent_confidence: float
    meta: Dict[str, Any] = field(default_factory=dict)


_DOC_CUE_PATTERNS: Dict[str, re.Pattern[str]] = {
    "document_ref": re.compile(
        r"\b(?:document|docs?|pdf|file|files|report|spec|specification|rfq|appendix|drawing|drawings)\b",
        re.IGNORECASE,
    ),
    "corpus_scope": re.compile(
        r"\b(?:collection|uploaded files?|uploaded docs?|my files?|my docs?|knowledge base|evidence)\b",
        re.IGNORECASE,
    ),
    "doc_verbs": re.compile(
        r"\b(?:what does (?:the|this|that|my)?\s*(?:document|doc|spec|report|pdf|rfq)\s+say|"
        r"check (?:the|this|that|my)?\s*(?:document|doc|spec|report|pdf|rfq)|"
        r"show me (?:the )?evidence|according to (?:the|this|that|my)?\s*(?:document|doc|spec|report|pdf|rfq))\b",
        re.IGNORECASE,
    ),
}

_SOCIAL_ONLY_RE = re.compile(
    r"^\s*(?:hi|hello|hey|thanks|thank you|thx|ok(?:ay)?|cool|sounds good|goodnight|bye)\W*\s*$",
    re.IGNORECASE,
)

_FACTUALISH_INTENTS = frozenset(
    {
        "metadata_lookup",
        "formula_lookup",
        "section_lookup",
        "comparison",
        "list_lookup",
        "summary",
        "fact_lookup",
        "exploratory",
        "implicit_followup",
    }
)


def _count_recent_sources(prior_sources: Optional[List[str]], source: str) -> int:
    return sum(1 for s in (prior_sources or [])[-3:] if s == source)


def _detect_doc_cues(query: str) -> List[str]:
    q = (query or "").strip()
    hits: List[str] = []
    for label, pattern in _DOC_CUE_PATTERNS.items():
        if pattern.search(q):
            hits.append(label)
    return hits


def _is_chat_only_query(
    query: str,
    *,
    intent: str,
    doc_cues: List[str],
    needs_web: bool,
) -> bool:
    q = (query or "").strip()
    if needs_web or doc_cues:
        return False
    if intent in ("conversational_meta", "user_profile"):
        return True
    return bool(_SOCIAL_ONLY_RE.match(q))


def _score_sources(
    query: str,
    *,
    intent: str,
    needs_web: bool,
    collection_id: Optional[str],
    doc_ids: Optional[List[str]],
    prior_sources: Optional[List[str]],
) -> tuple[Dict[str, int], Dict[str, Any]]:
    doc_cues = _detect_doc_cues(query)
    collection_active = bool(collection_id)
    doc_filter_active = bool(doc_ids)
    rag_momentum = _count_recent_sources(prior_sources, "rag")
    web_momentum = _count_recent_sources(prior_sources, "web")

    scores = {"rag": 0, "web": 0, "chat": 0}

    if collection_active:
        scores["rag"] += 4
    if doc_filter_active:
        scores["rag"] += 5
    if doc_cues:
        scores["rag"] += 4
        scores["chat"] -= 4
    if rag_momentum:
        scores["rag"] += min(4, 1 + rag_momentum)
    if web_momentum and needs_web:
        scores["web"] += min(4, 1 + web_momentum)

    if intent in _FACTUALISH_INTENTS and (collection_active or doc_filter_active or doc_cues or rag_momentum):
        scores["rag"] += 2

    if needs_web:
        scores["web"] += 6
        scores["chat"] -= 3
    if intent == "current_data_lookup":
        scores["web"] += 3

    if intent in ("conversational_meta", "user_profile"):
        scores["chat"] += 6
    elif intent == "conversational":
        scores["chat"] += 4
    if _SOCIAL_ONLY_RE.match(query or ""):
        scores["chat"] += 4

    return scores, {
        "doc_cues": doc_cues,
        "collection_active": collection_active,
        "doc_filter_active": doc_filter_active,
        "rag_momentum": rag_momentum,
        "web_momentum": web_momentum,
    }


def _should_force_web_first(
    *,
    intent: str,
    needs_web: bool,
    doc_cues: List[str],
) -> bool:
    if not needs_web:
        return False
    if doc_cues:
        return False
    return intent == "current_data_lookup" or True


def plan_query(
    query: str,
    *,
    force_rag: bool = False,
    force_web: bool = False,
    history_available: bool = False,
    clarification_pending: bool = False,
    collection_id: Optional[str] = None,
    doc_ids: Optional[List[str]] = None,
    prior_sources: Optional[List[str]] = None,
) -> QueryPlan:
    """Return an ordered list of tool steps to execute for *query*.

    Parameters
    ----------
    query                 : raw user query string
    force_rag             : user toggled "Force RAG" — always search the corpus
    force_web             : user toggled "Force Web" — always search the internet
    history_available     : True when prior conversation messages exist (used for
                            ``conversational_meta`` routing)
    clarification_pending : True when the previous assistant turn was a
                            disambiguation question — the user's reply is an
                            answer to that question, not a new retrieval request.
    collection_id         : currently selected collection, if any
    doc_ids               : active document filters, if any
    prior_sources         : recent resolved source modes ("rag" | "web" | "chat")
    """
    from retrieval.router import classify_intent, _needs_web as _check_needs_web

    intent, meta = classify_intent(query)
    confidence: float = meta.get("ml_confidence", 0.0)
    steps: List[PlanStep] = []

    # If the previous turn was a clarification question, the user is answering
    # it — not issuing a new retrieval request.  Route directly to chat.
    if clarification_pending and not force_rag and not force_web:
        steps.append(PlanStep("history", "Answering from conversation history…", query))
        return QueryPlan(
            steps=steps,
            intent=intent,
            intent_confidence=confidence,
            meta={**meta, "clarification_followup": True},
        )

    if force_rag and force_web:
        # User wants both: search docs first, then web.
        steps.append(PlanStep("rag", "Searching your documents…", query))
        steps.append(PlanStep("web", "Searching the web…",        query))
    elif force_rag:
        steps.append(PlanStep("rag", "Searching your documents…", query))
    elif force_web:
        steps.append(PlanStep("web", "Searching the web…",        query))
    else:
        needs_web, needs_web_reason = _check_needs_web(
            query,
            intent=intent,
            secondary_intent=meta.get("secondary_intent"),
            nli_scores=meta.get("nli_scores"),
        )
        source_scores, source_meta = _score_sources(
            query,
            intent=intent,
            needs_web=needs_web,
            collection_id=collection_id,
            doc_ids=doc_ids,
            prior_sources=prior_sources,
        )
        meta["source_scores"] = source_scores
        meta["source_signals"] = source_meta
        meta["needs_web"] = needs_web
        meta["needs_web_reason"] = needs_web_reason

        if _should_force_web_first(
            intent=intent,
            needs_web=needs_web,
            doc_cues=source_meta["doc_cues"],
        ):
            steps.append(PlanStep("web", "Searching the web…", query))
            meta["selected_source"] = "web"
            meta["web_priority_override"] = True
            return QueryPlan(
                steps=steps,
                intent=intent,
                intent_confidence=confidence,
                meta=meta,
            )

        if _is_chat_only_query(
            query,
            intent=intent,
            doc_cues=source_meta["doc_cues"],
            needs_web=needs_web,
        ):
            steps.append(PlanStep("history", "Answering from conversation history…", query))
            meta["selected_source"] = "chat"
            return QueryPlan(
                steps=steps,
                intent=intent,
                intent_confidence=confidence,
                meta=meta,
            )

        selected_source = "chat"
        if source_scores["rag"] >= source_scores["web"] and source_scores["rag"] >= source_scores["chat"]:
            selected_source = "rag"
        elif source_scores["web"] >= source_scores["chat"]:
            selected_source = "web"

        explicit_doc_context = bool(
            source_meta["collection_active"]
            or source_meta["doc_filter_active"]
            or source_meta["doc_cues"]
            or source_meta["rag_momentum"]
        )

        from retrieval.corpus_scope import is_in_scope
        in_scope, scope_score, matched = is_in_scope(query)
        meta["scope_score"] = round(scope_score, 3)
        meta["scope_matched"] = matched

        if selected_source == "rag" and not in_scope and not explicit_doc_context:
            if needs_web or intent in ("fact_lookup", "current_data_lookup"):
                selected_source = "web"
                meta["scope_veto"] = "no_overlap_without_doc_context"
        elif (
            selected_source == "rag"
            and len(matched) == 1
            and intent in ("exploratory", "summary")
            and not explicit_doc_context
        ):
            # Single ambiguous match on an open-ended or summary query —
            # could be the technical ML concept or something unrelated. Ask first.
            meta["clarify_topic"] = matched[0]
            steps.append(PlanStep("clarify", "Checking what you meant…", query))
            meta["selected_source"] = "clarify"
        else:
            if selected_source == "rag":
                steps.append(PlanStep("rag", "Searching your documents…", query))
            elif selected_source == "web":
                steps.append(PlanStep("web", "Searching the web…", query))
            else:
                steps.append(PlanStep("history", "Answering from conversation history…", query))
            meta["selected_source"] = selected_source

    return QueryPlan(
        steps=steps,
        intent=intent,
        intent_confidence=confidence,
        meta=meta,
    )
