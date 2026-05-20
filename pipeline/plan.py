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
from typing import Dict, List, Any


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


def plan_query(
    query: str,
    *,
    force_rag: bool = False,
    force_web: bool = False,
    history_available: bool = False,
    clarification_pending: bool = False,
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
    elif intent in ("conversational_meta", "user_profile"):
        # These reference the conversation or user context — always answered from
        # history regardless of temporal signals in the query.
        steps.append(PlanStep("history", "Answering from conversation history…", query))
    elif intent == "current_data_lookup" or _check_needs_web(
        query, intent=intent, secondary_intent=meta.get("secondary_intent")
    )[0]:
        # Web wins over conversational too: "hey, what's happening in the markets
        # right now?" may be classified as conversational but still needs live data.
        steps.append(PlanStep("web", "Searching the web…", query))
    elif intent == "conversational":
        steps.append(PlanStep("history", "Answering from conversation history…", query))
    else:
        # Scope check: applied to ALL remaining intents.
        # Even if the classifier is confident about the intent type (e.g.
        # concept_explanation, procedural, comparison), if the query has zero
        # lexical overlap with the corpus there is nothing useful to retrieve.
        # This prevents off-topic messages from reaching RAG regardless of how
        # they are labelled.
        from retrieval.corpus_scope import is_in_scope
        in_scope, scope_score, matched = is_in_scope(query)
        meta["scope_score"] = round(scope_score, 3)
        meta["scope_matched"] = matched
        if not in_scope:
            # No overlap with corpus topics → corpus can't help; use web.
            steps.append(PlanStep("web", "Searching the web…", query))
        elif len(matched) == 1 and intent in ("exploratory", "summary"):
            # Single ambiguous match on an open-ended or summary query —
            # could be the technical ML concept or something unrelated. Ask first.
            meta["clarify_topic"] = matched[0]
            steps.append(PlanStep("clarify", "Checking what you meant…", query))
        elif len(matched) < 2 and intent == "fact_lookup":
            # Single weak match on a factual query → not reliable for RAG.
            steps.append(PlanStep("web", "Searching the web…", query))
        else:
            steps.append(PlanStep("rag", "Searching your documents…", query))

    return QueryPlan(
        steps=steps,
        intent=intent,
        intent_confidence=confidence,
        meta=meta,
    )
