"""Prompt templates for RAG answering."""

from __future__ import annotations

from typing import Any, Dict, List

from llm.coverage import is_overview_query


def build_system_prompt(prompt_config: Dict[str, Any] | None = None, source_mode: str = "corpus") -> str:
	cfg = prompt_config or {}
	require_inline = bool(cfg.get("require_inline_citations", True))

	if source_mode == "web":
		citation_rule = (
			"Place [cNNNNNN] citation tags after claims drawn from the web results. "
			"Do not cite general knowledge sentences."
			if require_inline else
			"Use citations where possible for web-sourced claims."
		)
		return (
			"You are a knowledgeable colleague who has just retrieved live web search results to answer a question. "
			"Your job is to synthesize those results into a clear, direct answer — "
			"not to summarize each source one by one, but to extract the relevant facts and state them plainly.\n\n"
			"Treat the web results as supporting evidence. Prioritize specific facts, numbers, and named sources over vague generalities. "
			"If results conflict, note the conflict briefly rather than picking one silently. "
			"If results are too vague or off-topic to answer well, say so in one sentence — then give your best answer from general knowledge.\n\n"
			"Never fabricate URLs, publication dates, or statistics not present in the provided results. "
			"Do not open your response by quoting the raw search result text — synthesize it. "
		f"{citation_rule}\n\n"
		"FORMATTING RULES — follow these for every response:\n"
		"• Start with a concise direct answer (1–2 sentences).\n"
		"• For anything longer than two paragraphs, break up the substance into bullet points — "
		"avoid walls of unbroken prose.\n"
		"• **Bold** key terms; use `code` for names and formulas.\n"
		"• Keep paragraphs short (2–3 sentences max)."
		)

	if source_mode == "general":
		return (
			"You are a knowledgeable colleague answering from your own expertise — "
			"no documents or search results are available for this question.\n\n"
			"Answer directly and confidently from general knowledge. "
			"Be honest about uncertainty — 'I think' or 'typically' are fine, fabricating specifics is not. "
			"Use precise technical terminology — do not paraphrase standard terms (e.g. say 'learning rate', 'true positive', 'overfitting', not vague synonyms). "
			"Match the depth of the question: 2–3 sentences for simple facts, 2–3 paragraphs for conceptual questions. "
			"Use markdown where it aids clarity: **bold** key terms, bullet lists for genuinely list-like content, "
			"`code` for variable names and formulas. Plain prose is fine when structure adds nothing."
		)

	# Default: corpus RAG
	citation_rule = (
		"Place [cNNNNNN] citation tags at the end of sentences or claims that draw directly from the provided context — "
		"after the full claim, not mid-sentence. Do not cite general-knowledge sentences."
		if require_inline else
		"Use citations where possible for context-sourced claims."
	)
	return (
		"You are a knowledgeable colleague answering questions from a personal document corpus. "
		"You are technically fluent, direct, and precise — matching the depth of the subject matter: "
		"machine learning, finance, reinforcement learning, or whatever domain is in the documents.\n\n"
		"Synthesize your answer from the provided context chunks — do not open your response by quoting or "
		"copying the raw chunk text verbatim. Use the context as evidence to construct your own explanation. "
		"Preserve exact technical terms, formula notation, and variable names as they appear in the source.\n\n"
		"When context supports the answer, use it and cite your sources. "
		"When context only partially supports the answer, cover what it does support and briefly note what it doesn't. "
		"When the retrieved context does not cover the question at all, answer it yourself from your own knowledge. "
		"Prefix that answer with '⚠️ GENERAL KNOWLEDGE:' to signal to the user it comes from your knowledge, not the documents. "
		"You MUST still provide a real answer after the prefix — do not just say 'the context doesn't cover this' and stop. "
		"The ⚠️ prefix is informational, not an apology.\n\n"
		"Never invent citations, entities, or numbers not present in context. "
		f"{citation_rule}\n\n"
		"FORMATTING RULES — follow these for every response:\n"
		"• Start with a concise direct answer (1–2 sentences).\n"
		"• For anything longer than two paragraphs, break up the substance into bullet points "
		"or a numbered list — never write 3+ consecutive sentences on the same theme as a wall of prose.\n"
		"• **Bold** the first occurrence of each key technical term.\n"
		"• Use `code` for variable names, formula symbols, and function names.\n"
		"• Use ## section headers only when the answer has two or more clearly distinct major sections.\n"
		"• Keep paragraphs short (2–3 sentences max)."
	)


def format_context_blocks(
	hits: List[Dict[str, Any]],
	max_chunks: int = 6,
	max_chars_per_chunk: int = 1600,
	include_retrieval_score: bool = True,
	include_document_path: bool = False,
	include_neighbor_context: bool = True,
	max_neighbors_per_chunk: int = 2,
	max_chars_per_neighbor: int = 280,
) -> str:
	parts: List[str] = []
	for h in hits[:max_chunks]:
		chunk_id = h.get("chunk_id", "unknown")
		path = h.get("path_text") or ""
		pages = f"{h.get('page_start')}-{h.get('page_end')}"
		score = h.get("score")
		collection_id = h.get("collection_id") or (h.get("metadata") or {}).get("collection_id") or ""
		source_name = h.get("source_name") or (h.get("metadata") or {}).get("source_name") or ""
		document_title = h.get("document_title") or (h.get("metadata") or {}).get("document_title") or ""
		document_path = h.get("document_path") or (h.get("metadata") or {}).get("document_path") or ""
		page_number = h.get("page_number") or (h.get("metadata") or {}).get("page_number")
		section_header = h.get("section_header") or (h.get("metadata") or {}).get("section_header") or ""
		registry = ((h.get("metadata") or {}).get("document_registry") or {})
		text = (h.get("text") or "").strip()
		if len(text) > max_chars_per_chunk:
			text = text[:max_chars_per_chunk].rstrip() + " ..."
		score_line = f"Score: {float(score):.4f}" if include_retrieval_score and isinstance(score, (int, float)) else None
		lines = [
			f"[CHUNK {chunk_id}]",
			f"Collection: {collection_id}",
			f"Source: {source_name}",
			f"Document Title: {document_title}",
			f"Path: {path}",
			f"Section Header: {section_header}",
			f"Page Number: {page_number}",
			f"Pages: {pages}",
		]
		if include_document_path:
			lines.insert(4, f"Document Path: {document_path}")
		if score_line:
			lines.append(score_line)
		registry_parts = []
		for key, label in (("title", "Title"), ("authors", "Authors"), ("publisher", "Publisher"), ("year", "Year")):
			value = registry.get(key)
			if value:
				registry_parts.append(f"{label}: {value}")
		if registry_parts:
			lines.append("Book Metadata: " + " | ".join(registry_parts))
		lines.append(text)

		if include_neighbor_context:
			neighbors = ((h.get("metadata") or {}).get("neighbors") or [])[:max_neighbors_per_chunk]
			for n in neighbors:
				nid = n.get("chunk_id", "unknown")
				ntext = (n.get("text") or "").strip()
				if not ntext:
					continue
				if len(ntext) > max_chars_per_neighbor:
					ntext = ntext[:max_chars_per_neighbor].rstrip() + " ..."
				lines.append(f"Neighbor [{nid}]: {ntext}")
		parts.append(
			"\n".join(lines)
		)
	return "\n\n".join(parts)


# Per-intent style notes.  These override the generic confidence-band style
# for medium/high bands.  "low" confidence always keeps its trust-signal prefix.
_INTENT_STYLE_NOTES: Dict[str, str] = {
	"formula_lookup": (
		"Lead with the formula or equation exactly as it appears in context — "
		"use inline code notation (e.g. `E = mc²`, `loss = -Σ y log ŷ`). "
		"Preserve every symbol, subscript, and variable name verbatim. "
		"Follow with a 1–2 sentence plain-language explanation of what each term means. "
		"No verbose prose preamble before the formula."
	),
	"list_lookup": (
		"Answer as a concise bullet list — no prose preamble. "
		"One item per line. "
		"Cite with [cNNNNNN] at the end of items drawn directly from context. "
		"If the list is short (≤5 items), a tight paragraph is also acceptable."
	),
	"comparison": (
		"Structure your answer as a markdown comparison table when comparing two or more distinct items. "
		"Format: | Feature | Item A | Item B | with a header row and separator row (| --- | --- | --- |). "
		"Choose 4–6 of the most meaningful features to compare — not exhaustive, just the key differentiators. "
		"Follow the table with one short paragraph (2–3 sentences) stating when to prefer each option. "
		"If the comparison is nuanced or the items are asymmetric, use brief labeled sections instead of a table. "
		"Do not pad with background that does not bear on the contrast."
	),
}


def build_user_prompt(
	query: str,
	hits: List[Dict[str, Any]],
	prompt_config: Dict[str, Any] | None = None,
	*,
	confidence_band: str = "medium",
	evidence_facts: str | None = None,
	history: List[Dict[str, str]] | None = None,
	source_mode: str = "corpus",
	intent: str | None = None,
) -> str:
	cfg = prompt_config or {}
	max_chunks = int(cfg.get("max_chunks", 6))
	max_chars_per_chunk = int(cfg.get("max_chars_per_chunk", 1600))
	min_citations = int(cfg.get("min_citations", 2))
	max_citations = int(cfg.get("max_citations", 3))
	include_score = bool(cfg.get("include_retrieval_score", True))
	include_document_path = bool(cfg.get("include_document_path", False))
	include_neighbors = bool(cfg.get("include_neighbor_context", True))
	max_neighbors_per_chunk = int(cfg.get("max_neighbors_per_chunk", 2))
	max_chars_per_neighbor = int(cfg.get("max_chars_per_neighbor", 280))

	overview = is_overview_query(query, intent=intent)

	context = format_context_blocks(
		hits,
		max_chunks=max_chunks,
		max_chars_per_chunk=max_chars_per_chunk,
		include_retrieval_score=include_score,
		include_document_path=include_document_path,
		include_neighbor_context=include_neighbors,
		max_neighbors_per_chunk=max_neighbors_per_chunk,
		max_chars_per_neighbor=max_chars_per_neighbor,
	)
	if overview:
		# Overview always wins — it's a structural decision, not just a style.
		style_note = (
			"Write a flowing, well-organized overview using markdown formatting. "
			"Use ## section headings to break the answer into logical parts, "
			"and **bold** key terms on first use. "
			"Synthesize information from all provided context into coherent paragraphs — "
			"do not copy verbatim lists; convert them to prose. "
			"Place 1–2 citation tags at the end of each paragraph, not after every sentence."
		)
	elif confidence_band == "low":
		# Low confidence keeps its trust-signal prefix regardless of intent.
		style_note = (
			"Provide a best-effort answer from direct evidence only, 1–3 sentences. "
			"Start with 'Low confidence:' and include a brief note for unsupported parts."
		)
	elif intent is not None and intent in _INTENT_STYLE_NOTES:
		# Intent-specific style overrides the generic medium/high confidence style.
		style_note = _INTENT_STYLE_NOTES[intent]
	elif confidence_band == "high":
		style_note = (
			"Answer in 2–4 sentences. Use exact numbers, formulas, technical terms, and notation "
			"as they appear in the context — do not paraphrase precise factual content. "
			"Use **bold** for key terms or named values when it aids scannability. "
			"Write in your own words; do not open by quoting the raw context text."
		)
	else:
		style_note = (
			"Be direct and grounded — write as a colleague explaining to a peer. "
			"Start with a 1–2 sentence direct answer. "
			"For any answer covering more than one main point, use bullet points or a numbered list — "
			"do not write 3+ consecutive sentences on the same theme as unbroken prose. "
			"**Bold** key technical terms on first use. "
			"Use `code` for variable names, formula symbols, and function names. "
			"Keep paragraphs to 2–3 sentences max. "
			"Use numbered steps for sequential processes."
		)

	evidence_block = ""
	if evidence_facts and not overview:
		evidence_block = (
			f"Key facts to include:\n{evidence_facts}\n\n"
			"Incorporate these facts into your answer. "
			"Preserve exact numbers, units, currency amounts, formulas, and quoted terms exactly as written — "
			"do not substitute equivalent forms (e.g., write '$1,000' not 'thousands of dollars', "
			"write '252' not 'approximately 250'). "
			"Write the surrounding prose in your own words.\n\n"
		)

	# Prepend conversation history for follow-up/pronoun resolution.
	# Capped at last 10 messages (~5 turns), 500 chars each to stay token-efficient.
	history_block = ""
	if history:
		lines = ["Prior conversation (for follow-up context only — do not cite as a source):"]
		for msg in history[-10:]:
			role_label = "User" if msg.get("role") == "user" else "Assistant"
			content = (msg.get("content") or "")[:500]
			lines.append(f"{role_label}: {content}")
		history_block = "\n".join(lines) + "\n\n"

	# ── General knowledge mode: no context chunks, no citation machinery ──────
	if source_mode == "general":
		return (
			history_block
			+ f"Question:\n{query}\n\n"
			"Answer from your own knowledge. Be direct and honest about any uncertainty."
		)

	# ── Web search mode ───────────────────────────────────────────────────────
	if source_mode == "web":
		context = format_context_blocks(
			hits,
			max_chunks=max_chunks,
			max_chars_per_chunk=max_chars_per_chunk,
			include_retrieval_score=include_score,
			include_document_path=False,
			include_neighbor_context=False,
		)
		return (
			history_block
			+ f"Question:\n{query}\n\n"
			f"Web search results:\n{context}\n\n"
			"Synthesize a clear answer from the web results above. "
			"Extract the key facts — do not summarize each source separately. "
			"If the results don't fully answer the question, say so briefly then give your best answer. "
			f"Include {min_citations}–{max_citations} [cNNNNNN] citations for web-sourced claims."
		)

	# ── Corpus RAG mode (default) ─────────────────────────────────────────────
	return (
		history_block
		+ "Answer the question below using the provided context.\n\n"
		f"Question:\n{query}\n\n"
		f"{evidence_block}"
		f"Context:\n{context}\n\n"
		f"Confidence band: {confidence_band}\n\n"
		f"Style: {style_note}\n\n"
		"Citation rules: "
		+ (
			"Place [cNNNNNN] tags at the end of each paragraph (1–2 per paragraph). "
			"Do not interrupt prose with mid-sentence citation tags.\n"
			if overview else
			"Place [cNNNNNN] tags at the end of sentences or claims that draw from context. "
			f"Include {min_citations}–{max_citations} citations total. "
			"A single tag can cover a group of closely related claims in the same sentence — "
			"you do not need a separate tag after every clause.\n"
		)
		+ (
			"Only use the ⚠️ GENERAL KNOWLEDGE prefix if the context chunks contain no relevant information "
			"for the question at all — not as a hedge for your own uncertainty. "
			"Do not invent facts or refuse when relevant context exists."
		)
	)
