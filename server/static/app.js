/* ═══════════════════════════════════════════════════════════════════════
   Personal AI — App logic
   Handles: mode switching, collection loading, sending messages,
   rendering responses, citations panel, history, markdown rendering.
   ═══════════════════════════════════════════════════════════════════════ */

"use strict";

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  force_rag: false,           // force corpus search regardless of intent
  force_web: false,           // force web search regardless of intent
  collection_id: null,        // filter: selected collection or null
  top_k: 5,
  doc_ids: [],                // filter: restrict to specific documents
  history: [],                // [{role, content}, …]  local mirror for quick context
  isLoading: false,
  pendingCitations: [],       // citations from latest response
  conversationId: null,       // persistent conversation ID (server-side)
  priorIntents: [],           // rolling list of recent intent labels (last 6)
  priorSources: [],           // rolling list of resolved sources (rag/web/chat)
  clarificationPending: false, // true when last assistant turn was a clarify question
};

// ── DOM refs ──────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const messages        = $("messages");
const inputBox        = $("inputBox");
const sendBtn         = $("sendBtn");
const collectionSel   = $("collectionSelect");
const topKInput       = $("topKInput");
const forceRagBtn     = $("forceRagBtn");
const forceWebBtn     = $("forceWebBtn");
const filtersToggle   = $("filtersToggle");
const filtersBody     = $("filtersBody");
const topbarTitle     = $("topbarTitle");
const citationsPanel  = $("citationsPanel");
const citationsList   = $("citationsList");
const closeCitations  = $("closeCitations");
const statusDot       = $("statusDot");
const sidebarToggle   = $("sidebarToggle");
const sidebarRailToggle = $("sidebarRailToggle");
const sidebar         = $("sidebar");
const newChatBtn      = $("newChatBtn");
const historyList     = $("historyList");
const welcomeSubtitle = $("welcomeSubtitle");
const welcomeChips    = $("welcomeChips");
const welcome         = $("welcome");
const docFilterList    = $("docFilterList");
const docFilterCount   = $("docFilterCount");
const docFilterClear   = $("docFilterClear");
const docSearch        = $("docSearch");

function _setOverlayState(className, isOpen) {
  document.body.classList.toggle(className, !!isOpen);
}

// ── Utilities ─────────────────────────────────────────────────────────────

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// Configure marked once (GFM + line breaks).
// Falls back to escapeHtml() if marked/DOMPurify are not loaded (e.g. offline
// with CDN-only build), so the app never hard-crashes.
(function _setupMarked() {
  if (typeof marked !== "undefined") {
    marked.setOptions({ gfm: true, breaks: true });
  }
})();

function renderMarkdown(text) {
  if (!text) return "";
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    // Normalise: ensure ATX headings (## …) always start on their own line.
    // LLMs sometimes emit them mid-paragraph without a preceding blank line,
    // which marked.js then treats as inline text rather than a heading block.
    const normalised = text
      .replace(/([^\n])([ \t]*\n?[ \t]*)(#{1,6} )/g, "$1\n\n$3");
    return DOMPurify.sanitize(marked.parse(normalised));
  }
  // Minimal safe fallback (plain text with line-break preservation)
  return `<p>${escapeHtml(text).replace(/\n{2,}/g, "</p><p>").replace(/\n/g, "<br>")}</p>`;
}

function formatElapsed(sec) {
  if (sec < 1) return `${Math.round(sec * 1000)}ms`;
  return `${sec.toFixed(1)}s`;
}

function showToast(msg) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  clearTimeout(toast._tid);
  toast._tid = setTimeout(() => toast.classList.remove("show"), 4000);
}

// ── Server health check ────────────────────────────────────────────────────

async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    if (res.ok) {
      statusDot.className = "status-dot ok";
      statusDot.title = "Server connected";
    } else {
      statusDot.className = "status-dot error";
    }
  } catch {
    statusDot.className = "status-dot error";
    statusDot.title = "Server unreachable";
  }
}

// ── Document filter ────────────────────────────────────────────────────────

async function loadDocuments(collectionId) {
  try {
    const url = collectionId
      ? `/api/documents?collection_id=${encodeURIComponent(collectionId)}`
      : "/api/documents";
    const res = await fetch(url);
    if (!res.ok) return;
    const docs = await res.json();

    // Reset selections when collection changes
    state.doc_ids = [];
    _updateDocFilterUI();

    docFilterList.innerHTML = "";
    docs.forEach(doc => {
      const label = document.createElement("label");
      label.className = "doc-filter-item";
      const title = doc.title || doc.filename;
      label.innerHTML = `
        <input type="checkbox" value="${escapeHtml(doc.doc_id)}" />
        <span class="doc-title" title="${escapeHtml(doc.filename)}">${escapeHtml(title)}</span>
        <span class="doc-chunks">${doc.chunk_count != null ? doc.chunk_count.toLocaleString() : ""}</span>
      `;
      label.querySelector("input").addEventListener("change", _onDocCheckboxChange);
      docFilterList.appendChild(label);
    });
  } catch {
    // silently ignore
  }
}

function _onDocCheckboxChange() {
  state.doc_ids = Array.from(
    docFilterList.querySelectorAll("input[type='checkbox']:checked")
  ).map(el => el.value);
  _updateDocFilterUI();
}

function _updateDocFilterUI() {
  const n = state.doc_ids.length;
  docFilterCount.textContent = n > 0 ? `${n} selected` : "";
  docFilterClear.style.display = n > 0 ? "block" : "none";
}

docFilterClear.addEventListener("click", () => {
  docFilterList.querySelectorAll("input[type='checkbox']").forEach(el => { el.checked = false; });
  state.doc_ids = [];
  _updateDocFilterUI();
});

docSearch.addEventListener("input", () => {
  const q = docSearch.value.toLowerCase();
  docFilterList.querySelectorAll(".doc-filter-item").forEach(item => {
    const t = item.querySelector(".doc-title").textContent.toLowerCase();
    item.style.display = t.includes(q) ? "" : "none";
  });
});

// ── Collections ────────────────────────────────────────────────────────────

async function loadCollections() {
  try {
    const res = await fetch("/api/collections");
    if (!res.ok) return;
    const cols = await res.json();
    collectionSel.innerHTML = '<option value="">All documents</option>';
    cols.forEach(c => {
      const indent = c.parent_id ? "\u00a0\u00a0\u00a0└ " : "";
      const label = `${indent}${c.name} (${c.chunk_count} chunks)`;
      const opt = document.createElement("option");
      opt.value = c.collection_id;
      opt.textContent = label;
      collectionSel.appendChild(opt);
    });
  } catch {
    // silently ignore — server may not be fully up yet
  }
}

// ── Mode switching ─────────────────────────────────────────────────────────

// ── Message rendering ──────────────────────────────────────────────────────

function hideWelcome() {
  if (welcome && welcome.parentElement) {
    welcome.remove();
  }
}

// ── Feedback (thumbs up / down) ────────────────────────────────────────────

function _makeFeedbackRow(traceId, query, answer) {
  const row = document.createElement("div");
  row.className = "feedback-row";

  const up = document.createElement("button");
  up.className = "feedback-btn";
  up.title = "Good answer";
  up.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>`;

  const down = document.createElement("button");
  down.className = "feedback-btn";
  down.title = "Bad answer";
  down.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg>`;

  const label = document.createElement("span");
  label.className = "feedback-label";

  async function sendFeedback(rating) {
    up.disabled = true;
    down.disabled = true;
    (rating === 1 ? up : down).classList.add("feedback-active");
    label.textContent = rating === 1 ? "Thanks!" : "Noted";
    try {
      await fetch("/api/feedback", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trace_id: traceId,
          rating,
          query,
          answer: (answer || "").slice(0, 200),
        }),
      });
    } catch (_) {}
  }

  up.addEventListener("click",   () => sendFeedback(1));
  down.addEventListener("click", () => sendFeedback(-1));

  row.appendChild(up);
  row.appendChild(down);
  row.appendChild(label);
  return row;
}

function appendMessage(role, content, meta = {}) {
  hideWelcome();
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  if (role === "ai" || role === "system") {
    bubble.innerHTML = renderMarkdown(content);
  } else {
    bubble.innerHTML = `<p>${escapeHtml(content).replace(/\n/g, "<br>")}</p>`;
  }

  row.appendChild(bubble);

  // Meta row for AI messages
  if (role === "ai" && (meta.elapsed || meta.mode || meta.citations_count)) {
    const metaRow = document.createElement("div");
    metaRow.className = "msg-meta";

    if (meta.mode) {
      const tag = document.createElement("span");
      tag.className = `meta-tag ${meta.mode}`;
      tag.textContent = meta.mode.toUpperCase();
      metaRow.appendChild(tag);
    }
    if (meta.internet_used) {
      const tag = document.createElement("span");
      tag.className = "meta-tag web";
      tag.textContent = "WEB";
      metaRow.appendChild(tag);
    }
    if (meta.citations_count > 0) {
      const btn = document.createElement("button");
      btn.className = "citations-btn";
      btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>${meta.citations_count} source${meta.citations_count !== 1 ? "s" : ""}`;
      btn.addEventListener("click", () => showCitations(meta.citations));
      metaRow.appendChild(btn);
    }
    if (meta.elapsed) {
      const t = document.createElement("span");
      t.textContent = formatElapsed(meta.elapsed);
      metaRow.appendChild(t);
    }

    row.appendChild(metaRow);

    // Feedback buttons (only when a trace_id is available)
    if (meta.trace_id) {
      row.appendChild(_makeFeedbackRow(meta.trace_id, meta.query || "", content));
    }
  }

  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function appendTypingIndicator() {
  const row = document.createElement("div");
  row.className = "msg-row ai";
  row.id = "typing";
  row.innerHTML = `
    <div class="msg-bubble">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function removeTypingIndicator() {
  const el = $("typing");
  if (el) el.remove();
}

// ── Citations panel ────────────────────────────────────────────────────────

function appendWebSources(sources) {
  if (!sources || sources.length === 0) return;
  const row = document.createElement("div");
  row.className = "msg-row ai";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";

  const header = document.createElement("div");
  header.style.cssText = "font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;color:var(--text-dim);margin-bottom:8px;display:flex;align-items:center;gap:6px;";
  header.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>Web sources`;
  bubble.appendChild(header);

  const list = document.createElement("div");
  list.className = "web-sources";
  sources.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "web-source-card";
    card.innerHTML = `
      <div class="web-source-title">[${i + 1}] ${escapeHtml(s.title || s.url)}</div>
      <div class="web-source-url">${escapeHtml(s.url)}</div>
      ${s.snippet ? `<div class="web-source-snippet">${escapeHtml(s.snippet)}</div>` : ""}
    `;
    list.appendChild(card);
  });
  bubble.appendChild(list);
  row.appendChild(bubble);
  messages.appendChild(row);
  messages.scrollTop = messages.scrollHeight;
}

function showCitations(citations) {
  if (!citations || citations.length === 0) return;

  citationsList.innerHTML = "";
  citations.forEach(c => {
    const card = document.createElement("div");
    card.className = `citation-card${c.cited ? " cited" : ""}`;

    const sourceType = c.source_type === "internet" ? "web" : (c.source_type || "pdf");
    const pageStr   = c.page ? ` · p.${c.page}` : "";
    const scoreStr  = c.score ? ` · ${c.score}` : "";

    card.innerHTML = `
      <div class="citation-top">
        <span class="citation-source" title="${escapeHtml(c.source)}">${escapeHtml(c.source)}</span>
        <div class="citation-badges">
          ${c.cited ? '<span class="citation-badge cited-badge">cited</span>' : ""}
          ${c.source_type === "internet" ? '<span class="citation-badge internet-badge">web</span>' : ""}
          <span class="citation-badge">${sourceType}${pageStr}${scoreStr}</span>
        </div>
      </div>
      <p class="citation-snippet">${escapeHtml(c.snippet)}</p>
    `;
    citationsList.appendChild(card);
  });

  citationsPanel.style.display = "flex";
  citationsPanel.style.flexDirection = "column";
  _setOverlayState("citations-open", true);
  messages.scrollTop = messages.scrollHeight;
}

closeCitations.addEventListener("click", () => {
  citationsPanel.style.display = "none";
  _setOverlayState("citations-open", false);
});

// ── Send message ───────────────────────────────────────────────────────────

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || state.isLoading) return;

  // Lazily create a server-side conversation on first send
  if (!state.conversationId) await createNewConversation();

  state.isLoading = true;
  inputBox.value = "";
  autoResize();
  sendBtn.disabled = true;

  // Render user message
  appendMessage("user", trimmed);

  // Build history snapshot for chat mode (before adding this message)
  // (kept for potential non-streaming path; server manages DB history)

  // Show typing indicator
  appendTypingIndicator();

  const payload = {
    message: trimmed,
    force_rag: state.force_rag,
    force_web: state.force_web,
    collection_id: state.collection_id || null,
    doc_ids: state.doc_ids.length ? state.doc_ids : undefined,
    history: [],   // server manages history — send empty to avoid duplication
    top_k: state.top_k,
    stream: true,
    conversation_id: state.conversationId || null,
    prior_intents: state.priorIntents.length ? state.priorIntents : undefined,
    prior_sources: state.priorSources.length ? state.priorSources : undefined,
    clarification_pending: state.clarificationPending || undefined,
  };

  let streamBubble = null;
  let streamRow = null;
  let accumulated = "";
  let lastStageMsg = "";   // tracks the most recent status/plan text for the stage caption

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      removeTypingIndicator();
      const err = await res.text();
      appendMessage("system", `Error ${res.status}: ${err}`);
      state.isLoading = false;
      sendBtn.disabled = false;
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // keep potentially incomplete trailing chunk

      for (const part of parts) {
        const dataLine = part.split("\n").find(l => l.startsWith("data: "));
        if (!dataLine) continue;
        let event;
        try { event = JSON.parse(dataLine.slice(6)); } catch { continue; }

        if (event.type === "plan") {
          const typingEl = $("typing");
          if (typingEl && event.steps) {
            const summary = event.steps.map(s => `${s.icon || ""} ${s.label}`).join(" → ");
            lastStageMsg = summary;
            typingEl.querySelector(".msg-bubble").innerHTML =
              `<p style="color:var(--text-muted);font-size:13px;">${escapeHtml(summary)}</p>`;
          }
        } else if (event.type === "status") {
          lastStageMsg = event.message || "";
          const typingEl = $("typing");
          if (typingEl) {
            typingEl.querySelector(".msg-bubble").innerHTML =
              `<p style="color:var(--text-muted);font-size:13px;">${escapeHtml(event.message)}</p>`;
          }
          // If streaming has already started, update the stage caption
          const captionEl = document.getElementById("stream-stage-caption");
          if (captionEl) captionEl.textContent = event.message;

        } else if (event.type === "token") {
          if (!streamBubble) {
            removeTypingIndicator();
            hideWelcome();
            streamRow = document.createElement("div");
            streamRow.className = "msg-row ai";
            streamBubble = document.createElement("div");
            streamBubble.className = "msg-bubble";
            // Show the last known pipeline stage as a small caption above the text
            if (lastStageMsg) {
              const caption = document.createElement("p");
              caption.id = "stream-stage-caption";
              caption.className = "stream-stage-caption";
              caption.textContent = lastStageMsg;
              streamBubble.appendChild(caption);
            }
            streamRow.appendChild(streamBubble);
            messages.appendChild(streamRow);
          }
          accumulated += event.content;
          // Update only the text node (not the caption) for performance
          let textNode = streamBubble._textNode;
          if (!textNode) {
            textNode = document.createTextNode("");
            streamBubble._textNode = textNode;
            streamBubble.appendChild(textNode);
          }
          textNode.textContent = accumulated;
          messages.scrollTop = messages.scrollHeight;

        } else if (event.type === "done") {
          removeTypingIndicator();
          const answerText = event.answer || "(no response)";
          const citations  = event.citations || [];
          const webSources = event.web_sources || [];

          if (streamBubble) {
            // Finalize: upgrade plain text to markdown
            streamBubble.innerHTML = renderMarkdown(answerText);
            // Add meta row
            const metaRow = document.createElement("div");
            metaRow.className = "msg-meta";
            if (event.mode) {
              const tag = document.createElement("span");
              tag.className = `meta-tag ${event.mode}`;
              tag.textContent = event.mode.toUpperCase();
              metaRow.appendChild(tag);
            }
            if (event.internet_used) {
              const tag = document.createElement("span");
              tag.className = "meta-tag web";
              tag.textContent = "WEB";
              metaRow.appendChild(tag);
            }
            if (citations.length > 0) {
              const btn = document.createElement("button");
              btn.className = "citations-btn";
              btn.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>${citations.length} source${citations.length !== 1 ? "s" : ""}`;
              btn.addEventListener("click", () => showCitations(citations));
              metaRow.appendChild(btn);
            }
            const t = document.createElement("span");
            t.textContent = formatElapsed(event.elapsed_seconds);
            metaRow.appendChild(t);
            streamRow.appendChild(metaRow);

            // Feedback buttons for streamed answer
            if (event.trace_id) {
              streamRow.appendChild(_makeFeedbackRow(event.trace_id, trimmed, answerText));
            }

          } else {
            // RAG path: no tokens were streamed — full answer arrived in done event
            appendMessage("ai", answerText, {
              mode: event.mode,
              elapsed: event.elapsed_seconds,
              internet_used: event.internet_used,
              citations_count: citations.length,
              citations,
              trace_id: event.trace_id || null,
              query: trimmed,
            });
          }

          if (webSources.length > 0) appendWebSources(webSources);

          // Update local history mirror for quick context (both modes)
          state.history.push({ role: "user",      content: trimmed });
          state.history.push({ role: "assistant", content: answerText });
          if (state.history.length > 20) state.history.splice(0, 2);

          // Track intent for carry-forward routing on the next turn
          if (event.intent) {
            state.priorIntents.push(event.intent);
            if (state.priorIntents.length > 6) state.priorIntents.shift();
          }

          const resolvedSource = event.mode === "rag"
            ? "rag"
            : (event.internet_used || webSources.length > 0 ? "web" : "chat");
          state.priorSources.push(resolvedSource);
          if (state.priorSources.length > 6) state.priorSources.shift();

          // Track whether last turn was a clarification question
          state.clarificationPending = event.clarification_asked === true;

          // Update conversation ID from server (handles first-message creation)
          if (event.conversation_id) {
            state.conversationId = event.conversation_id;
            localStorage.setItem("rag_conv_id", event.conversation_id);
          }

          if (citations.length > 0) {
            showCitations(citations);
          }

          await loadConversationList();
          break outer;

        } else if (event.type === "error") {
          removeTypingIndicator();
          appendMessage("system", event.message || "An error occurred.");
          break outer;
        }
      }
    }

  } catch (err) {
    removeTypingIndicator();
    if (accumulated && streamBubble) {
      // Partial content was streamed — finalize what we have and show an error note
      streamBubble.innerHTML = renderMarkdown(accumulated);
      const errNote = document.createElement("p");
      errNote.className = "stream-error-note";
      errNote.textContent = "⚠ Stream interrupted — response may be incomplete.";
      streamBubble.appendChild(errNote);
    } else {
      appendMessage("system", "Connection lost — is the server running?");
    }
    showToast(err.message || "Network error");
  }

  state.isLoading = false;
  sendBtn.disabled = inputBox.value.trim().length === 0;
  inputBox.focus();
}

// ── Conversation management ────────────────────────────────────────────────

async function createNewConversation() {
  try {
    const res = await fetch("/api/conversations", { method: "POST" });
    if (!res.ok) return;
    const data = await res.json();
    state.conversationId = data.conversation_id;
    localStorage.setItem("rag_conv_id", state.conversationId);
    state.priorIntents = [];
    state.priorSources = [];
    state.clarificationPending = false;
  } catch (e) {
    state.conversationId = null;
  }
}

async function loadConversation(conversationId) {
  try {
    const res = await fetch(`/api/conversations/${conversationId}`);
    if (!res.ok) return false;
    const conv = await res.json();
    messages.innerHTML = "";
    state.conversationId = conversationId;
    state.history = [];
    state.priorIntents = [];
    state.priorSources = [];
    state.clarificationPending = false;
    localStorage.setItem("rag_conv_id", conversationId);

    if (conv.summary) {
      const note = document.createElement("div");
      note.className = "history-summary-note";
      note.textContent = "↑ Earlier messages summarized";
      messages.appendChild(note);
    }

    for (const msg of (conv.messages || [])) {
      if (msg.role === "user") {
        appendMessage("user", msg.content);
      } else {
        appendMessage("ai", msg.content, { mode: msg.mode });
        const restoredSource = msg.mode === "rag" ? "rag" : "chat";
        state.priorSources.push(restoredSource);
        if (state.priorSources.length > 6) state.priorSources.shift();
      }
      state.history.push({ role: msg.role === "user" ? "user" : "assistant", content: msg.content });
      if (state.history.length > 20) state.history.splice(0, 2);
    }

    // Hide welcome screen if we have messages
    if (conv.messages && conv.messages.length > 0) hideWelcome();
    return true;
  } catch (e) {
    return false;
  }
}

async function loadConversationList() {
  try {
    const res = await fetch("/api/conversations");
    if (!res.ok) return;
    const convs = await res.json();
    historyList.innerHTML = "";
    if (convs.length === 0) {
      historyList.innerHTML = '<p class="history-empty">No conversations yet</p>';
      return;
    }
    for (const conv of convs) {
      const item = document.createElement("div");
      item.className = "history-item" + (conv.conversation_id === state.conversationId ? " active" : "");
      item.dataset.convId = conv.conversation_id;
      const titleSpan = document.createElement("span");
      titleSpan.className = "history-title";
      titleSpan.textContent = conv.title || "Untitled";
      titleSpan.title = conv.title || "";
      const delBtn = document.createElement("button");
      delBtn.className = "history-delete";
      delBtn.title = "Delete conversation";
      delBtn.textContent = "×";
      delBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        await fetch(`/api/conversations/${conv.conversation_id}`, { method: "DELETE" });
        if (conv.conversation_id === state.conversationId) {
          await startNewChat();
        }
        await loadConversationList();
      });
      titleSpan.addEventListener("click", () => switchConversation(conv.conversation_id));
      item.appendChild(titleSpan);
      item.appendChild(delBtn);
      historyList.appendChild(item);
    }
  } catch (e) {
    // silently fail
  }
}

async function switchConversation(conversationId) {
  if (conversationId === state.conversationId) return;
  const ok = await loadConversation(conversationId);
  if (ok) {
    if (window.innerWidth <= 680) {
      sidebar.classList.remove("mobile-open");
    }
    citationsPanel.style.display = "none";
    _setOverlayState("citations-open", false);
    await loadConversationList();
  }
}

// ── New chat ───────────────────────────────────────────────────────────────

async function startNewChat() {
  state.history = [];
  state.priorIntents = [];
  state.priorSources = [];
  state.clarificationPending = false;
  messages.innerHTML = "";
  state.conversationId = null;
  localStorage.removeItem("rag_conv_id");
  if (window.innerWidth <= 680) {
    sidebar.classList.remove("mobile-open");
  }

  // Re-insert welcome
  const w = document.createElement("div");
  w.id = "welcome";
  w.className = "welcome";
  w.innerHTML = `
    <div class="welcome-icon">
      <svg width="48" height="48" viewBox="0 0 28 28" fill="none">
        <circle cx="14" cy="14" r="13" stroke="url(#wg2)" stroke-width="1.5"/>
        <circle cx="14" cy="14" r="6" fill="url(#wg2)" opacity="0.9"/>
        <defs>
          <linearGradient id="wg2" x1="0" y1="0" x2="28" y2="28" gradientUnits="userSpaceOnUse">
            <stop stop-color="#818cf8"/><stop offset="1" stop-color="#38bdf8"/>
          </linearGradient>
        </defs>
      </svg>
    </div>
    <h1 class="welcome-title">Personal AI</h1>
    <p class="welcome-subtitle" id="welcomeSubtitle">Ask anything — I’ll search your notes, browse the web, or answer from general knowledge.</p>
    <div class="welcome-chips" id="welcomeChips"></div>
  `;
  messages.appendChild(w);

  window._welcomeRef = w;
  await loadWelcomeChips();

  citationsPanel.style.display = "none";
  _setOverlayState("citations-open", false);
  inputBox.focus();

  await loadConversationList();
}

newChatBtn.addEventListener("click", () => startNewChat());

// ── Sidebar toggle ─────────────────────────────────────────────────────────

function handleSidebarToggle(e) {
  e.stopPropagation();
  const isMobile = window.innerWidth <= 680;
  if (isMobile) {
    sidebar.classList.toggle("mobile-open");
  } else {
    sidebar.classList.toggle("collapsed");
  }
}

sidebarToggle.addEventListener("click", handleSidebarToggle);
if (sidebarRailToggle) sidebarRailToggle.addEventListener("click", handleSidebarToggle);

// Close mobile sidebar when clicking outside
document.addEventListener("click", e => {
  const clickedToggle = e.target.closest("#sidebarToggle, #sidebarRailToggle");
  if (window.innerWidth <= 680
      && sidebar.classList.contains("mobile-open")
      && !sidebar.contains(e.target)
      && !clickedToggle) {
    sidebar.classList.remove("mobile-open");
  }
});

// ── Input handling ─────────────────────────────────────────────────────────

function autoResize() {
  inputBox.style.height = "auto";
  inputBox.style.height = Math.min(inputBox.scrollHeight, 160) + "px";
}

inputBox.addEventListener("input", () => {
  autoResize();
  sendBtn.disabled = inputBox.value.trim().length === 0 || state.isLoading;
});

inputBox.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) sendMessage(inputBox.value);
  }
});

sendBtn.addEventListener("click", () => sendMessage(inputBox.value));

// ── Force mode buttons ────────────────────────────────────────────────

function setForceMode(forceRag, forceWeb) {
  state.force_rag = forceRag;
  state.force_web = forceWeb;
  forceRagBtn.classList.toggle("active", forceRag);
  forceWebBtn.classList.toggle("active", forceWeb);
  topbarTitle.textContent = forceRag ? "Personal AI · RAG" : forceWeb ? "Personal AI · Web" : "Personal AI";
}

forceRagBtn.addEventListener("click", () => setForceMode(!state.force_rag, false));
forceWebBtn.addEventListener("click", () => setForceMode(false, !state.force_web));

// ── Filters collapse ───────────────────────────────────────────────

filtersToggle.addEventListener("click", () => {
  const expanded = filtersToggle.getAttribute("aria-expanded") === "true";
  filtersToggle.setAttribute("aria-expanded", String(!expanded));
  filtersBody.classList.toggle("collapsed", expanded);
  const chevron = filtersToggle.querySelector(".collapse-chevron");
  if (chevron) chevron.style.transform = expanded ? "" : "rotate(180deg)";
});

// ── Collection / top-k controls ────────────────────────────────────────

collectionSel.addEventListener("change", () => {
  state.collection_id = collectionSel.value || null;
  loadDocuments(state.collection_id);
});

topKInput.addEventListener("change", () => {
  const v = parseInt(topKInput.value, 10);
  if (v >= 1 && v <= 20) state.top_k = v;
});

// ── Chip handlers ──────────────────────────────────────────────────────────

function attachChipHandlers() {
  document.querySelectorAll(".chip").forEach(btn => {
    btn.addEventListener("click", () => {
      const prompt = btn.dataset.prompt;
      if (prompt) sendMessage(prompt);
    });
  });
}

// ── Init ───────────────────────────────────────────────────────────────────

async function loadWelcomeChips() {
  const container = document.getElementById("welcomeChips");
  if (!container) return;
  try {
    const resp = await fetch("/api/welcome-chips");
    if (!resp.ok) return;
    const data = await resp.json();
    const chips = (data.chips || []);
    container.innerHTML = chips.map(c =>
      `<button class="chip" data-prompt="${escapeAttr(c.prompt)}">${escapeHtml(c.label)}</button>`
    ).join("");
    attachChipHandlers();
  } catch (_) {}
}

function escapeHtml(str) {
  return String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function escapeAttr(str) {
  return String(str).replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;");
}

(async function init() {
  await checkHealth();
  await loadCollections();
  await loadDocuments(null);
  await loadWelcomeChips();
  inputBox.focus();

  // Restore last conversation if available (lazy: no server create until first send)
  const storedId = localStorage.getItem("rag_conv_id");
  if (storedId) {
    await loadConversation(storedId);
    // if load fails, state.conversationId stays null → created on first send
  }
  await loadConversationList();

  // Ping health every 30s
  setInterval(checkHealth, 30_000);
})();

// ── Personal context / settings modal ─────────────────────────────────────

function switchSettingsTab(name) {
  ["personal", "inference"].forEach(t => {
    document.getElementById(`stab-${t}`).classList.toggle("active", t === name);
    document.getElementById(`spanel-${t}`).style.display = t === name ? "" : "none";
  });
  if (name === "inference") loadInfsvcStatus();
}

const settingsBtn     = $("settingsBtn");
const settingsOverlay = $("settingsOverlay");
const settingsClose   = $("settingsClose");
const settingsForm    = $("settingsForm");
const settingsClear   = $("settingsClear");
const ctxName         = $("ctxName");
const ctxRole         = $("ctxRole");
const ctxProjects     = $("ctxProjects");
const ctxTech         = $("ctxTech");
const ctxNotes        = $("ctxNotes");
const ctxSystemPrompt = $("ctxSystemPrompt");

async function openSettings() {
  try {
    const res = await fetch("/api/context");
    if (res.ok) {
      const ctx = await res.json();
      ctxName.value     = ctx.name     || "";
      ctxRole.value     = ctx.role     || "";
      ctxProjects.value = ctx.current_projects || "";
      ctxTech.value     = ctx.technologies || "";
      ctxNotes.value        = ctx.notes         || "";
      ctxSystemPrompt.value = ctx.system_prompt   || "";
    }
  } catch { /* ignore */ }
  settingsOverlay.style.display = "flex";
  _setOverlayState("modal-open", true);
  ctxName.focus();
}

function closeSettings() {
  settingsOverlay.style.display = "none";
  _setOverlayState("modal-open", false);
}

settingsBtn.addEventListener("click", openSettings);
settingsClose.addEventListener("click", closeSettings);
settingsOverlay.addEventListener("click", e => {
  if (e.target === settingsOverlay) closeSettings();
});
document.addEventListener("keydown", e => {
  if (e.key === "Escape" && settingsOverlay.style.display !== "none") closeSettings();
  if (e.key === "Escape" && citationsPanel.style.display !== "none") {
    citationsPanel.style.display = "none";
    _setOverlayState("citations-open", false);
  }
});

settingsClear.addEventListener("click", async () => {
  if (!confirm("Clear all personal context?")) return;
  try {
    await fetch("/api/context", { method: "PUT", headers: { "Content-Type": "application/json" }, body: "{}" });
    ctxName.value = ctxRole.value = ctxProjects.value = ctxTech.value = ctxNotes.value = ctxSystemPrompt.value = "";
    showToast("Context cleared.");
    closeSettings();
  } catch { showToast("Failed to clear context."); }
});

settingsForm.addEventListener("submit", async e => {
  e.preventDefault();
  const body = {
    name:             ctxName.value.trim(),
    role:             ctxRole.value.trim(),
    current_projects: ctxProjects.value.trim(),
    technologies:     ctxTech.value.trim(),
    notes:            ctxNotes.value.trim(),
      system_prompt:    ctxSystemPrompt.value.trim(),
  };
  // Strip empty keys so we don't store blanks
  Object.keys(body).forEach(k => { if (!body[k]) delete body[k]; });
  try {
    const res = await fetch("/api/context", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (res.ok || res.status === 204) {
      showToast("Context saved.");
      closeSettings();
    } else {
      showToast("Failed to save context.");
    }
  } catch { showToast("Network error saving context."); }
});

// ── Inference Service settings ─────────────────────────────────────────────

const infsvcDot         = $("infsvcDot");
const infsvcStatusText  = $("infsvcStatusText");
const infsvcCapsRow     = $("infsvcCapsRow");
const infsvcUrlDisplay  = $("infsvcUrlDisplay");
const infsvcRefreshBtn  = $("infsvcRefreshBtn");

async function loadInfsvcStatus() {
  infsvcDot.className       = "infsvc-dot checking";
  infsvcStatusText.textContent = "Checking…";
  try {
    const resp = await fetch("/api/inference-service/status");
    const d    = await resp.json();
    if (infsvcUrlDisplay) infsvcUrlDisplay.textContent = d.url || "—";
    if (d.connected) {
      infsvcDot.className       = "infsvc-dot ok";
      infsvcStatusText.textContent = `Connected — ${d.url}`;
      const eps    = d.capabilities && d.capabilities.endpoints;
      const ollama = d.capabilities && d.capabilities.ollama;
      const parts  = [];
      if (eps) {
        parts.push(...Object.entries(eps)
          .map(([k, v]) => `<td class="infsvc-cap-key">${k}</td><td class="infsvc-cap-val">${v.model || "?"}</td><td class="infsvc-cap-status">${v.loaded ? "✓" : "loading…"}</td>`));
      }
      if (ollama && ollama.pinned_model) {
        parts.push(`<td class="infsvc-cap-key">ollama</td><td class="infsvc-cap-val">${ollama.pinned_model}</td><td class="infsvc-cap-status">pinned</td>`);
      }
      if (parts.length) {
        infsvcCapsRow.innerHTML    = `<table class="infsvc-caps-table">${parts.map(p => `<tr>${p}</tr>`).join("")}</table>`;
        document.getElementById("infsvcCapsRowWrapper").style.display = "";
      } else {
        document.getElementById("infsvcCapsRowWrapper").style.display = "none";
      }
    } else {
      infsvcDot.className       = "infsvc-dot err";
      infsvcStatusText.textContent = d.url
        ? `Unreachable — ${d.url}`
        : "Not configured — using local models";
      document.getElementById("infsvcCapsRowWrapper").style.display = "none";
    }
  } catch (e) {
    infsvcDot.className       = "infsvc-dot err";
    infsvcStatusText.textContent = "Error checking status";
    infsvcCapsRow.style.display = "none";
  }
}

infsvcRefreshBtn.addEventListener("click", loadInfsvcStatus);

// Inference status loads when the Inference Service tab is opened (switchSettingsTab)
