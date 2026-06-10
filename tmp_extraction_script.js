
"use strict";

// â”€â”€ State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
let state = {
  projects: [],       // [{slug, name}]
  current: null,      // active ProjectConfig object
  runResult: null,    // last run result
  latestCheckpointPath: null,
  runAborted: false,
  _sse: null,         // active EventSource or fetch AbortController
};

const GUIDED_REPORT_PRESETS = {
  structured_summary: {
    label: "Structured Summary",
    document_type: "general_document_set",
    sections: [
      "Overview",
      "Most Important Findings",
      "Key Requirements or Obligations",
      "Risks, Gaps, or Uncertainties",
      "Recommended Next Steps",
    ],
  },
  action_checklist: {
    label: "Action Checklist",
    document_type: "general_document_set",
    sections: [
      "Required Items",
      "Submission or Process Requirements",
      "Technical or Content Items to Address",
      "Missing or Unclear Items",
      "Questions Before Completion",
    ],
  },
  risk_issues_review: {
    label: "Risk and Issues Review",
    document_type: "general_document_set",
    sections: [
      "Confirmed Risk Signals",
      "Ambiguities or Unknowns",
      "Operational or Technical Risks",
      "Commercial, Legal, or Compliance Risks",
      "Items Requiring Human Review",
      "Overall Risk Posture",
    ],
  },
  key_concepts_definitions: {
    label: "Key Concepts and Definitions",
    document_type: "general_document_set",
    sections: ["Core Concepts", "Definitions", "Important Relationships or Rules", "Common Misunderstandings or Edge Cases", "Questions for Further Review"],
  },
  requirements_breakdown: {
    label: "Requirements Breakdown",
    document_type: "general_document_set",
    sections: ["Primary Requirements", "Technical or Content Requirements", "Process, Submission, or Timing Requirements", "Qualifications, Dependencies, or Preconditions", "Open Questions or Unclear Requirements"],
  },
  questions_gaps: {
    label: "Questions and Gaps",
    document_type: "general_document_set",
    sections: ["Missing or Incomplete Information", "Ambiguities", "Potential Conflicts or Inconsistencies", "Questions to Resolve", "Recommended Follow-up Actions"],
  },
  decision_support_memo: {
    label: "Decision Support Memo",
    document_type: "general_document_set",
    sections: ["Situation Overview", "Key Decision Factors", "Evidence Supporting Action", "Evidence Raising Concern", "Unknowns or Assumptions Requiring Review", "Recommended Next Decision Steps"],
  },
};

const DOCUMENT_TYPE_LABELS = {
  public_bid_spec: "Public bid / spec package",
  textbook_course_material: "Textbook / course material",
  policy_law_regulation: "Policy / law / regulation",
  technical_manual_sop: "Technical manual / SOP",
  financial_accounting: "Financial / accounting material",
  general_document_set: "General document set",
  other_custom: "Other / custom",
};

const DOMAIN_LENS_LABELS = {
  general: "General",
  engineering_technical: "Engineering / Technical",
  legal_regulatory: "Legal / Regulatory",
  financial_accounting: "Financial / Accounting",
  educational_textbook: "Educational / Textbook",
  business_operations: "Business / Operations",
  healthcare_clinical: "Healthcare / Clinical",
  public_procurement_government: "Public Procurement / Government",
};

const DOMAIN_LENS_GUIDANCE = {
  general: "Keep the report broadly useful and avoid assuming a specialized reader unless the audience or instructions say otherwise.",
  engineering_technical: "Prioritize technical requirements, interfaces, systems, testing, constraints, dependencies, and implementation details.",
  legal_regulatory: "Prioritize obligations, exceptions, deadlines, penalties, compliance language, and items requiring counsel review.",
  financial_accounting: "Prioritize reporting requirements, controls, deadlines, obligations, exceptions, and financially material risks.",
  educational_textbook: "Prioritize definitions, concepts, formulas, examples, likely exam material, and common misunderstandings.",
  business_operations: "Prioritize objectives, deliverables, dependencies, process requirements, decision points, and operational risks.",
  healthcare_clinical: "Prioritize patient-impacting requirements, workflows, compliance duties, safety constraints, and unresolved clinical or operational questions.",
  public_procurement_government: "Prioritize submission requirements, procurement rules, deadlines, evaluation factors, mandatory forms, and clarification needs.",
};

const BRANCH_PACK_SUGGESTIONS = {
  public_bid_spec: ["scope", "requirements", "deadlines", "submission items", "qualifications", "risks", "questions"],
  textbook_course_material: ["definitions", "core concepts", "formulas or rules", "examples", "review topics"],
  policy_law_regulation: ["obligations", "exceptions", "deadlines", "penalties", "ambiguities"],
  technical_manual_sop: ["procedure steps", "inputs and outputs", "warnings", "maintenance", "troubleshooting"],
  financial_accounting: ["reporting requirements", "controls", "deadlines", "exceptions", "financial risks"],
  general_document_set: ["key findings", "requirements", "risks", "questions", "next actions"],
  other_custom: ["key findings", "requirements", "risks", "questions", "next actions"],
};

const DOMAIN_BRANCH_ADDONS = {
  engineering_technical: ["interfaces", "systems", "testing", "constraints"],
  legal_regulatory: ["obligations", "exceptions", "definitions", "penalties"],
  financial_accounting: ["controls", "reporting", "exceptions", "financial risks"],
  educational_textbook: ["definitions", "examples", "review topics", "common mistakes"],
  business_operations: ["deliverables", "dependencies", "decision points", "next actions"],
  healthcare_clinical: ["safety constraints", "workflows", "compliance duties", "open clinical questions"],
  public_procurement_government: ["evaluation factors", "mandatory forms", "clarifications", "procurement rules"],
};

const COMMON_SECTION_SUGGESTIONS = [
  "Overview",
  "Key Findings",
  "Requirements",
  "Risks or Uncertainties",
  "Open Questions",
  "Recommended Next Steps",
];

function normalizeProjectPasses(project) {
  if (!Array.isArray(project.second_passes)) project.second_passes = [];
  project.second_passes = project.second_passes.map(sp => ({
    ...sp,
    enabled: sp?.enabled !== false,
    source_branches: Array.isArray(sp?.source_branches) ? sp.source_branches : [],
  }));
}

const SECOND_PASS_LIBRARY = {
  organize_by_category: { name: "Organize by Category", title: "Organized Evidence by Category" },
  summarize_by_category: { name: "Summarize by Category", title: "Category Summaries" },
  executive_summary: { name: "Executive Summary", title: "Executive Summary" },
  key_findings: { name: "Key Findings", title: "Key Findings" },
  next_actions: { name: "Next Actions", title: "Next Actions" },
  assemble_report: { name: "Assemble Report", title: "Final Report" },
};

function _parseLineOrCommaList(text) {
  return String(text || "")
    .split(/\r?\n|,/)
    .map(part => part.trim())
    .filter(Boolean);
}

async function refreshLatestCheckpointForCurrentProject() {
  const slug = state.current?.slug;
  if (!slug) return;
  try {
    const res = await fetch(`/api/extraction/projects/${encodeURIComponent(slug)}/latest-checkpoint`);
    if (!res.ok) return;
    const data = await res.json();
    state.latestCheckpointPath = data.checkpoint_path || null;
  } catch (_) {}
}

function toggleSidebar() {
  document.getElementById("app").classList.toggle("sidebar-closed");
}

document.addEventListener("DOMContentLoaded", () => {
  if (window.innerWidth <= 700) {
    document.getElementById("app").classList.add("sidebar-closed");
  }
  loadProjectList();
});

function showProjectUI() {
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("tab-bar").style.display = "";
  document.getElementById("panel-setup").style.display = "flex";
  document.getElementById("panel-run").style.display = "none";
  document.getElementById("panel-report").style.display = "none";
  document.getElementById("tab-setup").classList.add("active");
  document.getElementById("tab-run").classList.remove("active");
  document.getElementById("tab-report").classList.remove("active");
}

async function loadProjectList() {
  const container = document.getElementById("project-list");
  try {
    const res = await fetch("/api/extraction/projects");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const projects = await res.json();
    state.projects = Array.isArray(projects) ? projects : [];
    container.innerHTML = "";
    if (!state.projects.length) {
      container.innerHTML = '<div style="color:var(--text-dim);font-size:0.82rem;padding:8px 2px;">No projects yet.</div>';
      return;
    }
    for (const project of state.projects) {
      const row = document.createElement("div");
      row.className = "project-row";

      const item = document.createElement("button");
      item.type = "button";
      item.className = `project-item${state.current?.slug === project.slug ? " active" : ""}`;
      item.style.width = "100%";
      item.style.textAlign = "left";
      item.onclick = () => {
        item.blur();
        loadProject(String(project.slug || ""));
      };

      const nameEl = document.createElement("div");
      nameEl.className = "pname";
      nameEl.textContent = String(project.name || project.slug || "");

      const slugEl = document.createElement("div");
      slugEl.className = "pslug";
      slugEl.textContent = String(project.slug || "");

      item.appendChild(nameEl);
      item.appendChild(slugEl);

      const deleteBtn = document.createElement("button");
      deleteBtn.type = "button";
      deleteBtn.className = "project-delete";
      deleteBtn.textContent = "Delete";
      deleteBtn.onclick = (event) => {
        event.stopPropagation();
        deleteProject(String(project.slug || ""), String(project.name || project.slug || ""));
      };

      row.appendChild(item);
      row.appendChild(deleteBtn);
      container.appendChild(row);
    }
  } catch (e) {
    container.innerHTML = `<div style="color:var(--danger);font-size:0.82rem;padding:8px 2px;">Could not load projects: ${esc(String(e))}</div>`;
  }
}

async function loadProject(slug) {
  try {
    const res = await fetch(`/api/extraction/projects/${encodeURIComponent(slug)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const project = await res.json();
    normalizeProjectPasses(project);
    state.current = project;
    state.runResult = null;
    state.latestCheckpointPath = project.latest_checkpoint_path || null;
    showProjectUI();
    renderProjectForm();
    renderSources();
    renderBranches();
    renderSecondPasses();
    updateRunSummary();
    document.getElementById("report-md").textContent = "";
    document.getElementById("report-md-rendered").textContent = "No report yet. Run extraction first.";
    await loadProjectList();
    await refreshLatestCheckpointForCurrentProject();
  } catch (e) {
    setStatus("err", `Could not load project: ${String(e)}`);
  }
}

// â”€â”€ Tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function showTab(name) {
  ["setup","run","report"].forEach(t => {
    document.getElementById("panel-"+t).style.display = "none";
    document.getElementById("tab-"+t).classList.remove("active");
  });
  document.getElementById("panel-"+name).style.display = "flex";
  document.getElementById("tab-"+name).classList.add("active");
  if (name === "report") refreshReportHistory();
}

// â”€â”€ New project modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function openNewProjectModal() {
  document.getElementById("np-name").value = "";
  document.getElementById("np-slug").value = "";
  openModal("modal-new-project");
  document.getElementById("np-name").oninput = () => {
    const slug = document.getElementById("np-name").value
      .toLowerCase().replace(/[^a-z0-9]+/g,"_").replace(/^_+|_+$/g,"").slice(0,40);
    document.getElementById("np-slug").value = slug;
  };
}

function confirmNewProject() {
  const name = document.getElementById("np-name").value.trim();
  const slug = document.getElementById("np-slug").value.trim();
  if (!name || !slug) { alert("Name and slug are required."); return; }
  state.current = {
    slug, name,
    document_sources: [],
    branches: [],
    second_passes: [],
    collection_id: `extraction_${slug}`,
    keep_collection_after_run: false,
    cross_branch_dedup: false,
    report_output_path: "data/extraction_reports",
  };
  normalizeProjectPasses(state.current);
  state.runResult = null;
  state.latestCheckpointPath = null;
  closeModal("modal-new-project");
  showProjectUI();
  renderProjectForm();
  renderSources();
  renderBranches();
  renderSecondPasses();
  updateRunSummary();
}

// â”€â”€ Project form â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderProjectForm() {
  const p = state.current;
  if (!p) return;
  document.getElementById("proj-name").value = p.name || "";
  document.getElementById("proj-slug").value = p.slug || "";
  document.getElementById("proj-collection").value = p.collection_id || "";
  document.getElementById("proj-keep-collection").checked = !!p.keep_collection_after_run;
  document.getElementById("proj-cross-branch-dedup").checked = !!p.cross_branch_dedup;
}

function readProjectForm() {
  const p = state.current;
  p.name = document.getElementById("proj-name").value.trim() || p.name;
  p.slug = document.getElementById("proj-slug").value.trim() || p.slug;
  p.collection_id = document.getElementById("proj-collection").value.trim() || `extraction_${p.slug}`;
  p.keep_collection_after_run = document.getElementById("proj-keep-collection").checked;
  p.cross_branch_dedup = document.getElementById("proj-cross-branch-dedup").checked;
}

// â”€â”€ Save project â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function saveProject() {
  readProjectForm();
  const p = state.current;
  try {
    const res = await fetch(`/api/extraction/projects/${encodeURIComponent(p.slug)}`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({project: p}),
    });
    if (!res.ok) { const d=await res.json(); alert("Save failed: "+(d.detail||res.status)); return; }
    setStatus("ok", `Project '${p.name}' saved.`);
    await loadProjectList();
  } catch(e) { alert("Error saving: "+e); }
}

// â”€â”€ Delete project â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function deleteProject(slug, name) {
  if (!confirm(`Delete project "${name}"?\nThis cannot be undone.`)) return;
  try {
    const res = await fetch(`/api/extraction/projects/${encodeURIComponent(slug)}`, { method: "DELETE" });
    if (!res.ok) { alert("Delete failed: " + res.status); return; }
    if (state.current?.slug === slug) {
      state.current = null;
      document.getElementById("empty-state").style.display = "";
      document.getElementById("tab-bar").style.display = "none";
      ["setup","run","report"].forEach(t => document.getElementById("panel-"+t).style.display="none");
    }
    await loadProjectList();
    setStatus("ok", `Project '${name}' deleted.`);
  } catch(e) { alert("Error: "+e); }
}

// â”€â”€ Pick collection modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
async function openPickCollectionModal() {
  const container = document.getElementById("pick-collection-list");
  container.innerHTML = '<div style="padding:10px; color:var(--text-dim); font-size:0.8rem;">Loadingâ€¦</div>';
  openModal("modal-pick-collection");
  try {
    const res  = await fetch("/api/collections");
    const cols = await res.json();
    if (!cols.length) {
      container.innerHTML = '<div style="padding:10px; color:var(--text-dim); font-size:0.8rem;">No collections found.</div>';
      return;
    }
    container.innerHTML = cols.map(c => `
      <div style="padding:10px 14px; cursor:pointer; border-bottom:1px solid var(--border);
                  display:flex; align-items:center; gap:10px;"
           onmouseover="this.style.background='#1e2a3a'" onmouseout="this.style.background=''"
           onclick="pickCollection('${esc(c.collection_id)}','${esc(c.name)}')">
        <div style="flex:1;">
          <div style="font-size:0.88rem; font-weight:500;">${esc(c.name)}</div>
          <div style="font-size:0.75rem; color:var(--text-dim);">${esc(c.collection_id)}</div>
        </div>
        <span style="font-size:0.75rem; color:var(--text-dim);">${c.document_count??''} docs</span>
      </div>`).join("");
  } catch(e) {
    container.innerHTML = `<div style="padding:10px; color:var(--danger); font-size:0.8rem;">Error: ${esc(String(e))}</div>`;
  }
}

function pickCollection(colId, name) {
  document.getElementById("proj-collection").value = colId;
  // Also set keep_collection to true â€” we didn't create it, don't destroy it
  document.getElementById("proj-keep-collection").checked = true;
  document.getElementById("proj-cross-branch-dedup").checked = true;
  closeModal("modal-pick-collection");
  setStatus("ok", `Collection set to '${name}' â€” existing chunks will be used directly.`);
}

// â”€â”€ Source documents â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderSources() {
  const ul = document.getElementById("source-list");
  const srcs = state.current?.document_sources || [];
  if (!srcs.length) {
    ul.innerHTML = '<li style="color:var(--text-dim);font-size:0.8rem;">No documents added yet.</li>';
    return;
  }
  ul.innerHTML = srcs.map((s,i) => `
    <li class="source-item">
      <span class="spath">${esc(s.path)}</span>
      <span class="srole">${esc(s.role||"primary")}</span>
      <button class="btn-icon" title="Remove" onclick="removeSource(${i})">âœ•</button>
    </li>`).join("");
}

// â”€â”€ Add document modal â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
let _addDocTab = "browse";
let _addDocSelected = null; // {path, filename} when browse tab has a selection

function setAddDocTab(tab) {
  _addDocTab = tab;
  document.getElementById("add-doc-browse").style.display      = tab==="browse" ? "" : "none";
  document.getElementById("add-doc-path-tab").style.display     = tab==="path"   ? "" : "none";
  document.getElementById("add-doc-tab-browse").classList.toggle("active", tab==="browse");
  document.getElementById("add-doc-tab-path").classList.toggle("active",   tab==="path");
}

async function openAddDocModal() {
  _addDocSelected = null;
  document.getElementById("add-doc-path").value  = "";
  document.getElementById("add-doc-label").value = "";
  document.getElementById("add-doc-role").value  = "primary";
  setAddDocTab("browse");
  openModal("modal-add-doc");
  // Populate collection picker
  try {
    const res = await fetch("/api/collections");
    const cols = await res.json();
    const sel = document.getElementById("add-doc-collection");
    sel.innerHTML = '<option value="">All documents</option>' +
      cols.map(c => `<option value="${esc(c.collection_id)}">${esc(c.name)}</option>`).join("");
  } catch {}
  await loadDocsForCollection();
}

async function loadDocsForCollection() {
  const colId = document.getElementById("add-doc-collection").value;
  const container = document.getElementById("add-doc-doc-list");
  container.innerHTML = '<div style="padding:10px; color:var(--text-dim); font-size:0.8rem;">Loadingâ€¦</div>';
  _addDocSelected = null;
  try {
    const url = colId ? `/api/documents?collection_id=${encodeURIComponent(colId)}` : "/api/documents";
    const res = await fetch(url);
    const docs = await res.json();
    if (!docs.length) {
      container.innerHTML = '<div style="padding:10px; color:var(--text-dim); font-size:0.8rem;">No documents found.</div>';
      return;
    }
    container.innerHTML = docs.map(d => `
      <div class="doc-pick-row" data-path="${esc(d.source_path)}" data-name="${esc(d.filename)}"
           onclick="selectDocRow(this)"
           style="padding:8px 12px; cursor:pointer; border-bottom:1px solid var(--border);
                  display:flex; align-items:center; gap:10px;">
        <span style="flex:1; font-size:0.85rem;">${esc(d.title||d.filename)}</span>
        <span style="font-size:0.75rem; color:var(--text-dim);">${esc(d.source_type||'')}</span>
        <span style="font-size:0.72rem; color:var(--text-dim);">${d.chunk_count||0} chunks</span>
      </div>`).join("");
  } catch(e) {
    container.innerHTML = `<div style="padding:10px; color:var(--danger); font-size:0.8rem;">Error: ${esc(String(e))}</div>`;
  }
}

function selectDocRow(el) {
  document.querySelectorAll(".doc-pick-row").forEach(r => r.style.background = "");
  el.style.background = "#1e2a3a";
  _addDocSelected = { path: el.dataset.path, filename: el.dataset.name };
  if (!document.getElementById("add-doc-label").value)
    document.getElementById("add-doc-label").value = el.dataset.name;
}

function confirmAddDoc() {
  let path, label;
  if (_addDocTab === "browse") {
    if (!_addDocSelected) { alert("Select a document from the list."); return; }
    path  = _addDocSelected.path;
    label = document.getElementById("add-doc-label").value.trim() || _addDocSelected.filename;
  } else {
    path  = document.getElementById("add-doc-path").value.trim();
    label = document.getElementById("add-doc-label").value.trim();
    if (!path) { alert("File path is required."); return; }
  }
  const role = document.getElementById("add-doc-role").value;
  state.current.document_sources = state.current.document_sources || [];
  state.current.document_sources.push({ path, role, label: label || undefined });
  renderSources();
  closeModal("modal-add-doc");
}

function removeSource(i) {
  state.current.document_sources.splice(i,1);
  renderSources();
}

// â”€â”€ Branches â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderBranches() {
  const ul = document.getElementById("branch-list");
  const branches = state.current?.branches || [];
  if (!branches.length) {
    ul.innerHTML = '<li style="color:var(--text-dim);font-size:0.8rem;">No branches configured yet.</li>';
    return;
  }
  ul.innerHTML = branches.map((b,i) => {
    const modeClass = b.mode==="semantic" ? "badge-semantic" : "badge-keyword";
    const detail = b.mode==="keyword"
      ? (b.keywords||[]).slice(0,5).join(", ")
      : (b.topic_description||"").slice(0,60);
    return `
    <li class="branch-card">
      <div class="branch-card-header">
        <span class="bname">${esc(b.name)}</span>
        <span class="branch-mode-badge ${modeClass}">${esc(b.mode)}</span>
        <button class="btn-icon" onclick="editBranch(${i})" title="Edit">Edit</button>
        <button class="btn-icon" onclick="removeBranch(${i})" title="Remove">Remove</button>
        <button class="btn-icon" onclick="saveBranchToLibrary(${i})" title="Save to flag library">Save</button>
      </div>
      <div class="branch-detail">${esc(detail)}</div>
    </li>`;
  }).join("");
}

function openAddBranchModal(editIdx=-1) {
  document.getElementById("add-branch-edit-idx").value = editIdx;
  document.getElementById("add-branch-title").textContent = editIdx>=0 ? "Edit Evidence Category" : "Add Evidence Category";
  const b = editIdx>=0 ? state.current.branches[editIdx] : null;
  document.getElementById("add-branch-name").value    = b?.name || "";
  document.getElementById("add-branch-mode").value    = b?.mode || "keyword";
  document.getElementById("add-branch-keywords").value = (b?.keywords||[]).join(", ");
  document.getElementById("add-branch-topic").value   = b?.topic_description || "";
  document.getElementById("add-branch-heading").value = b?.output_heading || "";
  document.getElementById("add-branch-format").value  = b?.output_format || "bullets";
  document.getElementById("add-branch-max").value     = b?.max_items ?? 200;
  document.getElementById("add-branch-system-override").value = b?.prompt_context || "";
  onBranchModeChange();
  openModal("modal-add-branch");
}

function editBranch(i) { openAddBranchModal(i); }

function onBranchModeChange() {
  const mode = document.getElementById("add-branch-mode").value;
  document.getElementById("branch-keyword-field").style.display  = mode==="keyword"  ? "" : "none";
  document.getElementById("branch-semantic-field").style.display = mode==="semantic" ? "" : "none";
}

function confirmAddBranch() {
  const idx   = parseInt(document.getElementById("add-branch-edit-idx").value, 10);
  const name  = document.getElementById("add-branch-name").value.trim();
  const mode  = document.getElementById("add-branch-mode").value;
  const kwRaw = document.getElementById("add-branch-keywords").value;
  const topic = document.getElementById("add-branch-topic").value.trim();
  const heading = document.getElementById("add-branch-heading").value.trim();
  const format  = document.getElementById("add-branch-format").value;
  const maxItems = parseInt(document.getElementById("add-branch-max").value, 10) || 200;
  const promptContext = document.getElementById("add-branch-system-override").value.trim() || null;
  if (!name) { alert("Branch name is required."); return; }
  const kws = kwRaw.split(",").map(k=>k.trim()).filter(Boolean);
  const branch = {
    name, mode,
    keywords: mode==="keyword" ? kws : [],
    topic_description: mode==="semantic" ? topic : "",
    output_heading: heading || name,
    output_format: format,
    max_items: maxItems,
    prompt_context: promptContext,
    enabled: true,
  };
  if (idx >= 0) state.current.branches[idx] = branch;
  else { state.current.branches = state.current.branches || []; state.current.branches.push(branch); }
  renderBranches();
  closeModal("modal-add-branch");
}

function normalizedTokens(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .map(s => s.trim())
    .filter(Boolean);
}

function categoryMatchesSuggestion(categoryName, suggestion) {
  const cat = normalizedTokens(categoryName);
  const sug = normalizedTokens(suggestion);
  if (!cat.length || !sug.length) return false;
  const catJoined = cat.join(" ");
  const sugJoined = sug.join(" ");
  if (catJoined.includes(sugJoined) || sugJoined.includes(catJoined)) return true;
  return sug.some(tok => cat.includes(tok));
}

function suggestedBranchPack(docType, lens) {
  const base = [...(BRANCH_PACK_SUGGESTIONS[docType] || BRANCH_PACK_SUGGESTIONS.general_document_set)];
  const addon = DOMAIN_BRANCH_ADDONS[lens] || [];
  const seen = new Set();
  const merged = [];
  for (const item of [...base, ...addon]) {
    const key = item.toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      merged.push(item);
    }
  }
  return merged;
}

function removeBranch(i) {
  if (!confirm("Remove this branch?")) return;
  state.current.branches.splice(i,1);
  renderBranches();
}

// â”€â”€ Second passes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function renderSecondPasses() {
  const ul = document.getElementById("second-pass-list");
  if (!ul) return;
  const passes = state.current?.second_passes || [];
  if (!passes.length) {
    ul.innerHTML = '<li style="color:var(--text-dim);font-size:0.8rem;">No second passes configured yet.</li>';
    return;
  }
  ul.innerHTML = passes.map((sp, i) => `
    <li>
      <div class="pass-head">
        <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
          <strong>${esc(sp.title || sp.name)}</strong>
          <span class="badge-pass">${esc((sp.pass_type || "").replaceAll("_", " "))}</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px;">
          <button class="btn-icon" onclick="openSecondPassModal('${esc(sp.pass_type || "")}', ${i})" title="Edit">Edit</button>
          <button class="btn-icon" onclick="removeSecondPass(${i})" title="Remove">Remove</button>
        </div>
      </div>
      <div class="pass-detail">${esc(sp.instructions || "Runs after branch extraction as part of the serial second-pass pipeline.")}</div>
      ${(sp.report_categories || []).length ? `<div class="pass-detail">Report categories: ${esc((sp.report_categories || []).join(", "))}</div>` : ""}
      ${(sp.source_branches || []).length ? `<div class="pass-detail">Source branches: ${esc((sp.source_branches || []).join(", "))}</div>` : ""}
    </li>`).join("");
}

function onSecondPassTypeChange() {
  const passType = document.getElementById("second-pass-type").value;
  const showCategories = passType === "organize_by_category" || passType === "summarize_by_category";
  document.getElementById("second-pass-report-categories-row").style.display = showCategories ? "" : "none";
}

function openSecondPassModal(passType, editIdx = -1) {
  if (!state.current) return;
  const spec = SECOND_PASS_LIBRARY[passType];
  if (!spec) return;
  const existing = editIdx >= 0 ? state.current.second_passes?.[editIdx] : null;
  document.getElementById("second-pass-title").textContent = editIdx >= 0 ? "Edit Second Pass" : "Add Second Pass";
  document.getElementById("second-pass-edit-idx").value = String(editIdx);
  document.getElementById("second-pass-type").value = passType;
  document.getElementById("second-pass-name").value = existing?.name || spec.name;
  document.getElementById("second-pass-heading").value = existing?.title || spec.title;
  document.getElementById("second-pass-source-branches").value = (existing?.source_branches || []).join(", ");
  document.getElementById("second-pass-report-categories").value = (existing?.report_categories || []).join("\n");
  document.getElementById("second-pass-instructions").value = existing?.instructions || "";
  onSecondPassTypeChange();
  openModal("modal-second-pass");
}

function confirmSecondPass() {
  if (!state.current) return;
  const idx = parseInt(document.getElementById("second-pass-edit-idx").value, 10);
  const passType = document.getElementById("second-pass-type").value;
  const name = document.getElementById("second-pass-name").value.trim() || (SECOND_PASS_LIBRARY[passType]?.name || "Second Pass");
  const title = document.getElementById("second-pass-heading").value.trim() || (SECOND_PASS_LIBRARY[passType]?.title || name);
  const sourceBranches = _parseLineOrCommaList(document.getElementById("second-pass-source-branches").value);
  const reportCategories = _parseLineOrCommaList(document.getElementById("second-pass-report-categories").value);
  const instructions = document.getElementById("second-pass-instructions").value.trim();
  const passConfig = {
    name,
    pass_type: passType,
    title,
    enabled: true,
    source_branches: sourceBranches,
    report_categories: reportCategories,
    instructions,
  };
  state.current.second_passes = state.current.second_passes || [];
  if (idx >= 0) state.current.second_passes[idx] = passConfig;
  else state.current.second_passes.push(passConfig);
  renderSecondPasses();
  updateRunSummary();
  closeModal("modal-second-pass");
}

function removeSecondPass(i) {
  if (!state.current?.second_passes) return;
  state.current.second_passes.splice(i, 1);
  renderSecondPasses();
  updateRunSummary();
}

function handleSseEvent(obj) {
  if (obj.type === "progress") {
    const msg = obj.message || "";
    const cls = msg.startsWith("âœ“") ? "ok" : msg.startsWith("âœ—") ? "err"
              : msg.startsWith("Branch") || msg.startsWith("Starting") ? "head" : "";
    appendLog(msg, cls);
    _rsbParse(msg);
    _rsbUpdate();
  } else if (obj.type === "error") {
    appendLog("ERROR: " + obj.message, "err");
    _rs.phase = "error"; _rs.main = obj.message; _rsbUpdate();
    setStatus("err", "Failed");
  } else if (obj.type === "result") {
    state.runResult = obj;
    state.latestCheckpointPath = obj.checkpoint_path || state.latestCheckpointPath;
    renderRunResults(obj);
    document.getElementById("report-md").textContent = obj.report_markdown || "(empty)";
    _renderReport(obj.report_markdown || "(empty)");
    _rs.phase = "done"; _rs.main = `Complete â€” ${obj.elapsed_seconds}s`; _rs.sub = ""; _rsbUpdate();
    setStatus("ok", `Done â€” ${obj.elapsed_seconds}s`);
    appendLog(`\nâœ“ Extraction complete in ${obj.elapsed_seconds}s`, "ok");
    refreshReportHistory();
  }
}

function renderRunResults(result) {
  const el = document.getElementById("run-results");
  const branches   = result.branches || [];
  const secondPasses = result.second_passes || [];
  let html = "<h3 style='margin-bottom:12px; color:var(--text-dim);'>Branches</h3>";
  html += branches.map(b => {
    const icon = b.status==="ok"?"âœ…":b.status==="empty"?"âš ï¸":"âŒ";
    return `<div class="branch-result-block">
      <h3>${icon} ${esc(b.name)} <span style="font-weight:normal;color:var(--text-dim)">(${b.items} items)</span></h3>
    </div>`;
  }).join("");
  if (secondPasses.length) {
    html += "<h3 style='margin-bottom:12px; margin-top:16px; color:var(--text-dim);'>Second Passes</h3>";
    html += secondPasses.filter(pp => pp.pass_name !== "report_plan").map(pp => {
      const icon = pp.status==="ok" ? "âœ…" : "âŒ";
      const detail = pp.status==="ok" ? `${(pp.response_text||'').length} chars` : (pp.error||'error');
      return `<div class="branch-result-block">
        <h3>${icon} ${esc(pp.pass_name)} <span style="font-weight:normal;color:var(--text-dim)">(${detail})</span></h3>
      </div>`;
    }).join("");
  }
  el.innerHTML = html;
}

function abortRun() {
  state.runAborted = true;
  if (state._sse) { state._sse.abort(); state._sse = null; }
  appendLog("Aborted by user.", "err");
  setStatus("err", "Aborted");
  document.getElementById("btn-run").disabled = false;
  document.getElementById("btn-abort").style.display = "none";
}

// â”€â”€ Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function _renderReport(text) {
  document.getElementById("report-md").textContent = text;
  const rendered = document.getElementById("report-md-rendered");
  if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
    const normalised = text.replace(/([^\n])([ \t]*\n?[ \t]*)(#{1,6} )/g, "$1\n\n$3");
    rendered.innerHTML = DOMPurify.sanitize(marked.parse(normalised));
  } else {
    rendered.textContent = text;
  }
}
function copyReport() {
  const text = document.getElementById("report-md").textContent;
  navigator.clipboard.writeText(text).then(()=>setStatus("ok","Copied!"));
}

function downloadReport() {
  const slug = state.current?.slug || "report";
  const text = document.getElementById("report-md").textContent;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], {type:"text/markdown"}));
  a.download = `${slug}_report.md`;
  a.click();
}

async function refreshReportHistory() {
  try {
    const resp = await fetch("/api/extraction/reports");
    if (!resp.ok) return;
    const list = await resp.json();
    const sel = document.getElementById("report-history-select");
    const cur = sel.value;
    // Keep the placeholder, rebuild options
    while (sel.options.length > 1) sel.remove(1);
    list.forEach(r => {
      const opt = document.createElement("option");
      opt.value = r.filename;
      // Format label: "slug_YYYYMMDD_HHMMSS.md" â†’ "slug Â· YYYY-MM-DD HH:MM"
      const m = r.filename.match(/^(.+?)_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.md$/);
      opt.textContent = m
        ? `${m[1]} Â· ${m[2]}-${m[3]}-${m[4]} ${m[5]}:${m[6]}`
        : r.filename.replace(/\.md$/, "");
      sel.appendChild(opt);
    });
    if (cur) sel.value = cur;
  } catch (_) {}
}

async function loadHistoryReport(filename) {
  if (!filename) return;
  try {
    const resp = await fetch(`/api/extraction/report-file/${encodeURIComponent(filename)}`);
    if (!resp.ok) { setStatus("err", "Could not load report"); return; }
    const text = await resp.text();
    _renderReport(text);
  } catch (e) {
    setStatus("err", "Failed to load report");
  }
}

// â”€â”€ Status bar â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function setStatus(state_, msg) {
  const dot  = document.getElementById("status-dot");
  const text = document.getElementById("status-text");
  dot.className  = "status-dot " + (state_==="running"?"running":state_==="ok"?"ok":state_==="err"?"err":"");
  text.textContent = msg;
}

// â”€â”€ Modal helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function openModal(id)  { document.getElementById(id).classList.add("open"); }
function closeModal(id) { document.getElementById(id).classList.remove("open"); }

// â”€â”€ Utils â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
function esc(s) {
  return String(s||"").replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

