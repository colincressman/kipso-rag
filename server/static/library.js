/* library.js — Collection Management Dashboard */

// ── Stable DOM references (cached once so grid.innerHTML never loses them) ──
const _emptyEl = document.getElementById('docGridEmpty');

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  collections: [],          // flat list from /api/collections
  selectedId: '__unassigned__',  // currently viewed collection
  docs: [],                 // docs in current view
  draggingDocId: null,      // doc_id being dragged
  pendingJobs: {},          // jobId -> { filename, status, timerId }
  uploadDraft: null,        // { filename, collectionId, startedAt }
  confirmCallback: null,    // pending confirm modal callback
};

const PENDING_UPLOADS_KEY = 'rag.library.pendingUploads';
const UPLOAD_DRAFT_KEY = 'rag.library.uploadDraft';

function _serializePendingJobs() {
  const out = {};
  for (const [jobId, job] of Object.entries(state.pendingJobs || {})) {
    out[jobId] = {
      filename: job.filename,
      collectionId: job.collectionId || null,
      status: job.status || 'running',
      error: job.error || null,
      destPath: job.destPath || null,
      queuePosition: Number.isFinite(job.queuePosition) ? job.queuePosition : null,
      isActive: !!job.isActive,
      isWaiting: !!job.isWaiting,
      gpuHolder: job.gpuHolder || null,
      stage: job.stage || null,
      stageDetail: job.stageDetail || null,
    };
  }
  return out;
}

function _persistPendingJobs() {
  try {
    localStorage.setItem(PENDING_UPLOADS_KEY, JSON.stringify(_serializePendingJobs()));
  } catch {}
}

function _persistUploadDraft() {
  try {
    if (state.uploadDraft) {
      localStorage.setItem(UPLOAD_DRAFT_KEY, JSON.stringify(state.uploadDraft));
    } else {
      localStorage.removeItem(UPLOAD_DRAFT_KEY);
    }
  } catch {}
}

function _restorePendingJobs() {
  try {
    const raw = localStorage.getItem(PENDING_UPLOADS_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    state.pendingJobs = parsed;
  } catch {
    state.pendingJobs = {};
  }
}

function _restoreUploadDraft() {
  try {
    const raw = localStorage.getItem(UPLOAD_DRAFT_KEY);
    if (!raw) return;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== 'object') return;
    state.uploadDraft = parsed;
  } catch {
    state.uploadDraft = null;
  }
}

async function _hydratePendingJobsFromServer() {
  try {
    const jobs = await api('GET', '/api/upload-active');
    if (!Array.isArray(jobs)) return;
    const activeJobIds = new Set();
    for (const job of jobs) {
      if (!job || !job.job_id) continue;
      activeJobIds.add(job.job_id);
      const existing = state.pendingJobs[job.job_id] || {};
      state.pendingJobs[job.job_id] = {
        ...existing,
        filename: job.filename || existing.filename || 'upload',
        collectionId: job.collection_id || existing.collectionId || null,
        status: job.status || existing.status || 'running',
        error: job.error || existing.error || null,
        destPath: job.dest_path || existing.destPath || null,
        queuePosition: Number.isFinite(job.queue_position) ? job.queue_position : existing.queuePosition,
        isActive: !!job.is_active,
        isWaiting: !!job.is_waiting,
        gpuHolder: job.gpu_holder || existing.gpuHolder || null,
        stage: job.stage || existing.stage || null,
        stageDetail: job.stage_detail || existing.stageDetail || null,
        timerId: existing.timerId,
      };
    }
    for (const jobId of Object.keys(state.pendingJobs || {})) {
      if (activeJobIds.has(jobId)) continue;
      const existing = state.pendingJobs[jobId];
      if (!existing || existing.status === 'error') continue;
      try {
        const status = await api('GET', `/api/upload/${jobId}`);
        state.pendingJobs[jobId] = {
          ...existing,
          status: status.status || existing.status || 'error',
          error: status.error || existing.error || null,
          destPath: status.dest_path || existing.destPath || null,
          queuePosition: Number.isFinite(status.queue_position) ? status.queue_position : existing.queuePosition,
          isActive: !!status.is_active,
          isWaiting: !!status.is_waiting,
          gpuHolder: status.gpu_holder || existing.gpuHolder || null,
          stage: status.stage || existing.stage || null,
          stageDetail: status.stage_detail || existing.stageDetail || null,
          timerId: existing.timerId,
        };
      } catch {
        existing.status = 'error';
        existing.error = 'Job not found on server';
        existing.stage = 'interrupted';
        existing.stageDetail = 'Server no longer has this upload job';
      }
    }
    if (state.uploadDraft && Array.isArray(jobs) && jobs.some(job => {
      if (!job) return false;
      const sameName = (job.filename || '') === (state.uploadDraft.filename || '');
      const sameCollection = (job.collection_id || null) === (state.uploadDraft.collectionId || null);
      return sameName && sameCollection;
    })) {
      state.uploadDraft = null;
    }
    _persistUploadDraft();
    _persistPendingJobs();
  } catch {
    for (const job of Object.values(state.pendingJobs || {})) {
      if (!job || job.status === 'error') continue;
      job.status = 'error';
      job.error = 'Lost connection to server';
      job.stage = 'disconnected';
      job.stageDetail = 'Server unavailable';
    }
    _persistPendingJobs();
  }
}

// ── API helpers ────────────────────────────────────────────────────────────
async function api(method, url, body) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(url, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try { detail = (await res.json()).detail || detail; } catch {}
    throw new Error(detail);
  }
  if (res.status === 204) return null;
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('application/json')) return res.json();
  return null;
}

// ── Load / refresh ─────────────────────────────────────────────────────────
async function loadAll() {
  await _hydratePendingJobsFromServer();
  await loadCollections();
  await loadDocs(state.selectedId);
  _resumePendingJobPollers();
}

async function loadCollections() {
  try {
    state.collections = await api('GET', '/api/collections');
  } catch (e) {
    state.collections = [];
    showToast('Failed to load collections: ' + e.message, 'error');
  }
  renderTree();
  populateCollectionSelects();
}

async function loadDocs(collectionId) {
  state.selectedId = collectionId;
  updateTreeSelection();
  try {
    if (collectionId === '__unassigned__') {
      state.docs = await api('GET', '/api/documents?unassigned=true');
    } else {
      state.docs = await api('GET', `/api/documents?collection_id=${encodeURIComponent(collectionId)}`);
    }
  } catch (e) {
    state.docs = [];
    showToast('Failed to load documents: ' + e.message, 'error');
  }
  renderDocGrid();
  updateViewHeader();
}

// ── Tree rendering ─────────────────────────────────────────────────────────
function buildTree(collections) {
  const map = {};
  const roots = [];
  for (const c of collections) {
    map[c.collection_id] = { ...c, children: [] };
  }
  for (const c of collections) {
    if (c.parent_id && map[c.parent_id]) {
      map[c.parent_id].children.push(map[c.collection_id]);
    } else if (!c.parent_id) {
      roots.push(map[c.collection_id]);
    }
  }
  return roots;
}

function folderIcon(open) {
  return `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
  </svg>`;
}

function renderTreeNode(node, depth) {
  const isSelected = state.selectedId === node.collection_id;
  const hasChildren = node.children && node.children.length > 0;
  const selClass = isSelected ? ' selected' : '';
  const domId = 'tree-' + node.collection_id.replace(/[^a-zA-Z0-9]/g, '_');

  let html = `
    <div class="tree-item${selClass}" data-id="${esc(node.collection_id)}"
         onclick="selectCollection(this)"
         ondragover="onTreeDragOver(event)"
         ondragleave="onTreeDragLeave(event)"
         ondrop="onTreeDrop(event, '${esc(node.collection_id)}')">
      <span class="tree-icon">${folderIcon(false)}</span>
      <span class="tree-label" title="${esc(node.collection_id)}">${esc(node.name)}</span>
      <span class="tree-badge">${node.doc_count || 0}</span>
      <button class="tree-delete-btn" onclick="confirmDeleteCollection(event, '${esc(node.collection_id)}')" title="Delete collection">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
        </svg>
      </button>
    </div>`;

  if (hasChildren) {
    html += `<div class="tree-children">`;
    for (const child of node.children) {
      html += renderTreeNode(child, depth + 1);
    }
    html += `</div>`;
  }
  return html;
}

function renderTree() {
  const container = document.getElementById('collectionTree');
  const roots = buildTree(state.collections);

  if (roots.length === 0) {
    container.innerHTML = `<p style="padding:8px 8px;font-size:12px;color:var(--text-dim)">No collections yet.</p>`;
  } else {
    container.innerHTML = roots.map(n => renderTreeNode(n, 0)).join('');
  }

  // Update unassigned badge
  const unassignedCount = state.selectedId === '__unassigned__' ? state.docs.length : '?';
  if (state.selectedId === '__unassigned__') {
    document.getElementById('unassignedBadge').textContent = state.docs.length;
  }
}

function updateTreeSelection() {
  document.querySelectorAll('.tree-item').forEach(el => {
    el.classList.toggle('selected', el.dataset.id === state.selectedId);
  });
}

function selectCollection(el) {
  const id = el.dataset.id;
  loadDocs(id);
}

// ── Doc grid rendering ─────────────────────────────────────────────────────
function getExt(filename) {
  return (filename || '').split('.').pop().toLowerCase();
}

function extIcon(ext) {
  const icons = {
    pdf: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
      <line x1="9" y1="15" x2="15" y2="15"/><line x1="9" y1="11" x2="15" y2="11"/>
    </svg>`,
    docx: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/>
      <line x1="9" y1="15" x2="15" y2="15"/>
    </svg>`,
  };
  return icons[ext] || icons.docx;
}

function renderDocGrid() {
  const grid = document.getElementById('docGrid');
  const filterEl = document.getElementById('docFilter');
  const filterText = (filterEl ? filterEl.value : '').trim().toLowerCase();

  const pendingHtml = Object.keys(state.pendingJobs)
    .map(jid => _renderPendingCard(jid))
    .join('');
  const draftHtml = state.uploadDraft ? _renderUploadDraftCard() : '';

  const visibleDocs = (state.docs || []).filter(doc => {
    if (!filterText) return true;
    const name = (doc.document_title || doc.title || doc.filename || '').toLowerCase();
    return name.includes(filterText);
  });

  if (visibleDocs.length === 0) {
    grid.innerHTML = draftHtml + pendingHtml;
    if (!draftHtml && !pendingHtml) {
      _emptyEl.style.display = '';
      grid.appendChild(_emptyEl);
    }
    return;
  }
  // Remove empty placeholder before overwriting innerHTML
  if (_emptyEl.parentNode === grid) _emptyEl.remove();

  grid.innerHTML = draftHtml + pendingHtml + visibleDocs.map(doc => {
    const ext = getExt(doc.filename);
    // Prefer document_title (rich title) > title > filename
    const displayName = doc.document_title || doc.title || doc.filename;
    return `
      <div class="doc-card" draggable="true" data-doc-id="${esc(doc.doc_id)}"
           ondragstart="onDocDragStart(event, '${esc(doc.doc_id)}')"
           ondragend="onDocDragEnd(event)"
           onclick="onDocCardTap(event, '${esc(doc.doc_id)}')">
        <div class="doc-card-icon ${ext}">${extIcon(ext)}</div>
        <div class="doc-card-title" title="${esc(doc.filename)}">${esc(displayName)}</div>
        <div class="doc-card-meta">
          <span class="doc-badge doc-badge--${ext}">${ext.toUpperCase()}</span>
          <span class="doc-badge doc-badge--chunks">${doc.chunk_count} chunks</span>
        </div>
        <button class="doc-card-delete"
                onclick="confirmDeleteDocument(event, '${esc(doc.doc_id)}', '${esc(doc.filename)}')"
                title="Remove from library">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            <path d="M10 11v6"/><path d="M14 11v6"/>
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
          </svg>
        </button>
        <a class="doc-card-open" href="/api/documents/${esc(doc.doc_id)}/file"
           target="_blank" title="Open / download file" onclick="event.stopPropagation()">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
        </a>
      </div>`;
  }).join('');
}

function updateViewHeader() {
  const titleEl = document.getElementById('viewTitle');
  const countEl = document.getElementById('docCount');
  const descEl = document.getElementById('viewDescription');

  const n = state.docs.length;
  const filterEl = document.getElementById('docFilter');
  const filterText = (filterEl ? filterEl.value : '').trim().toLowerCase();
  const visible = filterText
    ? state.docs.filter(d => (d.document_title || d.title || d.filename || '').toLowerCase().includes(filterText)).length
    : n;
  countEl.textContent = filterText
    ? `${visible} of ${n} file${n !== 1 ? 's' : ''}`
    : `${n} file${n !== 1 ? 's' : ''}`;

  if (state.selectedId === '__unassigned__') {
    titleEl.textContent = 'Unassigned Files';
    document.getElementById('unassignedBadge').textContent = n;
    descEl.style.display = 'none';
  } else {
    const col = state.collections.find(c => c.collection_id === state.selectedId);
    titleEl.textContent = col ? col.name : state.selectedId;
    if (col && col.description) {
      descEl.textContent = col.description;
      descEl.style.display = '';
    } else {
      descEl.style.display = 'none';
    }
  }
}

// ── Drag and drop ──────────────────────────────────────────────────────────
function onDocDragStart(event, docId) {
  state.draggingDocId = docId;
  event.currentTarget.classList.add('dragging');
  event.dataTransfer.effectAllowed = 'move';
  event.dataTransfer.setData('text/plain', docId);
}

function onDocDragEnd(event) {
  event.currentTarget.classList.remove('dragging');
  state.draggingDocId = null;
  // Clean up any drag-over states
  document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
  document.getElementById('mainDropHint').classList.add('hidden');
}

function onTreeDragOver(event) {
  // Always prevent default so the browser allows the drop — even if our
  // state hasn't been set yet (e.g. drag started from a re-render).
  event.preventDefault();
  event.stopPropagation();
  if (!state.draggingDocId && !event.dataTransfer.types.includes('text/plain')) return;
  event.dataTransfer.dropEffect = 'move';
  event.currentTarget.classList.add('drag-over');
}

function onTreeDragLeave(event) {
  event.currentTarget.classList.remove('drag-over');
}

function onTreeDrop(event, targetCollectionId) {
  event.preventDefault();
  event.stopPropagation();
  event.currentTarget.classList.remove('drag-over');

  const docId = event.dataTransfer.getData('text/plain') || state.draggingDocId;
  if (!docId) return;

  const collectionId = targetCollectionId === '__unassigned__' ? null : targetCollectionId;
  doAssignDoc(docId, collectionId);
}

// When dragging over the main content area (assign to currently viewed collection)
function onMainDragOver(event) {
  if (!state.draggingDocId) return;
  event.preventDefault();
  const hint = document.getElementById('mainDropHint');
  if (hint.classList.contains('hidden')) {
    const colName = state.selectedId === '__unassigned__'
      ? 'Unassigned'
      : (state.collections.find(c => c.collection_id === state.selectedId)?.name || state.selectedId);
    document.getElementById('mainDropLabel').textContent = `Move to "${colName}"`;
    hint.classList.remove('hidden');
  }
}

function onMainDrop(event) {
  event.preventDefault();
  document.getElementById('mainDropHint').classList.add('hidden');
  const docId = event.dataTransfer.getData('text/plain') || state.draggingDocId;
  if (!docId) return;
  const targetId = state.selectedId === '__unassigned__' ? null : state.selectedId;
  // Don't move if already in this collection
  const doc = state.docs.find(d => d.doc_id === docId);
  if (doc) return; // already here
  doAssignDoc(docId, targetId);
}

async function doAssignDoc(docId, collectionId) {
  try {
    await api('PUT', `/api/documents/${encodeURIComponent(docId)}/collection`, { collection_id: collectionId });
    showToast(collectionId ? `Moved to "${collectionId}"` : 'Removed from collection', 'success');
    await loadAll();
  } catch (e) {
    showToast('Failed to assign: ' + e.message, 'error');
  }
}

// ── Collection actions ─────────────────────────────────────────────────────
function confirmDeleteCollection(event, collectionId) {
  event.stopPropagation();
  showConfirm(
    `Delete collection "${collectionId}"?`,
    `Chunks will be un-assigned but documents will NOT be deleted from the library.`,
    async () => {
      try {
        await api('DELETE', `/api/collections/${encodeURIComponent(collectionId)}`);
        showToast(`Collection "${collectionId}" deleted`, 'success');
        if (state.selectedId === collectionId) {
          state.selectedId = '__unassigned__';
        }
        await loadAll();
      } catch (e) {
        showToast('Delete failed: ' + e.message, 'error');
      }
    }
  );
}

async function submitCreateCollection() {
  const id = document.getElementById('colIdInput').value.trim();
  const name = document.getElementById('colNameInput').value.trim();
  const desc = document.getElementById('colDescInput').value.trim();
  const parentId = document.getElementById('colParentSelect').value || null;
  const errEl = document.getElementById('createColError');

  errEl.classList.add('hidden');
  if (!id) { errEl.textContent = 'ID is required'; errEl.classList.remove('hidden'); return; }
  if (!name) { errEl.textContent = 'Name is required'; errEl.classList.remove('hidden'); return; }

  try {
    await api('POST', '/api/collections', {
      collection_id: id,
      name,
      description: desc || null,
      parent_id: parentId,
    });
    closeModal('createCollectionModal');
    showToast(`Collection "${id}" created`, 'success');
    await loadCollections();
    // Select the new collection
    await loadDocs(id);
  } catch (e) {
    errEl.textContent = e.message;
    errEl.classList.remove('hidden');
  }
}

// ── Document actions ───────────────────────────────────────────────────────
function confirmDeleteDocument(event, docId, filename) {
  event.stopPropagation();
  showConfirm(
    `Remove "${filename}" from the library?`,
    `This will delete all chunks from the database. The raw file in data/raw/ will NOT be deleted.`,
    async () => {
      try {
        await api('DELETE', `/api/documents/${encodeURIComponent(docId)}`);
        showToast(`"${filename}" removed`, 'success');
        await loadAll();
      } catch (e) {
        showToast('Delete failed: ' + e.message, 'error');
      }
    }
  );
}

// ── Upload ─────────────────────────────────────────────────────────────────
let _selectedFile = null;

function openUploadModal() {
  _selectedFile = null;
  document.getElementById('uploadFileInfo').classList.add('hidden');
  document.getElementById('uploadPreSummary').classList.add('hidden');
  document.getElementById('uploadSubmitBtn').disabled = true;
  document.getElementById('uploadDropzone').classList.remove('hidden');
  document.getElementById('uploadProgress').classList.add('hidden');
  document.getElementById('progressBarFill').style.width = '0%';
  document.getElementById('progressLabel').textContent = 'Uploading…';
  // Pre-select current collection
  const sel = document.getElementById('uploadCollectionSelect');
  if (state.selectedId && state.selectedId !== '__unassigned__') {
    sel.value = state.selectedId;
  } else {
    sel.value = '';
  }
  openModal('uploadModal');
}

function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) setUploadFile(file);
}

function handleUploadDrop(event) {
  event.preventDefault();
  event.currentTarget.classList.remove('drag-active');
  const file = event.dataTransfer.files[0];
  if (file) setUploadFile(file);
}

function setUploadFile(file) {
  _selectedFile = file;
  document.getElementById('uploadFileName').textContent = file.name;
  document.getElementById('uploadFileInfo').classList.remove('hidden');
  document.getElementById('uploadSubmitBtn').disabled = false;
  _showPreSummary(file);
}

function _showPreSummary(file) {
  const ext = (file.name.split('.').pop() || '').toLowerCase();
  const typeLabels = { pdf: 'PDF', docx: 'DOCX', txt: 'TXT', md: 'MD' };
  const typeLabel = typeLabels[ext] || ext.toUpperCase() || 'FILE';
  // Rough chunk estimate: PDF ~1500 chars/chunk, text ~3000 chars/chunk
  const bytesPerChunk = (ext === 'pdf' || ext === 'docx') ? 1500 : 3000;
  const estChunks = Math.max(1, Math.round(file.size / bytesPerChunk));
  const sizeMB = (file.size / (1024 * 1024)).toFixed(2);

  document.getElementById('preSummaryType').textContent = typeLabel;
  document.getElementById('preSummaryChunks').textContent = estChunks;
  document.getElementById('preSummarySize').textContent = sizeMB + ' MB';
  document.getElementById('uploadPreSummary').classList.remove('hidden');
}

function clearUploadFile() {
  _selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('uploadFileInfo').classList.add('hidden');
  document.getElementById('uploadPreSummary').classList.add('hidden');
  document.getElementById('uploadSubmitBtn').disabled = true;
}

async function submitUpload() {
  if (!_selectedFile) return;

  const collectionId = document.getElementById('uploadCollectionSelect').value || null;
  const submitBtn = document.getElementById('uploadSubmitBtn');
  const progressDiv = document.getElementById('uploadProgress');
  const progressFill = document.getElementById('progressBarFill');
  const progressLabel = document.getElementById('progressLabel');

  submitBtn.disabled = true;
  progressDiv.classList.remove('hidden');
  progressFill.style.width = '30%';
  progressLabel.textContent = 'Uploading file…';

  try {
    state.uploadDraft = {
      filename: _selectedFile.name,
      collectionId: collectionId,
      startedAt: new Date().toISOString(),
    };
    _persistUploadDraft();
    renderDocGrid();

    const initRes = await fetch('/api/upload-init', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        filename: _selectedFile.name,
        collection_id: collectionId,
      }),
      keepalive: true,
    });
    if (!initRes.ok) {
      const err = await initRes.json().catch(() => ({ detail: `HTTP ${initRes.status}` }));
      throw new Error(err.detail || 'Upload init failed');
    }
    const initJob = await initRes.json();
    state.pendingJobs[initJob.job_id] = {
      filename: initJob.filename,
      collectionId: collectionId,
      status: initJob.status || 'uploading',
      destPath: null,
      queuePosition: null,
      isActive: false,
      isWaiting: false,
      gpuHolder: null,
      stage: 'uploading',
      stageDetail: 'Receiving file upload',
    };
    _persistPendingJobs();
    renderDocGrid();

    const formData = new FormData();
    formData.append('file', _selectedFile);
    formData.append('job_id', initJob.job_id);
    if (collectionId) formData.append('collection_id', collectionId);

    const res = await fetch('/api/upload', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `HTTP ${res.status}` }));
      throw new Error(err.detail || `Upload failed`);
    }
    const job = await res.json();
    progressFill.style.width = '100%';
    progressLabel.textContent = 'Upload received — indexing in background…';

    // Register pending job and close modal — ingest shown on card in grid
    state.pendingJobs[job.job_id] = {
      ...(state.pendingJobs[job.job_id] || {}),
      filename: job.filename,
      collectionId: collectionId,
      status: job.status || 'queued',
      stage: job.stage || 'queued',
      stageDetail: job.stage_detail || 'Waiting for ingest worker',
    };
    state.uploadDraft = null;
    _persistUploadDraft();
    _persistPendingJobs();
    renderDocGrid(); // show the pending card immediately

    setTimeout(() => {
      closeModal('uploadModal');
      submitBtn.disabled = false;
    }, 800);

    _startJobPoller(job.job_id);
  } catch (e) {
    const message = String(e && e.message ? e.message : e || '');
    const likelyInterrupted =
      /aborted|aborterror|failed to fetch|networkerror/i.test(message);
    if (!likelyInterrupted) {
      state.uploadDraft = null;
      _persistUploadDraft();
    }
    progressLabel.textContent = 'Error: ' + e.message;
    progressFill.style.width = '0%';
    submitBtn.disabled = false;
    showToast('Upload failed: ' + e.message, 'error');
  }
}

/**
 * Polls a single ingest job and updates its pending card.
 * Removes the pending card and reloads the grid when the job completes.
 */
function _startJobPoller(jobId) {
  const job = state.pendingJobs[jobId];
  if (!job) return;
  if (job.timerId) return;
  let attempts = 0;
  let failures = 0;
  const maxAttempts = 240; // 240 × 5 s = 20 min

  const interval = setInterval(async () => {
    attempts++;
    try {
      const status = await api('GET', `/api/upload/${jobId}`);
      failures = 0;
      state.pendingJobs[jobId] = {
        ...(state.pendingJobs[jobId] || {}),
        status: status.status || state.pendingJobs[jobId]?.status || 'running',
        error: status.error || null,
        destPath: status.dest_path || state.pendingJobs[jobId]?.destPath || null,
        queuePosition: Number.isFinite(status.queue_position) ? status.queue_position : state.pendingJobs[jobId]?.queuePosition,
        isActive: !!status.is_active,
        isWaiting: !!status.is_waiting,
        gpuHolder: status.gpu_holder || state.pendingJobs[jobId]?.gpuHolder || null,
        stage: status.stage || state.pendingJobs[jobId]?.stage || null,
        stageDetail: status.stage_detail || state.pendingJobs[jobId]?.stageDetail || null,
      };
      if (status.status === 'done') {
        clearInterval(interval);
        delete job.timerId;
        delete state.pendingJobs[jobId];
        _persistPendingJobs();
        api('DELETE', `/api/upload/${jobId}`).catch(() => {});
        showToast(`"${job.filename}" indexed successfully`, 'success');
        const colTarget = job.collectionId || state.selectedId;
        if (colTarget !== '__unassigned__' && colTarget !== state.selectedId) {
          await loadDocs(state.selectedId); // stay in current view
        } else {
          await loadDocs(state.selectedId);
        }
        await loadCollections();
      } else if (status.status === 'error') {
        clearInterval(interval);
        delete job.timerId;
        state.pendingJobs[jobId].status = 'error';
        state.pendingJobs[jobId].error = status.error || 'Unknown error';
        state.pendingJobs[jobId].stage = status.stage || state.pendingJobs[jobId].stage || 'error';
        state.pendingJobs[jobId].stageDetail = status.stage_detail || state.pendingJobs[jobId].stageDetail || 'Ingest failed';
        _persistPendingJobs();
        _updatePendingCard(jobId);
        showToast('Ingest failed: ' + (status.error || 'Unknown'), 'error');
      } else {
        _updatePendingCard(jobId, attempts);
      }
      if (attempts >= maxAttempts) {
        clearInterval(interval);
        delete job.timerId;
        delete state.pendingJobs[jobId];
        _persistPendingJobs();
        renderDocGrid();
      }
    } catch {
      failures++;
      if (failures >= 3 && state.pendingJobs[jobId]) {
        clearInterval(interval);
        delete job.timerId;
        state.pendingJobs[jobId].status = 'error';
        state.pendingJobs[jobId].error = 'Lost connection to server';
        state.pendingJobs[jobId].stage = 'disconnected';
        state.pendingJobs[jobId].stageDetail = 'Server unavailable';
        _persistPendingJobs();
        _updatePendingCard(jobId);
      }
    }
  }, 5000);
  job.timerId = interval;
}

/** Update the status text on an existing pending card without a full re-render. */
function _updatePendingCard(jobId, attempts) {
  const card = document.querySelector(`.doc-card-pending[data-job-id="${CSS.escape(jobId)}"]`);
  if (!card) return;
  const job = state.pendingJobs[jobId];
  if (!job) return;
  const label = card.querySelector('.pending-card-status');
  if (!label) return;
  if (job.status === 'error') {
    label.textContent = '✗ ' + (job.error || 'Ingest failed');
    label.style.color = '#f87171';
    card.querySelector('.pending-spinner')?.remove();
  } else {
    const dots = '.'.repeat(((attempts || 1) % 3) + 1);
    if (job.status === 'uploading') {
      label.textContent = (job.stageDetail || 'Uploading') + (job.stageDetail ? '' : dots);
    } else if (job.status === 'queued') {
      label.textContent = (job.stageDetail || 'Queued') + (job.stageDetail ? '' : dots);
    } else {
      label.textContent = (job.stageDetail || _humanizeStage(job.stage) || 'Indexing') + (job.stageDetail ? '' : dots);
    }
  }

  const detail = card.querySelector('.pending-card-detail');
  if (!detail) return;
  if (job.status === 'uploading') {
    detail.textContent = _humanizeStage(job.stage) || 'Uploading';
  } else if (job.status === 'queued') {
    detail.textContent = Number.isFinite(job.queuePosition)
      ? `Queue position #${job.queuePosition}`
      : (_humanizeStage(job.stage) || 'Waiting for GPU slot');
  } else if (job.status === 'running') {
    detail.textContent = _humanizeStage(job.stage) || (job.destPath ? `Working on ${_basename(job.destPath)}` : 'Ingest in progress');
  } else {
    detail.textContent = '';
  }
}

function _humanizeStage(stage) {
  const value = String(stage || '').trim();
  if (!value) return '';
  return value.replace(/_/g, ' ').replace(/\b\w/g, ch => ch.toUpperCase());
}

function _jobStageBadgeText(job) {
  if (!job) return 'Ingesting';
  if (job.status === 'error') return 'Failed';
  if (job.status === 'done') return 'Complete';
  return _humanizeStage(job.stage) || _humanizeStage(job.status) || 'Ingesting';
}

function _jobPrimaryStatusText(job, attempts) {
  if (!job) return '';
  if (job.status === 'error') return 'Ingest failed';
  if (job.stageDetail) return job.stageDetail;
  const dots = '.'.repeat(((attempts || 1) % 3) + 1);
  if (job.status === 'uploading') return `Uploading${dots}`;
  if (job.status === 'queued') return `Queued${dots}`;
  return `${_jobStageBadgeText(job)}${dots}`;
}

function _jobSecondaryDetailText(job) {
  if (!job) return '';
  if (job.status === 'error') return job.error || '';
  if (job.status === 'uploading') return 'Transferring file to server';
  if (job.status === 'queued') {
    return Number.isFinite(job.queuePosition)
      ? `Queue position #${job.queuePosition}`
      : 'Waiting for ingest worker';
  }
  if (job.status === 'running') {
    return job.destPath
      ? `Working on ${_basename(job.destPath)}`
      : 'Processing in background';
  }
  return '';
}

/**
 * Build the HTML for a single pending (in-progress ingest) card.
 */
function _renderPendingCard(jobId) {
  const job = state.pendingJobs[jobId];
  if (!job) return '';
  const ext = job.filename.split('.').pop().toLowerCase() || 'doc';
  const stageLabel = _humanizeStage(job.stage);
  const statusText = job.status === 'error'
    ? ('✗ ' + (job.error || 'Ingest failed'))
    : (job.status === 'uploading'
      ? (job.stageDetail || 'Uploading…')
      : (job.status === 'queued'
        ? (job.stageDetail || 'Queued…')
        : (job.stageDetail || (stageLabel ? `${stageLabel}…` : 'Indexing…'))));
  const spinnerHtml = job.status !== 'error'
    ? `<div class="pending-spinner"></div>`
    : '';
  const dismissHtml = job.status === 'error'
    ? `<button class="doc-card-dismiss" type="button" onclick="dismissPendingJob('${esc(jobId)}')" title="Dismiss upload job">×</button>`
    : '';
  const detailText = job.status === 'uploading'
    ? (stageLabel || 'Uploading')
    : (job.status === 'queued'
      ? (Number.isFinite(job.queuePosition) ? `Queue position #${job.queuePosition}` : (stageLabel || 'Waiting for GPU slot'))
      : (job.status === 'running'
        ? (stageLabel || (job.destPath ? `Working on ${_basename(job.destPath)}` : 'Ingest in progress'))
        : ''));
  return `
    <div class="doc-card doc-card-pending" data-job-id="${esc(jobId)}">
      ${dismissHtml}
      <div class="doc-card-icon ${esc(ext)}">${extIcon(ext)}</div>
      <div class="doc-card-title" title="${esc(job.filename)}">${esc(job.filename)}</div>
      <div class="doc-card-meta">
        <span class="doc-badge doc-badge--${esc(ext)}">${ext.toUpperCase()}</span>
        <span class="doc-badge ingest-badge">${esc(job.stage || job.status || 'ingesting')}</span>
      </div>
      <div class="pending-status-row">
        ${spinnerHtml}
        <span class="pending-card-status">${esc(statusText)}</span>
      </div>
      <div class="pending-card-detail">${esc(detailText)}</div>
    </div>`;
}

function _renderUploadDraftCard() {
  const draft = state.uploadDraft;
  if (!draft) return '';
  const ext = (draft.filename || 'doc').split('.').pop().toLowerCase() || 'doc';
  return `
    <div class="doc-card doc-card-pending" data-job-id="upload-draft">
      <div class="doc-card-icon ${esc(ext)}">${extIcon(ext)}</div>
      <div class="doc-card-title" title="${esc(draft.filename || 'uploading')}">${esc(draft.filename || 'uploading')}</div>
      <div class="doc-card-meta">
        <span class="doc-badge doc-badge--${esc(ext)}">${ext.toUpperCase()}</span>
        <span class="doc-badge ingest-badge">uploading</span>
      </div>
      <div class="pending-status-row">
        <div class="pending-spinner"></div>
        <span class="pending-card-status">Starting upload…</span>
      </div>
      <div class="pending-card-detail">Waiting for server job registration</div>
    </div>`;
}

function _basename(path) {
  const parts = String(path || '').split(/[\\/]/);
  return parts[parts.length - 1] || '';
}

function _jobStageBadgeText(job) {
  if (!job) return 'Ingesting';
  if (job.status === 'error') return 'Failed';
  if (job.status === 'done') return 'Complete';
  return _humanizeStage(job.stage) || _humanizeStage(job.status) || 'Ingesting';
}

function _jobPrimaryStatusText(job, attempts) {
  if (!job) return '';
  if (job.status === 'error') return 'Ingest failed';
  if (job.stageDetail) return job.stageDetail;
  const dots = '.'.repeat(((attempts || 1) % 3) + 1);
  if (job.status === 'uploading') return `Uploading${dots}`;
  if (job.status === 'queued') return `Queued${dots}`;
  return `${_jobStageBadgeText(job)}${dots}`;
}

function _jobSecondaryDetailText(job) {
  if (!job) return '';
  if (job.status === 'error') return job.error || '';
  if (job.status === 'uploading') return 'Transferring file to server';
  if (job.status === 'queued') {
    return Number.isFinite(job.queuePosition)
      ? `Queue position #${job.queuePosition}`
      : 'Waiting for ingest worker';
  }
  if (job.status === 'running') {
    return job.destPath
      ? `Working on ${_basename(job.destPath)}`
      : 'Processing in background';
  }
  return '';
}

function _updatePendingCard(jobId, attempts) {
  const card = document.querySelector(`.doc-card-pending[data-job-id="${CSS.escape(jobId)}"]`);
  if (!card) return;
  const job = state.pendingJobs[jobId];
  if (!job) return;

  const badge = card.querySelector('.ingest-badge');
  const label = card.querySelector('.pending-card-status');
  const detail = card.querySelector('.pending-card-detail');
  if (!label || !detail) return;

  if (badge) {
    badge.textContent = _jobStageBadgeText(job);
    badge.classList.toggle('ingest-badge--error', job.status === 'error');
  }

  if (job.status === 'error') {
    label.textContent = 'Ingest failed';
    label.classList.add('pending-card-status--error');
    card.querySelector('.pending-spinner')?.remove();
  } else {
    label.textContent = _jobPrimaryStatusText(job, attempts);
    label.classList.remove('pending-card-status--error');
  }

  detail.textContent = _jobSecondaryDetailText(job);
}

function _renderPendingCard(jobId) {
  const job = state.pendingJobs[jobId];
  if (!job) return '';
  const ext = job.filename.split('.').pop().toLowerCase() || 'doc';
  const statusText = _jobPrimaryStatusText(job, 1);
  const detailText = _jobSecondaryDetailText(job);
  const spinnerHtml = job.status !== 'error'
    ? `<div class="pending-spinner"></div>`
    : '';
  const dismissHtml = job.status === 'error'
    ? `<button class="doc-card-dismiss" type="button" onclick="dismissPendingJob('${esc(jobId)}')" title="Dismiss upload job">×</button>`
    : '';
  const badgeClass = job.status === 'error'
    ? 'doc-badge ingest-badge ingest-badge--error'
    : 'doc-badge ingest-badge';
  return `
    <div class="doc-card doc-card-pending" data-job-id="${esc(jobId)}">
      ${dismissHtml}
      <div class="doc-card-icon ${esc(ext)}">${extIcon(ext)}</div>
      <div class="doc-card-title" title="${esc(job.filename)}">${esc(job.filename)}</div>
      <div class="doc-card-meta">
        <span class="doc-badge doc-badge--${esc(ext)}">${ext.toUpperCase()}</span>
        <span class="${badgeClass}">${esc(_jobStageBadgeText(job))}</span>
      </div>
      <div class="pending-status-row">
        ${spinnerHtml}
        <span class="pending-card-status${job.status === 'error' ? ' pending-card-status--error' : ''}">${esc(statusText)}</span>
      </div>
      <div class="pending-card-detail">${esc(detailText)}</div>
    </div>`;
}

function pollJob(jobId, filename, progressFill, progressLabel, submitBtn, collectionId) {
  // Legacy shim — new code uses _startJobPoller. Kept for any external callers.
  state.pendingJobs[jobId] = { filename, collectionId, status: 'running' };
  _persistPendingJobs();
  _startJobPoller(jobId);
}

async function dismissPendingJob(jobId) {
  if (!jobId || !state.pendingJobs[jobId]) return;
  try {
    await api('DELETE', `/api/upload/${jobId}`);
  } catch (_) {}
  delete state.pendingJobs[jobId];
  _persistPendingJobs();
  renderDocGrid();
}

function _resumePendingJobPollers() {
  for (const [jobId, job] of Object.entries(state.pendingJobs || {})) {
    if (!job || job.status === 'error') continue;
    _startJobPoller(jobId);
  }
}



// ── Modals ─────────────────────────────────────────────────────────────────
function openModal(id) {
  document.getElementById(id).classList.remove('hidden');
}
function closeModal(id) {
  document.getElementById(id).classList.add('hidden');
}

function showConfirm(title, text, onOk) {
  document.getElementById('confirmModalTitle').textContent = title;
  document.getElementById('confirmModalText').textContent = text;
  state.confirmCallback = onOk;
  openModal('confirmModal');
}

document.getElementById('confirmModalOk').addEventListener('click', () => {
  closeModal('confirmModal');
  if (state.confirmCallback) {
    state.confirmCallback();
    state.confirmCallback = null;
  }
});

// Close backdrop on click outside
document.querySelectorAll('.modal-backdrop').forEach(backdrop => {
  backdrop.addEventListener('click', e => {
    if (e.target === backdrop) closeModal(backdrop.id);
  });
});

// ── Populate selects ───────────────────────────────────────────────────────
function populateCollectionSelects() {
  const leafCollections = getLeafCollections();

  // Upload modal select
  const uploadSel = document.getElementById('uploadCollectionSelect');
  const uploadCurrent = uploadSel.value;
  uploadSel.innerHTML = '<option value="">None — leave unassigned</option>';
  for (const c of leafCollections) {
    const opt = document.createElement('option');
    opt.value = c.collection_id;
    opt.textContent = c.collection_id + ' — ' + c.name;
    uploadSel.appendChild(opt);
  }
  if (uploadCurrent) uploadSel.value = uploadCurrent;

  // Create collection parent select
  const parentSel = document.getElementById('colParentSelect');
  const parentCurrent = parentSel.value;
  parentSel.innerHTML = '<option value="">None (top-level)</option>';
  // Only top-level collections can be parents (two-level hierarchy)
  const topLevel = state.collections.filter(c => !c.parent_id);
  for (const c of topLevel) {
    const opt = document.createElement('option');
    opt.value = c.collection_id;
    opt.textContent = c.collection_id + ' — ' + c.name;
    parentSel.appendChild(opt);
  }
  if (parentCurrent) parentSel.value = parentCurrent;
}

function getLeafCollections() {
  // Return collections that have no children (the ones docs actually live in)
  const parentIds = new Set(state.collections.filter(c => c.parent_id).map(c => c.parent_id));
  const leaves = state.collections.filter(c => !parentIds.has(c.collection_id));
  return leaves;
}

// ── Utility ────────────────────────────────────────────────────────────────
function esc(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

let _toastTimer = null;
function showToast(message, type) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'toast toast--' + (type || 'info');
  toast.classList.remove('hidden');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => toast.classList.add('hidden'), 3500);
}

// ── Button wiring ──────────────────────────────────────────────────────────
document.getElementById('uploadBtn').addEventListener('click', openUploadModal);
document.getElementById('refreshBtn').addEventListener('click', async () => {
  const btn = document.getElementById('refreshBtn');
  btn.disabled = true;
  await loadAll();
  btn.disabled = false;
});
document.getElementById('newCollectionBtn').addEventListener('click', () => {
  document.getElementById('colIdInput').value = '';
  document.getElementById('colNameInput').value = '';
  document.getElementById('colDescInput').value = '';
  document.getElementById('createColError').classList.add('hidden');
  openModal('createCollectionModal');
});

// Unassigned tree item click
document.querySelector('.tree-item--unassigned').addEventListener('click', () => {
  loadDocs('__unassigned__');
});

// Keyboard: Escape closes modals
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    ['uploadModal', 'createCollectionModal', 'confirmModal'].forEach(id => {
      if (!document.getElementById(id).classList.contains('hidden')) closeModal(id);
    });
    clearMobileSelection();
  }
});

// ── Sidebar toggle (mobile) ────────────────────────────────────────────────
function openSidebar() {
  document.getElementById('libSidebar').classList.add('open');
  document.getElementById('sidebarBackdrop').classList.remove('hidden');
}
function closeSidebar() {
  document.getElementById('libSidebar').classList.remove('open');
  document.getElementById('sidebarBackdrop').classList.add('hidden');
}
document.getElementById('sidebarToggle').addEventListener('click', () => {
  const sidebar = document.getElementById('libSidebar');
  if (sidebar.classList.contains('open')) closeSidebar();
  else openSidebar();
});

// Close sidebar when a collection is tapped on mobile
const _origSelectCollection = selectCollection;
window.selectCollection = function(el) {
  _origSelectCollection(el);
  closeSidebar();
};
document.querySelector('.tree-item--unassigned').addEventListener('click', closeSidebar);

// ── Mobile tap-to-assign ───────────────────────────────────────────────────
let _mobileSelectedDocId = null;

function isTouchDevice() {
  return window.matchMedia('(max-width: 640px)').matches;
}

// Called when a doc card is tapped on mobile
function onDocCardTap(event, docId) {
  if (!isTouchDevice()) return; // desktop: do nothing, DnD handles it
  event.stopPropagation();

  // If tapping the same card again, deselect
  if (_mobileSelectedDocId === docId) {
    clearMobileSelection();
    return;
  }

  _mobileSelectedDocId = docId;

  // Highlight selected card
  document.querySelectorAll('.doc-card').forEach(c => c.classList.remove('touch-selected'));
  const card = document.querySelector(`.doc-card[data-doc-id="${CSS.escape(docId)}"]`);
  if (card) card.classList.add('touch-selected');

  // Find the doc's display name
  const doc = state.docs.find(d => d.doc_id === docId);
  const name = doc ? (doc.document_title || doc.title || doc.filename) : docId;
  document.getElementById('mobileAssignTitle').textContent =
    'Move "' + name.substring(0, 30) + (name.length > 30 ? '…' : '') + '" to…';

  // Build collection list
  const container = document.getElementById('mobileAssignCollections');
  let html = '';
  // "Unassigned" option only if not already unassigned
  if (state.selectedId !== '__unassigned__') {
    html += `<div class="mobile-assign-item mobile-assign-item--unassign"
                  onclick="mobileAssignTo(null)">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                 <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
               </svg>
               Unassigned
             </div>`;
  }
  for (const col of state.collections) {
    if (col.collection_id === state.selectedId) continue;
    html += `<div class="mobile-assign-item" onclick="mobileAssignTo('${esc(col.collection_id)}')">
               <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                 <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"/>
               </svg>
               <span>${esc(col.name)}</span>
               <span style="margin-left:auto;font-size:11px;color:var(--text-dim)">${esc(col.collection_id)}</span>
             </div>`;
  }
  container.innerHTML = html;

  document.getElementById('mobileAssignBar').classList.remove('hidden');
}

async function mobileAssignTo(collectionId) {
  if (!_mobileSelectedDocId) return;
  const docId = _mobileSelectedDocId;
  clearMobileSelection();
  await doAssignDoc(docId, collectionId);
}

function clearMobileSelection() {
  _mobileSelectedDocId = null;
  document.querySelectorAll('.doc-card').forEach(c => c.classList.remove('touch-selected'));
  document.getElementById('mobileAssignBar').classList.add('hidden');
}

// ── Init ───────────────────────────────────────────────────────────────────
_restorePendingJobs();
_restoreUploadDraft();
loadAll();
