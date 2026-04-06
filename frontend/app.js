/* global state */
const API = window.location.origin;   // works from inside Docker or localhost
let currentResults = { merged: [], exact: [], regex: [], semantic: [] };
let currentTab = "merged";

// ── DOM refs ──────────────────────────────────────────────────────────────────
const dropZone        = document.getElementById("drop-zone");
const fileInput       = document.getElementById("file-input");
const uploadQueue     = document.getElementById("upload-queue");
const searchInput     = document.getElementById("search-input");
const modeSelect      = document.getElementById("mode-select");
const topkInput       = document.getElementById("topk-input");
const searchBtn       = document.getElementById("search-btn");
const resultsContainer= document.getElementById("results-container");
const tabs            = document.getElementById("tabs");
const tabBtns         = document.querySelectorAll(".tab-btn");
const docsList        = document.getElementById("docs-list");
const refreshBtn      = document.getElementById("refresh-docs-btn");
const toast           = document.getElementById("toast");

// ── Drop-zone / file upload ──────────────────────────────────────────────────
dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  handleFiles([...e.dataTransfer.files]);
});
fileInput.addEventListener("change", () => handleFiles([...fileInput.files]));

// ── Dry-run modal ─────────────────────────────────────────────────────────────
const dryRunModal    = document.getElementById("dry-run-modal");
const dryRunBackdrop = document.getElementById("dry-run-backdrop");
const dryRunClose    = document.getElementById("dry-run-close");
const dryRunMeta     = document.getElementById("dry-run-meta");
const dryRunBody     = document.getElementById("dry-run-body");

function openDryRunModal(data) {
  dryRunMeta.innerHTML = `
    <span class="dr-pill">engine: ${escHtml(data.ocr_engine)}</span>
    <span class="dr-pill">langs: ${escHtml((data.ocr_langs || []).join(", "))}</span>
    <span class="dr-pill">${data.pages} page(s)</span>
    <span class="dr-pill">${data.total_chunks} chunk(s)</span>`;

  dryRunBody.innerHTML = (data.preview || []).map(page => `
<div class="dr-page">
  <div class="dr-page-header">Page ${page.page_number}</div>
  <div class="dr-section-label">Raw extracted text</div>
  <pre class="dr-raw">${escHtml(page.raw_text || "(empty)")}</pre>
  <div class="dr-section-label">Chunks (${page.chunks.length})</div>
  <ol class="dr-chunks">
    ${page.chunks.map(c => `<li><pre>${escHtml(c)}</pre></li>`).join("")}
  </ol>
</div>`).join("") || '<p class="dim">No pages extracted.</p>';

  dryRunModal.classList.remove("hidden");
  document.body.style.overflow = "hidden";
}

function closeDryRunModal() {
  dryRunModal.classList.add("hidden");
  document.body.style.overflow = "";
}

dryRunClose.addEventListener("click", closeDryRunModal);
dryRunBackdrop.addEventListener("click", closeDryRunModal);
document.addEventListener("keydown", e => { if (e.key === "Escape") closeDryRunModal(); });

// Show/hide the page-limit row when dry-run is toggled
document.getElementById("dry-run-check").addEventListener("change", function () {
  const row = document.getElementById("dry-run-pages-row");
  if (row) row.style.display = this.checked ? "" : "none";
});

function handleFiles(files) {
  // Capture OCR settings once for this batch of uploads
  const engine   = document.querySelector('input[name="ocr_engine"]:checked')?.value || "tesseract";
  const langs    = [...document.querySelectorAll('input[name="ocr_lang"]:checked')].map(el => el.value);
  const isDryRun = document.getElementById("dry-run-check")?.checked ?? false;
  const dryRunPages = isDryRun
    ? parseInt(document.getElementById("dry-run-pages")?.value || "3", 10)
    : 0;
  files.forEach(file => uploadFile(file, engine, langs, isDryRun, dryRunPages));
}

async function uploadFile(file, ocrEngine, ocrLangs, isDryRun, dryRunPages) {
  const item = createQueueItem(file.name);
  uploadQueue.prepend(item.el);
  item.setBadge("uploading", isDryRun ? "Analysing…" : "Uploading…");

  const fd = new FormData();
  fd.append("file", file, file.name);

  // Append OCR params for PDF; dry_run applies to all types
  if (file.name.toLowerCase().endsWith(".pdf")) {
    fd.append("ocr_engine", ocrEngine);
    ocrLangs.forEach(lang => fd.append("ocr_langs", lang));
  }
  if (isDryRun) {
    fd.append("dry_run", "true");
    fd.append("dry_run_pages", String(dryRunPages));
  }

  try {
    const resp = await fetch(`${API}/documents/upload`, { method: "POST", body: fd });
    const data = await resp.json();

    if (!resp.ok) {
      item.setBadge("error", `Error ${resp.status}`);
      showToast(data.detail || `Upload failed (${resp.status})`, "error");
      return;
    }

    // Dry-run: open preview modal instead of normal badge
    if (isDryRun) {
      item.setBadge("skip", `Dry run — ${data.pages} page(s), ${data.total_chunks} chunk(s)`);
      openDryRunModal(data);
      return;
    }

    const { status, chunks, pages } = data;
    if (status === "duplicate") {
      item.setBadge("skip", "Already indexed");
    } else if (status === "empty") {
      item.setBadge("skip", "No text found");
    } else {
      item.setBadge("ok", `${pages} page(s) · ${chunks} chunk(s)`);
      showToast(`${file.name} indexed (${chunks} chunks)`, "success");
    }
  } catch (err) {
    item.setBadge("error", "Network error");
    showToast(`Upload failed: ${err.message}`, "error");
  }
}

function createQueueItem(name) {
  const el = document.createElement("div");
  el.className = "upload-item";
  el.innerHTML = `<span class="name" title="${escHtml(name)}">${escHtml(name)}</span>
                  <span class="badge badge-pending">Pending</span>`;
  const badge = el.querySelector(".badge");

  return {
    el,
    setBadge(type, label) {
      badge.className = `badge badge-${type}`;
      badge.textContent = label;
    },
  };
}

// ── Search ────────────────────────────────────────────────────────────────────
searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", e => { if (e.key === "Enter") doSearch(); });

async function doSearch() {
  const query = searchInput.value.trim();
  if (!query) { showToast("Enter a search query.", "info"); return; }

  searchBtn.textContent = "Searching…";
  searchBtn.disabled = true;
  resultsContainer.innerHTML = '<p class="loading">Searching…</p>';
  tabs.classList.add("hidden");

  const params = new URLSearchParams({
    query,
    top_k: topkInput.value || 10,
    mode: modeSelect.value,
  });

  try {
    const resp = await fetch(`${API}/search/?${params}`);
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    currentResults = data;
    renderResults(currentTab);
    updateTabCounts(data);
    tabs.classList.remove("hidden");
  } catch (err) {
    resultsContainer.innerHTML = `<p class="dim">Search failed: ${escHtml(err.message)}</p>`;
    showToast(err.message, "error");
  } finally {
    searchBtn.textContent = "Search";
    searchBtn.disabled = false;
  }
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    tabBtns.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    currentTab = btn.dataset.tab;
    renderResults(currentTab);
  });
});

function updateTabCounts(data) {
  tabBtns.forEach(btn => {
    const key = btn.dataset.tab;
    const count = data[key]?.length ?? 0;
    btn.textContent = `${key.charAt(0).toUpperCase() + key.slice(1)} (${count})`;
    if (key === currentTab) btn.classList.add("active");
  });
}

// ── Render results ────────────────────────────────────────────────────────────
function renderResults(tab) {
  const items = currentResults[tab] || [];
  if (items.length === 0) {
    resultsContainer.innerHTML = `<p class="dim">No results in the "${tab}" layer.</p>`;
    return;
  }

  const query = searchInput.value.trim();
  resultsContainer.innerHTML = items.map(item => renderCard(item, query)).join("");
}

function renderCard(item, query) {
  const meta   = item.metadata || {};
  const file   = meta.file_name || "Unknown file";
  const page   = meta.page_number ?? 0;
  const chunk  = meta.chunk_index ?? "?";
  const total  = meta.total_chunks ?? "?";
  const docId  = meta.doc_id || "";
  const score  = (item.score * 100).toFixed(1);
  const source = item.source || "?";

  const text   = highlightQuery(item.text || "", query);
  const fileUrl = docId ? `${API}/documents/${docId}/file` : null;
  const pageHint = page > 0 ? `page ${page}` : "no page";

  return `
<div class="result-card">
  <div class="result-header">
    <span class="result-filename">${escHtml(file)}</span>
    <span class="source-badge source-${source}">${source}</span>
    <span class="result-meta">${escHtml(pageHint)} · chunk ${chunk}/${total}</span>
    <span class="score-pill">score ${score}%</span>
  </div>
  <div class="result-text">${text}</div>
  <div class="result-actions">
    ${fileUrl ? `<a class="result-link" href="${fileUrl}" target="_blank" rel="noopener">
      📄 Open file${page > 0 ? ` (page ${page})` : ""}
    </a>` : ""}
    ${docId ? `<span class="result-meta">doc: ${docId.slice(0,8)}…</span>` : ""}
  </div>
</div>`;
}

function highlightQuery(text, query) {
  const preview = text.slice(0, 400) + (text.length > 400 ? "…" : "");
  const safe    = escHtml(preview);
  if (!query) return safe;
  try {
    const re = new RegExp(`(${escRegex(query)})`, "gi");
    return safe.replace(re, "<mark>$1</mark>");
  } catch {
    return safe;
  }
}

// ── Documents list ────────────────────────────────────────────────────────────
refreshBtn.addEventListener("click", loadDocuments);

async function loadDocuments() {
  docsList.innerHTML = '<p class="loading">Loading…</p>';
  try {
    const resp = await fetch(`${API}/documents/?limit=200`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const docs = await resp.json();

    if (!docs.length) {
      docsList.innerHTML = '<p class="dim">No documents indexed yet.</p>';
      return;
    }

    docsList.innerHTML = docs.map(doc => `
<div class="doc-item">
  <span class="doc-type">${escHtml(doc.file_type || "?").toUpperCase()}</span>
  <span class="doc-name" title="${escHtml(doc.file_name || "")}">${escHtml(doc.file_name || "—")}</span>
  <span class="doc-meta">chunks: ${doc.total_chunks ?? "?"} · page: ${doc.page_count ?? "?"}</span>
  <span class="doc-meta">${(doc.upload_ts || "").slice(0, 10)}</span>
  <button class="btn-danger" data-id="${escHtml(doc.doc_id || "")}" onclick="deleteDoc(this)">Delete</button>
</div>`).join("");
  } catch (err) {
    docsList.innerHTML = `<p class="dim">Failed to load: ${escHtml(err.message)}</p>`;
  }
}

async function deleteDoc(btn) {
  const docId = btn.dataset.id;
  if (!docId) return;
  if (!confirm(`Delete document ${docId.slice(0, 8)}…?`)) return;

  btn.disabled = true;
  btn.textContent = "…";
  try {
    const resp = await fetch(`${API}/documents/${docId}`, { method: "DELETE" });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    showToast("Document deleted.", "success");
    loadDocuments();
  } catch (err) {
    showToast(`Delete failed: ${err.message}`, "error");
    btn.disabled = false;
    btn.textContent = "Delete";
  }
}

// ── Toast ─────────────────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(msg, type = "info") {
  toast.textContent = msg;
  toast.className = `toast ${type}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.add("hidden"), 3500);
}

// ── Utilities ─────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
