document.addEventListener("DOMContentLoaded", () => {

  // Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const uploadStatus = document.getElementById("uploadStatus");
  const uploadSpinner = document.getElementById("uploadSpinner");
  const uploadStatusText = document.getElementById("uploadStatusText");
  const docList = document.getElementById("docList");

  const indexedDocsCount = document.getElementById("indexedDocsCount");
  const indexedChunksCount = document.getElementById("indexedChunksCount");
  const healthStatusBadge = document.getElementById("healthStatusBadge");

  const queryForm = document.getElementById("queryForm");
  const questionInput = document.getElementById("questionInput");
  const submitBtn = document.getElementById("submitBtn");
  const pillsWrapper = document.getElementById("pillsWrapper");

  const emptyState = document.getElementById("emptyState");
  const resultContainer = document.getElementById("resultContainer");
  const answerText = document.getElementById("answerText");
  const answerBadges = document.getElementById("answerBadges");
  const groundedBadge = document.getElementById("groundedBadge");
  const latencyBadge = document.getElementById("latencyBadge");
  const sourceCount = document.getElementById("sourceCount");
  const sourcesList = document.getElementById("sourcesList");

  const openManualBtn = document.getElementById("openManualBtn");
  const closeManualBtn = document.getElementById("closeManualBtn");
  const manualModal = document.getElementById("manualModal");
  const resetIndexBtn = document.getElementById("resetIndexBtn");
  const summarizeBtn = document.getElementById("summarizeBtn");

  // Multi-document selection elements
  const selectionToolbar = document.getElementById("selectionToolbar");
  const selectAllDocsCheckbox = document.getElementById("selectAllDocsCheckbox");
  const summarizeSelectedBtn = document.getElementById("summarizeSelectedBtn");
  const selectedDocsCount = document.getElementById("selectedDocsCount");

  // Custom text summarizer elements
  const toggleCustomTextBtn = document.getElementById("toggleCustomTextBtn");
  const customTextSection = document.getElementById("customTextSection");
  const customTextInput = document.getElementById("customTextInput");
  const summarizeCustomTextBtn = document.getElementById("summarizeCustomTextBtn");
  const clearCustomTextBtn = document.getElementById("clearCustomTextBtn");

  let indexedDocsSet = new Set();
  let selectedDocsSet = new Set();

  // 1. Check System Health & Sync Document List
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        indexedDocsCount.textContent = data.indexed_documents;
        indexedChunksCount.textContent = data.indexed_chunks;
        healthStatusBadge.textContent = "Online";
        healthStatusBadge.className = "badge badge-success";

        // Sync indexed documents list from backend
        if (Array.isArray(data.documents)) {
          indexedDocsSet = new Set(data.documents);
          // Keep existing selections that still exist, or select all if new
          const nextSelected = new Set();
          indexedDocsSet.forEach((d) => {
            if (selectedDocsSet.has(d) || selectedDocsSet.size === 0) {
              nextSelected.add(d);
            }
          });
          selectedDocsSet = nextSelected.size > 0 ? nextSelected : new Set(indexedDocsSet);
          renderDocList();
          fetchSuggestions();
        }
      }
    } catch (e) {
      healthStatusBadge.textContent = "Offline";
      healthStatusBadge.className = "badge badge-warning";
    }
  }

  fetchHealth();

  // Reset / Clear Index Handler
  if (resetIndexBtn) {
    resetIndexBtn.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to clear all indexed documents and vectors?")) {
        return;
      }
      try {
        const res = await fetch("/documents/reset", { method: "POST" });
        if (res.ok) {
          indexedDocsSet.clear();
          selectedDocsSet.clear();
          renderDocList();
          await fetchHealth();
          fetchSuggestions();
          showUploadStatus("All indexed documents and vectors have been cleared.", false, false);
          emptyState.classList.remove("hidden");
          resultContainer.classList.add("hidden");
        }
      } catch (err) {
        showUploadStatus(`Failed to reset: ${err.message}`, true, false);
      }
    });
  }

  // 2. Drag & Drop File Upload
  dropZone.addEventListener("click", () => fileInput.click());

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    const fileName = file.name;
    const ext = fileName.split('.').pop().toLowerCase();
    
    if (ext !== 'pdf' && ext !== 'txt') {
      showUploadStatus(`Unsupported format .${ext}. Only PDF and TXT files are allowed.`, true);
      return;
    }

    showUploadStatus(`Indexing "${fileName}"...`, false, true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/documents/upload", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (res.ok) {
        showUploadStatus(`Successfully indexed "${fileName}" (${data.chunks_created} chunks)`, false, false);
        indexedDocsSet.add(fileName);
        renderDocList();
        fetchHealth();
        fetchSuggestions();
      } else {
        showUploadStatus(`Upload failed: ${data.detail || 'Unknown error'}`, true, false);
      }
    } catch (err) {
      showUploadStatus(`Network error during upload: ${err.message}`, true, false);
    }
  }

  function showUploadStatus(msg, isError = false, isLoading = false) {
    uploadStatus.classList.remove("hidden");
    uploadStatusText.textContent = msg;
    if (isLoading) {
      uploadSpinner.classList.remove("hidden");
    } else {
      uploadSpinner.classList.add("hidden");
    }
    if (isError) {
      uploadStatus.style.background = "rgba(239, 68, 68, 0.15)";
      uploadStatus.style.borderColor = "rgba(239, 68, 68, 0.3)";
    } else {
      uploadStatus.style.background = "rgba(16, 185, 129, 0.15)";
      uploadStatus.style.borderColor = "rgba(16, 185, 129, 0.3)";
    }
  }

  function renderDocList() {
    if (indexedDocsSet.size === 0) {
      docList.innerHTML = `<li class="empty-doc-msg">No files indexed yet. Upload a PDF or TXT to start!</li>`;
      if (selectionToolbar) selectionToolbar.classList.add("hidden");
      return;
    }

    if (selectionToolbar) {
      selectionToolbar.classList.remove("hidden");
      if (selectedDocsCount) selectedDocsCount.textContent = selectedDocsSet.size;
      if (selectAllDocsCheckbox) {
        selectAllDocsCheckbox.checked = selectedDocsSet.size === indexedDocsSet.size && indexedDocsSet.size > 0;
      }
    }

    docList.innerHTML = "";
    indexedDocsSet.forEach((docName) => {
      const safeDocName = escapeHtml(docName);
      const isChecked = selectedDocsSet.has(docName) ? "checked" : "";
      const li = document.createElement("li");
      li.className = "doc-item";
      li.innerHTML = `
        <div class="doc-info">
          <input type="checkbox" class="doc-checkbox" data-doc="${safeDocName}" ${isChecked} title="Select ${safeDocName} for summary">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <span class="doc-name" title="${safeDocName}">${safeDocName}</span>
        </div>
        <div class="doc-item-actions">
          <button class="btn-summarize-doc" data-doc="${safeDocName}" title="Summarize only ${safeDocName}">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
              <polyline points="14 2 14 8 20 8"/>
            </svg>
            <span>Summarize</span>
          </button>
          <span class="badge badge-success">Indexed</span>
        </div>
      `;
      docList.appendChild(li);
    });

    // Checkbox change handlers
    docList.querySelectorAll(".doc-checkbox").forEach((cb) => {
      cb.addEventListener("change", (e) => {
        const doc = cb.getAttribute("data-doc");
        if (cb.checked) {
          selectedDocsSet.add(doc);
        } else {
          selectedDocsSet.delete(doc);
        }
        if (selectedDocsCount) selectedDocsCount.textContent = selectedDocsSet.size;
        if (selectAllDocsCheckbox) {
          selectAllDocsCheckbox.checked = selectedDocsSet.size === indexedDocsSet.size;
        }
        if (selectedDocsSet.size === 1) {
          fetchSuggestions(Array.from(selectedDocsSet)[0]);
        } else {
          fetchSuggestions();
        }
      });
    });

    // Attach click events to individual doc summarize buttons
    docList.querySelectorAll(".btn-summarize-doc").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        e.stopPropagation();
        const targetDoc = btn.getAttribute("data-doc");
        executeSummary({ documents: [targetDoc] });
      });
    });
  }

  // Select All Checkbox Handler
  if (selectAllDocsCheckbox) {
    selectAllDocsCheckbox.addEventListener("change", () => {
      if (selectAllDocsCheckbox.checked) {
        selectedDocsSet = new Set(indexedDocsSet);
      } else {
        selectedDocsSet.clear();
      }
      renderDocList();
      fetchSuggestions();
    });
  }

  // Summarize Selected Documents Button Handler
  if (summarizeSelectedBtn) {
    summarizeSelectedBtn.addEventListener("click", () => {
      if (selectedDocsSet.size === 0) {
        showUploadStatus("Please select at least one document using the checkboxes to summarize.", true, false);
        return;
      }
      executeSummary({ documents: Array.from(selectedDocsSet) });
    });
  }

  // Custom Text Toggle & Actions
  if (toggleCustomTextBtn && customTextSection) {
    toggleCustomTextBtn.addEventListener("click", () => {
      customTextSection.classList.toggle("hidden");
    });
  }

  if (clearCustomTextBtn && customTextInput) {
    clearCustomTextBtn.addEventListener("click", () => {
      customTextInput.value = "";
    });
  }

  if (summarizeCustomTextBtn && customTextInput) {
    summarizeCustomTextBtn.addEventListener("click", () => {
      const text = customTextInput.value.trim();
      if (!text) {
        alert("Please paste or type text into the box first to generate a summary.");
        return;
      }
      executeSummary({ text: text });
    });
  }

  // 3. Query & Summary Submission
  if (summarizeBtn) {
    summarizeBtn.addEventListener("click", () => {
      if (selectedDocsSet.size > 0) {
        executeSummary({ documents: Array.from(selectedDocsSet) });
      } else {
        executeSummary();
      }
    });
  }

  queryForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (q) {
      executeQuery(q);
    }
  });

  // Render dynamic suggestions into pills wrapper
  function renderSuggestions(suggestions) {
    if (!pillsWrapper || !Array.isArray(suggestions) || suggestions.length === 0) return;
    pillsWrapper.innerHTML = "";
    suggestions.forEach((item, index) => {
      const btn = document.createElement("button");
      const isExecutive = item.label.toLowerCase().includes("summary") || item.label.toLowerCase().includes("executive") || index === 0;
      btn.className = `query-pill${isExecutive ? " pill-accent" : ""}`;
      btn.setAttribute("data-query", item.query);
      btn.textContent = item.label;
      btn.title = item.query;
      btn.addEventListener("click", () => {
        questionInput.value = item.query;
        executeQuery(item.query);
      });
      pillsWrapper.appendChild(btn);
    });
  }

  // Fetch dynamic suggestions from backend
  async function fetchSuggestions(documentName = null) {
    try {
      const url = documentName ? `/documents/suggestions?document=${encodeURIComponent(documentName)}` : "/documents/suggestions";
      const res = await fetch(url);
      if (res.ok) {
        const data = await res.json();
        if (data.suggestions && data.suggestions.length > 0) {
          renderSuggestions(data.suggestions);
        }
      }
    } catch (e) {
      console.warn("Could not fetch dynamic suggestions:", e);
    }
  }

  // Initial listener attachment for any pre-rendered fallback query pills
  if (pillsWrapper) {
    pillsWrapper.querySelectorAll(".query-pill").forEach((pill) => {
      pill.addEventListener("click", () => {
        const q = pill.getAttribute("data-query");
        questionInput.value = q;
        executeQuery(q);
      });
    });
  }

  async function executeSummary(options = {}) {
    // Determine payload based on arguments
    let payload = {};
    let label = "all indexed documents";

    if (typeof options === "string") {
      payload = { document: options };
      label = `"${options}"`;
    } else if (options.text) {
      payload = { text: options.text };
      label = "Custom Text Input";
    } else if (options.documents && options.documents.length > 0) {
      payload = { documents: options.documents };
      label = options.documents.length === 1 ? `"${options.documents[0]}"` : `${options.documents.length} selected documents`;
    } else {
      if (indexedDocsSet.size === 0) {
        showUploadStatus("Please upload a PDF or TXT document first before generating a summary.", true, false);
        return;
      }
      payload = {};
      label = "all indexed documents";
    }

    emptyState.classList.add("hidden");
    resultContainer.classList.remove("hidden");

    answerText.textContent = `Synthesizing comprehensive executive summary for ${label}...`;
    sourcesList.innerHTML = "";
    groundedBadge.textContent = "Executive Summary";
    groundedBadge.className = "badge badge-primary";
    sourceCount.textContent = "Full Context";

    const startTime = performance.now();

    try {
      const res = await fetch("/documents/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const elapsedMs = Math.round(performance.now() - startTime);
      latencyBadge.textContent = `${elapsedMs} ms`;

      const data = await res.json();

      if (res.ok) {
        // Convert simple markdown bullets and headers to rich HTML
        answerText.innerHTML = formatMarkdown(data.summary);
        sourceCount.textContent = `${data.chunks_used} Chunks Synthesized`;
        sourcesList.innerHTML = `
          <div class="source-card">
            <div class="source-meta">
              <span class="source-doc-name">📄 Scope: ${escapeHtml(data.document)}</span>
              <span class="badge badge-primary">Context: ${data.chunks_used} Chunks</span>
            </div>
            <div class="source-text">
              This summary was synthesized by analyzing all ${data.chunks_used} sequential document sections to produce a holistic executive overview.
            </div>
          </div>
        `;
      } else {
        answerText.textContent = `Error: ${data.detail || 'Failed to generate summary'}`;
        groundedBadge.textContent = "Error";
        groundedBadge.className = "badge badge-warning";
      }
    } catch (err) {
      answerText.textContent = `Network error: ${err.message}`;
    }
  }

  function formatMarkdown(text) {
    if (!text) return "";
    let html = escapeHtml(text);
    // Headers ###
    html = html.replace(/^### (.*$)/gim, '<h4 style="margin: 12px 0 6px; color: var(--accent-cyan);">$1</h4>');
    // Bold **text**
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Bullets - text
    html = html.replace(/^\- (.*$)/gim, '<li style="margin-left: 20px; margin-bottom: 6px;">$1</li>');
    // Line breaks
    html = html.replace(/\n\n/g, '<p style="margin-bottom: 12px;"></p>');
    return html;
  }

  async function executeQuery(question) {
    emptyState.classList.add("hidden");
    resultContainer.classList.remove("hidden");
    
    answerText.textContent = "Searching vectors and generating grounded response...";
    sourcesList.innerHTML = "";
    submitBtn.disabled = true;

    const startTime = performance.now();

    try {
      const res = await fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question })
      });

      const elapsedMs = Math.round(performance.now() - startTime);
      latencyBadge.textContent = `${elapsedMs} ms`;

      const data = await res.json();

      if (res.ok) {
        answerText.textContent = data.answer;
        
        const isUnknown = data.answer.includes("could not be found");
        if (isUnknown) {
          groundedBadge.textContent = "Not Found";
          groundedBadge.className = "badge badge-warning";
        } else {
          groundedBadge.textContent = "100% Grounded";
          groundedBadge.className = "badge badge-primary";
        }

        renderSources(data.sources || []);
      } else {
        answerText.textContent = `Error: ${data.detail || 'Failed to process query'}`;
        groundedBadge.textContent = "Error";
        groundedBadge.className = "badge badge-warning";
      }
    } catch (err) {
      answerText.textContent = `Network error: ${err.message}`;
    } finally {
      submitBtn.disabled = false;
    }
  }

  function renderSources(sources) {
    sourceCount.textContent = sources.length;

    if (sources.length === 0) {
      sourcesList.innerHTML = `<p class="empty-doc-msg">No relevant source chunks passed the similarity threshold.</p>`;
      return;
    }

    sourcesList.innerHTML = "";
    sources.forEach((src) => {
      const card = document.createElement("div");
      card.className = "source-card";
      
      const pageInfo = src.page ? `Page ${src.page}` : 'N/A';
      const simPercent = Math.round(src.similarity * 100);

      card.innerHTML = `
        <div class="source-meta">
          <span class="source-doc-name">📄 ${src.document} (${pageInfo}, Chunk #${src.chunk_id})</span>
          <span class="badge badge-primary">Score: ${src.similarity} (${simPercent}%)</span>
        </div>
        <div class="source-text">${escapeHtml(src.text)}</div>
      `;
      sourcesList.appendChild(card);
    });
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // 4. Manual Modal Controls
  openManualBtn.addEventListener("click", () => manualModal.classList.remove("hidden"));
  closeManualBtn.addEventListener("click", () => manualModal.classList.add("hidden"));
  manualModal.addEventListener("click", (e) => {
    if (e.target === manualModal) manualModal.classList.add("hidden");
  });

});
