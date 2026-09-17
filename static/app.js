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
  const queryPills = document.querySelectorAll(".query-pill");

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

  let indexedDocsSet = new Set();

  // 1. Check System Health
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      if (res.ok) {
        const data = await res.json();
        indexedDocsCount.textContent = data.indexed_documents || indexedDocsSet.size;
        indexedChunksCount.textContent = data.indexed_chunks || 0;
        healthStatusBadge.textContent = "Online";
        healthStatusBadge.className = "badge badge-success";
      }
    } catch (e) {
      healthStatusBadge.textContent = "Offline";
      healthStatusBadge.className = "badge badge-warning";
    }
  }

  fetchHealth();

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
      return;
    }

    docList.innerHTML = "";
    indexedDocsSet.forEach((docName) => {
      const li = document.createElement("li");
      li.className = "doc-item";
      li.innerHTML = `
        <div class="doc-info">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
            <polyline points="14 2 14 8 20 8"/>
          </svg>
          <span class="doc-name" title="${docName}">${docName}</span>
        </div>
        <span class="badge badge-success">Indexed</span>
      `;
      docList.appendChild(li);
    });
  }

  // 3. Query Submission
  queryForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const q = questionInput.value.trim();
    if (q) {
      executeQuery(q);
    }
  });

  queryPills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const q = pill.getAttribute("data-query");
      questionInput.value = q;
      executeQuery(q);
    });
  });

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
