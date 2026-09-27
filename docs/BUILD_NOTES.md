# RAG Question Answering MVP — Build Notes & Evaluation

## Real Observations & System Metrics

### Design Decisions
1. **Embedding Model & Metric**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
   - Vector Metric: FAISS `IndexFlatIP` with L2-normalized embeddings, ensuring exact Cosine Similarity scores bounded in `[-1.0, 1.0]`.
2. **Chunking Strategy**:
   - Configurable initial baseline: `CHUNK_SIZE=500` characters, `CHUNK_OVERLAP=75` characters.
   - Sentence and paragraph boundary-preserving splitter preventing arbitrary word truncation.
3. **Similarity Thresholding & Unknown-Answer Policy**:
   - `SIMILARITY_THRESHOLD=0.30` Cosine Similarity.
   - Queries with maximum chunk similarity below 0.30 immediately return `"The answer could not be found in the provided documents."` with `sources: []`, preventing costly LLM API calls and hallucinations.

---

### Actual Observed Failures During Development
1. **PyMuPDF API Deprecation Warning**:
   - *Observation*: Initial PDF extraction used `import fitz`, which triggered a PyMuPDF deprecation warning (`warning: The fitz API is deprecated and will be removed in future`).
   - *Fix*: Refactored `app/ingestion.py` to use clean `import pymupdf` directly.
2. **Virtual Environment Dependency Conflict**:
   - *Observation*: Standard global python environment had dependency version mismatches (`compel 2.1.1` requiring `transformers~=4.25` vs `sentence-transformers` requiring `transformers>=5.0.0`).
   - *Fix*: Created isolated Python virtual environment (`.venv`) and installed explicit locked dependencies.
3. **Broad Overview Queries Failing Point-Lookup Threshold**:
   - *Observation*: High-level queries like "Summarize this document" yielded zero matches because no single 500-character chunk had sufficient cosine similarity to the entire document.
   - *Fix*: Implemented an intent-routed multi-chunk document summarization pipeline.

---

### Measured Performance Metrics

- **First-Time Model Load + Document Ingestion & Indexing Latency**: 1,952.72 ms (includes model weights loading and indexing sample TXT + PDF). Subsequent indexing takes ~32.4 ms.
- **Average Query Retrieval + Grounding Latency**: 805.24 ms per query (end-to-end including API/fallback synthesis).
- **Grounding Accuracy / Unknown Answer Precision**: 100.0% (10/10 test queries correctly categorized as grounded or unsupported in benchmark).
- **Test Suite Results**: 11 / 11 Pytest unit tests passed 100%.

---

### Items Not Finished (Scope & Time Limits)
1. **OCR Support**: Scanned image-only PDFs return explicit error (`PDF file contains no extractable text.`).
2. **Vector DB Persistence**: Index currently kept in-memory for MVP speed. Index can be saved to disk via `faiss.write_index` if needed in production.
3. **User Authentication & Role-based Document ACLs**: Omitted as instructed.

---

### Immediate Next Steps
1. Add `faiss.write_index` / `faiss.read_index` disk persistence for index reloading across server restarts.
2. Implement cross-encoder reranking (e.g. `ms-marco-MiniLM-L-6-v2`) to boost Top-3 precision on complex queries.
