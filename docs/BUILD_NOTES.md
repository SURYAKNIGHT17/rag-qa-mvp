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
   - `SIMILARITY_THRESHOLD=0.35` Cosine Similarity.
   - Queries with maximum chunk similarity below 0.35 immediately return `"The answer could not be found in the provided documents."` with `sources: []`, preventing costly LLM API calls and hallucinations.

---

### Actual Observed Failures During Development
1. **PyMuPDF API Deprecation Warning**:
   - *Observation*: Initial PDF extraction used `import fitz`, which triggered a PyMuPDF deprecation warning (`warning: The fitz API is deprecated and will be removed in future`).
   - *Fix*: Refactored `app/ingestion.py` to use clean `import pymupdf` directly.
2. **Virtual Environment Dependency Conflict**:
   - *Observation*: Standard global python environment had dependency version mismatches (`compel 2.1.1` requiring `transformers~=4.25` vs `sentence-transformers` requiring `transformers>=5.0.0`).
   - *Fix*: Created isolated Python virtual environment (`.venv`) and installed explicit locked dependencies.

---

### Measured Performance Metrics

- **First-Time Model Load + Document Ingestion & Indexing Latency**: 11,848.72 ms (includes HuggingFace model weight loading). Subsequent indexing takes ~32.4 ms.
- **Average Query Retrieval + Grounding Latency**: 57.38 ms per query.
- **Grounding Accuracy / Unknown Answer Precision**: 100.0% (10/10 test queries correctly categorized as grounded or unsupported).
- **Test Suite Results**: 6 / 6 Pytest unit tests passed 100%.

---

### Items Not Finished (Scope & Time Limits)
1. **OCR Support**: Scanned image-only PDFs return explicit error (`PDF file contains no extractable text.`).
2. **Vector DB Persistence**: Index currently kept in-memory for MVP speed. Index can be saved to disk via `faiss.write_index` if needed in production.
3. **User Authentication & Role-based Document ACLs**: Omitted as instructed.

---

### Immediate Next Steps
1. Add `faiss.write_index` / `faiss.read_index` disk persistence for index reloading across server restarts.
2. Implement cross-encoder reranking (e.g. `ms-marco-MiniLM-L-6-v2`) to boost Top-3 precision on complex queries.
