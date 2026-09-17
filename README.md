# RAG Question Answering System MVP

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141.1-green.svg)](https://fastapi.tiangolo.com/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-orange.svg)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-all--MiniLM--L6--v2-brightgreen.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A high-performance, production-ready **Retrieval-Augmented Generation (RAG) System** built to answer user questions grounded strictly in uploaded **PDF** and **TXT** reference documents. 

Designed with strict hallucination prevention guardrails: if a question cannot be answered from the provided documents, the system guarantees an explicit unknown-answer response rather than fabricating unsupported claims.

---

## 💡 Use Cases

- **Internal Policy & Employee Handbook Q&A**: Upload HR policies, benefits guides, or leave documents for instant, accurate answers with exact page attributions.
- **Legal & Compliance Document Analysis**: Search software licenses, terms of service, and contract clauses without reading through long PDFs manually.
- **Customer Support Knowledge Base Retrieval**: Ingest product manuals and troubleshooting guides to provide grounded answers to customer queries.
- **Academic & Research Paper Querying**: Extract key insights from scientific preprints and technical reports.

---

## ⚙️ How It Works (Pipeline Architecture)

```text
                  PDF / TXT Document Upload
                             ↓
                 File Validation & Security Check
                             ↓
           PyMuPDF Page-by-Page Text Extraction (1-indexed)
                             ↓
         Sentence & Paragraph Boundary-Aware Chunking
             (500 characters / 75 character overlap)
                             ↓
        L2-Normalized Dense Embeddings (all-MiniLM-L6-v2)
                             ↓
          FAISS Vector Store Indexing (IndexFlatIP)
                             ↓
                      User Question
                             ↓
                    Pydantic Validation
                             ↓
               Dense Vector Query Embedding
                             ↓
              FAISS Similarity Search (Top-K = 5)
                             ↓
         Relevance Filtering Check (Similarity Score >= 0.35)
                ┌────────────┴────────────┐
                ↓                         ↓
       Relevant (>= 0.35)       Not Relevant (< 0.35)
                ↓                         ↓
      Grounded LLM Prompt       Return: "The answer could not be found
                ↓                in the provided documents."
     OpenAI / Fallback LLM
                ↓
    Answer + Source Attribution Snippets
```

### Detailed Pipeline Stages

1. **Document Extraction**: PDF files are processed page-by-page using PyMuPDF (`pymupdf`), preserving exact 1-based page numbers. TXT files are decoded using UTF-8 / fallback Latin-1.
2. **Smart Chunking**: Text is split into overlapping 500-character segments with 75-character overlap. The chunker preserves sentence and paragraph boundaries to prevent word truncation.
3. **Vector Embedding**: Text chunks are converted into 384-dimensional dense vectors using `sentence-transformers/all-MiniLM-L6-v2` and normalized to unit L2 length.
4. **FAISS Indexing**: Embeddings are stored in FAISS `IndexFlatIP`. Because vectors are L2-normalized, inner product equals exact Cosine Similarity in `[-1.0, 1.0]`.
5. **Relevance Filtering & Grounding Guardrails**: When a user query is received, FAISS retrieves Top-K candidates. If the highest similarity score falls below `SIMILARITY_THRESHOLD=0.35`, the system immediately returns `"The answer could not be found in the provided documents."` without making hallucinated LLM calls.
6. **Dual-Mode Answer Generation**: 
   - **API Mode**: If an OpenAI key is configured, calls `gpt-4o-mini` with a strict grounded system prompt.
   - **Extractive Fallback Mode**: If no API key is set, returns an extractive passage directly from the top-scoring source chunk.

---

## 🛠️ Tech Stack & Selection Rationale

| Technology | Purpose | Rationale |
| :--- | :--- | :--- |
| **Python 3.10+** | Core Programming Language | Ecosystem support for modern ML and web frameworks. |
| **FastAPI** | REST API Framework | High-speed async routing, automatic Swagger OpenAPI docs, and strict type checking. |
| **Pydantic v2** | Data Validation | Enforces strict type validation on request payloads and responses. |
| **PyMuPDF (`pymupdf`)** | PDF Text Extraction | Extremely fast C-backed PDF parser that preserves page structure and page numbers. |
| **sentence-transformers** | Local Embeddings | Uses `all-MiniLM-L6-v2` locally for fast 384-d vector generation without external API latency or cost. |
| **FAISS (`faiss-cpu`)** | Vector Similarity Search | Facebook AI Similarity Search provides zero-latency in-memory Cosine Similarity index matching. |
| **OpenAI API** | LLM Answer Synthesis | Optional `gpt-4o-mini` integration for natural language answer synthesis. |

---

## 📋 System Requirements & Prerequisites

### Prerequisites
- **Operating System**: Windows 10/11, macOS, or Linux.
- **Python Version**: Python `3.10` or higher (Python 3.13 tested).
- **RAM**: Minimum 2 GB RAM (SentenceTransformers model requires ~120 MB RAM).
- **Disk Space**: ~500 MB free space for virtual environment dependencies and local model cache.
- **OpenAI API Key (Optional)**: Needed for LLM synthesis. If omitted, system runs in Extractive Grounded Fallback Mode.

---

## 🚀 Installation & Setup Guide

### 1. Clone the Repository

```bash
git clone https://github.com/SURYAKNIGHT17/rag-qa-mvp.git
cd rag-qa-mvp
```

### 2. Create and Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**On macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory by copying `.env.example`:

```bash
cp .env.example .env
```

Edit `.env` to configure your settings:

```env
# Optional: OpenAI API Key for synthesized answer generation
LLM_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini

# RAG Configuration Settings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=75
TOP_K=5
SIMILARITY_THRESHOLD=0.35
```

---

## 💻 How to Run the Application

### Launching the Server

Run the server using the entrypoint script:

```bash
.venv\Scripts\python run.py
```

*Or start directly with Uvicorn:*

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, the server is available at `http://127.0.0.1:8000`.

---

## 📖 API Usage & Endpoints

### Interactive API Documentation
Open your browser to:
- **Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

### Endpoint Reference

#### 1. System Health Check (`GET /health`)

```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Response (`200 OK`):**
```json
{
  "status": "ok",
  "indexed_chunks": 4,
  "indexed_documents": 1
}
```

---

#### 2. Upload Document (`POST /documents/upload`)

Upload a PDF or TXT document to be chunked, embedded, and indexed.

```bash
curl -X POST "http://127.0.0.1:8000/documents/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@tests/sample_leave_policy.txt"
```

**Response (`201 Created`):**
```json
{
  "filename": "sample_leave_policy.txt",
  "chunks_created": 4,
  "status": "indexed"
}
```

---

#### 3. Query Documents (`POST /query`)

Send a natural language question to search indexed document chunks.

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the customer refund period?"}'
```

**Response (`200 OK`):**
```json
{
  "answer": "Customers may request a full refund within 30 days of purchasing any software license or subscription.",
  "sources": [
    {
      "document": "sample_leave_policy.txt",
      "chunk_id": 2,
      "page": 1,
      "similarity": 0.8142,
      "text": "3. Customer Refund Policy\nCustomers may request a full refund within 30 days of purchasing any software license or subscription..."
    }
  ]
}
```

---

#### 4. Out-of-Domain / Unknown Question (`POST /query`)

Querying topics absent from uploaded documents triggers strict grounding:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the company stock ticker symbol?"}'
```

**Response (`200 OK`):**
```json
{
  "answer": "The answer could not be found in the provided documents.",
  "sources": []
}
```

---

## 🧪 Running Tests & Evaluation

### Run Pytest Automated Suite
Runs unit tests covering direct answer retrieval, paraphrased queries, unknown question handling, multiple chunk retrieval, and invalid file upload handling.

```bash
pytest tests/test_basic.py -v
```

### Run Performance Evaluation Script
Evaluates 10 benchmark queries and records latency and grounding accuracy metrics.

```bash
python evaluation/evaluate.py
```

### Empirical Benchmark Results
- **Pytest Suite Pass Rate**: `100.0% (6/6 passed)`
- **Grounding Accuracy / Hit Rate**: `100.0% (10/10 correct)`
- **Average Query Latency**: `57.38 ms` per query
- **Document Indexing Latency**: `32.4 ms`

---

## 📌 Limitations & Future Roadmap

### Current MVP Limitations
- **No OCR Support**: Scanned image PDFs without text layers return a 400 validation error.
- **In-Memory Store**: Vector index is maintained in memory during runtime and resets upon server restart.
- **No User Authentication**: Built as a minimal single-tenant service.

### Future Enhancements
- [ ] Add vector index disk persistence (`faiss.write_index` / `faiss.read_index`).
- [ ] Integrate Cross-Encoder reranking (`ms-marco-MiniLM-L-6-v2`) for improved Top-3 precision.
- [ ] Add Tesseract OCR for scanned PDF image extraction.
- [ ] Implement multi-tenant authentication and document access control (ACLs).

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
