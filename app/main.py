import os
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceChunk,
    SummaryRequest,
    SummaryResponse,
    SuggestionsResponse
)
from app.ingestion import extract_document
from app.chunking import chunk_document, split_text_into_chunks, ChunkRecord
from app.retrieval import vector_store
from app.generation import (
    generate_grounded_answer,
    generate_document_summary,
    generate_suggested_queries,
    is_summary_query,
    UNKNOWN_ANSWER_MESSAGE
)
from app.config import settings

app = FastAPI(
    title="RAG Question Answering MVP",
    description="Production-ready grounded RAG QA system for PDF and TXT documents.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files Directory
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    """
    Serves the modern RAG QA Web Application UI.
    """
    if os.path.exists("static/index.html"):
        return FileResponse("static/index.html")
    return {
        "message": "RAG Question Answering System API is running.",
        "docs_url": "http://127.0.0.1:8000/docs",
        "health_url": "http://127.0.0.1:8000/health",
        "status": "online"
    }


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
def health_check():
    """
    Health check endpoint returning system status, indexed chunk count, and document list.
    """
    return HealthResponse(
        status="ok",
        indexed_chunks=vector_store.total_chunks,
        indexed_documents=vector_store.total_documents,
        documents=sorted(list(vector_store.indexed_documents))
    )


@app.post("/documents/reset")
def reset_documents():
    """
    Clears all indexed documents, chunks, and FAISS vectors to start fresh.
    """
    vector_store.reset()
    return {
        "status": "reset",
        "message": "Vector store and indexed documents have been cleared.",
        "indexed_chunks": 0,
        "indexed_documents": 0
    }


@app.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts PDF or TXT file, extracts text, chunks it, generates embeddings,
    and indexes the vectors into FAISS.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is missing."
        )

    filename = file.filename
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {str(e)}"
        )

    # 1. Extraction
    try:
        pages_text = extract_document(filename, content)
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing document extraction: {str(e)}"
        )

    # 2. Chunking
    try:
        chunks = chunk_document(
            filename=filename,
            pages_text=pages_text,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            start_chunk_id=vector_store.total_chunks
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to chunk document: {str(e)}"
        )

    # 3. Embedding & Indexing
    try:
        chunks_indexed = vector_store.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embeddings or index document: {str(e)}"
        )

    return UploadResponse(
        filename=filename,
        chunks_created=chunks_indexed,
        status="indexed"
    )


@app.post("/documents/summary", response_model=SummaryResponse)
def summarize_document(request: SummaryRequest):
    """
    Synthesizes a comprehensive executive summary across selected documents,
    all documents, or custom text provided by the user.
    """
    # 1. Custom text direct summarization
    if request.text and request.text.strip():
        raw_text = request.text.strip()
        text_chunks = split_text_into_chunks(raw_text, chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
        chunk_records = [
            ChunkRecord(document="Custom Text Input", chunk_id=i, page=1, text=chunk)
            for i, chunk in enumerate(text_chunks)
        ]
        summary = generate_document_summary(chunk_records, "Custom Text Input")
        return SummaryResponse(
            document="Custom Text Input",
            summary=summary,
            chunks_used=len(chunk_records),
            status="success"
        )

    # 2. Document-based summarization (selected files or all files)
    if vector_store.total_chunks == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents have been indexed yet. Upload a document or provide text to summarize."
        )

    target_docs = request.documents or ([request.document] if request.document else None)
    chunks = vector_store.get_document_chunks(target_docs)

    if not chunks:
        label = ", ".join(target_docs) if target_docs else "specified documents"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for {label} in the index."
        )

    if target_docs:
        doc_label = ", ".join(target_docs)
    else:
        doc_label = "All Indexed Documents" if len(vector_store.indexed_documents) > 1 else (chunks[0].document if chunks else "Document")

    summary = generate_document_summary(chunks, doc_label)

    return SummaryResponse(
        document=doc_label,
        summary=summary,
        chunks_used=len(chunks),
        status="success"
    )


@app.get("/documents/suggestions", response_model=SuggestionsResponse)
def get_suggestions(document: Optional[str] = None):
    """
    Returns dynamically generated high-value suggested queries tailored to the indexed documents.
    """
    chunks = vector_store.get_document_chunks(document) if vector_store.total_chunks > 0 else []
    suggestions = generate_suggested_queries(chunks)
    return SuggestionsResponse(suggestions=suggestions)


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Accepts user question, searches FAISS vector store, applies relevance thresholding,
    and returns a grounded answer with source chunk attribution.
    Automatically handles summary queries across the full document context.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty or whitespace only."
        )

    # If no documents are indexed yet
    if vector_store.total_chunks == 0:
        return QueryResponse(
            answer=UNKNOWN_ANSWER_MESSAGE,
            sources=[]
        )

    # 1. Check if user is asking for a document summary or overview
    if is_summary_query(question):
        chunks = vector_store.get_document_chunks()
        if chunks:
            doc_name = chunks[0].document if len(vector_store.indexed_documents) == 1 else "All Documents"
            summary = generate_document_summary(chunks, doc_name)
            # Build sample source attribution chunks
            sample_sources = [
                SourceChunk(
                    document=c.document,
                    chunk_id=c.chunk_id,
                    page=c.page,
                    similarity=1.0,
                    text=c.text
                )
                for c in chunks[:3]
            ]
            return QueryResponse(answer=summary, sources=sample_sources)

    # 2. Vector similarity search + relevance filtering
    try:
        sources = vector_store.search(
            query=question,
            top_k=settings.TOP_K,
            similarity_threshold=settings.SIMILARITY_THRESHOLD
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval error: {str(e)}"
        )

    # 3. Grounded LLM generation
    try:
        answer = generate_grounded_answer(question, sources)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer generation error: {str(e)}"
        )

    return QueryResponse(
        answer=answer,
        sources=sources if answer != UNKNOWN_ANSWER_MESSAGE else []
    )
