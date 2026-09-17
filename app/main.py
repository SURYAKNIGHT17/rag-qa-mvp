from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.models import (
    UploadResponse,
    QueryRequest,
    QueryResponse,
    HealthResponse,
    SourceChunk
)
from app.ingestion import extract_document
from app.chunking import chunk_document
from app.retrieval import vector_store
from app.generation import generate_grounded_answer, UNKNOWN_ANSWER_MESSAGE
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


@app.api_route("/", methods=["GET", "HEAD"])
def root():
    """
    Root endpoint for browser sanity checks and navigation.
    """
    return {
        "message": "RAG Question Answering System API is running.",
        "docs_url": "http://127.0.0.1:8000/docs",
        "health_url": "http://127.0.0.1:8000/health",
        "status": "online"
    }


@app.api_route("/health", methods=["GET", "HEAD"], response_model=HealthResponse)
def health_check():
    """
    Health check endpoint returning system status and indexed chunk count.
    """
    return HealthResponse(
        status="ok",
        indexed_chunks=vector_store.total_chunks,
        indexed_documents=vector_store.total_documents
    )


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


@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    """
    Accepts user question, searches FAISS vector store, applies relevance thresholding,
    and returns a grounded answer with source chunk attribution.
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

    # 1. Vector similarity search + relevance filtering
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

    # 2. Grounded LLM generation
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
