from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class UploadResponse(BaseModel):
    filename: str
    chunks_created: int
    status: str = "indexed"


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user question to be answered.")

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question must not be empty or whitespace only.")
        return cleaned


class SourceChunk(BaseModel):
    document: str
    chunk_id: int
    page: Optional[int] = None
    similarity: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk] = []


class HealthResponse(BaseModel):
    status: str = "ok"
    indexed_chunks: int = 0
    indexed_documents: int = 0
    documents: List[str] = []


class SummaryRequest(BaseModel):
    document: Optional[str] = Field(None, description="Optional single document name to summarize.")
    documents: Optional[List[str]] = Field(None, description="Optional list of specific document names to summarize.")
    text: Optional[str] = Field(None, description="Optional raw text or profile content to summarize directly.")


class SummaryResponse(BaseModel):
    document: str
    summary: str
    chunks_used: int
    status: str = "success"


class QuerySuggestion(BaseModel):
    label: str
    query: str


class SuggestionsResponse(BaseModel):
    suggestions: List[QuerySuggestion] = []
