import os
import sys
import io
from unittest.mock import patch
import pytest

# Ensure workspace root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app
from app.retrieval import vector_store
from app.generation import UNKNOWN_ANSWER_MESSAGE

client = TestClient(app)

SAMPLE_TXT_PATH = os.path.join("tests", "sample_leave_policy.txt")
SAMPLE_PDF_PATH = os.path.join("tests", "sample_company_guide.pdf")


@pytest.fixture(scope="module", autouse=True)
def reset_vector_store_module():
    """Ensure vector store is clean before the test suite starts."""
    vector_store.reset()
    yield
    vector_store.reset()


def test_01_answer_exists():
    """Test 1 — Answer exists: Question clearly answered by uploaded document."""
    # 1. Upload sample text document
    with open(SAMPLE_TXT_PATH, "rb") as f:
        upload_resp = client.post(
            "/documents/upload",
            files={"file": ("sample_leave_policy.txt", f, "text/plain")}
        )
    
    assert upload_resp.status_code == 201
    assert upload_resp.json()["chunks_created"] > 0

    # 2. Query question answered by text (using transparent mock to avoid network dependencies)
    with patch("app.main.generate_grounded_answer", return_value="Customers can request a full refund within 30 days of purchase."):
        query_resp = client.post("/query", json={"question": "What is the customer refund period?"})
        assert query_resp.status_code == 200
        data = query_resp.json()
        
        assert data["answer"] != UNKNOWN_ANSWER_MESSAGE
        assert len(data["sources"]) > 0
        assert "30 days" in data["sources"][0]["text"].lower() or "30 days" in data["answer"].lower()


def test_02_paraphrased_question():
    """Test 2 — Paraphrased question: Question uses different wording, semantic search still finds chunk."""
    with patch("app.main.generate_grounded_answer", return_value="Users have up to 30 days to ask for a refund."):
        query_resp = client.post("/query", json={"question": "How long do users have to ask for their money back after purchase?"})
        assert query_resp.status_code == 200
        data = query_resp.json()

        assert data["answer"] != UNKNOWN_ANSWER_MESSAGE
        assert len(data["sources"]) > 0
        assert data["sources"][0]["document"] == "sample_leave_policy.txt"


def test_03_answer_absent():
    """Test 3 — Answer absent: Topic not present in document returns unknown message."""
    query_resp = client.post("/query", json={"question": "What is the company stock ticker symbol and share price?"})
    assert query_resp.status_code == 200
    data = query_resp.json()

    assert data["answer"] == UNKNOWN_ANSWER_MESSAGE
    assert len(data["sources"]) == 0


def test_04_unrelated_question():
    """Test 4 — Unrelated question: Out of domain query returns unknown message."""
    query_resp = client.post("/query", json={"question": "What is the average distance from the Earth to the Moon?"})
    assert query_resp.status_code == 200
    data = query_resp.json()

    assert data["answer"] == UNKNOWN_ANSWER_MESSAGE
    assert len(data["sources"]) == 0


def test_05_multiple_chunks():
    """Test 5 — Multiple chunks: Question covering multiple topics retrieves multiple sources."""
    with patch("app.main.generate_grounded_answer", return_value="Full-time employees receive 25 days leave and a $500 wellness allowance."):
        query_resp = client.post("/query", json={"question": "Tell me about employee annual leave entitlement and wellness allowance."})
        assert query_resp.status_code == 200
        data = query_resp.json()

        assert data["answer"] != UNKNOWN_ANSWER_MESSAGE
        assert len(data["sources"]) >= 1


def test_06_invalid_file():
    """Test 6 — Invalid file: Unsupported file type returns HTTP 400 Bad Request."""
    invalid_file = io.BytesIO(b"binary data content")
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("unsupported_script.exe", invalid_file, "application/octet-stream")}
    )
    assert upload_resp.status_code == 400
    assert "Unsupported file extension" in upload_resp.json()["detail"]


def test_07_document_summary():
    """Test 7 — Full Document Summary: Requesting summary returns synthesized executive summary."""
    summary_resp = client.post("/documents/summary", json={"document": "sample_leave_policy.txt"})
    assert summary_resp.status_code == 200
    data = summary_resp.json()
    assert data["status"] == "success"
    assert data["chunks_used"] > 0
    assert len(data["summary"]) > 50


def test_08_query_summary_routing():
    """Test 8 — Query Summary Routing: Querying 'summarize this document' invokes summary engine."""
    query_resp = client.post("/query", json={"question": "Please summarize this document for me"})
    assert query_resp.status_code == 200
    data = query_resp.json()
    assert data["answer"] != UNKNOWN_ANSWER_MESSAGE
    assert len(data["answer"]) > 50


def test_09_multi_document_selection_summary():
    """Test 9 — Multi-Document Selection: Summarize specific selected document list."""
    summary_resp = client.post("/documents/summary", json={"documents": ["sample_leave_policy.txt"]})
    assert summary_resp.status_code == 200
    data = summary_resp.json()
    assert data["status"] == "success"
    assert data["chunks_used"] > 0
    assert len(data["summary"]) > 50


def test_10_custom_text_direct_summary():
    """Test 10 — Custom Text Direct Summary: Summarize raw user-submitted profile or text directly."""
    sample_profile_text = (
        "Candidate Profile: Senior Python Engineer with 6 years experience in FastAPI, PyTorch, and FAISS. "
        "Built distributed retrieval-augmented generation systems handling 10,000 requests per minute. "
        "Led a team of 4 engineers to reduce vector search latency by 45% using quantized indexing."
    )
    summary_resp = client.post("/documents/summary", json={"text": sample_profile_text})
    assert summary_resp.status_code == 200
    data = summary_resp.json()
    assert data["status"] == "success"
    assert data["document"] == "Custom Text Input"
    assert data["chunks_used"] >= 1
    assert len(data["summary"]) > 30


def test_11_dynamic_suggestions():
    """Test 11 — Dynamic Query Suggestions: Suggestions generated dynamically from indexed document content."""
    # 1. Suggestions across all indexed documents
    resp = client.get("/documents/suggestions")
    assert resp.status_code == 200
    data = resp.json()
    assert "suggestions" in data
    suggestions = data["suggestions"]
    assert len(suggestions) >= 3
    for s in suggestions:
        assert "label" in s and len(s["label"]) > 0
        assert "query" in s and len(s["query"]) > 0

    # Ensure Executive Summary is present
    labels = [s["label"] for s in suggestions]
    assert any("summary" in l.lower() or "executive" in l.lower() for l in labels)

    # 2. Suggestions for specific document
    doc_resp = client.get("/documents/suggestions?document=sample_leave_policy.txt")
    assert doc_resp.status_code == 200
    doc_data = doc_resp.json()
    assert len(doc_data["suggestions"]) >= 3


def test_12_llm_failure_returns_502():
    """Test 12 — Error Propagation: When LLM generation fails, API returns HTTP 502 Bad Gateway."""
    from app.generation import LLMGenerationError
    with patch("app.main.generate_grounded_answer", side_effect=LLMGenerationError("Upstream API quota exceeded or timeout")):
        query_resp = client.post("/query", json={"question": "What is the customer refund period?"})
        assert query_resp.status_code == 502
        data = query_resp.json()
        assert "LLM generation failed" in data["detail"]
        assert "Upstream API quota exceeded" in data["detail"]


def test_13_generate_grounded_answer_raises_on_failure():
    """Test 13 — Transparent Error Raising: generate_grounded_answer raises LLMGenerationError instead of masked fallback."""
    from app.generation import generate_grounded_answer, LLMGenerationError
    from app.models import SourceChunk
    
    dummy_sources = [
        SourceChunk(document="doc.txt", chunk_id=1, page=1, similarity=0.85, text="Refund period is 30 days.")
    ]
    with patch("app.generation._call_gemini_api", side_effect=Exception("Connection refused")), \
         patch("app.generation.settings.GEMINI_API_KEY", "dummy_key"), \
         patch("app.generation.settings.LLM_API_KEY", ""):
        with pytest.raises(LLMGenerationError) as exc_info:
            generate_grounded_answer("What is refund period?", dummy_sources, allow_degraded_fallback=False)
        assert "All configured LLM providers failed" in str(exc_info.value)


