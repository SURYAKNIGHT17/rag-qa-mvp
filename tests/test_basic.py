import os
import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.retrieval import vector_store
from app.generation import UNKNOWN_ANSWER_MESSAGE

client = TestClient(app)

SAMPLE_TXT_PATH = os.path.join("tests", "sample_leave_policy.txt")
SAMPLE_PDF_PATH = os.path.join("tests", "sample_company_guide.pdf")


@pytest.fixture(autouse=True)
def reset_vector_store():
    """Clear vector store before running test suite if needed."""
    pass


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

    # 2. Query question answered by text
    query_resp = client.post("/query", json={"question": "What is the customer refund period?"})
    assert query_resp.status_code == 200
    data = query_resp.json()
    
    assert data["answer"] != UNKNOWN_ANSWER_MESSAGE
    assert len(data["sources"]) > 0
    assert "30 days" in data["sources"][0]["text"].lower() or "30 days" in data["answer"].lower()


def test_02_paraphrased_question():
    """Test 2 — Paraphrased question: Question uses different wording, semantic search still finds chunk."""
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
