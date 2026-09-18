import time
import os
import sys
from typing import List, Dict, Any

# Ensure workspace root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app
from app.generation import UNKNOWN_ANSWER_MESSAGE
from app.retrieval import vector_store

client = TestClient(app)

EVAL_QUESTIONS = [
    {
        "question": "What is the annual paid leave entitlement for full-time employees?",
        "should_find": True,
        "expected_keyword": "25 working days"
    },
    {
        "question": "How many weeks of parental leave do primary caregivers get?",
        "should_find": True,
        "expected_keyword": "16 weeks"
    },
    {
        "question": "Within how many days can customers request a full refund?",
        "should_find": True,
        "expected_keyword": "30 days"
    },
    {
        "question": "What is the annual wellness allowance amount for employees?",
        "should_find": True,
        "expected_keyword": "$500"
    },
    {
        "question": "What remote work hours are specified for core availability?",
        "should_find": True,
        "expected_keyword": "10:00 AM"
    },
    {
        "question": "How long must passwords be according to the cybersecurity policy?",
        "should_find": True,
        "expected_keyword": "16 characters"
    },
    {
        "question": "What is the company stock ticker symbol and current share price?",
        "should_find": False,
        "expected_keyword": None
    },
    {
        "question": "What is the recipe for baking chocolate chip cookies?",
        "should_find": False,
        "expected_keyword": None
    },
    {
        "question": "Where is the company headquarters office located?",
        "should_find": False,
        "expected_keyword": None
    },
    {
        "question": "What is the minimum age requirement for job applicants?",
        "should_find": False,
        "expected_keyword": None
    }
]


def run_evaluation() -> Dict[str, Any]:
    print("=== STARTING RAG SYSTEM EVALUATION ===")
    vector_store.reset()
    
    # 1. Measure Document Ingestion Latency
    txt_path = os.path.join("tests", "sample_leave_policy.txt")
    pdf_path = os.path.join("tests", "sample_company_guide.pdf")

    start_ingest = time.perf_counter()
    with open(txt_path, "rb") as f:
        resp1 = client.post("/documents/upload", files={"file": ("sample_leave_policy.txt", f, "text/plain")})
    
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            resp2 = client.post("/documents/upload", files={"file": ("sample_company_guide.pdf", f, "application/pdf")})
    ingest_latency_ms = (time.perf_counter() - start_ingest) * 1000.0

    print(f"Ingestion & Indexing Latency: {ingest_latency_ms:.2f} ms")

    # 2. Evaluate Query Performance
    correct_retrievals = 0
    total_latency_ms = 0.0
    eval_results = []

    for idx, item in enumerate(EVAL_QUESTIONS, 1):
        q = item["question"]
        should_find = item["should_find"]
        
        t0 = time.perf_counter()
        resp = client.post("/query", json={"question": q})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        if should_find:
            is_success = (answer != UNKNOWN_ANSWER_MESSAGE) and (len(sources) > 0)
        else:
            is_success = (answer == UNKNOWN_ANSWER_MESSAGE) and (len(sources) == 0)

        if is_success:
            correct_retrievals += 1

        eval_results.append({
            "id": idx,
            "question": q,
            "should_find": should_find,
            "success": is_success,
            "latency_ms": round(elapsed_ms, 2),
            "source_count": len(sources)
        })

    avg_latency_ms = total_latency_ms / len(EVAL_QUESTIONS)
    accuracy = (correct_retrievals / len(EVAL_QUESTIONS)) * 100.0

    print("\n--- SUMMARY RESULTS ---")
    print(f"Total Evaluated Questions: {len(EVAL_QUESTIONS)}")
    print(f"Hit Accuracy / Grounding Correctness: {accuracy:.1f}% ({correct_retrievals}/{len(EVAL_QUESTIONS)})")
    print(f"Average Query Latency: {avg_latency_ms:.2f} ms")

    return {
        "ingest_latency_ms": round(ingest_latency_ms, 2),
        "avg_query_latency_ms": round(avg_latency_ms, 2),
        "accuracy_percent": accuracy,
        "correct_count": correct_retrievals,
        "total_count": len(EVAL_QUESTIONS),
        "results": eval_results
    }


if __name__ == "__main__":
    run_evaluation()
