import time
import os
import sys
from typing import List, Dict, Any, Optional

# Ensure workspace root is on Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app.main import app
from app.generation import UNKNOWN_ANSWER_MESSAGE
from app.retrieval import vector_store

client = TestClient(app)

EVAL_TEST_SUITE = [
    # Category 1: In-Domain Factual Questions (Verified against ground-truth document facts)
    {
        "id": 1,
        "category": "factual_in_domain",
        "question": "What is the annual paid leave entitlement for full-time employees?",
        "should_find": True,
        "expected_facts": ["25 working days", "25 days"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Annual leave quota verification"
    },
    {
        "id": 2,
        "category": "factual_in_domain",
        "question": "How many weeks of parental leave do primary caregivers get?",
        "should_find": True,
        "expected_facts": ["16 weeks"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Primary caregiver leave policy"
    },
    {
        "id": 3,
        "category": "factual_in_domain",
        "question": "Within how many days can customers request a full refund?",
        "should_find": True,
        "expected_facts": ["30 days"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Customer refund window"
    },
    {
        "id": 4,
        "category": "factual_in_domain",
        "question": "What is the annual wellness allowance amount for employees?",
        "should_find": True,
        "expected_facts": ["$500", "500"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Employee wellness stipend"
    },
    {
        "id": 5,
        "category": "factual_in_domain",
        "question": "How many weeks of paid leave do secondary caregivers receive?",
        "should_find": True,
        "expected_facts": ["4 weeks"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Secondary caregiver leave policy"
    },
    {
        "id": 6,
        "category": "factual_in_domain",
        "question": "How many days in advance must annual leave be requested?",
        "should_find": True,
        "expected_facts": ["14 days"],
        "expected_doc": "sample_leave_policy.txt",
        "description": "Advance notice requirement"
    },
    {
        "id": 7,
        "category": "factual_in_domain",
        "question": "How many days per week are employees permitted to work remotely?",
        "should_find": True,
        "expected_facts": ["3 days"],
        "expected_doc": "sample_company_guide.pdf",
        "description": "Remote work policy limit"
    },
    {
        "id": 8,
        "category": "factual_in_domain",
        "question": "What authentication is mandatory for corporate systems?",
        "should_find": True,
        "expected_facts": ["multi-factor authentication", "mfa"],
        "expected_doc": "sample_company_guide.pdf",
        "description": "Cybersecurity MFA mandate"
    },

    # Category 2: Out-of-Domain Questions (Must be refused without hallucination)
    {
        "id": 9,
        "category": "out_of_domain",
        "question": "What is the company stock ticker symbol and current share price?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Stock market inquiry outside document scope"
    },
    {
        "id": 10,
        "category": "out_of_domain",
        "question": "What is the recipe for baking chocolate chip cookies?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Completely irrelevant general query"
    },
    {
        "id": 11,
        "category": "out_of_domain",
        "question": "Where is the company global headquarters office located?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Corporate headquarters query absent from docs"
    },

    # Category 3: Adversarial Vocabulary Distractors (Partial lexical overlap, absent facts)
    {
        "id": 12,
        "category": "distractor_refusal",
        "question": "What is the annual paid leave entitlement for independent contractors and interns?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Lexical overlap (leave) but targeting unmentioned worker types"
    },
    {
        "id": 13,
        "category": "distractor_refusal",
        "question": "What is the 401(k) retirement matching percentage for full-time employees?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Benefits overlap but 401(k) matching is absent"
    },
    {
        "id": 14,
        "category": "distractor_refusal",
        "question": "What is the company policy for company car rentals and parking spaces?",
        "should_find": False,
        "expected_facts": [],
        "expected_doc": None,
        "description": "Workplace policy overlap with unmentioned parking topic"
    }
]


def verify_answer_and_citations(
    test_case: Dict[str, Any],
    answer: str,
    sources: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Rigorously verifies:
    1. Factual correctness of the answer against ground-truth facts.
    2. Citation support: whether cited chunks contain verifiable evidence.
    3. Proper refusal for unanswerable / distractor questions.
    """
    should_find = test_case["should_find"]
    expected_facts = [f.lower() for f in test_case["expected_facts"]]
    expected_doc = test_case.get("expected_doc")

    # If question is unanswerable or a distractor
    if not should_find:
        is_refusal = (UNKNOWN_ANSWER_MESSAGE in answer)
        no_hallucinated_sources = (len(sources) == 0)
        passed = is_refusal and no_hallucinated_sources
        failure_reasons = []
        if not is_refusal:
            failure_reasons.append("Failed to return unknown answer refusal message for unanswerable query.")
        if not no_hallucinated_sources:
            failure_reasons.append(f"Expected 0 sources, but returned {len(sources)} source chunk(s).")
        return {
            "passed": passed,
            "factual_match": is_refusal,
            "citation_grounded": no_hallucinated_sources,
            "failure_reasons": failure_reasons
        }

    # For answerable questions:
    failure_reasons = []

    # 1. Answer non-empty and not refused
    if answer == UNKNOWN_ANSWER_MESSAGE or not answer.strip():
        failure_reasons.append("System erroneously refused an in-domain question.")

    # 2. Factual check in generated answer
    answer_lower = answer.lower()
    factual_match = any(fact in answer_lower for fact in expected_facts)
    if not factual_match:
        failure_reasons.append(f"Answer does not contain expected factual terms: {test_case['expected_facts']}")

    # 3. Citation Support & Source Grounding
    if len(sources) == 0:
        failure_reasons.append("No source chunks returned to support the answer.")
        citation_grounded = False
    else:
        # Check if at least one source chunk contains the expected factual term
        chunk_texts_combined = " ".join([s.get("text", "").lower() for s in sources])
        chunk_fact_match = any(fact in chunk_texts_combined for fact in expected_facts)
        doc_match = any(s.get("document", "") == expected_doc for s in sources) if expected_doc else True

        if not chunk_fact_match:
            failure_reasons.append("Retrieved source chunks do not contain the verified ground truth fact.")
        if not doc_match:
            failure_reasons.append(f"Retrieved sources do not cite expected document '{expected_doc}'.")

        citation_grounded = chunk_fact_match and doc_match

    passed = (len(failure_reasons) == 0)
    return {
        "passed": passed,
        "factual_match": factual_match,
        "citation_grounded": citation_grounded,
        "failure_reasons": failure_reasons
    }


def run_evaluation() -> Dict[str, Any]:
    print("=" * 70)
    print("RIGOROUS RAG EVALUATION SUITE: Grounding, Facts, and Distractors")
    print("=" * 70)

    # 1. Reset vector store
    vector_store.reset()

    # 2. Ingestion latency benchmark
    txt_path = os.path.join("tests", "sample_leave_policy.txt")
    pdf_path = os.path.join("tests", "sample_company_guide.pdf")

    start_ingest = time.perf_counter()
    with open(txt_path, "rb") as f:
        resp1 = client.post("/documents/upload", files={"file": ("sample_leave_policy.txt", f, "text/plain")})
    assert resp1.status_code == 201, f"Failed to upload {txt_path}: {resp1.text}"

    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            resp2 = client.post("/documents/upload", files={"file": ("sample_company_guide.pdf", f, "application/pdf")})
        assert resp2.status_code == 201, f"Failed to upload {pdf_path}: {resp2.text}"

    ingest_latency_ms = (time.perf_counter() - start_ingest) * 1000.0
    print(f"\n[+] Ingestion Benchmark: Ingested documents in {ingest_latency_ms:.2f} ms")
    print(f"[+] Total Indexed Chunks: {vector_store.total_chunks}")
    print(f"[+] Total Indexed Documents: {vector_store.total_documents}\n")

    # 3. Query evaluation
    total_latency_ms = 0.0
    eval_results = []
    factual_correct_count = 0
    citation_grounded_count = 0
    refusal_correct_count = 0
    in_domain_total = sum(1 for q in EVAL_TEST_SUITE if q["should_find"])
    out_of_domain_total = sum(1 for q in EVAL_TEST_SUITE if not q["should_find"])

    print(f"{'ID':<4} | {'Category':<20} | {'Status':<6} | {'Lat(ms)':<8} | {'Question'}")
    print("-" * 75)

    for item in EVAL_TEST_SUITE:
        q_id = item["id"]
        q_text = item["question"]
        cat = item["category"]
        should_find = item["should_find"]

        t0 = time.perf_counter()
        resp = client.post("/query", json={"question": q_text, "allow_fallback": True})
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        data = resp.json()
        answer = data.get("answer", "")
        sources = data.get("sources", [])

        verification = verify_answer_and_citations(item, answer, sources)
        passed = verification["passed"]

        if should_find:
            if verification["factual_match"]:
                factual_correct_count += 1
            if verification["citation_grounded"]:
                citation_grounded_count += 1
        else:
            if passed:
                refusal_correct_count += 1

        status_str = "PASS" if passed else "FAIL"
        print(f"{q_id:<4} | {cat:<20} | {status_str:<6} | {elapsed_ms:<8.1f} | {q_text[:35]}...")

        if not passed:
            for r in verification["failure_reasons"]:
                print(f"      [!] Failure: {r}")

        eval_results.append({
            "id": q_id,
            "category": cat,
            "question": q_text,
            "should_find": should_find,
            "passed": passed,
            "factual_match": verification["factual_match"],
            "citation_grounded": verification["citation_grounded"],
            "failure_reasons": verification["failure_reasons"],
            "latency_ms": round(elapsed_ms, 2),
            "sources_returned": len(sources)
        })

    total_tests = len(EVAL_TEST_SUITE)
    overall_passed = sum(1 for r in eval_results if r["passed"])
    overall_accuracy = (overall_passed / total_tests) * 100.0
    factual_accuracy = (factual_correct_count / in_domain_total) * 100.0 if in_domain_total > 0 else 0.0
    citation_accuracy = (citation_grounded_count / in_domain_total) * 100.0 if in_domain_total > 0 else 0.0
    refusal_accuracy = (refusal_correct_count / out_of_domain_total) * 100.0 if out_of_domain_total > 0 else 0.0
    avg_latency = total_latency_ms / total_tests

    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY & METRICS")
    print("=" * 70)
    print(f"Total Test Cases Evaluated:       {total_tests}")
    print(f"Overall Test Pass Rate:           {overall_accuracy:.1f}% ({overall_passed}/{total_tests})")
    print(f"Factual Correctness Rate:         {factual_accuracy:.1f}% ({factual_correct_count}/{in_domain_total})")
    print(f"Citation Grounding Rate:          {citation_accuracy:.1f}% ({citation_grounded_count}/{in_domain_total})")
    print(f"Distractor / Refusal Accuracy:    {refusal_accuracy:.1f}% ({refusal_correct_count}/{out_of_domain_total})")
    print(f"Average Query Latency:            {avg_latency:.2f} ms")
    print(f"Ingestion & Indexing Latency:     {ingest_latency_ms:.2f} ms")
    print("=" * 70)

    return {
        "total_tests": total_tests,
        "overall_accuracy_percent": round(overall_accuracy, 2),
        "factual_accuracy_percent": round(factual_accuracy, 2),
        "citation_accuracy_percent": round(citation_accuracy, 2),
        "refusal_accuracy_percent": round(refusal_accuracy, 2),
        "avg_query_latency_ms": round(avg_latency, 2),
        "ingest_latency_ms": round(ingest_latency_ms, 2),
        "results": eval_results
    }


if __name__ == "__main__":
    run_evaluation()
