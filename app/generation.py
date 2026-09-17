from typing import List
import os
import openai
from app.config import settings
from app.models import SourceChunk

UNKNOWN_ANSWER_MESSAGE = "The answer could not be found in the provided documents."

GROUNDED_PROMPT_TEMPLATE = """You are a document question-answering assistant.

Answer the user's question using ONLY the provided document context.

Do not use outside knowledge.

Do not invent or infer unsupported facts.

If the provided context does not contain enough information to answer the question, say:

"The answer could not be found in the provided documents."

Context:
{context}

Question:
{question}
"""


def generate_grounded_answer(question: str, sources: List[SourceChunk]) -> str:
    """
    Generates a grounded answer using retrieved source chunks.
    Supports Google Gemini API, OpenAI API, or Extractive Fallback.
    """
    if not sources:
        return UNKNOWN_ANSWER_MESSAGE

    context_str = "\n\n".join(
        [f"[Doc: {s.document}, Page: {s.page or 'N/A'}, Chunk: {s.chunk_id}]\n{s.text}" for s in sources]
    )

    prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context_str, question=question)

    gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    openai_key = settings.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")

    # 1. Google Gemini API
    if gemini_key:
        try:
            return _call_gemini_api(prompt, gemini_key)
        except Exception as e:
            return _extractive_fallback_answer(question, sources, error_note=f"Gemini API Error: {str(e)}")

    # 2. OpenAI API
    if openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document Q&A assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return _extractive_fallback_answer(question, sources, error_note=f"OpenAI API Error: {str(e)}")

    # 3. Local Extractive Fallback Engine (when no API key is provided)
    return _extractive_fallback_answer(question, sources)


def _call_gemini_api(prompt: str, api_key: str) -> str:
    """
    Calls Google Gemini REST API using httpx.
    """
    import httpx
    model_name = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 500
        }
    }

    with httpx.Client(timeout=15.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates and "content" in candidates[0]:
            parts = candidates[0]["content"].get("parts", [])
            if parts and "text" in parts[0]:
                return parts[0]["text"].strip()

    raise ValueError("Invalid response structure from Gemini API.")


def _extractive_fallback_answer(question: str, sources: List[SourceChunk], error_note: str = "") -> str:
    """
    Provides a grounded fallback answer extracted directly from top source chunk
    when no API key is provided or when an LLM API error occurs.
    """
    if not sources:
        return UNKNOWN_ANSWER_MESSAGE

    top_source = sources[0]
    # Simple verification that query terms overlap with text
    q_words = set(question.lower().split()) - {"what", "is", "the", "a", "an", "of", "in", "to", "for", "and", "or", "how", "why", "where", "who"}
    source_words = set(top_source.text.lower().split())

    if q_words and not (q_words & source_words):
        return UNKNOWN_ANSWER_MESSAGE

    return f"{top_source.text}"
