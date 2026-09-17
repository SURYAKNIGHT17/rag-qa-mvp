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
    If sources are empty, returns UNKNOWN_ANSWER_MESSAGE directly.
    """
    if not sources:
        return UNKNOWN_ANSWER_MESSAGE

    context_str = "\n\n".join(
        [f"[Doc: {s.document}, Page: {s.page or 'N/A'}, Chunk: {s.chunk_id}]\n{s.text}" for s in sources]
    )

    prompt = GROUNDED_PROMPT_TEMPLATE.format(context=context_str, question=question)

    api_key = settings.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")

    if api_key:
        try:
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful document Q&A assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=500
            )
            answer = response.choices[0].message.content.strip()
            return answer
        except Exception as e:
            # Fallback to extractive answer if API call fails
            return _extractive_fallback_answer(question, sources, error_note=str(e))
    else:
        # Extractive fallback when no LLM API key is provided
        return _extractive_fallback_answer(question, sources)


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
