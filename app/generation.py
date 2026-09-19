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

SUMMARIZE_PROMPT_TEMPLATE = """You are an expert document analyst.
Provide a clear, structured, and comprehensive executive summary of the following document content:

{context}

Format your summary clearly with the following sections:
- **Executive Overview**: High-level purpose and summary of the document.
- **Key Findings & Core Points**: Bullet points detailing the most critical takeaways.
- **Important Requirements & Data**: Key numbers, dates, policies, metrics, or guidelines mentioned.
- **Actionable Takeaways / Next Steps**: What the reader should do or understand based on this content.

Strictly ground all facts in the document text provided above. Do not invent any outside facts.
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

    # 1. Try Google Gemini API first if configured
    if gemini_key:
        try:
            return _call_gemini_api(prompt, gemini_key)
        except Exception:
            # Fall through to OpenAI or extractive fallback
            pass

    # 2. Try OpenAI API if configured (or as fallback from Gemini)
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
            raw_content = response.choices[0].message.content
            if raw_content and raw_content.strip():
                return raw_content.strip()
        except Exception:
            # Fall through to extractive fallback
            pass

    # 3. Deterministic Extractive Grounded Fallback (when offline or APIs fail)
    return _extractive_fallback_answer(question, sources)


def _call_gemini_api(prompt: str, api_key: str) -> str:
    """
    Calls Google Gemini REST API securely using header-based authentication.
    """
    import httpx
    model_name = settings.GEMINI_MODEL
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }
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
                answer = parts[0]["text"].strip()
                if answer:
                    return answer

    raise ValueError("Invalid response structure from Gemini API.")


def _extractive_fallback_answer(question: str, sources: List[SourceChunk], error_note: str = "") -> str:
    """
    Provides a grounded fallback answer extracted directly from the most relevant source chunk
    when no API key is active or when upstream LLM APIs error.
    """
    if not sources:
        return UNKNOWN_ANSWER_MESSAGE

    top_source = sources[0]
    return top_source.text.strip()


def is_summary_query(question: str) -> bool:
    """
    Detects whether a user query is asking for a general document summary or overview.
    """
    q = question.lower().strip()
    summary_triggers = [
        "summarize",
        "summary",
        "overview",
        "brief",
        "synopsis",
        "tldr",
        "tl;dr",
        "what is this document about",
        "what are these documents about",
        "main points",
        "key takeaways",
        "give me an overview",
        "give me a summary"
    ]
    # Check if query is short and contains trigger, or directly matches trigger patterns
    words = q.split()
    if len(words) <= 8 and any(trigger in q for trigger in summary_triggers):
        return True
    return False


def generate_document_summary(chunks: list, document_name: str = "") -> str:
    """
    Synthesizes a full-document structured summary from sequential chunks.
    Uses Gemini API (or OpenAI fallback) and local structured extraction if offline.
    """
    if not chunks:
        return "No document content available to summarize. Please upload a PDF or TXT file first."

    # Build sequential context across chunks
    doc_header = f"Document: {document_name}\n\n" if document_name else ""
    context_body = "\n\n".join([f"[Page {getattr(c, 'page', 'N/A') or 'N/A'}]\n{c.text}" for c in chunks])
    full_context = f"{doc_header}{context_body}"

    prompt = SUMMARIZE_PROMPT_TEMPLATE.format(context=full_context)

    gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    openai_key = settings.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")

    # 1. Try Gemini API
    if gemini_key:
        try:
            return _call_gemini_api(prompt, gemini_key)
        except Exception:
            pass

    # 2. Try OpenAI API
    if openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert document summarization assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=800
            )
            raw = response.choices[0].message.content
            if raw and raw.strip():
                return raw.strip()
        except Exception:
            pass

    # 3. Deterministic Extractive Summary Fallback (offline mode)
    header = f"### Executive Summary for {document_name or 'Uploaded Document'}\n\n"
    sections = []
    for idx, c in enumerate(chunks[:5]):
        # Extract first 2 sentences of each key chunk
        sentences = [s.strip() for s in c.text.split(".") if len(s.strip()) > 15]
        snippet = ". ".join(sentences[:2])
        if snippet:
            sections.append(f"- **Key Section {idx + 1} (Page {getattr(c, 'page', 'N/A') or 'N/A'})**: {snippet}.")

    fallback_body = "\n".join(sections)
    return f"{header}**Key Findings & Highlights:**\n{fallback_body}\n\n*(Note: Generated via extractive synthesis across {len(chunks)} document chunks)*"


def generate_suggested_queries(chunks: list) -> list:
    """
    Dynamically analyzes document chunks and generates 4 tailored suggested questions.
    Falls back gracefully if offline.
    """
    default_suggestions = [
        {"label": "📝 Executive Summary", "query": "Summarize this document"},
        {"label": "Key Insights", "query": "What are the primary insights and core takeaways?"},
        {"label": "Critical Requirements", "query": "What are the main requirements and guidelines?"},
        {"label": "Next Steps", "query": "What action items or next steps are specified?"}
    ]

    if not chunks:
        return default_suggestions

    # Select representative chunks (up to 4 chunks across the document)
    step = max(1, len(chunks) // 4)
    sampled = [chunks[i] for i in range(0, len(chunks), step)][:4]
    context_sample = "\n---\n".join([c.text[:400] for c in sampled])

    prompt = f"""You are an intelligent document Q&A assistant.
Analyze this document excerpt:
{context_sample}

Generate exactly 4 high-value, specific questions that a user would ask about this exact document.
Format your output strictly as 4 lines, each line in this exact format:
Short Label | Question

Keep the Short Label under 3 words. Do not number the lines. Do not add markdown or explanation.
"""

    gemini_key = settings.GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    openai_key = settings.LLM_API_KEY or os.getenv("OPENAI_API_KEY", "")

    raw_output = None
    if gemini_key:
        try:
            raw_output = _call_gemini_api(prompt, gemini_key)
        except Exception:
            pass

    if not raw_output and openai_key:
        try:
            client = openai.OpenAI(api_key=openai_key)
            resp = client.chat.completions.create(
                model=settings.LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=250
            )
            raw_output = resp.choices[0].message.content
        except Exception:
            pass

    if raw_output:
        parsed = []
        # Always keep Executive Summary as first pill
        parsed.append({"label": "📝 Executive Summary", "query": "Summarize this document"})
        for line in raw_output.strip().split("\n"):
            line = line.strip().lstrip("0123456789.-) ")
            if "|" in line:
                parts = line.split("|", 1)
                label = parts[0].strip()
                query = parts[1].strip()
                if label and query and len(query) > 5:
                    parsed.append({"label": label, "query": query})
            elif line.endswith("?"):
                # Fallback if delimiter wasn't included
                words = line.split()
                label = " ".join(words[:2]).title()
                parsed.append({"label": label, "query": line})
            if len(parsed) >= 5:
                break
        if len(parsed) >= 2:
            return parsed

    # Extractive rule-based fallback based on first chunk words
    first_chunk_words = [w for w in chunks[0].text.split() if len(w) > 4][:10]
    topic = " ".join(first_chunk_words[:2]) if first_chunk_words else "the document"
    return [
        {"label": "📝 Executive Summary", "query": "Summarize this document"},
        {"label": f"{topic.title()}", "query": f"What are the key points regarding {topic}?"},
        {"label": "Requirements", "query": "What are the specific requirements or conditions mentioned?"},
        {"label": "Timeline & Rules", "query": "What timelines, periods, or policies are outlined?"}
    ]


