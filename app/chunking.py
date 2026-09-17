import re
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ChunkRecord:
    document: str
    chunk_id: int
    page: Optional[int]
    text: str


def split_text_into_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 75
) -> List[str]:
    """
    Splits a single text block into overlapping chunks.
    Attempts to split cleanly on paragraph or sentence boundaries where possible.
    """
    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    # Split into paragraphs/sentences first
    sentences = re.split(r'(?<=[.!?\n])\s+', text)
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # If a single sentence exceeds chunk_size, split by character
        if len(sentence) > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
            
            start = 0
            while start < len(sentence):
                end = min(start + chunk_size, len(sentence))
                chunks.append(sentence[start:end])
                start += chunk_size - chunk_overlap if chunk_size > chunk_overlap else chunk_size
            continue

        if current_len + len(sentence) + 1 <= chunk_size:
            current_chunk.append(sentence)
            current_len += len(sentence) + 1
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))

            # Calculate overlap from the end of current_chunk
            overlap_sentences: List[str] = []
            overlap_len = 0
            for prev_s in reversed(current_chunk):
                if overlap_len + len(prev_s) + 1 <= chunk_overlap:
                    overlap_sentences.insert(0, prev_s)
                    overlap_len += len(prev_s) + 1
                else:
                    break

            current_chunk = overlap_sentences + [sentence]
            current_len = sum(len(s) + 1 for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def chunk_document(
    filename: str,
    pages_text: List[Tuple[int, str]],
    chunk_size: int = 500,
    chunk_overlap: int = 75,
    start_chunk_id: int = 0
) -> List[ChunkRecord]:
    """
    Chunks pages of a document into ChunkRecord instances.
    """
    chunk_records: List[ChunkRecord] = []
    current_id = start_chunk_id

    for page_num, page_text in pages_text:
        page_chunks = split_text_into_chunks(page_text, chunk_size, chunk_overlap)
        for text_chunk in page_chunks:
            chunk_records.append(
                ChunkRecord(
                    document=filename,
                    chunk_id=current_id,
                    page=page_num,
                    text=text_chunk
                )
            )
            current_id += 1

    return chunk_records
