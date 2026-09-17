from typing import List, Tuple, Dict, Set
import faiss
import numpy as np
from app.chunking import ChunkRecord
from app.embeddings import embedding_manager
from app.config import settings
from app.models import SourceChunk


class VectorStore:
    """
    FAISS-backed local vector store with associated metadata store.
    Uses IndexFlatIP for Cosine Similarity search over L2-normalized embeddings.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: List[ChunkRecord] = []
        self.indexed_documents: Set[str] = set()

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    @property
    def total_documents(self) -> int:
        return len(self.indexed_documents)

    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        """
        Embeds and indexes a list of ChunkRecord objects.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_manager.embed_texts(texts)

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError("Mismatch between number of chunks and generated embeddings.")

        self.index.add(embeddings)
        self.metadata_store.extend(chunks)

        for chunk in chunks:
            self.indexed_documents.add(chunk.document)

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> List[SourceChunk]:
        """
        Searches FAISS for top_k relevant chunks for a given query string.
        Filters out candidates below similarity_threshold.
        Returns a list of SourceChunk objects.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = embedding_manager.embed_query(query)
        actual_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, actual_k)

        results: List[SourceChunk] = []
        if scores.size == 0 or indices.size == 0:
            return results

        # Process top-k results
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            
            float_score = float(score)
            # Relevance filtering check
            if float_score < similarity_threshold:
                continue

            chunk_meta = self.metadata_store[idx]
            results.append(
                SourceChunk(
                    document=chunk_meta.document,
                    chunk_id=chunk_meta.chunk_id,
                    page=chunk_meta.page,
                    similarity=round(float_score, 4),
                    text=chunk_meta.text
                )
            )

        return results


# Global in-memory vector store instance
vector_store = VectorStore()
