import os
import json
import logging
import re
from typing import List, Tuple, Dict, Set, Optional, Union
import faiss
import numpy as np
from app.chunking import ChunkRecord
from app.embeddings import embedding_manager
from app.config import settings
from app.models import SourceChunk

logger = logging.getLogger(__name__)


class VectorStore:
    """
    FAISS-backed local vector store with associated metadata store and disk persistence.
    Uses IndexFlatIP for Cosine Similarity search over L2-normalized embeddings,
    augmented with lexical keyword matching for robust retrieval ranking.
    """
    def __init__(self, dimension: int = 384, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir if persist_dir is not None else settings.PERSIST_DIRECTORY
        try:
            model = embedding_manager.get_model()
            if hasattr(model, "get_embedding_dimension"):
                self.dimension = model.get_embedding_dimension()
            else:
                self.dimension = model.get_sentence_embedding_dimension()
        except Exception:
            self.dimension = dimension

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store: List[ChunkRecord] = []
        self.indexed_documents: Set[str] = set()

        if self.persist_dir:
            self.load()

    def reset(self):
        """
        Resets the index and clears all stored chunk metadata and documents,
        including persisted files on disk.
        """
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata_store.clear()
        self.indexed_documents.clear()

        if self.persist_dir:
            index_file = os.path.join(self.persist_dir, "index.faiss")
            meta_file = os.path.join(self.persist_dir, "metadata.json")
            if os.path.exists(index_file):
                try:
                    os.remove(index_file)
                except OSError:
                    pass
            if os.path.exists(meta_file):
                try:
                    os.remove(meta_file)
                except OSError:
                    pass

    def save(self) -> bool:
        """
        Persists FAISS index and chunk metadata to disk.
        """
        if not self.persist_dir:
            return False
        try:
            os.makedirs(self.persist_dir, exist_ok=True)
            index_file = os.path.join(self.persist_dir, "index.faiss")
            meta_file = os.path.join(self.persist_dir, "metadata.json")

            faiss.write_index(self.index, index_file)
            meta_payload = {
                "indexed_documents": sorted(list(self.indexed_documents)),
                "chunks": [
                    {
                        "document": c.document,
                        "chunk_id": c.chunk_id,
                        "page": c.page,
                        "text": c.text
                    }
                    for c in self.metadata_store
                ]
            }
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta_payload, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.warning("Failed to persist vector store: %s", str(e))
            return False

    def load(self) -> bool:
        """
        Loads FAISS index and chunk metadata from disk if available.
        """
        if not self.persist_dir:
            return False
        index_file = os.path.join(self.persist_dir, "index.faiss")
        meta_file = os.path.join(self.persist_dir, "metadata.json")

        if os.path.exists(index_file) and os.path.exists(meta_file):
            try:
                self.index = faiss.read_index(index_file)
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta_payload = json.load(f)
                self.indexed_documents = set(meta_payload.get("indexed_documents", []))
                self.metadata_store = [
                    ChunkRecord(
                        document=item["document"],
                        chunk_id=item["chunk_id"],
                        page=item.get("page"),
                        text=item["text"]
                    )
                    for item in meta_payload.get("chunks", [])
                ]
                logger.info("Loaded persisted vector store with %d chunks and %d docs from %s",
                            self.total_chunks, self.total_documents, self.persist_dir)
                return True
            except Exception as e:
                logger.warning("Failed to load persisted vector store: %s", str(e))
                return False
        return False

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    @property
    def total_documents(self) -> int:
        return len(self.indexed_documents)

    def get_document_chunks(self, document_names: Optional[Union[str, List[str]]] = None) -> List[ChunkRecord]:
        """
        Returns all chunk records for specific document(s), or across all documents.
        Sorted by document, page, and chunk_id for sequential reading.
        """
        if not document_names:
            return sorted(self.metadata_store, key=lambda c: (c.document, c.page or 0, c.chunk_id))

        if isinstance(document_names, str):
            names_set = {document_names.lower()}
        else:
            names_set = {d.lower() for d in document_names if d}

        matching = [c for c in self.metadata_store if c.document.lower() in names_set]
        return sorted(matching, key=lambda c: (c.document, c.page or 0, c.chunk_id))

    def add_chunks(self, chunks: List[ChunkRecord]) -> int:
        """
        Embeds and indexes a list of ChunkRecord objects.
        Returns the number of chunks added and persists to disk.
        """
        if not chunks:
            return 0

        texts = [chunk.text for chunk in chunks]
        embeddings = embedding_manager.embed_texts(texts)

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError("Mismatch between number of chunks and generated embeddings.")

        start_id = len(self.metadata_store)
        for i, chunk in enumerate(chunks):
            chunk.chunk_id = start_id + i

        self.index.add(embeddings)
        self.metadata_store.extend(chunks)

        for chunk in chunks:
            self.indexed_documents.add(chunk.document)

        # Persist to disk
        self.save()

        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = settings.TOP_K,
        similarity_threshold: float = settings.SIMILARITY_THRESHOLD
    ) -> List[SourceChunk]:
        """
        Searches FAISS for top_k relevant chunks for a given query string.
        Applies similarity thresholding and hybrid keyword-aware re-ranking to ensure
        high semantic relevance.
        """
        if self.index.ntotal == 0:
            return []

        query_vec = embedding_manager.embed_query(query)
        # Fetch candidate pool (up to 2x top_k)
        candidate_k = min(max(top_k * 2, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_vec, candidate_k)

        if scores.size == 0 or indices.size == 0:
            return []

        # Extract meaningful terms from query (tokens >= 3 chars, lowercase)
        query_terms = set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', query.lower()))

        candidates: List[Tuple[float, float, ChunkRecord]] = []

        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue

            float_score = float(score)
            if float_score < similarity_threshold:
                continue

            chunk_meta = self.metadata_store[idx]
            chunk_lower = chunk_meta.text.lower()

            # Compute keyword overlap score
            keyword_matches = sum(1 for term in query_terms if term in chunk_lower)
            keyword_ratio = keyword_matches / max(len(query_terms), 1)

            # Combined ranking score: base cosine similarity + up to 0.05 lexical bonus
            ranking_score = float_score + (0.05 * keyword_ratio)
            candidates.append((ranking_score, float_score, chunk_meta))

        # Sort by ranking score descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: List[SourceChunk] = []
        for _, orig_score, chunk_meta in candidates[:top_k]:
            results.append(
                SourceChunk(
                    document=chunk_meta.document,
                    chunk_id=chunk_meta.chunk_id,
                    page=chunk_meta.page,
                    similarity=round(orig_score, 4),
                    text=chunk_meta.text
                )
            )

        return results


# Global vector store instance with disk persistence
vector_store = VectorStore()
