from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import settings


class EmbeddingManager:
    """
    Singleton wrapper for sentence-transformers embedding model.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance._model = None
        return cls._instance

    def get_model(self) -> SentenceTransformer:
        if self._model is None:
            try:
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL)
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model '{settings.EMBEDDING_MODEL}': {str(e)}")
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embeds a list of string texts.
        Returns a float32 numpy array of shape (N, D), normalized to unit L2 norm.
        """
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        model = self.get_model()
        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.astype(np.float32)
        except Exception as e:
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embeds a single query string.
        Returns a float32 numpy array of shape (1, D), normalized to unit L2 norm.
        """
        return self.embed_texts([query])


embedding_manager = EmbeddingManager()
