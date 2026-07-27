"""
Embedding Service Module
==========================
Provides dense vector embeddings for text using sentence-transformers.

Replaces the old TF-IDF refit-per-request approach with a pre-trained
sentence transformer model that produces fixed 384-dimensional embeddings.

Key benefits over the old TF-IDF approach:
1. **No refitting**: Embeddings are computed once, not refit on every request
2. **Semantic understanding**: Captures meaning, not just word overlap
3. **Consistent vector space**: Same text always produces the same vector
4. **Fast inference**: ~10ms per text on CPU

Model: all-MiniLM-L6-v2 (384-dim, ~80MB, CPU-friendly)
- Trained on 1B+ sentence pairs
- Optimized for semantic similarity tasks
- Good balance of quality vs. speed for CPU deployment

Design Decision:
- Model is loaded lazily on first use to avoid blocking app startup.
- Embeddings are L2-normalized for cosine similarity via dot product.
- Batch encoding is supported for bulk operations (e.g., indexing all jobs).
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Encodes text into dense vector embeddings using sentence-transformers.

    Usage:
        service = EmbeddingService()
        vector = service.encode("Python developer with 5 years experience...")
        vectors = service.encode_batch(["text1", "text2", "text3"])
    """

    # Default model — good balance of quality vs. speed for CPU
    DEFAULT_MODEL = 'all-MiniLM-L6-v2'
    EMBEDDING_DIM = 384

    def __init__(self, model_name=None):
        self._model = None
        self._model_name = model_name or self.DEFAULT_MODEL
        self._is_loaded = False

    @property
    def is_loaded(self):
        return self._is_loaded

    def _load_model(self):
        """Lazy-load the sentence transformer model on first use."""
        if self._is_loaded:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f'Loading embedding model: {self._model_name}...')
            self._model = SentenceTransformer(self._model_name)
            self._is_loaded = True
            logger.info(f'Embedding model loaded successfully ({self.EMBEDDING_DIM}-dim)')
        except Exception as e:
            logger.error(f'Failed to load embedding model: {e}')
            raise RuntimeError(
                f'Failed to load sentence-transformers model "{self._model_name}". '
                f'Ensure sentence-transformers is installed: pip install sentence-transformers'
            ) from e

    def encode(self, text):
        """
        Encode a single text string into a normalized embedding vector.

        Args:
            text: Input text string.

        Returns:
            numpy array of shape (384,) with L2-normalized values.
        """
        if not text or not text.strip():
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        self._load_model()

        embedding = self._model.encode(
            text,
            normalize_embeddings=True,  # L2 normalize for cosine via dot product
            show_progress_bar=False,
        )
        return embedding.astype(np.float32)

    def encode_batch(self, texts, batch_size=32):
        """
        Encode multiple texts into normalized embedding vectors.

        Args:
            texts: List of text strings.
            batch_size: Number of texts to encode at once.

        Returns:
            numpy array of shape (len(texts), 384) with L2-normalized values.
        """
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM), dtype=np.float32)

        self._load_model()

        # Replace empty/None texts with empty string
        cleaned = [t if t and t.strip() else '' for t in texts]

        embeddings = self._model.encode(
            cleaned,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def similarity(self, vec_a, vec_b):
        """
        Compute cosine similarity between two normalized vectors.

        Since vectors are L2-normalized, dot product = cosine similarity.

        Args:
            vec_a: numpy array of shape (384,)
            vec_b: numpy array of shape (384,)

        Returns:
            Float similarity score in [-1, 1] (typically [0, 1] for text).
        """
        return float(np.dot(vec_a, vec_b))


# Singleton instance — lazy-loaded
embedding_service = EmbeddingService()
