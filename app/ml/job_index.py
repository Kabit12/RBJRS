"""
Job Index Module
==================
FAISS-based vector index for fast approximate nearest neighbor search
over job embeddings.

Replaces the old brute-force KNN + TF-IDF refit-per-request approach with
a pre-built FAISS index that supports millisecond search times.

Architecture:
- Job embeddings are computed once when the index is built.
- The index is stored in memory and rebuilt when jobs change.
- Search returns top-K most similar jobs with cosine similarity scores.
- Uses IndexFlatIP (Inner Product) since embeddings are L2-normalized,
  making dot product equivalent to cosine similarity.

Design Decision:
- IndexFlatIP (exact search) is used instead of approximate indexes
  because job counts are typically <10k, making exact search fast enough.
- For >100k jobs, switch to IndexIVFFlat or IndexHNSW.
"""

import numpy as np
import logging
import threading

logger = logging.getLogger(__name__)

# Try to import faiss; fall back to brute-force numpy if not available
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning(
        'faiss-cpu not installed. Falling back to brute-force numpy search. '
        'Install with: pip install faiss-cpu'
    )


class JobIndex:
    """
    FAISS index over job embeddings for fast nearest neighbor search.

    Usage:
        index = JobIndex(embedding_dim=384)
        index.build(job_ids, job_embeddings)
        scores, ids = index.search(query_vector, top_k=10)
    """

    def __init__(self, embedding_dim=384):
        self.embedding_dim = embedding_dim
        self._index = None
        self._job_ids = []
        self._embeddings = None
        self._is_built = False
        self._lock = threading.Lock()

    @property
    def is_built(self):
        return self._is_built

    @property
    def size(self):
        return len(self._job_ids)

    def build(self, job_ids, embeddings):
        """
        Build the FAISS index from job embeddings.

        Args:
            job_ids: List of integer job IDs (parallel to embeddings).
            embeddings: numpy array of shape (n_jobs, embedding_dim), L2-normalized.
        """
        with self._lock:
            if len(job_ids) == 0:
                self._is_built = False
                return

            embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            self._job_ids = list(job_ids)
            self._embeddings = embeddings

            if HAS_FAISS:
                # IndexFlatIP = Inner Product (cosine similarity for normalized vectors)
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._index.add(embeddings)
                logger.info(f'FAISS index built with {len(job_ids)} jobs')
            else:
                # Fallback: store embeddings for brute-force numpy search
                logger.info(f'Numpy fallback index built with {len(job_ids)} jobs')

            self._is_built = True

    def search(self, query_vector, top_k=20):
        """
        Search for the top-K most similar jobs to the query vector.

        Args:
            query_vector: numpy array of shape (embedding_dim,), L2-normalized.
            top_k: Number of results to return.

        Returns:
            Tuple of (scores, job_ids):
                - scores: list of float similarity scores (descending)
                - job_ids: list of integer job IDs
        """
        if not self._is_built or len(self._job_ids) == 0:
            return [], []

        top_k = min(top_k, len(self._job_ids))
        query = np.ascontiguousarray(
            query_vector.reshape(1, -1), dtype=np.float32
        )

        with self._lock:
            if HAS_FAISS and self._index is not None:
                scores, indices = self._index.search(query, top_k)
                scores = scores[0].tolist()
                indices = indices[0].tolist()
            else:
                # Brute-force fallback using numpy dot product
                similarities = np.dot(self._embeddings, query_vector)
                indices = np.argsort(similarities)[::-1][:top_k].tolist()
                scores = [float(similarities[i]) for i in indices]

        # Map indices back to job IDs
        result_ids = [self._job_ids[i] for i in indices if i < len(self._job_ids)]
        result_scores = scores[:len(result_ids)]

        return result_scores, result_ids

    def upsert_job(self, job_id, embedding):
        """
        Add or update a single job embedding in the FAISS index (Directive 3 Fix).

        Args:
            job_id: Integer job ID
            embedding: numpy array of shape (embedding_dim,), L2-normalized
        """
        with self._lock:
            emb = np.ascontiguousarray(embedding.reshape(1, -1), dtype=np.float32)
            if self._embeddings is None or len(self._job_ids) == 0:
                self._job_ids = [job_id]
                self._embeddings = emb
            elif job_id in self._job_ids:
                idx = self._job_ids.index(job_id)
                self._embeddings[idx] = emb[0]
            else:
                self._job_ids.append(job_id)
                self._embeddings = np.vstack([self._embeddings, emb])

            if HAS_FAISS:
                self._index = faiss.IndexFlatIP(self.embedding_dim)
                self._index.add(self._embeddings)
            self._is_built = True
            logger.info(f'Job {job_id} upserted into FAISS index (total: {len(self._job_ids)})')

    def remove_job(self, job_id):
        """
        Remove a job from the FAISS index.

        Args:
            job_id: Integer job ID to remove
        """
        with self._lock:
            if job_id not in self._job_ids:
                return

            idx = self._job_ids.index(job_id)
            self._job_ids.pop(idx)
            self._embeddings = np.delete(self._embeddings, idx, axis=0)

            if len(self._job_ids) == 0:
                self.clear()
            else:
                if HAS_FAISS:
                    self._index = faiss.IndexFlatIP(self.embedding_dim)
                    self._index.add(self._embeddings)
                logger.info(f'Job {job_id} removed from FAISS index (total: {len(self._job_ids)})')

    def clear(self):
        """Clear the index."""
        with self._lock:
            self._index = None
            self._job_ids = []
            self._embeddings = None
            self._is_built = False


# Singleton instance
job_index = JobIndex()
