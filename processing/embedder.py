"""
Embedding module using multilingual-e5-base.
============================================

Key design decisions:

1. multilingual-e5-base requires a prefix for queries vs. passages:
   - Passages (chunks):  "passage: {text}"
   - Queries (at search time): "query: {text}"
   This is critical — without prefixes, retrieval quality drops significantly.

2. We embed in batches to manage GPU/CPU memory. Batch size of 32 works
   well on most machines. Reduce if you hit OOM errors.

3. Embeddings are L2-normalized so we can use dot product (faster than cosine)
   in Qdrant while getting equivalent ranking.
"""

import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 768


class Embedder:
    """
    Wrapper around multilingual-e5-base for producing chunk embeddings.

    Usage:
        embedder = Embedder()
        vectors = embedder.encode_passages(["text1", "text2", ...])
        query_vec = embedder.encode_query("How do I register my address?")
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str = None):
        from sentence_transformers import SentenceTransformer
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Loading embedding model: {model_name} on {device}")
        self.model = SentenceTransformer(model_name, device=device)
        self.device = device
        logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")

    def encode_passages(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Encode document passages (chunks) for indexing.

        IMPORTANT: multilingual-e5 requires "passage: " prefix for documents.
        This prefix tells the model the text is a document to be retrieved,
        not a query. Skipping this drops recall significantly.
        """
        prefixed = [f"passage: {t}" for t in texts]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # L2 normalize → dot product = cosine
        )
        return embeddings

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a user query for searching.

        IMPORTANT: multilingual-e5 requires "query: " prefix for queries.
        """
        embedding = self.model.encode(
            f"query: {query}",
            normalize_embeddings=True,
        )
        return embedding

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple queries (for multi-query expansion)."""
        prefixed = [f"query: {q}" for q in queries]
        embeddings = self.model.encode(
            prefixed,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return embeddings