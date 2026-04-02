"""
Cross-Encoder Reranker
======================

Why rerank? Bi-encoder retrieval (embedding similarity) is fast but
approximate — it compares query and document independently. A cross-encoder
sees query AND document together, enabling much deeper relevance scoring.

The pipeline: retrieve 20-30 candidates fast with bi-encoder (Qdrant),
then rerank top candidates with cross-encoder to get the best 5-8.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Tiny (22M params), fast even on CPU
- Trained on MS MARCO passage ranking
- Outputs a relevance score (higher = more relevant)

For multilingual queries, you could swap to multilingual rerankers like
jeffwan/mmarco-mMiniLMv2-L12-H384-v1, but ms-marco works well enough
since most of your corpus is English.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    """
    Cross-encoder reranker for retrieved chunks.

    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self, model_name: str = RERANKER_MODEL):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker loaded")

    def rerank(
        self,
        query: str,
        chunks: list,
        top_k: int = 5,
    ) -> list:
        """
        Rerank retrieved chunks by cross-encoder relevance score.

        Args:
            query: the user's query
            chunks: list of RetrievedChunk objects from the retriever
            top_k: how many to keep after reranking

        Returns:
            Top-k chunks sorted by cross-encoder score (descending).
            Each chunk's .score is REPLACED with the cross-encoder score.
        """
        if not chunks:
            return []

        # Build query-document pairs for cross-encoder
        pairs = [(query, chunk.text) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach scores to chunks
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Update chunk scores and return top-k
        results = []
        for chunk, score in scored_chunks[:top_k]:
            chunk.score = float(score)
            results.append(chunk)

        logger.info(
            f"Reranked {len(chunks)} → top {len(results)} "
            f"(best: {results[0].score:.3f}, worst: {results[-1].score:.3f})"
        )

        return results