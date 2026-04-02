"""
Hierarchical Retriever — searches the right chunk levels based on intent.
=========================================================================

The key insight: not every query should search every collection.

- "How does health insurance work in Germany?" (broad)
    → Search L1 (summaries) + L2 (sections)
    → Returns overview + relevant sections

- "What's the income threshold for private insurance?" (specific)
    → Search L2 (sections) + L3 (fine-grained)
    → Returns the exact detail

The retriever also supports:
- Category filtering: narrow search to matching topic
- Multi-query: search multiple query variants, merge results
- Deduplication: remove chunks from the same section appearing multiple times
"""

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION_MAP = {
    "L1": "chunks_l1",
    "L2": "chunks_l2",
    "L3": "chunks_l3",
}


@dataclass
class RetrievedChunk:
    """A single retrieval result with score and metadata."""
    chunk_id: str
    text: str
    score: float
    level: str
    doc_url: str
    doc_title: str
    source: str
    category: str
    section_header: str
    parent_chunk_id: str = ""

    def __hash__(self):
        return hash(self.chunk_id)

    def __eq__(self, other):
        return self.chunk_id == other.chunk_id


class HierarchicalRetriever:
    """
    Searches Qdrant collections based on query intent.

    Usage:
        retriever = HierarchicalRetriever(qdrant_client, embedder)
        results = retriever.search(
            query="What documents for Anmeldung?",
            intent=intent,
            top_k=10,
        )
    """

    def __init__(self, qdrant_client, embedder):
        self.client = qdrant_client
        self.embedder = embedder

    def search(
        self,
        query: str,
        search_levels: list[str],
        category_filter: str = None,
        top_k_per_level: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Search specified collection levels for a single query.

        Args:
            query: search query text
            search_levels: which levels to search ("L1", "L2", "L3")
            category_filter: optional category to filter by
            top_k_per_level: max results per level
        """
        query_vector = self.embedder.encode_query(query).tolist()
        all_results = []

        for level in search_levels:
            collection_name = COLLECTION_MAP.get(level)
            if not collection_name:
                continue

            # Build optional category filter
            query_filter = None
            if category_filter and category_filter != "general":
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="category",
                            match=MatchValue(value=category_filter),
                        )
                    ]
                )

            try:
                results = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k_per_level,
                )

                for point in results.points:
                    chunk = RetrievedChunk(
                        chunk_id=point.payload["chunk_id"],
                        text=point.payload["text"],
                        score=point.score,
                        level=level,
                        doc_url=point.payload["doc_url"],
                        doc_title=point.payload["doc_title"],
                        source=point.payload["source"],
                        category=point.payload["category"],
                        section_header=point.payload["section_header"],
                        parent_chunk_id=point.payload.get("parent_chunk_id", ""),
                    )
                    all_results.append(chunk)

            except Exception as e:
                logger.error(f"Search failed on {collection_name}: {e}")

        return all_results

    def search_multi_query(
        self,
        queries: list[str],
        search_levels: list[str],
        category_filter: str = None,
        top_k_per_level: int = 10,
    ) -> list[RetrievedChunk]:
        """
        Search with multiple query variants and merge results.

        This is the "diverse retrieval" strategy — each query variant
        might match different relevant chunks. Results are deduplicated
        and the best score for each chunk is kept.
        """
        seen_chunks: dict[str, RetrievedChunk] = {}

        for query in queries:
            results = self.search(
                query=query,
                search_levels=search_levels,
                category_filter=category_filter,
                top_k_per_level=top_k_per_level,
            )

            for chunk in results:
                if chunk.chunk_id not in seen_chunks:
                    seen_chunks[chunk.chunk_id] = chunk
                else:
                    # Keep the higher score
                    if chunk.score > seen_chunks[chunk.chunk_id].score:
                        seen_chunks[chunk.chunk_id] = chunk

        # Sort by score descending
        merged = sorted(seen_chunks.values(), key=lambda c: c.score, reverse=True)
        logger.info(
            f"Multi-query search: {len(queries)} queries → "
            f"{len(merged)} unique chunks"
        )

        return merged