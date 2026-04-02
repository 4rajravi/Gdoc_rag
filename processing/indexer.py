"""
Qdrant Indexer — stores chunk embeddings with metadata.
========================================================

Design decisions:

1. Three separate collections: chunks_l1, chunks_l2, chunks_l3
   Why not one collection with a "level" filter?
   - Separate collections let Qdrant optimize each index independently
   - L1 is tiny (~70 vectors), L2 is medium, L3 can be large
   - At query time, you control which levels to search explicitly

2. Payload (metadata) stored alongside each vector enables:
   - Category filtering: only search "anmeldung" docs
   - Source filtering: prefer official sources over Reddit
   - Returning context: show doc_title, section_header, url in results

3. We use DOT product distance (not cosine) because embeddings are
   already L2-normalized by the Embedder. Dot product is faster.

4. Batch upsert (100 points at a time) to avoid overwhelming Qdrant.
"""

import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

COLLECTION_NAMES = {
    "L1": "chunks_l1",
    "L2": "chunks_l2",
    "L3": "chunks_l3",
}

EMBEDDING_DIM = 768  # multilingual-e5-base


class QdrantIndexer:
    """
    Manages Qdrant collections for hierarchical chunk storage.

    Usage:
        indexer = QdrantIndexer()
        indexer.create_collections()
        indexer.index_chunks(chunks, embeddings)
    """

    def __init__(self, host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.client = QdrantClient(host=host, port=port)
        self.Distance = Distance
        self.VectorParams = VectorParams
        logger.info(f"Connected to Qdrant at {host}:{port}")

    def create_collections(self, recreate: bool = False):
        """
        Create the three chunk collections in Qdrant.

        Args:
            recreate: If True, delete existing collections first.
                      Use during development; avoid in production.
        """
        for level, name in COLLECTION_NAMES.items():
            if recreate:
                try:
                    self.client.delete_collection(name)
                    logger.info(f"Deleted existing collection: {name}")
                except Exception:
                    pass

            try:
                self.client.create_collection(
                    collection_name=name,
                    vectors_config=self.VectorParams(
                        size=EMBEDDING_DIM,
                        distance=self.Distance.DOT,
                    ),
                )
                logger.info(f"Created collection: {name} (dim={EMBEDDING_DIM}, distance=DOT)")
            except Exception as e:
                # Collection might already exist
                logger.info(f"Collection {name} already exists or error: {e}")

    def index_chunks(
        self,
        chunks: list,
        embeddings: np.ndarray,
        batch_size: int = 100,
    ):
        """
        Upload chunk vectors + metadata to appropriate Qdrant collections.

        Args:
            chunks: list of Chunk dataclass instances
            embeddings: numpy array of shape (len(chunks), 768)
            batch_size: upsert batch size
        """
        from qdrant_client.models import PointStruct

        assert len(chunks) == len(embeddings), (
            f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings"
        )

        # Group by level
        level_groups: dict[str, list[tuple]] = {"L1": [], "L2": [], "L3": []}
        for chunk, embedding in zip(chunks, embeddings):
            level_groups[chunk.level].append((chunk, embedding))

        for level, items in level_groups.items():
            if not items:
                continue

            collection_name = COLLECTION_NAMES[level]
            points = []

            for idx, (chunk, embedding) in enumerate(items):
                # Build payload from chunk metadata
                payload = {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "doc_url": chunk.doc_url,
                    "doc_title": chunk.doc_title,
                    "source": chunk.source,
                    "category": chunk.category,
                    "language": chunk.language,
                    "section_header": chunk.section_header,
                    "parent_chunk_id": chunk.parent_chunk_id,
                    "token_count": chunk.token_count,
                }

                points.append(PointStruct(
                    id=idx,
                    vector=embedding.tolist(),
                    payload=payload,
                ))

            # Batch upsert
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch,
                )

            logger.info(f"Indexed {len(points)} vectors into {collection_name}")

    def get_collection_info(self):
        """Print info about all collections."""
        print("\n" + "=" * 50)
        print("QDRANT COLLECTION INFO")
        print("=" * 50)
        for level, name in COLLECTION_NAMES.items():
            try:
                info = self.client.get_collection(name)
                print(f"\n  {name}:")
                print(f"    Vectors: {info.points_count}")
                print(f"    Status:  {info.status}")
            except Exception as e:
                print(f"\n  {name}: not found ({e})")
        print()