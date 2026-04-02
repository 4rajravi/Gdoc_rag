"""
German Bureaucracy RAG — Indexing Pipeline
==========================================

Reads scraped documents → hierarchical chunking → embedding → Qdrant indexing.

Prerequisites:
    pip install sentence-transformers qdrant-client torch
    docker run -p 6333:6333 qdrant/qdrant

Usage:
    # Full pipeline: chunk → embed → index
    python run_indexing.py

    # Only chunk (no embedding/indexing, useful for inspecting chunks)
    python run_indexing.py --chunk-only

    # Recreate Qdrant collections (wipes existing data)
    python run_indexing.py --recreate

    # Custom paths
    python run_indexing.py --input data/raw/documents.jsonl --output data/processed/
"""

import argparse
import logging
import sys
import time

from processing.chunker import HierarchicalChunker
from processing.embedder import Embedder
from processing.indexer import QdrantIndexer


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("indexing.log", encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Chunk, embed, and index scraped documents into Qdrant.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/documents.jsonl",
        help="Path to scraped documents JSONL (default: data/raw/documents.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/processed",
        help="Output directory for chunk files (default: data/processed/)",
    )
    parser.add_argument(
        "--chunk-only",
        action="store_true",
        help="Only run chunking (skip embedding and indexing)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate Qdrant collections (deletes existing data)",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
        help="Qdrant host (default: localhost)",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size (reduce if OOM, default: 32)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for embedding model: 'cpu', 'cuda', or 'mps' (auto-detect if not set)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ─── Step 1: Hierarchical Chunking ──────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Hierarchical Chunking")
    logger.info("=" * 60)

    t0 = time.time()

    chunker = HierarchicalChunker(
        l3_max_tokens=300,
        l3_overlap=50,
        l2_split_threshold=500,
        l1_summary_tokens=200,
    )
    chunker.process_documents(args.input)
    chunker.save(args.output)
    chunker.print_stats()

    t_chunk = time.time() - t0
    logger.info(f"Chunking completed in {t_chunk:.1f}s")

    if args.chunk_only:
        logger.info("--chunk-only flag set. Stopping here.")
        return

    # ─── Step 2: Embedding ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2: Embedding with multilingual-e5-base")
    logger.info("=" * 60)

    t0 = time.time()

    embedder = Embedder(device=args.device)

    all_chunks = chunker.get_all_chunks()
    texts = [chunk.text for chunk in all_chunks]

    logger.info(f"Encoding {len(texts)} chunks...")
    embeddings = embedder.encode_passages(texts, batch_size=args.batch_size)

    t_embed = time.time() - t0
    logger.info(f"Embedding completed in {t_embed:.1f}s ({len(texts)/t_embed:.1f} chunks/sec)")
    logger.info(f"Embedding matrix shape: {embeddings.shape}")

    # ─── Step 3: Qdrant Indexing ────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3: Indexing into Qdrant")
    logger.info("=" * 60)

    t0 = time.time()

    indexer = QdrantIndexer(host=args.qdrant_host, port=args.qdrant_port)
    indexer.create_collections(recreate=args.recreate)
    indexer.index_chunks(all_chunks, embeddings)
    indexer.get_collection_info()

    t_index = time.time() - t0
    logger.info(f"Indexing completed in {t_index:.1f}s")

    # ─── Summary ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INDEXING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Documents processed: {args.input}")
    print(f"  Chunks created:      L1={len(chunker.chunks['L1'])}, "
          f"L2={len(chunker.chunks['L2'])}, L3={len(chunker.chunks['L3'])}")
    print(f"  Vectors indexed:     {len(all_chunks)}")
    print(f"  Embedding dim:       {embeddings.shape[1]}")
    print(f"  Time — chunking:     {t_chunk:.1f}s")
    print(f"  Time — embedding:    {t_embed:.1f}s")
    print(f"  Time — indexing:     {t_index:.1f}s")
    print(f"  Chunk files:         {args.output}/chunks_l{{1,2,3}}.jsonl")
    print()


if __name__ == "__main__":
    main()