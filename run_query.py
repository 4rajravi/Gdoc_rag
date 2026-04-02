#!/usr/bin/env python3
"""
German Bureaucracy RAG — Query Interface
=========================================

Interactive CLI to ask questions about German bureaucracy.

Prerequisites:
    - Qdrant running: docker run -p 6333:6333 qdrant/qdrant
    - Ollama running with model: ollama serve & ollama pull llama3.1:8b
    - Indexing pipeline completed: python run_indexing.py --recreate

Usage:
    # Interactive mode
    python run_query.py

    # Single question
    python run_query.py --question "How do I register my address?"

    # Debug mode (shows all pipeline stages)
    python run_query.py --debug

    # Fast mode (skip query expansion — lower latency)
    python run_query.py --fast
"""

import argparse
import logging
import sys

from processing.embedder import Embedder
from retrieval.pipeline import RAGPipeline


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask questions about German bureaucracy.",
    )
    parser.add_argument(
        "--question", "-q",
        default=None,
        help="Ask a single question (non-interactive mode)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show pipeline internals (intent, retrieval, reranking)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode — skip query reformulation and expansion",
    )
    parser.add_argument(
        "--model",
        default="llama3.1:8b",
        help="Ollama model to use (default: llama3.1:8b)",
    )
    parser.add_argument(
        "--qdrant-host",
        default="localhost",
    )
    parser.add_argument(
        "--qdrant-port",
        type=int,
        default=6333,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
    )
    return parser.parse_args()


def print_result(result):
    """Pretty-print a pipeline result."""
    print("\n" + "=" * 60)
    print(result.answer)
    print("=" * 60)

    if result.sources:
        print("\nSources:")
        for src in result.sources:
            print(f"  - {src['title']}")
            print(f"    {src['url']}")

    print(f"\n[{result.chunks_used} chunks | "
          f"{result.intent_category} | "
          f"{result.intent_specificity} | "
          f"{sum(result.timing.values()):.1f}s total]")


def main():
    args = parse_args()
    setup_logging(args.verbose)

    mode = "debug" if args.debug else ("fast" if args.fast else "full")

    # Initialize pipeline
    print("Loading models...")
    from qdrant_client import QdrantClient
    client = QdrantClient(host=args.qdrant_host, port=args.qdrant_port)
    embedder = Embedder()

    pipeline = RAGPipeline(
        qdrant_client=client,
        embedder=embedder,
        ollama_model=args.model,
    )
    print("Ready!\n")

    # Single question mode
    if args.question:
        result = pipeline.query(args.question, mode=mode)
        print_result(result)
        return

    # Interactive mode
    print("German Bureaucracy Helper (RAG)")
    print("Type your question, or 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = pipeline.query(query, mode=mode)
        print_result(result)
        print()


if __name__ == "__main__":
    main()