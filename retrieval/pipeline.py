"""
RAG Pipeline — Orchestrates the full query-to-answer flow.
===========================================================

Flow:
1. Intent classification (rule-based, instant)
2. Query reformulation (Ollama, if query is vague)
3. Multi-query expansion (Ollama, generates 2-3 variants)
4. Hierarchical retrieval (Qdrant, searches appropriate levels)
5. Merge + deduplicate results from all query variants
6. Cross-encoder reranking (ms-marco-MiniLM)
7. Context assembly + LLM generation (Ollama)
8. Return answer with sources

The pipeline can be run in different modes:
- full: all stages (default)
- fast: skip reformulation + expansion (for low-latency)
- debug: print intermediate results at each stage
"""

import logging
import time
from dataclasses import dataclass, field

from .query_processor import classify_intent, reformulate_query, expand_query
from .retriever import HierarchicalRetriever
from .reranker import Reranker
from .generator import OllamaClient, Generator

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result from the RAG pipeline."""
    query: str
    answer: str
    sources: list
    chunks_used: int

    # Debug info
    intent_category: str = ""
    intent_specificity: str = ""
    search_levels: list = field(default_factory=list)
    reformulated_query: str = ""
    expanded_queries: list = field(default_factory=list)
    retrieved_count: int = 0
    reranked_count: int = 0
    timing: dict = field(default_factory=dict)


class RAGPipeline:
    """
    Full RAG pipeline from query to answer.

    Usage:
        pipeline = RAGPipeline(
            qdrant_client=client,
            embedder=embedder,
            ollama_model="llama3.1:8b",
        )
        result = pipeline.query("How do I register my address in Germany?")
        print(result.answer)
    """

    def __init__(
        self,
        qdrant_client,
        embedder,
        ollama_model: str = "llama3.1:8b",
        ollama_url: str = "http://localhost:11434",
        reranker_top_k: int = 5,
        retriever_top_k_per_level: int = 10,
    ):
        self.retriever = HierarchicalRetriever(qdrant_client, embedder)
        self.reranker = Reranker()
        self.ollama = OllamaClient(model=ollama_model, base_url=ollama_url)
        self.generator = Generator(self.ollama)

        self.reranker_top_k = reranker_top_k
        self.retriever_top_k = retriever_top_k_per_level

    def query(
        self,
        user_query: str,
        mode: str = "full",  # "full", "fast", or "debug"
        category_override: str = None,
    ) -> PipelineResult:
        """
        Run the full RAG pipeline on a user query.

        Args:
            user_query: the question to answer
            mode: "full" (all stages), "fast" (skip expansion), "debug" (verbose)
            category_override: force a specific category filter

        Returns:
            PipelineResult with answer, sources, and debug info
        """
        timings = {}
        debug = mode == "debug"

        # ─── 1. Intent Classification ──────────────────────────
        t0 = time.time()
        intent = classify_intent(user_query)
        timings["intent"] = time.time() - t0

        if category_override:
            intent.category = category_override

        if debug:
            print(f"\n[Intent] category={intent.category} "
                  f"specificity={intent.specificity} "
                  f"levels={intent.search_levels}")

        # ─── 2. Query Reformulation ────────────────────────────
        reformulated = user_query
        if mode == "full" or mode == "debug":
            t0 = time.time()
            reformulated = reformulate_query(user_query, intent, self.ollama)
            timings["reformulation"] = time.time() - t0

            if debug and reformulated != user_query:
                print(f"[Reformulated] '{user_query}' → '{reformulated}'")

        # ─── 3. Multi-Query Expansion ──────────────────────────
        query_variants = [reformulated]
        if mode == "full" or mode == "debug":
            t0 = time.time()
            query_variants = expand_query(reformulated, intent, self.ollama)
            timings["expansion"] = time.time() - t0

            if debug:
                print(f"[Expanded] {len(query_variants)} variants: {query_variants}")

        # ─── 4. Hierarchical Retrieval ─────────────────────────
        t0 = time.time()

        category_filter = intent.category if intent.category_confidence > 0.3 else None

        retrieved = self.retriever.search_multi_query(
            queries=query_variants,
            search_levels=intent.search_levels,
            category_filter=category_filter,
            top_k_per_level=self.retriever_top_k,
        )
        timings["retrieval"] = time.time() - t0

        if debug:
            print(f"[Retrieved] {len(retrieved)} unique chunks")
            for c in retrieved[:5]:
                print(f"  [{c.score:.3f}] {c.level} | {c.source} — {c.section_header}")

        # ─── 5. Reranking ──────────────────────────────────────
        t0 = time.time()
        reranked = self.reranker.rerank(
            query=user_query,  # rerank against ORIGINAL query
            chunks=retrieved,
            top_k=self.reranker_top_k,
        )
        timings["reranking"] = time.time() - t0

        if debug:
            print(f"[Reranked] top {len(reranked)}:")
            for c in reranked:
                print(f"  [{c.score:.3f}] {c.level} | {c.source} — {c.section_header}")

        # ─── 6. Generation ─────────────────────────────────────
        t0 = time.time()
        gen_result = self.generator.generate_answer(
            query=user_query,  # generate against ORIGINAL query
            chunks=reranked,
        )
        timings["generation"] = time.time() - t0

        if debug:
            total = sum(timings.values())
            print(f"\n[Timing] total={total:.1f}s")
            for stage, t in timings.items():
                print(f"  {stage}: {t:.1f}s")

        return PipelineResult(
            query=user_query,
            answer=gen_result["answer"],
            sources=gen_result["sources"],
            chunks_used=gen_result["chunks_used"],
            intent_category=intent.category,
            intent_specificity=intent.specificity,
            search_levels=intent.search_levels,
            reformulated_query=reformulated,
            expanded_queries=query_variants,
            retrieved_count=len(retrieved),
            reranked_count=len(reranked),
            timing=timings,
        )