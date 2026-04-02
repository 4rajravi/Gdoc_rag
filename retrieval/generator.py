"""
LLM Generator — Ollama-powered answer generation with source citations.
========================================================================

The generator takes reranked chunks and assembles them into a prompt
that tells the LLM to answer based ONLY on the provided context.

Prompt engineering decisions:
- System prompt establishes the "German bureaucracy helper" persona
- Retrieved chunks are formatted with source labels
- LLM is instructed to cite sources and say "I don't know" when context
  doesn't cover the question (prevents hallucination)
- Response includes source URLs for verification
"""

import logging
import requests
import json

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1:8b"
OLLAMA_URL = "http://localhost:11434"

SYSTEM_PROMPT = """You are a helpful assistant specializing in German bureaucracy for international residents and students.

Your role is to answer questions about living in Germany: Anmeldung (address registration), visa and residence permits, health insurance (Krankenversicherung), taxes (Steuererklärung), banking, housing, and work permits.

Rules:
1. Answer ONLY based on the provided context below. Do not use outside knowledge.
2. If the context doesn't contain enough information, say so clearly.
3. When citing information, mention the source (e.g., "According to allaboutberlin.com...").
4. Use German terms in parentheses where helpful (e.g., "registration certificate (Meldebescheinigung)").
5. Be practical and actionable — give step-by-step guidance when possible.
6. If the user writes in German, respond in German."""


class OllamaClient:
    """
    Lightweight Ollama API client.

    Supports both generation (for answers) and short completions
    (for query reformulation/expansion).
    """

    def __init__(self, model: str = DEFAULT_MODEL, base_url: str = OLLAMA_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._verify_connection()

    def _verify_connection(self):
        """Check that Ollama is running and model is available."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]

            # Check if our model is available (handle tag variations)
            model_base = self.model.split(":")[0]
            available = any(model_base in m for m in models)

            if available:
                logger.info(f"Ollama connected. Model '{self.model}' available.")
            else:
                logger.warning(
                    f"Model '{self.model}' not found. Available: {models}. "
                    f"Run: ollama pull {self.model}"
                )
        except requests.exceptions.ConnectionError:
            logger.error(
                "Cannot connect to Ollama. Make sure it's running: ollama serve"
            )
            raise

    def generate(self, prompt: str, system: str = None, temperature: float = 0.1) -> str:
        """
        Generate a completion from Ollama.

        Uses low temperature (0.1) for factual responses.
        For query reformulation, you might want 0.3-0.5.
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    def chat(self, messages: list[dict], temperature: float = 0.1) -> str:
        """
        Chat-style completion (for multi-turn or system+user messages).
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error(f"Ollama chat failed: {e}")
            raise


def assemble_context(chunks: list) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.

    Each chunk is labeled with its source and section for citation.
    """
    context_parts = []

    for i, chunk in enumerate(chunks, 1):
        source_label = f"[Source {i}: {chunk.source} — {chunk.section_header}]"
        context_parts.append(f"{source_label}\n{chunk.text}\n")

    return "\n---\n".join(context_parts)


def build_source_list(chunks: list) -> str:
    """Build a source reference list for the response footer."""
    seen = set()
    sources = []

    for chunk in chunks:
        key = chunk.doc_url
        if key not in seen:
            seen.add(key)
            sources.append(f"- {chunk.doc_title}: {chunk.doc_url}")

    return "\n".join(sources)


class Generator:
    """
    Generates answers from retrieved context using Ollama.

    Usage:
        generator = Generator(ollama_client)
        answer = generator.generate_answer(query, reranked_chunks)
    """

    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client

    def generate_answer(
        self,
        query: str,
        chunks: list,
        include_sources: bool = True,
    ) -> dict:
        """
        Generate an answer grounded in retrieved chunks.

        Returns dict with:
        - answer: the generated text
        - sources: list of source URLs
        - chunks_used: number of chunks in context
        """
        if not chunks:
            return {
                "answer": "I couldn't find relevant information to answer your question. Could you try rephrasing it?",
                "sources": [],
                "chunks_used": 0,
            }

        # Assemble context from chunks
        context = assemble_context(chunks)

        # Build the prompt
        user_prompt = f"""Context (use ONLY this information to answer):

{context}

---

Question: {query}

Answer the question based on the context above. Cite sources by their labels (e.g., "According to Source 1..."). If the context doesn't fully answer the question, say what you can and note what's missing."""

        # Generate with Ollama
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer = self.client.chat(messages, temperature=0.1)

        # Build source list
        source_urls = []
        seen = set()
        for chunk in chunks:
            if chunk.doc_url not in seen:
                seen.add(chunk.doc_url)
                source_urls.append({
                    "title": chunk.doc_title,
                    "url": chunk.doc_url,
                    "source": chunk.source,
                })

        result = {
            "answer": answer,
            "sources": source_urls,
            "chunks_used": len(chunks),
        }

        return result