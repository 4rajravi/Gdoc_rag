# German Bureaucracy RAG

An end-to-end Retrieval-Augmented Generation system that helps international residents and students navigate German bureaucracy — Anmeldung, visa applications, health insurance, taxes, and more.

Built with hierarchical chunking, multilingual embeddings, cross-encoder reranking, and a multi-stage query processing pipeline. Fully local — no API keys required.

![Landing Page](screenshots/landing.png)

![Chat with Debug](screenshots/chat-debug.png)

---

## What it does

Ask a question in English or German about any aspect of living in Germany. The system scrapes authoritative sources, chunks them hierarchically, retrieves the most relevant passages across multiple levels of detail, reranks them with a cross-encoder, and generates a grounded answer with source citations.

**Covers:** Anmeldung (address registration) · Visa & residence permits · Health insurance (GKV/PKV) · Taxes (Steuererklärung) · Banking & Schufa · Housing · Work permits · University enrollment

**Sources indexed:** make-it-in-germany.com (official government portal) · allaboutberlin.com · howtogermany.com · hamburg.de · Reddit threads · 47 official PDFs

---

## Architecture

![Architecture Diagram](screenshots/architecture.png)

**Indexing pipeline:** Raw documents → hierarchical chunking (L1 summaries, L2 sections, L3 fine-grained) → multilingual-e5-base embeddings → Qdrant vector store (3 collections)

**Query pipeline:** User query → intent classification → query reformulation → multi-query expansion → hierarchical retrieval → cross-encoder reranking → LLM generation with source citations

---

## Key technical decisions

**Hierarchical chunking** — Three levels of chunks serve different query types. Broad questions ("How does health insurance work?") hit L1 summaries + L2 sections. Specific questions ("What's the income threshold for private insurance?") hit L2 sections + L3 fine-grained passages. Each chunk carries metadata (source, category, section header, URL) enabling filtered retrieval.

**Multilingual embeddings** — `intfloat/multilingual-e5-base` handles both English and German queries against a mixed-language corpus. Critical detail: e5 models require `"passage: "` and `"query: "` prefixes — without them, retrieval quality drops significantly.

**Cross-encoder reranking** — Bi-encoder retrieval (embedding similarity) is fast but approximate. The cross-encoder (`ms-marco-MiniLM-L-6-v2`) reads query and document together, enabling much deeper relevance scoring. Pipeline: retrieve 20 candidates fast → rerank to best 5 precisely.

**Query processing** — Three strategies for handling unclear inputs: (1) rule-based intent classification detects topic and query specificity to route to the right chunk levels, (2) LLM-powered query reformulation rewrites vague inputs into precise searches, (3) multi-query expansion generates 2-3 alternative phrasings for diverse retrieval, results merged and deduplicated before reranking.

**Fully local stack** — Ollama for generation, sentence-transformers for embeddings and reranking, Qdrant for vector search. No external API dependencies. Swappable LLM backend via config.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Scraping | Python, BeautifulSoup, requests |
| Chunking | Custom hierarchical splitter (L1/L2/L3) |
| Embeddings | multilingual-e5-base (768d, sentence-transformers) |
| Vector store | Qdrant (Docker, 3 collections, DOT product) |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Query processing | Rule-based intent + LLM reformulation/expansion |
| Generation | Ollama (llama3.1, local) |
| Backend | FastAPI |
| Frontend | React + Vite |

---

## Quickstart

**Prerequisites:** Docker, Python 3.11+, Node.js 18+, Ollama

```bash
# 1. Clone and install
git clone https://github.com/yourusername/german-bureaucracy-rag.git
cd german-bureaucracy-rag
pip install -r requirements.txt

# 2. Start services
docker compose up -d                  # Qdrant
ollama pull llama3.1                  # download LLM (~4.7GB)

# 3. Build the knowledge base
python run_scraper.py --all           # scrape sources (~5 min)
python run_indexing.py --recreate     # chunk + embed + index (~6 min)

# 4. Start the app
python run_server.py                  # FastAPI on :8000

# 5. Start frontend (separate terminal)
cd frontend && npm install && npm run dev   # React on :3000
```

Open **http://localhost:3000**

---

## Usage

**Web interface** — Full chat UI with suggested questions, topic sidebar, debug mode toggle, and source citations.

**CLI** — Direct terminal access to the pipeline:
```bash
python run_query.py --debug           # interactive with pipeline internals
python run_query.py -q "How do I register my address?"
python run_query.py --fast            # skip query expansion for lower latency
```

**API** — REST endpoints for integration:
```bash
# Ask a question
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What documents for Anmeldung?", "mode": "full"}'

# Health check
curl http://localhost:8000/api/health
```

---

## Debug mode

Toggle debug in the UI to inspect every pipeline stage per query:

![Debug Panel](screenshots/debug-panel.png)

Shows intent classification, search levels, retrieved/reranked chunk counts, reformulated query, expanded variants, and per-stage timing bars (intent → reformulation → expansion → retrieval → reranking → generation).

---

## Corpus stats

| Metric | Value |
|--------|-------|
| Documents scraped | 156 |
| Total words | 125,637 |
| PDFs downloaded | 47 (41.9 MB) |
| L1 chunks (summaries) | 156 |
| L2 chunks (sections) | 710 |
| L3 chunks (fine-grained) | 528 |
| Total vectors indexed | 1,394 |
| Embedding dimension | 768 |
| Categories covered | 9 |

---

## Adding new sources

1. Create a scraper class in `scraper/sources.py` following the existing pattern
2. Register it in `AVAILABLE_SOURCES` in `run_scraper.py`
3. Re-run scraping and indexing:
```bash
python run_scraper.py --sources your-new-source
python run_indexing.py --recreate
```

---

## License

MIT