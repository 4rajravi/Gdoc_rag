"""
FastAPI backend for the German Bureaucracy RAG system.
======================================================

Endpoints:
    POST /api/query     — Ask a question, get a grounded answer
    GET  /api/health    — Check system status (Qdrant, Ollama)
    GET  /api/categories — List available topic categories

Usage:
    uvicorn api.main:app --reload --port 8000
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from api import QueryRequest, QueryResponse, HealthResponse, Source
from processing.embedder import Embedder
from retrieval.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

# ─── Globals (initialized at startup) ────────────────────────────────────────

pipeline: RAGPipeline | None = None
qdrant_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to services at startup."""
    global pipeline, qdrant_client

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info("Starting up...")

    from qdrant_client import QdrantClient

    qdrant_host = os.environ.get("QDRANT_HOST", "localhost")
    qdrant_port = int(os.environ.get("QDRANT_PORT", "6333"))
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    ollama_url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    logger.info("Connecting to Qdrant...")
    qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)

    logger.info("Loading embedding model...")
    embedder = Embedder()

    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(
        qdrant_client=qdrant_client,
        embedder=embedder,
        ollama_model=ollama_model,
        ollama_url=ollama_url,
    )

    logger.info("Ready to serve requests!")
    yield

    logger.info("Shutting down...")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="German Bureaucracy RAG",
    description="AI-powered assistant for navigating German bureaucracy",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Ask a question about German bureaucracy."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        result = pipeline.query(
            user_query=request.question,
            mode=request.mode,
            category_override=request.category_override,
        )

        return QueryResponse(
            answer=result.answer,
            sources=[Source(**s) for s in result.sources],
            chunks_used=result.chunks_used,
            intent_category=result.intent_category,
            intent_specificity=result.intent_specificity,
            search_levels=result.search_levels,
            reformulated_query=result.reformulated_query,
            expanded_queries=result.expanded_queries,
            retrieved_count=result.retrieved_count,
            reranked_count=result.reranked_count,
            timing=result.timing,
        )
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Check system health."""
    qdrant_status = "unknown"
    ollama_status = "unknown"
    collections = {}

    # Check Qdrant
    try:
        for name in ["chunks_l1", "chunks_l2", "chunks_l3"]:
            info = qdrant_client.get_collection(name)
            collections[name] = info.points_count
        qdrant_status = "healthy"
    except Exception as e:
        qdrant_status = f"error: {e}"

    # Check Ollama
    try:
        import requests as req
        resp = req.get("http://localhost:11434/v1/models", timeout=3)
        if resp.status_code == 200:
            ollama_status = "healthy"
        else:
            ollama_status = f"error: HTTP {resp.status_code}"
    except Exception as e:
        ollama_status = f"error: {e}"

    status = "healthy" if qdrant_status == "healthy" and ollama_status == "healthy" else "degraded"

    return HealthResponse(
        status=status,
        qdrant=qdrant_status,
        ollama=ollama_status,
        collections=collections,
    )


@app.get("/api/categories")
async def categories():
    """List available topic categories."""
    return {
        "categories": [
            {"id": "anmeldung", "label": "Address Registration (Anmeldung)"},
            {"id": "visa", "label": "Visa & Residence Permits"},
            {"id": "tax", "label": "Taxes (Steuererklärung)"},
            {"id": "health_insurance", "label": "Health Insurance (Krankenversicherung)"},
            {"id": "banking", "label": "Banking & Schufa"},
            {"id": "housing", "label": "Housing & Apartments"},
            {"id": "work", "label": "Work & Employment"},
            {"id": "university", "label": "University & Students"},
        ]
    }


# ─── Serve React frontend (production) ───────────────────────────────────────

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dir):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dir, "assets")), name="assets")

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve React app for all non-API routes."""
        return FileResponse(os.path.join(frontend_dir, "index.html"))