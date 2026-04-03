"""
API request and response models.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    mode: str = Field(default="full", pattern="^(full|fast|debug)$")
    category_override: str | None = None


class Source(BaseModel):
    title: str
    url: str
    source: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    chunks_used: int
    intent_category: str
    intent_specificity: str
    search_levels: list[str]
    reformulated_query: str
    expanded_queries: list[str]
    retrieved_count: int
    reranked_count: int
    timing: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    ollama: str
    collections: dict[str, int]