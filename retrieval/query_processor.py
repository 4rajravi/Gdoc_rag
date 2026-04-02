"""
Query Processor — handles unclear, vague, or multilingual user inputs.
======================================================================

Three stages:

1. Intent Classification
   - Detects topic category (anmeldung, visa, tax, etc.)
   - Detects query specificity: broad vs specific
   - Detects language (for bilingual corpus)
   This is rule-based + lightweight — no LLM call needed.

2. Query Reformulation (LLM-powered)
   - Rewrites vague inputs into precise, searchable queries
   - "I just moved here" → "How to register address (Anmeldung) after moving to Germany"
   - Expands abbreviations and adds German terms

3. Multi-Query Expansion (LLM-powered)
   - Generates 2-3 alternative phrasings of the query
   - Each variant is searched independently, results merged
   - Captures different aspects the user might mean

Design decisions:
- Intent classification is rule-based (fast, no API call)
- Reformulation + expansion use Ollama (same model as generation)
- We cache the Ollama client to avoid reconnection overhead
"""

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─── Category keyword map (reused from scraper, expanded) ────────────────────

CATEGORY_KEYWORDS = {
    "anmeldung": [
        "anmeldung", "register", "registration", "address", "bürgeramt",
        "einwohnermeldeamt", "meldebescheinigung", "wohnsitz", "moved",
        "moving", "new apartment", "wohnungsgeberbestätigung",
    ],
    "visa": [
        "visa", "visum", "residence permit", "aufenthaltstitel", "blue card",
        "blaue karte", "work permit", "aufenthaltserlaubnis", "niederlassung",
        "residence", "stay", "immigration", "einreise", "entry",
    ],
    "tax": [
        "tax", "steuer", "steuererklärung", "tax return", "tax id",
        "steuer-id", "lohnsteuer", "finanzamt", "elster", "tax class",
        "steuerklasse", "einkommensteuer",
    ],
    "health_insurance": [
        "health insurance", "krankenversicherung", "krankenkasse", "gkv", "pkv",
        "tk", "aok", "barmer", "doctor", "arzt", "medical", "insurance",
        "gesundheit",
    ],
    "banking": [
        "bank", "konto", "girokonto", "schufa", "account", "sparkasse",
        "volksbank", "credit",
    ],
    "housing": [
        "apartment", "wohnung", "wg", "flat", "rent", "miete", "kaution",
        "nebenkosten", "warmmiete", "kaltmiete", "landlord", "vermieter",
        "housing", "room", "zimmer",
    ],
    "work": [
        "job", "employment", "arbeitsvertrag", "contract", "minijob",
        "werkstudent", "working student", "salary", "gehalt", "arbeit",
    ],
    "university": [
        "university", "studium", "enrollment", "immatrikulation",
        "semester", "bafög", "student", "hochschule",
    ],
}

# Broad query indicators (user asking for overview, not specifics)
BROAD_INDICATORS = [
    "how does", "what is", "explain", "overview", "tell me about",
    "guide", "help me understand", "introduction", "basics",
    "was ist", "wie funktioniert", "erkläre",
]

# Specific query indicators (user wants a precise answer)
SPECIFIC_INDICATORS = [
    "how much", "where", "when", "deadline", "which documents",
    "how long", "cost", "fee", "address", "phone", "appointment",
    "form", "formular", "threshold", "limit", "requirement",
    "wieviel", "wo ", "wann", "welche",
]


@dataclass
class QueryIntent:
    """Analyzed intent of a user query."""
    original_query: str
    category: str  # detected topic
    category_confidence: float  # 0-1
    specificity: str  # "broad" or "specific"
    language: str  # "en" or "de"
    search_levels: list[str] = field(default_factory=list)  # which Qdrant collections to search


def classify_intent(query: str) -> QueryIntent:
    """
    Rule-based intent classification. Fast, no LLM call.

    Returns QueryIntent with:
    - category: best-matching topic
    - specificity: broad vs specific (determines which chunk levels to search)
    - language: en or de
    - search_levels: which collections to query
    """
    query_lower = query.lower()

    # ─── Detect category ───────────────────────────────────
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in query_lower)
        if score > 0:
            scores[category] = score

    if scores:
        category = max(scores, key=scores.get)
        max_score = scores[category]
        confidence = min(max_score / 3.0, 1.0)  # normalize
    else:
        category = "general"
        confidence = 0.0

    # ─── Detect specificity ────────────────────────────────
    broad_score = sum(1 for ind in BROAD_INDICATORS if ind in query_lower)
    specific_score = sum(1 for ind in SPECIFIC_INDICATORS if ind in query_lower)

    # Short queries are usually broad; long queries are usually specific
    word_count = len(query.split())
    if word_count <= 5:
        broad_score += 1
    elif word_count >= 12:
        specific_score += 1

    specificity = "specific" if specific_score > broad_score else "broad"

    # ─── Detect language ───────────────────────────────────
    german_markers = ["wie", "was", "wo", "ich", "mein", "für", "und", "der", "die", "das"]
    german_count = sum(1 for w in query_lower.split() if w in german_markers)
    language = "de" if german_count >= 2 else "en"

    # ─── Determine search levels ───────────────────────────
    if specificity == "broad":
        search_levels = ["L1", "L2"]  # overview + sections
    else:
        search_levels = ["L2", "L3"]  # sections + fine-grained

    return QueryIntent(
        original_query=query,
        category=category,
        category_confidence=confidence,
        specificity=specificity,
        language=language,
        search_levels=search_levels,
    )


def reformulate_query(query: str, intent: QueryIntent, ollama_client) -> str:
    """
    Use Ollama to rewrite a vague query into a precise, searchable query.

    Only called when the query seems vague or ambiguous.
    """
    # Skip reformulation for already-clear queries
    if intent.category_confidence >= 0.7 and len(query.split()) >= 6:
        logger.debug(f"Query clear enough, skipping reformulation: {query}")
        return query

    prompt = f"""You are a query rewriter for a German bureaucracy information system.

Rewrite the following user query to be more specific and searchable.
Add relevant German terms in parentheses where helpful.
Keep the rewritten query under 30 words.
Return ONLY the rewritten query, nothing else.

User query: {query}

Rewritten query:"""

    try:
        response = ollama_client.generate(prompt)
        reformulated = response.strip().strip('"').strip("'")
        if reformulated and len(reformulated) > 5:
            logger.info(f"Reformulated: '{query}' → '{reformulated}'")
            return reformulated
    except Exception as e:
        logger.warning(f"Reformulation failed: {e}")

    return query  # fallback to original


def expand_query(query: str, intent: QueryIntent, ollama_client) -> list[str]:
    """
    Generate 2-3 alternative phrasings of the query for diverse retrieval.

    Each variant captures a different angle or interpretation.
    Results from all variants are merged and deduplicated before reranking.
    """
    prompt = f"""You are a query expansion tool for a German bureaucracy information system.

Generate exactly 3 alternative search queries for the following user question.
Each alternative should capture a different aspect or phrasing.
Include relevant German terms where appropriate.
Return ONLY the 3 queries, one per line, no numbering or bullets.

User question: {query}
Topic category: {intent.category}

Alternative queries:"""

    try:
        response = ollama_client.generate(prompt)
        lines = [l.strip().strip("-").strip("•").strip() for l in response.strip().split("\n")]
        variants = [l for l in lines if l and len(l) > 5 and len(l) < 200]

        if variants:
            # Always include original query
            all_queries = [query] + variants[:3]
            logger.info(f"Expanded to {len(all_queries)} variants: {all_queries}")
            return all_queries
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")

    return [query]  # fallback to original only