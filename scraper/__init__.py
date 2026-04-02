from .core import ScraperEngine, ScrapedDocument
from .sources import (
    MakeItInGermanyScraper,
    HowToGermanyScraper,
    AllAboutBerlinScraper,
    HamburgDeScraper,
    RedditScraper,
    detect_category,
    detect_language,
)

__all__ = [
    "ScraperEngine",
    "ScrapedDocument",
    "MakeItInGermanyScraper",
    "HowToGermanyScraper",
    "AllAboutBerlinScraper",
    "HamburgDeScraper",
    "RedditScraper",
    "detect_category",
    "detect_language",
]