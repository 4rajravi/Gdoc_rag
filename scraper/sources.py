"""
Source-specific scrapers for German bureaucracy content.

Each scraper knows the structure of its target site:
- Where to find topic pages
- How to extract clean content
- How to categorize documents
"""

import logging
import re
from urllib.parse import urljoin, urlparse

from .core import ScraperEngine, ScrapedDocument

logger = logging.getLogger(__name__)


# ─── Category detection ──────────────────────────────────────────────────────

CATEGORY_KEYWORDS = {
    "anmeldung": [
        "anmeldung", "registration", "register your address", "residents registration",
        "einwohnermeldeamt", "bürgeramt", "meldebescheinigung", "wohnsitz",
    ],
    "visa": [
        "visa", "visum", "residence permit", "aufenthaltstitel", "aufenthaltserlaubnis",
        "blue card", "blaue karte", "work permit", "arbeitserlaubnis",
        "aufenthaltsgenehmigung", "niederlassungserlaubnis",
    ],
    "tax": [
        "tax", "steuer", "steuererklärung", "tax return", "tax id", "steuer-id",
        "lohnsteuer", "einkommensteuer", "finanzamt", "elster", "tax class",
        "steuerklasse",
    ],
    "health_insurance": [
        "health insurance", "krankenversicherung", "gesetzliche krankenversicherung",
        "private krankenversicherung", "gkv", "pkv", "tk", "aok", "barmer",
        "krankenkasse",
    ],
    "banking": [
        "bank account", "bankkonto", "girokonto", "schufa", "opening a bank",
        "sparkasse", "volksbank",
    ],
    "housing": [
        "apartment", "wohnung", "wg", "wohnungssuche", "mietvertrag", "kaution",
        "nebenkosten", "warmmiete", "kaltmiete", "schufa", "landlord", "vermieter",
    ],
    "work": [
        "employment", "arbeitsvertrag", "job contract", "minijob", "werkstudent",
        "working student", "sozialversicherung", "social security number",
        "rentenversicherung",
    ],
    "university": [
        "enrollment", "immatrikulation", "semester ticket", "semesterbeitrag",
        "student visa", "studienkolleg", "bafög",
    ],
    "general": [],  # fallback
}


def detect_category(text: str, url: str = "") -> str:
    """Detect document category from content and URL."""
    combined = (text + " " + url).lower()
    scores = {}
    for category, keywords in CATEGORY_KEYWORDS.items():
        if category == "general":
            continue
        score = sum(1 for kw in keywords if kw in combined)
        if score > 0:
            scores[category] = score

    if scores:
        return max(scores, key=scores.get)
    return "general"


def detect_language(text: str) -> str:
    """Simple heuristic language detection (de vs en)."""
    german_markers = [
        "und", "oder", "für", "mit", "bei", "nach", "über", "unter",
        "können", "müssen", "werden", "haben", "sein",
        "der", "die", "das", "ein", "eine", "ist", "sind",
    ]
    words = text.lower().split()[:200]  # check first 200 words
    german_count = sum(1 for w in words if w in german_markers)
    return "de" if german_count > 10 else "en"


# ─── Make it in Germany ──────────────────────────────────────────────────────

class MakeItInGermanyScraper:
    """
    Scrapes make-it-in-germany.com — the official German government portal
    for skilled workers and international residents.

    Structure:
    - /en/living-in-germany/ → housing, registration, insurance
    - /en/working-in-germany/ → permits, contracts, recognition
    - /en/visa/ → all visa types
    - /en/studying-in-germany/ → student-specific info
    """

    BASE_URL = "https://www.make-it-in-germany.com"

    # Seed pages covering major topic areas (verified April 2026)
    SEED_URLS = [
        # Living in Germany
        "/en/living-in-germany/housing-mobility/looking",
        "/en/living-in-germany/housing/settling-in",
        "/en/living-in-germany/money-insurance/bank-account",
        "/en/living-in-germany/money-insurance/health-insurance",
        "/en/living-in-germany/money-insurance/additional",
        "/en/living-in-germany/german/experiences-students",
        "/en/living-in-germany/discover-germany/immigration",
        # Visa & residence
        "/en/visa-residence/types",
        "/en/visa-residence/procedure/do-i-need-visa",
        "/en/visa-residence/procedure/entry-process",
        "/en/visa-residence/procedure/application-forms",
        "/en/visa-residence/types/studying",
        "/en/visa-residence/types/work-qualified-professionals",
        "/en/visa-residence/types/other",
        "/en/visa-residence/types/recognition",
        "/en/visa-residence/living-permanently",
        "/en/visa-residence/living-permanently/settlement-permit",
        "/en/visa-residence/opportunity-card/job-search-opportunity-card",
        # Working in Germany
        "/en/working-in-germany/working-environment/salary-taxes-social-security",
        "/en/working-in-germany/setting-up-business/steps",
        # Service
        "/en/service/faq",
    ]

    def __init__(self, engine: ScraperEngine):
        self.engine = engine
        self.visited: set[str] = set()

    def scrape(self, max_pages: int = 100, follow_links: bool = True):
        """Scrape seed pages and optionally follow internal links."""
        to_visit = [urljoin(self.BASE_URL, path) for path in self.SEED_URLS]

        pages_scraped = 0
        while to_visit and pages_scraped < max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            html = self.engine.fetch(url)
            if not html:
                continue

            self.engine.save_html(url, html)
            title, content = self.engine.extract_text(
                html, content_selector=".c-content-area, .content-area, main"
            )

            # Collect PDFs linked from this page
            for pdf_url in self.engine.discover_pdf_links(html, url):
                self.engine.download_pdf(pdf_url, source="make-it-in-germany", found_on_page=url)

            if not content:
                continue

            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                source="make-it-in-germany",
                category=detect_category(content, url),
                language=detect_language(content),
            )

            if self.engine.add_document(doc):
                pages_scraped += 1
                logger.info(
                    f"[make-it-in-germany] ({pages_scraped}/{max_pages}) "
                    f"{doc.category}: {title[:60]}"
                )

            # Discover and queue internal links
            if follow_links:
                links = self.engine.discover_links(html, url, same_domain=True)
                relevant = [
                    l for l in links
                    if any(seg in l for seg in [
                        "/en/living", "/en/visa-residence", "/en/working",
                        "/en/studying", "/en/service",
                    ])
                    and l not in self.visited
                ]
                to_visit.extend(relevant)


# ─── How To Germany ──────────────────────────────────────────────────────────

class HowToGermanyScraper:
    """
    Scrapes howtogermany.com — comprehensive English-language guide
    for expats in Germany. Long-form articles organized by topic.
    """

    BASE_URL = "https://www.howtogermany.com"

    SEED_URLS = [
        # Visa & residency
        "/visa-residency/residency/residence-permits/",
        "/visa-residency/residency/the-eu-blue-card-for-germany/",
        # Taxes
        "/taxes/german-taxes/",
        "/taxes/american-expats-and-the-irs-in-germany/",
        # Insurance
        "/insurance/health-insurance/health-insurance-options-germany/",
        "/insurance/health-insurance/german-government-health-insurance-gkv/",
        "/insurance/health-insurance/paying-medical-expenses-health-insurance-claims-germany/",
        "/insurance/health-insurance/tips-about-german-statutory-health-insurance/",
        # Housing
        "/housing/rental/housing-what-you-need-to-know/",
        # Jobs & work
        "/jobs/rights/social-security-and-employee-benefits-in-germany/",
        "/jobs/",
    ]

    def __init__(self, engine: ScraperEngine):
        self.engine = engine
        self.visited: set[str] = set()

    def scrape(self, max_pages: int = 50, follow_links: bool = True):
        """Scrape seed pages and follow internal links."""
        to_visit = [urljoin(self.BASE_URL, path) for path in self.SEED_URLS]

        pages_scraped = 0
        while to_visit and pages_scraped < max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            html = self.engine.fetch(url)
            if not html:
                continue

            self.engine.save_html(url, html)
            title, content = self.engine.extract_text(
                html, content_selector=".entry-content, .post-content, article, .content"
            )

            # Collect PDFs linked from this page
            for pdf_url in self.engine.discover_pdf_links(html, url):
                self.engine.download_pdf(pdf_url, source="howtogermany", found_on_page=url)

            if not content:
                continue

            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                source="howtogermany",
                category=detect_category(content, url),
                language="en",
            )

            if self.engine.add_document(doc):
                pages_scraped += 1
                logger.info(
                    f"[howtogermany] ({pages_scraped}/{max_pages}) "
                    f"{doc.category}: {title[:60]}"
                )

            if follow_links:
                links = self.engine.discover_links(html, url, same_domain=True)
                relevant = [
                    l for l in links
                    if any(seg in l for seg in [
                        "/visa-residency/", "/taxes/", "/insurance/",
                        "/housing/", "/jobs/", "/banking/",
                    ])
                    and l not in self.visited
                    and "/storefronts/" not in l  # skip sponsor pages
                ]
                to_visit.extend(relevant)


# ─── All About Berlin ────────────────────────────────────────────────────────

class AllAboutBerlinScraper:
    """
    Scrapes allaboutberlin.com — excellent practical guides
    for bureaucracy in Germany (Berlin-focused but broadly applicable).
    """

    BASE_URL = "https://allaboutberlin.com"

    SEED_URLS = [
        "/guides/first-steps-germany",
        "/guides/anmeldung",
        "/guides/residence-permit",
        "/guides/german-tax-system",
        "/guides/tax-id-germany",
        "/guides/first-tax-return-germany",
        "/guides/health-insurance-germany",
        "/guides/public-health-insurance-germany",
        "/guides/private-health-insurance-germany",
        "/guides/bank-account-germany",
        "/guides/schufa",
        "/guides/find-apartment-germany",
        "/guides/german-apartment-scams",
        "/guides/drivers-license-germany",
        "/guides/freelance-germany",
        "/guides/settle-in-berlin",
        "/guides/moving-to-berlin",
    ]

    def __init__(self, engine: ScraperEngine):
        self.engine = engine
        self.visited: set[str] = set()

    def scrape(self, max_pages: int = 50, follow_links: bool = True):
        to_visit = [urljoin(self.BASE_URL, path) for path in self.SEED_URLS]

        pages_scraped = 0
        while to_visit and pages_scraped < max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            html = self.engine.fetch(url)
            if not html:
                continue

            self.engine.save_html(url, html)
            title, content = self.engine.extract_text(
                html, content_selector="article, .guide-content, .entry-content, main"
            )

            # Collect PDFs linked from this page
            for pdf_url in self.engine.discover_pdf_links(html, url):
                self.engine.download_pdf(pdf_url, source="allaboutberlin", found_on_page=url)

            if not content:
                continue

            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                source="allaboutberlin",
                category=detect_category(content, url),
                language="en",
            )

            if self.engine.add_document(doc):
                pages_scraped += 1
                logger.info(
                    f"[allaboutberlin] ({pages_scraped}/{max_pages}) "
                    f"{doc.category}: {title[:60]}"
                )

            if follow_links:
                links = self.engine.discover_links(html, url, same_domain=True)
                relevant = [
                    l for l in links
                    if "/guides/" in l and l not in self.visited
                ]
                to_visit.extend(relevant)


# ─── Hamburg.de (official city portal) ────────────────────────────────────────

class HamburgDeScraper:
    """
    Scrapes hamburg.de and welcome.hamburg.de — official Hamburg city
    government pages for immigration, registration, and administrative services.

    Content is primarily in German but extremely authoritative.
    """

    SEED_URLS = [
        # welcome.hamburg.de (Welcome Center portal)
        "https://welcome.hamburg.de/leben-in-hamburg/deutsch-lernen/integrationskurse-416418",
        "https://welcome.hamburg.de/unsere-services/fk-drittstaat-417190",
        # hamburg.de — international students info
        "https://www.hamburg.de/politik-und-verwaltung/behoerden/behoerde-fuer-inneres-und-sport/aemter/amt-fuer-migration/zentrale-auslaenderangelegenheiten/allgemeines-aufenthaltsrecht-und-zentraler-service/studenten-18018",
    ]

    def __init__(self, engine: ScraperEngine):
        self.engine = engine
        self.visited: set[str] = set()

    def scrape(self, max_pages: int = 30, follow_links: bool = True):
        to_visit = list(self.SEED_URLS)

        pages_scraped = 0
        while to_visit and pages_scraped < max_pages:
            url = to_visit.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)

            html = self.engine.fetch(url)
            if not html:
                continue

            self.engine.save_html(url, html)
            title, content = self.engine.extract_text(
                html, content_selector=".module-text, .content-main, article, main"
            )

            # Collect PDFs linked from this page
            for pdf_url in self.engine.discover_pdf_links(html, url):
                self.engine.download_pdf(pdf_url, source="hamburg.de", found_on_page=url)

            if not content:
                continue

            doc = ScrapedDocument(
                url=url,
                title=title,
                content=content,
                source="hamburg.de",
                category=detect_category(content, url),
                language=detect_language(content),
            )

            if self.engine.add_document(doc):
                pages_scraped += 1
                logger.info(
                    f"[hamburg.de] ({pages_scraped}/{max_pages}) "
                    f"{doc.category}: {title[:60]}"
                )

            if follow_links:
                links = self.engine.discover_links(html, url, same_domain=True)
                relevant = [
                    l for l in links
                    if any(seg in l for seg in [
                        "/leben-in-hamburg/", "/unsere-services/",
                        "/amt-fuer-migration/",
                    ])
                    and l not in self.visited
                ]
                to_visit.extend(relevant)


# ─── Reddit Scraper (via old.reddit.com, no API key needed) ──────────────────

class RedditScraper:
    """
    Scrapes Reddit threads from relevant subreddits using old.reddit.com
    (no API key required, just HTTP requests).

    Targets:
    - r/germany — general expat questions
    - r/de — German-language discussions
    - r/hamburg — Hamburg-specific threads
    - r/iwantout — immigration experiences
    """

    SUBREDDIT_SEARCHES = {
        "germany": [
            "anmeldung", "residence permit", "tax return steuererklärung",
            "health insurance", "blue card", "bank account schufa",
            "apartment search", "werkstudent visa",
        ],
        "hamburg": [
            "anmeldung", "bürgeramt", "apartment",
            "visa", "registration",
        ],
    }

    def __init__(self, engine: ScraperEngine):
        self.engine = engine
        self.visited: set[str] = set()
        # Reddit needs a browser-like User-Agent
        self.engine.session.headers["User-Agent"] = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    def scrape(self, max_threads_per_query: int = 5):
        """Search subreddits and scrape top threads."""
        for subreddit, queries in self.SUBREDDIT_SEARCHES.items():
            for query in queries:
                self._search_and_scrape(subreddit, query, max_threads_per_query)

    def _search_and_scrape(self, subreddit: str, query: str, max_threads: int):
        """Search a subreddit and scrape matching threads."""
        search_url = (
            f"https://old.reddit.com/r/{subreddit}/search"
            f"?q={query}&restrict_sr=on&sort=relevance&t=all"
        )

        html = self.engine.fetch(search_url)
        if not html:
            return

        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        thread_links = []

        for link in soup.select("a.search-title, a.title"):
            href = link.get("href", "")
            if "/comments/" in href:
                full_url = href if href.startswith("http") else f"https://old.reddit.com{href}"
                thread_links.append(full_url)

        for thread_url in thread_links[:max_threads]:
            if thread_url in self.visited:
                continue
            self.visited.add(thread_url)

            self._scrape_thread(thread_url, subreddit, query)

    def _scrape_thread(self, url: str, subreddit: str, search_query: str):
        """Scrape a single Reddit thread (post + comments)."""
        html = self.engine.fetch(url)
        if not html:
            return

        soup = BeautifulSoup(html, "html.parser")

        # Get post title
        title_el = soup.select_one("a.title, p.title a")
        title = title_el.get_text(strip=True) if title_el else "Reddit Thread"

        # Get post body
        post_body = ""
        usertext = soup.select_one(".expando .usertext-body")
        if usertext:
            post_body = usertext.get_text(separator="\n", strip=True)

        # Get comments (top-level and replies)
        comments = []
        for comment in soup.select(".comment .usertext-body .md"):
            text = comment.get_text(separator="\n", strip=True)
            if text and len(text) > 30:  # skip very short comments
                comments.append(text)

        # Combine into a structured document
        parts = [f"# {title}", ""]
        if post_body:
            parts.extend(["## Original Post", post_body, ""])
        if comments:
            parts.append("## Community Responses")
            for i, comment in enumerate(comments[:15], 1):  # cap at 15 comments
                parts.extend([f"\n### Response {i}", comment])

        content = "\n".join(parts)

        doc = ScrapedDocument(
            url=url,
            title=title,
            content=content,
            source="reddit",
            category=detect_category(content, url),
            language=detect_language(content),
            meta={
                "subreddit": subreddit,
                "search_query": search_query,
                "comment_count": len(comments),
            },
        )

        if self.engine.add_document(doc):
            logger.info(
                f"[reddit/r/{subreddit}] {doc.category}: {title[:60]} "
                f"({len(comments)} comments)"
            )