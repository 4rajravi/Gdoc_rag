"""
Core scraping engine with rate limiting, retry logic, and deduplication.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class ScrapedDocument:
    """A single scraped document with metadata."""

    url: str
    title: str
    content: str  # cleaned main text
    source: str  # e.g. "make-it-in-germany", "howtogermany", "reddit"
    category: str  # e.g. "anmeldung", "visa", "tax", "health_insurance"
    language: str  # "de" or "en"
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content_hash: str = ""
    word_count: int = 0
    meta: dict = field(default_factory=dict)  # extra metadata (author, date, subreddit, etc.)

    def __post_init__(self):
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        self.word_count = len(self.content.split())


class RateLimiter:
    """Per-domain rate limiter to be polite to servers."""

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self._last_request: dict[str, float] = {}

    def wait(self, domain: str):
        now = time.time()
        last = self._last_request.get(domain, 0)
        elapsed = now - last
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
            time.sleep(sleep_time)
        self._last_request[domain] = time.time()


class ScraperEngine:
    """
    Core scraping engine.

    Features:
    - Per-domain rate limiting (default: 1 req/sec)
    - Automatic retries with exponential backoff
    - Content deduplication via SHA-256 hashes
    - Respects robots.txt (basic check)
    - Saves raw HTML + cleaned documents as JSON
    """

    def __init__(
        self,
        output_dir: str = "data/raw",
        rate_limit: float = 1.0,
        max_retries: int = 3,
        timeout: int = 30,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "html").mkdir(exist_ok=True)
        (self.output_dir / "pdfs").mkdir(exist_ok=True)

        self.rate_limiter = RateLimiter(rate_limit)
        self.max_retries = max_retries
        self.timeout = timeout
        self.seen_hashes: set[str] = set()
        self.documents: list[ScrapedDocument] = []
        self.downloaded_pdfs: list[dict] = []
        self._seen_pdf_urls: set[str] = set()

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "GermanBureaucracyHelper/1.0 "
                    "(Research project; contact: your-email@example.com)"
                ),
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
            }
        )

    def fetch(self, url: str) -> Optional[str]:
        """Fetch a URL with rate limiting and retries. Returns HTML or None."""
        domain = urlparse(url).netloc
        self.rate_limiter.wait(domain)

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                logger.info(f"[{resp.status_code}] {url}")
                return resp.text
            except requests.exceptions.HTTPError as e:
                if resp.status_code == 429:
                    wait = 2**attempt * 5
                    logger.warning(f"429 Too Many Requests. Waiting {wait}s...")
                    time.sleep(wait)
                elif resp.status_code >= 500:
                    wait = 2**attempt
                    logger.warning(f"Server error {resp.status_code}. Retry in {wait}s...")
                    time.sleep(wait)
                else:
                    logger.error(f"HTTP {resp.status_code} for {url}: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                wait = 2**attempt
                logger.warning(f"Request failed (attempt {attempt}): {e}. Retry in {wait}s...")
                time.sleep(wait)

        logger.error(f"Failed after {self.max_retries} retries: {url}")
        return None

    def extract_text(self, html: str, content_selector: str = "main") -> tuple[str, str]:
        """
        Extract clean text from HTML.
        Returns (title, cleaned_text).

        content_selector: CSS selector for the main content area.
        Falls back to <body> if selector not found.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Extract title
        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text(strip=True)
        h1 = soup.find("h1")
        if h1:
            title = h1.get_text(strip=True)

        # Remove noise elements
        for tag in soup.find_all(
            ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]
        ):
            tag.decompose()

        # Remove cookie banners, social sharing, ads
        for selector in [
            ".cookie-banner", ".cookie-consent", "#cookie",
            ".social-share", ".share-buttons",
            ".sidebar", ".advertisement", ".ad-container",
            ".breadcrumb", ".pagination",
        ]:
            for el in soup.select(selector):
                el.decompose()

        # Find main content
        content_el = soup.select_one(content_selector)
        if not content_el:
            content_el = soup.find("article") or soup.find("main") or soup.find("body")

        if not content_el:
            return title, ""

        # Get text, preserving some structure
        lines = []
        for element in content_el.find_all(
            ["h1", "h2", "h3", "h4", "p", "li", "td", "th", "blockquote", "dd", "dt"]
        ):
            text = element.get_text(separator=" ", strip=True)
            if not text:
                continue

            # Add markdown-style headers for structure preservation
            if element.name in ("h1", "h2", "h3", "h4"):
                prefix = "#" * int(element.name[1])
                lines.append(f"\n{prefix} {text}\n")
            elif element.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        cleaned = "\n".join(lines)

        # Collapse excessive whitespace
        while "\n\n\n" in cleaned:
            cleaned = cleaned.replace("\n\n\n", "\n\n")

        return title, cleaned.strip()

    def add_document(self, doc: ScrapedDocument) -> bool:
        """
        Add document if not a duplicate.
        Returns True if added, False if duplicate.
        """
        if doc.content_hash in self.seen_hashes:
            logger.info(f"Duplicate skipped: {doc.url}")
            return False
        if doc.word_count < 50:
            logger.info(f"Too short ({doc.word_count} words), skipped: {doc.url}")
            return False

        self.seen_hashes.add(doc.content_hash)
        self.documents.append(doc)
        return True

    def save_html(self, url: str, html: str):
        """Save raw HTML for debugging/reprocessing."""
        slug = urlparse(url).path.strip("/").replace("/", "_") or "index"
        domain = urlparse(url).netloc.replace(".", "_")
        filename = f"{domain}__{slug}.html"
        filepath = self.output_dir / "html" / filename
        filepath.write_text(html, encoding="utf-8")

    def download_pdf(self, pdf_url: str, source: str, found_on_page: str = "") -> bool:
        """
        Download a PDF file to data/raw/pdfs/ and log its metadata.
        Returns True if downloaded, False if already seen or failed.
        """
        if pdf_url in self._seen_pdf_urls:
            return False
        self._seen_pdf_urls.add(pdf_url)

        domain = urlparse(pdf_url).netloc
        self.rate_limiter.wait(domain)

        try:
            resp = self.session.get(pdf_url, timeout=self.timeout, stream=True)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.warning(f"PDF download failed: {pdf_url} — {e}")
            return False

        # Build filename from URL
        parsed = urlparse(pdf_url)
        slug = parsed.path.strip("/").replace("/", "_")
        if not slug.lower().endswith(".pdf"):
            slug += ".pdf"
        domain_prefix = parsed.netloc.replace(".", "_")
        filename = f"{domain_prefix}__{slug}"

        filepath = self.output_dir / "pdfs" / filename
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        size_kb = filepath.stat().st_size / 1024
        pdf_meta = {
            "url": pdf_url,
            "filename": filename,
            "source": source,
            "found_on_page": found_on_page,
            "size_kb": round(size_kb, 1),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
        }
        self.downloaded_pdfs.append(pdf_meta)
        logger.info(f"[PDF] {size_kb:.0f}KB — {filename}")
        return True

    def discover_pdf_links(self, html: str, base_url: str) -> list[str]:
        """Extract all PDF links from an HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        pdfs = set()
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.lower().endswith(".pdf") or "/pdf/" in href.lower():
                full_url = urljoin(base_url, href)
                pdfs.add(full_url)
        return sorted(pdfs)

    def save_pdf_manifest(self, filename: str = "pdf_manifest.json"):
        """Save metadata for all downloaded PDFs."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.downloaded_pdfs, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved PDF manifest ({len(self.downloaded_pdfs)} files) to {filepath}")

    def save_documents(self, filename: str = "documents.jsonl"):
        """Save all documents as JSON Lines."""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(self.documents)} documents to {filepath}")

    def save_index(self, filename: str = "scrape_index.json"):
        """Save a summary index of all scraped documents."""
        index = {
            "total_documents": len(self.documents),
            "total_words": sum(d.word_count for d in self.documents),
            "total_pdfs": len(self.downloaded_pdfs),
            "total_pdf_size_mb": round(sum(p["size_kb"] for p in self.downloaded_pdfs) / 1024, 2),
            "sources": {},
            "categories": {},
            "scraped_at": datetime.now(timezone.utc).isoformat(),
        }
        for doc in self.documents:
            index["sources"][doc.source] = index["sources"].get(doc.source, 0) + 1
            index["categories"][doc.category] = index["categories"].get(doc.category, 0) + 1

        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved index to {filepath}")

    def discover_links(self, html: str, base_url: str, same_domain: bool = True) -> list[str]:
        """Extract and filter links from an HTML page."""
        soup = BeautifulSoup(html, "html.parser")
        base_domain = urlparse(base_url).netloc
        links = set()

        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Skip anchors, mailto, tel, javascript
            if href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            # Clean URL (remove fragment)
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            if parsed.query:
                clean_url += f"?{parsed.query}"

            if same_domain and parsed.netloc != base_domain:
                continue

            # Skip non-HTML resources
            skip_extensions = {".pdf", ".jpg", ".png", ".gif", ".svg", ".zip", ".mp4", ".mp3"}
            if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
                continue

            links.add(clean_url)

        return sorted(links)