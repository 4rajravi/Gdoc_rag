#!/usr/bin/env python3
"""
German Bureaucracy RAG — Data Scraper
=====================================

Scrapes official guides, expat portals, and Reddit threads about
German bureaucracy (Anmeldung, visas, taxes, health insurance, etc.)
and saves structured documents for downstream RAG ingestion.

Usage:
    # Scrape all sources
    python run_scraper.py --all

    # Scrape specific sources
    python run_scraper.py --sources make-it-in-germany howtogermany reddit

    # Dry run — show what would be scraped
    python run_scraper.py --all --dry-run

    # Custom output directory and rate limit
    python run_scraper.py --all --output data/my_corpus --rate-limit 0.5
"""

import argparse
import logging
import sys

from scraper import (
    ScraperEngine,
    MakeItInGermanyScraper,
    HowToGermanyScraper,
    AllAboutBerlinScraper,
    HamburgDeScraper,
    RedditScraper,
)


AVAILABLE_SOURCES = {
    "make-it-in-germany": {
        "class": MakeItInGermanyScraper,
        "description": "Official German gov portal for international workers/students",
        "default_max_pages": 80,
    },
    "howtogermany": {
        "class": HowToGermanyScraper,
        "description": "Comprehensive English-language expat guide",
        "default_max_pages": 40,
    },
    "allaboutberlin": {
        "class": AllAboutBerlinScraper,
        "description": "Practical bureaucracy guides (Berlin-focused, broadly applicable)",
        "default_max_pages": 40,
    },
    "hamburg.de": {
        "class": HamburgDeScraper,
        "description": "Official Hamburg city portal (German language)",
        "default_max_pages": 25,
    },
    "reddit": {
        "class": RedditScraper,
        "description": "Reddit threads from r/germany, r/hamburg",
        "default_max_pages": None,  # uses max_threads_per_query instead
    },
}


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("scraper.log", encoding="utf-8"),
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape German bureaucracy content for RAG corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(AVAILABLE_SOURCES.keys()),
        help="Which sources to scrape",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Scrape all available sources",
    )
    parser.add_argument(
        "--output",
        default="data/raw",
        help="Output directory for scraped documents (default: data/raw)",
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Max requests per second per domain (default: 1.0)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Override max pages per source",
    )
    parser.add_argument(
        "--no-follow",
        action="store_true",
        help="Don't follow internal links (only scrape seed URLs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be scraped without actually scraping",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if not args.all and not args.sources:
        parser.error("Specify --all or --sources <source1> <source2> ...")

    return args


def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    sources = list(AVAILABLE_SOURCES.keys()) if args.all else args.sources

    # Dry run mode
    if args.dry_run:
        print("\n=== DRY RUN — Would scrape the following sources ===\n")
        for name in sources:
            info = AVAILABLE_SOURCES[name]
            max_p = args.max_pages or info["default_max_pages"] or "N/A"
            print(f"  {name:25s} | max_pages: {max_p:>5} | {info['description']}")
            scraper_cls = info["class"]
            if hasattr(scraper_cls, "SEED_URLS"):
                for url in scraper_cls.SEED_URLS[:5]:
                    print(f"    → {url}")
                remaining = len(scraper_cls.SEED_URLS) - 5
                if remaining > 0:
                    print(f"    → ... and {remaining} more seed URLs")
        print(f"\n  Output directory: {args.output}")
        print(f"  Rate limit: {args.rate_limit} req/s per domain")
        print(f"  Follow links: {not args.no_follow}\n")
        return

    # Initialize engine
    engine = ScraperEngine(
        output_dir=args.output,
        rate_limit=args.rate_limit,
    )

    logger.info(f"Starting scrape of {len(sources)} sources → {args.output}")
    logger.info(f"Rate limit: {args.rate_limit} req/s | Follow links: {not args.no_follow}")

    # Run each source scraper
    for name in sources:
        info = AVAILABLE_SOURCES[name]
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping: {name} — {info['description']}")
        logger.info(f"{'='*60}")

        scraper = info["class"](engine)

        try:
            if name == "reddit":
                max_threads = args.max_pages or 5
                scraper.scrape(max_threads_per_query=max_threads)
            else:
                max_pages = args.max_pages or info["default_max_pages"]
                scraper.scrape(
                    max_pages=max_pages,
                    follow_links=not args.no_follow,
                )
        except KeyboardInterrupt:
            logger.warning(f"Interrupted while scraping {name}. Saving progress...")
            break
        except Exception as e:
            logger.error(f"Error scraping {name}: {e}", exc_info=True)
            continue

    # Save results
    engine.save_documents()
    engine.save_pdf_manifest()
    engine.save_index()

    # Print summary
    total_pdf_mb = sum(p["size_kb"] for p in engine.downloaded_pdfs) / 1024
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)
    print(f"  Total documents: {len(engine.documents)}")
    print(f"  Total words:     {sum(d.word_count for d in engine.documents):,}")
    print(f"  Duplicates:      {len(engine.seen_hashes) - len(engine.documents)}")
    print(f"  PDFs downloaded: {len(engine.downloaded_pdfs)} ({total_pdf_mb:.1f} MB)")

    # Per-source breakdown
    source_counts: dict[str, int] = {}
    for doc in engine.documents:
        source_counts[doc.source] = source_counts.get(doc.source, 0) + 1
    print("\n  By source:")
    for source, count in sorted(source_counts.items()):
        print(f"    {source:25s} {count:>4} docs")

    # Per-category breakdown
    cat_counts: dict[str, int] = {}
    for doc in engine.documents:
        cat_counts[doc.category] = cat_counts.get(doc.category, 0) + 1
    print("\n  By category:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat:25s} {count:>4} docs")

    print(f"\n  Output: {args.output}/documents.jsonl")
    print(f"  PDFs:   {args.output}/pdfs/")
    print(f"  Index:  {args.output}/scrape_index.json")
    print(f"  HTML:   {args.output}/html/")
    print()


if __name__ == "__main__":
    main()