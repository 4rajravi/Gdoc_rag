"""
Hierarchical Chunking for German Bureaucracy RAG
=================================================

Three levels of chunks, each serving a different retrieval purpose:

L1 (Document Summary)
    One per document. Title + opening paragraph + category.
    Hit when user asks broad questions: "How does health insurance work?"

L2 (Section Chunks)  
    One per markdown ## section. Primary retrieval layer.
    Hit for most queries: "What documents do I need for Anmeldung?"

L3 (Fine-grained)
    Sub-splits of large L2 sections (~300 tokens, 50-token overlap).
    Hit for very specific queries: "Income threshold for private insurance?"

Design decisions:
- We preserve section headers in chunk text (helps embedding model understand topic)
- Each chunk carries rich metadata (doc_title, section, source, category, url)
- L3 chunks carry a `parent_chunk_id` linking back to their L2 parent
- This parent link enables "drill-down": retrieve L2 for context, L3 for detail
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk at any level of the hierarchy."""

    chunk_id: str
    text: str
    level: str  # "L1", "L2", "L3"

    # Metadata for filtering and context assembly
    doc_url: str
    doc_title: str
    source: str  # "make-it-in-germany", "allaboutberlin", etc.
    category: str  # "anmeldung", "visa", "tax", etc.
    language: str

    section_header: str = ""  # the ## header this chunk belongs to
    parent_chunk_id: str = ""  # L3 → its L2 parent
    token_count: int = 0

    def __post_init__(self):
        self.token_count = len(self.text.split())  # rough word-based estimate


def generate_chunk_id(level: str, doc_url: str, section: str, index: int = 0) -> str:
    """Generate a deterministic chunk ID."""
    raw = f"{level}:{doc_url}:{section}:{index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def split_into_sections(text: str) -> list[tuple[str, str]]:
    """
    Split markdown text by ## headers.
    Returns list of (header, body) tuples.

    If text starts without a header, the first tuple has header="introduction".
    """
    # Pattern: split on lines starting with ## (but not ### which is sub-section)
    # We keep ## and ### together in the same section
    parts = re.split(r'\n(?=## (?!#))', text)

    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        # Extract header if present
        lines = part.split('\n', 1)
        first_line = lines[0].strip()

        if first_line.startswith('## '):
            header = first_line.lstrip('# ').strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        elif first_line.startswith('# ') and not first_line.startswith('## '):
            header = first_line.lstrip('# ').strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        else:
            header = "introduction"
            body = part

        if body:  # skip empty sections
            sections.append((header, body))

    return sections


def split_tokens(text: str, max_tokens: int = 300, overlap: int = 50) -> list[str]:
    """
    Split text into sub-chunks of ~max_tokens words with overlap.

    Uses sentence boundaries where possible to avoid cutting mid-sentence.
    """
    # Split into sentences (rough but effective for this content)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk: list[str] = []
    current_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())

        if current_count + word_count > max_tokens and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))

            # Keep overlap: take last N tokens worth of sentences
            overlap_chunk: list[str] = []
            overlap_count = 0
            for s in reversed(current_chunk):
                s_count = len(s.split())
                if overlap_count + s_count > overlap:
                    break
                overlap_chunk.insert(0, s)
                overlap_count += s_count

            current_chunk = overlap_chunk
            current_count = overlap_count

        current_chunk.append(sentence)
        current_count += word_count

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


class HierarchicalChunker:
    """
    Creates L1, L2, L3 chunks from scraped documents.

    Usage:
        chunker = HierarchicalChunker()
        chunker.process_documents("data/raw/documents.jsonl")
        chunker.save("data/processed/")
        chunker.print_stats()
    """

    def __init__(
        self,
        l3_max_tokens: int = 300,
        l3_overlap: int = 50,
        l2_split_threshold: int = 500,  # L2 sections above this get L3 sub-splits
        l1_summary_tokens: int = 200,  # max tokens for L1 summary
    ):
        self.l3_max_tokens = l3_max_tokens
        self.l3_overlap = l3_overlap
        self.l2_split_threshold = l2_split_threshold
        self.l1_summary_tokens = l1_summary_tokens

        self.chunks: dict[str, list[Chunk]] = {"L1": [], "L2": [], "L3": []}

    def process_documents(self, jsonl_path: str):
        """Process all documents from a JSONL file."""
        path = Path(jsonl_path)
        if not path.exists():
            raise FileNotFoundError(f"Document file not found: {jsonl_path}")

        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                    self._process_single_doc(doc)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {line_num}: {e}")

        logger.info(
            f"Chunking complete: "
            f"L1={len(self.chunks['L1'])}, "
            f"L2={len(self.chunks['L2'])}, "
            f"L3={len(self.chunks['L3'])}"
        )

    def _process_single_doc(self, doc: dict):
        """Create all chunk levels for a single document."""
        url = doc["url"]
        title = doc["title"]
        content = doc["content"]
        source = doc["source"]
        category = doc["category"]
        language = doc["language"]

        # ─── L1: Document summary ──────────────────────────────────
        # Take the title + first N tokens as a summary
        words = content.split()
        summary_text = ' '.join(words[:self.l1_summary_tokens])
        l1_text = f"{title}\n\n{summary_text}"

        l1_chunk = Chunk(
            chunk_id=generate_chunk_id("L1", url, "summary"),
            text=l1_text,
            level="L1",
            doc_url=url,
            doc_title=title,
            source=source,
            category=category,
            language=language,
            section_header="document_summary",
        )
        self.chunks["L1"].append(l1_chunk)

        # ─── L2: Section chunks ────────────────────────────────────
        sections = split_into_sections(content)

        if not sections:
            # Document has no headers — treat entire content as one section
            sections = [("full_content", content)]

        for sec_idx, (header, body) in enumerate(sections):
            # Prepend header to body so the embedding model sees the topic
            l2_text = f"{title} — {header}\n\n{body}"

            l2_id = generate_chunk_id("L2", url, header, sec_idx)

            l2_chunk = Chunk(
                chunk_id=l2_id,
                text=l2_text,
                level="L2",
                doc_url=url,
                doc_title=title,
                source=source,
                category=category,
                language=language,
                section_header=header,
            )
            self.chunks["L2"].append(l2_chunk)

            # ─── L3: Fine-grained sub-splits ───────────────────────
            body_tokens = len(body.split())
            if body_tokens > self.l2_split_threshold:
                sub_chunks = split_tokens(
                    body,
                    max_tokens=self.l3_max_tokens,
                    overlap=self.l3_overlap,
                )

                for sub_idx, sub_text in enumerate(sub_chunks):
                    # Prepend header for context
                    l3_text = f"{title} — {header}\n\n{sub_text}"

                    l3_chunk = Chunk(
                        chunk_id=generate_chunk_id("L3", url, header, sub_idx),
                        text=l3_text,
                        level="L3",
                        doc_url=url,
                        doc_title=title,
                        source=source,
                        category=category,
                        language=language,
                        section_header=header,
                        parent_chunk_id=l2_id,
                    )
                    self.chunks["L3"].append(l3_chunk)

    def save(self, output_dir: str = "data/processed"):
        """Save chunks as JSONL files, one per level."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for level, chunks in self.chunks.items():
            filepath = out / f"chunks_{level.lower()}.jsonl"
            with open(filepath, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(chunks)} {level} chunks to {filepath}")

    def print_stats(self):
        """Print chunking statistics."""
        print("\n" + "=" * 50)
        print("HIERARCHICAL CHUNKING STATS")
        print("=" * 50)

        for level, chunks in self.chunks.items():
            if not chunks:
                print(f"\n  {level}: 0 chunks")
                continue

            token_counts = [c.token_count for c in chunks]
            avg = sum(token_counts) / len(token_counts)
            min_t = min(token_counts)
            max_t = max(token_counts)

            print(f"\n  {level}: {len(chunks)} chunks")
            print(f"    Avg tokens: {avg:.0f}")
            print(f"    Min/Max:    {min_t} / {max_t}")

            # Category distribution
            cats = {}
            for c in chunks:
                cats[c.category] = cats.get(c.category, 0) + 1
            print(f"    Categories: {dict(sorted(cats.items(), key=lambda x: -x[1]))}")

        total = sum(len(c) for c in self.chunks.values())
        print(f"\n  Total chunks: {total}")
        print()

    def get_all_chunks(self) -> list[Chunk]:
        """Get all chunks across all levels (for batch embedding)."""
        all_chunks = []
        for level_chunks in self.chunks.values():
            all_chunks.extend(level_chunks)
        return all_chunks