from __future__ import annotations

import hashlib
import json
import re
import sys
import textwrap
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Literal

from bs4 import BeautifulSoup, NavigableString, Tag

# Ensure the console can handle Unicode on Windows (cp1252 would choke on arrows)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Paths
CLEAN_DIR = Path("corpus_clean")
CORPUS_META = Path("corpus") / "metadata.json"
OUTPUT_DIR = Path("chunks")

# Configuration — easy to tune without touching strategy logic


# Heading chunker
HEADING_MIN_CHARS = 100        # discard chunks with fewer chars than this
HEADING_MAX_CHARS = 4000       #very large sections are split further

# Fixed-size chunker
FIXED_CHUNK_SIZE = 800         
FIXED_OVERLAP = 150           
FIXED_MIN_CHARS = 80          

# Heading levels to split on
HEADING_TAGS: list[str] = ["h1", "h2", "h3", "h4"]


# Data model

Strategy = Literal["heading", "fixed"]


@dataclass
class Chunk:
    chunk_id: str           # globally unique, deterministic
    strategy: Strategy      # which chunker produced this
    text: str               # plain-text content (no HTML)
    source_file: str        # relative path inside corpus_clean/
    source_url: str         # original URL (from metadata.json)
    doc_type: str           # tutorial / advanced / reference / …
    access_level: str       # public / internal / …
    last_updated: str       # ISO date from metadata
    scraped_on: str         # ISO date from metadata

    # Heading-strategy extras (empty string for fixed strategy)
    section: str = ""              # leaf heading text, e.g. "Dependency Injection"
    section_path: str = ""         # breadcrumb, e.g. "Background Tasks > Dependency Injection"
    heading_level: int = 0         # 1-4; 0 means document-level intro

    # Fixed-strategy extras (0 for heading strategy)
    chunk_index: int = 0           # position within the document
    chunk_start_char: int = 0      # character offset in the full document text
    chunk_end_char: int = 0        # character offset end

    # Common
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


# Shared helpers


def _load_metadata() -> dict[str, dict]:
    """Load corpus/metadata.json keyed by relative path (with backslashes)."""
    return json.loads(CORPUS_META.read_text(encoding="utf-8"))


def _rel_key(clean_path: Path) -> str:
    """
    Convert a corpus_clean path to the key used in metadata.json.

    corpus_clean/tutorial/foo.html  →  tutorial\\foo.html   (Windows key)
    """
    rel = clean_path.relative_to(CLEAN_DIR)
    return str(rel)   # keeps the OS separator — matches metadata.json on Windows


def _make_id(strategy: str, source_file: str, suffix: str) -> str:
    """
    Deterministic chunk_id that survives re-runs.

    Format:  <strategy>__<slug>__<suffix>
    The slug is derived from the source_file path.
    """
    slug = re.sub(r"[\\/:.]", "_", source_file)
    raw = f"{strategy}__{slug}__{suffix}"
    # Keep it readable but bounded: truncate slug, append 8-char hash
    h = hashlib.sha1(raw.encode()).hexdigest()[:8]
    short_slug = slug[:60]
    return f"{strategy}__{short_slug}__{suffix}__{h}"


def _html_to_text(element) -> str:
    """
    Extract readable plain text from a BeautifulSoup element.

    • Preserves code blocks as-is (important for FastAPI examples).
    • Collapses excessive whitespace but keeps paragraph breaks.
    """
    # Replace <code>/<pre> blocks with their text verbatim (no stripping inside)
    for code in element.find_all(["code", "pre"]):
        code.replace_with(f"\n{code.get_text()}\n")

    lines = []
    for chunk in element.strings:
        text = str(chunk)
        if text.strip():
            lines.append(text)
        else:
            # Preserve a single blank line as paragraph break
            if lines and lines[-1] != "":
                lines.append("")

    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
    """
    When a heading section is larger than max_chars, sub-split it with
    the fixed-size algorithm so nothing exceeds the ceiling.
    Returns a list of sub-chunks (may be a single-element list).
    """
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        # Try to break at a sentence boundary
        boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 1  # include the period / newline

        parts.append(text[start:boundary].strip())
        start = boundary - overlap if boundary - overlap > start else boundary

    return [p for p in parts if p]


# Strategy 1 — HeadingChunker
class HeadingChunker:
    """
    Splits a cleaned HTML document into chunks at heading boundaries.

    Algorithm
    ---------
    1. Parse the HTML with BeautifulSoup.
    2. Walk top-level children of the root element.
    3. When a heading tag (h1–h4) is encountered, close the current
       accumulation buffer and start a new one.
    4. Build a breadcrumb path by tracking the most-recent heading at
       each level (like a call stack for headings).
    5. Text before the first heading is collected as level-0 "intro".
    6. Oversized chunks (> HEADING_MAX_CHARS) are sub-split with overlap.
    """

    def __init__(
        self,
        min_chars: int = HEADING_MIN_CHARS,
        max_chars: int = HEADING_MAX_CHARS,
        overlap: int = FIXED_OVERLAP,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap = overlap

    
    # Public API
    

    def chunk_file(
        self,
        html_path: Path,
        source_file: str,
        meta: dict,
    ) -> list[Chunk]:
        raw_html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")

        # The cleaner wraps content in <article> or <body>
        root = soup.find("article") or soup.find("body") or soup

        sections = self._extract_sections(root)
        chunks: list[Chunk] = []

        for section_path, heading_level, section_title, content_el_list in sections:
            # Build text from the accumulated elements
            container = BeautifulSoup("<div></div>", "html.parser").div
            for el in content_el_list:
                container.append(el.__copy__() if hasattr(el, "__copy__") else el)
            text = _html_to_text(container)

            if not text or len(text) < self.min_chars:
                continue

            # Sub-split oversized sections
            sub_texts = _split_oversized(text, self.max_chars, self.overlap)
            for sub_idx, sub_text in enumerate(sub_texts):
                if len(sub_text) < self.min_chars:
                    continue
                suffix = f"h{heading_level}_{sub_idx:03d}"
                chunk = Chunk(
                    chunk_id=_make_id("heading", source_file, suffix),
                    strategy="heading",
                    text=sub_text,
                    source_file=source_file,
                    source_url=meta.get("source", ""),
                    doc_type=meta.get("doc_type", ""),
                    access_level=meta.get("access_level", ""),
                    last_updated=meta.get("last_updated", ""),
                    scraped_on=meta.get("scraped_on", ""),
                    section=section_title,
                    section_path=section_path,
                    heading_level=heading_level,
                    chunk_index=len(chunks),
                )
                chunks.append(chunk)

        return chunks

    
    # Internal helpers
   

    def _extract_sections(
        self, root
    ) -> list[tuple[str, int, str, list]]:
        """
        Returns a list of (section_path, heading_level, section_title, elements).

        Elements are the BeautifulSoup nodes that belong to that section.
        """
        # heading_stack[i] = title of most-recent heading at level (i+1)
        heading_stack: list[str] = [""] * len(HEADING_TAGS)

        sections: list[tuple[str, int, str, list]] = []
        current_level = 0
        current_title = "__intro__"
        current_path = ""
        current_els: list = []

        for child in root.children:
            if isinstance(child, NavigableString):
                if str(child).strip():
                    current_els.append(child)
                continue

            if not isinstance(child, Tag):
                continue

            tag_name = child.name.lower() if child.name else ""

            if tag_name in HEADING_TAGS:
                # Flush current buffer
                if current_els:
                    sections.append(
                        (current_path, current_level, current_title, current_els)
                    )

                # Update heading stack
                level = HEADING_TAGS.index(tag_name) + 1  # 1-indexed
                heading_stack[level - 1] = child.get_text(strip=True)
                # Clear deeper levels
                for i in range(level, len(heading_stack)):
                    heading_stack[i] = ""

                # Build breadcrumb path from non-empty stack entries
                current_path = " > ".join(h for h in heading_stack[:level] if h)
                current_level = level
                current_title = heading_stack[level - 1]
                current_els = []
            else:
                current_els.append(child)

        # Flush last buffer
        if current_els:
            sections.append(
                (current_path, current_level, current_title, current_els)
            )

        return sections

# Strategy 2 — FixedSizeChunker

class FixedSizeChunker:
    """
    Splits the plain-text content of a document into fixed-size windows
    with a configurable character overlap.

    Why characters and not tokens?
    --------------------------------
    No tokeniser dependency means this runs on any machine without a model
    download. The overhead per-chunk is small. You can convert to token-based
    splitting later once you settle on an embedding model.

    Overlap strategy
    ----------------
    Each chunk starts FIXED_OVERLAP characters before the end of the
    previous chunk. The overlap text is included in BOTH chunks so the
    retriever never loses context at a boundary.
    """

    def __init__(
        self,
        chunk_size: int = FIXED_CHUNK_SIZE,
        overlap: int = FIXED_OVERLAP,
        min_chars: int = FIXED_MIN_CHARS,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chars = min_chars

   
    # Public API
    

    def chunk_file(
        self,
        html_path: Path,
        source_file: str,
        meta: dict,
    ) -> list[Chunk]:
        raw_html = html_path.read_text(encoding="utf-8")
        soup = BeautifulSoup(raw_html, "html.parser")
        root = soup.find("article") or soup.find("body") or soup
        full_text = _html_to_text(root)

        windows = self._sliding_windows(full_text)
        chunks: list[Chunk] = []

        for idx, (start, end, text) in enumerate(windows):
            if len(text) < self.min_chars:
                continue
            chunk = Chunk(
                chunk_id=_make_id("fixed", source_file, f"{idx:04d}"),
                strategy="fixed",
                text=text,
                source_file=source_file,
                source_url=meta.get("source", ""),
                doc_type=meta.get("doc_type", ""),
                access_level=meta.get("access_level", ""),
                last_updated=meta.get("last_updated", ""),
                scraped_on=meta.get("scraped_on", ""),
                chunk_index=idx,
                chunk_start_char=start,
                chunk_end_char=end,
            )
            chunks.append(chunk)

        return chunks

    
    # Internal helpers
    

    def _sliding_windows(
        self, text: str
    ) -> list[tuple[int, int, str]]:
        """
        Yield (start, end, window_text) tuples.

        Attempts to break at sentence / paragraph boundaries within
        ±10 % of the target size to avoid mid-sentence cuts.
        """
        windows: list[tuple[int, int, str]] = []
        n = len(text)
        start = 0

        while start < n:
            target_end = start + self.chunk_size

            if target_end >= n:
                windows.append((start, n, text[start:].strip()))
                break

            # Prefer breaking at a paragraph boundary
            boundary = text.rfind("\n\n", start, target_end)
            if boundary == -1 or boundary - start < self.chunk_size // 2:
                # Fall back to sentence boundary
                boundary = text.rfind(". ", start, target_end)
            if boundary == -1 or boundary - start < self.chunk_size // 4:
                # Fall back to any newline
                boundary = text.rfind("\n", start, target_end)
            if boundary == -1 or boundary <= start:
                boundary = target_end  # hard cut

            chunk_text = text[start : boundary + 1].strip()
            if chunk_text:
                windows.append((start, boundary + 1, chunk_text))

            # Next window starts overlap chars before the boundary
            next_start = boundary + 1 - self.overlap
            start = next_start if next_start > start else boundary + 1

        return windows



# Runner — processes all corpus_clean files with both strategies


def build_chunk_store(
    clean_dir: Path = CLEAN_DIR,
    output_dir: Path = OUTPUT_DIR,
    metadata: dict | None = None,
    heading_min_chars: int = HEADING_MIN_CHARS,
    heading_max_chars: int = HEADING_MAX_CHARS,
    fixed_chunk_size: int = FIXED_CHUNK_SIZE,
    fixed_overlap: int = FIXED_OVERLAP,
) -> dict:
    """
    Run both chunkers over every file in corpus_clean/ and write:
      chunks_heading.jsonl, chunks_fixed.jsonl, chunks_all.jsonl, chunk_stats.json

    Returns the stats dict.
    """
    if metadata is None:
        metadata = _load_metadata()

    output_dir.mkdir(parents=True, exist_ok=True)

    heading_chunker = HeadingChunker(
        min_chars=heading_min_chars,
        max_chars=heading_max_chars,
        overlap=fixed_overlap,
    )
    fixed_chunker = FixedSizeChunker(
        chunk_size=fixed_chunk_size,
        overlap=fixed_overlap,
    )

    heading_chunks: list[Chunk] = []
    fixed_chunks: list[Chunk] = []

    # Walk every HTML file in corpus_clean/
    html_files = sorted(clean_dir.rglob("*.html"))
    print(f"Found {len(html_files)} cleaned HTML files.")

    for html_path in html_files:
        # Build the metadata key — try both OS-native and forward-slash forms
        rel_key = _rel_key(html_path)
        rel_key_fwd = rel_key.replace("\\", "/")

        meta = metadata.get(rel_key) or metadata.get(rel_key_fwd) or {}
        if not meta:
            print(f"  [WARN] No metadata for {rel_key} — using defaults.")

        source_file = rel_key

        # --- Heading strategy ---
        try:
            h_chunks = heading_chunker.chunk_file(html_path, source_file, meta)
            heading_chunks.extend(h_chunks)
            print(
                f"  heading: {html_path.name:<50s}  -> {len(h_chunks):>3d} chunks"
            )
        except Exception as exc:
            print(f"  [ERROR] heading chunker on {html_path}: {exc}")

        # Fixed strategy
        try:
            f_chunks = fixed_chunker.chunk_file(html_path, source_file, meta)
            fixed_chunks.extend(f_chunks)
            print(
                f"  fixed:   {html_path.name:<50s}  -> {len(f_chunks):>3d} chunks"
            )
        except Exception as exc:
            print(f"  [ERROR] fixed chunker on {html_path}: {exc}")

    # Write strategy-wise JSONL files
    _write_jsonl(output_dir / "chunks_heading.jsonl", heading_chunks)
    _write_jsonl(output_dir / "chunks_fixed.jsonl", fixed_chunks)

    # Merged file (heading first, then fixed;  chunk_ids are unique)
    _write_jsonl(output_dir / "chunks_all.jsonl", heading_chunks + fixed_chunks)

    # Statistics
    stats = _compute_stats(heading_chunks, fixed_chunks)
    (output_dir / "chunk_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    _print_summary(stats)
    print(f"\nDone. Chunks written to {output_dir}/")
    return stats



# I/O helpers


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"  wrote {len(chunks):>5d} chunks -> {path}")


def _compute_stats(
    heading_chunks: list[Chunk],
    fixed_chunks: list[Chunk],
) -> dict:
    def _agg(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"total": 0}
        char_counts = [c.char_count for c in chunks]
        by_doc_type: dict[str, int] = {}
        for c in chunks:
            by_doc_type[c.doc_type] = by_doc_type.get(c.doc_type, 0) + 1
        return {
            "total": len(chunks),
            "avg_chars": round(sum(char_counts) / len(char_counts), 1),
            "min_chars": min(char_counts),
            "max_chars": max(char_counts),
            "by_doc_type": dict(sorted(by_doc_type.items())),
        }

    return {
        "heading": _agg(heading_chunks),
        "fixed": _agg(fixed_chunks),
        "combined_total": len(heading_chunks) + len(fixed_chunks),
    }


def _print_summary(stats: dict) -> None:
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    for strategy in ("heading", "fixed"):
        s = stats[strategy]
        print(f"\n  Strategy: {strategy.upper()}")
        print(f"    Total chunks : {s['total']}")
        if s["total"] > 0:
            print(f"    Avg chars    : {s['avg_chars']}")
            print(f"    Min / Max    : {s['min_chars']} / {s['max_chars']}")
            print("    By doc type  :")
            for dt, count in s["by_doc_type"].items():
                print(f"      {dt:<20s}: {count}")
    print(f"\n  Combined total : {stats['combined_total']}")
    print("=" * 60)



# CLI entry point


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk the FastAPI corpus into JSONL for RAG retrieval."
    )
    parser.add_argument(
        "--clean-dir",
        default=str(CLEAN_DIR),
        help="Path to corpus_clean/ directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory to write chunk JSONL files (default: %(default)s)",
    )
    parser.add_argument(
        "--heading-min-chars",
        type=int,
        default=HEADING_MIN_CHARS,
        help="Min chars for a heading chunk to be kept (default: %(default)s)",
    )
    parser.add_argument(
        "--heading-max-chars",
        type=int,
        default=HEADING_MAX_CHARS,
        help="Soft max chars per heading chunk before sub-splitting (default: %(default)s)",
    )
    parser.add_argument(
        "--fixed-chunk-size",
        type=int,
        default=FIXED_CHUNK_SIZE,
        help="Fixed-size window in characters (default: %(default)s)",
    )
    parser.add_argument(
        "--fixed-overlap",
        type=int,
        default=FIXED_OVERLAP,
        help="Overlap in characters for fixed-size strategy (default: %(default)s)",
    )
    args = parser.parse_args()

    build_chunk_store(
        clean_dir=Path(args.clean_dir),
        output_dir=Path(args.output_dir),
        heading_min_chars=args.heading_min_chars,
        heading_max_chars=args.heading_max_chars,
        fixed_chunk_size=args.fixed_chunk_size,
        fixed_overlap=args.fixed_overlap,
    )
