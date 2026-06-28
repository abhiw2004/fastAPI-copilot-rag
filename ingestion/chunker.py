from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from bs4 import BeautifulSoup, NavigableString, Tag

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CLEAN_DIR   = Path("corpus_clean")
CORPUS_META = Path("corpus") / "metadata.json"
OUTPUT_DIR  = Path("chunks")

HEADING_MIN_CHARS = 100
HEADING_MAX_CHARS = 4000
FIXED_CHUNK_SIZE  = 800
FIXED_OVERLAP     = 150
FIXED_MIN_CHARS   = 80

HEADING_TAGS: list[str] = ["h1", "h2", "h3", "h4"]

Strategy = Literal["heading", "fixed"]


@dataclass
class Chunk:
    chunk_id:    str
    strategy:    Strategy
    text:        str
    source_file: str
    source_url:  str
    doc_type:    str
    access_level: str
    last_updated: str
    scraped_on:   str

    section:       str = ""
    section_path:  str = ""
    heading_level: int = 0

    chunk_index:      int = 0
    chunk_start_char: int = 0
    chunk_end_char:   int = 0

    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)


def _load_metadata() -> dict[str, dict]:
    return json.loads(CORPUS_META.read_text(encoding="utf-8"))


def _rel_key(clean_path: Path) -> str:
    return str(clean_path.relative_to(CLEAN_DIR))


def _make_id(strategy: str, source_file: str, suffix: str) -> str:
    slug = re.sub(r"[\\/:.]", "_", source_file)
    raw  = f"{strategy}__{slug}__{suffix}"
    h    = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{strategy}__{slug[:60]}__{suffix}__{h}"


def _html_to_text(element) -> str:
    for code in element.find_all(["code", "pre"]):
        code.replace_with(f"\n{code.get_text()}\n")

    lines = []
    for chunk in element.strings:
        text = str(chunk)
        if text.strip():
            lines.append(text)
        else:
            if lines and lines[-1] != "":
                lines.append("")

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _split_oversized(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    parts: list[str] = []
    start = 0
    while start < len(text):
        end      = start + max_chars
        boundary = text.rfind(". ", start, end)
        if boundary == -1 or boundary <= start:
            boundary = text.rfind("\n", start, end)
        if boundary == -1 or boundary <= start:
            boundary = end
        else:
            boundary += 1

        parts.append(text[start:boundary].strip())
        start = boundary - overlap if boundary - overlap > start else boundary

    return [p for p in parts if p]


class HeadingChunker:
    """Splits a cleaned HTML document into chunks at h1–h4 boundaries."""

    def __init__(
        self,
        min_chars: int = HEADING_MIN_CHARS,
        max_chars: int = HEADING_MAX_CHARS,
        overlap:   int = FIXED_OVERLAP,
    ) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.overlap   = overlap

    def chunk_file(self, html_path: Path, source_file: str, meta: dict) -> list[Chunk]:
        soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        root = soup.find("article") or soup.find("body") or soup

        chunks: list[Chunk] = []
        for section_path, heading_level, section_title, content_els in self._extract_sections(root):
            container = BeautifulSoup("<div></div>", "html.parser").div
            for el in content_els:
                container.append(el.__copy__() if hasattr(el, "__copy__") else el)
            text = _html_to_text(container)

            if not text or len(text) < self.min_chars:
                continue

            for sub_idx, sub_text in enumerate(_split_oversized(text, self.max_chars, self.overlap)):
                if len(sub_text) < self.min_chars:
                    continue
                chunks.append(Chunk(
                    chunk_id=_make_id("heading", source_file,
                                      f"h{heading_level}_{len(chunks):04d}_{sub_idx:03d}"),
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
                ))

        return chunks

    def _extract_sections(self, root) -> list[tuple[str, int, str, list]]:
        heading_stack: list[str] = [""] * len(HEADING_TAGS)
        sections: list[tuple[str, int, str, list]] = []
        current_level = 0
        current_title = "__intro__"
        current_path  = ""
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
                if current_els:
                    sections.append((current_path, current_level, current_title, current_els))

                level = HEADING_TAGS.index(tag_name) + 1
                heading_stack[level - 1] = child.get_text(strip=True)
                for i in range(level, len(heading_stack)):
                    heading_stack[i] = ""

                current_path  = " > ".join(h for h in heading_stack[:level] if h)
                current_level = level
                current_title = heading_stack[level - 1]
                current_els   = []
            else:
                current_els.append(child)

        if current_els:
            sections.append((current_path, current_level, current_title, current_els))

        return sections


class FixedSizeChunker:
    """Splits document plain-text into fixed-size overlapping windows."""

    def __init__(
        self,
        chunk_size: int = FIXED_CHUNK_SIZE,
        overlap:    int = FIXED_OVERLAP,
        min_chars:  int = FIXED_MIN_CHARS,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap    = overlap
        self.min_chars  = min_chars

    def chunk_file(self, html_path: Path, source_file: str, meta: dict) -> list[Chunk]:
        soup      = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
        root      = soup.find("article") or soup.find("body") or soup
        full_text = _html_to_text(root)

        chunks: list[Chunk] = []
        for idx, (start, end, text) in enumerate(self._sliding_windows(full_text)):
            if len(text) < self.min_chars:
                continue
            chunks.append(Chunk(
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
            ))

        return chunks

    def _sliding_windows(self, text: str) -> list[tuple[int, int, str]]:
        windows: list[tuple[int, int, str]] = []
        n = len(text)
        start = 0

        while start < n:
            target_end = start + self.chunk_size

            if target_end >= n:
                windows.append((start, n, text[start:].strip()))
                break

            boundary = text.rfind("\n\n", start, target_end)
            if boundary == -1 or boundary - start < self.chunk_size // 2:
                boundary = text.rfind(". ", start, target_end)
            if boundary == -1 or boundary - start < self.chunk_size // 4:
                boundary = text.rfind("\n", start, target_end)
            if boundary == -1 or boundary <= start:
                boundary = target_end

            chunk_text = text[start : boundary + 1].strip()
            if chunk_text:
                windows.append((start, boundary + 1, chunk_text))

            next_start = boundary + 1 - self.overlap
            start = next_start if next_start > start else boundary + 1

        return windows


def build_chunk_store(
    clean_dir:         Path = CLEAN_DIR,
    output_dir:        Path = OUTPUT_DIR,
    metadata:          dict | None = None,
    heading_min_chars: int = HEADING_MIN_CHARS,
    heading_max_chars: int = HEADING_MAX_CHARS,
    fixed_chunk_size:  int = FIXED_CHUNK_SIZE,
    fixed_overlap:     int = FIXED_OVERLAP,
) -> dict:
    """Run both chunkers over every file in clean_dir and write JSONL output."""
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
    fixed_chunks:   list[Chunk] = []

    html_files = sorted(clean_dir.rglob("*.html"))
    print(f"Found {len(html_files)} cleaned HTML files.")

    for html_path in html_files:
        rel_key     = _rel_key(html_path)
        rel_key_fwd = rel_key.replace("\\", "/")
        meta        = metadata.get(rel_key) or metadata.get(rel_key_fwd) or {}

        if not meta:
            print(f"  [WARN] No metadata for {rel_key} — using defaults.")

        try:
            h_chunks = heading_chunker.chunk_file(html_path, rel_key, meta)
            heading_chunks.extend(h_chunks)
            print(f"  heading: {html_path.name:<50s}  -> {len(h_chunks):>3d} chunks")
        except Exception as exc:
            print(f"  [ERROR] heading chunker on {html_path}: {exc}")

        try:
            f_chunks = fixed_chunker.chunk_file(html_path, rel_key, meta)
            fixed_chunks.extend(f_chunks)
            print(f"  fixed:   {html_path.name:<50s}  -> {len(f_chunks):>3d} chunks")
        except Exception as exc:
            print(f"  [ERROR] fixed chunker on {html_path}: {exc}")

    _write_jsonl(output_dir / "chunks_heading.jsonl", heading_chunks)
    _write_jsonl(output_dir / "chunks_fixed.jsonl",   fixed_chunks)
    _write_jsonl(output_dir / "chunks_all.jsonl",     heading_chunks + fixed_chunks)

    stats = _compute_stats(heading_chunks, fixed_chunks)
    (output_dir / "chunk_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    _print_summary(stats)
    print(f"\nDone. Chunks written to {output_dir}/")
    return stats


def _write_jsonl(path: Path, chunks: list[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    print(f"  wrote {len(chunks):>5d} chunks -> {path}")


def _compute_stats(heading_chunks: list[Chunk], fixed_chunks: list[Chunk]) -> dict:
    def _agg(chunks: list[Chunk]) -> dict:
        if not chunks:
            return {"total": 0}
        char_counts  = [c.char_count for c in chunks]
        by_doc_type: dict[str, int] = {}
        for c in chunks:
            by_doc_type[c.doc_type] = by_doc_type.get(c.doc_type, 0) + 1
        return {
            "total":       len(chunks),
            "avg_chars":   round(sum(char_counts) / len(char_counts), 1),
            "min_chars":   min(char_counts),
            "max_chars":   max(char_counts),
            "by_doc_type": dict(sorted(by_doc_type.items())),
        }

    return {
        "heading":       _agg(heading_chunks),
        "fixed":         _agg(fixed_chunks),
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Chunk the FastAPI corpus into JSONL for RAG retrieval."
    )
    parser.add_argument("--clean-dir",          default=str(CLEAN_DIR))
    parser.add_argument("--output-dir",         default=str(OUTPUT_DIR))
    parser.add_argument("--heading-min-chars",  type=int, default=HEADING_MIN_CHARS)
    parser.add_argument("--heading-max-chars",  type=int, default=HEADING_MAX_CHARS)
    parser.add_argument("--fixed-chunk-size",   type=int, default=FIXED_CHUNK_SIZE)
    parser.add_argument("--fixed-overlap",      type=int, default=FIXED_OVERLAP)
    args = parser.parse_args()

    build_chunk_store(
        clean_dir=Path(args.clean_dir),
        output_dir=Path(args.output_dir),
        heading_min_chars=args.heading_min_chars,
        heading_max_chars=args.heading_max_chars,
        fixed_chunk_size=args.fixed_chunk_size,
        fixed_overlap=args.fixed_overlap,
    )
