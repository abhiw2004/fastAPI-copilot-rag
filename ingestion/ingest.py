"""
ingest.py  --  Full pipeline: clean, chunk, index.

Usage
-----
  python ingest.py --source corpus/ --rebuild
  python ingest.py --source corpus/ --rebuild --dry-run
  python ingest.py --skip-clean --skip-chunk --rebuild
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from pathlib import Path


def _header(text: str) -> None:
    print()
    print("=" * 60)
    print(f"  {text}")
    print("=" * 60)


def _step(text: str) -> None:
    print(f"\n{text}")


def _done(text: str) -> None:
    print(f"  {text}")


def _warn(text: str) -> None:
    print(f"  [WARN] {text}", file=sys.stderr)


def _abort(text: str) -> None:
    print(f"\n[ERROR] {text}", file=sys.stderr)
    sys.exit(1)


def _require_dir(path: Path, label: str) -> None:
    if not path.exists():
        _abort(f"{label} directory not found: {path}")


def run_clean(source_dir: Path, clean_dir: Path, dry_run: bool) -> None:
    _step("Stage 1 -- Clean")

    html_files = sorted(source_dir.rglob("*.html"))
    if not html_files:
        _abort(f"No .html files found under {source_dir}")

    print(f"  source  : {source_dir}  ({len(html_files)} files)")
    print(f"  dest    : {clean_dir}")

    if dry_run:
        print("  [dry-run] skipping.")
        return

    from ingestion.normalise import clean_html

    report: dict = {}
    for html_path in html_files:
        rel      = html_path.relative_to(source_dir)
        out_path = clean_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cleaned, stats = clean_html(html_path.read_text(encoding="utf-8"))
        out_path.write_text(cleaned, encoding="utf-8")
        report[str(rel)] = stats

        flag = ""
        if stats["raw_char_count"] > 0 and stats["kept_ratio"] < 0.05:
            flag = "  [low kept ratio]"
        elif stats["selector_used"] == "fallback_body" and stats["raw_char_count"] > 3000:
            flag = "  [fallback on large page]"
        print(f"    {str(rel):<55s}  {stats['selector_used']:<20s}  {stats['kept_ratio']:.0%}{flag}")

    report_path = Path("cleaning_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _done(f"{len(html_files)} files cleaned -> {clean_dir}/")
    _done(f"report -> {report_path}")


def run_chunk(clean_dir: Path, chunks_dir: Path, dry_run: bool) -> None:
    _step("Stage 2 -- Chunk")

    html_files = list(clean_dir.rglob("*.html"))
    print(f"  source  : {clean_dir}  ({len(html_files)} files)")
    print(f"  dest    : {chunks_dir}")

    if dry_run:
        print("  [dry-run] skipping.")
        return

    from ingestion.chunker import build_chunk_store

    meta_path = Path("corpus") / "metadata.json"
    metadata: dict | None = None
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"  metadata: {meta_path}  ({len(metadata)} entries)")
    else:
        _warn(f"No metadata.json at {meta_path} -- chunk metadata fields will be empty.")

    stats = build_chunk_store(clean_dir=clean_dir, output_dir=chunks_dir, metadata=metadata)
    _done(f"{stats['combined_total']:,} chunks written -> {chunks_dir}/")


def run_index(chunks_dir: Path, index_dir: Path, rebuild: bool, dry_run: bool) -> None:
    _step("Stage 3 -- Index")

    chunks_file = chunks_dir / "chunks_all.jsonl"
    if not chunks_file.exists():
        _abort(f"chunks_all.jsonl not found at {chunks_file}")

    print(f"  source  : {chunks_file}")
    print(f"  dest    : {index_dir}")
    print(f"  rebuild : {rebuild}")

    if dry_run:
        print("  [dry-run] skipping.")
        return

    from ingestion.indexer import build_bm25_index, build_metadata_store, build_vector_index, load_chunks

    chunks = load_chunks(chunks_file)

    index_dir.mkdir(parents=True, exist_ok=True)
    meta_out = index_dir / "metadata.json"
    store    = build_metadata_store(chunks)
    meta_out.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")
    _done(f"{len(store):,} metadata records -> {meta_out}")

    build_vector_index(chunks, index_dir / "qdrant", reset=rebuild)
    build_bm25_index(chunks, index_dir / "bm25.pkl")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="ingest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            Full pipeline: clean, chunk, index.

            Examples
            --------
              python ingest.py --source corpus/ --rebuild
              python ingest.py --source corpus/ --rebuild --dry-run
              python ingest.py --skip-clean --skip-chunk --rebuild
        """),
    )
    p.add_argument("--source",      default="corpus",       metavar="PATH")
    p.add_argument("--clean-dir",   default="corpus_clean", metavar="PATH")
    p.add_argument("--chunks-dir",  default="chunks",       metavar="PATH")
    p.add_argument("--index-dir",   default="indexes",      metavar="PATH")
    p.add_argument("--rebuild",     action="store_true")
    p.add_argument("--skip-clean",  action="store_true")
    p.add_argument("--skip-chunk",  action="store_true")
    p.add_argument("--skip-index",  action="store_true")
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    source_dir = Path(args.source)
    clean_dir  = Path(args.clean_dir)
    chunks_dir = Path(args.chunks_dir)
    index_dir  = Path(args.index_dir)

    if not args.skip_clean and not source_dir.exists():
        _abort(f"Source directory not found: {source_dir}")
    if args.skip_clean and not args.skip_chunk:
        _require_dir(clean_dir, "--clean-dir")

    _header("INGEST PIPELINE" + ("  [dry-run]" if args.dry_run else ""))
    print(f"  source  : {source_dir}")
    print(f"  rebuild : {args.rebuild}")

    skipped: list[str] = []
    t0 = time.monotonic()

    if not args.skip_clean:
        run_clean(source_dir, clean_dir, dry_run=args.dry_run)
    else:
        skipped.append("clean")
        print("\n  [skip] clean")

    if not args.skip_chunk:
        run_chunk(clean_dir, chunks_dir, dry_run=args.dry_run)
    else:
        skipped.append("chunk")
        print("\n  [skip] chunk")

    if not args.skip_index:
        run_index(chunks_dir, index_dir, rebuild=args.rebuild, dry_run=args.dry_run)
    else:
        skipped.append("index")
        print("\n  [skip] index")

    elapsed = time.monotonic() - t0

    _header("INGEST COMPLETE" + ("  [dry-run]" if args.dry_run else ""))
    print(f"  source dir  : {source_dir}")
    print(f"  clean dir   : {clean_dir}")
    print(f"  chunks dir  : {chunks_dir}")
    print(f"  index dir   : {index_dir}")
    if skipped:
        print(f"  skipped     : {', '.join(skipped)}")
    print(f"  elapsed     : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
