"""
indexer.py  --  Build the Qdrant vector index and BM25 index from a chunk JSONL.

Outputs
-------
  indexes/
    qdrant/        Qdrant on-disk collection  (cosine, 384-dim)
    bm25.pkl       BM25Okapi + id list
    metadata.json  Shared metadata keyed by chunk_id

Usage
-----
  python indexer.py
  python indexer.py --chunks chunks/chunks_heading.jsonl
  python indexer.py --reset
  python indexer.py --bm25-only | --vector-only
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

CHUNKS_PATH = Path("chunks/chunks_all.jsonl")
INDEX_DIR   = Path("indexes")
QDRANT_DIR  = INDEX_DIR / "qdrant"
BM25_PATH   = INDEX_DIR / "bm25.pkl"
META_PATH   = INDEX_DIR / "metadata.json"

COLLECTION_NAME  = "fastapi_docs"
EMBED_MODEL      = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE      = 384
EMBED_BATCH_SIZE = 128
UPSERT_BATCH     = 256


def load_chunks(path: Path) -> list[dict]:
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks):,} chunks from {path}")
    return chunks


def build_metadata_store(chunks: list[dict]) -> dict[str, dict]:
    store: dict[str, dict] = {}
    for c in chunks:
        store[c["chunk_id"]] = {
            "source_file":      c["source_file"],
            "source_url":       c["source_url"],
            "doc_type":         c["doc_type"],
            "access_level":     c["access_level"],
            "last_updated":     c["last_updated"],
            "scraped_on":       c["scraped_on"],
            "strategy":         c["strategy"],
            "section":          c.get("section", ""),
            "section_path":     c.get("section_path", ""),
            "heading_level":    c.get("heading_level", 0),
            "chunk_index":      c.get("chunk_index", 0),
            "chunk_start_char": c.get("chunk_start_char", 0),
            "chunk_end_char":   c.get("chunk_end_char", 0),
            "char_count":       c["char_count"],
        }
    return store


def build_vector_index(chunks: list[dict], qdrant_dir: Path, reset: bool = False) -> None:
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print(f"\n[ERROR] Missing dependency: {exc}")
        print("Run:  pip install qdrant-client sentence-transformers")
        sys.exit(1)

    print("\nVector index")

    qdrant_dir.mkdir(parents=True, exist_ok=True)
    client   = QdrantClient(path=str(qdrant_dir))
    existing = [c.name for c in client.get_collections().collections]

    if reset and COLLECTION_NAME in existing:
        print(f"  Deleting collection '{COLLECTION_NAME}'")
        client.delete_collection(COLLECTION_NAME)
        existing = []

    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        print(f"  Created collection '{COLLECTION_NAME}'")
    else:
        print(f"  Reusing collection '{COLLECTION_NAME}'")

    print(f"  Loading {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    texts     = [c["text"]     for c in chunks]
    ids       = [c["chunk_id"] for c in chunks]
    total     = len(chunks)
    id_to_int = {cid: i for i, cid in enumerate(ids)}

    print(f"  Embedding {total:,} chunks ...")
    all_vectors: list[list[float]] = []
    for start in range(0, total, EMBED_BATCH_SIZE):
        vecs = model.encode(
            texts[start : start + EMBED_BATCH_SIZE],
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        all_vectors.extend(vecs.tolist())
        done = min(start + EMBED_BATCH_SIZE, total)
        print(f"    {done:>5,}/{total:,}  ({done/total*100:.1f}%)", end="\r")
    print()

    print("  Upserting ...")
    for start in range(0, total, UPSERT_BATCH):
        sl     = slice(start, start + UPSERT_BATCH)
        points = [
            PointStruct(
                id=id_to_int[cid],
                vector=vec,
                payload={
                    "chunk_id":     cid,
                    "text":         texts[i],
                    "source_url":   chunks[i]["source_url"],
                    "doc_type":     chunks[i]["doc_type"],
                    "strategy":     chunks[i]["strategy"],
                    "section":      chunks[i].get("section", ""),
                    "section_path": chunks[i].get("section_path", ""),
                    "char_count":   chunks[i]["char_count"],
                },
            )
            for i, (cid, vec) in enumerate(zip(ids[sl], all_vectors[sl]), start=start)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points, wait=True)
        done = min(start + UPSERT_BATCH, total)
        print(f"    {done:>5,}/{total:,}", end="\r")
    print()

    count = client.get_collection(COLLECTION_NAME).points_count
    print(f"  {count:,} vectors -> {qdrant_dir}")


def _tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_bm25_index(chunks: list[dict], bm25_path: Path) -> None:
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("\n[ERROR] rank-bm25 not installed.  Run:  pip install rank-bm25")
        sys.exit(1)

    print("\nBM25 index")
    print(f"  Tokenising {len(chunks):,} chunks ...")

    ids    = [c["chunk_id"] for c in chunks]
    corpus = [_tokenise(c["text"]) for c in chunks]

    print("  Building index ...")
    index = BM25Okapi(corpus)

    bm25_path.parent.mkdir(parents=True, exist_ok=True)
    with bm25_path.open("wb") as fh:
        pickle.dump({"index": index, "ids": ids, "corpus": corpus}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)

    size_kb = bm25_path.stat().st_size / 1024
    print(f"  {size_kb:.0f} KB -> {bm25_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Qdrant vector + BM25 indexes."
    )
    parser.add_argument("--chunks",      default=str(CHUNKS_PATH))
    parser.add_argument("--index-dir",   default=str(INDEX_DIR))
    parser.add_argument("--reset",       action="store_true")
    parser.add_argument("--bm25-only",   action="store_true")
    parser.add_argument("--vector-only", action="store_true")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    index_dir   = Path(args.index_dir)
    meta_path   = index_dir / "metadata.json"

    if not chunks_path.exists():
        print(f"[ERROR] Chunk file not found: {chunks_path}")
        sys.exit(1)

    chunks = load_chunks(chunks_path)

    print("\nMetadata store")
    index_dir.mkdir(parents=True, exist_ok=True)
    meta_store = build_metadata_store(chunks)
    meta_path.write_text(
        json.dumps(meta_store, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  {len(meta_store):,} records -> {meta_path}")

    if not args.bm25_only:
        build_vector_index(chunks, index_dir / "qdrant", reset=args.reset)

    if not args.vector_only:
        build_bm25_index(chunks, index_dir / "bm25.pkl")

    print("\n" + "=" * 58)
    print("INDEX BUILD COMPLETE")
    print("=" * 58)
    print(f"  chunks   : {len(chunks):,}")
    print(f"  vectors  : {index_dir / 'qdrant'}")
    print(f"  bm25     : {index_dir / 'bm25.pkl'}")
    print(f"  metadata : {meta_path}")
    print("=" * 58)


if __name__ == "__main__":
    main()
