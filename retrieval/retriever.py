"""
retriever.py  --  Dense and sparse retrieval over the indexed corpus.

Usage
-----
  from retriever import DenseRetriever, SparseRetriever

  dense  = DenseRetriever()
  sparse = SparseRetriever()

  hits = dense.query("how do I add OAuth2 JWT authentication?", k=5)
  hits = sparse.query("HTTPException status_code 422", k=5)

CLI
---
  python retriever.py "how do I add OAuth2 JWT authentication?"
  python retriever.py "HTTPException 422" --mode sparse
  python retriever.py "dependency injection" --mode both --k 5
"""

from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

INDEX_DIR       = Path("indexes")
QDRANT_DIR      = INDEX_DIR / "qdrant"
BM25_PATH       = INDEX_DIR / "bm25.pkl"
META_PATH       = INDEX_DIR / "metadata.json"

COLLECTION_NAME = "fastapi_docs"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

DEFAULT_K = 10


@dataclass
class RetrievalResult:
    chunk_id:     str
    score:        float
    text:         str
    source_url:   str
    doc_type:     str
    strategy:     str
    section:      str
    section_path: str
    retriever:    Literal["dense", "sparse"]
    meta:         dict = field(default_factory=dict)


class DenseRetriever:
    """Embeds the query and retrieves top-k chunks by cosine similarity."""

    def __init__(
        self,
        qdrant_dir:  Path = QDRANT_DIR,
        meta_path:   Path = META_PATH,
        embed_model: str  = EMBED_MODEL,
    ) -> None:
        self._qdrant_dir  = qdrant_dir
        self._meta_path   = meta_path
        self._embed_model = embed_model
        self._client      = None
        self._model       = None
        self._meta: dict[str, dict] = {}

    def _load(self) -> None:
        if self._client is not None:
            return
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer
        self._client = QdrantClient(path=str(self._qdrant_dir))
        self._model  = SentenceTransformer(self._embed_model)
        self._meta   = json.loads(self._meta_path.read_text(encoding="utf-8"))

    def query(
        self,
        question: str,
        k:        int = DEFAULT_K,
        doc_type: str | None = None,
    ) -> list[RetrievalResult]:
        self._load()

        vec = self._model.encode(
            question,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        query_filter = None
        if doc_type:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            query_filter = Filter(
                must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type))]
            )

        hits = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=vec,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        ).points

        results = []
        for hit in hits:
            p    = hit.payload
            meta = self._meta.get(p["chunk_id"], {})
            results.append(RetrievalResult(
                chunk_id=     p["chunk_id"],
                score=        hit.score,
                text=         p.get("text", ""),
                source_url=   p.get("source_url", ""),
                doc_type=     p.get("doc_type", ""),
                strategy=     p.get("strategy", ""),
                section=      p.get("section", ""),
                section_path= p.get("section_path", ""),
                retriever=    "dense",
                meta=         meta,
            ))

        return results


class SparseRetriever:
    """Scores the query with BM25 and returns top-k keyword-matched chunks."""

    def __init__(
        self,
        bm25_path: Path = BM25_PATH,
        meta_path: Path = META_PATH,
    ) -> None:
        self._bm25_path = bm25_path
        self._meta_path = meta_path
        self._index     = None
        self._ids: list[str]       = []
        self._meta: dict[str, dict] = {}

    def _load(self) -> None:
        if self._index is not None:
            return
        with self._bm25_path.open("rb") as fh:
            payload = pickle.load(fh)
        self._index = payload["index"]
        self._ids   = payload["ids"]
        self._meta  = json.loads(self._meta_path.read_text(encoding="utf-8"))

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return re.findall(r"[a-z0-9_]+", text.lower())

    def query(
        self,
        question: str,
        k:        int = DEFAULT_K,
        doc_type: str | None = None,
    ) -> list[RetrievalResult]:
        self._load()

        tokens = self._tokenise(question)
        scores = self._index.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RetrievalResult] = []
        for idx in ranked:
            if len(results) >= k:
                break
            if scores[idx] == 0.0:
                break

            chunk_id = self._ids[idx]
            meta     = self._meta.get(chunk_id, {})

            if doc_type and meta.get("doc_type") != doc_type:
                continue

            results.append(RetrievalResult(
                chunk_id=     chunk_id,
                score=        float(scores[idx]),
                text=         "",
                source_url=   meta.get("source_url", ""),
                doc_type=     meta.get("doc_type", ""),
                strategy=     meta.get("strategy", ""),
                section=      meta.get("section", ""),
                section_path= meta.get("section_path", ""),
                retriever=    "sparse",
                meta=         meta,
            ))

        return results


def _print_results(results: list[RetrievalResult], label: str) -> None:
    print(f"\n{label}  ({len(results)} results)")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        snippet = r.text[:100].replace("\n", " ") if r.text else "(no text)"
        print(f"  {i:>2}.  score={r.score:.4f}  [{r.doc_type}]  {r.section[:50]}")
        print(f"       {r.source_url}")
        print(f"       {snippet}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k",        type=int, default=5)
    parser.add_argument("--mode",     choices=["dense", "sparse", "both"], default="both")
    parser.add_argument("--doc-type", default=None)
    args = parser.parse_args()

    if args.mode in ("dense", "both"):
        hits = DenseRetriever().query(args.question, k=args.k, doc_type=args.doc_type)
        _print_results(hits, f"Dense -- {args.question}")

    if args.mode in ("sparse", "both"):
        hits = SparseRetriever().query(args.question, k=args.k, doc_type=args.doc_type)
        _print_results(hits, f"Sparse -- {args.question}")
