"""
fusion.py  --  Reciprocal Rank Fusion over dense and sparse result lists.

Usage
-----
  from retriever import DenseRetriever, SparseRetriever
  from fusion import RRFFuser

  fuser   = RRFFuser()
  dense   = DenseRetriever()
  sparse  = SparseRetriever()

  results = fuser.fuse(dense.query(q, k=20), sparse.query(q, k=20), k=10)

CLI
---
  python fusion.py "how do I add OAuth2 JWT authentication?" --k 5
  python fusion.py "dependency injection" --k 5 --weight-dense 2.0
  python fusion.py "dependency injection" --k 5 --weight-sparse 2.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from retriever import DEFAULT_K, RetrievalResult

RRF_K = 60


@dataclass
class FusedResult:
    chunk_id:     str
    rrf_score:    float
    text:         str
    source_url:   str
    doc_type:     str
    strategy:     str
    section:      str
    section_path: str
    sources:      list[Literal["dense", "sparse"]] = field(default_factory=list)
    dense_rank:   int   = 0
    dense_score:  float = 0.0
    sparse_rank:  int   = 0
    sparse_score: float = 0.0
    meta:         dict  = field(default_factory=dict)


class RRFFuser:
    """Merges dense and sparse result lists with Reciprocal Rank Fusion."""

    def __init__(
        self,
        rrf_k:         int   = RRF_K,
        weight_dense:  float = 1.0,
        weight_sparse: float = 1.0,
    ) -> None:
        self.rrf_k         = rrf_k
        self.weight_dense  = weight_dense
        self.weight_sparse = weight_sparse

    def fuse(
        self,
        dense_results:  list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        k:              int = DEFAULT_K,
    ) -> list[FusedResult]:
        scores:       dict[str, float]           = {}
        dense_rank:   dict[str, int]             = {}
        dense_score:  dict[str, float]           = {}
        sparse_rank:  dict[str, int]             = {}
        sparse_score: dict[str, float]           = {}
        texts:        dict[str, str]             = {}
        meta_by_id:   dict[str, dict]            = {}
        payload:      dict[str, RetrievalResult] = {}

        for rank, result in enumerate(dense_results, start=1):
            cid = result.chunk_id
            scores[cid]      = scores.get(cid, 0.0) + self.weight_dense / (self.rrf_k + rank)
            dense_rank[cid]  = rank
            dense_score[cid] = result.score
            if result.text:
                texts[cid] = result.text
            meta_by_id[cid] = result.meta
            payload[cid]    = result

        for rank, result in enumerate(sparse_results, start=1):
            cid = result.chunk_id
            scores[cid]       = scores.get(cid, 0.0) + self.weight_sparse / (self.rrf_k + rank)
            sparse_rank[cid]  = rank
            sparse_score[cid] = result.score
            if result.text and cid not in texts:
                texts[cid] = result.text
            if cid not in meta_by_id:
                meta_by_id[cid] = result.meta
            if cid not in payload:
                payload[cid] = result

        ranked = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)[:k]

        results: list[FusedResult] = []
        for cid in ranked:
            r       = payload[cid]
            sources = []
            if cid in dense_rank:
                sources.append("dense")
            if cid in sparse_rank:
                sources.append("sparse")

            results.append(FusedResult(
                chunk_id=     cid,
                rrf_score=    round(scores[cid], 6),
                text=         texts.get(cid, ""),
                source_url=   r.source_url,
                doc_type=     r.doc_type,
                strategy=     r.strategy,
                section=      r.section,
                section_path= r.section_path,
                sources=      sources,
                dense_rank=   dense_rank.get(cid, 0),
                dense_score=  round(dense_score.get(cid, 0.0), 4),
                sparse_rank=  sparse_rank.get(cid, 0),
                sparse_score= round(sparse_score.get(cid, 0.0), 4),
                meta=         meta_by_id.get(cid, {}),
            ))

        return results


def _print_fused(results: list[FusedResult], label: str) -> None:
    print(f"\n{label}  ({len(results)} results)")
    print("-" * 68)
    for i, r in enumerate(results, 1):
        via     = "+".join(r.sources)
        snippet = r.text[:90].replace("\n", " ") if r.text else "(no text)"
        dr      = r.dense_rank  or "-"
        sr      = r.sparse_rank or "-"
        print(f"  {i:>2}.  rrf={r.rrf_score:.5f}  [{r.doc_type}]  via={via}  d_rank={dr}  s_rank={sr}")
        print(f"       d_score={r.dense_score:.4f}  s_score={r.sparse_score:.4f}")
        print(f"       {r.section[:60]}")
        print(f"       {r.source_url}")
        print(f"       {snippet}")


if __name__ == "__main__":
    import argparse

    from retriever import DenseRetriever, SparseRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k",             type=int,   default=5)
    parser.add_argument("--fetch",         type=int,   default=20)
    parser.add_argument("--weight-dense",  type=float, default=1.0)
    parser.add_argument("--weight-sparse", type=float, default=1.0)
    parser.add_argument("--rrf-k",         type=int,   default=RRF_K)
    parser.add_argument("--doc-type",      default=None)
    args = parser.parse_args()

    fuser  = RRFFuser(rrf_k=args.rrf_k, weight_dense=args.weight_dense, weight_sparse=args.weight_sparse)
    d_hits = DenseRetriever().query(args.question,  k=args.fetch, doc_type=args.doc_type)
    s_hits = SparseRetriever().query(args.question, k=args.fetch, doc_type=args.doc_type)
    fused  = fuser.fuse(d_hits, s_hits, k=args.k)

    _print_fused(fused, f"RRF  wd={args.weight_dense}  ws={args.weight_sparse}  -- {args.question}")
