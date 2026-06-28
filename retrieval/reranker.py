"""
reranker.py  --  Cross-encoder reranking over fused retrieval candidates.

Usage
-----
  from retriever import DenseRetriever, SparseRetriever
  from fusion    import RRFFuser
  from reranker  import CrossEncoderReranker

  dense    = DenseRetriever()
  sparse   = SparseRetriever()
  fuser    = RRFFuser()
  reranker = CrossEncoderReranker()

  candidates = fuser.fuse(dense.query(q, k=20), sparse.query(q, k=20), k=20)
  final      = reranker.rerank(q, candidates, k=5)

CLI
---
  python reranker.py "how do I add OAuth2 JWT authentication?"
  python reranker.py "HTTPException 422" --candidates 20 --k 5
  python reranker.py "background tasks FastAPI" --show-drop
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from retrieval.fusion import FusedResult

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DEFAULT_CANDIDATES = 20
DEFAULT_K          = 5


@dataclass
class RankedResult:
    chunk_id:     str
    ce_score:     float
    rrf_score:    float
    rrf_rank:     int
    ce_rank:      int
    rank_delta:   int
    text:         str
    source_url:   str
    doc_type:     str
    strategy:     str
    section:      str
    section_path: str
    sources:      list[Literal["dense", "sparse"]] = field(default_factory=list)
    meta:         dict                             = field(default_factory=dict)


class CrossEncoderReranker:
    """Reranks fused candidates using a cross-encoder model."""

    def __init__(self, model_name: str = CROSS_ENCODER_MODEL) -> None:
        self._model_name = model_name
        self._model      = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import CrossEncoder
        self._model = CrossEncoder(self._model_name)

    def rerank(
        self,
        question:   str,
        candidates: list[FusedResult],
        k:          int = DEFAULT_K,
    ) -> list[RankedResult]:
        self._load()

        if not candidates:
            return []

        pairs = [
            (question, c.text if c.text else "[no text available]")
            for c in candidates
        ]

        scores: list[float] = self._model.predict(pairs).tolist()

        for i, c in enumerate(candidates):
            if not c.text:
                scores[i] = float("-inf")

        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        results: list[RankedResult] = []
        for ce_rank, idx in enumerate(order[:k], start=1):
            c = candidates[idx]
            results.append(RankedResult(
                chunk_id=     c.chunk_id,
                ce_score=     round(scores[idx], 4),
                rrf_score=    c.rrf_score,
                rrf_rank=     idx + 1,
                ce_rank=      ce_rank,
                rank_delta=   (idx + 1) - ce_rank,
                text=         c.text,
                source_url=   c.source_url,
                doc_type=     c.doc_type,
                strategy=     c.strategy,
                section=      c.section,
                section_path= c.section_path,
                sources=      c.sources,
                meta=         c.meta,
            ))

        return results


def _print_ranked(results: list[RankedResult], dropped: list[RankedResult] | None = None) -> None:
    print(f"\nTop {len(results)} after reranking")
    print("-" * 68)
    for r in results:
        delta   = f"+{r.rank_delta}" if r.rank_delta > 0 else str(r.rank_delta)
        snippet = r.text[:100].replace("\n", " ") if r.text else "(no text)"
        print(
            f"  ce_rank={r.ce_rank}  ce={r.ce_score:>8.3f}"
            f"  rrf_rank={r.rrf_rank}  delta={delta:>3}"
            f"  [{r.doc_type}]  via={'+'.join(r.sources)}"
        )
        print(f"       {r.section[:60]}")
        print(f"       {r.source_url}")
        print(f"       {snippet}")

    if dropped:
        print(f"\nDropped ({len(dropped)})")
        for r in dropped:
            print(f"  rrf_rank={r.rrf_rank}  [{r.doc_type}]  {r.section[:50]}")


if __name__ == "__main__":
    import argparse

    from retrieval.fusion    import RRFFuser
    from retrieval.retriever import DenseRetriever, SparseRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--candidates",    type=int,   default=DEFAULT_CANDIDATES)
    parser.add_argument("--k",             type=int,   default=DEFAULT_K)
    parser.add_argument("--weight-dense",  type=float, default=1.0)
    parser.add_argument("--weight-sparse", type=float, default=1.0)
    parser.add_argument("--doc-type",      default=None)
    parser.add_argument("--show-drop",     action="store_true")
    args = parser.parse_args()

    dense    = DenseRetriever()
    sparse   = SparseRetriever()
    fuser    = RRFFuser(weight_dense=args.weight_dense, weight_sparse=args.weight_sparse)
    reranker = CrossEncoderReranker()

    d_hits     = dense.query(args.question,  k=args.candidates, doc_type=args.doc_type)
    s_hits     = sparse.query(args.question, k=args.candidates, doc_type=args.doc_type)
    candidates = fuser.fuse(d_hits, s_hits,  k=args.candidates)
    final      = reranker.rerank(args.question, candidates, k=args.k)

    kept_ids = {r.chunk_id for r in final}
    dropped  = [
        RankedResult(
            chunk_id=c.chunk_id, ce_score=0.0, rrf_score=c.rrf_score,
            rrf_rank=i + 1, ce_rank=0, rank_delta=0,
            text=c.text, source_url=c.source_url, doc_type=c.doc_type,
            strategy=c.strategy, section=c.section, section_path=c.section_path,
            sources=c.sources, meta=c.meta,
        )
        for i, c in enumerate(candidates)
        if c.chunk_id not in kept_ids
    ] if args.show_drop else None

    print(f"\nQuery      : {args.question}")
    print(f"Candidates : {len(candidates)}")
    print(f"Model      : {CROSS_ENCODER_MODEL}")
    _print_ranked(final, dropped)
