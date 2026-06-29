"""
llm.py  --  LLM integration via Groq (Llama 3.3 70B).

Requires GROQ_API_KEY in environment or .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from generation.prompt import SYSTEM_PROMPT, build_user_prompt
from retrieval.reranker import RankedResult

MODEL = "llama-3.3-70b-versatile"

load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com")
    return Groq(api_key=api_key)


def _filter_outdated(chunks: list[RankedResult]) -> list[RankedResult]:
    """Drop all outdated chunks. Current docs always take priority."""
    filtered = [c for c in chunks if c.doc_type != "outdated"]
    return filtered if filtered else chunks


def _prefer_heading_chunks(chunks: list[RankedResult]) -> list[RankedResult]:
    """
    When both heading and fixed chunks exist for the same source file,
    keep heading chunks (complete sections) over fixed chunks (arbitrary windows).
    """
    heading_sources = set()
    for c in chunks:
        if c.strategy == "heading":
            heading_sources.add(c.source_url)

    filtered = []
    for c in chunks:
        if c.strategy == "fixed" and c.source_url in heading_sources:
            continue
        filtered.append(c)

    if not filtered:
        return chunks
    return filtered


def generate(question: str, chunks: list[RankedResult]) -> str:
    client = _get_client()
    chunks = _filter_outdated(chunks)
    chunks = _prefer_heading_chunks(chunks)
    user_prompt = build_user_prompt(question, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from generation.confidence import compute_confidence, compute_retrieval_score
    from generation.fallback import build_fallback_response, should_fallback
    from generation.verifier import CitationVerifier
    from retrieval.fusion import RRFFuser
    from retrieval.reranker import CrossEncoderReranker
    from retrieval.retriever import DenseRetriever, SparseRetriever

    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    dense    = DenseRetriever()
    sparse   = SparseRetriever()
    fuser    = RRFFuser()
    reranker = CrossEncoderReranker()

    d_hits     = dense.query(args.question, k=20)
    s_hits     = sparse.query(args.question, k=20)
    candidates = fuser.fuse(d_hits, s_hits, k=20)

    candidates_filtered = [c for c in candidates if c.doc_type != "outdated"]
    if not candidates_filtered:
        candidates_filtered = candidates

    top_chunks = reranker.rerank(args.question, candidates_filtered, k=args.k)

    retrieval_score = compute_retrieval_score(top_chunks)

    if should_fallback(retrieval_score):
        response = build_fallback_response(args.question, top_chunks, retrieval_score)
        print(response.model_dump_json(indent=2))
        sys.exit(0)

    print(f"Generating answer (model={MODEL}) ...")
    answer = generate(args.question, top_chunks)

    chunks_by_id = {c.chunk_id: c for c in top_chunks}
    verifier     = CitationVerifier(chunks_by_id)
    citations, support_rate = verifier.verify(answer)

    confidence, breakdown = compute_confidence(
        question=args.question,
        answer=answer,
        top_chunks=top_chunks,
        citations=citations,
        support_rate=support_rate,
    )

    print(f"\nAnswer:\n{answer}")
    print(f"\nConfidence: {confidence}")
    print(f"  retrieval_score:       {breakdown.retrieval_score}")
    print(f"  citation_support_rate: {breakdown.citation_support_rate}")
    print(f"  answer_completeness:   {breakdown.answer_completeness}")
    print(f"\nCitations:")
    for c in citations:
        status = "supported" if c.supports_claim else "UNSUPPORTED"
        print(f"  [{status}] {c.chunk_id}")
