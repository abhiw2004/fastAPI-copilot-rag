"""
fallback.py  --  Handle low-confidence retrievals gracefully.
"""

from __future__ import annotations

from contract import AnswerResponse, ConfidenceBreakdown
from retrieval.reranker import RankedResult

RETRIEVAL_THRESHOLD = 0.45


def should_fallback(retrieval_score: float) -> bool:
    return retrieval_score < RETRIEVAL_THRESHOLD


def build_fallback_response(
    question:        str,
    top_chunks:      list[RankedResult],
    retrieval_score: float,
) -> AnswerResponse:
    closest = _format_closest_sections(top_chunks[:3])

    answer = (
        f"I could not find a confident answer to this in the FastAPI documentation.\n\n"
        f"The closest sections I found were:\n{closest}\n\n"
        f"You may want to check these sections directly, or rephrase the question "
        f"with more specific terms (e.g. class names, function names, error codes)."
    )

    unverified = (
        f"Retrieval confidence too low ({retrieval_score:.2f} < {RETRIEVAL_THRESHOLD}). "
        f"No generation attempted."
    )

    return AnswerResponse(
        answer=answer,
        citations=[],
        confidence=0.0,
        confidence_breakdown=ConfidenceBreakdown(
            retrieval_score=retrieval_score,
            citation_support_rate=0.0,
            answer_completeness=0.0,
        ),
        unverified=unverified,
    )


def _format_closest_sections(chunks: list[RankedResult]) -> str:
    if not chunks:
        return "  (none)"

    lines = []
    for i, c in enumerate(chunks, 1):
        section = c.section or "(untitled section)"
        lines.append(f"  {i}. {section}\n     {c.source_url}")
    return "\n".join(lines)


if __name__ == "__main__":
    from generation.confidence import compute_retrieval_score

    fake_chunks = [
        RankedResult(
            chunk_id="chunk_001", ce_score=-1.5, rrf_score=0.01, rrf_rank=1,
            ce_rank=1, rank_delta=0,
            text="Something tangential about middleware.",
            source_url="https://fastapi.tiangolo.com/tutorial/middleware/",
            doc_type="tutorial", strategy="heading",
            section="Create a Middleware", section_path="Middleware",
        ),
        RankedResult(
            chunk_id="chunk_002", ce_score=-2.0, rrf_score=0.008, rrf_rank=2,
            ce_rank=2, rank_delta=0,
            text="CORS allows cross-origin requests.",
            source_url="https://fastapi.tiangolo.com/tutorial/cors/",
            doc_type="tutorial", strategy="heading",
            section="CORS (Cross-Origin Resource Sharing)", section_path="CORS",
        ),
        RankedResult(
            chunk_id="chunk_003", ce_score=-2.5, rrf_score=0.006, rrf_rank=3,
            ce_rank=3, rank_delta=0,
            text="Background tasks run after the response.",
            source_url="https://fastapi.tiangolo.com/tutorial/background-tasks/",
            doc_type="tutorial", strategy="heading",
            section="Background Tasks", section_path="Background Tasks",
        ),
    ]

    ret_score = compute_retrieval_score(fake_chunks)
    print(f"Retrieval score: {ret_score}")
    print(f"Should fallback: {should_fallback(ret_score)}")

    if should_fallback(ret_score):
        response = build_fallback_response(
            question="How do I deploy FastAPI on Kubernetes with Helm charts?",
            top_chunks=fake_chunks,
            retrieval_score=ret_score,
        )
        print(f"\n{response.model_dump_json(indent=2)}")
