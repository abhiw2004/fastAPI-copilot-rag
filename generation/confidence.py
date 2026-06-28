"""
confidence.py  --  Combine retrieval, citation, and completeness signals
into a single confidence score with full breakdown.
"""

from __future__ import annotations

import math
import re

from contract import Citation, ConfidenceBreakdown
from retrieval.reranker import RankedResult

NO_ANSWER_PHRASES = [
    "i don't have enough information",
    "not available in the documentation",
    "cannot answer",
    "no relevant information",
    "not covered in the provided context",
]

WEIGHT_RETRIEVAL    = 0.35
WEIGHT_CITATION     = 0.40
WEIGHT_COMPLETENESS = 0.25


def compute_retrieval_score(top_chunks: list[RankedResult]) -> float:
    if not top_chunks:
        return 0.0
    best_ce = max(c.ce_score for c in top_chunks)
    score   = 1.0 / (1.0 + math.exp(-best_ce))
    return round(min(max(score, 0.0), 1.0), 3)


def compute_answer_completeness(
    question:  str,
    answer:    str,
    citations: list[Citation],
) -> float:
    if _is_no_answer(answer):
        return 0.0

    score = 0.0

    supported       = sum(1 for c in citations if c.supports_claim)
    citation_factor = min(supported / 3.0, 1.0)
    score += 0.5 * citation_factor

    answer_len    = len(answer.split())
    length_factor = min(answer_len / 50.0, 1.0)
    score += 0.3 * length_factor

    question_terms = set(re.findall(r"[a-z0-9_]+", question.lower()))
    answer_terms   = set(re.findall(r"[a-z0-9_]+", answer.lower()))
    if question_terms:
        overlap = len(question_terms & answer_terms) / len(question_terms)
        score += 0.2 * overlap

    return round(min(max(score, 0.0), 1.0), 3)


def _is_no_answer(answer: str) -> bool:
    lower = answer.lower()
    return any(phrase in lower for phrase in NO_ANSWER_PHRASES)


def compute_confidence(
    question:     str,
    answer:       str,
    top_chunks:   list[RankedResult],
    citations:    list[Citation],
    support_rate: float,
) -> tuple[float, ConfidenceBreakdown]:
    if _is_no_answer(answer):
        breakdown = ConfidenceBreakdown(
            retrieval_score=compute_retrieval_score(top_chunks),
            citation_support_rate=0.0,
            answer_completeness=0.0,
        )
        return 0.0, breakdown

    retrieval_score     = compute_retrieval_score(top_chunks)
    answer_completeness = compute_answer_completeness(question, answer, citations)

    confidence = (
        WEIGHT_RETRIEVAL    * retrieval_score
        + WEIGHT_CITATION   * support_rate
        + WEIGHT_COMPLETENESS * answer_completeness
    )
    confidence = round(min(max(confidence, 0.0), 1.0), 3)

    breakdown = ConfidenceBreakdown(
        retrieval_score=retrieval_score,
        citation_support_rate=round(support_rate, 3),
        answer_completeness=answer_completeness,
    )

    return confidence, breakdown


if __name__ == "__main__":
    fake_chunks = [
        RankedResult(
            chunk_id="chunk_001", ce_score=4.0, rrf_score=0.02, rrf_rank=1,
            ce_rank=1, rank_delta=0,
            text="Use HTTPException to return errors.",
            source_url="https://fastapi.tiangolo.com/tutorial/handling-errors/",
            doc_type="tutorial", strategy="heading",
            section="Use HTTPException", section_path="",
        ),
        RankedResult(
            chunk_id="chunk_002", ce_score=2.5, rrf_score=0.01, rrf_rank=2,
            ce_rank=2, rank_delta=0,
            text="Import HTTPException from fastapi.",
            source_url="https://fastapi.tiangolo.com/tutorial/handling-errors/",
            doc_type="tutorial", strategy="heading",
            section="Import", section_path="",
        ),
    ]

    citations = [
        Citation(chunk_id="chunk_001", source="handling-errors/", section="Use HTTPException", supports_claim=True),
        Citation(chunk_id="chunk_002", source="handling-errors/", section="Import", supports_claim=True),
    ]

    answer = (
        "To return HTTP error responses in FastAPI, raise HTTPException "
        "with the appropriate status_code parameter [chunk_001]. "
        "First import it from the fastapi module [chunk_002]."
    )

    conf, breakdown = compute_confidence(
        question="How do I raise HTTP errors in FastAPI?",
        answer=answer,
        top_chunks=fake_chunks,
        citations=citations,
        support_rate=1.0,
    )

    print(f"Confidence: {conf}")
    print(f"  retrieval_score:       {breakdown.retrieval_score}")
    print(f"  citation_support_rate: {breakdown.citation_support_rate}")
    print(f"  answer_completeness:   {breakdown.answer_completeness}")

    no_answer = "I don't have enough information to answer this based on the available documentation."
    conf2, bd2 = compute_confidence(
        question="What is quantum computing?",
        answer=no_answer,
        top_chunks=fake_chunks,
        citations=[],
        support_rate=0.0,
    )
    print(f"\nNo-answer confidence: {conf2}")
    print(f"  retrieval_score: {bd2.retrieval_score}")
