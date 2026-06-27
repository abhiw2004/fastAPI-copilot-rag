from typing import Optional

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    """What goes IN to the system."""

    question: str = Field(..., min_length=1, description="The user's raw question")
    doc_type: Optional[str] = Field(
        default=None,
        description="Optional filter, e.g. 'tutorial', 'advanced', 'reference'",
    )


class Citation(BaseModel):
    """One piece of evidence backing a claim in the answer."""

    chunk_id: str
    source: str
    section: str
    supports_claim: bool = Field(
        ..., description="Set by the citation verifier, not the generation step"
    )


class ConfidenceBreakdown(BaseModel):
    """The components that get combined into the single `confidence` score."""

    retrieval_score: float = Field(..., ge=0.0, le=1.0)
    citation_support_rate: float = Field(..., ge=0.0, le=1.0)
    answer_completeness: float = Field(..., ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    """What comes OUT of the system. This is the contract's core shape."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    unverified: Optional[str] = Field(
        default=None,
        description="Set when the system could not find or verify part of the answer",
    )

    @model_validator(mode="after")
    def citations_or_unverified(self) -> "AnswerResponse":
        """
        Rule: if the system found nothing worth citing, it must say so via
        `unverified` rather than silently returning an answer with zero
        citations. Keeps the "no-answer handling" requirement enforced in
        code, not just in a prompt instruction the LLM might ignore.
        """
        if not self.citations and self.unverified is None:
            raise ValueError(
                "citations is empty but unverified is None — "
                "either provide at least one citation or explain what could not be verified"
            )
        return self


if __name__ == "__main__":
    # Quick self-test. Run `python contract.py` to confirm the contract works.
    good = AnswerResponse(
        answer="You can add OAuth2 scopes using the SecurityScopes dependency.",
        citations=[
            Citation(
                chunk_id="advanced_security_oauth2-scopes_004",
                source="advanced/security/oauth2-scopes.html",
                section="Use SecurityScopes",
                supports_claim=True,
            )
        ],
        confidence=0.87,
        confidence_breakdown=ConfidenceBreakdown(
            retrieval_score=0.9, citation_support_rate=1.0, answer_completeness=0.8
        ),
    )
    print("Valid response OK:")
    print(good.model_dump_json(indent=2))

    print("\nTesting invalid response (no citations, no unverified)...")
    try:
        AnswerResponse(
            answer="Something",
            citations=[],
            confidence=0.5,
            confidence_breakdown=ConfidenceBreakdown(
                retrieval_score=0.5, citation_support_rate=0.0, answer_completeness=0.5
            ),
        )
    except Exception as exc:
        print(f"Correctly rejected: {exc}")