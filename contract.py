from typing import Optional

from pydantic import BaseModel, Field, model_validator


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    doc_type: Optional[str] = Field(default=None)


class Citation(BaseModel):
    chunk_id:      str
    source:        str
    section:       str
    supports_claim: bool


class ConfidenceBreakdown(BaseModel):
    retrieval_score:       float = Field(..., ge=0.0, le=1.0)
    citation_support_rate: float = Field(..., ge=0.0, le=1.0)
    answer_completeness:   float = Field(..., ge=0.0, le=1.0)


class AnswerResponse(BaseModel):
    answer:               str
    citations:            list[Citation] = Field(default_factory=list)
    confidence:           float = Field(..., ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    unverified:           Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def citations_or_unverified(self) -> "AnswerResponse":
        if not self.citations and self.unverified is None:
            raise ValueError(
                "citations is empty but unverified is None — "
                "provide at least one citation or set unverified"
            )
        return self


if __name__ == "__main__":
    good = AnswerResponse(
        answer="You can add OAuth2 scopes using the SecurityScopes dependency.",
        citations=[Citation(
            chunk_id="advanced_security_oauth2-scopes_004",
            source="advanced/security/oauth2-scopes.html",
            section="Use SecurityScopes",
            supports_claim=True,
        )],
        confidence=0.87,
        confidence_breakdown=ConfidenceBreakdown(
            retrieval_score=0.9, citation_support_rate=1.0, answer_completeness=0.8
        ),
    )
    print(good.model_dump_json(indent=2))

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
