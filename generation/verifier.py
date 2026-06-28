"""
verifier.py  --  Parse and verify citations from a generated answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contract import Citation
from retrieval.reranker import RankedResult

CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")
MIN_TERM_OVERLAP = 0.3


@dataclass
class ClaimCitation:
    chunk_id: str
    claim:    str
    position: int


class CitationVerifier:
    def __init__(
        self,
        chunks_by_id: dict[str, RankedResult],
        min_overlap:  float = MIN_TERM_OVERLAP,
    ) -> None:
        self._chunks      = chunks_by_id
        self._min_overlap = min_overlap

    def verify(self, answer: str) -> tuple[list[Citation], float]:
        claim_citations = self._extract_citations(answer)

        if not claim_citations:
            return [], 0.0

        citations: list[Citation] = []
        supported_count = 0

        for cc in claim_citations:
            chunk = self._chunks.get(cc.chunk_id)

            if chunk is None:
                citations.append(Citation(
                    chunk_id=cc.chunk_id,
                    source="",
                    section="",
                    supports_claim=False,
                ))
                continue

            supports = self._check_support(cc.claim, chunk.text)
            if supports:
                supported_count += 1

            citations.append(Citation(
                chunk_id=cc.chunk_id,
                source=chunk.source_url,
                section=chunk.section,
                supports_claim=supports,
            ))

        support_rate = supported_count / len(citations) if citations else 0.0
        return citations, round(support_rate, 3)

    def _extract_citations(self, answer: str) -> list[ClaimCitation]:
        results: list[ClaimCitation] = []
        seen:    set[str] = set()

        for match in CITATION_PATTERN.finditer(answer):
            chunk_id = match.group(1)
            if chunk_id in seen:
                continue
            seen.add(chunk_id)

            claim = self._extract_sentence(answer, match.start())
            results.append(ClaimCitation(
                chunk_id=chunk_id,
                claim=claim,
                position=match.start(),
            ))

        return results

    def _extract_sentence(self, text: str, citation_pos: int) -> str:
        sentence_ends = re.compile(r"[.!?\n]")

        start = citation_pos
        while start > 0 and not sentence_ends.match(text[start - 1]):
            start -= 1

        end = citation_pos
        while end < len(text) and not sentence_ends.match(text[end]):
            end += 1

        sentence = text[start:end].strip()
        sentence = CITATION_PATTERN.sub("", sentence).strip()
        return sentence

    def _check_support(self, claim: str, chunk_text: str) -> bool:
        claim_terms = self._tokenise(claim)
        if not claim_terms:
            return True

        chunk_terms = set(self._tokenise(chunk_text))
        overlap     = sum(1 for t in claim_terms if t in chunk_terms)
        ratio       = overlap / len(claim_terms)

        return ratio >= self._min_overlap

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        tokens = re.findall(r"[a-z0-9_]+", text.lower())
        stop   = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                  "to", "of", "in", "for", "on", "with", "at", "by", "from",
                  "it", "this", "that", "and", "or", "but", "not", "you", "can",
                  "do", "if", "as", "your", "will", "has", "have", "had"}
        return [t for t in tokens if t not in stop and len(t) > 1]


if __name__ == "__main__":
    fake_chunks = {
        "chunk_001": RankedResult(
            chunk_id="chunk_001", ce_score=4.0, rrf_score=0.02, rrf_rank=1,
            ce_rank=1, rank_delta=0,
            text="Use HTTPException to return HTTP error responses. "
                 "Import it from fastapi. Raise HTTPException with a status_code.",
            source_url="https://fastapi.tiangolo.com/tutorial/handling-errors/",
            doc_type="tutorial", strategy="heading",
            section="Use HTTPException", section_path="Handling Errors > Use HTTPException",
        ),
        "chunk_002": RankedResult(
            chunk_id="chunk_002", ce_score=3.5, rrf_score=0.01, rrf_rank=2,
            ce_rank=2, rank_delta=0,
            text="CORS middleware allows cross-origin requests. "
                 "Add CORSMiddleware to your application.",
            source_url="https://fastapi.tiangolo.com/tutorial/cors/",
            doc_type="tutorial", strategy="heading",
            section="CORS", section_path="CORS",
        ),
    }

    answer = (
        "To return an error response, raise HTTPException with a status_code [chunk_001]. "
        "You can also configure CORS using CORSMiddleware [chunk_002]. "
        "For WebSocket errors, use a different approach [chunk_999]."
    )

    verifier = CitationVerifier(fake_chunks)
    citations, rate = verifier.verify(answer)

    print(f"Support rate: {rate}")
    for c in citations:
        print(f"  {c.chunk_id}: supports={c.supports_claim}  source={c.source}")
