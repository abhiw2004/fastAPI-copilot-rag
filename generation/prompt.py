"""
prompt.py  --  System and user prompt templates for the generation step.
"""

from __future__ import annotations

from retrieval.reranker import RankedResult

SYSTEM_PROMPT = """\
You are a FastAPI documentation assistant.

Rules:
1. Answer ONLY from the provided context chunks. Do not use prior knowledge.
2. Cite every factual claim with [chunk_id]. Place the citation inline, immediately after the claim it supports.
3. If the context does not contain enough information to answer, respond with: "I don't have enough information to answer this based on the available documentation."
4. Do not guess, speculate, or infer beyond what the chunks explicitly state.
5. If only part of the question can be answered, answer that part with citations and state what remains unanswered.
6. Use code examples from the chunks when relevant. Attribute them with their chunk_id.
7. Keep answers concise and direct.
"""


def build_user_prompt(question: str, chunks: list[RankedResult]) -> str:
    context_block = "\n\n".join(
        f"[{c.chunk_id}]\n"
        f"source: {c.source_url}\n"
        f"section: {c.section}\n"
        f"---\n"
        f"{c.text}"
        for c in chunks
    )

    return f"Context:\n{context_block}\n\nQuestion: {question}"
