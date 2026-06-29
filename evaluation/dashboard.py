"""
dashboard.py  --  Streamlit dashboard for inspecting RAG pipeline outputs.

Shows:
  - Question and generated answer
  - Retrieved chunks with scores
  - Citation verdicts (supported / unsupported)
  - Confidence breakdown
  - Toggle to compare dense-only vs hybrid (dense+sparse) retrieval

Run
---
  streamlit run evaluation/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from generation.confidence import compute_retrieval_score
from generation.fallback import should_fallback
from generation.verifier import CitationVerifier
from retrieval.fusion import RRFFuser
from retrieval.reranker import CrossEncoderReranker, RankedResult
from retrieval.retriever import DenseRetriever, SparseRetriever


@st.cache_resource
def load_dense():
    return DenseRetriever()


@st.cache_resource
def load_sparse():
    return SparseRetriever()


@st.cache_resource
def load_reranker():
    return CrossEncoderReranker()


def run_pipeline(question: str, mode: str, k_retrieve: int, k_rerank: int) -> dict:
    dense    = load_dense()
    sparse   = load_sparse()
    reranker = load_reranker()
    fuser    = RRFFuser()

    d_hits = dense.query(question, k=k_retrieve)

    if mode == "hybrid":
        s_hits     = sparse.query(question, k=k_retrieve)
        fused      = fuser.fuse(d_hits, s_hits, k=k_retrieve)
    else:
        from retrieval.fusion import FusedResult
        fused = [
            FusedResult(
                chunk_id=r.chunk_id, rrf_score=r.score, text=r.text,
                source_url=r.source_url, doc_type=r.doc_type, strategy=r.strategy,
                section=r.section, section_path=r.section_path,
                sources=["dense"], dense_rank=i + 1, dense_score=r.score,
            )
            for i, r in enumerate(d_hits[:k_retrieve])
        ]

    fused_filtered = [c for c in fused if c.doc_type != "outdated"] or fused
    candidates = reranker.rerank(question, fused_filtered, k=k_rerank)

    retrieval_score = compute_retrieval_score(candidates)
    fallback        = should_fallback(retrieval_score)

    chunks_by_id = {c.chunk_id: c for c in candidates}
    verifier     = CitationVerifier(chunks_by_id)

    return {
        "candidates":      candidates,
        "retrieval_score": retrieval_score,
        "fallback":        fallback,
        "chunks_by_id":    chunks_by_id,
        "verifier":        verifier,
    }


def main():
    st.set_page_config(page_title="RAG Pipeline Dashboard", layout="wide")
    st.title("RAG Pipeline Dashboard")

    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("Settings")
        mode = st.radio("Retrieval mode", ["hybrid", "dense-only"], index=0)
        k_retrieve = st.slider("Candidates per retriever", 5, 40, 20)
        k_rerank   = st.slider("Final chunks after reranking", 3, 10, 5)

    with col_left:
        question = st.text_input("Question", placeholder="How do I add OAuth2 JWT authentication?")

    if not question:
        st.info("Enter a question above to run the pipeline.")
        return

    mode_key = "hybrid" if mode == "hybrid" else "dense_only"
    result   = run_pipeline(question, mode_key, k_retrieve, k_rerank)

    candidates      = result["candidates"]
    retrieval_score = result["retrieval_score"]
    fallback        = result["fallback"]

    st.divider()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Retrieval score", f"{retrieval_score:.3f}")
    col_b.metric("Chunks retrieved", len(candidates))
    col_c.metric("Would fallback", "Yes" if fallback else "No")

    st.subheader("Retrieved chunks")
    for i, c in enumerate(candidates, 1):
        with st.expander(f"{i}. [{c.doc_type}] {c.section[:60]}  (ce={c.ce_score:.3f}, delta={c.rank_delta:+d})"):
            st.caption(c.source_url)
            st.text(c.text[:500] if c.text else "(no text)")
            st.json({
                "chunk_id":   c.chunk_id,
                "ce_score":   c.ce_score,
                "rrf_score":  c.rrf_score,
                "rrf_rank":   c.rrf_rank,
                "ce_rank":    c.ce_rank,
                "rank_delta": c.rank_delta,
                "strategy":   c.strategy,
                "sources":    c.sources,
            })

    st.divider()
    st.subheader("Generated answer")

    if fallback:
        from generation.fallback import build_fallback_response
        response = build_fallback_response(question, candidates, retrieval_score)
        st.warning("Retrieval confidence too low. Showing fallback response.")
        st.markdown(response.answer)
        st.json({
            "confidence": response.confidence,
            "unverified": response.unverified,
        })
    else:
        if st.button("Generate answer", type="primary"):
            with st.spinner("Generating ..."):
                from generation.llm import generate
                from generation.confidence import compute_confidence

                answer = generate(question, candidates)

            st.markdown("**Answer:**")
            st.markdown(answer)

            st.divider()
            st.subheader("Citation verification")

            verifier = result["verifier"]
            citations, support_rate = verifier.verify(answer)

            col_x, col_y = st.columns(2)
            col_x.metric("Support rate", f"{support_rate:.1%}")
            col_y.metric("Citations found", len(citations))

            for c in citations:
                status = "supported" if c.supports_claim else "unsupported"
                icon   = "+" if c.supports_claim else "-"
                st.text(f"  [{icon}] {c.chunk_id}  ({status})  source={c.source}")

            st.divider()
            st.subheader("Confidence breakdown")

            confidence, breakdown = compute_confidence(
                question=question,
                answer=answer,
                top_chunks=candidates,
                citations=citations,
                support_rate=support_rate,
            )

            col_m, col_n, col_o, col_p = st.columns(4)
            col_m.metric("Confidence", f"{confidence:.3f}")
            col_n.metric("Retrieval", f"{breakdown.retrieval_score:.3f}")
            col_o.metric("Citation rate", f"{breakdown.citation_support_rate:.1%}")
            col_p.metric("Completeness", f"{breakdown.answer_completeness:.3f}")

    if mode == "hybrid":
        with st.expander("Compare: what would dense-only return?"):
            dense_result = run_pipeline(question, "dense_only", k_retrieve, k_rerank)
            dense_score  = dense_result["retrieval_score"]
            st.metric("Dense-only retrieval score", f"{dense_score:.3f}")
            for i, c in enumerate(dense_result["candidates"], 1):
                st.text(f"  {i}. ce={c.ce_score:.3f}  [{c.doc_type}]  {c.section[:50]}")


if __name__ == "__main__":
    main()
