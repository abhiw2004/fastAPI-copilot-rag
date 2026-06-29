"""
eval.py  --  Run the retrieval pipeline against the golden Q&A set and generate a report.

Usage
-----
  python eval.py --strategy hybrid
  python eval.py --strategy dense-only
  python eval.py --strategy hybrid --output evaluation/report.md

This runs every question through retrieval + reranking (no LLM generation),
measures retrieval quality and refusal accuracy, and writes a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generation.confidence import compute_retrieval_score
from generation.fallback import RETRIEVAL_THRESHOLD, should_fallback
from retrieval.fusion import FusedResult, RRFFuser
from retrieval.reranker import CrossEncoderReranker, DEFAULT_CANDIDATES, DEFAULT_K
from retrieval.retriever import DenseRetriever, SparseRetriever

GOLDEN_QA_PATH = Path("evaluation/golden_qa.json")
DEFAULT_OUTPUT = Path("evaluation/report.md")


def load_golden_qa(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_single_question(
    question: str,
    strategy: str,
    dense: DenseRetriever,
    sparse: SparseRetriever,
    fuser: RRFFuser,
    reranker: CrossEncoderReranker,
    k_retrieve: int,
    k_rerank: int,
) -> dict:
    d_hits = dense.query(question, k=k_retrieve)

    if strategy == "hybrid":
        s_hits     = sparse.query(question, k=k_retrieve)
        fused      = fuser.fuse(d_hits, s_hits, k=k_retrieve)
        candidates = reranker.rerank(question, fused, k=k_rerank)
    else:
        fused_from_dense = [
            FusedResult(
                chunk_id=r.chunk_id, rrf_score=r.score, text=r.text,
                source_url=r.source_url, doc_type=r.doc_type, strategy=r.strategy,
                section=r.section, section_path=r.section_path,
                sources=["dense"], dense_rank=i + 1, dense_score=r.score,
            )
            for i, r in enumerate(d_hits[:k_retrieve])
        ]
        candidates = reranker.rerank(question, fused_from_dense, k=k_rerank)

    retrieval_score = compute_retrieval_score(candidates)
    fallback        = should_fallback(retrieval_score)

    top_doc_types  = [c.doc_type for c in candidates]
    top_scores     = [c.ce_score for c in candidates]
    used_outdated  = any(c.doc_type == "outdated" for c in candidates)

    return {
        "retrieval_score": retrieval_score,
        "fallback":        fallback,
        "top_chunk_doc_types": top_doc_types,
        "top_chunk_scores":    top_scores,
        "used_outdated":       used_outdated,
        "candidates":          candidates,
    }


def run_eval(strategy: str, k_retrieve: int = DEFAULT_CANDIDATES, k_rerank: int = DEFAULT_K) -> tuple[list[dict], dict]:
    golden = load_golden_qa(GOLDEN_QA_PATH)

    print(f"Loading models ...")
    dense    = DenseRetriever()
    sparse   = SparseRetriever()
    fuser    = RRFFuser()
    reranker = CrossEncoderReranker()

    # Warm up models with a dummy query
    dense.query("warmup", k=1)
    reranker.rerank("warmup", [], k=1)
    print(f"Models loaded. Running {len(golden)} questions with strategy={strategy} ...")

    results = []
    t0 = time.monotonic()

    for i, qa in enumerate(golden, 1):
        q_result = run_single_question(
            qa["question"], strategy, dense, sparse, fuser, reranker, k_retrieve, k_rerank,
        )

        did_refuse = q_result["fallback"]
        did_answer = not did_refuse

        entry = {
            "question_id":         qa["id"],
            "question":            qa["question"],
            "category":            qa["category"],
            "has_answer":          qa["has_answer"],
            "expected_doc_type":   qa.get("expected_doc_type"),
            "did_answer":          did_answer,
            "did_refuse":          did_refuse,
            "retrieval_score":     q_result["retrieval_score"],
            "top_chunk_doc_types": q_result["top_chunk_doc_types"],
            "top_chunk_scores":    q_result["top_chunk_scores"],
            "used_outdated":       q_result["used_outdated"],
            "confidence":          q_result["retrieval_score"] if did_answer else 0.0,
            "citation_support_rate": 0.0,
            "answer_completeness":   0.0,
            "citations_valid":     0,
            "citations_invalid":   0,
            "answer_text":         "",
        }
        results.append(entry)

        status = "REFUSE" if did_refuse else "ANSWER"
        print(f"  [{i:>2}/{len(golden)}] {status}  score={q_result['retrieval_score']:.3f}  {qa['question'][:60]}")

    elapsed = time.monotonic() - t0
    print(f"\nDone. {len(golden)} questions in {elapsed:.1f}s ({elapsed/len(golden):.2f}s/question)")

    from evaluation.metrics import EvalResult, compute_all_metrics

    eval_results = []
    for r in results:
        eval_results.append(EvalResult(
            question_id=r["question_id"],
            question=r["question"],
            category=r["category"],
            has_answer=r["has_answer"],
            expected_doc_type=r["expected_doc_type"],
            did_answer=r["did_answer"],
            did_refuse=r["did_refuse"],
            retrieval_score=r["retrieval_score"],
            top_chunk_doc_types=r["top_chunk_doc_types"],
            top_chunk_scores=r["top_chunk_scores"],
            used_outdated=r["used_outdated"],
            confidence=r["confidence"],
            citation_support_rate=r["citation_support_rate"],
            answer_completeness=r["answer_completeness"],
            citations_valid=r["citations_valid"],
            citations_invalid=r["citations_invalid"],
        ))

    metrics = compute_all_metrics(eval_results)
    metrics["strategy"] = strategy
    metrics["elapsed_seconds"] = round(elapsed, 1)

    return results, metrics


def generate_report(results: list[dict], metrics: dict, strategy: str) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"# Evaluation Report",
        f"",
        f"**Date:** {now}",
        f"**Strategy:** {strategy}",
        f"**Questions:** {metrics['total_questions']}",
        f"**Elapsed:** {metrics['elapsed_seconds']}s",
        f"",
        f"## Retrieval",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Doc-type hit rate | {metrics['retrieval']['doc_type_hit_rate']} |",
        f"| Mean top-1 CE score | {metrics['retrieval']['mean_top1_score']} |",
        f"| Mean top-5 CE score | {metrics['retrieval']['mean_top5_score']} |",
        f"| Outdated leak rate | {metrics['retrieval']['outdated_leak_rate']} |",
        f"",
        f"## Answer",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Answer rate | {metrics['answer']['answer_rate']} |",
        f"| Mean confidence | {metrics['answer']['mean_confidence']} |",
        f"| Mean completeness | {metrics['answer']['mean_completeness']} |",
        f"",
        f"## Citation",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean support rate | {metrics['citation']['mean_support_rate']} |",
        f"| Hallucinated | {metrics['citation']['hallucinated_citations']}/{metrics['citation']['total_citations']} |",
        f"",
        f"## Refusal",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| True refusal | {metrics['refusal']['true_refusal']} |",
        f"| False refusal | {metrics['refusal']['false_refusal']} |",
        f"| True answer | {metrics['refusal']['true_answer']} |",
        f"| False answer | {metrics['refusal']['false_answer']} |",
        f"| Precision | {metrics['refusal']['refusal_precision']} |",
        f"| Recall | {metrics['refusal']['refusal_recall']} |",
        f"",
        f"## Per-question results",
        f"",
        f"| # | Category | Score | Decision | Doc-types | Question |",
        f"|---|----------|-------|----------|-----------|----------|",
    ]

    for r in results:
        decision  = "REFUSE" if r["did_refuse"] else "ANSWER"
        doc_types = ", ".join(r["top_chunk_doc_types"][:3]) if r["top_chunk_doc_types"] else "-"
        q_short   = r["question"][:50]
        lines.append(
            f"| {r['question_id']} | {r['category']} | {r['retrieval_score']:.3f} "
            f"| {decision} | {doc_types} | {q_short} |"
        )

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run retrieval evaluation and generate report.")
    parser.add_argument("--strategy", choices=["hybrid", "dense-only"], default="hybrid")
    parser.add_argument("--output",   default=str(DEFAULT_OUTPUT))
    parser.add_argument("--k-retrieve", type=int, default=DEFAULT_CANDIDATES)
    parser.add_argument("--k-rerank",   type=int, default=DEFAULT_K)
    args = parser.parse_args()

    results, metrics = run_eval(args.strategy, args.k_retrieve, args.k_rerank)

    results_path = Path(args.output).with_suffix(".json")
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  Results -> {results_path}")

    report = generate_report(results, metrics, args.strategy)
    output_path = Path(args.output)
    output_path.write_text(report, encoding="utf-8")
    print(f"  Report  -> {output_path}")

    from evaluation.metrics import print_metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()
