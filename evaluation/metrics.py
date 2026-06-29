"""
metrics.py  --  Evaluation harness for retrieval and answer quality.

Tracks four dimensions independently:
  1. Retrieval quality  -- did we fetch the right chunks?
  2. Answer correctness -- is the generated answer factually right?
  3. Citation validity  -- do citations point to chunks that support the claim?
  4. Refusal accuracy   -- did the system refuse when it should, and answer when it should?

Usage
-----
  python -m evaluation.metrics --results evaluation/run_results.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RetrievalMetrics:
    doc_type_hit_rate: float = 0.0
    mean_top1_score:   float = 0.0
    mean_top5_score:   float = 0.0
    outdated_leak_rate: float = 0.0


@dataclass
class AnswerMetrics:
    answer_rate:       float = 0.0
    mean_confidence:   float = 0.0
    mean_completeness: float = 0.0


@dataclass
class CitationMetrics:
    mean_support_rate:    float = 0.0
    hallucinated_citations: int = 0
    total_citations:      int = 0


@dataclass
class RefusalMetrics:
    true_refusal:   int = 0
    false_refusal:  int = 0
    true_answer:    int = 0
    false_answer:   int = 0
    refusal_precision: float = 0.0
    refusal_recall:    float = 0.0


@dataclass
class EvalResult:
    question_id:     int
    question:        str
    category:        str
    has_answer:      bool
    expected_doc_type: str | None

    did_answer:      bool = False
    did_refuse:      bool = False
    answer_text:     str = ""
    confidence:      float = 0.0
    retrieval_score: float = 0.0
    citation_support_rate: float = 0.0
    answer_completeness:   float = 0.0

    top_chunk_doc_types: list[str] = field(default_factory=list)
    top_chunk_scores:    list[float] = field(default_factory=list)
    citations_valid:     int = 0
    citations_invalid:   int = 0
    used_outdated:       bool = False


def compute_retrieval_metrics(results: list[EvalResult]) -> RetrievalMetrics:
    if not results:
        return RetrievalMetrics()

    doc_type_hits = 0
    top1_scores   = []
    top5_scores   = []
    outdated_leaks = 0
    answerable     = [r for r in results if r.has_answer and r.expected_doc_type]

    for r in answerable:
        if r.expected_doc_type in r.top_chunk_doc_types:
            doc_type_hits += 1
        if r.top_chunk_scores:
            top1_scores.append(r.top_chunk_scores[0])
            top5_scores.append(max(r.top_chunk_scores[:5]) if len(r.top_chunk_scores) >= 5 else max(r.top_chunk_scores))

    for r in results:
        if r.has_answer and r.expected_doc_type != "outdated" and r.used_outdated:
            outdated_leaks += 1

    non_outdated = [r for r in results if r.has_answer and r.expected_doc_type != "outdated"]

    return RetrievalMetrics(
        doc_type_hit_rate=doc_type_hits / len(answerable) if answerable else 0.0,
        mean_top1_score=sum(top1_scores) / len(top1_scores) if top1_scores else 0.0,
        mean_top5_score=sum(top5_scores) / len(top5_scores) if top5_scores else 0.0,
        outdated_leak_rate=outdated_leaks / len(non_outdated) if non_outdated else 0.0,
    )


def compute_answer_metrics(results: list[EvalResult]) -> AnswerMetrics:
    answerable = [r for r in results if r.has_answer]
    if not answerable:
        return AnswerMetrics()

    answered = [r for r in answerable if r.did_answer]

    return AnswerMetrics(
        answer_rate=len(answered) / len(answerable),
        mean_confidence=sum(r.confidence for r in answered) / len(answered) if answered else 0.0,
        mean_completeness=sum(r.answer_completeness for r in answered) / len(answered) if answered else 0.0,
    )


def compute_citation_metrics(results: list[EvalResult]) -> CitationMetrics:
    answered = [r for r in results if r.did_answer]
    if not answered:
        return CitationMetrics()

    total_valid   = sum(r.citations_valid for r in answered)
    total_invalid = sum(r.citations_invalid for r in answered)
    total         = total_valid + total_invalid
    support_rates = [r.citation_support_rate for r in answered if (r.citations_valid + r.citations_invalid) > 0]

    return CitationMetrics(
        mean_support_rate=sum(support_rates) / len(support_rates) if support_rates else 0.0,
        hallucinated_citations=total_invalid,
        total_citations=total,
    )


def compute_refusal_metrics(results: list[EvalResult]) -> RefusalMetrics:
    true_refusal  = 0
    false_refusal = 0
    true_answer   = 0
    false_answer  = 0

    for r in results:
        if r.has_answer and r.did_answer:
            true_answer += 1
        elif r.has_answer and r.did_refuse:
            false_refusal += 1
        elif not r.has_answer and r.did_refuse:
            true_refusal += 1
        elif not r.has_answer and r.did_answer:
            false_answer += 1

    refusal_precision = true_refusal / (true_refusal + false_refusal) if (true_refusal + false_refusal) > 0 else 0.0
    refusal_recall    = true_refusal / (true_refusal + false_answer) if (true_refusal + false_answer) > 0 else 0.0

    return RefusalMetrics(
        true_refusal=true_refusal,
        false_refusal=false_refusal,
        true_answer=true_answer,
        false_answer=false_answer,
        refusal_precision=refusal_precision,
        refusal_recall=refusal_recall,
    )


def compute_all_metrics(results: list[EvalResult]) -> dict:
    retrieval = compute_retrieval_metrics(results)
    answer    = compute_answer_metrics(results)
    citation  = compute_citation_metrics(results)
    refusal   = compute_refusal_metrics(results)

    return {
        "retrieval": {
            "doc_type_hit_rate":  round(retrieval.doc_type_hit_rate, 3),
            "mean_top1_score":    round(retrieval.mean_top1_score, 3),
            "mean_top5_score":    round(retrieval.mean_top5_score, 3),
            "outdated_leak_rate": round(retrieval.outdated_leak_rate, 3),
        },
        "answer": {
            "answer_rate":        round(answer.answer_rate, 3),
            "mean_confidence":    round(answer.mean_confidence, 3),
            "mean_completeness":  round(answer.mean_completeness, 3),
        },
        "citation": {
            "mean_support_rate":     round(citation.mean_support_rate, 3),
            "hallucinated_citations": citation.hallucinated_citations,
            "total_citations":       citation.total_citations,
        },
        "refusal": {
            "true_refusal":       refusal.true_refusal,
            "false_refusal":      refusal.false_refusal,
            "true_answer":        refusal.true_answer,
            "false_answer":       refusal.false_answer,
            "refusal_precision":  round(refusal.refusal_precision, 3),
            "refusal_recall":     round(refusal.refusal_recall, 3),
        },
        "total_questions": len(results),
    }


def load_eval_results(path: Path) -> list[EvalResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    results = []
    for d in data:
        results.append(EvalResult(
            question_id=d["question_id"],
            question=d["question"],
            category=d["category"],
            has_answer=d["has_answer"],
            expected_doc_type=d.get("expected_doc_type"),
            did_answer=d.get("did_answer", False),
            did_refuse=d.get("did_refuse", False),
            answer_text=d.get("answer_text", ""),
            confidence=d.get("confidence", 0.0),
            retrieval_score=d.get("retrieval_score", 0.0),
            citation_support_rate=d.get("citation_support_rate", 0.0),
            answer_completeness=d.get("answer_completeness", 0.0),
            top_chunk_doc_types=d.get("top_chunk_doc_types", []),
            top_chunk_scores=d.get("top_chunk_scores", []),
            citations_valid=d.get("citations_valid", 0),
            citations_invalid=d.get("citations_invalid", 0),
            used_outdated=d.get("used_outdated", False),
        ))
    return results


def print_metrics(metrics: dict) -> None:
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)

    print(f"\n  Retrieval")
    print(f"    doc_type hit rate  : {metrics['retrieval']['doc_type_hit_rate']}")
    print(f"    mean top-1 score   : {metrics['retrieval']['mean_top1_score']}")
    print(f"    mean top-5 score   : {metrics['retrieval']['mean_top5_score']}")
    print(f"    outdated leak rate : {metrics['retrieval']['outdated_leak_rate']}")

    print(f"\n  Answer")
    print(f"    answer rate        : {metrics['answer']['answer_rate']}")
    print(f"    mean confidence    : {metrics['answer']['mean_confidence']}")
    print(f"    mean completeness  : {metrics['answer']['mean_completeness']}")

    print(f"\n  Citation")
    print(f"    mean support rate  : {metrics['citation']['mean_support_rate']}")
    print(f"    hallucinated       : {metrics['citation']['hallucinated_citations']}/{metrics['citation']['total_citations']}")

    print(f"\n  Refusal")
    print(f"    true refusal       : {metrics['refusal']['true_refusal']}")
    print(f"    false refusal      : {metrics['refusal']['false_refusal']}")
    print(f"    true answer        : {metrics['refusal']['true_answer']}")
    print(f"    false answer       : {metrics['refusal']['false_answer']}")
    print(f"    precision          : {metrics['refusal']['refusal_precision']}")
    print(f"    recall             : {metrics['refusal']['refusal_recall']}")

    print(f"\n  Total questions: {metrics['total_questions']}")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    results = load_eval_results(Path(args.results))
    metrics = compute_all_metrics(results)
    print_metrics(metrics)

    output_path = Path(args.results).with_suffix(".metrics.json")
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {output_path}")
