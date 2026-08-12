#!/usr/bin/env python3
"""
Evaluation harness. Runs the fixed question set (eval/questions.json)
through the live retrieval + generation pipeline and computes all three
layers of evaluation required by the assignment:

  1. Retrieval quality  : Hit Rate@k, Recall@k, MRR, nDCG@k, context precision@k
  2. Answer quality     : EM, F1, LLM-judged faithfulness + relevance
  3. Latency            : p50/p95 retrieval and total latency

Writes a JSON results file to eval/results.json and prints a summary
table to stdout.

Usage:
    python eval/run_eval.py --top-k 5
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.retrieval import retrieve, filter_relevant
from app.generation import generate_answer
from app.config import settings
from app import vectorstore

from eval.retrieval_metrics import hit_rate_at_k, recall_at_k, mrr, ndcg_at_k, context_precision_at_k
from eval.answer_metrics import exact_match, f1_score, judge_answer

QUESTIONS_PATH = Path(__file__).parent / "questions.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def compute_gold_relevant_ids(source: str, gold_keyphrase: str) -> set[str]:
    """Find chunk IDs in the store that belong to `source` and contain the
    gold key-phrase, i.e. the ground-truth relevant chunk(s) for a question."""
    if not source or not gold_keyphrase:
        return set()
    collection = vectorstore.get_collection()
    result = collection.get(where={"source": source}, include=["documents"])
    ids = result.get("ids", [])
    docs = result.get("documents", [])
    relevant = {
        cid for cid, doc in zip(ids, docs) if gold_keyphrase.lower() in doc.lower()
    }
    return relevant


def run_eval(top_k: int):
    questions = json.loads(QUESTIONS_PATH.read_text())
    per_question_results = []

    for q in questions:
        t0 = time.perf_counter()
        retrieved = retrieve(q["question"], top_k=top_k)
        retrieval_ms = (time.perf_counter() - t0) * 1000

        relevant_chunks = filter_relevant(retrieved)

        t1 = time.perf_counter()
        gen = generate_answer(q["question"], relevant_chunks)
        generation_ms = (time.perf_counter() - t1) * 1000

        retrieved_ids = [c.chunk_id for c in retrieved]

        record = {
            "id": q["id"],
            "question": q["question"],
            "answerable": q["answerable"],
            "answer": gen.answer,
            "grounded": gen.grounded,
            "retrieval_latency_ms": round(retrieval_ms, 2),
            "generation_latency_ms": round(generation_ms, 2),
            "total_latency_ms": round(retrieval_ms + generation_ms, 2),
            "prompt_tokens": gen.prompt_tokens,
            "completion_tokens": gen.completion_tokens,
        }

        if q["answerable"]:
            gold_ids = compute_gold_relevant_ids(q["source"], q["gold_keyphrase"])
            record["retrieval"] = {
                "hit_rate": hit_rate_at_k(retrieved_ids, gold_ids),
                "recall": round(recall_at_k(retrieved_ids, gold_ids), 4),
                "mrr": round(mrr(retrieved_ids, gold_ids), 4),
                "ndcg": round(ndcg_at_k(retrieved_ids, gold_ids), 4),
                "context_precision": round(context_precision_at_k(retrieved_ids, gold_ids), 4),
                "num_gold_relevant": len(gold_ids),
            }
            record["em"] = exact_match(gen.answer, q["gold_answer"])
            record["f1"] = round(f1_score(gen.answer, q["gold_answer"]), 4)

            context_text = "\n".join(c.text for c in relevant_chunks)
            verdict = judge_answer(q["question"], context_text, gen.answer)
            record["judge"] = {
                "faithfulness": verdict.faithfulness,
                "relevance": verdict.relevance,
                "rationale": verdict.rationale,
            }
        else:
            record["correctly_abstained"] = int(not gen.grounded)

        per_question_results.append(record)

    return per_question_results


def summarize(results: list[dict]) -> dict:
    answerable = [r for r in results if r["answerable"]]
    unanswerable = [r for r in results if not r["answerable"]]

    total_latencies = [r["total_latency_ms"] for r in results]
    retrieval_latencies = [r["retrieval_latency_ms"] for r in results]

    def pctile(data, p):
        if not data:
            return 0.0
        s = sorted(data)
        idx = min(int(len(s) * p), len(s) - 1)
        return round(s[idx], 2)

    summary = {
        "num_questions": len(results),
        "num_answerable": len(answerable),
        "num_unanswerable": len(unanswerable),
        "retrieval": {
            "mean_hit_rate": round(statistics.mean(r["retrieval"]["hit_rate"] for r in answerable), 4) if answerable else 0,
            "mean_recall": round(statistics.mean(r["retrieval"]["recall"] for r in answerable), 4) if answerable else 0,
            "mean_mrr": round(statistics.mean(r["retrieval"]["mrr"] for r in answerable), 4) if answerable else 0,
            "mean_ndcg": round(statistics.mean(r["retrieval"]["ndcg"] for r in answerable), 4) if answerable else 0,
            "mean_context_precision": round(statistics.mean(r["retrieval"]["context_precision"] for r in answerable), 4) if answerable else 0,
        },
        "answer_quality": {
            "mean_em": round(statistics.mean(r["em"] for r in answerable), 4) if answerable else 0,
            "mean_f1": round(statistics.mean(r["f1"] for r in answerable), 4) if answerable else 0,
            "mean_faithfulness": round(statistics.mean(r["judge"]["faithfulness"] for r in answerable), 2) if answerable else 0,
            "mean_relevance": round(statistics.mean(r["judge"]["relevance"] for r in answerable), 2) if answerable else 0,
        },
        "abstention": {
            "correctly_abstained_rate": round(statistics.mean(r["correctly_abstained"] for r in unanswerable), 4) if unanswerable else None,
        },
        "latency": {
            "p50_total_ms": pctile(total_latencies, 0.50),
            "p95_total_ms": pctile(total_latencies, 0.95),
            "p50_retrieval_ms": pctile(retrieval_latencies, 0.50),
            "p95_retrieval_ms": pctile(retrieval_latencies, 0.95),
        },
        "cost": {
            "total_prompt_tokens": sum(r.get("prompt_tokens", 0) for r in results),
            "total_completion_tokens": sum(r.get("completion_tokens", 0) for r in results),
        },
    }
    return summary


def print_summary(summary: dict):
    print("\n=== EVAL SUMMARY ===")
    print(f"Questions: {summary['num_questions']} ({summary['num_answerable']} answerable, {summary['num_unanswerable']} unanswerable)")
    print("\n-- Retrieval --")
    for k, v in summary["retrieval"].items():
        print(f"  {k}: {v}")
    print("\n-- Answer Quality --")
    for k, v in summary["answer_quality"].items():
        print(f"  {k}: {v}")
    print("\n-- Abstention (unanswerable questions) --")
    print(f"  correctly_abstained_rate: {summary['abstention']['correctly_abstained_rate']}")
    print("\n-- Latency --")
    for k, v in summary["latency"].items():
        print(f"  {k}: {v} ms")
    print("\n-- Token Cost --")
    for k, v in summary["cost"].items():
        print(f"  {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=settings.default_top_k)
    args = parser.parse_args()

    print(f"Running eval with top_k={args.top_k}, mock_mode={settings.mock_mode} ...")
    results = run_eval(args.top_k)
    summary = summarize(results)

    output = {
        "config": {
            "top_k": args.top_k,
            "mock_mode": settings.mock_mode,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model,
            "min_relevance_score": settings.min_relevance_score,
        },
        "summary": summary,
        "per_question": results,
    }
    RESULTS_PATH.write_text(json.dumps(output, indent=2))
    print_summary(summary)
    print(f"\nFull results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
