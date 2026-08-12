"""
Standard IR metrics computed against binary relevance judgments.

Gold relevance for this eval set is derived automatically: a chunk is
"relevant" to a question if it belongs to the question's designated
source file AND contains the question's gold key-phrase (a short,
distinctive substring from the source document that only appears in the
passage that actually answers the question). This is recomputed against
whatever chunks are actually in the store, so it stays valid across
different chunk-size/overlap configurations.
"""
import math


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> int:
    return 1 if any(rid in relevant_ids for rid in retrieved_ids) else 0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(relevant_ids)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    dcg = 0.0
    for rank, rid in enumerate(retrieved_ids, start=1):
        rel = 1 if rid in relevant_ids else 0
        if rel:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: all relevant docs (up to k) ranked first.
    ideal_hits = min(len(relevant_ids), len(retrieved_ids))
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


def context_precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hit = len(set(retrieved_ids) & relevant_ids)
    return hit / len(retrieved_ids)
