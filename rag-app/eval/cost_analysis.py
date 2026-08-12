#!/usr/bin/env python3
"""
Cost-comparison model: self-hosted ChromaDB (on a small persistent VM)
vs. a fully managed vector DB (Pinecone-style pod-based pricing), at
100K / 1M / 10M vectors.

All numbers are estimates with stated assumptions -- meant to show the
shape of the cost curve and the crossover logic, not to be exact
production quotes. Sources for the managed-DB assumptions are in the
README's Discussion section.

Usage:
    python eval/cost_analysis.py
"""
import json
from pathlib import Path

# ---- Assumptions (stated explicitly per assignment requirements) ----

EMBEDDING_DIM = 1536          # text-embedding-3-small
BYTES_PER_FLOAT32 = 4
HNSW_OVERHEAD_FACTOR = 1.5    # graph index overhead on top of raw vector bytes, typical for HNSW

# Self-hosted Chroma: single small persistent VM, always-on, running Chroma
# as an embedded/server process. Assume a cloud VM (e.g. e2-small class)
# sized to hold the index in memory/disk comfortably at each scale.
SELF_HOSTED_VM_TIERS = [
    # (max_vectors_supported, monthly_cost_usd, vm_description)
    (200_000, 15, "2 vCPU / 4GB RAM VM (e.g. e2-small)"),
    (2_000_000, 55, "4 vCPU / 16GB RAM VM (e.g. e2-standard-4)"),
    (15_000_000, 220, "8 vCPU / 64GB RAM VM (e.g. e2-highmem-8)"),
]
STORAGE_COST_PER_GB_MONTH = 0.10  # persistent SSD disk, typical cloud pricing

# Managed vector DB: pod-based pricing where cost scales with pods
# provisioned to hold the vector count, billed whether queried or not.
# Modeled loosely on published Pinecone pod pricing (s1 pods, ~5M vectors
# capacity per pod at 1536 dims, ~$70-90/mo per pod depending on pod type).
MANAGED_VECTORS_PER_POD = 5_000_000
MANAGED_COST_PER_POD_MONTH = 80


def self_hosted_monthly_cost(num_vectors: int) -> tuple[float, str]:
    raw_bytes = num_vectors * EMBEDDING_DIM * BYTES_PER_FLOAT32
    index_bytes = raw_bytes * HNSW_OVERHEAD_FACTOR
    index_gb = index_bytes / (1024 ** 3)

    for max_vecs, vm_cost, desc in SELF_HOSTED_VM_TIERS:
        if num_vectors <= max_vecs:
            storage_cost = index_gb * STORAGE_COST_PER_GB_MONTH
            return round(vm_cost + storage_cost, 2), desc
    # Beyond largest tier, extrapolate.
    largest_vm_cost = SELF_HOSTED_VM_TIERS[-1][1]
    storage_cost = index_gb * STORAGE_COST_PER_GB_MONTH
    return round(largest_vm_cost + storage_cost, 2), "extrapolated beyond largest tier"


def managed_monthly_cost(num_vectors: int) -> float:
    pods_needed = max(1, -(-num_vectors // MANAGED_VECTORS_PER_POD))  # ceil division
    return pods_needed * MANAGED_COST_PER_POD_MONTH


def build_table():
    scales = [100_000, 1_000_000, 10_000_000]
    rows = []
    for n in scales:
        sh_cost, sh_desc = self_hosted_monthly_cost(n)
        mg_cost = managed_monthly_cost(n)
        savings_pct = round((1 - sh_cost / mg_cost) * 100, 1) if mg_cost else 0
        rows.append(
            {
                "num_vectors": n,
                "self_hosted_monthly_usd": sh_cost,
                "self_hosted_config": sh_desc,
                "managed_monthly_usd": mg_cost,
                "savings_pct": savings_pct,
            }
        )
    return rows


def print_table(rows):
    print(f"{'Vectors':>12} | {'Self-hosted ($/mo)':>19} | {'Managed ($/mo)':>15} | {'Savings':>8}")
    print("-" * 65)
    for r in rows:
        print(
            f"{r['num_vectors']:>12,} | {r['self_hosted_monthly_usd']:>19,.2f} | "
            f"{r['managed_monthly_usd']:>15,.2f} | {r['savings_pct']:>7.1f}%"
        )


if __name__ == "__main__":
    rows = build_table()
    print_table(rows)
    out_path = Path(__file__).parent / "cost_comparison.json"
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nWritten to {out_path}")
