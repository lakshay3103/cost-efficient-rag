# Cost-Efficient RAG Application

A QA service over a document corpus backed by a low-cost, self-hosted vector store (ChromaDB), with honest, measured evaluation of retrieval quality, answer quality, latency, and cost — built as an alternative to a fully managed vector DB.

## Why ChromaDB

ChromaDB was chosen over pgvector, Qdrant, LanceDB, FAISS, and sqlite-vec for this assignment because:

- **Embeddable, zero-ops**: runs in-process with a persistent local directory — no separate database server to provision, unlike pgvector (needs Postgres) or self-hosted Qdrant (needs its own server process). This matters directly for the "low infra cost" premise of the assignment: there's no always-on server bill beyond the app's own VM.
- **Native metadata filtering**: unlike FAISS (vectors only, no metadata store — you'd have to build filtering yourself) or sqlite-vec (filtering is possible via SQL but more manual), Chroma supports `where` filters on metadata out of the box, which the assignment requires ("at least one metadata filter").
- **Good-enough ANN performance at the target scale**: Chroma uses HNSW under the hood, which is fine up to a few million vectors on a single node — the scale most lightly-queried, cost-sensitive workloads actually operate at. FAISS would be faster at very large scale but trades away the metadata/filtering convenience.
- **Idempotent upserts by ID**: Chroma's `upsert` (insert-or-replace by ID) makes deterministic-ID-based idempotent re-ingestion straightforward, which the assignment explicitly requires.

The trade-off: Chroma is single-node (no built-in sharding/replication), so at very large scale (see the 10M-vector cost table below) a managed service's ability to shard across pods starts to win on raw infra cost. That crossover is the central finding of the cost analysis.

## Architecture

```
app/
  config.py       - all tunables, env-driven, no hardcoded secrets
  loaders.py       - PDF / HTML / MD -> plain text
  chunking.py       - deterministic character-based chunker (size + overlap)
  embeddings.py     - OpenAI embeddings wrapper (+ mock mode for offline testing)
  vectorstore.py    - ChromaDB wrapper; deterministic chunk IDs for idempotency
  ingest.py         - ties loaders+chunking+embeddings+store together, delta-only re-ingest
  retrieval.py      - query embedding -> top-k search -> relevance filtering
  generation.py     - grounded LLM answer with inline citations, JSON-structured output
  query_log.py      - structured per-query logging (latency, chunk count, tokens)
  main.py           - FastAPI app: /ingest, /query, /health, /stats
eval/
  questions.json         - fixed 22-question eval set (19 answerable + 3 unanswerable)
  retrieval_metrics.py    - Hit Rate, Recall@k, MRR, nDCG@k, context precision
  answer_metrics.py       - EM/F1 + LLM-as-judge faithfulness/relevance
  run_eval.py             - harness: runs questions through the live pipeline, writes results.json
  cost_analysis.py         - self-hosted vs managed cost model at 100K/1M/10M vectors
scripts/
  ingest_cli.py     - CLI alternative to the /ingest endpoint
```

## Setup

```bash
cd rag-app
pip install -r requirements.txt --break-system-packages   # or use a venv
cp .env.example .env
# edit .env: set OPENAI_API_KEY, and set MOCK_MODE=false for real use
```

### Ingest the sample corpus

```bash
python scripts/ingest_cli.py data/corpus
```

The sample corpus (`data/corpus/`) is three Markdown documents (Python language facts, Transformer architecture facts, oil refinery process facts) written with clear, verifiable factual content so the eval question set has unambiguous gold answers. Swap in your own PDF/HTML/MD files — ingestion handles all three.

### Run the API

```bash
uvicorn app.main:app --reload
```

- `POST /ingest {"path": "data/corpus"}` — ingest a file or directory
- `POST /query {"question": "...", "top_k": 5}` — ask a question
- `GET /health`, `GET /stats`

### Run the evaluation harness

```bash
python eval/run_eval.py --top-k 5
```

Writes `eval/results.json` (full per-question + summary) and prints a summary to stdout.

### Run the cost analysis

```bash
python eval/cost_analysis.py
```

## Important note on the numbers in this repo: MOCK_MODE

**This project was built and tested in a sandboxed environment with no network access to `api.openai.com`** (only package registries were reachable). To let every stage of the pipeline — chunking, idempotent ingestion, retrieval, relevance filtering, grounded generation, the eval harness, and the API — be built and verified end-to-end anyway, `app/embeddings.py` and `app/generation.py` include a `MOCK_MODE` path (`MOCK_MODE=true` in `.env`) that swaps in deterministic fake embeddings and canned LLM responses instead of real API calls.

**What mock mode proves:** all the plumbing is correct — idempotent re-ingest was verified to produce zero new embeddings on an unchanged re-run and to only re-embed the delta on a changed file; the no-hallucination path was verified to correctly abstain when nothing clears the relevance threshold; the eval harness was verified to compute every metric and produce a valid results file.

**What mock mode cannot prove:** actual retrieval/answer quality, since fake embeddings carry no real semantic signal (this shows up as near-chance retrieval metrics in `eval/results.json` if you inspect it — that's expected and not a bug).

**To get real numbers**: set `MOCK_MODE=false` and a real `OPENAI_API_KEY` in `.env`, then re-run ingestion and `eval/run_eval.py`. Everything is wired to switch over with no code changes.

## Evaluation Methodology

### Question set (`eval/questions.json`)

22 fixed questions: 19 answerable (7 on Python, 6 on Transformers, 6 on oil refining) plus 3 deliberately unanswerable ones (out-of-corpus facts) to test the abstention path. Each answerable question has a `gold_keyphrase` — a short, distinctive substring that only appears in the passage that actually answers the question. At eval time, the harness looks up which actual stored chunks (for the question's source file) contain that key-phrase, giving ground-truth relevant-chunk IDs that stay valid regardless of chunk-size/overlap configuration — so you can re-run the eval after changing chunking settings without hand-relabeling relevance.

### Retrieval metrics (`eval/retrieval_metrics.py`)

Standard binary-relevance IR metrics computed per question then averaged: **Hit Rate@k**, **Recall@k**, **MRR**, **nDCG@k**, **context precision@k** (fraction of retrieved chunks that are actually relevant).

### Answer metrics (`eval/answer_metrics.py`)

- **EM/F1** against the gold answer (token-overlap based — a rough automatic signal, since free-form answers rarely exact-match).
- **LLM-as-judge faithfulness + relevance**, 1–5 scale with a rationale. Faithfulness asks whether every claim in the answer is actually supported by the retrieved context (not just whether it's true); relevance asks whether the answer addresses the question. *Note on judge validity*: this eval currently uses the same model family (`gpt-4o-mini` in both the generator and the judge role by default) for simplicity — see Problem 2's self-enhancement-bias concern. For a real deployment gate, swap the judge to a different model family (e.g. a Claude model) via `JUDGE_MODEL` to avoid self-preference bias; this is a known limitation of the current setup, called out here rather than glossed over.

### Latency

p50/p95 retrieval and total (retrieval + generation) latency across the question set, from `eval/run_eval.py`.

### Cost (`eval/cost_analysis.py`)

Self-hosted Chroma cost = VM tier sized for the vector count + persistent SSD storage for the HNSW index (raw vector bytes × 1.5 overhead factor for the graph index, a typical HNSW rule of thumb). Managed cost = pod-based pricing modeled loosely on published Pinecone-style pod pricing (~5M vectors/pod at 1536 dims, ~$80/pod/month), billed whether queried or not — this is exactly the "always-on pods" cost driver the assignment background describes.

| Vectors | Self-hosted ($/mo) | Managed ($/mo) | Savings |
|---:|---:|---:|---:|
| 100,000 | $15.09 | $80.00 | 81.1% |
| 1,000,000 | $55.86 | $80.00 | 30.2% |
| 10,000,000 | $228.58 | $160.00 | -42.9% (managed wins) |

All figures are stated estimates with explicit assumptions (see `eval/cost_analysis.py` for the exact tiering logic) — meant to demonstrate the shape of the trade-off, not to be an exact vendor quote.

## Discussion

**When would you switch back to managed?** The cost table shows the crossover directly: self-hosted Chroma is dramatically cheaper at 100K vectors (a single small VM easily holds the index) and still meaningfully cheaper at 1M, but by 10M vectors a single-node Chroma deployment needs a much larger, more expensive VM to hold the HNSW graph in memory, while managed pod pricing scales more gracefully because it shards across pods. Beyond that, you'd also switch back to managed sooner than the raw cost crossover suggests if you need: multi-region replication/high availability (Chroma is single-node by default), horizontal query throughput scaling under heavy concurrent load, or you want to remove vector-DB operations entirely from your team's on-call burden — the "low-cost" story here explicitly trades infra dollars for some ops responsibility.

**Was retrieval or generation the weak link?** In this codebase's honest assessment: it cannot be answered from the numbers currently in `eval/results.json`, because those numbers were produced in `MOCK_MODE` due to the sandbox's lack of network access to the OpenAI API (see above) — the retrieval metrics there reflect random fake embeddings, not real semantic retrieval. The harness and metrics are fully built and verified to work correctly end-to-end; running `eval/run_eval.py` with `MOCK_MODE=false` and a real API key will produce the real answer. As a structural prediction based on the corpus design: with only ~13 chunks across 3 short documents and one dominant key-phrase per question, retrieval should score very highly (this is a favorable, low-ambiguity retrieval setting) — so generation quality (faithfulness in particular) is the more likely source of any real-run imperfection, especially since the grounding prompt is strict about not answering beyond the provided context.

## Verified Behaviors (from actual test runs in this repo, mock mode)

- **Idempotent re-ingest**: running `ingest_cli.py` twice on an unchanged corpus produces `new_or_changed_chunks=0` on the second run for every file, with the vector count unchanged (13 vectors before and after).
- **Delta-only re-ingest on edit**: appending a new section to one file and re-ingesting produced exactly 1 new/changed chunk and 1 deleted stale chunk, leaving the other 3 chunks of that file untouched and the total vector count unchanged (13) — confirming no duplicate vectors are ever created.
- **No-hallucination path**: when no retrieved chunk clears `MIN_RELEVANCE_SCORE`, the API returns the fixed "I don't have enough relevant information..." message with `grounded: false` rather than calling the LLM to guess, and the eval harness's `correctly_abstained_rate` on the 3 unanswerable questions was 1.0.
