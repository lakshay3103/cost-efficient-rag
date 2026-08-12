"""
FastAPI service exposing:
  POST /ingest       - ingest a file or directory path already on disk
  POST /query         - ask a question, get a grounded answer with citations
  GET  /health        - basic health/status check
  GET  /stats          - vector count etc.

Config is entirely env-driven (see app/config.py); no secrets in code.
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.ingest import ingest_directory, ingest_file
from app.retrieval import retrieve, filter_relevant
from app.generation import generate_answer
from app.query_log import QueryLogEntry, log_query, Timer
from app import vectorstore

app = FastAPI(title="Cost-Efficient RAG Service", version="1.0.0")


# ---------- Schemas ----------

class IngestRequest(BaseModel):
    path: str = Field(..., description="File or directory path on disk to ingest")
    chunk_size: int | None = None
    chunk_overlap: int | None = None


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = Field(default=None, description="Overrides DEFAULT_TOP_K")
    source_filter: str | None = Field(
        default=None, description="Restrict retrieval to chunks from this source path (metadata filter)"
    )


class CitedChunkResponse(BaseModel):
    chunk_id: str
    filename: str
    chunk_index: int
    similarity: float
    text_preview: str


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    cited_chunks: list[CitedChunkResponse]
    retrieved_chunk_count: int
    relevant_chunk_count: int
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    prompt_tokens: int
    completion_tokens: int


# ---------- Endpoints ----------

@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": settings.mock_mode}


@app.get("/stats")
def stats():
    return {"vector_count": vectorstore.count(), "collection": settings.chroma_collection_name}


@app.post("/ingest")
def ingest(req: IngestRequest):
    path = Path(req.path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    if path.is_dir():
        results = ingest_directory(path, req.chunk_size, req.chunk_overlap)
    else:
        results = [ingest_file(path, req.chunk_size, req.chunk_overlap)]

    return {"ingested_files": len(results), "results": [r.__dict__ for r in results]}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    top_k = req.top_k or settings.default_top_k

    with Timer() as retrieval_timer:
        retrieved = retrieve(req.question, top_k=top_k, source_filter=req.source_filter)
        relevant = filter_relevant(retrieved)

    with Timer() as generation_timer:
        result = generate_answer(req.question, relevant)

    total_ms = retrieval_timer.elapsed_ms + generation_timer.elapsed_ms

    log_query(
        QueryLogEntry(
            timestamp=__import__("time").time(),
            question=req.question,
            top_k=top_k,
            retrieved_chunk_count=len(retrieved),
            relevant_chunk_count=len(relevant),
            grounded=result.grounded,
            retrieval_latency_ms=retrieval_timer.elapsed_ms,
            generation_latency_ms=generation_timer.elapsed_ms,
            total_latency_ms=total_ms,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
    )

    return QueryResponse(
        answer=result.answer,
        grounded=result.grounded,
        cited_chunks=[
            CitedChunkResponse(
                chunk_id=c.chunk_id,
                filename=c.filename,
                chunk_index=c.chunk_index,
                similarity=c.similarity,
                text_preview=c.text[:200],
            )
            for c in result.cited_chunks
        ],
        retrieved_chunk_count=len(retrieved),
        relevant_chunk_count=len(relevant),
        retrieval_latency_ms=retrieval_timer.elapsed_ms,
        generation_latency_ms=generation_timer.elapsed_ms,
        total_latency_ms=total_ms,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
    )
