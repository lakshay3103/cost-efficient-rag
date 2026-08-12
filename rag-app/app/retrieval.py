"""
Retrieval: embed a query, fetch top-k chunks from Chroma, and convert
Chroma's cosine *distance* into a cosine *similarity* score so relevance
thresholds are intuitive (1.0 = identical, 0.0 = unrelated).
"""
from dataclasses import dataclass

from app.embeddings import embed_query
from app.config import settings
from app import vectorstore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    filename: str
    chunk_index: int
    similarity: float  # 1 - cosine_distance; higher is more relevant


def retrieve(
    query_text: str,
    top_k: int | None = None,
    source_filter: str | None = None,
) -> list[RetrievedChunk]:
    top_k = top_k or settings.default_top_k
    where = {"source": source_filter} if source_filter else None

    q_embedding = embed_query(query_text)
    results = vectorstore.query(q_embedding, top_k=top_k, where=where)

    chunks: list[RetrievedChunk] = []
    ids = results.get("ids", [[]])[0]
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        similarity = 1.0 - dist  # Chroma cosine space: distance = 1 - similarity
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                text=doc,
                source=meta.get("source", ""),
                filename=meta.get("filename", ""),
                chunk_index=meta.get("chunk_index", -1),
                similarity=round(similarity, 4),
            )
        )
    return chunks


def filter_relevant(chunks: list[RetrievedChunk], min_score: float | None = None) -> list[RetrievedChunk]:
    min_score = min_score if min_score is not None else settings.min_relevance_score
    return [c for c in chunks if c.similarity >= min_score]
