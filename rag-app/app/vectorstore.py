"""
Wrapper around a persistent ChromaDB collection.

Idempotency: each chunk's ID is a deterministic hash of
(source_path, chunk_index, chunk_text). Re-ingesting an unchanged file
produces the exact same IDs, and Chroma's `upsert` overwrites existing
IDs rather than duplicating them -- so re-running ingestion is a no-op
for unchanged content, and only changed/added chunks get re-embedded
and written. Deleted/changed chunks from a prior version of a file are
swept by first deleting all existing IDs for that source path.
"""
import hashlib
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.config import settings

_client = chromadb.PersistentClient(
    path=settings.chroma_persist_dir,
    settings=ChromaSettings(anonymized_telemetry=False),
)


def get_collection():
    return _client.get_or_create_collection(
        name=settings.chroma_collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def make_chunk_id(source_path: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha256()
    h.update(source_path.encode("utf-8"))
    h.update(str(chunk_index).encode("utf-8"))
    h.update(chunk_text.encode("utf-8"))
    return h.hexdigest()[:32]


def delete_by_source(source_path: str) -> None:
    """Remove all existing chunks for a given source file before re-ingest,
    so stale chunks (e.g. from a shrunk document) don't linger."""
    collection = get_collection()
    collection.delete(where={"source": source_path})


def upsert_chunks(
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict],
) -> None:
    collection = get_collection()
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )


def query(
    query_embedding: list[float],
    top_k: int,
    where: dict | None = None,
) -> dict:
    collection = get_collection()
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
    )


def count() -> int:
    return get_collection().count()
