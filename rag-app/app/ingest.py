"""
End-to-end ingestion pipeline for a single file or a directory of files.

Idempotency strategy:
1. Compute deterministic chunk IDs (hash of source path + chunk index + chunk text).
2. Look up which IDs already exist in the collection for this source file.
3. Only embed chunks whose ID is NOT already present (new or changed content).
4. Delete any existing IDs for this source that are no longer produced
   (handles shrinking/edited documents so stale chunks don't linger).
5. Upsert the (new/changed) embeddings.

Net effect: re-running ingest on an unchanged file does zero embedding
calls and zero writes. Re-running on a changed file only pays for the
delta.
"""
import time
from dataclasses import dataclass
from pathlib import Path

from app.loaders import load_document
from app.chunking import chunk_text
from app.embeddings import embed_texts
from app.config import settings
from app import vectorstore


@dataclass
class IngestResult:
    source: str
    total_chunks: int
    new_or_changed_chunks: int
    skipped_unchanged_chunks: int
    deleted_stale_chunks: int
    elapsed_seconds: float


def ingest_file(path: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> IngestResult:
    start = time.time()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    source = str(path)
    text = load_document(path)
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    computed_ids = [
        vectorstore.make_chunk_id(source, c.chunk_index, c.text) for c in chunks
    ]

    collection = vectorstore.get_collection()
    existing = collection.get(where={"source": source}, include=[])
    existing_ids = set(existing["ids"]) if existing and existing.get("ids") else set()

    new_ids, new_texts, new_metas = [], [], []
    for cid, chunk in zip(computed_ids, chunks):
        if cid not in existing_ids:
            new_ids.append(cid)
            new_texts.append(chunk.text)
            new_metas.append(
                {
                    "source": source,
                    "chunk_index": chunk.chunk_index,
                    "filename": path.name,
                }
            )

    # Stale IDs: existed before but not produced by this ingest run.
    stale_ids = list(existing_ids - set(computed_ids))
    if stale_ids:
        collection.delete(ids=stale_ids)

    if new_ids:
        embeddings = embed_texts(new_texts)
        vectorstore.upsert_chunks(new_ids, embeddings, new_texts, new_metas)

    elapsed = time.time() - start
    return IngestResult(
        source=source,
        total_chunks=len(chunks),
        new_or_changed_chunks=len(new_ids),
        skipped_unchanged_chunks=len(chunks) - len(new_ids),
        deleted_stale_chunks=len(stale_ids),
        elapsed_seconds=round(elapsed, 3),
    )


def ingest_directory(directory: Path, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[IngestResult]:
    supported = {".pdf", ".html", ".htm", ".md", ".markdown"}
    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in supported)
    return [ingest_file(p, chunk_size, chunk_overlap) for p in files]
