"""
Simple, deterministic character-based chunker with configurable size and
overlap. Character-based (not token-based) keeps it dependency-light and
fully deterministic, which matters for idempotent re-ingest (same input
text always produces the same chunks -> same hashes -> same IDs).

Splits on paragraph boundaries where possible so chunks stay coherent,
falling back to hard character slicing only when a single paragraph
exceeds chunk_size.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    chunk_index: int  # position within the source document


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [p for p in text.split("\n") if p.strip()]

    chunks: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = (current + "\n" + para).strip() if current else para

        if len(candidate) <= chunk_size:
            current = candidate
            continue

        # Current chunk is full; flush it.
        if current:
            chunks.append(current)

        if len(para) <= chunk_size:
            current = para
        else:
            # Single paragraph longer than chunk_size: hard-slice it with overlap.
            start = 0
            while start < len(para):
                end = start + chunk_size
                chunks.append(para[start:end])
                start = end - chunk_overlap
            current = ""

    if current:
        chunks.append(current)

    # Apply overlap between consecutive chunks (character-level, from the
    # tail of the previous chunk prepended to the next) for chunks that
    # came from the paragraph-packing path above.
    overlapped: list[str] = []
    for i, c in enumerate(chunks):
        if i == 0 or chunk_overlap == 0:
            overlapped.append(c)
        else:
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((tail + "\n" + c).strip())

    return [Chunk(text=c, chunk_index=i) for i, c in enumerate(overlapped)]
