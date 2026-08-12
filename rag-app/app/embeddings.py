"""
Thin wrapper around the OpenAI embeddings endpoint. Batches requests to
stay efficient and keeps the model/dimensionality in one place so it's
easy to swap or benchmark a second embedding model later.
"""
import hashlib
import numpy as np
from openai import OpenAI
from app.config import settings

BATCH_SIZE = 100  # OpenAI allows up to 2048 inputs/call; keep batches modest.

_client = None if settings.mock_mode else OpenAI(api_key=settings.openai_api_key)


def _mock_embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding derived from text content, used only
    when MOCK_MODE=true (no network access to OpenAI). Same text always
    produces the same vector, and textually similar inputs are NOT
    guaranteed to be semantically close -- this is for exercising the
    pipeline's plumbing (chunking/storage/retrieval/API), not for real
    retrieval quality testing."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=settings.embedding_dimensions)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts, batching to respect API limits."""
    if not texts:
        return []

    if settings.mock_mode:
        return [_mock_embed(t) for t in texts]

    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = _client.embeddings.create(
            model=settings.embedding_model,
            input=batch,
        )
        # API returns embeddings in the same order as input.
        all_embeddings.extend([item.embedding for item in response.data])
    return all_embeddings


def embed_query(text: str) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text])[0]
