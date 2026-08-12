"""
Generation: given a question and retrieved chunks, produce a grounded
answer that cites which chunks it used, or an explicit "no relevant
context" response if nothing retrieved clears the relevance bar (never
falls back to the model's own unsourced knowledge).
"""
import json
from dataclasses import dataclass, field

from openai import OpenAI
from app.config import settings
from app.retrieval import RetrievedChunk

_client = None if settings.mock_mode else OpenAI(api_key=settings.openai_api_key)

NO_CONTEXT_MESSAGE = (
    "I don't have enough relevant information in the indexed documents to "
    "answer that question."
)

SYSTEM_PROMPT = """You are a QA assistant that answers ONLY using the provided context chunks.

Rules:
- Answer strictly from the given context. Do not use outside knowledge.
- Every factual claim in your answer must be traceable to a specific chunk.
- Cite chunks inline using their bracketed number, e.g. [1], [2].
- If the context does not contain enough information to answer the question, say so explicitly instead of guessing.
- Respond ONLY with a JSON object of the form:
  {"answer": "<answer text with inline [n] citations>", "cited_chunk_numbers": [1, 3], "grounded": true}
  Set "grounded" to false if you could not answer from the context.
No markdown, no preamble, JSON only."""


@dataclass
class GeneratedAnswer:
    answer: str
    grounded: bool
    cited_chunks: list[RetrievedChunk] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(f"[{i}] (source: {c.filename})\n{c.text}")
    return "\n\n".join(parts)


def _mock_generate(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    """Canned response path for MOCK_MODE, so the API/eval harness can be
    exercised without a live OpenAI call."""
    if not chunks:
        return GeneratedAnswer(answer=NO_CONTEXT_MESSAGE, grounded=False)
    snippet = chunks[0].text[:120].replace("\n", " ")
    return GeneratedAnswer(
        answer=f"[MOCK ANSWER] Based on [1]: {snippet}...",
        grounded=True,
        cited_chunks=[chunks[0]],
        prompt_tokens=0,
        completion_tokens=0,
    )


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> GeneratedAnswer:
    if not chunks:
        return GeneratedAnswer(answer=NO_CONTEXT_MESSAGE, grounded=False)

    if settings.mock_mode:
        return _mock_generate(question, chunks)

    context_block = _build_context_block(chunks)
    user_prompt = f"Context:\n{context_block}\n\nQuestion: {question}"

    response = _client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    usage = response.usage

    try:
        parsed = json.loads(raw)
        answer_text = parsed.get("answer", NO_CONTEXT_MESSAGE)
        grounded = bool(parsed.get("grounded", False))
        cited_numbers = parsed.get("cited_chunk_numbers", [])
        cited = [chunks[n - 1] for n in cited_numbers if 1 <= n <= len(chunks)]
    except (json.JSONDecodeError, KeyError, TypeError):
        # Malformed JSON from the model: fail safe rather than hallucinate structure.
        answer_text = raw.strip() if raw else NO_CONTEXT_MESSAGE
        grounded = False
        cited = []

    return GeneratedAnswer(
        answer=answer_text,
        grounded=grounded,
        cited_chunks=cited,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
    )
