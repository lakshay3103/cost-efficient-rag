"""
Answer-quality evaluation:
  - EM/F1 against gold answers (token-overlap based; rough for free-form
    text but gives a cheap automatic signal alongside the judge).
  - Faithfulness (is every claim in the answer supported by the cited
    context?) and relevance (does the answer actually address the
    question?) via LLM-as-judge, each scored 1-5 with a rationale.

The judge is a separate call from the generator call, and in mock mode
returns deterministic canned scores so the harness can be exercised
without live API access.
"""
import json
import re
import string
from dataclasses import dataclass

from openai import OpenAI
from app.config import settings

_client = None if settings.mock_mode else OpenAI(api_key=settings.openai_api_key)

JUDGE_MODEL = settings.llm_model  # Note: ideally a different model family than the generator; see README discussion.


def _normalize(text: str) -> list[str]:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return text.split()


def exact_match(prediction: str, gold: str) -> int:
    return int(_normalize(prediction) == _normalize(gold))


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize(prediction)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = {}
    for t in pred_tokens:
        common[t] = min(pred_tokens.count(t), gold_tokens.count(t))
    num_same = sum(common.get(t, 0) for t in set(pred_tokens) & set(gold_tokens))
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


JUDGE_SYSTEM_PROMPT = """You are an evaluation judge scoring a QA system's answer.

Score two dimensions, each 1-5 (5 = best):
- faithfulness: is every claim in the answer actually supported by the provided context? (Penalize any unsupported claim heavily, even if the claim happens to be true.)
- relevance: does the answer actually address the question asked?

Respond ONLY with JSON: {"faithfulness": <1-5>, "relevance": <1-5>, "rationale": "<one sentence>"}"""


@dataclass
class JudgeVerdict:
    faithfulness: int
    relevance: int
    rationale: str


def _mock_judge(question: str, context: str, answer: str) -> JudgeVerdict:
    # Deterministic canned verdict for offline/no-network testing.
    if "MOCK ANSWER" in answer:
        return JudgeVerdict(faithfulness=4, relevance=3, rationale="Mock mode: canned plausible score.")
    return JudgeVerdict(faithfulness=1, relevance=1, rationale="Mock mode: no context, correctly abstained.")


def judge_answer(question: str, context: str, answer: str) -> JudgeVerdict:
    if settings.mock_mode:
        return _mock_judge(question, context, answer)

    user_prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer to evaluate:\n{answer}"
    response = _client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        return JudgeVerdict(
            faithfulness=int(parsed.get("faithfulness", 1)),
            relevance=int(parsed.get("relevance", 1)),
            rationale=parsed.get("rationale", ""),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return JudgeVerdict(faithfulness=1, relevance=1, rationale="[unparseable judge response]")
