"""LLM-as-judge for (1) answer correctness vs. the gold answer and (2) groundedness:
the fraction of the answer's factual claims that are supported by the retrieved context.

Design choices that limit known judge failure modes (see INTERVIEW_NOTES.md):
- temperature 0 + JSON-schema-constrained output: no free-form parsing, reproducible
- groundedness judges against the *retrieved context only*, never the gold answer, so a
  correct-but-unsupported answer (model used its own memory) is still flagged ungrounded
- claim decomposition: the judge must list claims and label each, instead of one holistic score
- the judge is a separate call with no access to the generator's system prompt, and a
  different model family (qwen3) from the generator (gemma4), to avoid self-preference bias
- deterministic keyword accuracy is reported next to judge accuracy, plus their agreement,
  and a sample is exported for human review (scripts/export_review_sample.py)
"""
from __future__ import annotations

import json

from .config import JUDGE_MODEL
from .llm import chat

CORRECT_SCHEMA = {
    "type": "object",
    "properties": {"reasoning": {"type": "string"}, "correct": {"type": "boolean"}},
    "required": ["reasoning", "correct"],
}

GROUND_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {
            "type": "object",
            "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
            "required": ["claim", "supported"]}},
    },
    "required": ["claims"],
}

CORRECT_PROMPT = """You are grading a question-answering system.
Question: {question}
Reference answer: {gold}
System answer: {answer}

Is the system answer correct? It is correct if it states the key facts of the reference answer
(wording may differ; extra correct detail is fine). It is incorrect if it contradicts the
reference, omits a key fact the question asks for, or declines to answer.
Give one short sentence of reasoning, then the verdict."""

GROUND_PROMPT = """You are checking whether an answer is supported by the given sources.

Sources:
{context}

Answer:
{answer}

Split the answer into its atomic factual claims (ignore citation markers like [1] and ignore
phrases that only restate the question). For each claim decide: is it directly supported by the
sources above? Use ONLY the sources, not your own knowledge. A claim that is true in the real
world but not stated in the sources is NOT supported."""


def judge_correct(question: str, gold: str, answer: str, model: str = JUDGE_MODEL) -> dict:
    r = chat([{"role": "user", "content": CORRECT_PROMPT.format(question=question, gold=gold, answer=answer)}],
             model, json_schema=CORRECT_SCHEMA, max_tokens=200)
    try:
        d = json.loads(r.text)
    except json.JSONDecodeError:
        d = {"reasoning": "unparseable judge output", "correct": False}
    return {"correct": bool(d.get("correct")), "reasoning": d.get("reasoning", ""),
            "tokens": r.input_tokens + r.output_tokens}


def judge_grounded(context: str, answer: str, model: str = JUDGE_MODEL) -> dict:
    r = chat([{"role": "user", "content": GROUND_PROMPT.format(context=context, answer=answer)}],
             model, json_schema=GROUND_SCHEMA, max_tokens=400)
    try:
        claims = json.loads(r.text).get("claims", [])
    except json.JSONDecodeError:
        claims = []
    n = len(claims)
    sup = sum(1 for c in claims if c.get("supported"))
    return {"claims": claims, "n_claims": n, "n_supported": sup,
            "groundedness": (sup / n) if n else None,
            "tokens": r.input_tokens + r.output_tokens}
