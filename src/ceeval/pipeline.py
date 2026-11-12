"""End-to-end answering: route -> retrieve (or SQL) -> assemble context -> generate.

Configs evaluated
  bm25 / dense / hybrid / hybrid_meta : text retrieval only (every question goes to the docs)
  routed                              : router sends lifecycle questions to the SQL tool and
                                        everything else to hybrid_meta
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import GEN_MODEL, MAX_CONTEXT_WORDS, TOP_K
from .llm import chat
from .retrieval import Hit, Retriever, detect_products
from .sql_tool import text_to_sql

ABSTAIN_TEXT = "I don't know based on the provided sources."
# A refusal is about the *sources* or the assistant's knowledge, not a negative fact about the
# product ("Debian does not include non-free firmware" is an answer, not a refusal).
REFUSAL_RE = re.compile(
    r"(\bi (?:do not|don.?t) know\b|\bi (?:cannot|can.?t) (?:answer|determine|find)\b|"
    r"\b(?:the )?(?:provided |given |available )?(?:sources?|context|documents?|information provided)\b[^.]{0,60}?"
    r"\b(?:do(?:es)? not|do(?:es)?n.?t|is not|isn.?t|are not|aren.?t|never)\b[^.]{0,30}?"
    r"\b(?:mention|contain|provide|include|have|state|specify|say|cover|address|discuss|list)|"
    r"\bno (?:information|mention|details?)\b|"
    r"\bnot (?:stated|mentioned|specified|covered|available|provided|included) (?:in|by) the (?:provided )?(?:sources?|context|documents?)\b|"
    r"\b(?:isn.?t|is not|are not|aren.?t) (?:stated|mentioned|specified|covered) in the (?:provided )?(?:sources?|context|documents?)\b)",
    re.I)
CITATION_RE = re.compile(r"\[\d+\]")

LIFECYCLE_RE = re.compile(
    r"\b(end[- ]of[- ]life|eol|end[- ]of[- ](?:standard |mainstream |active |extended |maintenance )?support|"
    r"support (?:end|ends)|still (?:supported|maintained)|maintenance support|extended (?:update |life ?cycle )?support|"
    r"mainstream|released (?:first|most recently)|release date|when was .+ released|latest patch|"
    r"newest .+ release|release lines?|lts releases?|marked as lts|long-term support releases|codename|"
    r"minor versions|lifecycle data)\b", re.I)

SYSTEM = (
    "You are a support assistant for enterprise infrastructure software. Answer ONLY from the "
    "numbered sources. Cite sources inline like [1]. Be concise: 1-3 sentences. If the sources do "
    f"not contain the answer, reply exactly: {ABSTAIN_TEXT}"
)


def is_abstention(text: str) -> bool:
    """Abstained = the answer declines AND asserts nothing from the sources.

    The system prompt asks for citations on every answer, so a reply that refuses and cites no
    source is an abstention; a reply that answers part of a question with citations and declines
    the rest is an answer (and gets judged for correctness and groundedness)."""
    t = text.strip()
    refuses = ABSTAIN_TEXT.lower() in t.lower() or bool(REFUSAL_RE.search(t))
    return refuses and not CITATION_RE.search(t)


def route(question: str) -> str:
    """Deterministic router: lifecycle facts live in the structured table, everything else in docs."""
    return "sql" if LIFECYCLE_RE.search(question) and detect_products(question) else "text"


def format_context(hits: list[Hit]) -> str:
    parts, used = [], 0
    for i, h in enumerate(hits, 1):
        c = h.chunk
        words = len(c.text.split())
        if used + words > MAX_CONTEXT_WORDS:
            break
        used += words
        parts.append(f"[{i}] {c.product} | {c.section} | source: Wikipedia rev {c.revid} ({c.doc_date})\n{c.text}")
    return "\n\n".join(parts)


def format_rows(sql: str, cols: list[str], rows: list[tuple]) -> str:
    header = " | ".join(cols)
    body = "\n".join(" | ".join("" if v is None else str(v) for v in r) for r in rows)
    return f"[1] Lifecycle database (endoflife.date snapshot) | query: {sql}\n{header}\n{body}"


@dataclass
class Answer:
    qid: str
    config: str
    route: str
    answer: str
    abstained: bool
    context: str
    retrieved_ids: list[str] = field(default_factory=list)
    sql: str | None = None
    sql_rows: list | None = None
    sql_error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    retrieval_ms: float = 0.0
    latency_s: float = 0.0       # end-to-end, excluding judging
    llm_calls: int = 0
    cached: bool = False


class Pipeline:
    def __init__(self, retriever: Retriever | None = None, model: str = GEN_MODEL):
        self.r = retriever or Retriever()
        self.model = model

    def _generate(self, question: str, context: str):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"}]
        return chat(msgs, self.model, max_tokens=250)

    def answer(self, qid: str, question: str, config: str) -> Answer:
        t0 = time.perf_counter()
        r = "sql" if config == "routed" and route(question) == "sql" else "text"
        llm_latency = 0.0  # measured LLM latency (replayed faithfully from cache)
        llm_wall = 0.0     # time actually spent in LLM calls during this invocation
        a = Answer(qid=qid, config=config, route=r, answer="", abstained=False, context="")

        if r == "sql":
            sq = text_to_sql(question, self.model)
            a.sql, a.sql_error, a.llm_calls = sq.sql, sq.error, sq.attempts
            a.input_tokens += sq.input_tokens
            a.output_tokens += sq.output_tokens
            llm_latency += sq.llm_latency_s
            llm_wall += sq.llm_wall_s
            if sq.error or not sq.rows or all(all(v is None for v in row) for row in sq.rows):
                # no data -> abstain without calling the generator (cheaper and safer)
                a.answer, a.abstained = ABSTAIN_TEXT, True
                a.sql_rows = [list(x) for x in sq.rows]
                a.latency_s = (time.perf_counter() - t0 - llm_wall) + llm_latency
                return a
            a.sql_rows = [list(x) for x in sq.rows]
            a.context = format_rows(sq.sql, sq.columns, sq.rows)
        else:
            variant = "hybrid_meta" if config == "routed" else config
            ts = time.perf_counter()
            hits = self.r.search(question, variant, k=TOP_K)
            a.retrieval_ms = (time.perf_counter() - ts) * 1000
            a.retrieved_ids = [h.chunk.chunk_id for h in hits]
            a.context = format_context(hits)

        g = self._generate(question, a.context)
        a.answer = g.text
        a.input_tokens += g.input_tokens
        a.output_tokens += g.output_tokens
        a.llm_calls += 1
        a.cached = g.cached
        llm_latency += g.latency_s
        llm_wall += g.wall_s
        a.abstained = is_abstention(g.text)
        # end-to-end latency = non-LLM wall time (retrieval, SQL, assembly) + measured LLM latency
        a.latency_s = (time.perf_counter() - t0 - llm_wall) + llm_latency
        return a
