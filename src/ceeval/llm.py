"""Thin LLM client. Default backend: a local model served by Ollama (zero API cost).

Every call returns text plus the accounting the eval needs: input/output tokens and wall-clock
latency. Responses are cached on disk keyed by (model, messages, options) so re-running the
eval (e.g. after changing only a metric) does not re-spend compute; cached calls keep the
latency that was measured when they actually ran.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass

import requests

from .config import OLLAMA_URL, PROCESSED

CACHE_DIR = PROCESSED / "llm_cache"


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float      # latency measured when the call actually ran
    cached: bool = False
    wall_s: float = 0.0   # time this invocation took (≈0 on a cache hit)


def chat(messages: list[dict], model: str, *, json_schema: dict | None = None,
         max_tokens: int = 300, use_cache: bool = True) -> LLMResult:
    options = {"temperature": 0, "seed": 7, "num_ctx": 4096, "num_predict": max_tokens}
    payload = {"model": model, "messages": messages, "stream": False, "think": False,
               "options": options, "keep_alive": "30m"}
    if json_schema is not None:
        payload["format"] = json_schema
    key = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    if use_cache and path.exists():
        t0 = time.perf_counter()
        d = json.loads(path.read_text())
        return LLMResult(d["text"], d["input_tokens"], d["output_tokens"], d["latency_s"], cached=True,
                         wall_s=time.perf_counter() - t0)

    t0 = time.perf_counter()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=600)
    r.raise_for_status()
    latency = time.perf_counter() - t0
    d = r.json()
    res = LLMResult(d["message"]["content"].strip(), d.get("prompt_eval_count", 0),
                    d.get("eval_count", 0), latency, wall_s=latency)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: v for k, v in res.__dict__.items() if k not in ("cached", "wall_s")}
                               | {"model": model}))
    return res


def warmup(model: str) -> None:
    """Load weights before timing anything, so the first query's latency isn't a model load.
    Best effort: if the model isn't installed, cached replays still work and uncached calls
    will raise a clear error from chat()."""
    try:
        _warm(model)
    except requests.RequestException:
        pass


def _warm(model: str) -> None:
    requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model, "messages": [{"role": "user", "content": "ok"}], "stream": False,
        "think": False, "keep_alive": "30m", "options": {"num_ctx": 4096, "num_predict": 1}},
        timeout=600).raise_for_status()
