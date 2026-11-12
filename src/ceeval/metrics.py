"""Metric functions. Pure, deterministic, unit-tested."""
from __future__ import annotations

import re
from datetime import date

import numpy as np

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def expand_alt(alt: str) -> list[str]:
    """An ISO date keyword also matches the common written forms of the same day."""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", alt)
    if not m:
        return [alt]
    d = date(int(m[1]), int(m[2]), int(m[3]))
    mon = MONTHS[d.month - 1]
    return [alt, f"{mon} {d.day}, {d.year}", f"{d.day} {mon} {d.year}", f"{mon} {d.day} {d.year}",
            f"{mon[:3]} {d.day}, {d.year}", f"{d.day} {mon[:3]} {d.year}"]


def _contains(text: str, alt: str) -> bool:
    # Leading word boundary always ('3' must not match '13'). Short or numeric keywords also need
    # a trailing boundary ('no' must not match 'not', '9.0' must not match '9.01'); longer words
    # may take a suffix so 'superstition' matches 'superstitions'.
    a = alt.lower()
    tail = r"(?!\w|\.\d)" if (len(a) <= 3 or re.search(r"\d", a)) else ""
    return re.search(r"(?<![\w.])" + re.escape(a) + tail, text.lower()) is not None


def keyword_match(answer: str, groups: list[list[str]]) -> bool:
    """True iff for every group at least one alternative (or its date variants) appears."""
    return all(any(_contains(answer, v) for alt in g for v in expand_alt(alt)) for g in groups)


def gold_chunk_sets(evidence: list[dict], chunks, norm) -> list[set[str]]:
    """Map each evidence quote to the set of chunk ids containing it (overlap can put a quote
    in two adjacent chunks; either one satisfies that evidence item)."""
    out = []
    for ev in evidence:
        q = norm(ev["quote"])
        out.append({c.chunk_id for c in chunks if c.product_id == ev["product_id"] and q in norm(c.text)})
    return out


def recall_at_k(retrieved: list[str], gold_sets: list[set[str]], k: int) -> float:
    """Fraction of required evidence items with at least one gold chunk in the top-k."""
    top = set(retrieved[:k])
    return sum(1 for g in gold_sets if g & top) / len(gold_sets)


def all_found_at_k(retrieved: list[str], gold_sets: list[set[str]], k: int) -> bool:
    top = set(retrieved[:k])
    return all(g & top for g in gold_sets)


def reciprocal_rank(retrieved: list[str], gold_sets: list[set[str]]) -> float:
    gold = set().union(*gold_sets)
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def pctl(xs: list[float], p: float) -> float:
    return float(np.percentile(xs, p)) if xs else float("nan")


def paired_bootstrap(a: list[float], b: list[float], n: int = 10000, seed: int = 0) -> dict:
    """Mean of (b - a) over paired questions with a 95% bootstrap CI. If the CI contains 0 the
    difference is not distinguishable from noise at this sample size."""
    x = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"diff": round(float(x.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n": len(x), "significant": bool(lo > 0 or hi < 0)}
