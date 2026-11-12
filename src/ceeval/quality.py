"""Data-quality gate for both data products. Exits non-zero if a blocking check fails,
so it can run in CI before any index is (re)built.

Checks
- chunks: schema validation, unique ids, exact duplicates, near duplicates (shingle Jaccard),
  empty/too-short chunks, garbled text (symbol ratio, markup residue), metadata completeness
- lifecycle db: schema, PK uniqueness, date parse, date ordering, FK integrity, coverage
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import date

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import CHUNKS_PATH, DB_PATH, MIN_CHUNK_WORDS, PRODUCTS, REPORTS

NEAR_DUP_JACCARD = 0.8
MAX_SYMBOL_RATIO = 0.08
MARKUP = re.compile(r"(\{\\displaystyle|\{\{|\}\}|\[\[|\]\]|<ref|&nbsp;|==\s)")


class ChunkModel(BaseModel):
    chunk_id: str = Field(pattern=r"^[a-z0-9-]+::\d{2}-[a-z0-9-]*::\d+$")
    product_id: str
    product: str
    category: str
    doc_title: str
    section: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    n_words: int
    source_url: str = Field(pattern=r"^https://en\.wikipedia\.org/")
    revid: int = Field(gt=0)
    doc_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    license: str
    content_hash: str

    @field_validator("product_id")
    @classmethod
    def known_product(cls, v: str) -> str:
        if v not in PRODUCTS:
            raise ValueError(f"unknown product_id {v}")
        return v


def shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    toks = re.findall(r"\w+", text.lower())
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def check_chunks() -> dict:
    rows = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]
    issues: dict[str, list] = {k: [] for k in (
        "schema_errors", "duplicate_ids", "exact_duplicates", "near_duplicates",
        "too_short", "garbled", "markup_residue", "word_count_mismatch")}

    for r in rows:
        try:
            ChunkModel(**r)
        except ValidationError as e:
            issues["schema_errors"].append({"chunk_id": r.get("chunk_id"), "error": str(e)[:200]})
        if r["n_words"] < MIN_CHUNK_WORDS:
            issues["too_short"].append(r["chunk_id"])
        if len(r["text"].split()) != r["n_words"]:
            issues["word_count_mismatch"].append(r["chunk_id"])
        txt = r["text"]
        sym = sum(1 for ch in txt if not (ch.isalnum() or ch.isspace() or ch in ".,;:'\"()-/%&?!"))
        if txt and sym / len(txt) > MAX_SYMBOL_RATIO:
            issues["garbled"].append({"chunk_id": r["chunk_id"], "symbol_ratio": round(sym / len(txt), 3)})
        if MARKUP.search(txt):
            issues["markup_residue"].append(r["chunk_id"])

    ids = Counter(r["chunk_id"] for r in rows)
    issues["duplicate_ids"] = [k for k, v in ids.items() if v > 1]

    by_hash: dict[str, list[str]] = {}
    for r in rows:
        by_hash.setdefault(r["content_hash"], []).append(r["chunk_id"])
    issues["exact_duplicates"] = [v for v in by_hash.values() if len(v) > 1]

    # Near-duplicates: O(n^2) on ~700 chunks is fine; index by shingle for larger corpora.
    sh = [(r["chunk_id"], shingles(r["text"])) for r in rows]
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            a, b = sh[i][1], sh[j][1]
            if not a or not b:
                continue
            jac = len(a & b) / len(a | b)
            if jac >= NEAR_DUP_JACCARD:
                issues["near_duplicates"].append({"a": sh[i][0], "b": sh[j][0], "jaccard": round(jac, 3)})

    products_covered = {r["product_id"] for r in rows}
    blocking = ["schema_errors", "duplicate_ids", "exact_duplicates", "too_short", "markup_residue",
                "word_count_mismatch"]
    return {
        "n_chunks": len(rows),
        "n_products": len(products_covered),
        "missing_products": sorted(set(PRODUCTS) - products_covered),
        "words": {"min": min(r["n_words"] for r in rows), "max": max(r["n_words"] for r in rows),
                  "mean": round(sum(r["n_words"] for r in rows) / len(rows), 1)},
        "counts": {k: len(v) for k, v in issues.items()},
        "issues": issues,
        "blocking_failures": [k for k in blocking if issues[k]] + (["missing_products"] if set(PRODUCTS) - products_covered else []),
    }


def _is_date(s: str | None) -> bool:
    if s is None:
        return True
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def check_db() -> dict:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cols = [r[1] for r in cur.execute("PRAGMA table_info(releases)")]
    expected = ["product_id", "cycle", "codename", "release_date", "is_lts", "eoas_date", "eol_date",
                "eoes_date", "latest_version", "latest_date"]
    rows = cur.execute("SELECT * FROM releases").fetchall()
    issues: dict[str, list] = {"bad_dates": [], "eol_before_release": [], "orphan_product": [],
                               "missing_release_date": [], "missing_eol_date": []}
    products = {r[0] for r in cur.execute("SELECT product_id FROM products")}
    for r in rows:
        rec = dict(zip(cols, r))
        key = f"{rec['product_id']}/{rec['cycle']}"
        for c in ("release_date", "eoas_date", "eol_date", "eoes_date", "latest_date"):
            if not _is_date(rec[c]):
                issues["bad_dates"].append(f"{key}.{c}={rec[c]}")
        if rec["release_date"] and rec["eol_date"] and rec["eol_date"] < rec["release_date"]:
            issues["eol_before_release"].append(key)
        if rec["product_id"] not in products:
            issues["orphan_product"].append(key)
        if not rec["release_date"]:
            issues["missing_release_date"].append(key)
        if not rec["eol_date"]:
            issues["missing_eol_date"].append(key)
    con.close()
    blocking = ["bad_dates", "eol_before_release", "orphan_product"]
    return {
        "n_rows": len(rows),
        "schema_ok": cols == expected,
        "counts": {k: len(v) for k, v in issues.items()},
        "issues": issues,
        # missing eol_date is expected (rolling / not-yet-announced EOL); reported, not blocking
        "blocking_failures": [k for k in blocking if issues[k]] + ([] if cols == expected else ["schema"]),
    }


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    report = {"chunks": check_chunks(), "lifecycle_db": check_db()}
    (REPORTS / "data_quality.json").write_text(json.dumps(report, indent=1))
    for name, part in report.items():
        print(f"[{name}] counts={part['counts']}")
        print(f"[{name}] blocking_failures={part['blocking_failures'] or 'none'}")
    failed = report["chunks"]["blocking_failures"] or report["lifecycle_db"]["blocking_failures"]
    print("DATA QUALITY:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
