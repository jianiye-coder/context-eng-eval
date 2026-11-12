"""Turn the raw snapshot into two AI-ready data products.

1. data/processed/chunks.jsonl : cleaned, section-aware text chunks with metadata
2. data/processed/lifecycle.db : SQLite table of product release cycles

Both are validated by `ceeval.quality` before anything downstream uses them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass

from .config import (
    AS_OF, CHUNK_MAX_WORDS, CHUNK_OVERLAP_SENTENCES, CHUNK_TARGET_WORDS, CHUNKS_PATH,
    DB_PATH, DROP_SECTIONS, MIN_CHUNK_WORDS, PROCESSED, PRODUCTS, RAW_EOL, RAW_WIKI,
)

HEADING = re.compile(r"^(={2,6})\s*(.+?)\s*\1\s*$")
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


@dataclass
class Chunk:
    chunk_id: str
    product_id: str
    product: str
    category: str
    doc_title: str
    section: str          # "History > Early years"
    chunk_index: int      # position within the section
    text: str             # body only
    n_words: int
    source_url: str
    revid: int
    doc_date: str         # Wikipedia revision timestamp (as-of date of the text)
    license: str
    content_hash: str

    def contextual_text(self) -> str:
        """Body prefixed with a short header so a chunk is self-describing out of context.

        Used for indexing (BM25 + embeddings) and for the LLM context window.
        """
        return f"[{self.product} | {self.section}]\n{self.text}"


def clean_text(text: str) -> str:
    text = text.replace(" ", " ")
    # pronunciation / IPA residue left by the plain-text extract, e.g. "Ubuntu (  uu-BUUN-too)"
    text = re.sub(r"\(\s*[;,]?\s*\)", "", text)
    text = re.sub(r"\(\s{2,}[^)]{0,60}\)", "", text)
    # leftover LaTeX from math templates
    text = re.sub(r"\{\\displaystyle[^}]*\}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([,.;:])", r"\1", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[list[str], str]]:
    """Return [(heading_path, body)] in document order. Lead section gets path ['Overview']."""
    sections: list[tuple[list[str], list[str]]] = [(["Overview"], [])]
    path: list[str] = []
    for line in text.splitlines():
        m = HEADING.match(line.strip())
        if m:
            level = len(m.group(1)) - 1  # '==' -> 1
            path = path[: level - 1] + [m.group(2).strip()]
            sections.append((list(path), []))
        else:
            sections[-1][1].append(line)
    out = []
    for p, lines in sections:
        if any(h.lower() in DROP_SECTIONS for h in p):
            continue
        body = clean_text("\n".join(lines))
        if body:
            out.append((p, body))
    return out


def pack_sentences(body: str) -> list[str]:
    """Greedy sentence packing to ~CHUNK_TARGET_WORDS, never splitting a sentence,
    with a one-sentence overlap so facts on a boundary are not orphaned."""
    paras = [p.strip() for p in body.split("\n") if p.strip()]
    sents: list[str] = []
    for p in paras:
        sents.extend(s.strip() for s in SENT_SPLIT.split(p) if s.strip())
    chunks, cur, cur_words = [], [], 0
    for s in sents:
        w = len(s.split())
        if cur and (cur_words + w > CHUNK_MAX_WORDS or cur_words >= CHUNK_TARGET_WORDS):
            chunks.append(" ".join(cur))
            cur = cur[-CHUNK_OVERLAP_SENTENCES:] if CHUNK_OVERLAP_SENTENCES else []
            cur_words = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_words += w
    if cur:
        tail = " ".join(cur)
        # avoid a tiny trailing chunk that is mostly overlap: merge it back
        if chunks and len(tail.split()) < 40:
            prev_sents = set(SENT_SPLIT.split(chunks[-1]))
            extra = [s for s in cur if s not in prev_sents]
            chunks[-1] = (chunks[-1] + " " + " ".join(extra)).strip()
        else:
            chunks.append(tail)
    return chunks


def merge_small_sections(sections: list[tuple[list[str], str]]) -> list[tuple[list[str], str]]:
    """A section shorter than MIN_CHUNK_WORDS (often a heading whose content was a table)
    is folded into the next section instead of becoming a near-empty chunk."""
    out: list[tuple[list[str], str]] = []
    carry = ""
    for path, body in sections:
        body = (carry + "\n" + body).strip() if carry else body
        carry = ""
        if len(body.split()) < MIN_CHUNK_WORDS:
            carry = f"{path[-1]}: {body}"
            continue
        out.append((path, body))
    if carry and out:
        p, b = out[-1]
        out[-1] = (p, b + "\n" + carry)
    return out


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40]


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    for pid, (label, category, _title) in PRODUCTS.items():
        doc = json.loads((RAW_WIKI / f"{pid}.json").read_text())
        for sec_idx, (path, body) in enumerate(merge_small_sections(split_sections(doc["text"]))):
            section = " > ".join(path)
            for i, piece in enumerate(pack_sentences(body)):
                h = hashlib.sha1(piece.encode()).hexdigest()[:10]
                chunks.append(Chunk(
                    chunk_id=f"{pid}::{sec_idx:02d}-{slug(path[-1])}::{i}",
                    product_id=pid, product=label, category=category,
                    doc_title=doc["title"], section=section, chunk_index=i,
                    text=piece, n_words=len(piece.split()),
                    source_url=doc["url"], revid=doc["revid"],
                    doc_date=doc["rev_timestamp"][:10], license=doc["license"],
                    content_hash=h,
                ))
    return chunks


SCHEMA = """
CREATE TABLE products (
    product_id   TEXT PRIMARY KEY,
    product      TEXT NOT NULL,
    category     TEXT NOT NULL
);
CREATE TABLE releases (
    product_id      TEXT NOT NULL REFERENCES products(product_id),
    cycle           TEXT NOT NULL,      -- release line, e.g. '9', '22.04', '2019'
    codename        TEXT,
    release_date    TEXT,               -- ISO date the cycle was first released
    is_lts          INTEGER,            -- 1 if a long-term-support line
    eoas_date       TEXT,               -- end of active/full support (if the vendor has that phase)
    eol_date        TEXT,               -- end of standard security support / end of life
    eoes_date       TEXT,               -- end of extended (paid) support, if offered
    latest_version  TEXT,               -- latest patch release in this cycle
    latest_date     TEXT,
    PRIMARY KEY (product_id, cycle)
);
"""


def _date(v) -> str | None:
    """endoflife.date mixes ISO dates and booleans; keep only real dates."""
    return v if isinstance(v, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", v) else None


def build_db() -> int:
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    n = 0
    for pid, (label, category, _t) in PRODUCTS.items():
        con.execute("INSERT INTO products VALUES (?,?,?)", (pid, label, category))
        data = json.loads((RAW_EOL / f"{pid}.json").read_text())["result"]
        for r in data["releases"]:
            latest = r.get("latest") or {}
            con.execute(
                "INSERT OR IGNORE INTO releases VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pid, r["name"], r.get("codename"), _date(r.get("releaseDate")),
                 int(bool(r.get("isLts"))), _date(r.get("eoasFrom")), _date(r.get("eolFrom")),
                 _date(r.get("eoesFrom")), latest.get("name"), _date(latest.get("date"))),
            )
            n += 1
    con.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    con.execute("INSERT INTO meta VALUES ('as_of', ?)", (AS_OF,))
    con.commit()
    con.close()
    return n


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    chunks = build_chunks()
    with CHUNKS_PATH.open("w") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    n_rows = build_db()
    print(f"wrote {len(chunks)} chunks -> {CHUNKS_PATH.relative_to(PROCESSED.parent.parent)}")
    print(f"wrote {n_rows} release rows -> {DB_PATH.relative_to(PROCESSED.parent.parent)}")


def load_chunks() -> list[Chunk]:
    return [Chunk(**json.loads(l)) for l in CHUNKS_PATH.read_text().splitlines() if l.strip()]


if __name__ == "__main__":
    main()
