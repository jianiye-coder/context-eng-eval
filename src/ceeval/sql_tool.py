"""Text-to-SQL tool for lifecycle questions, with guardrails.

The LLM sees a documented schema (a small "knowledge model" of what each column means),
the snapshot date, and few-shot examples that are NOT in the eval set. Generated SQL is
validated (single read-only SELECT on known tables), executed on a read-only connection,
and repaired once if it errors. Zero rows means "not in the data" -> the answerer abstains.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from .config import AS_OF, DB_PATH, GEN_MODEL, PRODUCTS
from .llm import chat

SCHEMA_DOC = f"""SQLite database of software release lifecycles (snapshot date {AS_OF}).

TABLE products(product_id TEXT PK, product TEXT, category TEXT)
TABLE releases(
  product_id TEXT      -- FK to products; one of: {", ".join(PRODUCTS)}
  cycle TEXT           -- release line as a string, e.g. '9' (RHEL), '22.04' (Ubuntu), '2019' (Windows Server),
                       --   '16.0' (SQL Server 2022 is cycle '16.0', 2019 is '15.0'), '1.35' (Kubernetes), '9.14.1' (ONTAP)
  codename TEXT        -- e.g. 'Noble Numbat', 'Bookworm'; may be NULL
  release_date TEXT    -- ISO date the cycle was first released
  is_lts INTEGER       -- 1 = long-term-support line
  eoas_date TEXT       -- end of active / mainstream / full support (NULL if vendor has no such phase)
  eol_date TEXT        -- end of standard support = end of life / end of maintenance support
  eoes_date TEXT       -- end of extended (usually paid) support, e.g. RHEL ELS, Ubuntu Pro, OpenShift EUS; may be NULL
  latest_version TEXT  -- latest patch release in this cycle
  latest_date TEXT
)
Dates are ISO strings, compare them as strings. "Still supported" means eol_date > '{AS_OF}'.
Never use date('now'); use the literal snapshot date '{AS_OF}'."""

FEW_SHOT = [
    ("When does Debian 12 reach end of life?",
     "SELECT eol_date FROM releases WHERE product_id='debian' AND cycle='12'"),
    ("Which SUSE Linux Enterprise 15 service packs are still supported?",
     f"SELECT cycle, eol_date FROM releases WHERE product_id='sles' AND cycle LIKE '15.%' AND eol_date > '{AS_OF}' ORDER BY cycle"),
    ("What is the latest patch version of Redis 7.2?",
     "SELECT latest_version FROM releases WHERE product_id='redis' AND cycle='7.2'"),
    ("Which was released first, CentOS 7 or RHEL 7?",
     "SELECT product_id, release_date FROM releases WHERE cycle='7' AND product_id IN ('centos','rhel') ORDER BY release_date"),
]

ALLOWED_TABLES = {"releases", "products", "meta"}
FORBIDDEN = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b|\breplace\s+into\b|\bsqlite_", re.I)


@dataclass
class SQLResult:
    sql: str | None
    columns: list[str] = field(default_factory=list)
    rows: list[tuple] = field(default_factory=list)
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    llm_latency_s: float = 0.0
    llm_wall_s: float = 0.0
    attempts: int = 0


def extract_sql(text: str) -> str:
    m = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    sql = (m.group(1) if m else text).strip()
    m = re.search(r"\bselect\b.*", sql, re.S | re.I)
    sql = m.group(0) if m else sql
    return sql.strip().rstrip(";").strip()


DOTTED_VERSION = re.compile(r"(?<![\d-])\d+(?:\.\d+)+(?![\d-])")
# a bare 1-2 digit number right after a capitalised name: "RHEL 11", "Debian 12", "Proxmox VE 9"
NAMED_MAJOR = re.compile(r"\b[A-Z][\w-]*\s(\d{1,2})\b(?![.\d])")


def version_literals(question: str) -> list[str]:
    """Version-like tokens the user named (e.g. '9.14.1', '30.04', 'RHEL 11'), excluding ISO
    dates and lowercase phrases like 'top 3'. 4-digit cycles (Windows Server 2019) are not
    guarded because they are indistinguishable from years ('since 2010')."""
    q = re.sub(r"\d{4}-\d{2}-\d{2}", " ", question)
    return DOTTED_VERSION.findall(q) + NAMED_MAJOR.findall(q)


def _mentions_version(sql: str, v: str) -> bool:
    ev = re.escape(v)
    return bool(re.search(rf"'{ev}'", sql) or re.search(rf"like\s+'{ev}(?:\.%|%)'", sql, re.I))


def validate(sql: str, question: str = "") -> str | None:
    """Return an error message, or None if the SQL is acceptable."""
    for v in version_literals(question):
        # Guard against the model silently swapping the version the user asked about
        # (observed: '30.04' rewritten to '20.04'), which would answer a different question.
        if not _mentions_version(sql, v):
            return f"the query must filter on the version the user asked about: '{v}'"
    if not re.match(r"^\s*(select|with)\b", sql, re.I):
        return "only SELECT queries are allowed"
    if ";" in sql:
        return "only one statement is allowed"
    if FORBIDDEN.search(sql):
        return "write/DDL keywords are not allowed"
    tables = set()
    # every FROM clause, including ones nested in subqueries; comma-separated lists included
    for m in re.finditer(r"(?=\bfrom\s+(.+?)(?:\bwhere\b|\bjoin\b|\border\b|\bgroup\b|\blimit\b|\bfrom\b|\)|$))", sql, re.I | re.S):
        tables |= {p.strip().split()[0].lower() for p in m.group(1).split(",")
                   if p.strip() and not p.strip().startswith("(")}  # "(SELECT ..." is a subquery
    tables |= {t.lower() for t in re.findall(r"\bjoin\s+([a-zA-Z_]+)", sql, re.I)}
    if not tables or not tables <= ALLOWED_TABLES:
        return f"unknown table(s): {sorted(tables - ALLOWED_TABLES) or 'none'}"
    if re.search(r"date\(\s*'now'\s*\)", sql, re.I):
        return f"do not use date('now'); use '{AS_OF}'"
    return None


def run_readonly(sql: str) -> tuple[list[str], list[tuple]]:
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        cur = con.execute(sql if re.search(r"\blimit\b", sql, re.I) else sql + " LIMIT 50")
        return [d[0] for d in cur.description], cur.fetchall()
    finally:
        con.close()


def text_to_sql(question: str, model: str = GEN_MODEL) -> SQLResult:
    shots = "\n\n".join(f"Q: {q}\nSQL: {s}" for q, s in FEW_SHOT)
    messages = [
        {"role": "system", "content": "You translate questions into a single SQLite SELECT query. "
                                      "Reply with the SQL only, no explanation."},
        {"role": "user", "content": f"{SCHEMA_DOC}\n\nExamples:\n{shots}\n\nQ: {question}\nSQL:"},
    ]
    res = SQLResult(sql=None)
    for attempt in range(2):  # one self-repair round
        out = chat(messages, model, max_tokens=200)
        res.input_tokens += out.input_tokens
        res.output_tokens += out.output_tokens
        res.llm_latency_s += out.latency_s
        res.llm_wall_s += out.wall_s
        res.attempts = attempt + 1
        sql = extract_sql(out.text)
        res.sql = sql
        err = validate(sql, question)
        if err is None:
            try:
                res.columns, res.rows = run_readonly(sql)
                res.error = None
                return res
            except sqlite3.Error as e:
                err = f"sqlite error: {e}"
        res.error = err
        messages += [{"role": "assistant", "content": out.text},
                     {"role": "user", "content": f"That query failed: {err}. Reply with a corrected SQL query only."}]
    return res
