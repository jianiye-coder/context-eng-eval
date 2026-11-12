"""Assemble eval/questions.jsonl from the reviewed drafts + the structured (SQL) set below.

Structured questions carry a hand-written gold SQL; the gold answer is whatever that SQL
returns on the pinned snapshot, so the gold can never drift from the data.
Run: python eval/build_eval_set.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ceeval.config import AS_OF, CHUNKS_PATH, DB_PATH, QUESTIONS_PATH  # noqa: E402

DRAFTS = ["single_a.jsonl", "single_b.jsonl", "multi.jsonl", "unanswerable.jsonl"]
# near-duplicates of multi-doc questions (same evidence); dropped during review
DROP = {"sd-a-04", "sd-a-13", "sd-b-07"}
# review fixes: "no" as a keyword matches almost anything
PATCH_KEYWORDS = {"sd-a-14": [["agpl", "affero"], ["without", "not required", "not needed", "no subscription"]]}

S = AS_OF
STRUCTURED = [
    ("st-01", "When does Red Hat Enterprise Linux 9 reach the end of its maintenance support (EOL)?",
     "SELECT eol_date FROM releases WHERE product_id='rhel' AND cycle='9'", [["2032-05-31"]]),
    ("st-02", "What is the end-of-life date for PostgreSQL 13?",
     "SELECT eol_date FROM releases WHERE product_id='postgresql' AND cycle='13'", [["2025-11-13"]]),
    ("st-03", "What is the latest patch release in the PostgreSQL 16 line?",
     "SELECT latest_version FROM releases WHERE product_id='postgresql' AND cycle='16'", [["16.16"]]),
    ("st-04", "When was Ubuntu 24.04 released, and what is its codename?",
     "SELECT release_date, codename FROM releases WHERE product_id='ubuntu' AND cycle='24.04'",
     [["2024-04-25"], ["noble numbat"]]),
    ("st-05", f"Which Ubuntu LTS releases are still within standard support as of {S}?",
     f"SELECT cycle FROM releases WHERE product_id='ubuntu' AND is_lts=1 AND eol_date > '{S}' ORDER BY cycle",
     [["22.04"], ["24.04"], ["26.04"]]),
    ("st-06", "When does extended life cycle support for RHEL 8 end?",
     "SELECT eoes_date FROM releases WHERE product_id='rhel' AND cycle='8'", [["2033-05-31"]]),
    ("st-07", f"How many Kubernetes minor versions are still maintained (not past EOL) as of {S}?",
     f"SELECT COUNT(*) FROM releases WHERE product_id='kubernetes' AND eol_date > '{S}'", [["3", "three"]]),
    ("st-08", "When does Windows Server 2019 reach end of support?",
     "SELECT eol_date FROM releases WHERE product_id='windows-server' AND cycle='2019'", [["2029-01-09"]]),
    ("st-09", "Which VMware ESXi versions reached end of support in 2022?",
     "SELECT cycle FROM releases WHERE product_id='esxi' AND eol_date LIKE '2022%' ORDER BY cycle",
     [["6.5"], ["6.7"]]),
    ("st-10", "What is the newest MongoDB release cycle and when was it released?",
     "SELECT cycle, release_date FROM releases WHERE product_id='mongodb' ORDER BY release_date DESC LIMIT 1",
     [["9.0"], ["2026-09-30"]]),
    ("st-11", "What is the codename of Debian 11?",
     "SELECT codename FROM releases WHERE product_id='debian' AND cycle='11'", [["bullseye"]]),
    ("st-12", "How many major Proxmox VE release lines are listed in the lifecycle data?",
     "SELECT COUNT(*) FROM releases WHERE product_id='proxmox-ve'",
     # bare "9" would also match the version "Proxmox VE 9", so require it as a count
     [["nine", "9 major", "9 release", "9 lines", "9 versions", "9 cycles", "9 entries", "9 distinct",
       "total of 9", "are 9", "is 9", ": 9", "count is 9"]]),
    ("st-13", "Which Oracle Database releases since 2010 are long-term support releases?",
     "SELECT cycle FROM releases WHERE product_id='oracle-database' AND is_lts=1 AND release_date >= '2010-01-01' ORDER BY release_date",
     [["12.1", "12c"], ["19", "19c"], ["23", "23ai", "23c"]]),
    ("st-14", "Which was released first: RHEL 9, Rocky Linux 9, or AlmaLinux 9?",
     "SELECT product_id, release_date FROM releases WHERE cycle='9' AND product_id IN ('rhel','rocky-linux','almalinux') ORDER BY release_date",
     [["rhel", "red hat enterprise linux"], ["2022-05-18"]]),
    ("st-15", "What is the end-of-support date for NetApp ONTAP 9.14.1?",
     "SELECT eol_date FROM releases WHERE product_id='netapp-ontap' AND cycle='9.14.1'", [["2027-01-31"]]),
    ("st-16", "When does extended update support for OpenShift 4.18 end?",
     "SELECT eoes_date FROM releases WHERE product_id='red-hat-openshift' AND cycle='4.18'", [["2027-02-25"]]),
    ("st-17", f"Is MySQL 8.0 still supported as of {S}, and when did or does its support end?",
     "SELECT eol_date FROM releases WHERE product_id='mysql' AND cycle='8.0'",
     [["no longer", "ended", "reached", "not supported", "no,", "past its", "expired", "is not still", "not still"],
      ["2026-04-30"]]),
    ("st-18", "Which Kubernetes version was released most recently?",
     "SELECT cycle FROM releases WHERE product_id='kubernetes' ORDER BY release_date DESC LIMIT 1", [["1.37"]]),
    ("st-19", "When does mainstream (active) support for Windows Server 2022 end?",
     "SELECT eoas_date FROM releases WHERE product_id='windows-server' AND cycle='2022'", [["2026-10-13"]]),
    ("st-20", "Which MySQL release lines are marked as LTS?",
     "SELECT cycle FROM releases WHERE product_id='mysql' AND is_lts=1 ORDER BY release_date",
     [["8.0"], ["8.4"], ["9.7"]]),
]
# Structured-looking but unanswerable: the cycle does not exist in the data.
STRUCTURED_UNANSWERABLE = [
    ("ua-s1", "When does Ubuntu 30.04 LTS reach end of standard support?"),
    ("ua-s2", "When does maintenance support end for RHEL 11?"),
    ("ua-s3", "What is the end-of-life date for PostgreSQL 25?"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    chunks = [json.loads(l) for l in CHUNKS_PATH.read_text().splitlines()]
    out = []
    for name in DRAFTS:
        for line in (ROOT / "eval" / "drafts" / name).read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            if q["qid"] in DROP:
                continue
            q["answer_keywords"] = PATCH_KEYWORDS.get(q["qid"], q["answer_keywords"])
            q["route"] = "text"
            for ev in q["evidence"]:
                hits = [c["chunk_id"] for c in chunks
                        if c["product_id"] == ev["product_id"] and norm(ev["quote"]) in norm(c["text"])]
                if not hits:
                    raise SystemExit(f"{q['qid']}: evidence quote not found: {ev['quote'][:60]}")
            out.append(q)

    con = sqlite3.connect(DB_PATH)
    # readable gold answers where the bare SQL result would not let a judge grade the claim
    GOLD_TEXT = {
        "st-14": "RHEL 9 was released first (2022-05-18), before AlmaLinux 9 (2022-05-26) and Rocky Linux 9 (2022-07-14).",
        "st-17": "No. MySQL 8.0 support ended on 2026-04-30.",
        "st-12": "9 release lines.",
        "st-07": "3 minor versions.",
    }
    for qid, question, sql, kw in STRUCTURED:
        rows = con.execute(sql).fetchall()
        if not rows:
            raise SystemExit(f"{qid}: gold SQL returned no rows")
        out.append({"qid": qid, "type": "structured", "phrasing": "paraphrase", "route": "sql",
                    "question": question, "gold_sql": sql, "gold_result": rows,
                    "gold_answer": GOLD_TEXT.get(qid) or "; ".join(", ".join(str(v) for v in r) for r in rows),
                    "answer_keywords": kw, "evidence": [], "notes": ""})
    for qid, question in STRUCTURED_UNANSWERABLE:
        out.append({"qid": qid, "type": "unanswerable", "phrasing": "paraphrase", "route": "sql",
                    "question": question, "gold_answer": "ABSTAIN", "answer_keywords": [],
                    "evidence": [], "notes": "cycle does not exist in lifecycle.db"})

    QUESTIONS_PATH.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in out))
    from collections import Counter
    print(len(out), dict(Counter(q["type"] for q in out)))


if __name__ == "__main__":
    main()
