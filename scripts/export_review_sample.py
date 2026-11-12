"""Export a stratified sample of judged answers for HUMAN review of the LLM judge.

Writes reports/human_review_sample.csv. Fill the two empty columns (human_correct,
human_grounded: 1/0) by reading the context and answer, then run scripts/score_review.py to
get judge-vs-human agreement and Cohen's kappa.
"""
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ceeval.evaluate import ANSWERS, load_questions  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
CFG = sys.argv[2] if len(sys.argv) > 2 else "routed"

qs = {q["qid"]: q for q in load_questions()}
rows = [json.loads(l) for l in (ANSWERS / f"{CFG}.jsonl").read_text().splitlines() if l.strip()]
judged = [a for a in rows if a.get("judge", {}).get("grounded")]
random.seed(0)
by_type: dict = {}
for a in judged:
    by_type.setdefault(qs[a["qid"]]["type"], []).append(a)
sample = []
while len(sample) < min(N, len(judged)):  # round-robin across question types
    for t in sorted(by_type):
        if by_type[t] and len(sample) < N:
            sample.append(by_type[t].pop(random.randrange(len(by_type[t]))))

out = ROOT / "reports" / "human_review_sample.csv"
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["qid", "type", "question", "gold_answer", "answer", "context",
                "judge_correct", "judge_groundedness", "judge_claims", "human_correct", "human_grounded"])
    for a in sample:
        q, j = qs[a["qid"]], a["judge"]
        w.writerow([a["qid"], q["type"], q["question"], q["gold_answer"], a["answer"], a["context"],
                    j.get("correct", {}).get("correct", ""), j["grounded"]["groundedness"],
                    json.dumps(j["grounded"]["claims"], ensure_ascii=False), "", ""])
print(f"wrote {len(sample)} rows -> {out.relative_to(ROOT)}")
