"""Agreement between the LLM judge and the human labels in reports/human_review_sample.csv.

"grounded" is compared as a binary: judge says fully grounded (all claims supported) vs. the
reviewer's 1/0. Reports raw agreement and Cohen's kappa (agreement corrected for chance).
Usage: python scripts/score_review.py [csv_path] [human_col_prefix]
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "human_review_sample.csv"
prefix = sys.argv[2] if len(sys.argv) > 2 else "human"


def kappa(a: list[int], b: list[int]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


rows = list(csv.DictReader(path.open()))
for name, jcol, hcol, conv in [
    ("correct", "judge_correct", f"{prefix}_correct", lambda v: int(v == "True")),
    ("grounded", "judge_groundedness", f"{prefix}_grounded", lambda v: int(float(v) == 1.0)),
]:
    pairs = [(conv(r[jcol]), int(r[hcol])) for r in rows if r.get(hcol, "").strip() in ("0", "1") and r[jcol] != ""]
    if not pairs:
        print(f"{name}: no labels yet in column {hcol}")
        continue
    j, h = zip(*pairs)
    agree = sum(x == y for x, y in pairs) / len(pairs)
    fp = sum(1 for x, y in pairs if x == 1 and y == 0)
    fn = sum(1 for x, y in pairs if x == 0 and y == 1)
    print(f"{name}: n={len(pairs)} agreement={agree:.1%} kappa={kappa(list(j), list(h)):.2f} "
          f"judge-too-lenient={fp} judge-too-strict={fn}")
