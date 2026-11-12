"""CI-style regression gate. Compares the latest metrics with eval/baseline.json and exits 1 if
any tracked metric got worse by more than its tolerance.

  python scripts/check_regression.py                    # retrieval + (if present) answer metrics
  python scripts/check_regression.py --retrieval-only   # what CI runs (no LLM needed)
  python scripts/check_regression.py --update-baseline  # accept current numbers as the new baseline

Fails closed: a missing baseline or a tracked metric missing from the current run is a failure.
Higher-is-better metrics may drop by at most `tolerance` (absolute). Latency may rise by at most
`latency_tolerance` (relative), since wall-clock on a shared laptop is noisy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "eval" / "baseline.json"
RET = ROOT / "reports" / "retrieval_metrics.json"
ANS = ROOT / "reports" / "metrics.json"

TRACK_RETRIEVAL = ["recall@5", "mrr@10", "all_evidence@5"]
TRACK_ANSWER = ["accuracy", "groundedness", "abstention_on_unanswerable"]
LOWER_BETTER = ["latency_p95_s", "false_abstention_on_answerable"]


def current(retrieval_only: bool) -> dict:
    cur: dict = {"retrieval": {}, "answer": {}}
    for v, m in json.loads(RET.read_text())["variants"].items():
        cur["retrieval"][v] = {k: m[k] for k in TRACK_RETRIEVAL}
    if not retrieval_only and ANS.exists():
        for c, m in json.loads(ANS.read_text())["configs"].items():
            cur["answer"][c] = {k: m[k] for k in TRACK_ANSWER + LOWER_BETTER if m.get(k) is not None}
    return cur


def main(argv: list[str]) -> int:
    retrieval_only = "--retrieval-only" in argv
    cur = current(retrieval_only)
    if not BASELINE.exists() and "--update-baseline" not in argv:
        print(f"FAIL: no baseline at {BASELINE.relative_to(ROOT)}; create one with --update-baseline")
        return 1
    if "--update-baseline" in argv:
        base = {"tolerance": 0.02, "latency_tolerance": 0.5, **cur}
        BASELINE.write_text(json.dumps(base, indent=1))
        print(f"baseline written: {BASELINE.relative_to(ROOT)}")
        return 0
    base = json.loads(BASELINE.read_text())
    tol, ltol = base["tolerance"], base["latency_tolerance"]
    failures, checked = [], 0
    for section in ("retrieval", "answer"):
        for name, metrics in base.get(section, {}).items():
            if name not in cur[section]:
                if section == "answer" and retrieval_only:
                    continue
                failures.append(f"{section}/{name}: missing from current run")
                continue
            for k, b in metrics.items():
                c = cur[section][name].get(k)
                if c is None:  # e.g. judge never ran: a missing metric is a failure, not a pass
                    failures.append(f"{section}/{name}/{k}: missing from current run")
                    continue
                checked += 1
                if k == "latency_p95_s":
                    bad = c > b * (1 + ltol)
                elif k in LOWER_BETTER:
                    bad = c > b + tol
                else:
                    bad = c < b - tol
                mark = "FAIL" if bad else "ok"
                print(f"{mark:4s} {section}/{name}/{k}: baseline={b:.3f} current={c:.3f}")
                if bad:
                    failures.append(f"{section}/{name}/{k}: {b:.3f} -> {c:.3f}")
    print(f"\nchecked {checked} metrics; {len(failures)} regression(s)")
    for f in failures:
        print("  REGRESSION", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
