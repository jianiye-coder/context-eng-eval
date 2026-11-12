"""Evaluation runner.

  python -m ceeval.evaluate retrieval              # retrieval-only metrics, no LLM (CI-friendly)
  python -m ceeval.evaluate generate [configs...]  # answer every question per config (resumable)
  python -m ceeval.evaluate judge [configs...]     # LLM-judge correctness + groundedness
  python -m ceeval.evaluate summarize              # metrics.json + results.md + dashboard.html
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import defaultdict
from dataclasses import asdict

from .config import (
    TOP_K, ELECTRICITY_USD_PER_KWH, GEN_MODEL, HOSTED_PRICE_PER_MTOK, LOCAL_POWER_WATTS, QUESTIONS_PATH,
    REPORTS,
)
from .metrics import (
    all_found_at_k, gold_chunk_sets, keyword_match, paired_bootstrap, pctl, recall_at_k,
    reciprocal_rank,
)

from .pipeline import is_abstention  # noqa: E402

ANSWERS = REPORTS / "answers"
CONFIGS = ["bm25", "dense", "hybrid", "hybrid_meta", "routed"]
TEXT_TYPES = ("single_doc", "multi_doc")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS_PATH.read_text().splitlines() if l.strip()]


# ------------------------------------------------------------------ retrieval
def run_retrieval(use_headers: bool = False, out_name: str = "retrieval_metrics.json") -> dict:
    from .retrieval import VARIANTS, Retriever
    r = Retriever(use_headers=use_headers)
    r.emb  # build/load index outside the timed loop
    qs = [q for q in load_questions() if q["type"] in TEXT_TYPES]
    out: dict = {"n_questions": len(qs), "variants": {}}
    per_q: dict = defaultdict(dict)
    for v in VARIANTS:
        rows, lat = defaultdict(list), []
        for q in qs:
            gold = gold_chunk_sets(q["evidence"], r.chunks, norm)
            # Each k gets its own search(k) call: the metadata variants' quota and escape-hatch
            # slot depend on k, so the top-5 of a k=10 list is not what the generator sees.
            ids_at = {}
            for k in (1, 3, 5, 10):
                t0 = time.perf_counter()
                hits = r.search(q["question"], v, k=k)
                if k == TOP_K:
                    lat.append((time.perf_counter() - t0) * 1000)
                ids_at[k] = [h.chunk.chunk_id for h in hits]
                rows[f"recall@{k}"].append(recall_at_k(ids_at[k], gold, k))
            ids5 = ids_at[5]
            rows["all_evidence@5"].append(float(all_found_at_k(ids5, gold, 5)))
            # MRR: rank of the first chunk matching ANY evidence item (for multi-doc questions
            # "all evidence found" is tracked separately by all_evidence@5)
            rows["mrr@10"].append(reciprocal_rank(ids_at[10], gold))
            for t in ("single_doc", "multi_doc"):
                if q["type"] == t:
                    rows[f"{t}:recall@5"].append(recall_at_k(ids5, gold, 5))
            rows[f"{q['phrasing']}:recall@5"].append(recall_at_k(ids5, gold, 5))
            per_q[q["qid"]][v] = {"recall@5": recall_at_k(ids5, gold, 5), "rr": reciprocal_rank(ids_at[10], gold)}
        out["variants"][v] = {k: round(sum(x) / len(x), 4) for k, x in rows.items()} | {
            "n_by_slice": {k: len(x) for k, x in rows.items() if ":" in k},
            "retrieval_ms_p50": round(pctl(lat, 50), 2), "retrieval_ms_p95": round(pctl(lat, 95), 2)}
    out["per_question"] = per_q
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / out_name).write_text(json.dumps(out, indent=1))
    print(f"{'variant':12s} " + " ".join(f"{m:>10s}" for m in ("recall@1", "recall@5", "recall@10", "mrr@10", "all_ev@5", "p50 ms")))
    for v, m in out["variants"].items():
        print(f"{v:12s} " + " ".join(f"{m[k]:10.3f}" for k in ("recall@1", "recall@5", "recall@10", "mrr@10", "all_evidence@5", "retrieval_ms_p50")))
    return out


# ------------------------------------------------------------------ generation
def _read_jsonl(path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def run_generate(configs: list[str]) -> None:
    from .llm import warmup
    from .pipeline import Pipeline
    ANSWERS.mkdir(parents=True, exist_ok=True)
    qs = load_questions()
    p = Pipeline()
    p.r.emb
    p.r.model.encode(["warm up"])
    warmup(GEN_MODEL)
    for cfg in configs:
        path = ANSWERS / f"{cfg}.jsonl"
        done = {a["qid"] for a in _read_jsonl(path)}
        with path.open("a") as f:
            for i, q in enumerate(qs):
                if q["qid"] in done:
                    continue
                a = p.answer(q["qid"], q["question"], cfg)
                f.write(json.dumps(asdict(a), ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{cfg}] {i + 1}/{len(qs)} {q['qid']} {a.route} {a.latency_s:.1f}s abstain={a.abstained}", flush=True)


def run_judge(configs: list[str]) -> None:
    from .judge import judge_correct, judge_grounded
    qs = {q["qid"]: q for q in load_questions()}
    for cfg in configs:
        path = ANSWERS / f"{cfg}.jsonl"
        rows = _read_jsonl(path)
        for i, a in enumerate(rows):
            q = qs[a["qid"]]
            if "judge" in a:
                continue
            a["abstained"] = is_abstention(a["answer"])
            j: dict = {}
            if q["type"] != "unanswerable" and not a["abstained"]:
                j["correct"] = judge_correct(q["question"], q["gold_answer"], a["answer"])
            if not a["abstained"] and a["context"]:
                j["grounded"] = judge_grounded(a["context"], a["answer"])
            a["judge"] = j
            print(f"[judge {cfg}] {i + 1}/{len(rows)} {a['qid']}", flush=True)
            path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


# ------------------------------------------------------------------ summary
def _cost(a: dict) -> dict:
    kwh = LOCAL_POWER_WATTS * a["latency_s"] / 3.6e6
    pin, pout = HOSTED_PRICE_PER_MTOK["hosted-small"]
    return {"local_usd": kwh * ELECTRICITY_USD_PER_KWH,
            "hosted_usd": a["input_tokens"] / 1e6 * pin + a["output_tokens"] / 1e6 * pout}


def summarize_config(cfg: str, qs: dict) -> dict | None:
    rows = _read_jsonl(ANSWERS / f"{cfg}.jsonl")
    if not rows:
        return None
    by_type: dict = defaultdict(list)
    judge_acc, agree, grounded, fully, ans_abst = [], [], [], [], []
    for a in rows:
        q = qs[a["qid"]]
        a["abstained"] = is_abstention(a["answer"])  # re-derive: detector may be newer than the run
        if q["type"] == "unanswerable":
            ok = a["abstained"]
        else:
            ok = (not a["abstained"]) and keyword_match(a["answer"], q["answer_keywords"])
            ans_abst.append(a["abstained"])
            j = a.get("judge", {})
            # judge accuracy/agreement only where the judge actually graded an answer
            if not a["abstained"] and "correct" in j:
                judge_acc.append(j["correct"]["correct"])
                agree.append(j["correct"]["correct"] == ok)
        by_type[q["type"]].append(ok)
        g = a.get("judge", {}).get("grounded")
        if g and g["groundedness"] is not None:
            grounded.append(g["groundedness"])
            fully.append(g["n_supported"] == g["n_claims"])
    lat = [a["latency_s"] for a in rows]
    costs = [_cost(a) for a in rows]
    unans = by_type.get("unanswerable", [])
    allok = [x for v in by_type.values() for x in v]
    mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None  # noqa: E731
    return {
        "n": len(rows),
        "accuracy": mean(allok),
        "accuracy_by_type": {t: mean(v) for t, v in by_type.items()},
        "n_by_type": {t: len(v) for t, v in by_type.items()},
        "judge_accuracy": mean(judge_acc),          # answerable, non-abstained answers only
        "n_judge_graded": len(judge_acc),
        "judge_keyword_agreement": mean(agree),
        "groundedness": mean(grounded),
        "fully_grounded_rate": mean(fully),
        "n_grounded_judged": len(grounded),
        "abstention_on_unanswerable": mean(unans),
        "false_abstention_on_answerable": mean(ans_abst),
        "latency_p50_s": round(pctl(lat, 50), 2),
        "latency_p95_s": round(pctl(lat, 95), 2),
        "avg_input_tokens": round(sum(a["input_tokens"] for a in rows) / len(rows), 1),
        "avg_output_tokens": round(sum(a["output_tokens"] for a in rows) / len(rows), 1),
        "avg_llm_calls": round(sum(a["llm_calls"] for a in rows) / len(rows), 2),
        "cost_per_query_local_usd": round(sum(c["local_usd"] for c in costs) / len(rows), 6),
        "cost_per_query_hosted_est_usd": round(sum(c["hosted_usd"] for c in costs) / len(rows), 6),
        "routing": {"sql": sum(a["route"] == "sql" for a in rows), "text": sum(a["route"] == "text" for a in rows)},
    }


def _per_question_ok(cfg: str, qs: dict) -> dict:
    out = {}
    for a in _read_jsonl(ANSWERS / f"{cfg}.jsonl"):
        q = qs[a["qid"]]
        ab = is_abstention(a["answer"])
        out[a["qid"]] = float(ab if q["type"] == "unanswerable"
                              else (not ab) and keyword_match(a["answer"], q["answer_keywords"]))
    return out


def run_summarize() -> dict:
    qs = {q["qid"]: q for q in load_questions()}
    summary = {"generator": GEN_MODEL, "n_questions": len(qs), "configs": {}}
    for cfg in CONFIGS:
        s = summarize_config(cfg, qs)
        if s:
            summary["configs"][cfg] = s
    rpath = REPORTS / "retrieval_metrics.json"
    if rpath.exists():
        rj = json.loads(rpath.read_text())
        summary["retrieval"] = rj["variants"]
        pq = rj["per_question"]
        qids = sorted(pq)
        summary["significance"] = {"retrieval_recall@5_vs_bm25": {
            v: paired_bootstrap([pq[q]["bm25"]["recall@5"] for q in qids], [pq[q][v]["recall@5"] for q in qids])
            for v in rj["variants"] if v != "bm25"}}
    # paired answer-accuracy comparison against the text-only hybrid_meta config
    ok = {c: _per_question_ok(c, qs) for c in summary["configs"]}
    if "hybrid_meta" in ok:
        base = ok["hybrid_meta"]
        summary.setdefault("significance", {})["answer_accuracy_vs_hybrid_meta"] = {
            c: paired_bootstrap([base[q] for q in sorted(base)], [ok[c][q] for q in sorted(base)])
            for c in ok if c != "hybrid_meta" and set(ok[c]) == set(base)}
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=1))
    from .report import write_reports
    write_reports(summary)
    return summary


def main(argv: list[str]) -> None:
    cmd, rest = argv[0], argv[1:]
    if cmd == "retrieval":
        run_retrieval()
    elif cmd == "generate":
        run_generate(rest or CONFIGS)
    elif cmd == "judge":
        run_judge(rest or CONFIGS)
    elif cmd == "summarize":
        s = run_summarize()
        print(json.dumps({k: {m: v[m] for m in ("accuracy", "groundedness", "latency_p50_s")}
                          for k, v in s["configs"].items()}, indent=1))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
