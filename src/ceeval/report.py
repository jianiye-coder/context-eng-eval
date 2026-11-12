"""Render metrics.json into reports/results.md (tables pasted into the README) and a
self-contained static dashboard (reports/dashboard.html, no server needed)."""
from __future__ import annotations

import html
import json

from .config import REPORTS

ANS_COLS = [
    ("accuracy", "Answer acc.", "pct"),
    ("judge_accuracy", "Judge acc.", "pct"),
    ("groundedness", "Groundedness", "pct"),
    ("abstention_on_unanswerable", "Abstain (unans.)", "pct"),
    ("false_abstention_on_answerable", "False abstain", "pct"),
    ("latency_p50_s", "p50 s", "num"),
    ("latency_p95_s", "p95 s", "num"),
    ("avg_input_tokens", "In tok", "int"),
    ("cost_per_query_hosted_est_usd", "Hosted $/q", "usd"),
]
TYPES = ["single_doc", "multi_doc", "structured", "unanswerable"]
RET_COLS = [("recall@1", "R@1"), ("recall@5", "R@5"), ("recall@10", "R@10"), ("mrr@10", "MRR@10"),
            ("all_evidence@5", "All evid.@5"), ("multi_doc:recall@5", "Multi-doc R@5"),
            ("retrieval_ms_p50", "p50 ms")]


def _fmt(v, kind: str) -> str:
    if v is None:
        return "–"
    return {"pct": f"{v * 100:.1f}%", "num": f"{v:.1f}", "int": f"{v:.0f}",
            "usd": f"${v:.5f}"}.get(kind, str(v))


def tables_md(s: dict) -> str:
    out = []
    if "retrieval" in s:
        out.append("| Retriever | " + " | ".join(c[1] for c in RET_COLS) + " |")
        out.append("|---|" + "---:|" * len(RET_COLS))
        for v, m in s["retrieval"].items():
            cells = [(f"{m[k]:.3f}" if k != "retrieval_ms_p50" else f"{m[k]:.1f}") for k, _ in RET_COLS]
            out.append(f"| `{v}` | " + " | ".join(cells) + " |")
        out.append("")
    if s.get("configs"):
        out.append("| Config | " + " | ".join(c[1] for c in ANS_COLS) + " |")
        out.append("|---|" + "---:|" * len(ANS_COLS))
        for cfg, m in s["configs"].items():
            out.append(f"| `{cfg}` | " + " | ".join(_fmt(m.get(k), kind) for k, _, kind in ANS_COLS) + " |")
        out.append("")
        out.append("| Config | " + " | ".join(f"{t} (n={next(iter(s['configs'].values()))['n_by_type'].get(t, 0)})" for t in TYPES) + " |")
        out.append("|---|" + "---:|" * len(TYPES))
        for cfg, m in s["configs"].items():
            out.append(f"| `{cfg}` | " + " | ".join(_fmt(m["accuracy_by_type"].get(t), "pct") for t in TYPES) + " |")
        out.append("")
    for name, comp in s.get("significance", {}).items():
        out.append(f"| {name} | diff | 95% CI | n | distinguishable from noise? |")
        out.append("|---|---:|---:|---:|:-:|")
        for v, r in comp.items():
            out.append(f"| `{v}` | {r['diff']:+.3f} | [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['n']} | "
                       f"{'yes' if r['significant'] else 'no'} |")
        out.append("")
    return "\n".join(out) + "\n"


def dashboard_html(s: dict) -> str:
    def bars(metric: str, label: str, kind: str = "pct") -> str:
        cfgs = s.get("configs", {})
        vals = {c: m.get(metric) for c, m in cfgs.items() if m.get(metric) is not None}
        if not vals:
            return ""
        top = max(vals.values()) or 1
        rows = "".join(
            f'<div class="row"><span class="lab">{html.escape(c)}</span>'
            f'<span class="bar" style="width:{(v / top) * 100 if kind != "pct" else v * 100:.1f}%"></span>'
            f'<span class="val">{_fmt(v, kind)}</span></div>' for c, v in vals.items())
        return f'<section><h3>{label}</h3>{rows}</section>'

    md = tables_md(s)
    table_html = []
    for block in md.strip().split("\n\n"):
        lines = [l for l in block.splitlines() if not l.startswith("|---")]
        if not lines:
            continue
        hdr = [c.strip() for c in lines[0].strip("|").split("|")]
        body = [[c.strip().strip("`") for c in l.strip("|").split("|")] for l in lines[1:]]
        table_html.append("<table><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in hdr) + "</tr>" +
                          "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>" for r in body) +
                          "</table>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Context-engineering eval</title>
<style>
body{{font:14px/1.45 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:32px auto;max-width:1060px;color:#1d2329;padding:0 16px}}
h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#5b6672;margin:0 0 24px}}
table{{border-collapse:collapse;margin:8px 0 24px;width:100%}} th,td{{padding:6px 8px;border-bottom:1px solid #e3e7eb;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{font-weight:600;color:#3c4650;background:#f6f8fa}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}}
section h3{{font-size:13px;margin:0 0 8px;color:#3c4650}}
.row{{display:grid;grid-template-columns:110px 1fr 64px;align-items:center;gap:8px;margin:4px 0}}
.lab{{font-family:ui-monospace,Menlo,monospace;font-size:12px}} .bar{{height:12px;background:#3b6ea5;border-radius:2px;display:block}}
.val{{text-align:right;font-variant-numeric:tabular-nums}}
</style></head><body>
<h1>Grounded Q&amp;A over enterprise product data: eval dashboard</h1>
<p class="sub">Generator: {html.escape(s.get('generator', ''))} (local) · {s.get('n_questions')} questions · generated from reports/metrics.json</p>
<div class="grid">{bars('accuracy', 'Answer accuracy (deterministic)')}{bars('groundedness', 'Groundedness (LLM judge, claim-level)')}
{bars('abstention_on_unanswerable', 'Correct abstention on unanswerable')}{bars('latency_p95_s', 'p95 latency (s)', 'num')}</div>
{''.join(table_html)}
</body></html>"""


def write_reports(summary: dict) -> None:
    (REPORTS / "results.md").write_text(tables_md(summary))
    (REPORTS / "dashboard.html").write_text(dashboard_html(summary))


if __name__ == "__main__":
    write_reports(json.loads((REPORTS / "metrics.json").read_text()))
