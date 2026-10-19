#!/usr/bin/env python3
"""Night 3 template A/B analysis with item-clustered statistics.
Per bench: accuracy, tool-markup leaks, runaways, API errors, prompt/completion tokens, latency; then every variant
paired against a reference on the items both ran: per item the mean score over repeats, mean difference with a
cluster bootstrap 95% CI (resampling items, not rows) and a sign test on items that differ.
usage: analyze_templates.py RESULTS_DIR REF TAG [TAG...]     (TAG or TAG@EFFORT = folder RESULTS_DIR/TAG/EFFORT, default medium)"""
import json, math, random, statistics as st, sys
from collections import defaultdict
from pathlib import Path

R = Path(sys.argv[1]); REF = sys.argv[2]; TAGS = sys.argv[3:]
BENCHES = ["bfcl", "evalplus", "longctx", "lcb"]


def load(tag, bench):  # TAG or TAG@EFFORT (results folder TAG/EFFORT, default medium)
    t, _, eff = tag.partition("@")
    f = R / t / (eff or "medium") / f"{bench}.jsonl"
    if not f.exists():
        return []
    return [json.loads(l) for l in f.open() if l.strip()]


def sign_p(a, b):  # two-sided exact sign test
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def per_item(rows):
    d = defaultdict(list)
    for r in rows:
        d[r["id"]].append(float(r.get("score", 1.0 if r.get("correct") else 0.0)))
    return {k: sum(v) / len(v) for k, v in d.items()}


out = []
for bench in BENCHES:
    data = {t: load(t, bench) for t in [REF] + [t for t in TAGS if t != REF]}
    data = {t: rows for t, rows in data.items() if rows}
    if len(data) < 1:
        continue
    out.append(f"\n## {bench}\n")
    out.append("| variant | rows | acc | tool markup leak | runaway | api err | prompt tok | completion tok | median latency s |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for t, rows in data.items():
        n = len(rows)
        acc = sum(r.get("score", r.get("correct", 0)) for r in rows) / n
        leak = sum(bool(r.get("raw_tool_markup")) for r in rows)
        run = sum(bool(r.get("runaway")) for r in rows)
        err = sum(bool(r.get("api_error")) for r in rows)
        pt = st.mean(r.get("prompt_tokens") or 0 for r in rows)
        ct = st.mean(r.get("completion_tokens") or 0 for r in rows)
        lat = st.median(r.get("latency_s") or 0 for r in rows)
        out.append(f"| {t} | {n} | {acc:.1%} | {leak} | {run} | {err} | {pt:.0f} | {ct:.0f} | {lat:.2f} |")
    if REF not in data:
        continue
    ref = per_item(data[REF])
    out.append(f"\nPaired against **{REF}** (items both ran; item score = mean over repeats; CI = item bootstrap):\n")
    out.append("| variant | items | diff | 95% CI | items better / worse | sign-test p |")
    out.append("|---|---|---|---|---|---|")
    rnd = random.Random(1)
    for t, rows in data.items():
        if t == REF:
            continue
        b = per_item(rows)
        ids = sorted(set(ref) & set(b))
        if not ids:
            continue
        diffs = [b[i] - ref[i] for i in ids]
        m = st.mean(diffs)
        boots = sorted(st.mean(rnd.choices(diffs, k=len(diffs))) for _ in range(2000))
        lo, hi = boots[50], boots[1949]
        better = sum(x > 0 for x in diffs); worse = sum(x < 0 for x in diffs)
        out.append(f"| {t} | {len(ids)} | {m * 100:+.1f} pt | {lo * 100:+.1f} .. {hi * 100:+.1f} | {better} / {worse} | {sign_p(better, worse):.3f} |")
    if bench == "bfcl":
        subsets = sorted({r["subset"] for rows in data.values() for r in rows})
        out.append("\n| subset | " + " | ".join(data) + " |")
        out.append("|---|" + "---|" * len(data))
        for s in subsets:
            cells = []
            for t, rows in data.items():
                rs = [r for r in rows if r["subset"] == s]
                cells.append(f"{sum(r.get('score', 0) for r in rs) / len(rs):.1%}" if rs else "-")
            out.append(f"| {s} | " + " | ".join(cells) + " |")
print("\n".join(out))
