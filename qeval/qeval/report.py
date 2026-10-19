"""Summaries and paired model comparisons for qeval results (stdlib only)."""
import glob
import math
import os
import random
import statistics

from .common import read_jsonl, wilson

CAPS = [2048, 4096, 8192, 12288, 16384, 24576, 32768]


def _load(path):
    recs = {}
    for r in read_jsonl(path):
        if r.get("transport_error"):
            continue  # server unreachable: not a model result (these items are retried on resume)
        recs[(r["id"], r.get("repeat", 0))] = r  # last one wins (resume / rescore)
    return list(recs.values())


def _pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 3)


def _block(recs):
    n = len(recs)
    if n == 0:
        return {"n": 0}
    k = sum(1 for r in recs if r.get("correct"))
    run = sum(1 for r in recs if r.get("runaway"))
    lo, hi = wilson(k, n)
    rlo, rhi = wilson(run, n)
    toks = [r.get("completion_tokens") or 0 for r in recs]
    nr = [r for r in recs if not r.get("runaway")]
    out = {
        "n": n, "correct": k, "acc": round(k / n, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "runaway": run, "runaway_rate": round(run / n, 4), "runaway_ci95": [round(rlo, 4), round(rhi, 4)],
        "runaway_no_answer": sum(1 for r in recs if r.get("runaway_no_answer")),
        "looping": sum(1 for r in recs if r.get("looping")),
        "acc_excl_runaway": round(sum(1 for r in nr if r.get("correct")) / len(nr), 4) if nr else None,
        "errors_api": sum(1 for r in recs if r.get("api_error")),
        "errors_harness": sum(1 for r in recs if r.get("harness_error")),
        "raw_tool_markup": sum(1 for r in recs if r.get("raw_tool_markup")),
        "tokens_mean": round(statistics.mean(toks), 1), "tokens_median": _pct(toks, 0.5), "tokens_p90": _pct(toks, 0.9),
        "reasoning_tokens_mean": round(statistics.mean([r.get("est_reasoning_tokens") or 0 for r in recs]), 1),
        "prompt_tokens_mean": round(statistics.mean([r.get("prompt_tokens") or 0 for r in recs]), 1),
        "calls_mean": round(statistics.mean([r.get("n_calls") or 0 for r in recs]), 2),
        "latency_median_s": _pct([r.get("latency_s") for r in recs], 0.5),
        "latency_p90_s": _pct([r.get("latency_s") for r in recs], 0.9),
        "ttft_median_s": _pct([r.get("ttft_s") for r in recs], 0.5),
        "t_answer_median_s": _pct([r.get("t_answer_s") for r in recs], 0.5),
    }
    # accuracy if the per-call token cap had been lower (pass only if it also fit under the cap)
    out["acc_at_cap"] = {str(c): round(sum(1 for r in recs if r.get("correct") and
                                           (r.get("max_call_completion_tokens") or 0) <= c) / n, 4) for c in CAPS}
    return out


def summarize_records(recs):
    s = {"overall": _block(recs), "by_subset": {}}
    keys = {}
    for r in recs:
        keys.setdefault(r.get("subset") or "all", []).append(r)
    for k, v in sorted(keys.items()):
        s["by_subset"][k] = _block(v)
    if recs and recs[0].get("bench") == "longctx":
        for field in ("kind", "bin"):
            g = {}
            for r in recs:
                g.setdefault(str(r.get(field)), []).append(r)
            s[f"by_{field}"] = {k: _block(v) for k, v in sorted(g.items())}
        g = {}
        for r in recs:
            d = r.get("depth") or 0
            g.setdefault(f"depth_{min(4, int(d * 5)) * 20}-{min(4, int(d * 5)) * 20 + 20}%", []).append(r)
        s["by_depth"] = {k: _block(v) for k, v in sorted(g.items())}
    return s


def summarize_file(path):
    return summarize_records(_load(path))


# ------------------------------------------------------------------ paired comparison


def _binom_two_sided(b, c):
    """Exact McNemar: two-sided binomial test of b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _per_item(recs):
    d = {}
    for r in recs:
        d.setdefault(r["id"], []).append(1.0 if r.get("correct") else 0.0)
    return {k: sum(v) / len(v) for k, v in d.items()}, {k: len(v) for k, v in d.items()}


def _groups(recs):
    return {r["id"]: (r.get("group") or r["id"]) for r in recs}


def paired(recs_a, recs_b, boot=10000, seed=1):
    ia, na = _per_item(recs_a)
    ib, nb = _per_item(recs_b)
    ids = sorted(set(ia) & set(ib))
    if not ids:
        return None
    grp = _groups(recs_a)
    diffs = {i: ia[i] - ib[i] for i in ids}
    n = len(ids)
    mean = sum(diffs.values()) / n
    by_g = {}
    for i in ids:
        by_g.setdefault(grp.get(i, i), []).append(diffs[i])
    glist = list(by_g.values())
    rnd = random.Random(seed)
    bs = []
    clustered = len(glist) < n
    for _ in range(boot):
        if clustered:
            pick = rnd.choices(glist, k=len(glist))
            tot = sum(sum(g) for g in pick)
            cnt = sum(len(g) for g in pick)
            bs.append(tot / cnt)
        else:
            vals = list(diffs.values())
            bs.append(sum(rnd.choices(vals, k=n)) / n)
    bs.sort()
    lo, hi = bs[int(0.025 * boot)], bs[int(0.975 * boot) - 1]
    binary = all(na[i] == 1 and nb[i] == 1 for i in ids)
    out = {"n_items": n, "acc_a": round(sum(ia[i] for i in ids) / n, 4), "acc_b": round(sum(ib[i] for i in ids) / n, 4),
           "diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)], "clustered_bootstrap": clustered}
    if binary:
        b = sum(1 for i in ids if ia[i] == 1 and ib[i] == 0)
        c = sum(1 for i in ids if ia[i] == 0 and ib[i] == 1)
        out.update({"a_only": b, "b_only": c, "p_mcnemar": round(_binom_two_sided(b, c), 4)})
    else:
        # sign-flip permutation test on per-item mean differences
        vals = [d for d in diffs.values() if d != 0]
        obs = abs(sum(vals))
        ge = 0
        for _ in range(boot):
            s = sum(v if rnd.random() < 0.5 else -v for v in vals)
            if abs(s) >= obs - 1e-12:
                ge += 1
        out["p_signflip"] = round((ge + 1) / (boot + 1), 4)
    return out


def _bench_files(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        name = os.path.basename(p)[:-6]
        if name.endswith(".rescored"):
            continue
        resc = p.replace(".jsonl", ".rescored.jsonl")
        out[name] = resc if os.path.exists(resc) else p
    return out


def compare_dirs(dirs, labels=None, boot=10000):
    labels = labels or [os.path.relpath(d) for d in dirs]
    files = [_bench_files(d) for d in dirs]
    benches = sorted(set.intersection(*[set(f) for f in files])) if files else []
    res = {"labels": labels, "dirs": dirs, "benches": {}}
    for b in benches:
        recs = [_load(f[b]) for f in files]
        common_ids = set.intersection(*[{r["id"] for r in rs} for rs in recs])
        recs_c = [[r for r in rs if r["id"] in common_ids] for rs in recs]
        entry = {"n_common_items": len(common_ids), "models": {}, "pairs": {}}
        for lab, rs, full in zip(labels, recs_c, recs):
            s = summarize_records(rs)
            entry["models"][lab] = {"common": s["overall"], "by_subset": {k: {kk: v[kk] for kk in (
                "n", "acc", "ci95", "runaway_rate", "tokens_mean")} for k, v in s["by_subset"].items()},
                "n_total_run": len(full)}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                entry["pairs"][f"{labels[i]} - {labels[j]}"] = paired(recs_c[i], recs_c[j], boot=boot)
        res["benches"][b] = entry
    # macro average over benches (equal weight), stratified bootstrap
    if len(benches) > 1:
        res["macro"] = {}
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                key = f"{labels[i]} - {labels[j]}"
                per = [res["benches"][b]["pairs"][key] for b in benches if res["benches"][b]["pairs"].get(key)]
                if not per:
                    continue
                diff = sum(p["diff"] for p in per) / len(per)
                # approximate CI by combining per-bench bootstrap half-widths as independent normals
                se = math.sqrt(sum(((p["ci95"][1] - p["ci95"][0]) / 3.92) ** 2 for p in per)) / len(per)
                res["macro"][key] = {"diff": round(diff, 4), "ci95": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
                                     "benches": len(per)}
    return res


def to_markdown(res):
    L = []
    labels = res["labels"]
    for b, e in res["benches"].items():
        L.append(f"## {b}  (items run by every model: {e['n_common_items']})\n")
        L.append("| model | acc | 95% CI | runaway | tokens/item | reasoning tok | median latency s |")
        L.append("|---|---|---|---|---|---|---|")
        for lab in labels:
            m = e["models"][lab]["common"]
            if not m.get("n"):
                continue
            L.append(f"| {lab} | {m['acc']*100:.1f}% ({m['correct']}/{m['n']}) | {m['ci95'][0]*100:.1f}-{m['ci95'][1]*100:.1f} "
                     f"| {m['runaway_rate']*100:.1f}% ({m['runaway']}) | {m['tokens_mean']:.0f} | {m['reasoning_tokens_mean']:.0f} "
                     f"| {m['latency_median_s']} |")
        L.append("")
        L.append("| pair (A - B) | diff | 95% CI | A-only / B-only | p |")
        L.append("|---|---|---|---|---|")
        for k, p in e["pairs"].items():
            if not p:
                continue
            pv = p.get("p_mcnemar", p.get("p_signflip"))
            ab = f"{p.get('a_only', '-')} / {p.get('b_only', '-')}"
            L.append(f"| {k} | {p['diff']*100:+.1f} pt | {p['ci95'][0]*100:+.1f} .. {p['ci95'][1]*100:+.1f} | {ab} | {pv} |")
        subs = sorted({s for lab in labels for s in e["models"][lab]["by_subset"]})
        if len(subs) > 1:
            L.append("")
            L.append("| subset | " + " | ".join(labels) + " |")
            L.append("|---|" + "---|" * len(labels))
            for s in subs:
                row = []
                for lab in labels:
                    v = e["models"][lab]["by_subset"].get(s)
                    row.append(f"{v['acc']*100:.1f}% (n={v['n']}, run {v['runaway_rate']*100:.0f}%)" if v else "-")
                L.append(f"| {s} | " + " | ".join(row) + " |")
        L.append("")
    if res.get("macro"):
        L.append("## Macro average over benchmarks (equal weight)\n")
        L.append("| pair | diff | approx 95% CI |")
        L.append("|---|---|---|")
        for k, p in res["macro"].items():
            L.append(f"| {k} | {p['diff']*100:+.1f} pt | {p['ci95'][0]*100:+.1f} .. {p['ci95'][1]*100:+.1f} |")
    return "\n".join(L) + "\n"
