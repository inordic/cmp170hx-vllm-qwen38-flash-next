#!/usr/bin/env python3
"""Night 5 decision: decide.py REF CAND...  (candidates in increasing completeness, e.g. C1 C2 C3)

Picks the most complete candidate that passes every gate, compared with REF on the same night and harness:
  quality  BFCL >= REF - 1.5 pt, EvalPlus >= REF - 2.0 pt, long context >= REF - 2.0 pt
  speed    mean median task latency of BFCL and EvalPlus <= 1.10 x REF (what a user waits for an answer)
  checks   func + multiturn PASS (res-NAME.json); INT8 KV candidates also need $N/k1.ok (all K1 checks passed)
Prints a table and 'PICK NAME' (or 'PICK none' -> production stays as it is)."""
import json, os, sys
from pathlib import Path

N = Path(os.environ.get("N5DIR", "/root/tune/night5"))
R = Path(os.environ.get("QRES", "/root/qeval/results"))
MARGIN = {"bfcl": 0.015, "evalplus": 0.02, "longctx": 0.02}


def suite(name):
    out = {}
    for b in MARGIN:
        try:
            o = json.load(open(R / f"n5-{name}" / "medium" / f"{b}.summary.json"))["overall"]
            out[b] = (o["acc"], o.get("latency_median_s"), o.get("n"))
        except Exception:
            out[b] = None
    return out


def res(name):
    try:
        return json.load(open(N / f"res-{name}.json"))
    except Exception:
        return {}


def int8(name):
    try:
        return "KV_DTYPE=int8" in (N / "profiles" / f"{name}.env").read_text()
    except Exception:
        return False


def lat(s):
    v = [s[b][1] for b in ("bfcl", "evalplus") if s.get(b) and s[b][1]]
    return sum(v) / len(v) if len(v) == 2 else None


ref, cands = sys.argv[1], sys.argv[2:]
rs = suite(ref); rl = lat(rs)
f = lambda s, b: f"{s[b][0] * 100:.1f}" if s.get(b) else "n/a"
print(f"{'':6} {'BFCL':>6} {'EvalP':>6} {'LongC':>6} {'lat s':>6} {'code':>6}  verdict")
print(f"{ref:6} {f(rs, 'bfcl'):>6} {f(rs, 'evalplus'):>6} {f(rs, 'longctx'):>6} {rl or 0:6.2f} {res(ref).get('code', 0):6.1f}  reference")
pick = None
for c in cands:
    s, r, why = suite(c), res(c), []
    for b, m in MARGIN.items():
        if not s.get(b) or not rs.get(b):
            why.append(f"{b} missing")
        elif s[b][0] < rs[b][0] - m:
            why.append(f"{b} {s[b][0] * 100:.1f} < {rs[b][0] * 100:.1f}-{m * 100:.1f}")
    cl = lat(s)
    if not cl or not rl:
        why.append("latency missing")
    elif cl > 1.10 * rl:
        why.append(f"latency {cl:.2f} > 1.1x {rl:.2f}")
    if not (r.get("func_ok") and r.get("multi_ok")):
        why.append("func/multiturn not passed")
    if int8(c) and not (N / "k1.ok").exists():
        why.append("INT8 KV without passed K1 checks")
    ok = not why
    if ok:
        pick = c   # later = more complete; keep the last one that passes
    print(f"{c:6} {f(s, 'bfcl'):>6} {f(s, 'evalplus'):>6} {f(s, 'longctx'):>6} {cl or 0:6.2f} {r.get('code', 0):6.1f}  "
          + ("PASS" if ok else "fail: " + "; ".join(why)))
print(f"PICK {pick or 'none'}")
