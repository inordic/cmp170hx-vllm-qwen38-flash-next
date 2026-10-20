#!/usr/bin/env python3
"""Pick the best speed config: score = geometric mean of code decode and code-edit (thinking off) decode.
A candidate must pass func + multiturn and keep 8K/32K prefill within 3% of the reference.
usage: select.py REF NAME... -> prints 'BEST <name> <score> <gain vs ref>' (REF itself if nothing beats it)"""
import json, math, os, sys

N = os.environ.get("N4DIR", "/root/tune/night4")
ref, names = sys.argv[1], sys.argv[2:]


def load(n):
    try:
        return json.load(open(f"{N}/res-{n}.json"))
    except Exception:
        return None


def score(r):
    if not r or not r.get("code") or not r.get("edit_off"):
        return 0.0
    return math.sqrt(r["code"] * r["edit_off"])


R = load(ref)
base = score(R)
best, best_s = ref, base
for n in names:
    r = load(n)
    if not r or not (r.get("func_ok") and r.get("multi_ok")):
        print(f"skip {n}: missing or failed checks"); continue
    ok_pp = all(r.get("pp", {}).get(str(k), r.get("pp", {}).get(k, 0)) >= 0.97 * R["pp"].get(str(k), R["pp"].get(k, 0))
                for k in (8192, 32768) if R and R.get("pp"))
    s = score(r)
    print(f"{n}: score {s:.1f} ({s / base - 1:+.1%} vs {ref}), prefill ok {ok_pp}")
    if ok_pp and s > best_s:
        best, best_s = n, s
print(f"BEST {best} {best_s:.1f} {best_s / base - 1:+.3f}")
