#!/usr/bin/env python3
"""accept_mb.py BEST MB -> 'ACCEPT +x.xxx' when MB (BEST + INT4 MTP experts) passes func/multiturn and its decode score
(sqrt(code x edit_off)) is at least 99% of BEST's; else 'REJECT ...'. The INT4 drafter frees 3.47 GiB on PP1 (KV)."""
import json, math, os, sys

N = os.environ.get("N4DIR", "/root/tune/night4")


def sc(n):
    try:
        r = json.load(open(f"{N}/res-{n}.json"))
    except Exception:
        return 0.0, False
    s = math.sqrt(r["code"] * r["edit_off"]) if r.get("code") and r.get("edit_off") else 0.0
    return s, bool(r.get("func_ok") and r.get("multi_ok"))


(b, _), (m, ok) = sc(sys.argv[1]), sc(sys.argv[2])
rel = f"{m / b - 1:+.3f}" if b else "n/a"
print(("ACCEPT " if ok and b and m >= 0.99 * b else "REJECT ") + rel + ("" if ok else " (func/multiturn failed)"))
