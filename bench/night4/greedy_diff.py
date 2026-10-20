#!/usr/bin/env python3
"""greedy_diff.py A B [REF] - show where the greedy answers of two configs diverge (context + both continuations),
and which of the two a third config REF agrees with at that point (near-tie flips vs systematic change)."""
import json, sys
from pathlib import Path

D = Path("/root/tune/night4")
ld = lambda n: json.loads((D / f"greedy-{n}.json").read_text())
a, b = ld(sys.argv[1]), ld(sys.argv[2])
r = ld(sys.argv[3]) if len(sys.argv) > 3 else None
for i, (x, y) in enumerate(zip(a, b)):
    if x == y:
        continue
    k = next((j for j, (u, v) in enumerate(zip(x, y)) if u != v), min(len(x), len(y)))
    print(f"#{i:2} at char {k:3}: ...{x[max(0, k - 45):k]!r}")
    print(f"     {sys.argv[1]:7}: {x[k:k + 32]!r}")
    print(f"     {sys.argv[2]:7}: {y[k:k + 32]!r}")
    if r:
        same = sys.argv[1] if r[i][:k + 32] == x[:k + 32] else (sys.argv[2] if r[i][:k + 32] == y[:k + 32] else "neither")
        print(f"     {sys.argv[3]} agrees with: {same}")
