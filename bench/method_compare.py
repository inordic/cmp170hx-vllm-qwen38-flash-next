#!/usr/bin/env python3
"""Prefill measured the way the fastest public builds measure it (iIIusi0n/2dameneko bench_prefill.py):
deterministic random token IDs via /v1/completions, max_tokens=1, whole-request latency. Median of REPEATS.
usage: method_compare.py NAME PORT [REPEATS]       -> one table on stdout
"""
import json, random, statistics, sys, time, urllib.request
import os
from pathlib import Path
NAME, PORT = sys.argv[1], int(sys.argv[2]); REP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
B = f"http://127.0.0.1:{PORT}/v1"; K = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
H = {"Authorization": f"Bearer {K}", "Content-Type": "application/json"}
M = json.load(urllib.request.urlopen(urllib.request.Request(f"{B}/models", headers=H)))["data"][0]["id"]
print(f"| {NAME} (random token ids, max_tokens=1) | prompt | latency (median) | prefill tok/s (median, min-max) |")
for n in (2048, 8192, 32768, 131072):
    rows = []
    for r in range(REP + 1):   # first run per size = warm-up, discarded
        rng = random.Random(7391 + n * 7 + r + 1000 * (r == 0))
        toks = [rng.randrange(1000, 30000) for _ in range(n)]
        body = {"model": M, "prompt": toks, "max_tokens": 1, "temperature": 0, "ignore_eos": True}
        t = time.perf_counter()
        d = json.load(urllib.request.urlopen(urllib.request.Request(f"{B}/completions", data=json.dumps(body).encode(), headers=H), timeout=1800))
        dt = time.perf_counter() - t
        if r:
            rows.append((dt, d["usage"]["prompt_tokens"] / dt))
    pps = [x[1] for x in rows]
    print(f"| | {n:,} | {statistics.median(x[0] for x in rows):.2f} s | {statistics.median(pps):,.0f} ({min(pps):,.0f}-{max(pps):,.0f}) |", flush=True)
