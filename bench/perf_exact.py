#!/usr/bin/env python3
"""Single-stream prefill/decode A/B test with exact prompt lengths.

Prompts are cut to an exact token count with the model's own tokenizer (random window of a large text, decoded
back), each with a unique first line so the prefix cache never helps. One untimed warm-up request per size, then
RUNS timed requests per size; medians are reported.

usage: perf_exact.py NAME PORT TOKENIZER_DIR [RUNS] [GEN]     results: /root/tune/exact-NAME.json
run with the vLLM venv python (needs transformers).
"""
import json, os, random, statistics, sys, time, urllib.request
from pathlib import Path
from transformers import AutoTokenizer

NAME, PORT, TOKDIR = sys.argv[1], int(sys.argv[2]), sys.argv[3]
RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
GEN = int(sys.argv[5]) if len(sys.argv) > 5 else 256
SIZES = [int(x) for x in os.environ.get("PERF_SIZES", "1024,4096,8192,16384,32768").split(",")]
BASE = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/models", headers=HDR)))["data"][0]["id"]
tok = AutoTokenizer.from_pretrained(TOKDIR)
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))
IDS = tok.encode(TEXT, add_special_tokens=False)


def prompt(n: int, tag: str) -> str:
    off = random.Random(tag).randrange(0, len(IDS) - n)
    return f"Request {tag}\n\n" + tok.decode(IDS[off:off + n - 40]) + "\n\nSummarise the text above in a few sentences."


def one(n: int, tag: str) -> dict:
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": GEN,
            "temperature": 0.6, "messages": [{"role": "user", "content": prompt(n, tag)}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    out = {"t_send": time.time()}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                out.setdefault("t_first", time.time())
                out["t_last"] = time.time()
            if ev.get("usage"):
                out["usage"] = ev["usage"]
    p, c = out["usage"]["prompt_tokens"], out["usage"]["completion_tokens"]
    ttft = out["t_first"] - out["t_send"]
    return {"size": n, "prompt": p, "ttft": round(ttft, 3), "pp": round(p / ttft), "gen": c,
            "tg": round((c - 1) / max(1e-6, out["t_last"] - out["t_first"]), 1)}


rows = []
for n in SIZES:
    one(n, f"{NAME}-warm-{n}-{random.random()}")
for n in SIZES:
    for r in range(RUNS):
        rows.append(one(n, f"{NAME}-{n}-{r}-{random.random()}"))
        print(json.dumps(rows[-1]), flush=True)
Path(f"/root/tune/exact-{NAME}.json").write_text(json.dumps(rows, indent=1))
print(f"\n| {NAME} | prompt | TTFT (median) | prefill tok/s (median, min-max) | decode tok/s (median) |")
print("|---|---|---|---|---|")
for n in SIZES:
    rs = [x for x in rows if x["size"] == n]
    pps = [x["pp"] for x in rs]
    print(f"| | {statistics.median(x['prompt'] for x in rs):,.0f} | {statistics.median(x['ttft'] for x in rs):.2f} s | "
          f"{statistics.median(pps):,.0f} ({min(pps):,}-{max(pps):,}) | {statistics.median(x['tg'] for x in rs):.1f} |")
print("EXACT DONE")
