#!/usr/bin/env python3
"""How much does the n-gram table on NVMe cost during prefill?

Sends fresh 32K and 128K prompts (one at a time) and samples the storage bytes read by all vLLM processes
(/proc/<pid>/io read_bytes, which counts O_DIRECT reads) between send and first token. After start-up the weights
are in VRAM, so these reads are the PLE table rows. Reports bytes per prompt token and the read rate the prefill
needed.

usage: ple_disk.py PORT
"""
import json, random, subprocess, sys, time, urllib.request
import os
from pathlib import Path

PORT = int(sys.argv[1])
BASE = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/models", headers=HDR)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))


def vllm_read_bytes() -> int:
    pids = subprocess.run(["pgrep", "-f", "vllm|VLLM"], capture_output=True, text=True).stdout.split()
    total = 0
    for pid in pids:
        try:
            for line in Path(f"/proc/{pid}/io").read_text().splitlines():
                if line.startswith("read_bytes:"):
                    total += int(line.split()[1])
        except OSError:
            pass
    return total


for tokens in (32768, 131072):
    n = int(tokens * 3.27)
    off = random.randrange(0, len(TEXT) - n)
    p = f"Request disk-{random.random()}\n\n" + TEXT[off:off + n] + "\n\nSummarise the text above in one sentence."
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 32,
            "messages": [{"role": "user", "content": p}]}
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    b0, t0 = vllm_read_bytes(), time.time()
    t_first = b_first = None
    usage = {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and t_first is None:
                t_first, b_first = time.time(), vllm_read_bytes()
            if ev.get("usage"):
                usage = ev["usage"]
    pt = usage.get("prompt_tokens", tokens)
    ttft, mb = t_first - t0, (b_first - b0) / 1e6
    print(f"disk {pt:,} tok: TTFT {ttft:.2f} s, read {mb:,.0f} MB during prefill = {mb * 1e6 / pt / 1024:.1f} KiB/token, "
          f"{mb / ttft:,.0f} MB/s needed", flush=True)
