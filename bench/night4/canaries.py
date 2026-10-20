#!/usr/bin/env python3
"""Night-4 canaries (server on PORT).
  preempt PORT           one ~131K-token prompt (425K chars); vllm:num_preemptions_total before/after (Mamba state-block leak:
                         DominikBucko saw preemptions at 131K with align mode + async scheduling)
  stalerows PORT         3/5/6/7 concurrent code requests; per-request decode tok/s. A request that drafts 1 token
                         per step for its whole turn (azampatti's padded-graph stale-row bug) shows up as an outlier
                         far below the median of its batch."""
import json, os, statistics, sys, threading, time, urllib.request
from pathlib import Path

PORT = int(sys.argv[2]); B = f"http://127.0.0.1:{PORT}"


def key():
    return os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip()


H = {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"}


def metric(name):
    txt = urllib.request.urlopen(urllib.request.Request(B + "/metrics", headers=H), timeout=30).read().decode()
    return sum(float(l.split()[-1]) for l in txt.splitlines() if l.startswith(name) and not l.startswith("#"))


def stream(msgs, max_tokens, extra=None):
    body = {"model": "qwen3.8-flash-next", "messages": msgs, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 1.0, "top_p": 0.95, "top_k": 20, **(extra or {})}
    req = urllib.request.Request(B + "/v1/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0 = time.time(); tf = tl = None; usage = {}
    with urllib.request.urlopen(req, timeout=900) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for c in ev.get("choices", []):
                d = c.get("delta", {})
                if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                    tf = tf or time.time(); tl = time.time()
    gen = usage.get("completion_tokens", 0)
    return {"ttft": (tf or t0) - t0, "tg": (gen - 1) / max(1e-6, (tl or t0) - (tf or t0)), "gen": gen,
            "prompt": usage.get("prompt_tokens", 0)}


if sys.argv[1] == "preempt":
    corpus = "".join(p.read_text(errors="ignore") for p in sorted(Path("/root/tune/corpus").rglob("*.md")))
    while len(corpus) < 440_000:
        corpus += corpus
    p0 = metric("vllm:num_preemptions_total")
    r = stream([{"role": "user", "content": f"[{time.time()}]\n" + corpus[:425_000] + "\n\nSummarize the text in one sentence."}],
               64, {"chat_template_kwargs": {"enable_thinking": False}})
    p1 = metric("vllm:num_preemptions_total")
    print(f"PREEMPT canary: prompt {r['prompt']:,} tokens, TTFT {r['ttft']:.1f} s, preemptions {p1 - p0:.0f} "
          f"-> {'OK (no leak)' if p1 == p0 else 'PREEMPTIONS (Mamba state-block leak?)'}")
else:
    task = "Write a complete Python module implementing an LRU cache class with get/put, type hints and docstrings."
    worst = 1.0
    for n in (3, 5, 6, 7):
        res = [None] * n

        def go(i):
            res[i] = stream([{"role": "user", "content": f"[req {i} {time.time()}] {task}"}], 400,
                            {"chat_template_kwargs": {"enable_thinking": False}})

        ts = [threading.Thread(target=go, args=(i,)) for i in range(n)]
        [t.start() for t in ts]; [t.join() for t in ts]
        tgs = [x["tg"] for x in res]; med = statistics.median(tgs); lo = min(tgs) / med
        worst = min(worst, lo)
        print(f"STALEROWS canary n={n}: per-request decode {', '.join(f'{x:.0f}' for x in tgs)} tok/s; min/median {lo:.2f}"
              f" -> {'OK' if lo > 0.6 else 'OUTLIER (stale-row drafter bug?)'}")
    print(f"STALEROWS worst min/median {worst:.2f}")
