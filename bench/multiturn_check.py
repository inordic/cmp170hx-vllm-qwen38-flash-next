#!/usr/bin/env python3
"""Multi-turn correctness + prefix-cache check (agent-style conversations).

Known risk in our vLLM tree: prefix caching on hybrid (GDN) models with a drafter (MTP) can restore the wrong
recurrent state after a cache hit (upstream fix #55601) -> wrong or garbled answers on follow-up turns.

Each conversation: a long shared document holding several facts, then 5 turns. Every turn asks for one fact
(answer must contain it) and resends the whole history like a chat client (thinking dropped from history).
Checks: answer correct, no degenerate repetition, TTFT of follow-up turns (prefix-cache hits should make them
much faster than turn 1), and vLLM's prefix-cache hit counter.

usage: multiturn_check.py PORT [CONVERSATIONS] [DOC_TOKENS]
"""
import json, random, re, sys, time, urllib.request
import os
from pathlib import Path

PORT = int(sys.argv[1])
CONVS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
DOC_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
BASE = f"http://127.0.0.1:{PORT}"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/v1/models", headers=HDR)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))


def metric(name: str) -> float:
    try:
        body = urllib.request.urlopen(f"{BASE}/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return float("nan")
    return sum(float(l.split()[-1]) for l in body.splitlines() if l.startswith(name) and not l.startswith("#"))


def ask(messages: list) -> tuple[str, float, float, dict]:
    body = {"model": MODEL, "messages": messages, "max_tokens": 3000, "temperature": 0.6, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    t0, t_first, content, usage = time.time(), None, [], {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if (d.get("content") or d.get("reasoning") or d.get("reasoning_content")) and t_first is None:
                t_first = time.time()
            if d.get("content"):
                content.append(d["content"])
            if ev.get("usage"):
                usage = ev["usage"]
    return "".join(content).strip(), (t_first or time.time()) - t0, time.time() - t0, usage


def degenerate(s: str) -> bool:
    words = s.split()
    if len(words) >= 40:
        top = max(words.count(w) for w in set(words))
        if top / len(words) > 0.3:
            return True
    return bool(re.search(r"(.{6,40})\1{5,}", s))


q0, h0 = metric("vllm:prefix_cache_queries"), metric("vllm:prefix_cache_hits")
ok_all = True
for c in range(CONVS):
    rnd = random.Random(f"conv-{c}-{time.time()}")
    n = int(DOC_TOKENS * 3.27)
    off = rnd.randrange(0, len(TEXT) - n)
    doc = TEXT[off:off + n]
    facts = {f"code {name}": str(rnd.randint(100000, 999999)) for name in ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO")}
    pieces, step = [], len(doc) // (len(facts) + 1)
    for i, (k, v) in enumerate(facts.items()):
        pieces.append(doc[i * step:(i + 1) * step] + f"\n\nNote: the {k} is {v}.\n\n")
    pieces.append(doc[len(facts) * step:])
    messages = [{"role": "system", "content": "You are a careful assistant. Answer briefly."},
                {"role": "user", "content": f"Conversation {c}. Read this document:\n\n" + "".join(pieces)
                 + "\n\nWhat is the code ALPHA? Answer with just the number."}]
    for turn, (k, v) in enumerate(facts.items()):
        if turn:
            messages.append({"role": "user", "content": f"And what is the {k}? Answer with just the number."})
        ans, ttft, total, usage = ask(messages)
        good = v in ans and not degenerate(ans)
        ok_all &= good
        print(f"conv {c} turn {turn + 1}: prompt {usage.get('prompt_tokens', 0):,} tok, TTFT {ttft:5.2f}s, "
              f"total {total:5.1f}s, answer {ans[:30]!r} -> {'PASS' if good else 'FAIL (want ' + v + ')'}", flush=True)
        messages.append({"role": "assistant", "content": ans})
q1, h1 = metric("vllm:prefix_cache_queries"), metric("vllm:prefix_cache_hits")
print(f"prefix cache: {h1 - h0:,.0f} of {q1 - q0:,.0f} queried tokens were hits "
      f"({(h1 - h0) / max(1, q1 - q0):.0%})")
print("MULTITURN ALL PASS" if ok_all else "MULTITURN SOME FAILED")
