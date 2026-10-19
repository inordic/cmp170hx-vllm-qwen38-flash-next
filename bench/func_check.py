#!/usr/bin/env python3
"""Functional check of an OpenAI-compatible endpoint: plain answer, tool call, needle in a haystack.

usage: func_check.py PORT
"""
import json, random, sys, time, urllib.request
import os
from pathlib import Path

PORT = int(sys.argv[1])
BASE = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/models", headers=HDR)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))


def chat(body: dict) -> tuple[dict, float]:
    body = {"model": MODEL, **body}
    t = time.time()
    req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    d = json.load(urllib.request.urlopen(req, timeout=1800))
    return d, time.time() - t


ok = True
d, dt = chat({"messages": [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}],
              "max_tokens": 2000, "temperature": 0})
m = d["choices"][0]["message"]
ans = (m.get("content") or "").strip()
print(f"[math] {ans!r} in {dt:.1f}s, reasoning {len(m.get('reasoning_content') or m.get('reasoning') or '')} chars,"
      f" {d['usage']['completion_tokens']} tok -> {'PASS' if '391' in ans else 'FAIL'}")
ok &= "391" in ans

tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
d, dt = chat({"messages": [{"role": "user", "content": "What's the weather in Piteå right now?"}], "tools": tools,
              "max_tokens": 2000, "temperature": 0})
tc = d["choices"][0]["message"].get("tool_calls") or []
good = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "pite" in tc[0]["function"]["arguments"].lower()
print(f"[tool] {[(t['function']['name'], t['function']['arguments']) for t in tc]} in {dt:.1f}s -> {'PASS' if good else 'FAIL'}")
ok &= good

for tokens in (32768, 131072):
    for depth in (0.1, 0.5, 0.9):
        code = f"{random.randint(100000, 999999)}"
        n = int(tokens * 3.27)
        off = random.randrange(0, len(TEXT) - n)
        hay = TEXT[off:off + n]
        cut = int(len(hay) * depth)
        hay = hay[:cut] + f"\n\nThe secret vault code is {code}. Remember it.\n\n" + hay[cut:]
        d, dt = chat({"messages": [{"role": "user", "content": hay + "\n\nWhat is the secret vault code mentioned in "
                                    "the text above? Answer with just the number."}],
                      "max_tokens": 3000, "temperature": 0})
        ans = (d["choices"][0]["message"].get("content") or "").strip()
        good = code in ans
        ok &= good
        print(f"[needle {d['usage']['prompt_tokens']:,} tok @ {depth:.0%}] {ans[:40]!r} in {dt:.1f}s -> "
              f"{'PASS' if good else 'FAIL (want ' + code + ')'}", flush=True)
print("FUNC ALL PASS" if ok else "FUNC SOME FAILED")
