#!/usr/bin/env python3
"""Tool-call adherence in the soak's scenario, with failure samples.

Each case: a 2K-16K-token document with a hidden fact, one fact question (answered by the model), then
"Now open the file src/app/config.py." with a read_file tool. Pass = a read_file call whose path contains config.py.
Failures are saved with finish reason, content and reasoning excerpt, so model behaviour can be told apart from
server faults. 4 cases run at a time.

usage: tool_adherence.py NAME PORT [CASES]       results: /root/tune/night/tools-NAME.json + one summary line
"""
import json, random, sys, urllib.request
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NAME, PORT = sys.argv[1], int(sys.argv[2])
CASES = int(sys.argv[3]) if len(sys.argv) > 3 else 100
B = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{B}/models", headers=H)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def chat(messages, tools=None, max_tokens=4000):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "seed": 7}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{B}/chat/completions", data=json.dumps(body).encode(), headers=H)
    return json.load(urllib.request.urlopen(req, timeout=1800))


def case(i: int) -> dict:
    rnd = random.Random(f"tool-case-{i}")
    tokens = rnd.choice([2048, 4096, 8192, 16384])
    n = int(tokens * 3.27)
    off = rnd.randrange(0, len(TEXT) - n)
    code = str(rnd.randint(100000, 999999))
    msgs = [{"role": "user", "content": TEXT[off:off + n // 2] + f"\n\nNote: the code RED is {code}.\n\n" +
             TEXT[off + n // 2:off + n] + "\n\nWhat is the code RED? Answer with just the number."}]
    d = chat(msgs)
    m = d["choices"][0]["message"]
    fact_ok = code in (m.get("content") or "")
    msgs += [{"role": "assistant", "content": m.get("content") or ""},
             {"role": "user", "content": "Now open the file src/app/config.py."}]
    d = chat(msgs, TOOLS)
    m = d["choices"][0]["message"]
    tc = m.get("tool_calls") or []
    ok = any(t["function"]["name"] == "read_file" and "config.py" in (t["function"]["arguments"] or "") for t in tc)
    return {"i": i, "tokens": tokens, "fact_ok": fact_ok, "tool_ok": ok, "finish": d["choices"][0]["finish_reason"],
            "calls": [(t["function"]["name"], t["function"]["arguments"]) for t in tc],
            "content": (m.get("content") or "")[:300], "reasoning_tail": (m.get("reasoning") or m.get("reasoning_content") or "")[-300:]}


with ThreadPoolExecutor(4) as ex:
    res = list(ex.map(case, range(CASES)))
Path(f"/root/tune/night/tools-{NAME}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
fails = [r for r in res if not r["tool_ok"]]
kinds = {}
for r in fails:
    k = "length" if r["finish"] == "length" else ("other-tool" if r["calls"] else "text-answer")
    kinds[k] = kinds.get(k, 0) + 1
print(f"| {NAME} | tool calls {sum(r['tool_ok'] for r in res)}/{len(res)} | facts {sum(r['fact_ok'] for r in res)}/{len(res)} | "
      f"failure kinds {kinds} |")
for r in fails[:3]:
    print(f"   fail {r['i']}: finish={r['finish']} calls={r['calls']} content={r['content'][:150]!r}")
