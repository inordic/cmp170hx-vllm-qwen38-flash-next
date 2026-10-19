#!/usr/bin/env python3
"""Production checks for the vLLM endpoint on 8080: model aliases, default + per-request reasoning effort,
tool call through the chromix template, Anthropic /v1/messages, streaming TTFT, API key enforcement."""
import json, sys, time, urllib.error, urllib.request
import os
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
B = f"http://127.0.0.1:{PORT}"
K = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
H = {"Authorization": f"Bearer {K}", "Content-Type": "application/json"}


def post(path, body, headers=H, timeout=600):
    req = urllib.request.Request(B + path, data=json.dumps(body).encode(), headers=headers)
    return json.load(urllib.request.urlopen(req, timeout=timeout))


ok = True
models = [m["id"] for m in json.load(urllib.request.urlopen(urllib.request.Request(B + "/v1/models", headers=H)))["data"]]
print("models:", models)
q = "A farmer has 17 sheep. All but 9 run away. Then he buys twice as many as he has left. How many sheep now? Answer with a number."
for effort in (None, "low", "medium", "high"):
    body = {"model": "swift-1.5-iq3_xxs", "messages": [{"role": "user", "content": q}], "max_tokens": 6000, "temperature": 0}
    if effort:
        body["reasoning_effort"] = effort
    d = post("/v1/chat/completions", body)
    m = d["choices"][0]["message"]
    r = m.get("reasoning") or m.get("reasoning_content") or ""
    good = "27" in (m.get("content") or "")
    ok &= good
    print(f"effort={effort or '(default)':9} reasoning {len(r):5d} chars, completion {d['usage']['completion_tokens']:4d} tok, "
          f"answer {(m.get('content') or '').strip()[:20]!r} -> {'PASS' if good else 'FAIL'}")
hard = ("Find all integer solutions (x, y) with 0 <= x, y <= 50 of x*y - 3*x - 4*y = 5. "
        "List them and explain briefly.")
for effort in ("low", "medium", "high"):
    d = post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "max_tokens": 12000, "temperature": 0,
             "reasoning_effort": effort, "messages": [{"role": "user", "content": hard}]})
    m = d["choices"][0]["message"]
    r = m.get("reasoning") or m.get("reasoning_content") or ""
    print(f"hard question, effort={effort:6}: reasoning {len(r):6d} chars, completion {d['usage']['completion_tokens']:5d} tok")
tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
d = post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "tools": tools, "max_tokens": 3000, "temperature": 0,
         "messages": [{"role": "user", "content": "Open the file src/main.py and tell me what it does."}]})
tc = d["choices"][0]["message"].get("tool_calls") or []
good = bool(tc) and tc[0]["function"]["name"] == "read_file" and "src/main.py" in tc[0]["function"]["arguments"]
ok &= good
print("tool call:", [(t["function"]["name"], t["function"]["arguments"]) for t in tc], "->", "PASS" if good else "FAIL")
if good:  # round trip with the tool result
    msgs = [{"role": "user", "content": "Open the file src/main.py and tell me what it does."},
            {"role": "assistant", "content": None, "tool_calls": tc},
            {"role": "tool", "tool_call_id": tc[0]["id"], "content": "def main():\n    print('hello from main')\n"}]
    d = post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "tools": tools, "max_tokens": 3000,
             "temperature": 0, "messages": msgs})
    c = d["choices"][0]["message"].get("content") or ""
    good = "hello" in c.lower() or "print" in c.lower()
    ok &= good
    print("tool result round trip:", repr(c[:80]), "->", "PASS" if good else "FAIL")
ah = {"x-api-key": K, "anthropic-version": "2023-06-01", "Content-Type": "application/json"}
d = post("/v1/messages", {"model": "qwen3.8-flash-next", "max_tokens": 2000,
         "messages": [{"role": "user", "content": "What is 6*7? Answer with just the number."}]}, headers=ah)
txt = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
good = "42" in txt
ok &= good
print("anthropic /v1/messages:", repr(txt[:40]), [b.get("type") for b in d.get("content", [])], "->", "PASS" if good else "FAIL")
t0 = time.time()
req = urllib.request.Request(B + "/v1/chat/completions", headers=H, data=json.dumps(
    {"model": "qwen3.8-flash-next", "stream": True, "max_tokens": 50,
     "messages": [{"role": "user", "content": "Say hi."}]}).encode())
with urllib.request.urlopen(req, timeout=120) as r:
    for raw in r:
        if raw.startswith(b"data:") and b'"delta"' in raw and (b'"content"' in raw or b'"reasoning' in raw):
            print(f"streaming first token: {time.time() - t0:.2f} s")
            break
try:
    post("/v1/chat/completions", {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": "hi"}]},
         headers={"Content-Type": "application/json"})
    print("no key: accepted -> FAIL"); ok = False
except urllib.error.HTTPError as e:
    print(f"no key: HTTP {e.code} -> {'PASS' if e.code == 401 else 'FAIL'}"); ok &= e.code == 401
print("PROD ALL PASS" if ok else "PROD SOME FAILED")
