#!/usr/bin/env python3
"""Soak test: realistic mixed load for DURATION minutes against one endpoint.

A controller changes the number of active users every 2-4 minutes (random 1..12).  Each user runs an agent-style
conversation: a document of 2K-40K tokens with hidden facts, then several follow-up turns asking for the facts
(answers are checked), with short chat turns and tool calls mixed in.  Every 5 minutes it samples the vLLM
processes' RSS, GPU memory, and logs per-window TTFT / decode / errors so drift is visible.

usage: soak_mixed.py PORT [DURATION_MIN]        log: /root/tune/night/soak.log, summary on stdout
"""
import json, os, random, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

PORT = int(sys.argv[1])
DUR = float(sys.argv[2]) * 60 if len(sys.argv) > 2 else 3600
B = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{B}/models", headers=H)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))
LOG = open(os.environ.get("NIGHT_DIR", "/root/tune/night") + "/soak.log", "a")
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
lock = threading.Lock()
events = []          # (time, kind, ttft, tg, ok, error)
T_END = time.time() + DUR
active_target = [1]


def log(msg: str) -> None:
    LOG.write(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}\n")
    LOG.flush()


def request(messages: list, tools=None, max_tokens=1500) -> tuple[dict, float, float]:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "stream": True,
            "stream_options": {"include_usage": True}}
    if tools:
        body["tools"] = tools
    req = urllib.request.Request(f"{B}/chat/completions", data=json.dumps(body).encode(), headers=H)
    t0, times, content, calls, usage = time.time(), [], [], {}, {}
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            d = (ev.get("choices") or [{}])[0].get("delta") or {}
            if d.get("content") or d.get("reasoning") or d.get("reasoning_content") or d.get("tool_calls"):
                times.append(time.time())
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                c = calls.setdefault(tc.get("index", 0), {"name": "", "args": ""})
                f = tc.get("function") or {}
                c["name"] += f.get("name") or ""
                c["args"] += f.get("arguments") or ""
            if ev.get("usage"):
                usage = ev["usage"]
    ttft = (times[0] - t0) if times else float("nan")
    tg = (usage.get("completion_tokens", 1) - 1) / max(1e-6, times[-1] - times[0]) if len(times) > 1 else float("nan")
    return {"content": "".join(content), "calls": list(calls.values())}, ttft, tg


def user(uid: int) -> None:
    rnd = random.Random(f"soak-{uid}-{time.time()}")
    while time.time() < T_END:
        if uid >= active_target[0]:
            time.sleep(2)
            continue
        tokens = rnd.choice([2048, 4096, 8192, 16384, 40000])
        n = int(tokens * 3.27)
        off = rnd.randrange(0, len(TEXT) - n)
        facts = {f"code {k}": str(rnd.randint(100000, 999999)) for k in ("RED", "GREEN", "BLUE")}
        doc, step = "", n // 4
        for i, (k, v) in enumerate(facts.items()):
            doc += TEXT[off + i * step:off + (i + 1) * step] + f"\n\nNote: the {k} is {v}.\n\n"
        msgs = [{"role": "user", "content": f"User {uid}. Document:\n\n{doc}\n\nWhat is the code RED? Answer with just the number."}]
        for turn, (k, v) in enumerate(facts.items()):
            if time.time() >= T_END:
                break
            if turn:
                msgs.append({"role": "user", "content": f"And the {k}? Just the number."})
            try:
                out, ttft, tg = request(msgs)
                ok = v in out["content"]
                with lock:
                    events.append((time.time(), f"fact{turn}", ttft, tg, ok, ""))
                msgs.append({"role": "assistant", "content": out["content"]})
                if rnd.random() < 0.4:   # a tool call turn
                    out, ttft, tg = request(msgs + [{"role": "user", "content": "Now open the file src/app/config.py."}], TOOLS)
                    ok = any(c["name"] == "read_file" and "config.py" in c["args"] for c in out["calls"])
                    with lock:
                        events.append((time.time(), "tool", ttft, tg, ok, ""))
            except Exception as e:  # noqa: BLE001
                with lock:
                    events.append((time.time(), "error", float("nan"), float("nan"), False, f"{type(e).__name__}: {e}"))
                time.sleep(5)
                break
            time.sleep(rnd.uniform(0.5, 4))


def sample_resources() -> str:
    rss = 0
    for pid in subprocess.run(["pgrep", "-f", "vllm|VLLM"], capture_output=True, text=True).stdout.split():
        try:
            rss += int([l for l in Path(f"/proc/{pid}/status").read_text().splitlines() if l.startswith("VmRSS")][0].split()[1])
        except (OSError, IndexError):
            pass
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.replace("\n", " | ").strip()
    return f"vLLM RSS {rss / 1048576:.1f} GiB, GPU mem/temp {gpu}"


def window_summary(t_from: float, t_to: float) -> str:
    with lock:
        ev = [e for e in events if t_from <= e[0] < t_to]
    if not ev:
        return "no requests"
    tt = [e[2] for e in ev if e[2] == e[2]]
    tg = [e[3] for e in ev if e[3] == e[3]]
    bad = [e for e in ev if not e[4]]
    return (f"{len(ev)} req, TTFT med {statistics.median(tt):.2f} s max {max(tt):.1f} s, decode med "
            f"{statistics.median(tg):.0f} tok/s, wrong/failed {len(bad)} ({', '.join(sorted({e[1] for e in bad}))})")


log(f"=== soak start: {DUR / 60:.0f} min, model {MODEL}, {sample_resources()}")
threads = [threading.Thread(target=user, args=(i,), daemon=True) for i in range(12)]
for t in threads:
    t.start()
start, last = time.time(), time.time()
rnd = random.Random()
next_change = time.time()
while time.time() < T_END:
    if time.time() >= next_change:
        active_target[0] = rnd.randint(1, 12)
        log(f"active users -> {active_target[0]}")
        next_change = time.time() + rnd.uniform(120, 240)
    if time.time() - last >= 300:
        log(f"window: {window_summary(last, time.time())} | {sample_resources()}")
        last = time.time()
    time.sleep(5)
for t in threads:
    t.join(timeout=600)
log(f"window: {window_summary(last, time.time())} | {sample_resources()}")
with lock:
    ev = list(events)
errs = [e for e in ev if e[1] == "error"]
wrong = [e for e in ev if not e[4] and e[1] != "error"]
facts = [e for e in ev if e[1].startswith("fact")]
tools = [e for e in ev if e[1] == "tool"]
summary = (f"SOAK {DUR / 60:.0f} min: {len(ev)} requests, errors {len(errs)}, wrong answers {len(wrong)} "
           f"(facts {sum(e[4] for e in facts)}/{len(facts)} correct, tool calls {sum(e[4] for e in tools)}/{len(tools)}), "
           f"overall {window_summary(start, time.time() + 1)}; end {sample_resources()}")
log(summary)
if errs:
    log("first errors: " + " | ".join(e[5] for e in errs[:3]))
print(summary)
