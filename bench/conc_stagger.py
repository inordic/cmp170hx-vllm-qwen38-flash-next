#!/usr/bin/env python3
"""Closed-loop multi-user load with staggered arrivals (how real users hit the server).

Each simulated user starts at a random offset within the first STAGGER seconds, then sends requests back to back
(fresh prompt each time, prompt length +-10 % around the target, so nothing comes from the prefix cache) until the
test window ends. Requests still running at the end are allowed to finish but only tokens inside the window count
towards the aggregate.

usage: conc_stagger.py NAME PORT SCENARIO [DURATION_S]
  SCENARIO: uN-P   N users with P-token prompts, e.g. u8-1024
            mixed  1 agent user with 32K prompts + 4 chat users with 0.5K prompts
results: appended as one JSON line to /root/tune/night/conc.jsonl, plus one table row on stdout
"""
import json, os, random, statistics, sys, threading, time, urllib.request
from pathlib import Path

NAME, PORT, SCEN = sys.argv[1], int(sys.argv[2]), sys.argv[3]
DUR = float(sys.argv[4]) if len(sys.argv) > 4 else 150
GEN, STAGGER, CPT = 256, 8.0, 3.27
BASE = f"http://127.0.0.1:{PORT}/v1"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
MODEL = json.load(urllib.request.urlopen(urllib.request.Request(f"{BASE}/models", headers=HDR)))["data"][0]["id"]
TEXT = "".join(p.read_text(errors="ignore") for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py")
               for p in sorted(Path("/root/tune/corpus/Strata").glob(pat)))

if SCEN == "mixed":
    USERS = [("agent", 32768)] + [("chat", 512)] * 4
else:
    n, p = SCEN[1:].split("-")
    USERS = [("user", int(p))] * int(n)


def prompt(tokens: int, rnd: random.Random) -> str:
    tokens = int(tokens * rnd.uniform(0.9, 1.1))
    n = int(max(64, tokens - 60) * CPT)
    off = rnd.randrange(0, len(TEXT) - n)
    return f"Request {rnd.random()}\n\n" + TEXT[off:off + n] + "\n\nSummarise the text above in a few sentences."


records, lock = [], threading.Lock()
T0 = time.time() + 1.0
T_END = T0 + DUR


def user(idx: int, kind: str, tokens: int) -> None:
    rnd = random.Random(f"{NAME}-{SCEN}-{idx}-{time.time()}")
    time.sleep(max(0.0, T0 + rnd.uniform(0, STAGGER) - time.time()))
    while time.time() < T_END:
        body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": GEN,
                "temperature": 0.6, "messages": [{"role": "user", "content": prompt(tokens, rnd)}]}
        req = urllib.request.Request(f"{BASE}/chat/completions", data=json.dumps(body).encode(), headers=HDR)
        rec = {"kind": kind, "t_send": time.time(), "tok_times": []}
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                for raw in r:
                    line = raw.decode(errors="ignore").strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    ev = json.loads(line[5:])
                    d = (ev.get("choices") or [{}])[0].get("delta") or {}
                    if d.get("content") or d.get("reasoning") or d.get("reasoning_content"):
                        rec["tok_times"].append(time.time())
                    if ev.get("usage"):
                        rec["usage"] = ev["usage"]
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        with lock:
            records.append(rec)


threads = [threading.Thread(target=user, args=(i, k, t)) for i, (k, t) in enumerate(USERS)]
for t in threads:
    t.start()
for t in threads:
    t.join()

ok = [r for r in records if "error" not in r and r["tok_times"] and "usage" in r]
errors = [r["error"] for r in records if "error" in r]


def tokens_in_window(r: dict) -> float:
    """completion tokens generated inside [T0, T_END], spread evenly over the stream chunks"""
    c, ts = r["usage"]["completion_tokens"], r["tok_times"]
    return c * sum(T0 <= t <= T_END for t in ts) / len(ts)


def stats(rs: list) -> dict:
    if not rs:
        return {}
    ttft = sorted(r["tok_times"][0] - r["t_send"] for r in rs)
    tg = [(r["usage"]["completion_tokens"] - 1) / max(1e-6, r["tok_times"][-1] - r["tok_times"][0]) for r in rs
          if len(r["tok_times"]) > 1]
    return {"requests": len(rs), "ttft_med": round(statistics.median(ttft), 2),
            "ttft_p95": round(ttft[min(len(ttft) - 1, int(0.95 * len(ttft)))], 2), "ttft_max": round(ttft[-1], 2),
            "tg_med": round(statistics.median(tg), 1) if tg else None,
            "prompt_med": int(statistics.median(r["usage"]["prompt_tokens"] for r in rs))}


res = {"name": NAME, "scenario": SCEN, "duration": DUR, "users": len(USERS), "errors": errors[:5],
       "n_errors": len(errors), "agg_out_tok_s": round(sum(tokens_in_window(r) for r in ok) / DUR, 1),
       "all": stats(ok)}
for kind in sorted({r["kind"] for r in ok}):
    res[kind] = stats([r for r in ok if r["kind"] == kind])
Path(os.environ.get("NIGHT_DIR", "/root/tune/night")).mkdir(parents=True, exist_ok=True)
with open(os.environ.get("NIGHT_DIR", "/root/tune/night") + "/conc.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
a = res["all"]
extra = ""
if SCEN == "mixed":
    extra = (f" | agent TTFT med {res['agent'].get('ttft_med')} s | chat TTFT med {res['chat'].get('ttft_med')} s "
             f"p95 {res['chat'].get('ttft_p95')} s")
print(f"| {NAME} | {SCEN} | {len(USERS)} users | {res['agg_out_tok_s']} tok/s total | {a.get('requests')} req | "
      f"TTFT med {a.get('ttft_med')} s p95 {a.get('ttft_p95')} s | tg/stream {a.get('tg_med')} | errors {len(errors)}{extra} |",
      flush=True)
