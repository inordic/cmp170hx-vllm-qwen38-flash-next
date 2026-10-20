#!/usr/bin/env python3
"""T1: does a stream lose speculative decoding when several run in parallel? ("dead drafter" check)

Idea from azampatti/Qwen3.8-Flash-Next-Int4-FAST tools/mtp_drafter_check.sh (their bug: with 3, 5, 6 or 7 running
requests the drafter replayed a padded CUDA graph whose spare rows read stale request ids, and one request then
drafted at ~1 token per step for the rest of its turn). Independent implementation, standard library only.

Method: one stream alone (baseline), then phases of N parallel streams started a moment apart. Every streamed chunk
is one engine step for that request, so tokens/chunk = accepted drafts + 1. A stream whose last 40 % of chunks
average ~1.0 token/step while the baseline is clearly above 1 is reported DEAD. Spec-decode counters from /metrics
are shown when available. Run against an otherwise idle server.

usage: python3 T1-mtp-drafter-check.py [--base http://127.0.0.1:8000] [--model M] [--phases 1,3,5,6,7]
                                      [--tokens 400] [--temperature 0] [--top-p 1] [--top-k -1] [--api-key KEY]
exit: 0 PASS, 1 FAIL, 2 not testable / error
"""
import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="")
ap.add_argument("--api-key", default="")
ap.add_argument("--phases", default="1,3,5,6,7")
ap.add_argument("--tokens", type=int, default=400)
ap.add_argument("--stagger", type=float, default=0.7)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=-1)
a = ap.parse_args()
BASE = a.base.rstrip("/").removesuffix("/v1")


def req(path, body=None, timeout=900):
    h = {"Content-Type": "application/json"}
    if a.api_key:
        h["Authorization"] = "Bearer " + a.api_key
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(urllib.request.Request(BASE + path, data, h), timeout=timeout)


def metrics():
    try:
        text = req("/metrics", timeout=10).read().decode()
    except Exception:
        return None
    out = {}
    for ln in text.splitlines():
        if ln.startswith("#") or " " not in ln:
            continue
        name, val = ln.rsplit(" ", 1)
        try:
            v = float(val)
        except ValueError:
            continue
        for key, prefix in (("acc", "vllm:spec_decode_num_accepted_tokens_total"),
                            ("drafts", "vllm:spec_decode_num_drafts_total"),
                            ("running", "vllm:num_requests_running")):
            if name.startswith(prefix):
                out[key] = out.get(key, 0.0) + v
    return out


TOPICS = ["the history of lighthouses", "how glaciers shape valleys", "the economics of container shipping",
          "the development of the telegraph", "how bees communicate", "the chemistry of bread baking",
          "the life of a medieval stonemason", "how radio telescopes work"]


class Stream:
    def __init__(self, label, i):
        self.label = label
        self.prompt = (f"Write a long, detailed essay of at least 1500 words about {TOPICS[i % len(TOPICS)]}. "
                       f"Plain prose, no headings or lists. (run {i} {time.time_ns() % 100000})\n\n")
        self.chunks = []
        self.tokens = None
        self.error = None

    def run(self):
        body = {"model": MODEL, "prompt": self.prompt, "max_tokens": a.tokens, "min_tokens": a.tokens,
                "temperature": a.temperature, "top_p": a.top_p, "top_k": a.top_k, "stream": True,
                "stream_options": {"include_usage": True}}
        try:
            for raw in req("/v1/completions", body):
                ln = raw.decode(errors="replace").strip()
                if not ln.startswith("data:"):
                    continue
                ln = ln[5:].strip()
                if ln == "[DONE]":
                    break
                ev = json.loads(ln)
                if ev.get("usage") and ev["usage"].get("completion_tokens") is not None:
                    self.tokens = int(ev["usage"]["completion_tokens"])
                for ch in ev.get("choices") or []:
                    if ch.get("text"):
                        self.chunks.append(len(ch["text"]))
        except urllib.error.HTTPError as e:
            self.error = f"HTTP {e.code} {e.read()[:200].decode(errors='replace')}"
        except Exception as e:  # noqa: BLE001
            self.error = str(e)[:200]

    def stats(self):
        steps, chars = len(self.chunks), sum(self.chunks)
        tokens = self.tokens if self.tokens is not None else round(chars / 4)
        if not steps or not tokens:
            return None
        k = steps if steps < 40 else max(20, int(steps * 0.4))
        tail = tokens * (sum(self.chunks[-k:]) / max(chars, 1)) / k
        return {"tokens": tokens, "steps": steps, "tps": tokens / steps, "tail": tail}


def phase(n):
    streams = [Stream(f"R{i + 1}", i) for i in range(n)]
    m0 = metrics()
    threads = []
    for s in streams:
        t = threading.Thread(target=s.run, daemon=True)
        t.start()
        threads.append(t)
        time.sleep(a.stagger)
    for t in threads:
        t.join()
    m1 = metrics()
    srv = None
    if m0 and m1 and m1.get("drafts", 0) > m0.get("drafts", 0):
        srv = 1 + (m1.get("acc", 0) - m0.get("acc", 0)) / (m1["drafts"] - m0["drafts"])
    return streams, srv


try:
    models = json.loads(req("/v1/models", timeout=20).read().decode())["data"]
except Exception as e:  # noqa: BLE001
    print(f"cannot reach {BASE}/v1/models: {e}")
    sys.exit(2)
MODEL = a.model or models[0]["id"]
m = metrics()
if m and m.get("running", 0) > 0:
    print(f"note: {int(m['running'])} other request(s) running; results are less reliable")
print(f"server {BASE}  model {MODEL}  T={a.temperature} top_p={a.top_p} top_k={a.top_k}")

phases = [int(x) for x in a.phases.split(",") if x.strip()]
if phases[0] != 1:
    phases.insert(0, 1)
base_tps, dead, checked, failed = None, 0, 0, 0
for n in phases:
    streams, srv = phase(n)
    print(f"\n{n} stream(s)" + (f"   server mean accepted length {srv:.2f}" if srv else ""))
    print(f"  {'stream':7} {'tokens':>7} {'steps':>6} {'tok/step':>9} {'last40%':>8}  status")
    for s in streams:
        st = s.stats()
        if s.error or not st:
            failed += 1
            print(f"  {s.label:7} request failed: {s.error or 'no tokens'}")
            continue
        if n == 1 and base_tps is None:
            base_tps = st["tps"]
        thr = 1.0 + min(0.2, (base_tps - 1.0) * 0.4) if base_tps else 1.1
        is_dead = n > 1 and base_tps is not None and base_tps >= 1.2 and st["tail"] < thr
        if n > 1:
            checked += 1
            dead += is_dead
        print(f"  {s.label:7} {st['tokens']:7d} {st['steps']:6d} {st['tps']:9.2f} {st['tail']:8.2f}  "
              f"{'DEAD DRAFTER' if is_dead else 'ok'}")
    if n == 1 and base_tps is None:
        print("baseline failed; cannot test")
        sys.exit(2)

print()
if base_tps < 1.2:
    print(f"NOT TESTABLE: baseline {base_tps:.2f} tokens/step, speculative decoding looks inactive")
    sys.exit(2)
if dead:
    print(f"FAIL: {dead} of {checked} parallel streams fell to ~1 token/step (baseline {base_tps:.2f})")
    sys.exit(1)
if failed:
    print(f"INCONCLUSIVE: {failed} request(s) failed")
    sys.exit(2)
print(f"PASS: {checked} parallel streams kept speculative decoding (baseline {base_tps:.2f} tokens/step)")
sys.exit(0)
