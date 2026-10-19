#!/usr/bin/env python3
"""Soak test for the production Strata server: mixed agent-like traffic for N minutes.

Each round picks one of: a fresh prompt (1K-64K, sometimes 128K), a follow-up turn on an earlier
conversation (prefix reuse), or a streamed request. Errors, latencies and memory are logged; a
summary is written at the end.  usage: soak.py MINUTES [OUT.json]
"""
import json, random, statistics, subprocess, sys, time, urllib.request
import os
from pathlib import Path

URL = "http://127.0.0.1:8080/v1/chat/completions"
KEY = (os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip())
STRATA = Path("/opt/Strata")
CPT = 3.27
MINUTES = float(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/tune/soak.json")
LOG = Path("/root/tune/soak.log")
rng = random.Random(42)


def log(msg):
    line = time.strftime("%H:%M:%S ") + msg
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def corpus():
    parts = []
    for pat in ("src/**/*.cpp", "docs/*.md", "serve/*.py", "tools/*.py"):
        for p in sorted(STRATA.glob(pat)):
            parts.append(p.read_text(errors="ignore"))
    return "\n\n".join(parts)


def mem():
    g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.split()
    m = dict(l.split(":") for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l)
    avail = int(m["MemAvailable"].split()[0]) // 1024
    return [int(x) for x in g], avail


def chat(messages, max_tokens, stream=False):
    body = {"model": "x", "messages": messages, "max_tokens": max_tokens, "temperature": 0.6, "stream": stream}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={
        "Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        if not stream:
            d = json.load(r)
            m = d["choices"][0]["message"]
            return d, time.time() - t0, None, m.get("content") or ""
        first, text = None, []
        for raw in r:
            line = raw.decode(errors="ignore").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            delta = (ev.get("choices") or [{}])[0].get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece and first is None:
                first = time.time() - t0
            text.append(delta.get("content") or "")
        return None, time.time() - t0, first, "".join(text)


def main():
    text = corpus()
    convs = []
    stats = {"ok": 0, "err": 0, "errors": [], "fresh_ttft": [], "follow_ttft": [], "stream_first": [], "mem": []}
    end = time.time() + MINUTES * 60
    i = 0
    while time.time() < end:
        i += 1
        kind = rng.choices(["fresh", "follow", "stream"], weights=[4, 4, 2])[0]
        if kind == "follow" and not convs:
            kind = "fresh"
        try:
            if kind == "fresh":
                tok = rng.choice([1024, 2048, 4096, 8192, 16384, 32768, 65536] + ([131072] if i % 10 == 0 else []))
                n = int(tok * CPT)
                off = rng.randrange(0, max(1, len(text) - n))
                msgs = [{"role": "user", "content": f"Session {i}.\n\n" + text[off:off + n] +
                         "\n\nSummarise this and suggest one improvement."}]
                d, wall, _, reply = chat(msgs, 200)
                t = d.get("timings", {})
                stats["fresh_ttft"].append((d["usage"]["prompt_tokens"], round(t.get("prompt_ms", 0) / 1000, 2)))
                msgs.append({"role": "assistant", "content": reply})
                convs.append(msgs)
                convs[:] = convs[-6:]
                log(f"#{i} fresh {d['usage']['prompt_tokens']} tok: ttft {t.get('prompt_ms', 0) / 1000:.2f}s "
                    f"pp {t.get('prompt_per_second')} tg {t.get('predicted_per_second')}")
            elif kind == "follow":
                msgs = rng.choice(convs)
                msgs.append({"role": "user", "content": rng.choice([
                    "Now show the code for that improvement.", "What are the risks?",
                    "Write a unit test for it.", "Explain it to a junior developer."])})
                d, wall, _, reply = chat(msgs, 200)
                t = d.get("timings", {})
                stats["follow_ttft"].append((t.get("cache_n"), t.get("prompt_n"), round(t.get("prompt_ms", 0) / 1000, 2)))
                msgs.append({"role": "assistant", "content": reply})
                log(f"#{i} follow-up: cached {t.get('cache_n')} new {t.get('prompt_n')} ttft {t.get('prompt_ms', 0) / 1000:.2f}s")
            else:
                n = int(rng.choice([1024, 4096, 16384]) * CPT)
                off = rng.randrange(0, len(text) - n)
                _, wall, first, reply = chat([{"role": "user", "content": f"Stream {i}.\n\n" + text[off:off + n] +
                                               "\n\nList three facts from this."}], 200, stream=True)
                stats["stream_first"].append(round(first or -1, 2))
                log(f"#{i} stream: first token {first:.2f}s total {wall:.1f}s, {len(reply)} chars")
            stats["ok"] += 1
        except Exception as e:
            stats["err"] += 1
            stats["errors"].append(f"#{i} {kind}: {type(e).__name__}: {e}")
            log(f"#{i} {kind} ERROR {type(e).__name__}: {e}")
            time.sleep(10)
        if i % 5 == 0:
            g, avail = mem()
            stats["mem"].append({"i": i, "gpu_mib": g, "ram_avail_mib": avail})
            log(f"   mem: GPU {g} MiB, RAM available {avail} MiB")
    summ = {
        "minutes": MINUTES, "requests": i, "ok": stats["ok"], "errors": stats["err"],
        "fresh_ttft_by_len": stats["fresh_ttft"], "follow_ttft_median_s": statistics.median([x[2] for x in stats["follow_ttft"]]) if stats["follow_ttft"] else None,
        "stream_first_median_s": statistics.median(stats["stream_first"]) if stats["stream_first"] else None,
        "mem_first": stats["mem"][:1], "mem_last": stats["mem"][-1:], "error_list": stats["errors"][:20],
    }
    OUT.write_text(json.dumps(summ, indent=1))
    log("SOAK DONE " + json.dumps({k: summ[k] for k in ("requests", "ok", "errors", "follow_ttft_median_s", "stream_first_median_s")}))


if __name__ == "__main__":
    main()
