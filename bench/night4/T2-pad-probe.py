#!/usr/bin/env python3
"""T2: fresh one-token prompts while other requests decode (verifies C2, and C1 when padding is forced on).

The scheduler pads a request with one token left to the 1+k spec shape when other requests are running. On this
hybrid model a padded *fresh* prompt runs the GDN/PLE spec kernels from an uninitialised state slot (NaN logprobs ->
HTTP 400 "Out of range float values are not JSON compliant", or garbage text). vektorprime measured 11/12 failures
on stock code (mtp-spec-decode README section 7, "bug 3"). Independent implementation, standard library only.

Each trial sends a one-token prompt (completions API, token-id prompt) with logprobs while --bg background streams
decode long outputs. A trial FAILS on a non-200 answer, NaN/inf in logprobs, or an output whose first generated token
is token 0 ('!') more often than the target itself would produce it (reported as the '!' rate; C1 matters here).

usage: python3 T2-pad-probe.py [--base http://127.0.0.1:8000] [--model M] [--n 24] [--bg 6]
                              [--temperature 1.0] [--top-p 1.0] [--top-k -1] [--api-key KEY]
Compare: server with VLLM_SPEC_PAD_NEW_REQS=1 (stock padding) vs unset (C2 "auto"); expect failures only with =1.
exit: 0 all trials OK, 1 failures, 2 error
"""
import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="")
ap.add_argument("--api-key", default="")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--bg", type=int, default=6)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--top-p", type=float, default=1.0)
ap.add_argument("--top-k", type=int, default=-1)
ap.add_argument("--word", default="The", help="text whose first token is used as the one-token prompt")
a = ap.parse_args()
BASE = a.base.rstrip("/").removesuffix("/v1")


def post(path, body, timeout=1800):
    h = {"Content-Type": "application/json"}
    if a.api_key:
        h["Authorization"] = "Bearer " + a.api_key
    r = urllib.request.Request(BASE + path, json.dumps(body).encode(), h)
    try:
        return 200, urllib.request.urlopen(r, timeout=timeout).read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        return -1, str(e)


def get(path):
    h = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
    return json.loads(urllib.request.urlopen(urllib.request.Request(BASE + path, None, h), timeout=20).read())


try:
    MODEL = a.model or get("/v1/models")["data"][0]["id"]
except Exception as e:  # noqa: BLE001
    print(f"cannot reach {BASE}: {e}")
    sys.exit(2)
code, txt = post("/tokenize", {"model": MODEL, "prompt": a.word, "add_special_tokens": False})
if code != 200:
    print(f"/tokenize failed: {code} {txt[:200]}")
    sys.exit(2)
TOK = json.loads(txt)["tokens"][0]
print(f"model {MODEL}; one-token prompt = [{TOK}] ({a.word!r}); background streams: {a.bg}")

stop = threading.Event()


def background(i):
    while not stop.is_set():
        post("/v1/completions", {"model": MODEL, "prompt": f"Tell a very long story about island number {i}.",
                                 "max_tokens": 2000, "min_tokens": 2000, "temperature": 1.0, "seed": 1000 + i})


threads = [threading.Thread(target=background, args=(i,), daemon=True) for i in range(a.bg)]
for t in threads:
    t.start()
time.sleep(8)

fails, bang = 0, 0
for j in range(a.n):
    body = {"model": MODEL, "prompt": [TOK], "max_tokens": 8, "temperature": a.temperature, "top_p": a.top_p,
            "top_k": a.top_k, "seed": 77 + j, "logprobs": 5, "return_tokens_as_token_ids": True}
    code, txt = post("/v1/completions", body)
    bad, note = False, ""
    if code != 200:
        bad, note = True, f"HTTP {code}: {txt[:160]}"
    else:
        try:
            ch = json.loads(txt)["choices"][0]
            lp = ch.get("logprobs") or {}
            vals = [v for v in (lp.get("token_logprobs") or []) if v is not None]
            if any(isinstance(v, float) and (math.isnan(v) or math.isinf(v)) for v in vals):
                bad, note = True, "NaN/inf logprob"
            toks = lp.get("tokens") or []
            first = toks[0] if toks else ""
            if first in ("token_id:0", "!"):
                bang += 1
            note = note or f"text={ch.get('text', '')[:40]!r} first={first}"
        except Exception as e:  # noqa: BLE001
            bad, note = True, f"unparseable answer: {e}"
    fails += bad
    print(f"  trial {j + 1:2d}: {'FAIL' if bad else 'ok  '} {note}")
stop.set()
print(f"\n{a.n - fails}/{a.n} trials OK; first token '!' in {bang}/{a.n} "
      "(a high rate with top_p=1/top_k=-1 points at accepted placeholder drafts, i.e. C1 missing)")
sys.exit(1 if fails else 0)
