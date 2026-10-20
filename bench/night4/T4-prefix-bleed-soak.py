#!/usr/bin/env python3
"""T4: prefix-cache state-bleed soak (end-to-end check for the Mamba/align/PP items: P0-0010/11/12, PP skew, leaks).

Each "case" is a long document (several Mamba blocks of 1600 tokens) with a unique code planted near the start and a
question at the end. Every case is sent COLD (unique prefix) and then WARM twice: once with the same prompt (prefix
hit up to the last block boundary before the tail) and once with an extra follow-up turn appended (hit plus a new
tail). Cases run concurrently with --conc in flight. With greedy sampling every answer must contain the case's own
code and no other case's code, and must not degenerate into a loop. A cross-request state bleed shows up as a wrong
or foreign code, a loop ("duct duct ..."), or an HTTP error.

usage: python3 T4-prefix-bleed-soak.py [--base http://127.0.0.1:8000] [--model M] [--cases 24] [--conc 6]
                                      [--blocks 5] [--api-key KEY]
Also watch the server log during the run (C4 probe line if VLLM_PP_SKEW_PROBE=1) and vllm:num_preemptions_total.
exit: 0 all answers correct, 1 failures, 2 error
"""
import argparse
import json
import random
import re
import sys
import threading
import urllib.error
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="")
ap.add_argument("--api-key", default="")
ap.add_argument("--cases", type=int, default=24)
ap.add_argument("--conc", type=int, default=6)
ap.add_argument("--blocks", type=int, default=5, help="approximate document length in 1600-token blocks")
a = ap.parse_args()
BASE = a.base.rstrip("/").removesuffix("/v1")


def post(body, timeout=1800):
    h = {"Content-Type": "application/json"}
    if a.api_key:
        h["Authorization"] = "Bearer " + a.api_key
    r = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), h)
    try:
        return 200, json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")[:300]
    except Exception as e:  # noqa: BLE001
        return -1, str(e)[:300]


try:
    h = {"Authorization": "Bearer " + a.api_key} if a.api_key else {}
    MODEL = a.model or json.loads(urllib.request.urlopen(
        urllib.request.Request(BASE + "/v1/models", None, h), timeout=20).read())["data"][0]["id"]
except Exception as e:  # noqa: BLE001
    print(f"cannot reach {BASE}: {e}")
    sys.exit(2)

WORDS = ("river stone lantern harbor meadow copper signal orchard timber canvas ledger compass beacon granite "
         "willow quarry falcon ember thistle cobalt marble prairie glacier saddle anchor").split()


def document(rng, code):
    # ~1.3 tokens per word; build roughly blocks*1600 tokens of filler sentences.
    n_words = int(a.blocks * 1600 / 1.3)
    words = [rng.choice(WORDS) for _ in range(n_words)]
    sents = [" ".join(words[i:i + 12]).capitalize() + "." for i in range(0, n_words, 12)]
    return f"Access code: {code}. Remember it.\n\n" + " ".join(sents)


CODES = {}
fails = []
lock = threading.Lock()


def check(case, tag, code, resp):
    status, body = resp
    if status != 200:
        return f"HTTP {status}: {body}"
    text = body["choices"][0]["message"].get("content") or ""
    found = set(re.findall(r"[A-Z]{3}-\d{4}", text))
    if code not in found:
        foreign = found & set(CODES.values())
        return f"missing own code (found {sorted(found)}{' FOREIGN' if foreign else ''}): {text[:80]!r}"
    if re.search(r"(\b\w+\b)(\s+\1\b){8,}", text):
        return f"loop: {text[:80]!r}"
    return None


def run_case(i):
    rng = random.Random(1000 + i)
    code = f"{''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ') for _ in range(3))}-{rng.randint(1000, 9999)}"
    with lock:
        CODES[i] = code
    doc = document(rng, code)
    q = "What is the access code stated at the very beginning of the text? Answer with the code only."
    base = [{"role": "user", "content": doc + "\n\n" + q}]
    params = {"model": MODEL, "temperature": 0, "max_tokens": 32,
              "chat_template_kwargs": {"enable_thinking": False}}
    r1 = post({**params, "messages": base})
    r2 = post({**params, "messages": base})
    turn = base + [{"role": "assistant", "content": code},
                   {"role": "user", "content": "Repeat the access code once more, code only."}]
    r3 = post({**params, "messages": turn})
    for tag, r in (("cold", r1), ("warm", r2), ("turn", r3)):
        err = check(i, tag, code, r)
        with lock:
            print(f"  case {i:3d} {tag}: {'ok' if err is None else 'FAIL ' + err}", flush=True)
            if err:
                fails.append((i, tag, err))


sem = threading.Semaphore(a.conc)


def worker(i):
    with sem:
        run_case(i)


print(f"model {MODEL}: {a.cases} cases x 3 requests, {a.conc} in flight, ~{a.blocks * 1600} prompt tokens")
ts = [threading.Thread(target=worker, args=(i,)) for i in range(a.cases)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print(f"\n{3 * a.cases - len(fails)}/{3 * a.cases} answers correct")
sys.exit(1 if fails else 0)
