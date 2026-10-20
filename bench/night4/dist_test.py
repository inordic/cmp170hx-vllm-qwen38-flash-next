#!/usr/bin/env python3
"""Output-distribution test for speculative drafting (is probabilistic drafting unbiased?).

Samples short random-letter / random-digit answers at temperature 1.0 (thinking off) many times and stores the symbol
at every position. Positions >= 2 are produced through the drafter, so a biased draft/verify path shows up there.
  collect NAME PORT [N]      -> /root/tune/night4/dist-NAME.json  (N samples per prompt, default 1500)
  compare REF NAME...        -> TV distance per prompt (pooled over drafted positions) vs REF, plus the
                                noise floor (TV between the two halves of REF) and a permutation p-value
"""
import json, os, random, sys, threading, urllib.request
from collections import Counter
from pathlib import Path

D = Path("/root/tune/night4")
PROMPTS = {
    "letters": "Write 12 random lowercase letters a-z, separated by single spaces. Output only the letters, nothing else.",
    "digits": "Write 16 random digits 0-9, separated by single spaces. Output only the digits, nothing else.",
}
LEN = {"letters": 12, "digits": 16}


def key():
    return os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip()


def one(port, prompt, seed):
    body = {"model": "qwen3.8-flash-next", "messages": [{"role": "user", "content": prompt}], "max_tokens": 60,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["choices"][0]["message"].get("content") or ""


def collect(name, port, n):
    out = {}
    for pname, prompt in PROMPTS.items():
        res, lock = [], threading.Lock()
        seeds = list(range(1, n + 1))

        def worker():
            while True:
                with lock:
                    if not seeds:
                        return
                    s = seeds.pop()
                try:
                    txt = one(port, prompt, s)
                except Exception as e:  # noqa: BLE001
                    txt = f"ERR {e}"
                with lock:
                    res.append(txt)

        ts = [threading.Thread(target=worker) for _ in range(16)]
        [t.start() for t in ts]; [t.join() for t in ts]
        out[pname] = [t.split() for t in res if not t.startswith("ERR")]
        print(f"{name} {pname}: {len(out[pname])} samples")
    (D / f"dist-{name}.json").write_text(json.dumps(out))


def pooled(samples, positions):
    c = Counter()
    for s in samples:
        for i in positions:
            if i < len(s):
                c[(i, s[i])] += 1
    return c


def tv(a, b):
    na, nb = sum(a.values()), sum(b.values())
    return 0.5 * sum(abs(a[k] / na - b[k] / nb) for k in set(a) | set(b))


def compare(ref, names):
    R = json.loads((D / f"dist-{ref}.json").read_text())
    rnd = random.Random(7)
    for pname in PROMPTS:
        pos = range(1, LEN[pname])  # position 0 comes from the target's own sampling, 1.. through the drafter
        r = R[pname]
        half = len(r) // 2
        floor = tv(pooled(r[:half], pos), pooled(r[half:], pos))
        for n in names:
            X = json.loads((D / f"dist-{n}.json").read_text())[pname]
            obs = tv(pooled(r, pos), pooled(X, pos))
            both = r + X; perm = []
            for _ in range(200):
                rnd.shuffle(both)
                perm.append(tv(pooled(both[:len(r)], pos), pooled(both[len(r):], pos)))
            p = sum(v >= obs for v in perm) / len(perm)
            lens = Counter(len(s) for s in X).most_common(2)
            print(f"{pname:8} {n:16} TV vs {ref}: {obs:.4f}  (noise floor {floor:.4f}, permutation p={p:.3f}; "
                  f"n={len(X)}, lengths {lens})  -> {'BIASED?' if p < 0.01 else 'ok'}")


if __name__ == "__main__":
    if sys.argv[1] == "collect":
        collect(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 1500)
    else:
        compare(sys.argv[2], sys.argv[3:])
