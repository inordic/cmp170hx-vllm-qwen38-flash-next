#!/usr/bin/env python3
"""Patch K1 server check: INT8 KV + --prefix-match-unit + MTP follow-ups that hit the prefix cache mid-block.

Every follow-up turn resends the whole chat. With MTP and fine-grained hashing the hit ends inside a block
(the eagle drop alone moves it 32 tokens before a block edge), so the scheduler copies the shared tail block
(copy-on-write). Before K1 the INT8 scales of that copy were stale: empty/wrong answers, thinking to max_tokens.

Modes (all print one summary line "K1 <MODE> ALL PASS" or "K1 <MODE> SOME FAILED ..."):
  multiturn PORT  conversations over a long document; turn 1 prompt lengths are steered (via /tokenize) so the
                  hits land at chosen offsets inside a block, a fact sits in the copied tail, --cold also asks
                  every follow-up once more with a cache-busting nonce (no hit) as a same-server reference
  fanout PORT     one document cached, then --users concurrent follow-ups (many CoW copies of one source block)
  firstbatch PORT run right after a restart: --users concurrent first turns, then their follow-ups concurrently
  fill PORT       last-block fill-level sweep (#50749-style): unique prompts whose length leaves the last block
                  filled to 1, 31, 32, 33, ..., B-1 tokens, then each prompt repeated (full hit + CoW)
  pressure PORT   conversations keep asking follow-ups while a flood of unique long prompts recycles the pool
                  (#50702-style eviction/reuse); samples KV usage and preemptions
  compare A.json B.json   compare two --out files (e.g. INT8+K1 vs BF16)
Common: --block B (3200 INT8, 1600 BF16), --log /opt/vllm-qfn/logs/vllm.log (count K1 log lines written during
the run), --out FILE (JSON records), --extra-body JSON (merged into every request).
API key: env API_KEY, else the file in env KEY_FILE, else /root/llm-api-key. Standard library only.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

A = None  # parsed args
BASE = HDR = MODEL = None
TEXT = ""
LOCK = threading.Lock()
RECORDS: list[dict] = []


# --------------------------------------------------------------------------------------------------- HTTP
def http(path, body=None, timeout=1800):
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def metric(name):
    """Sum a counter/gauge from /metrics (counters are exported as <name>_total)."""
    try:
        text = urllib.request.urlopen(BASE + "/metrics", timeout=10).read().decode()
    except Exception:  # noqa: BLE001
        return float("nan")
    total, found = 0.0, False
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        head = line.split("{", 1)[0].split(" ", 1)[0]
        if head in (name, name + "_total"):
            total += float(line.rsplit(" ", 1)[-1])
            found = True
    return total if found else float("nan")


def chat(messages, max_tokens=None, seed=None, timeout=1800):
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens or A.max_tokens,
            "temperature": A.temperature, "top_p": 0.95, "stream": True, "stream_options": {"include_usage": True}}
    if seed is not None:
        body["seed"] = seed
    body.update(A.extra)
    req = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(body).encode(), headers=HDR)
    t0, first, content, reasoning, usage, finish = time.time(), None, [], [], {}, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode(errors="ignore").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                ch = (ev.get("choices") or [{}])[0]
                d = ch.get("delta") or {}
                piece_r = d.get("reasoning") or d.get("reasoning_content")
                if (d.get("content") or piece_r) and first is None:
                    first = time.time()
                if d.get("content"):
                    content.append(d["content"])
                if piece_r:
                    reasoning.append(piece_r)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                if ev.get("usage"):
                    usage = ev["usage"]
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"answer": "", "error": str(e)[:200], "ttft": None, "total": time.time() - t0, "finish": "error",
                "prompt_tokens": 0, "completion_tokens": 0, "reasoning_chars": 0}
    return {"answer": "".join(content).strip(), "ttft": round((first or time.time()) - t0, 3),
            "total": round(time.time() - t0, 2), "finish": finish, "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0), "reasoning_chars": len("".join(reasoning))}


def count_tokens(messages):
    try:
        body = {"model": MODEL, "messages": messages, "add_generation_prompt": True}
        if "chat_template_kwargs" in A.extra:
            body["chat_template_kwargs"] = A.extra["chat_template_kwargs"]
        return int(http("/tokenize", body, timeout=60)["count"])
    except Exception:  # noqa: BLE001
        return None


# --------------------------------------------------------------------------------------------- content
def load_text():
    parts = []
    for pat in A.corpus.split(","):
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                parts.append(Path(p).read_text(errors="ignore"))
            except OSError:
                pass
    text = "".join(parts)
    if len(text) < 2_000_000:  # synthetic, deterministic fallback
        rnd = random.Random(1234)
        words = ("valve pump sensor line relay north south east west batch filter level pressure crew shift "
                 "report inspected replaced calibrated logged checked moved stored tagged sealed opened").split()
        text += "".join(f"Record {i}: the {rnd.choice(words)} on unit {rnd.randint(1, 99)} was "
                        f"{rnd.choice(words)} by team {rnd.choice('ABCDEFGHJK')}{rnd.randint(1, 9)} at "
                        f"{rnd.randint(0, 23):02d}:{rnd.randint(0, 59):02d}; reading {rnd.randint(100, 999)}.\n"
                        for i in range(60_000))
    return text


def slice_text(rnd, tokens):
    n = int(tokens * A.chars_per_token)
    off = rnd.randrange(0, max(1, len(TEXT) - n))
    return TEXT[off:off + n]


FACT_NAMES = ("ALPHA", "BRAVO", "CHARLIE", "DELTA", "ECHO", "FOXTROT")


def build_doc(rnd, tokens):
    """Document with 5 facts spread over it and one (ECHO) in the last ~400 tokens: the tail ends up in the
    block that a follow-up's hit copies."""
    doc = slice_text(rnd, tokens)
    facts = {n: str(rnd.randint(100000, 999999)) for n in FACT_NAMES[:5]}
    tail = int(400 * A.chars_per_token)
    body, end = doc[:-tail], doc[-tail:]
    step = len(body) // 5
    pieces = [body[i * step:(i + 1) * step] + f"\n\nNote: the code {n} is {facts[n]}.\n\n"
              for i, n in enumerate(FACT_NAMES[:4])]
    pieces.append(body[4 * step:])
    pieces.append(end[: tail // 2] + f"\n\nNote: the code ECHO is {facts['ECHO']}.\n\n" + end[tail // 2:])
    return "".join(pieces), facts


def first_messages(conv, doc, nonce="", filler=""):
    sysmsg = f"{nonce}You are a careful assistant. Answer briefly."
    return [{"role": "system", "content": sysmsg},
            {"role": "user", "content": f"Conversation {conv}. Read this document:\n\n{doc}{filler}"
             "\n\nWhat is the code ALPHA? Answer with just the number."}]


def steer(conv, doc, target, nonce="", tol=12):
    """Pad the first user turn so that (prompt tokens % block) lands on `target` (within `tol` tokens short):
    corpus text for big gaps, ' ok' (one token each in Qwen's tokenizer) for the last stretch."""
    filler = ""
    msgs = first_messages(conv, doc, nonce)
    for _ in range(14):
        n = count_tokens(msgs)
        if n is None:
            return msgs, None
        delta = (target - n) % A.block
        if delta <= tol:
            return msgs, n
        if delta < 200:
            filler += " ok" * delta
        else:
            start = (conv * 7919 + len(filler)) % (len(TEXT) - 50_000)
            filler += "\n" + TEXT[start:start + int(delta * A.chars_per_token * 0.9)]
        msgs = first_messages(conv, doc, nonce, filler)
    return msgs, count_tokens(msgs)


def degenerate(s):
    words = s.split()
    if len(words) >= 40 and max(words.count(w) for w in set(words)) / len(words) > 0.3:
        return True
    return bool(re.search(r"(.{6,40})\1{5,}", s))


def judge(rec, expect):
    rec["expect"] = expect
    rec["pass"] = bool(expect in rec["answer"] and rec["finish"] == "stop" and not degenerate(rec["answer"]))
    return rec


def record(rec):
    with LOCK:
        RECORDS.append(rec)
    hit = "" if rec.get("hit_tokens") is None else (
        f" hit {rec['hit_tokens']:,} (offset {rec['hit_offset']}{' MID-BLOCK' if rec['hit_offset'] else ''})")
    print(f"[{rec['mode']}] conv {rec.get('conv')} turn {rec.get('turn')} {rec.get('kind', '')}: prompt "
          f"{rec['prompt_tokens']:,} TTFT {rec['ttft']}s{hit} think {rec['reasoning_chars']} ch, finish "
          f"{rec['finish']}, answer {rec['answer'][:24]!r} -> {'PASS' if rec['pass'] else 'FAIL (want ' + rec['expect'] + ')'}",
          flush=True)


def ask_seq(mode, conv, turn, kind, messages, expect, seed):
    """Sequential request with the prefix-cache hit measured from the metrics delta."""
    h0 = metric("vllm:prefix_cache_hits")
    rec = chat(messages, seed=seed)
    time.sleep(0.3)
    hit = metric("vllm:prefix_cache_hits") - h0
    rec.update(mode=mode, conv=conv, turn=turn, kind=kind)
    if hit == hit:  # not NaN
        rec["hit_tokens"], rec["hit_offset"] = int(hit), int(hit) % A.block
    record(judge(rec, expect))
    return rec


# ------------------------------------------------------------------------------------------------ modes
def conversation(mode, conv, target, turns, cold, rnd):
    doc, facts = build_doc(rnd, A.doc_tokens)
    msgs, n = steer(conv, doc, target) if target is not None else (first_messages(conv, doc), None)
    if n is not None:
        print(f"[{mode}] conv {conv}: turn-1 prompt {n:,} tokens, {n % A.block} into a {A.block}-token block", flush=True)
    order = ["ALPHA", "ECHO", "BRAVO", "DELTA", "CHARLIE"][:turns]
    for turn, name in enumerate(order):
        if turn:
            msgs.append({"role": "user", "content": f"And what is the code {name}? Answer with just the number."})
        rec = ask_seq(mode, conv, turn + 1, "first" if turn == 0 else "hit", msgs, facts[name], seed=1000 * conv + turn)
        if turn and cold:
            busted = [{"role": "system", "content": f"[nonce {rnd.random():.12f}] " + msgs[0]["content"]}] + msgs[1:]
            ask_seq(mode, conv, turn + 1, "cold", busted, facts[name], seed=1000 * conv + turn)
        msgs.append({"role": "assistant", "content": rec["answer"]})
    return msgs, facts


def mode_multiturn():
    targets = [None] * A.convs
    if A.offsets != "none":
        offs = ([130, A.block // 2 + 50, A.block - 30, A.block - 5] if A.offsets == "auto"
                else [int(x) for x in A.offsets.split(",")])
        targets = [offs[i % len(offs)] for i in range(A.convs)]
    for c in range(A.convs):
        conversation("multiturn", c, targets[c], A.turns, A.cold, random.Random(f"k1-{c}-{A.seed}"))


def mode_fanout():
    rnd = random.Random(f"fan-{A.seed}")
    doc, facts = build_doc(rnd, A.doc_tokens)
    msgs = first_messages(0, doc)
    first = ask_seq("fanout", 0, 1, "first", msgs, facts["ALPHA"], seed=1)
    msgs.append({"role": "assistant", "content": first["answer"]})
    names = [FACT_NAMES[i % 5] for i in range(A.users)]

    def one(i):
        m = msgs + [{"role": "user", "content": f"(user {i}) And what is the code {names[i]}? Answer with just the number."}]
        rec = chat(m, seed=100 + i)
        rec.update(mode="fanout", conv=0, turn=2, kind=f"user{i}")
        record(judge(rec, facts[names[i]]))

    with cf.ThreadPoolExecutor(A.users) as ex:
        list(ex.map(one, range(A.users)))


def mode_firstbatch():
    for _ in range(120):
        try:
            urllib.request.urlopen(BASE + "/health", timeout=5)
            break
        except Exception:  # noqa: BLE001
            time.sleep(5)
    rnds = [random.Random(f"fb-{i}-{A.seed}") for i in range(A.users)]
    docs = [build_doc(r, A.doc_tokens) for r in rnds]
    convs = [first_messages(i, d) for i, (d, _) in enumerate(docs)]

    def turn(i, t, name):
        if t > 1:
            convs[i].append({"role": "user", "content": f"And what is the code {name}? Answer with just the number."})
        rec = chat(convs[i], seed=10 * i + t)
        rec.update(mode="firstbatch", conv=i, turn=t, kind="first" if t == 1 else "hit")
        record(judge(rec, docs[i][1][name]))
        convs[i].append({"role": "assistant", "content": rec["answer"]})

    for t, name in enumerate(["ALPHA", "ECHO", "BRAVO"], 1):
        with cf.ThreadPoolExecutor(A.users) as ex:
            list(ex.map(lambda i: turn(i, t, name), range(A.users)))


def mode_fill():
    b = A.block
    fills = sorted({1, 16, 31, 32, 33, 63, 64, 65, b // 2, b - 33, b - 32, b - 31, b - 1, b})
    for k, fill in enumerate(fills):
        rnd = random.Random(f"fill-{k}-{A.seed}")
        doc, facts = build_doc(rnd, A.doc_tokens)
        busted, n = steer(k, doc, fill % b, nonce=f"[nonce {rnd.random():.12f}] ", tol=0)
        if n is not None:
            print(f"[fill] case {k}: prompt {n:,} tokens -> last block holds {n % b or b} of {b}", flush=True)
        ask_seq("fill", k, 1, f"unique-fill{fill}", busted, facts["ALPHA"], seed=k)
        ask_seq("fill", k, 1, f"repeat-fill{fill}", busted, facts["ALPHA"], seed=k)  # same prompt: full hit + CoW


def mode_pressure():
    pool = A.pool_tokens or parse_pool()
    if not pool:
        sys.exit("pressure: give --pool-tokens (or --log with the startup lines)")
    total = int(pool * A.fill)
    n_flood = max(1, total // A.flood_len)
    print(f"[pressure] pool {pool:,} tokens -> {n_flood} unique prompts x {A.flood_len:,} tokens "
          f"({n_flood * A.flood_len / pool:.2f}x pool), {A.flood_conc} at a time", flush=True)
    rnd = random.Random(f"pr-{A.seed}")
    convs = []
    for c in range(A.convs):
        doc, facts = build_doc(rnd, A.doc_tokens)
        msgs = first_messages(c, doc)
        rec = chat(msgs, seed=c)
        rec.update(mode="pressure", conv=c, turn=1, kind="first")
        record(judge(rec, facts["ALPHA"]))
        msgs.append({"role": "assistant", "content": rec["answer"]})
        convs.append([msgs, facts, 1])
    stop = threading.Event()
    peak = {"usage": 0.0}
    pre0 = metric("vllm:num_preemptions")

    def sampler():
        while not stop.is_set():
            u = metric("vllm:kv_cache_usage_perc")
            if u == u:
                peak["usage"] = max(peak["usage"], u)
            time.sleep(2)

    def flood(i):
        r = random.Random(f"flood-{i}-{A.seed}")
        m = [{"role": "user", "content": f"[{r.random():.12f}] Summarise in one word:\n\n" + slice_text(r, A.flood_len)}]
        chat(m, max_tokens=4, seed=i, timeout=3600)

    def followups(tag):
        for c, conv in enumerate(convs):
            msgs, facts, t = conv
            name = FACT_NAMES[t % 5]
            msgs.append({"role": "user", "content": f"And what is the code {name}? Answer with just the number."})
            rec = chat(msgs, seed=100 * c + t)
            rec.update(mode="pressure", conv=c, turn=t + 1, kind=tag)
            record(judge(rec, facts[name]))
            msgs.append({"role": "assistant", "content": rec["answer"]})
            conv[2] = t + 1

    threading.Thread(target=sampler, daemon=True).start()
    with cf.ThreadPoolExecutor(A.flood_conc) as ex:
        futs = [ex.submit(flood, i) for i in range(n_flood)]
        while not all(f.done() for f in futs):
            followups("during-flood")
    followups("after-flood")
    stop.set()
    print(f"[pressure] peak KV usage {peak['usage']:.1%}, preemptions +{metric('vllm:num_preemptions') - pre0:.0f}",
          flush=True)


def parse_pool():
    if not A.log or not os.path.exists(A.log):
        return None
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", Path(A.log).read_text(errors="ignore"))
    return int(m[-1].replace(",", "")) if m else None


def mode_compare():
    a, b = (json.load(open(f)) for f in A.files)

    def stats(recs):
        out = {}
        for r in recs:
            k = (r["mode"], r.get("kind", "").split("-")[0] if r["mode"] == "fill" else r.get("kind", ""))
            s = out.setdefault(k, [0, 0, 0])
            s[0] += 1
            s[1] += r["pass"]
            s[2] += r.get("reasoning_chars", 0)
        return out

    sa, sb = stats(a), stats(b)
    print(f"{'mode/kind':28} {'A pass':>10} {'B pass':>10} {'A think/req':>12} {'B think/req':>12}")
    for k in sorted(set(sa) | set(sb)):
        x, y = sa.get(k, [0, 0, 0]), sb.get(k, [0, 0, 0])
        print(f"{'/'.join(k):28} {x[1]:>4}/{x[0]:<5} {y[1]:>4}/{y[0]:<5} {x[2] // max(1, x[0]):>12} {y[2] // max(1, y[0]):>12}")
    def key(r):
        return (r["mode"], r.get("conv"), r.get("turn"), r.get("kind"))

    ka, kb = {key(r): r["pass"] for r in a}, {key(r): r["pass"] for r in b}
    diff = [k for k in ka if k in kb and ka[k] != kb[k]]
    print(f"records with different verdicts: {len(diff)} {diff[:10]}")
    ok = all(r["pass"] for r in a) and all(r["pass"] for r in b)
    print("K1 COMPARE ALL PASS" if ok else "K1 COMPARE SOME FAILED")
    return ok


# ------------------------------------------------------------------------------------------------- main
def log_summary(start):
    if not A.log or not os.path.exists(A.log):
        return
    with open(A.log, "rb") as f:
        f.seek(start)
        text = f.read().decode(errors="ignore")
    steps = re.findall(r"Asgard patch K1: CoW step (\d+): (\d+) block pair", text)
    stale = [int(x) for x in re.findall(r"stale dst rows before copy (\d+)/", text)]
    print(f"server log during the run: K1 registrations {text.count('Asgard patch K1: per-block KV side tensors')}, "
          f"K1 CoW step lines {len(steps)} (pairs {sum(int(p) for _, p in steps)}), stale rows before copy "
          f"{sum(stale) if stale else 'n/a (VLLM_K1_DEBUG off)'}, KV partial-copy warnings "
          f"{text.count('differ between dst and src')}, 'NOT copied' {text.count('NOT copied (VLLM_K1_DISABLE')}, "
          f"errors {len(re.findall(r'(?i)traceback|out of memory|CUDA error', text))}")


def main():
    global A, BASE, HDR, MODEL, TEXT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["multiturn", "fanout", "firstbatch", "fill", "pressure", "compare"])
    ap.add_argument("args", nargs="+", help="PORT, or two JSON files for compare")
    ap.add_argument("--block", type=int, default=3200)
    ap.add_argument("--convs", type=int, default=4)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--doc-tokens", type=int, default=20000)
    ap.add_argument("--offsets", default="auto", help="auto | none | comma list of in-block offsets for turn 1")
    ap.add_argument("--cold", action="store_true", help="also ask each follow-up with a cache-busting nonce")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--extra-body", default="{}")
    ap.add_argument("--pool-tokens", type=int, default=0)
    ap.add_argument("--fill", type=float, default=1.2, help="pressure: flood size as a multiple of the pool")
    ap.add_argument("--flood-len", type=int, default=32000)
    ap.add_argument("--flood-conc", type=int, default=8)
    ap.add_argument("--corpus", default="/root/tune/corpus/Strata/src/**/*.cpp,/root/tune/corpus/Strata/docs/*.md")
    ap.add_argument("--chars-per-token", type=float, default=3.27)
    ap.add_argument("--seed", default="0")
    ap.add_argument("--log", default="")
    ap.add_argument("--out", default="")
    A = ap.parse_args()
    A.extra = json.loads(A.extra_body)
    if A.mode == "compare":
        A.files = A.args[:2]
        sys.exit(0 if mode_compare() else 1)
    BASE = f"http://127.0.0.1:{int(A.args[0])}"
    key = os.environ.get("API_KEY") or Path(os.environ.get("KEY_FILE", "/root/llm-api-key")).read_text().strip()
    HDR = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if A.mode != "firstbatch":
        MODEL = http("/v1/models", timeout=30)["data"][0]["id"]
    TEXT = load_text()
    log_start = os.path.getsize(A.log) if A.log and os.path.exists(A.log) else 0
    q0, h0 = metric("vllm:prefix_cache_queries"), metric("vllm:prefix_cache_hits")
    if A.mode == "firstbatch":
        for _ in range(120):
            try:
                MODEL = http("/v1/models", timeout=10)["data"][0]["id"]
                break
            except Exception:  # noqa: BLE001
                time.sleep(5)
    {"multiturn": mode_multiturn, "fanout": mode_fanout, "firstbatch": mode_firstbatch, "fill": mode_fill,
     "pressure": mode_pressure}[A.mode]()
    q1, h1 = metric("vllm:prefix_cache_queries"), metric("vllm:prefix_cache_hits")
    if A.out:
        Path(A.out).write_text(json.dumps(RECORDS, indent=1))
    hits = [r for r in RECORDS if r.get("hit_tokens") is not None
            and (r.get("kind") == "hit" or str(r.get("kind", "")).startswith("repeat"))]
    cold = [r for r in RECORDS if r.get("kind") == "cold" and r.get("hit_tokens") is not None]
    if cold:
        print(f"cold controls: {sum(1 for r in cold if r['hit_tokens'] == 0)} of {len(cold)} had no prefix hit")
    if hits:
        mid = sum(1 for r in hits if r["hit_offset"])
        ttft = sorted(r["ttft"] for r in hits if r["ttft"] is not None)
        print(f"follow-up hits: {mid} of {len(hits)} mid-block; offsets {sorted(r['hit_offset'] for r in hits)}; "
              f"median follow-up TTFT {ttft[len(ttft) // 2] if ttft else 'n/a'} s")
    print(f"prefix cache: {h1 - h0:,.0f} of {q1 - q0:,.0f} queried tokens were hits")
    log_summary(log_start)
    bad = [r for r in RECORDS if not r["pass"]]
    by_kind = {}
    for r in bad:
        by_kind[r.get("kind", "")] = by_kind.get(r.get("kind", ""), 0) + 1
    print(f"K1 {A.mode.upper()} " + ("ALL PASS" if not bad and RECORDS else
                                     f"SOME FAILED ({len(bad)} of {len(RECORDS)}; by kind {by_kind})"))
    sys.exit(0 if not bad and RECORDS else 1)


if __name__ == "__main__":
    main()
