"""Synthetic long-context code retrieval/reasoning (RULER-style, code haystack), 32K/64K/110K.

Each haystack (Python stdlib sources + synthetic needles) carries 10 questions:
4 x multi-key retrieval (near-duplicate CFG_* keys), 2 x multi-value (all ports of a route),
2 x variable tracking (TOKEN_* alias chains, 3 hops, with a distractor chain),
2 x function + far-away constant (compute quota_for_x(c)).
All questions on a haystack share the same prompt prefix, so after the first question the
server's prefix cache makes the rest cheap; the runner warms each group with one request.
"""
import os
import re

from .common import DATA, read_jsonl

_HAY = {}


def _hay(gid):
    if gid not in _HAY:
        with open(os.path.join(DATA, "longctx", "haystacks", gid + ".txt")) as f:
            _HAY[gid] = f.read()
    return _HAY[gid]


def load(bins=None, kinds=None, per_bin=None):
    items = []
    for it in read_jsonl(os.path.join(DATA, "longctx", "items.jsonl")):
        if bins and it["bin"] not in bins:
            continue
        if kinds and it["kind"] not in kinds:
            continue
        if per_bin is not None and int(it["group"].rsplit("_h", 1)[1]) >= per_bin:
            continue
        it = dict(it)
        it["subset"] = f"longctx_{it['bin']}"
        items.append(it)
    return items


def _answer_line(text):
    m = re.findall(r"ANSWER:\s*(.+)", text or "")
    if m:
        return m[-1].strip().strip("`*")
    lines = [l for l in (text or "").strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def score(item, text):
    a = _answer_line(text)
    k, exp = item["kind"], item["answer"]
    if k in ("kv", "func"):
        nums = re.findall(r"-?\d[\d,_]*", a)
        if not nums:
            return False, a
        try:
            v = int(nums[0].replace(",", "").replace("_", ""))
        except ValueError:
            return False, a
        return v == exp, a
    if k == "multivalue":
        nums = {int(x) for x in re.findall(r"\d{4,6}", a)}
        return nums == set(exp), a
    if k == "vartrack":
        names = set(re.findall(r"TOKEN_[A-Z0-9_]+", a))
        return names == set(exp), a
    return False, a


def make_fn(cfg):
    def fn(item, ctx):
        content = _hay(item["group"]) + item["question"]
        r = ctx.chat([{"role": "user", "content": content}], max_tokens=cfg.max_tokens)
        base = {"bin": item["bin"], "kind": item["kind"], "depth": item["depth"],
                "prompt_tokens_est": item["prompt_tokens_est"]}
        if r.get("error"):
            return dict(base, correct=False, status="api_error")
        ok, got = score(item, r["content"])
        return dict(base, correct=ok, status="pass" if ok else "wrong", got=got[:300], expected=item["answer"])
    return fn
