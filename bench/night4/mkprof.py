#!/usr/bin/env python3
"""Derive a night-4 test profile from the production profile.
usage: mkprof.py BASE OUT [opt=value ...]   (options may repeat where noted)
  venv=n4               run on /opt/vllm-qfn-n4 (tree with the night-4 patches, inactive unless their env is set)
  port=8008             PORT
  model=PATH            MODEL
  head=PATH             export VLLM_MTP_DRAFT_HEAD_PATH (patch D1, int4 draft lm_head)
  mtp=N                 num_speculative_tokens in the speculative config
  spec=KEY:VALUE        set a key in the speculative config JSON (repeatable; VALUE parsed as JSON, else string)
  nospec=1              drop --speculative-config (no drafting: reference for the output-distribution test)
  pmu=U                 --prefix-match-unit U
  chunk=C               export VLLM_MIXED_PREFILL_CHUNK (patch D2)
  batch=N               MAX_BATCH_TOKENS (max-num-batched-tokens)
  split=A,B             VLLM_PP_LAYER_PARTITION
  kv=DTYPE              KV_DTYPE (e.g. int8_per_token_head)
  gpumem=F              GPU_MEMORY
  maxlen=N              MAX_MODEL_LEN
  addflag=TEXT          append TEXT to EXTRA_FLAGS (repeatable)
  env=NAME=VALUE        export NAME=VALUE (repeatable)
  label=TEXT            name in the header comment"""
import json, re, sys

base, out, *kv = sys.argv[1:]
opts = []
for a in kv:
    k, _, v = a.partition("=")
    opts.append((k, v))
s = open(base).read()


def sub(pat, rep, count=1):
    global s
    s2, n = re.subn(pat, lambda m: rep, s, count=count, flags=re.M)
    assert n == count, f"pattern {pat!r} matched {n} times"
    s = s2


def get(k):
    return [v for kk, v in opts if kk == k]


def one(k):
    v = get(k)
    return v[-1] if v else None


exports = []
if one("venv") == "n4":
    sub(r"^VLLM_VENV=.*$", "VLLM_VENV=/opt/vllm-qfn-n4/.venv")
    sub(r"^VLLM_BIN=.*$", "VLLM_BIN=/opt/vllm-qfn-n4/.venv/bin/vllm")
if one("port"):
    sub(r"^PORT=\d+.*$", f"PORT={int(one('port'))}")
if one("model"):
    sub(r"^MODEL=.*$", f"MODEL={one('model')}")
if one("batch"):
    sub(r"^MAX_BATCH_TOKENS=\d+.*$", f"MAX_BATCH_TOKENS={int(one('batch'))}")
if one("split"):
    a, b = (int(x) for x in one("split").split(","))
    sub(r"^VLLM_PP_LAYER_PARTITION=.*$", f"VLLM_PP_LAYER_PARTITION={a},{b}")
if one("gpumem"):
    sub(r"^GPU_MEMORY=.*$", f"GPU_MEMORY={float(one('gpumem'))}")
if one("maxlen"):
    sub(r"^MAX_MODEL_LEN=\d+.*$", f"MAX_MODEL_LEN={int(one('maxlen'))}")
if one("kv"):
    s += f"\nKV_DTYPE={one('kv')}\n"
if one("pmu"):
    sub(r"--prefix-match-unit \d+", f"--prefix-match-unit {int(one('pmu'))}")

# speculative config inside EXTRA_FLAGS (JSON with escaped quotes)
m = re.search(r'--speculative-config (\{(?:[^ ]*)\})', s)
assert m, "no --speculative-config in EXTRA_FLAGS"
spec = json.loads(m.group(1).replace('\\"', '"'))
if one("nospec"):
    s = s[:m.start()].rstrip(" ") + s[m.end():]
    m = None
if m and one("mtp"):
    spec["num_speculative_tokens"] = int(one("mtp"))
for kvp in get("spec") if m else []:
    k, _, v = kvp.partition(":")
    try:
        spec[k] = json.loads(v)
    except json.JSONDecodeError:
        spec[k] = v
if m:
    s = s[:m.start(1)] + json.dumps(spec, separators=(",", ":")).replace('"', '\\"') + s[m.end(1):]

for f in get("addflag"):
    fm = re.search(r'^EXTRA_FLAGS="(.*)"\s*$', s, flags=re.M)
    assert fm, "no EXTRA_FLAGS line"
    s = s[:fm.end(1)] + " " + f.replace('"', '\\"') + s[fm.end(1):]

if one("head"):
    exports.append(f"export VLLM_MTP_DRAFT_HEAD_PATH={one('head')}")
if one("chunk"):
    exports.append(f"export VLLM_MIXED_PREFILL_CHUNK={int(one('chunk'))}")
for e in get("env"):
    n, _, v = e.partition("=")
    assert re.fullmatch(r"[A-Z0-9_]+", n), f"bad env name {n!r}"
    exports.append(f"export {n}={v}")

args = " ".join(f"{k}={v}" for k, v in opts)
s = f"# NIGHT 4 test profile {one('label') or out} (derived from {base} by mkprof.py {args})\n" + s
if exports:
    s += "\n# --- night 4 switches (exported: serve.sh only forwards an allowlist, a sourced export reaches the engine)\n" \
         + "\n".join(exports) + "\n"
open(out, "w").write(s)
print(out)
