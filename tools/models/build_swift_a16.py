#!/usr/bin/env python3
"""Build Swift-A16: Swift 1.5 W4A16 AWQ with BF16 attention projections.

Takes the served AWQ checkpoint (compressed-tensors, pack-quantized INT4) and the
BF16 attention file written by fetch_bf16_attention.py and creates a new model
directory:

  * every safetensors file without INT4 attention tensors is HARDLINKED
    (PLE-only shards, model-00002, model-00019, model_mtp, ...) - byte-identical,
    so the PLE-SSD loader's data_offsets stay valid and no disk space is used;
  * every shard that holds INT4 attention (weight_packed / weight_scale /
    weight_zero_point / weight_shape of q/k/v/o_proj or in_proj_qkv/in_proj_z/
    out_proj) is rewritten: the other tensors are copied byte for byte, the INT4
    attention parts are dropped and the BF16 `.weight` of the same modules is
    added (--placement inplace, default), or all BF16 tensors go into one extra
    shard (--placement extra);
  * model.safetensors.index.json is regenerated from the written headers;
  * config.json: the attention modules are appended to
    quantization_config.ignore (exact HF module names, the same form as the 728
    entries already there), so vLLM's compressed-tensors config returns no
    scheme for them -> UnquantizedLinearMethod.  All shards of each fused vLLM
    module (qkv_proj = q/k/v, in_proj_qkvz = in_proj_qkv/in_proj_z) are switched
    together;
  * every other file (tokenizer, chat template, generation config, README,
    recipe, ...) is copied.

Standard library only.  It refuses to touch an existing output directory and
builds in DST.building first, renaming to DST only after the verification.

    python3 build_swift_a16.py --src /opt/vllm-qfn/models/swift15-awq \
        --attn /opt/vllm-qfn/work/swift-a16/swift15-bf16-attention.safetensors \
        --dst /opt/vllm-qfn/models/swift15-a16 [--dry-run]
    python3 build_swift_a16.py --verify-only --src ... --attn ... --dst ...
    python3 build_swift_a16.py --src-dump swift15-awq-dump.json --attn-plan plan.json --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import sys
import time

ATTN_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$"
)
QSA_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
GDN_PROJ = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")
INT4_PARTS = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")
REQUIRED_PARTS = ("weight_packed", "weight_scale", "weight_shape")
# vLLM fused (packed) modules that are built with a quant_config in the
# qwen4_exp model; all shards of one fused module must share one scheme
# (vllm should_ignore_layer raises otherwise).
FUSED_GROUPS = {
    "self_attn.qkv_proj": ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"),
    "linear_attn.in_proj_qkvz": ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z"),
    "linear_attn.in_proj_ba": ("linear_attn.in_proj_b", "linear_attn.in_proj_a"),
    "shared_expert.gate_up_proj": ("shared_expert.gate_proj", "shared_expert.up_proj"),
    "ple.kv_proj": ("ple.key_proj", "ple.value_proj"),
}
PLE_MARK = ".ngram_embedding.shard_"
DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
MAX_HEADER = 100_000_000
CHUNK = 16 * 1024 * 1024
FLUSH_EVERY = 512 * 1024 * 1024
INDEX = "model.safetensors.index.json"
EXTRA_SHARD = "model-a16-attention.safetensors"
BUILD_MANIFEST = "SWIFT_A16_BUILD.json"
GiB = 2**30


def log(msg=""):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def die(msg):
    log("ERROR: " + msg)
    raise SystemExit(2)


def nbytes(dtype, shape):
    n = DTYPE_SIZE[dtype]
    for s in shape:
        n *= int(s)
    return n


def module_of(name):
    return name.rsplit(".", 1)[0]


def read_header(path):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            die("%s: shorter than 8 bytes (zero-byte file from a full disk?)" % path)
        (hlen,) = struct.unpack("<Q", head)
        if not 0 < hlen <= MAX_HEADER:
            die("%s: implausible safetensors header length %d" % (path, hlen))
        raw = f.read(hlen)
        if len(raw) != hlen:
            die("%s: truncated header" % path)
    hdr = json.loads(raw)
    meta = hdr.pop("__metadata__", None)
    return hdr, 8 + hlen, meta


def check_tensor_table(label, tensors, data_start, file_size):
    """safetensors invariants: sizes match shapes, data contiguous, no holes,
    file exactly as long as the header says."""
    spans = []
    for name, info in tensors.items():
        if info["dtype"] not in DTYPE_SIZE:
            die("%s: %s has unknown dtype %s" % (label, name, info["dtype"]))
        b, e = info["data_offsets"]
        if e - b != nbytes(info["dtype"], info["shape"]):
            die("%s: %s data_offsets %s do not match %s %s" % (label, name, info["data_offsets"],
                                                               info["dtype"], info["shape"]))
        spans.append((b, e, name))
    spans.sort()
    pos = 0
    for b, e, name in spans:
        if b != pos:
            die("%s: hole or overlap before %s (offset %d, expected %d)" % (label, name, b, pos))
        pos = e
    if file_size is not None and file_size != data_start + pos:
        die("%s: file is %d bytes, header implies %d (truncated or padded file)" % (
            label, file_size, data_start + pos))


# --------------------------------------------------------------------------
# checkpoint access (real directory or the header dump for offline dry runs)
# --------------------------------------------------------------------------
class DirCheckpoint:
    virtual = False

    def __init__(self, path):
        self.path = os.path.abspath(path)
        with open(os.path.join(self.path, INDEX)) as f:
            idx = json.load(f)
        self.index_meta = idx.get("metadata", {})
        self.weight_map = idx["weight_map"]
        with open(os.path.join(self.path, "config.json")) as f:
            self.config = json.load(f)
        self.files = sorted(set(self.weight_map.values()))
        self.headers = {}
        for fn in self.files:
            p = os.path.join(self.path, fn)
            if not os.path.isfile(p):
                die("%s: listed in the index but missing" % p)
            hdr, ds, meta = read_header(p)
            check_tensor_table(fn, hdr, ds, os.path.getsize(p))
            self.headers[fn] = (hdr, ds, meta, os.path.getsize(p))
        self.stray = sorted(f for f in os.listdir(self.path)
                            if f.endswith(".safetensors") and f not in self.headers)

    def file_path(self, fn):
        return os.path.join(self.path, fn)

    def pread(self, fn, offset, length):
        with open(self.file_path(fn), "rb") as f:
            f.seek(offset)
            data = f.read(length)
        if len(data) != length:
            die("short read %s @%d" % (fn, offset))
        return data


class DumpCheckpoint:
    """Header dump made on the server (files{name:{size,header_len,tensors}},
    config, ...).  The dump's index text is truncated, so the weight map is
    rebuilt from the headers.  No tensor data -> no byte-level checks."""
    virtual = True

    def __init__(self, dump_path):
        with open(dump_path) as f:
            d = json.load(f)
        self.path = "<dump:%s>" % os.path.basename(dump_path)
        self.config = d["config"]
        self.index_meta = {}
        self.headers = {}
        self.weight_map = {}
        for fn, v in d["files"].items():
            t = dict(v["tensors"])
            meta = t.pop("__metadata__", None)
            ds = 8 + v["header_len"]
            check_tensor_table(fn, t, ds, v["size"])
            self.headers[fn] = (t, ds, meta, v["size"])
            for k in t:
                if k in self.weight_map:
                    die("dump: %s in %s and %s" % (k, self.weight_map[k], fn))
                self.weight_map[k] = fn
        self.files = sorted(self.headers)
        self.stray = []

    def file_path(self, fn):
        return None

    def pread(self, fn, offset, length):
        return None


class AttnSource:
    def __init__(self, path=None, plan=None):
        self.path = path
        if path:
            self.tensors, self.data_start, self.meta = read_header(path)
            check_tensor_table(os.path.basename(path), self.tensors, self.data_start,
                               os.path.getsize(path))
            man = os.path.splitext(path)[0] + ".manifest.json"
            self.manifest = json.load(open(man)) if os.path.exists(man) else None
        else:
            with open(plan) as f:
                p = json.load(f)
            self.tensors = {}
            off = 0
            for t in p["tensors"]:
                self.tensors[t["name"]] = {"dtype": t["dtype"], "shape": t["shape"],
                                           "data_offsets": [off, off + t["size"]]}
                off += t["size"]
            self.data_start = 0
            self.meta = {"source_repo": p.get("repo"), "source_revision": p.get("revision")}
            self.manifest = None

    def pread(self, offset, length):
        if not self.path:
            return None
        with open(self.path, "rb") as f:
            f.seek(offset)
            data = f.read(length)
        if len(data) != length:
            die("short read of the attention file")
        return data


# --------------------------------------------------------------------------
# vLLM compressed-tensors ignore semantics (vllm/model_executor/layers/
# quantization/utils/config_utils.py: exact string or "re:" + re.match)
# --------------------------------------------------------------------------
def ignore_matches(name, ignore):
    for pat in ignore:
        if pat.startswith("re:"):
            if re.match(pat[3:], name):
                return True
        elif pat == name:
            return True
    return False


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def expected_attention_modules(config):
    tc = config.get("text_config", config)
    mods = []
    for L, t in enumerate(tc["layer_types"]):
        if t == "full_attention":
            projs = QSA_PROJ
        elif t == "linear_attention":
            projs = GDN_PROJ
        else:
            die("unknown layer type %r at layer %d" % (t, L))
        mods += ["model.language_model.layers.%d.%s" % (L, p) for p in projs]
    return mods


def plan_build(src, attn, placement):
    cfg = src.config
    qc = cfg.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        die("source quant_method is %r, expected compressed-tensors" % qc.get("quant_method"))
    ignore = list(qc.get("ignore") or [])

    # --- the BF16 attention tensors
    bf16 = {}
    for name, info in attn.tensors.items():
        if not ATTN_RE.match(name):
            die("attention file holds an unexpected tensor %s" % name)
        if info["dtype"] != "BF16":
            die("%s is %s, expected BF16" % (name, info["dtype"]))
        bf16[module_of(name)] = (name, info)
    want = expected_attention_modules(cfg)
    if set(bf16) != set(want):
        die("attention file covers %d modules, config needs %d (missing %s, extra %s)" % (
            len(bf16), len(want), sorted(set(want) - set(bf16))[:4], sorted(set(bf16) - set(want))[:4]))

    # --- what the source stores for those modules
    drop = {}        # tensor name -> module
    home = {}        # module -> file that held its weight_packed
    under = {}       # module -> tensor names directly under it
    for k in src.weight_map:
        under.setdefault(module_of(k), []).append(k)
    for mod in want:
        parts = {}
        for p in INT4_PARTS:
            n = "%s.%s" % (mod, p)
            if n in src.weight_map:
                parts[p] = n
        missing = [p for p in REQUIRED_PARTS if p not in parts]
        if missing:
            die("%s: source lacks %s - is the source really the INT4 AWQ?" % (mod, missing))
        if mod + ".weight" in src.weight_map:
            die("%s.weight already exists in the source" % mod)
        for n in parts.values():
            drop[n] = mod
        home[mod] = src.weight_map[parts["weight_packed"]]
        # any other tensor under this module would be an unknown quant param
        others = [k for k in under.get(mod, []) if k not in parts.values()]
        if others:
            die("%s: unexpected extra tensors %s" % (mod, others[:4]))
        # packed shape [out, in/8] must agree with the BF16 shape [out, in]
        fn = src.weight_map[parts["weight_packed"]]
        pshape = src.headers[fn][0][parts["weight_packed"]]["shape"]
        bshape = bf16[mod][1]["shape"]
        if pshape != [bshape[0], bshape[1] // 8] or bshape[1] % 8:
            die("%s: packed shape %s does not fit BF16 shape %s" % (mod, pshape, bshape))
        # weight_shape holds the true [out, in] as int64 data
        wsn = parts["weight_shape"]
        wfn = src.weight_map[wsn]
        whdr, wds = src.headers[wfn][0], src.headers[wfn][1]
        b, e = whdr[wsn]["data_offsets"]
        raw = src.pread(wfn, wds + b, e - b)
        if raw is not None:
            ws = list(struct.unpack("<%dq" % ((e - b) // 8), raw))
            if ws != bshape:
                die("%s: AWQ weight_shape %s != BF16 shape %s" % (mod, ws, bshape))

    # --- config: ignore entries for the attention modules
    added_ignore = [m for m in want if not ignore_matches(m, ignore)]
    new_ignore = ignore + added_ignore

    # --- per output file
    outputs = []
    by_file = {}
    for mod, fn in home.items():
        by_file.setdefault(fn, []).append(mod)
    for fn in src.files:
        hdr = src.headers[fn][0]
        dropped = sorted(n for n in hdr if n in drop)
        adds = sorted(by_file.get(fn, [])) if placement == "inplace" else []
        if not dropped and not adds:
            outputs.append({"name": fn, "action": "link", "src": fn,
                            "size": src.headers[fn][3], "tensors": len(hdr)})
            continue
        if any(PLE_MARK in n for n in hdr):
            die("%s holds PLE n-gram tensors AND INT4 attention; rewriting it would move "
                "PLE data_offsets - refusing (use --placement extra or adapt the script)" % fn)
        entries = [(n, hdr[n]["dtype"], hdr[n]["shape"], ("src", fn, n)) for n in hdr if n not in drop]
        entries += [(bf16[m][0], "BF16", bf16[m][1]["shape"], ("attn", None, bf16[m][0])) for m in adds]
        outputs.append({"name": fn, "action": "rewrite", "entries": entries, "meta": src.headers[fn][2],
                        "dropped": dropped, "added": [bf16[m][0] for m in adds]})
    if placement == "extra":
        if EXTRA_SHARD in src.headers:
            die("%s already exists in the source" % EXTRA_SHARD)
        entries = [(bf16[m][0], "BF16", bf16[m][1]["shape"], ("attn", None, bf16[m][0])) for m in sorted(bf16)]
        outputs.append({"name": EXTRA_SHARD, "action": "new", "entries": entries, "meta": {"format": "pt"},
                        "dropped": [], "added": [e[0] for e in entries]})
    for o in outputs:
        if o["action"] != "link":
            o["entries"] = order_entries(o["entries"])
            o["header"] = make_header(o["entries"], o["meta"])
            o["size"] = 8 + len(o["header"]) + sum(nbytes(d, s) for _, d, s, _ in o["entries"])
            o["tensors"] = len(o["entries"])

    new_cfg = json.loads(json.dumps(cfg))
    new_cfg["quantization_config"]["ignore"] = new_ignore
    return {"outputs": outputs, "drop": drop, "bf16": bf16, "modules": want,
            "added_ignore": added_ignore, "config": new_cfg}


def order_entries(entries):
    """Same order as the Rust safetensors serializer: by dtype size (desc), then
    name.  Every tensor then starts at a multiple of its element size."""
    return sorted(entries, key=lambda e: (-DTYPE_SIZE[e[1]], e[0]))


def make_header(entries, meta):
    hdr = {}
    if meta is not None:
        hdr["__metadata__"] = meta
    off = 0
    for name, dtype, shape, _ in entries:
        size = nbytes(dtype, shape)
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    return raw + b" " * ((8 - len(raw) % 8) % 8)


# --------------------------------------------------------------------------
# config checks emulating vLLM
# --------------------------------------------------------------------------
def check_config(cfg, out_tensors, attn_modules):
    ignore = cfg["quantization_config"]["ignore"]
    if len(set(ignore)) != len(ignore):
        die("config ignore list has duplicates")
    quantized = {module_of(n) for n in out_tensors if n.endswith(".weight_packed")}
    bad = sorted(m for m in quantized if ignore_matches(m, ignore))
    if bad:
        die("ignored modules still have INT4 tensors: %s" % bad[:4])
    for m in attn_modules:
        if not ignore_matches(m, ignore):
            die("%s is BF16 but not in quantization_config.ignore" % m)
        if m in quantized:
            die("%s still has INT4 tensors" % m)
        if m + ".weight" not in out_tensors:
            die("%s.weight missing" % m)
    # fused vLLM modules: all shards ignored or none (vllm raises otherwise)
    mods = {module_of(n) for n in out_tensors}
    checked = 0
    for fused, members in FUSED_GROUPS.items():
        parents = {m[: -len(members[0])] for m in mods if m.endswith("." + members[0])}
        for parent in parents:
            states = [ignore_matches(parent + mem, ignore) for mem in members]
            if any(states) and not all(states):
                die("fused %s%s: shards have different schemes %s" % (parent, fused, states))
            checked += 1
    return len(quantized), checked


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------
def write_all(fd, data):
    mv = memoryview(data)
    while mv:
        n = os.write(fd, mv)
        if n <= 0:
            die("write failed")
        mv = mv[n:]


def fadvise_dontneed(fd, offset, length):
    if hasattr(os, "posix_fadvise"):
        try:
            os.posix_fadvise(fd, offset, length, os.POSIX_FADV_DONTNEED)
        except OSError:
            pass


def write_shard(out_path, o, src, attn, do_hash):
    """Write one rewritten/new shard: header, then each tensor's raw bytes."""
    tmp = out_path + ".tmp"
    h = hashlib.sha256() if do_hash else None
    src_fds = {}
    written = 0
    t0 = time.time()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        head = struct.pack("<Q", len(o["header"])) + o["header"]
        write_all(fd, head)
        if h:
            h.update(head)
        flushed = len(head)
        pos = len(head)
        for name, dtype, shape, (kind, fn, sname) in o["entries"]:
            if kind == "src":
                info = src.headers[fn][0][sname]
                in_off = src.headers[fn][1] + info["data_offsets"][0]
                if fn not in src_fds:
                    src_fds[fn] = os.open(src.file_path(fn), os.O_RDONLY)
                in_fd = src_fds[fn]
            else:
                info = attn.tensors[sname]
                in_off = attn.data_start + info["data_offsets"][0]
                if "attn" not in src_fds:
                    src_fds["attn"] = os.open(attn.path, os.O_RDONLY)
                in_fd = src_fds["attn"]
            if info["dtype"] != dtype or info["shape"] != list(shape):
                die("internal: %s dtype/shape changed" % name)
            left = nbytes(dtype, shape)
            off = in_off
            while left:
                buf = os.pread(in_fd, min(CHUNK, left), off)
                if not buf:
                    die("unexpected EOF reading %s" % name)
                write_all(fd, buf)
                if h:
                    h.update(buf)
                fadvise_dontneed(in_fd, off, len(buf))
                off += len(buf)
                left -= len(buf)
                pos += len(buf)
                written += len(buf)
                if pos - flushed >= FLUSH_EVERY:
                    os.fdatasync(fd)
                    fadvise_dontneed(fd, flushed, pos - flushed)
                    flushed = pos
        os.fsync(fd)
        fadvise_dontneed(fd, 0, 0)
    except BaseException:
        os.close(fd)
        for x in src_fds.values():
            os.close(x)
        raise
    os.close(fd)
    for x in src_fds.values():
        os.close(x)
    size = os.path.getsize(tmp)
    if size != o["size"]:
        die("%s: wrote %d bytes, expected %d" % (tmp, size, o["size"]))
    os.rename(tmp, out_path)
    dt = time.time() - t0
    return (h.hexdigest() if h else None), written / max(dt, 1e-6)


def copy_small_files(src_dir, dst_dir):
    copied, skipped = [], []
    for fn in sorted(os.listdir(src_dir)):
        p = os.path.join(src_dir, fn)
        if fn.endswith(".safetensors") or fn in (INDEX, "config.json", BUILD_MANIFEST):
            continue
        if not os.path.isfile(p):
            skipped.append(fn + ("/" if os.path.isdir(p) else ""))
            continue
        target = fn
        if fn == "FILE_MANIFEST.json":
            # its sha256 values are stale for the rewritten shards
            target = "FILE_MANIFEST.swift15-awq.json"
        shutil.copy2(p, os.path.join(dst_dir, target))
        copied.append(target)
    return copied, skipped


# --------------------------------------------------------------------------
# verification of a built directory
# --------------------------------------------------------------------------
def verify(dst, src, attn, plan, bytes_mode, copied=False):
    log("verify %s" % dst)
    built = DirCheckpoint(dst)        # index <-> files, header sanity, file sizes
    seen = {}
    for fn in built.files:
        for n in built.headers[fn][0]:
            if n in seen:
                die("verify: %s in %s and %s" % (n, seen[n], fn))
            seen[n] = fn
            if built.weight_map.get(n) != fn:
                die("verify: index maps %s to %s, it is in %s" % (n, built.weight_map.get(n), fn))
    if set(seen) != set(built.weight_map):
        die("verify: index lists tensors not present in any file: %s" %
            sorted(set(built.weight_map) - set(seen))[:4])
    if built.stray:
        die("verify: safetensors files not in the index: %s" % built.stray)

    expected = (set(src.weight_map) - set(plan["drop"])) | {v[0] for v in plan["bf16"].values()}
    lost = expected - set(seen)
    extra = set(seen) - expected
    if lost or extra:
        die("verify: lost %d tensors (%s), unexpected %d (%s)" % (
            len(lost), sorted(lost)[:3], len(extra), sorted(extra)[:3]))
    for n, fn in seen.items():
        info = built.headers[fn][0][n]
        ref = attn.tensors[n] if n in attn.tensors else src.headers[src.weight_map[n]][0][n]
        if info["dtype"] != ref["dtype"] or info["shape"] != ref["shape"]:
            die("verify: %s is %s %s, expected %s %s" % (n, info["dtype"], info["shape"],
                                                         ref["dtype"], ref["shape"]))
    log("verify: %d tensors in %d files, each exactly once, none lost; dtypes/shapes OK" % (
        len(seen), len(built.files)))

    # hardlinks and PLE
    nlink = 0
    for o in plan["outputs"]:
        if o["action"] == "link":
            if copied:
                if built.headers[o["name"]][0] != src.headers[o["src"]][0] or \
                        built.headers[o["name"]][3] != src.headers[o["src"]][3]:
                    die("verify: copied %s differs from the source" % o["name"])
            elif not os.path.samefile(os.path.join(dst, o["name"]), src.file_path(o["src"])):
                die("verify: %s is not a hardlink of the source file" % o["name"])
            nlink += 1
    ple = sorted(n for n in seen if PLE_MARK in n)
    for n in ple:
        fn = seen[n]
        if fn != src.weight_map[n]:
            die("verify: PLE tensor %s moved from %s to %s" % (n, src.weight_map[n], fn))
        if not copied and not os.path.samefile(os.path.join(dst, fn), src.file_path(fn)):
            die("verify: PLE file %s is not byte-identical (not a hardlink)" % fn)
        if built.headers[fn][0][n]["data_offsets"] != src.headers[fn][0][n]["data_offsets"] \
                or built.headers[fn][1] != src.headers[fn][1]:
            die("verify: PLE tensor %s changed offsets" % n)
    ple_prefixes = {}
    for n in ple:
        pre, idx = n.rsplit(".shard_", 1)
        ple_prefixes.setdefault(pre, []).append(int(idx.split(".")[0]))
    for pre, idxs in ple_prefixes.items():
        if sorted(idxs) != list(range(len(idxs))):
            die("verify: PLE shard indices of %s not contiguous" % pre)
    log("verify: %d files %s; %d PLE tensors in %d tables unchanged (%s, same data_offsets)" % (
        nlink, "copied" if copied else "hardlinked", len(ple), len(ple_prefixes),
        "same header" if copied else "same inode"))

    nq, ngroups = check_config(built.config, seen, plan["modules"])
    with open(os.path.join(src.path, "config.json")) as f:
        src_cfg = json.load(f)
    a = {k: v for k, v in built.config.items() if k != "quantization_config"}
    b = {k: v for k, v in src_cfg.items() if k != "quantization_config"}
    qa = dict(built.config["quantization_config"])
    qb = dict(src_cfg["quantization_config"])
    qa.pop("ignore")
    qb.pop("ignore")
    if a != b or qa != qb:
        die("verify: config.json changed outside quantization_config.ignore")
    log("verify: config OK - %d attention modules ignored (BF16), %d INT4 modules still "
        "quantized, %d fused vLLM modules consistent" % (len(plan["modules"]), nq, ngroups))

    # bytes
    if bytes_mode != "none":
        compared = 0
        for o in plan["outputs"]:
            if o["action"] == "link":
                continue
            hdr, ds = built.headers[o["name"]][0], built.headers[o["name"]][1]
            with open(os.path.join(dst, o["name"]), "rb") as f:
                for name, dtype, shape, (kind, fn, sname) in o["entries"]:
                    size = nbytes(dtype, shape)
                    if kind == "src":
                        info = src.headers[fn][0][sname]
                        ref_off = src.headers[fn][1] + info["data_offsets"][0]
                        rd = lambda off, ln, fn=fn: src.pread(fn, off, ln)
                    else:
                        ref_off = attn.data_start + attn.tensors[sname]["data_offsets"][0]
                        rd = attn.pread
                    out_off = ds + hdr[name]["data_offsets"][0]
                    if bytes_mode == "full" or size <= 2 * 2**20:
                        ranges = [(0, size)]
                    else:
                        ranges = [(0, 2**20), (size - 2**20, 2**20)]
                    for r0, ln in ranges:
                        pos = 0
                        while pos < ln:
                            step = min(CHUNK, ln - pos)
                            f.seek(out_off + r0 + pos)
                            if f.read(step) != rd(ref_off + r0 + pos, step):
                                die("verify: bytes of %s differ in %s" % (name, o["name"]))
                            pos += step
                    compared += 1
        log("verify: bytes of %d tensors in rewritten shards equal their sources (%s)" % (
            compared, "full" if bytes_mode == "full" else "small tensors full, big ones first/last MiB"))
    if attn.manifest and bytes_mode == "full":
        for mod, (name, _) in plan["bf16"].items():
            fn = seen[name]
            hdr, ds = built.headers[fn][0], built.headers[fn][1]
            b, e = hdr[name]["data_offsets"]
            h = hashlib.sha256()
            with open(os.path.join(dst, fn), "rb") as f:
                f.seek(ds + b)
                left = e - b
                while left:
                    c = f.read(min(CHUNK, left))
                    h.update(c)
                    left -= len(c)
            if h.hexdigest() != attn.manifest["tensors"][name]["sha256"]:
                die("verify: sha256 of %s differs from the fetch manifest" % name)
        log("verify: sha256 of all %d BF16 tensors match the fetch manifest" % len(plan["bf16"]))
    return built


# --------------------------------------------------------------------------
def summarize(plan, src):
    outs = plan["outputs"]
    link = [o for o in outs if o["action"] == "link"]
    rew = [o for o in outs if o["action"] != "link"]
    ple_files = [o for o in link if src.headers[o["src"]][0] and
                 all(PLE_MARK in n for n in src.headers[o["src"]][0])]
    new_bytes = sum(o["size"] for o in rew)
    link_bytes = sum(o["size"] for o in link)
    dropped = sum(len(o["dropped"]) for o in rew)
    drop_bytes = sum(nbytes(src.headers[src.weight_map[n]][0][n]["dtype"],
                            src.headers[src.weight_map[n]][0][n]["shape"]) for n in plan["drop"])
    add_bytes = sum(nbytes("BF16", v[1]["shape"]) for v in plan["bf16"].values())
    log("plan:")
    log("  hardlink %d files, %.2f GiB (%d of them PLE-only, %.2f GiB) - no new space" % (
        len(link), link_bytes / GiB, len(ple_files), sum(o["size"] for o in ple_files) / GiB))
    log("  write    %d files, %.2f GiB new" % (len(rew), new_bytes / GiB))
    for o in rew:
        log("     %-44s %3d INT4 tensors out, %2d BF16 in, %6d tensors, %.3f GiB" % (
            o["name"], len(o["dropped"]), len(o["added"]), o["tensors"], o["size"] / GiB))
    log("  INT4 attention dropped: %d tensors, %.3f GiB; BF16 attention added: %d tensors, %.3f GiB" % (
        dropped, drop_bytes / GiB, len(plan["bf16"]), add_bytes / GiB))
    log("  weights grow by %.3f GiB (GPU memory for weights +%.2f GiB in total)" % (
        (add_bytes - drop_bytes) / GiB, (add_bytes - drop_bytes) / GiB))
    log("  config: %d ignore entries added (%d -> %d)" % (
        len(plan["added_ignore"]), len(plan["config"]["quantization_config"]["ignore"]) -
        len(plan["added_ignore"]), len(plan["config"]["quantization_config"]["ignore"])))
    return new_bytes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", help="source AWQ model dir")
    ap.add_argument("--src-dump", help="header dump JSON instead of --src (implies --dry-run)")
    ap.add_argument("--attn", help="BF16 attention safetensors from fetch_bf16_attention.py")
    ap.add_argument("--attn-plan", help="fetch plan.json instead of --attn (dry run only)")
    ap.add_argument("--dst", help="output model dir (must not exist)")
    ap.add_argument("--placement", choices=("inplace", "extra"), default="inplace",
                    help="BF16 tensors into the shard their INT4 parts came from (default) "
                         "or into one extra shard")
    ap.add_argument("--dry-run", action="store_true", help="plan, check and estimate only")
    ap.add_argument("--verify-only", action="store_true", help="only verify an existing --dst")
    ap.add_argument("--verify-bytes", choices=("none", "sample", "full"), default="sample")
    ap.add_argument("--no-sha256", action="store_true", help="do not hash the written shards")
    ap.add_argument("--emit-config", help="also write the new config.json here (for review)")
    ap.add_argument("--copy-unchanged", action="store_true",
                    help="copy instead of hardlink (needs ~102 GiB more; only if dst is on "
                         "another filesystem)")
    args = ap.parse_args(argv)

    if args.src_dump:
        src = DumpCheckpoint(args.src_dump)
        args.dry_run = True
    elif args.src:
        src = DirCheckpoint(args.src)
    else:
        die("--src or --src-dump is required")
    if args.attn:
        attn = AttnSource(path=os.path.abspath(args.attn))
    elif args.attn_plan and args.dry_run:
        attn = AttnSource(plan=args.attn_plan)
    else:
        die("--attn is required (--attn-plan only with --dry-run)")
    log("source %s: %d files, %d tensors" % (src.path, len(src.files), len(src.weight_map)))
    if src.stray:
        log("  note: %d safetensors files in the source are not in its index and are ignored: %s" % (
            len(src.stray), src.stray))
    log("attention: %d BF16 tensors from %s @ %s" % (
        len(attn.tensors), (attn.meta or {}).get("source_repo"), (attn.meta or {}).get("source_revision")))

    plan = plan_build(src, attn, args.placement)
    # tensor bookkeeping on the plan itself: each tensor exactly once
    names = []
    for o in plan["outputs"]:
        names += list(src.headers[o["src"]][0]) if o["action"] == "link" else [e[0] for e in o["entries"]]
    if len(names) != len(set(names)):
        dup = sorted({n for n in names if names.count(n) > 1})[:4]
        die("plan: duplicated tensors %s" % dup)
    expected = (set(src.weight_map) - set(plan["drop"])) | {v[0] for v in plan["bf16"].values()}
    if set(names) != expected:
        die("plan: lost/extra tensors %s" % sorted(set(names) ^ expected)[:4])
    nq, ngroups = check_config(plan["config"], set(names), plan["modules"])
    log("plan checks: %d tensors, each exactly once; %d INT4 modules stay quantized; "
        "%d fused vLLM modules consistent" % (len(names), nq, ngroups))
    new_bytes = summarize(plan, src)
    if args.emit_config:
        with open(args.emit_config, "w") as f:
            json.dump(plan["config"], f, indent=2)
            f.write("\n")
        log("new config.json written to %s" % args.emit_config)
    if args.copy_unchanged:
        new_bytes += sum(o["size"] for o in plan["outputs"] if o["action"] == "link")

    if args.verify_only:
        if not args.dst or not os.path.isdir(args.dst):
            die("--verify-only needs an existing --dst")
        verify(os.path.abspath(args.dst), src, attn, plan, args.verify_bytes, args.copy_unchanged)
        log("VERIFY OK")
        return 0

    small = 0
    if not src.virtual:
        small = sum(os.path.getsize(os.path.join(src.path, f)) for f in os.listdir(src.path)
                    if os.path.isfile(os.path.join(src.path, f)) and not f.endswith(".safetensors"))
    log("disk: needs %.2f GiB new space (+%.3f GiB small files)" % (new_bytes / GiB, small / GiB))

    if args.dry_run:
        if args.dst and not src.virtual:
            parent = os.path.dirname(os.path.abspath(args.dst)) or "."
            st = os.statvfs(parent)
            log("disk: %.2f GiB free in %s" % (st.f_bavail * st.f_frsize / GiB, parent))
        log("dry run: nothing written")
        return 0

    if not args.dst:
        die("--dst is required")
    dst = os.path.abspath(args.dst)
    building = dst + ".building"
    for p in (dst, building):
        if os.path.lexists(p):
            die("%s already exists - refusing to overwrite (remove it yourself)" % p)
    parent = os.path.dirname(dst)
    if not os.path.isdir(parent):
        die("parent directory %s does not exist" % parent)
    if not args.copy_unchanged and os.stat(parent).st_dev != os.stat(src.path).st_dev:
        die("%s and %s are on different filesystems: hardlinks impossible "
            "(use --copy-unchanged, needs ~102 GiB more)" % (parent, src.path))
    st = os.statvfs(parent)
    free = st.f_bavail * st.f_frsize
    need = new_bytes + small + 2 * GiB
    log("disk: %.2f GiB free in %s, need %.2f GiB incl. 2 GiB margin" % (free / GiB, parent, need / GiB))
    if free < need:
        die("not enough free space")

    os.mkdir(building)
    log("building in %s" % building)
    t0 = time.time()
    record = []
    for i, o in enumerate(plan["outputs"], 1):
        out_path = os.path.join(building, o["name"])
        if o["action"] == "link":
            if args.copy_unchanged:
                shutil.copyfile(src.file_path(o["src"]), out_path)
            else:
                os.link(src.file_path(o["src"]), out_path)
            record.append({"file": o["name"], "action": "copy" if args.copy_unchanged else "hardlink",
                           "size": o["size"], "tensors": o["tensors"]})
            continue
        sha, rate = write_shard(out_path, o, src, attn, not args.no_sha256)
        log("  [%d/%d] wrote %s %.3f GiB (%.0f MiB/s)" % (i, len(plan["outputs"]), o["name"],
                                                         o["size"] / GiB, rate / 2**20))
        record.append({"file": o["name"], "action": {"rewrite": "rewritten", "new": "new"}[o["action"]],
                       "size": o["size"], "tensors": o["tensors"], "sha256": sha,
                       "dropped_int4": o["dropped"], "added_bf16": o["added"]})

    # index from the headers that were just written
    weight_map = {}
    total = 0
    for o in plan["outputs"]:
        hdr = read_header(os.path.join(building, o["name"]))[0]
        for n, info in hdr.items():
            if n in weight_map:
                die("duplicate tensor %s" % n)
            weight_map[n] = o["name"]
            total += info["data_offsets"][1] - info["data_offsets"][0]
    meta = dict(src.index_meta)
    meta["total_size"] = total
    with open(os.path.join(building, INDEX), "w") as f:
        json.dump({"metadata": meta, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
        f.write("\n")
    with open(os.path.join(building, "config.json"), "w") as f:
        json.dump(plan["config"], f, indent=2)
        f.write("\n")
    copied, skipped = copy_small_files(src.path, building)
    log("copied %d small files: %s" % (len(copied), ", ".join(copied)))
    if skipped:
        log("not copied (not regular files): %s" % skipped)
    with open(os.path.join(building, BUILD_MANIFEST), "w") as f:
        json.dump({
            "name": "Swift-A16",
            "description": "Swift 1.5 Qwen3.8-Flash-Next W4A16 AWQ (compressed-tensors) with the "
                           "attention projections (QSA q/k/v/o_proj, GDN in_proj_qkv/in_proj_z/out_proj) "
                           "replaced by the BF16 originals; routed experts stay INT4.",
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_awq_dir": src.path,
            "attention_file": attn.path,
            "attention_source": attn.meta,
            "placement": args.placement,
            "ignore_entries_added": plan["added_ignore"],
            "files": record,
        }, f, indent=1)
    for p in [building] + [os.path.join(building, x) for x in os.listdir(building)]:
        try:
            fd = os.open(p, os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass

    verify(building, src, attn, plan, args.verify_bytes, args.copy_unchanged)
    os.rename(building, dst)
    log("DONE in %.0f s: %s" % (time.time() - t0, dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
