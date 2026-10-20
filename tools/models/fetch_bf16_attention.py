#!/usr/bin/env python3
"""Fetch only the BF16 attention projections of Swift 1.5 from Hugging Face.

Swift-A16 = the Swift 1.5 W4A16 AWQ checkpoint with its attention projections
replaced by the BF16 originals.  This script downloads exactly those tensors
(156 of them, about 4.98 GiB) from the BF16 release with HTTP Range requests,
using each shard's safetensors header to find the byte range of each tensor,
and writes them into ONE new safetensors file.  It never downloads a whole
shard (the BF16 repo is 335 GiB).

Python 3.8+ standard library only (urllib); no torch, no requests, no
huggingface_hub.

Tensors fetched (HF names, all BF16):
    model.language_model.layers.{L}.self_attn.{q,k,v,o}_proj.weight       (12 QSA layers)
    model.language_model.layers.{L}.linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight
                                                                           (36 GDN layers)

Usage (on the server):
    python3 fetch_bf16_attention.py --work-dir /opt/vllm-qfn/work/swift-a16 \
        --awq-dir /opt/vllm-qfn/models/swift15-awq
    python3 fetch_bf16_attention.py --work-dir ... --plan-only     # headers + sizes only

Resumable: run the same command again after an interruption; finished tensors
are kept, a partly downloaded tensor continues where its last checkpoint was.

Output (in --work-dir):
    swift15-bf16-attention.safetensors           the tensors (header + raw bytes)
    swift15-bf16-attention.manifest.json         source repo/revision, per-tensor
                                                 source shard, byte range, sha256
    headers/                                     cached shard headers of the BF16 repo
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import struct
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array

DEFAULT_REPO = "ukisai/Swift1.5-Qwen3.8-Flash-Next"
# Only commit of the BF16 repo ("initial release", 2026-09-24).  Pinned so the
# bytes cannot change under us between the header fetch and the data fetch.
DEFAULT_REVISION = "0bd4fe22431372cdad1979267d3ab45aa7e6150a"
OUT_NAME = "swift15-bf16-attention.safetensors"

# Attention projections that the AWQ checkpoint stores as INT4.
ATTN_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\."
    r"(self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
    r"|linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj))\.weight$"
)
QSA_PROJ = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
GDN_PROJ = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj")

DTYPE_SIZE = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8,
}
MAX_HEADER = 100_000_000
CHUNK = 1024 * 1024   # resp.read(n) blocks until n bytes: keep it small
CHECKPOINT_EVERY = 16 * 1024 * 1024   # --checkpoint-mib


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def nbytes(dtype: str, shape) -> int:
    n = DTYPE_SIZE[dtype]
    for s in shape:
        n *= int(s)
    return n


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class _StripAuthOnCrossHost(urllib.request.HTTPRedirectHandler):
    """Follow the huggingface.co -> CDN redirect, but never send the token to
    another host (urllib would otherwise forward the Authorization header)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            old_host = urllib.parse.urlsplit(req.full_url).hostname
            new_host = urllib.parse.urlsplit(new.full_url).hostname
            if old_host != new_host:
                new.remove_header("Authorization")
        return new


class Http:
    def __init__(self, endpoint, repo, revision, token, timeout, retries):
        self.endpoint = endpoint.rstrip("/")
        self.repo = repo
        self.revision = revision
        self.token = token
        self.timeout = timeout
        self.retries = retries
        self.opener = urllib.request.build_opener(_StripAuthOnCrossHost())

    def url(self, filename: str) -> str:
        return "%s/%s/resolve/%s/%s" % (
            self.endpoint,
            self.repo,
            urllib.parse.quote(self.revision, safe=""),
            urllib.parse.quote(filename),
        )

    def _request(self, filename, start=None, end=None):
        """Open filename; with start/end (inclusive) as a Range request."""
        req = urllib.request.Request(self.url(filename))
        req.add_header("User-Agent", "swift-a16-fetch/1.0 (python-urllib)")
        req.add_header("Accept-Encoding", "identity")
        if self.token:
            req.add_header("Authorization", "Bearer " + self.token)
        if start is not None:
            req.add_header("Range", "bytes=%d-%d" % (start, end))
        resp = self.opener.open(req, timeout=self.timeout)
        if start is not None:
            if resp.status != 206:
                resp.close()
                raise IOError(
                    "%s: server answered %s to a Range request (expected 206); "
                    "refusing to download the whole file" % (filename, resp.status)
                )
            cr = resp.headers.get("Content-Range", "")
            m = re.match(r"bytes (\d+)-(\d+)/(\d+|\*)", cr)
            if not m or int(m.group(1)) != start or int(m.group(2)) != end:
                resp.close()
                raise IOError("%s: bad Content-Range %r for %d-%d" % (filename, cr, start, end))
        return resp

    def with_retries(self, what, fn):
        delay = 2.0
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except urllib.error.HTTPError as e:
                # 4xx other than 403/408/429 will not get better by retrying
                # (403 can be an expired signed CDN URL -> re-resolve).
                if e.code in (401, 404) or (400 <= e.code < 500 and e.code not in (403, 408, 429)):
                    raise
                err = "HTTP %s" % e.code
                ra = e.headers.get("Retry-After") if e.headers else None
                if ra and ra.isdigit():
                    delay = max(delay, float(ra))
            except (urllib.error.URLError, OSError, ConnectionError) as e:
                err = repr(e)
            if attempt == self.retries:
                raise IOError("%s: giving up after %d attempts (%s)" % (what, attempt, err))
            log("  retry %d/%d for %s in %.0fs: %s" % (attempt, self.retries, what, delay, err))
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

    def get_small(self, filename: str) -> bytes:
        def go():
            with self._request(filename) as r:
                return r.read()
        return self.with_retries(filename, go)

    def get_range(self, filename: str, start: int, end: int) -> bytes:
        """Bytes [start, end] inclusive, fully read, length-checked."""
        def go():
            with self._request(filename, start, end) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError("short read %d of %d" % (len(data), end - start + 1))
            return data
        return self.with_retries("%s[%d-%d]" % (filename, start, end), go)


def fetch_header(http: Http, filename: str, cache_dir: str) -> tuple:
    """(header dict, data_start) of a remote safetensors shard; cached on disk."""
    cache = os.path.join(cache_dir, "%s@%s.header.json" % (filename, http.revision[:12]))
    if os.path.exists(cache):
        with open(cache) as f:
            c = json.load(f)
        return c["header"], c["data_start"], c["file_size"]
    raw = http.get_range(filename, 0, 7)
    (hlen,) = struct.unpack("<Q", raw)
    if not 0 < hlen <= MAX_HEADER:
        raise ValueError("%s: implausible header length %d" % (filename, hlen))
    hdr = json.loads(http.get_range(filename, 8, 8 + hlen - 1))
    # file size from a 1-byte probe at the end of the header region is not
    # available; compute what the header implies and record it.
    ends = [v["data_offsets"][1] for k, v in hdr.items() if k != "__metadata__"]
    data_start = 8 + hlen
    file_size = data_start + (max(ends) if ends else 0)
    tmp = cache + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"header": hdr, "data_start": data_start, "file_size": file_size}, f)
    os.replace(tmp, cache)
    return hdr, data_start, file_size


# --------------------------------------------------------------------------
# safetensors writing helpers
# --------------------------------------------------------------------------
def build_header(entries, metadata) -> bytes:
    """entries: list of (name, dtype, shape, size) in data order.  Returns the
    header bytes padded with spaces to a multiple of 8 (safetensors convention,
    keeps the data section 8-byte aligned)."""
    hdr = {}
    if metadata:
        hdr["__metadata__"] = metadata
    off = 0
    for name, dtype, shape, size in entries:
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + size]}
        off += size
    raw = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    raw += b" " * ((8 - len(raw) % 8) % 8)
    return raw


def read_local_header(path: str):
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8:
            raise ValueError("%s: too short for safetensors" % path)
        (hlen,) = struct.unpack("<Q", head)
        if not 0 < hlen <= MAX_HEADER:
            raise ValueError("%s: implausible header length %d" % (path, hlen))
        hdr = json.loads(f.read(hlen))
    return hdr, 8 + hlen


def bf16_sample_stats(data: bytes):
    """(n, nonfinite, zeros, mean_abs) of a little-endian BF16 sample."""
    a = array("H")
    a.frombytes(data[: len(data) - len(data) % 2])
    if sys.byteorder != "little":
        a.byteswap()
    nonfinite = zeros = 0
    tot = 0.0
    for u in a:
        e = (u >> 7) & 0xFF
        if e == 0xFF:
            nonfinite += 1
            continue
        if u & 0x7FFF == 0:
            zeros += 1
            continue
        m = u & 0x7F
        v = (1.0 + m / 128.0) * 2.0 ** (e - 127) if e else (m / 128.0) * 2.0 ** -126
        tot += v
    return len(a), nonfinite, zeros, (tot / len(a) if a else 0.0)


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def make_plan(http: Http, work_dir: str, config: dict):
    hdr_dir = os.path.join(work_dir, "headers")
    os.makedirs(hdr_dir, exist_ok=True)
    log("index: %s" % http.url("model.safetensors.index.json"))
    idx_raw = http.get_small("model.safetensors.index.json")
    with open(os.path.join(work_dir, "bf16-model.safetensors.index.json"), "wb") as f:
        f.write(idx_raw)
    weight_map = json.loads(idx_raw)["weight_map"]

    names = sorted(k for k in weight_map if ATTN_RE.match(k))
    check_coverage(names, config)
    shards = sorted({weight_map[k] for k in names})
    log("%d attention tensors in %d of %d BF16 shards" % (
        len(names), len(shards), len(set(weight_map.values()))))

    headers = {}
    for i, sh in enumerate(shards, 1):
        headers[sh] = fetch_header(http, sh, hdr_dir)
        if i % 6 == 0 or i == len(shards):
            log("  headers %d/%d" % (i, len(shards)))

    plan = []
    for name in names:
        sh = weight_map[name]
        hdr, data_start, _ = headers[sh]
        if name not in hdr:
            raise ValueError("%s: index says %s but its header lacks it" % (name, sh))
        info = hdr[name]
        b, e = info["data_offsets"]
        size = nbytes(info["dtype"], info["shape"])
        if info["dtype"] != "BF16":
            raise ValueError("%s is %s, expected BF16" % (name, info["dtype"]))
        if e - b != size:
            raise ValueError("%s: data_offsets span %d bytes, shape needs %d" % (name, e - b, size))
        plan.append({
            "name": name, "dtype": info["dtype"], "shape": info["shape"], "size": size,
            "src_file": sh, "src_start": data_start + b, "src_end": data_start + e,
        })
    return plan


def check_coverage(names, config: dict):
    """Every QSA layer must have q/k/v/o and every GDN layer qkv/z/out, so that
    vLLM's fused qkv_proj / in_proj_qkvz modules get one consistent scheme."""
    tc = config.get("text_config", config)
    layer_types = tc["layer_types"]
    want = set()
    for L, t in enumerate(layer_types):
        projs = QSA_PROJ if t == "full_attention" else GDN_PROJ if t == "linear_attention" else ()
        if not projs:
            raise ValueError("unknown layer type %r at layer %d" % (t, L))
        for p in projs:
            want.add("model.language_model.layers.%d.%s.weight" % (L, p))
    have = set(names)
    if have != want:
        raise ValueError("attention tensor set mismatch: missing %s, unexpected %s" % (
            sorted(want - have)[:6], sorted(have - want)[:6]))


# --------------------------------------------------------------------------
# AWQ cross-checks (optional, local AWQ dir)
# --------------------------------------------------------------------------
class LocalCheckpoint:
    def __init__(self, path: str):
        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self._headers = {}

    def header(self, filename):
        if filename not in self._headers:
            self._headers[filename] = read_local_header(os.path.join(self.path, filename))
        return self._headers[filename]

    def info(self, name):
        fn = self.weight_map.get(name)
        if fn is None:
            return None
        hdr, ds = self.header(fn)
        return fn, hdr[name], ds

    def read(self, name) -> bytes:
        fn, info, ds = self.info(name)
        b, e = info["data_offsets"]
        with open(os.path.join(self.path, fn), "rb") as f:
            f.seek(ds + b)
            data = f.read(e - b)
        if len(data) != e - b:
            raise IOError("short read of %s" % name)
        return data


def check_against_awq(plan, awq: LocalCheckpoint):
    """BF16 shape == AWQ weight_shape (read from the AWQ file) for every module."""
    bad = 0
    for t in plan:
        mod = t["name"][: -len(".weight")]
        ws = awq.info(mod + ".weight_shape")
        wp = awq.info(mod + ".weight_packed")
        if ws is None or wp is None:
            log("  AWQ lacks %s.weight_shape/_packed" % mod)
            bad += 1
            continue
        raw = awq.read(mod + ".weight_shape")
        shape = list(struct.unpack("<%dq" % (len(raw) // 8), raw))
        pshape = wp[1]["shape"]
        if shape != t["shape"] or pshape != [t["shape"][0], t["shape"][1] // 8]:
            log("  SHAPE MISMATCH %s: bf16 %s, awq weight_shape %s, packed %s" % (
                mod, t["shape"], shape, pshape))
            bad += 1
    if bad:
        raise SystemExit("AWQ cross-check failed for %d modules" % bad)
    log("AWQ cross-check: all %d BF16 shapes equal the AWQ weight_shape values" % len(plan))


def check_preserved_identical(http: Http, awq: LocalCheckpoint, bf16_map, hdr_dir, count: int):
    """Provenance: small tensors that AWQ kept in BF16 must be byte-identical to
    the BF16 repo if the AWQ was made from it."""
    pat = re.compile(
        r"^model\.language_model\.layers\.\d+\.(linear_attn\.(A_log|dt_bias|norm\.weight)"
        r"|self_attn\.(q_norm|k_norm)\.weight)$")
    cands = sorted(k for k in awq.weight_map if pat.match(k) and k in bf16_map)
    if not cands:
        log("provenance: no common small tensors found")
        return
    step = max(1, len(cands) // count)
    picked = cands[::step][:count]
    same = 0
    for name in picked:
        local = awq.read(name)
        sh = bf16_map[name]
        hdr, ds, _ = fetch_header(http, sh, hdr_dir)
        b, e = hdr[name]["data_offsets"]
        remote = http.get_range(sh, ds + b, ds + e - 1)
        ok = remote == local
        same += ok
        log("  provenance %-62s %s" % (name, "identical" if ok else "DIFFERENT"))
    log("provenance: %d/%d preserved tensors byte-identical" % (same, len(picked)))
    if same != len(picked):
        raise SystemExit("the local AWQ checkpoint does not match this BF16 revision")


# --------------------------------------------------------------------------
# download into a preallocated output file (resumable)
# --------------------------------------------------------------------------
class State:
    def __init__(self, path: str, plan_id: str, names):
        self.path = path
        self.lock = threading.Lock()
        self.done = {n: 0 for n in names}
        if os.path.exists(path):
            with open(path) as f:
                s = json.load(f)
            if s.get("plan_id") != plan_id:
                raise SystemExit(
                    "%s belongs to a different plan; delete it and the .partial "
                    "file to start over" % path)
            self.done.update({k: v for k, v in s["done"].items() if k in self.done})
        self.plan_id = plan_id

    def set(self, name, n):
        with self.lock:
            self.done[name] = n
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"plan_id": self.plan_id, "done": self.done}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)


class Progress:
    def __init__(self, total, already):
        self.total = total
        self.done = already
        self.start_done = already
        self.t0 = time.time()
        self.last = 0.0
        self.lock = threading.Lock()

    def add(self, n):
        with self.lock:
            self.done += n
            now = time.time()
            if now - self.last >= 10 or self.done == self.total:
                self.last = now
                rate = (self.done - self.start_done) / max(now - self.t0, 1e-6)
                eta = (self.total - self.done) / rate if rate > 0 else 0
                log("  %.2f / %.2f GiB  %.1f MiB/s  ETA %.0f s" % (
                    self.done / 2**30, self.total / 2**30, rate / 2**20, eta))


def download_tensor(http: Http, fd: int, t, state: State, prog: Progress):
    size, name = t["size"], t["name"]
    pos = state.done[name]
    if pos >= size:
        return
    attempt = 0
    delay = 2.0
    while pos < size:
        try:
            resp = http._request(t["src_file"], t["src_start"] + pos, t["src_end"] - 1)
            since_ckpt = 0
            with resp:
                while pos < size:
                    chunk = resp.read(min(CHUNK, size - pos))
                    if not chunk:
                        raise IOError("connection closed at %d of %d" % (pos, size))
                    written = os.pwrite(fd, chunk, t["out_start"] + pos)
                    if written != len(chunk):
                        raise IOError("short pwrite")
                    pos += len(chunk)
                    since_ckpt += len(chunk)
                    prog.add(len(chunk))
                    if since_ckpt >= CHECKPOINT_EVERY:
                        os.fsync(fd)
                        state.set(name, pos)
                        since_ckpt = 0
            attempt = 0
        except urllib.error.HTTPError as e:
            if e.code in (401, 404):
                raise
            attempt += 1
            err = "HTTP %s" % e.code
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            attempt += 1
            err = repr(e)
        else:
            break
        if attempt > http.retries:
            raise IOError("%s: giving up after %d attempts (%s)" % (name, attempt - 1, err))
        # bytes up to 'pos' were pwritten; make them durable, record, resume there
        os.fsync(fd)
        state.set(name, pos)
        log("  retry %d/%d %s at %d/%d in %.0fs: %s" % (attempt, http.retries, name, pos, size, delay, err))
        time.sleep(delay)
        delay = min(delay * 2, 120.0)
    os.fsync(fd)
    state.set(name, size)


def verify_output(path: str, plan, sample: bool):
    hdr, ds = read_local_header(path)
    names = [k for k in hdr if k != "__metadata__"]
    if sorted(names) != sorted(t["name"] for t in plan):
        raise SystemExit("verify: tensor names in %s differ from the plan" % path)
    if len(set(names)) != len(names):
        raise SystemExit("verify: duplicate tensor names")
    end = 0
    for t in plan:
        info = hdr[t["name"]]
        if info["dtype"] != t["dtype"] or info["shape"] != t["shape"]:
            raise SystemExit("verify: %s has %s %s, expected %s %s" % (
                t["name"], info["dtype"], info["shape"], t["dtype"], t["shape"]))
        b, e = info["data_offsets"]
        if e - b != nbytes(info["dtype"], info["shape"]):
            raise SystemExit("verify: %s byte range does not match shape" % t["name"])
        end = max(end, e)
    offs = sorted(tuple(hdr[n]["data_offsets"]) for n in names)
    pos = 0
    for b, e in offs:
        if b != pos:
            raise SystemExit("verify: hole/overlap in data section at %d" % b)
        pos = e
    if os.path.getsize(path) != ds + end:
        raise SystemExit("verify: file size %d != %d" % (os.path.getsize(path), ds + end))
    log("verify: %d tensors, dtype/shape/offsets OK, file size %d OK" % (len(names), ds + end))
    # sha256 + sampled BF16 sanity (finite values, not all zeros)
    sums = {}
    worst = []
    with open(path, "rb") as f:
        for t in plan:
            b, e = hdr[t["name"]]["data_offsets"]
            h = hashlib.sha256()
            f.seek(ds + b)
            left = e - b
            while left:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    raise SystemExit("verify: short read")
                h.update(chunk)
                left -= len(chunk)
            sums[t["name"]] = h.hexdigest()
            if sample:
                n_tot = nf_tot = z_tot = 0
                mags = []
                for frac in (0.0, 0.5, 0.999):
                    off = b + (int((e - b) * frac) & ~1)
                    off = min(off, e - 65536) if e - b > 65536 else b
                    f.seek(ds + off)
                    n, nf, z, mag = bf16_sample_stats(f.read(min(65536, e - off)))
                    n_tot += n
                    nf_tot += nf
                    z_tot += z
                    mags.append(mag)
                if nf_tot or z_tot == n_tot:
                    worst.append((t["name"], nf_tot, z_tot, n_tot))
                t["sample_mean_abs"] = sum(mags) / len(mags)
    if worst:
        for w in worst:
            log("  SUSPICIOUS %s: %d non-finite, %d zeros of %d sampled" % w)
        raise SystemExit("verify: sampled data looks wrong")
    if sample:
        log("verify: sampled values finite and non-zero in all tensors "
            "(mean|x| %.4f .. %.4f)" % (min(t["sample_mean_abs"] for t in plan),
                                        max(t["sample_mean_abs"] for t in plan)))
    return sums


# --------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", required=True, help="download/work directory (created)")
    ap.add_argument("--out", help="output safetensors (default: WORK_DIR/%s)" % OUT_NAME)
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--endpoint", default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"))
    ap.add_argument("--awq-dir", help="local Swift AWQ dir: cross-check shapes and provenance")
    ap.add_argument("--provenance-samples", type=int, default=12,
                    help="small preserved tensors compared byte-for-byte AWQ vs BF16 (0=off)")
    ap.add_argument("--jobs", type=int, default=4, help="parallel tensor downloads")
    ap.add_argument("--retries", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--plan-only", action="store_true", help="fetch headers, print plan + sizes, stop")
    ap.add_argument("--no-sample-check", action="store_true", help="skip the BF16 value sanity sample")
    ap.add_argument("--checkpoint-mib", type=int, default=16,
                    help="record resume progress every N MiB of a tensor (default 16)")
    args = ap.parse_args(argv)
    global CHECKPOINT_EVERY
    CHECKPOINT_EVERY = max(1, args.checkpoint_mib) * 1024 * 1024

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        tf = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(tf):
            token = open(tf).read().strip() or None
    http = Http(args.endpoint, args.repo, args.revision, token, args.timeout, args.retries)
    os.makedirs(args.work_dir, exist_ok=True)
    out = args.out or os.path.join(args.work_dir, OUT_NAME)
    if os.path.exists(out):
        raise SystemExit("%s already exists; refusing to overwrite (delete it to re-fetch)" % out)

    log("source: %s @ %s" % (args.repo, args.revision))
    config = json.loads(http.get_small("config.json"))
    with open(os.path.join(args.work_dir, "bf16-config.json"), "w") as f:
        json.dump(config, f, indent=2)
    awq = LocalCheckpoint(args.awq_dir) if args.awq_dir else None
    if awq is not None:
        with open(os.path.join(args.awq_dir, "config.json")) as f:
            awq_cfg = json.load(f)
        a = {k: v for k, v in awq_cfg.items() if k != "quantization_config"}
        if a != config:
            diff = sorted(k for k in set(a) | set(config) if a.get(k) != config.get(k))
            log("WARNING: AWQ config.json differs from the BF16 config.json in %s "
                "(check that these are local edits, not a different model)" % diff)
        else:
            log("AWQ config.json == BF16 config.json (apart from quantization_config)")

    plan = make_plan(http, args.work_dir, config)
    total = sum(t["size"] for t in plan)
    by_kind = {}
    for t in plan:
        k = t["name"].split(".", 4)[-1][: -len(".weight")]
        by_kind[k] = by_kind.get(k, 0) + t["size"]
    for k, v in sorted(by_kind.items()):
        log("  %-26s %8.3f GiB" % (k, v / 2**30))
    log("total to download: %d bytes = %.3f GiB (%d tensors)" % (total, total / 2**30, len(plan)))

    if awq is not None:
        check_against_awq(plan, awq)
        if args.provenance_samples:
            bf16_map = json.load(open(os.path.join(args.work_dir, "bf16-model.safetensors.index.json")))["weight_map"]
            check_preserved_identical(http, awq, bf16_map, os.path.join(args.work_dir, "headers"),
                                      args.provenance_samples)

    # output layout: all BF16 -> sort by name; header first, data contiguous
    metadata = {"format": "pt", "source_repo": args.repo, "source_revision": args.revision,
                "content": "Swift 1.5 BF16 attention projections (QSA q/k/v/o, GDN in_proj_qkv/in_proj_z/out_proj)"}
    entries = [(t["name"], t["dtype"], t["shape"], t["size"]) for t in plan]
    header = build_header(entries, metadata)
    data_start = 8 + len(header)
    off = 0
    for t in plan:
        t["out_start"] = data_start + off
        off += t["size"]
    out_size = data_start + total
    plan_id = hashlib.sha256(header).hexdigest()
    with open(os.path.join(args.work_dir, "plan.json"), "w") as f:
        json.dump({"plan_id": plan_id, "repo": args.repo, "revision": args.revision, "tensors": plan}, f, indent=1)
    log("output %s: %d bytes (header %d)" % (out, out_size, len(header)))

    if args.plan_only:
        log("--plan-only: stopping before the data download")
        return 0

    st = os.statvfs(args.work_dir)
    free = st.f_bavail * st.f_frsize
    partial = out + ".partial"
    have = os.path.getsize(partial) if os.path.exists(partial) else 0
    if free < out_size - have + 256 * 2**20:
        raise SystemExit("not enough free space in %s: %.2f GiB free, need %.2f GiB" % (
            args.work_dir, free / 2**30, (out_size - have) / 2**30))

    state = State(out + ".state.json", plan_id, [t["name"] for t in plan])
    fd = os.open(partial, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        os.pwrite(fd, struct.pack("<Q", len(header)) + header, 0)
        if os.fstat(fd).st_size != out_size:
            os.ftruncate(fd, out_size)
        already = sum(min(v, t["size"]) for t, v in zip(plan, (state.done[t["name"]] for t in plan)))
        todo = [t for t in plan if state.done[t["name"]] < t["size"]]
        log("downloading %d tensors (%.2f GiB left), %d parallel" % (
            len(todo), (total - already) / 2**30, args.jobs))
        prog = Progress(total, already)
        # biggest first so the tail of the run stays parallel
        todo.sort(key=lambda t: -t["size"])
        with concurrent.futures.ThreadPoolExecutor(max(1, args.jobs)) as ex:
            futs = [ex.submit(download_tensor, http, fd, t, state, prog) for t in todo]
            for fu in concurrent.futures.as_completed(futs):
                fu.result()
        os.fsync(fd)
    finally:
        os.close(fd)
    missing = [t["name"] for t in plan if state.done[t["name"]] != t["size"]]
    if missing:
        raise SystemExit("incomplete: %d tensors not finished (rerun to resume)" % len(missing))

    sums = verify_output(partial, plan, sample=not args.no_sample_check)
    manifest = {
        "source_repo": args.repo, "source_revision": args.revision, "endpoint": args.endpoint,
        "file": os.path.basename(out), "file_size": out_size, "total_tensor_bytes": total,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "tensors": {t["name"]: {"dtype": t["dtype"], "shape": t["shape"], "src_file": t["src_file"],
                                "src_range": [t["src_start"], t["src_end"]], "sha256": sums[t["name"]]}
                    for t in plan},
    }
    man_path = os.path.splitext(out)[0] + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=1)
    os.replace(partial, out)
    os.remove(state.path)
    log("done: %s (%.3f GiB), manifest %s" % (out, out_size / 2**30, man_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
