#!/usr/bin/env python3
"""Build intel-autoround-mtp4: the Intel Qwen3.8-Flash-Next W4A16 AutoRound checkpoint with the
hampsonw INT4 MTP-expert delta (only the 512 routed experts of the MTP draft layer become INT4).

Standard library only. Nothing in the source directory is modified.

What it does
  1. Checks that SRC is Intel/Qwen3.8-Flash-Next-W4A16-AutoRound @ 861536dd (== main 4c67bf6 for every
     model file): file sizes of all 17 safetensors files, sha256 of model_extra_tensors.safetensors
     (the file the delta was derived from; --verify-base all hashes every shard, ~280 GB of reads).
  2. Downloads the pinned delta (hampsonw/...-MTP-INT4-Experts @ b8988275) into a cache dir with
     urllib (resumable), checks size + sha256 of every file.
  3. Validates the delta sidecar against SRC: the 29 non-expert MTP tensors must be byte-identical,
     the 4608 expert tensors must have the GPTQ layout of Intel's own routed experts.
  4. Derives config.json / model.safetensors.index.json from SRC's own files (removes the 1536
     per-expert "16 bit" exclusions, adds the INT4 entry; swaps the 1536 BF16 expert weights for
     4608 qweight/scales/qzeros entries) and checks the result against the delta's copies.
     The config gets the INT4 entry under BOTH "mtp.layers.0.mlp.experts" (our vLLM tree remaps
     the draft's extra_config keys 0 -> 48, see README) and "mtp.layers.48.mlp.experts"
     (hampsonw/Whamp runtime, no remap).
  5. Builds DST in a staging dir next to it: every unchanged file is HARDLINKED (byte-identical,
     same inode: the PLE-SSD loader reads the PLE shard by data_offsets), the three changed files
     are written/linked, then the staging dir is renamed to DST. Refuses to touch an existing DST.

usage (on the server, any python3 >= 3.8):
  python3 intel_mtp4_install.py --dry-run          # checks + small downloads + plan, builds nothing
  python3 intel_mtp4_install.py                    # full build
options: --src --dst --cache --base-url --verify-base {sizes,sidecar,all} --offline --keep-staging
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import http.client
import json
import os
import re
import shutil
import struct
import sys
import time
import urllib.error
import urllib.request

# --------------------------------------------------------------------------------------------------
# Pins (verified 2026-10-19 against the HF API; sizes/sha256 are the LFS values)
# --------------------------------------------------------------------------------------------------
DELTA_REPO = "hampsonw/Qwen3.8-Flash-Next-W4A16-AutoRound-MTP-INT4-Experts"
DELTA_REV = "b898827562533ebb0e94b1b0cde0acf8062acc6b"
DELTA_FILES = {  # name: (size, sha256)   (sha256s also listed in the repo's own SHA256SUMS)
    "model_extra_tensors.safetensors": (1489128680, "674e7102079286693b1fa5e5ddec5619575010f07fa1d5dab6c6ebcdfc2db2fa"),
    "config.json": (106641, "6aec450fe396a65008b95da8c18eba8ace894438069455ac954c3838f8be7925"),
    "model.safetensors.index.json": (24063625, "6bcb9d762450f33facb59de83a1d157a4f397a2cdb822971b8d08a493cb8df7f"),
    "DERIVATION.txt": (547, "9a02ba42d7bbf0f6e81ffedf6b94381cab797995d6206518fd03555bc46e6101"),
    "SHA256SUMS": (1286, "6df943c87c1ea6fe4714b7bd2405ec575aa590e54e7c554ece92fabc49536e90"),
    "README.md": (6104, "8f01373313999f126b09daf6eccd963b7c77d59ad21c1c2ca2c3cbd418deb5ef"),
}
BASE_REPO = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
BASE_REV = "861536dda5bcb208376fc4cd879b2bf76bece9fe"  # main 4c67bf68 has identical model files
BASE_FILES = {  # LFS files of the base revision: name: (size, sha256)
    "model-00001-of-00017.safetensors": (2899311784, "70734bddc4db7476d4ba03a86f808290e88488d39bf513c639ad3957768692de"),
    "model-00003-of-00017.safetensors": (5370548192, "0ecb3e52cad98e4c3c042432f3db2f12ca6c41ea6844a964f29e66f673fb5be5"),
    "model-00004-of-00017.safetensors": (5370910416, "2865e0756b9a3cc0e045eeec349d3e495d1630476cc5355ea0da6a272092d9b0"),
    "model-00005-of-00017.safetensors": (5370140704, "5f0b9642174a7bfbfa57c6fda7917effeb3afb23c2377cfc5f8f9e29222dd048"),
    "model-00006-of-00017.safetensors": (5370205184, "0d06de9de63a1cdabd39a6c11ea1b6431b24be17c4a51fe4db236ea77c33d0b9"),
    "model-00007-of-00017.safetensors": (5370927288, "a632e777190fac1ee19b8706925862af7b4f3ce44e58b40f01a79c0ab7697e96"),
    "model-00008-of-00017.safetensors": (5344445352, "cfc3279b2981d2b6c319b6be03d33091c0dea1b94d194705450b4cece4df6a7d"),
    "model-00009-of-00017.safetensors": (5370355648, "f3f8390dc32f3a2ea3980fa1a44cc37d02394c40388bd4688b16779431791327"),
    "model-00010-of-00017.safetensors": (5370927288, "48f9d03732bc89e177b946da67702e3eeec55d185a4c795dfebfecf380014d42"),
    "model-00011-of-00017.safetensors": (5324173744, "8592f8599a883e1a57f4010b3cfa7516e8f1c5777d21a62f9b0e20e06e56840c"),
    "model-00012-of-00017.safetensors": (5370438520, "6d58f6a6ac680e6cde7e061025a55cac650ec32f00a7ae1fb6aa95839971ba40"),
    "model-00013-of-00017.safetensors": (5370927288, "901b9e3f509df9a051d99ad5a9815339b68b4e78006805175e7c1f75423883c0"),
    "model-00014-of-00017.safetensors": (5342689856, "a77beb75499e376d0de54232d510f533eaf730149c4d9f69e5b327850b3da66b"),
    "model-00015-of-00017.safetensors": (5050474736, "f457d062094bda4c6bd486fe373ae56d04f2b205ef083e95cfdee9400da86d2c"),
    "model-00016-of-00017.safetensors": (102400512256, "59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf"),
    "model-00017-of-00017.safetensors": (1284526608, "fdbe0ca6b1bfab1dece276c48efadbc7d24ebb4c2619497c19db47b400f31782"),
    "model_extra_tensors.safetensors": (5214492152, "e9e4786a8ef584c9cb112b0a2ce7063cb72db8e002694888d2a36d440edad83d"),
}
# git blob sha1 of the base revision's small files (only used to report "pristine" vs "locally modified")
BASE_GIT_BLOBS = {
    "config.json": "52f62d8c1ac5d34f245f400810274c0b9170bec5",
    "quantization_config.json": "87676249a21f0a6c38d33783ab2bde77b7068482",
}
BASE_INDEX_SHA256 = "35e503b758cf5441886042d315c48dfba70f947611816cbb90e02d4d5c5e58ff"

SIDECAR = "model_extra_tensors.safetensors"
CONFIG = "config.json"
INDEX = "model.safetensors.index.json"
REPLACED = (CONFIG, INDEX, SIDECAR)
PROVENANCE = "MTP4_PROVENANCE.txt"

NUM_EXPERTS = 512
HIDDEN = 2560
MOE_INTER = 640
GROUP = 128
RUNTIME_MTP_LAYER = 48  # num_hidden_layers: vLLM builds the draft layer with prefix mtp.layers.48
INT4_ENTRY = {"bits": 4, "group_size": GROUP, "sym": True, "data_type": "int"}

EXPERT_EXCLUSION_RE = re.compile(r"^mtp\.layers\.(\d+)\.mlp\.experts\.\d+\.(?:gate|up|down)_proj$")
EXPERT_WEIGHT_RE = re.compile(r"^(mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.weight$")
EXPERT_ANY_RE = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.([a-z_]+)$")
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "I16": 2, "I8": 1, "U8": 1,
              "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
MiB = 1024 * 1024


def log(msg: str) -> None:
    print(msg, flush=True)


class BuildError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------------
def sha256_file(path: str, label: str = "") -> str:
    h = hashlib.sha256()
    size = os.path.getsize(path)
    done, t0, last = 0, time.time(), 0.0
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(16 * MiB)
            if not b:
                break
            h.update(b)
            done += len(b)
            if size > 2 * 1024 * MiB and time.time() - last > 15:
                last = time.time()
                log(f"    sha256 {label or os.path.basename(path)}: {done / size:5.1%} "
                    f"({done / max(time.time() - t0, 1e-3) / MiB:.0f} MiB/s)")
    return h.hexdigest()


def git_blob_sha1(path: str) -> str:
    with open(path, "rb") as f:
        b = f.read()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def read_st_header(path: str) -> tuple[dict, int]:
    """safetensors header -> (header dict incl. __metadata__, absolute offset of the data section)"""
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise BuildError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n > 100_000_000:
            raise BuildError(f"{path}: header length {n} is implausible")
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


def tensor_nbytes(info: dict) -> int:
    n = DTYPE_SIZE[info["dtype"]]
    for s in info["shape"]:
        n *= s
    return n


def check_st_layout(path: str, hdr: dict, data_start: int) -> None:
    """Offsets contiguous from 0, sizes consistent, file size == header + data (what safetensors checks)."""
    items = sorted(((v["data_offsets"], k, v) for k, v in hdr.items() if k != "__metadata__"),
                   key=lambda x: x[0])
    pos = 0
    for (b, e), k, v in items:
        if b != pos or e - b != tensor_nbytes(v):
            raise BuildError(f"{path}: bad data_offsets for {k}")
        pos = e
    if data_start + pos != os.path.getsize(path):
        raise BuildError(f"{path}: file size {os.path.getsize(path)} != header+data {data_start + pos}")


def same_bytes(fa: str, oa: int, fb: str, ob: int, n: int) -> bool:
    with open(fa, "rb") as a, open(fb, "rb") as b:
        a.seek(oa)
        b.seek(ob)
        left = n
        while left:
            k = min(left, 8 * MiB)
            if a.read(k) != b.read(k):
                return False
            left -= k
    return True


def json_dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


# --------------------------------------------------------------------------------------------------
# download (urllib, resumable, size + sha256 checked)
# --------------------------------------------------------------------------------------------------
def download(url: str, dest: str, size: int, sha: str, offline: bool = False, retries: int = 8) -> str:
    if os.path.exists(dest):
        if os.path.getsize(dest) == size and sha256_file(dest) == sha:
            return "cached"
        log(f"    {os.path.basename(dest)}: cached copy is wrong, downloading again")
        os.replace(dest, dest + ".bad")
    if offline:
        raise BuildError(f"--offline and {dest} is missing or wrong")
    part = dest + ".part"
    have = os.path.getsize(part) if os.path.exists(part) else 0
    if have > size:
        os.remove(part)
        have = 0
    attempt = 0
    while have < size:
        before = have
        headers = {"User-Agent": "mtp4-install/1.0 (python-urllib)"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                status = getattr(r, "status", 200)
                if have and status != 206:
                    log("    server ignored the Range request, restarting from 0")
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                    if have:
                        cr = r.headers.get("Content-Range", "")
                        m = re.match(r"bytes (\d+)-", cr)
                        if not m or int(m.group(1)) != have:
                            raise BuildError(f"unexpected Content-Range {cr!r} for resume at {have}")
                t0, last, start = time.time(), 0.0, have
                with open(part, mode) as f:
                    while True:
                        b = r.read(4 * MiB)
                        if not b:
                            break
                        f.write(b)
                        have += len(b)
                        if have > size:
                            raise BuildError(f"{url}: more than {size} bytes")
                        if size > 64 * MiB and time.time() - last > 10:
                            last = time.time()
                            rate = (have - start) / max(time.time() - t0, 1e-3) / MiB
                            log(f"    {os.path.basename(dest)}: {have / size:6.1%} ({rate:.1f} MiB/s)")
                    f.flush()
                    os.fsync(f.fileno())
            if have < size:
                if have == before:
                    attempt += 1
                    if attempt > retries:
                        raise BuildError(f"{url}: no progress after {retries} retries")
                    time.sleep(min(60, 2 ** attempt))
                log(f"    connection closed at byte {have} of {size}, resuming")
            else:
                attempt = 0
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise BuildError(f"{url}: HTTP {e.code}") from e
            attempt += 1
            if attempt > retries:
                raise BuildError(f"{url}: giving up after {retries} retries: {e}") from e
            have = os.path.getsize(part) if os.path.exists(part) else 0
            log(f"    download error ({e}); retry {attempt}/{retries} from byte {have}")
            time.sleep(min(60, 2 ** attempt))
    if os.path.getsize(part) != size:
        raise BuildError(f"{dest}: size {os.path.getsize(part)} != {size}")
    got = sha256_file(part, os.path.basename(dest))
    if got != sha:
        os.replace(part, dest + ".bad")
        raise BuildError(f"{dest}: sha256 {got} != pinned {sha} (moved to .bad)")
    os.replace(part, dest)
    return "downloaded"


# --------------------------------------------------------------------------------------------------
# config / index derivation
# --------------------------------------------------------------------------------------------------
def derive_config(src_cfg: dict, runtime_layer: int = RUNTIME_MTP_LAYER) -> tuple[dict, int, list[str]]:
    cfg = copy.deepcopy(src_cfg)
    qc = cfg.get("quantization_config")
    if not isinstance(qc, dict) or qc.get("quant_method") != "auto-round":
        raise BuildError("config.json has no auto-round quantization_config")
    ec = qc.get("extra_config")
    if not isinstance(ec, dict):
        raise BuildError("quantization_config has no extra_config")
    removed = [k for k in ec if EXPERT_EXCLUSION_RE.fullmatch(k)]
    ckpt_layers = sorted({EXPERT_EXCLUSION_RE.fullmatch(k).group(1) for k in removed})
    for k in removed:
        if ec[k].get("bits", 16) < 16:
            raise BuildError(f"extra_config[{k!r}] is already quantized: {ec[k]}")
        del ec[k]
    added = []
    # checkpoint index (remapped 0 -> 48 by our tree's mtp.py) and runtime index (Whamp runtime)
    for layer in ckpt_layers + [str(runtime_layer)]:
        key = f"mtp.layers.{layer}.mlp.experts"
        if key in ec and ec[key] != INT4_ENTRY:
            raise BuildError(f"extra_config already has {key} = {ec[key]}")
        ec[key] = dict(INT4_ENTRY)
        added.append(key)
    return cfg, len(removed), added


def derive_index(src_index: dict, src_sidecar_hdr: dict, delta_sidecar_hdr: dict) -> tuple[dict, int, int]:
    """Replace each BF16 MTP expert weight by its qweight/scales/qzeros (same order as hampsonw's tool)."""
    wm = src_index["weight_map"]
    new_wm = {}
    removed_bytes = added_bytes = replaced = 0
    for name, shard in wm.items():
        m = EXPERT_WEIGHT_RE.fullmatch(name)
        if m is None:
            new_wm[name] = shard
            continue
        if shard != SIDECAR:
            raise BuildError(f"{name} lives in {shard}, expected {SIDECAR}")
        removed_bytes += tensor_nbytes(src_sidecar_hdr[name])
        for suffix in ("qweight", "scales", "qzeros"):
            q = f"{m.group(1)}.{suffix}"
            if q not in delta_sidecar_hdr:
                raise BuildError(f"delta sidecar lacks {q}")
            new_wm[q] = SIDECAR
            added_bytes += tensor_nbytes(delta_sidecar_hdr[q])
        replaced += 1
    idx = copy.deepcopy(src_index)
    idx["weight_map"] = new_wm
    md = idx.setdefault("metadata", {})
    if "total_size" in md:
        md["total_size"] = int(md["total_size"]) - removed_bytes + added_bytes
    # every tensor of the new sidecar must be indexed, and nothing else may point at it
    side = {k for k in delta_sidecar_hdr if k != "__metadata__"}
    idx_side = {k for k, v in new_wm.items() if v == SIDECAR}
    if side != idx_side:
        raise BuildError(f"index/sidecar mismatch: {len(side ^ idx_side)} names differ, "
                         f"e.g. {sorted(side ^ idx_side)[:3]}")
    return idx, replaced, removed_bytes - added_bytes


# --------------------------------------------------------------------------------------------------
# faithful port of our tree's INC layer resolution for the draft model (verification only)
#   vllm/models/qwen4_exp/nvidia/mtp.py  _make_draft_vllm_config / _remap_ignored_layers
#   vllm/model_executor/layers/quantization/inc/config_parser.py  INCConfigParser._resolve_raw
#   vllm/model_executor/layers/quantization/inc/inc.py  INCConfig.get_quant_method (bits>=16 shortcut)
# --------------------------------------------------------------------------------------------------
MTP_PACKED_MODULES = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"],
}


def remap_mtp_keys(d: dict, start: int) -> dict:
    out = {}
    for name, v in d.items():
        if name.startswith("mtp."):
            name = re.sub(r"(?<=\.layers\.)\d+", lambda m: str(start + int(m.group(0))), name)
        out[name] = v
    return out


def inc_resolve(qc: dict, layer_name: str, is_moe: bool, remap: bool = True,
                start: int = RUNTIME_MTP_LAYER) -> tuple[int, int, bool]:
    wb, gs, sym = qc.get("bits", 4), qc.get("group_size", 128), qc.get("sym", True)
    ec = qc.get("extra_config") or {}
    if remap:
        ec = remap_mtp_keys(ec, start)
    bn = qc.get("block_name_to_quantize")
    if isinstance(bn, str):
        bn = bn.split(",")
    special = set(r"*+?^$()[]{}|\\")

    def explicit(name):
        if name in ec:
            return True
        for p in ec:
            if isinstance(p, str) and any(c in special for c in p):
                try:
                    if re.search(p, name):
                        return True
                except re.error:
                    pass
        return False

    def get_config(name, quantized=True):
        if not ec:
            return (wb if quantized else 16, gs if quantized else -1, sym if quantized else True)
        if name in ec:
            c = ec[name]
            return (c.get("bits", wb if quantized else 16), c.get("group_size", gs if quantized else -1),
                    c.get("sym", sym if quantized else True))
        for p, c in ec.items():
            if not isinstance(p, str) or not any(ch in special for ch in p):
                continue
            try:
                if re.search(p, name):
                    return (c.get("bits", wb if quantized else 16), c.get("group_size", gs if quantized else -1),
                            c.get("sym", sym if quantized else True))
            except re.error:
                continue
        return (wb if quantized else 16, gs if quantized else -1, sym if quantized else True)

    # INCConfig.get_quant_method shortcut: explicit >=16-bit entry for exactly this prefix
    for k in ec:
        if (k == layer_name or k == f"model.{layer_name}") and ec[k].get("bits", 16) >= 16:
            return (16, -1, True)
    if layer_name in ec:
        return get_config(layer_name)
    for k in ec:
        if layer_name.endswith(f".{k}"):
            return get_config(k)
    quantized = True
    if bn:
        quantized = any(layer_name.startswith(n) for n in bn)
    if is_moe:
        cfgs = [get_config(n, quantized) for n in ec if n.startswith(layer_name)]
        if cfgs:
            if len(set(cfgs)) == 1:
                return cfgs[0]
            raise BuildError(f"inconsistent MoE config for {layer_name}: {set(cfgs)}")
    for fk, subs in MTP_PACKED_MODULES.items():
        if fk in layer_name and layer_name.count(fk) == 1:
            names = [layer_name.replace(fk, s) for s in subs]
            if not any(explicit(n) for n in names):
                continue
            cfgs = [get_config(n, quantized) for n in names]
            if len(set(cfgs)) == 1:
                return cfgs[0]
            raise BuildError(f"inconsistent fused config for {layer_name}: {set(cfgs)}")
    return get_config(layer_name, quantized)


DRAFT_LAYERS = [  # (runtime prefix, is RoutedExperts)
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.mlp.experts", True),
    ("mtp.fc_embedding", False),
    ("mtp.fc_hidden", False),
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.self_attn.qkv_proj", False),
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.self_attn.o_proj", False),
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.self_attn.indexer.index_qk_proj", False),
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.mlp.shared_expert.gate_up_proj", False),
    (f"mtp.layers.{RUNTIME_MTP_LAYER}.mlp.shared_expert.down_proj", False),
]


def draft_resolution_table(cfg: dict, remap: bool = True) -> dict[str, tuple]:
    qc = cfg["quantization_config"]
    return {name: inc_resolve(qc, name, moe, remap=remap) for name, moe in DRAFT_LAYERS}


def check_draft_resolution(cfg: dict) -> None:
    tab = draft_resolution_table(cfg, remap=True)
    exp = f"mtp.layers.{RUNTIME_MTP_LAYER}.mlp.experts"
    for name, r in tab.items():
        want = (4, GROUP, True) if name == exp else 16
        ok = r == want if name == exp else r[0] >= 16
        log(f"    {'ok ' if ok else 'BAD'} {name:58s} -> bits={r[0]} group={r[1]} sym={r[2]}")
        if not ok:
            raise BuildError(f"derived config would resolve {name} to {r}")


# --------------------------------------------------------------------------------------------------
# delta sidecar validation
# --------------------------------------------------------------------------------------------------
def expected_delta_tensor(name: str):
    m = EXPERT_ANY_RE.fullmatch(name)
    if not m:
        return None
    proj, suffix = m.group(2), m.group(3)
    k, n = (MOE_INTER, HIDDEN) if proj == "down" else (HIDDEN, MOE_INTER)  # K = in, N = out
    return {"qweight": ("I32", [k // 8, n]), "scales": ("F16", [k // GROUP, n]),
            "qzeros": ("I32", [k // GROUP, n // 8])}.get(suffix, ("?", None))


def validate_delta_sidecar(src_path: str, delta_path: str) -> dict:
    sh, sd = read_st_header(src_path)
    dh, dd = read_st_header(delta_path)
    check_st_layout(delta_path, dh, dd)
    src_names = {k for k in sh if k != "__metadata__"}
    dst_names = {k for k in dh if k != "__metadata__"}
    src_experts = {k for k in src_names if EXPERT_WEIGHT_RE.fullmatch(k)}
    if len(src_experts) != 3 * NUM_EXPERTS:
        raise BuildError(f"source sidecar has {len(src_experts)} BF16 MTP expert weights, expected 1536 "
                         "(already converted?)")
    non_expert = src_names - src_experts
    if non_expert != {k for k in dst_names if not EXPERT_ANY_RE.fullmatch(k)}:
        raise BuildError("delta sidecar: non-expert tensor names differ from the source sidecar")
    n_exp = 0
    for k in dst_names:
        exp = expected_delta_tensor(k)
        if exp is None:
            continue
        n_exp += 1
        if (dh[k]["dtype"], dh[k]["shape"]) != exp:
            raise BuildError(f"delta {k}: {dh[k]['dtype']} {dh[k]['shape']}, expected {exp}")
    if n_exp != 9 * NUM_EXPERTS:
        raise BuildError(f"delta sidecar has {n_exp} expert tensors, expected {9 * NUM_EXPERTS}")
    nbytes = 0
    for k in sorted(non_expert):
        a, b = sh[k], dh[k]
        if (a["dtype"], a["shape"]) != (b["dtype"], b["shape"]):
            raise BuildError(f"delta {k}: dtype/shape differ from source")
        n = tensor_nbytes(a)
        if not same_bytes(src_path, sd + a["data_offsets"][0], delta_path, dd + b["data_offsets"][0], n):
            raise BuildError(f"delta {k}: bytes differ from the source sidecar")
        nbytes += n
    # symmetric GPTQ: every qzeros word is 0x77777777 (zero point 8, stored minus one); spot-check
    with open(delta_path, "rb") as f:
        for e in (0, 1, NUM_EXPERTS // 2, NUM_EXPERTS - 1):
            for p in ("gate", "up", "down"):
                info = dh[f"mtp.layers.0.mlp.experts.{e}.{p}_proj.qzeros"]
                f.seek(dd + info["data_offsets"][0])
                words = struct.unpack(f"<{tensor_nbytes(info) // 4}I", f.read(tensor_nbytes(info)))
                if set(words) != {0x77777777}:
                    raise BuildError(f"expert {e} {p}_proj qzeros are not the symmetric constant")
    return {"non_expert_tensors": len(non_expert), "non_expert_bytes_identical": nbytes,
            "expert_tensors": n_exp, "data_bytes": os.path.getsize(delta_path) - dd}


# --------------------------------------------------------------------------------------------------
# hardlink tree
# --------------------------------------------------------------------------------------------------
def link_tree(src: str, dst: str, skip_top: set[str]) -> list[tuple[str, str]]:
    """Mirror src into dst with hardlinks (symlinks are re-created). Returns [(relpath, kind)]."""
    done = []
    src = os.path.abspath(src)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel_root = os.path.relpath(root, src)
        out_root = dst if rel_root == "." else os.path.join(dst, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for name in sorted(dirs + files):
            rel = name if rel_root == "." else os.path.join(rel_root, name)
            if rel_root == "." and name in skip_top:
                continue
            sp, dp = os.path.join(root, name), os.path.join(out_root, name)
            if os.path.islink(sp):
                target = os.readlink(sp)
                if not os.path.isabs(target):
                    resolved = os.path.normpath(os.path.join(root, target))
                    if not (resolved + os.sep).startswith(src + os.sep):
                        target = resolved  # points outside SRC: make it absolute
                os.symlink(target, dp)
                if name in dirs:
                    dirs.remove(name)  # never descend into symlinked dirs
                done.append((rel, "symlink"))
            elif os.path.isdir(sp):
                continue  # created when os.walk descends
            elif os.path.isfile(sp):
                os.link(sp, dp)
                done.append((rel, "hardlink"))
            else:
                log(f"    skipping special file {rel}")
    return done


# --------------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------------
VLLM_INC_CHECK = r"""
import json, re, sys
from vllm.model_executor.layers.quantization.inc.inc import INCConfig
cfg = json.load(open(sys.argv[1]))
names = json.loads(sys.argv[2])
qc = INCConfig.from_config(cfg["quantization_config"])
qc.packed_modules_mapping = json.loads(sys.argv[3])
# what vllm/models/qwen4_exp/nvidia/mtp.py _make_draft_vllm_config does to the draft's extra_config
qc.extra_config = {(re.sub(r"(?<=\.layers\.)\d+", lambda m: str(int(sys.argv[4]) + int(m.group(0))), k)
                    if k.startswith("mtp.") else k): v for k, v in qc.extra_config.items()}
class FusedMoE: pass
class ColumnParallelLinear: pass
out = {}
for name, moe in names:
    c = qc.config_parser.resolve(FusedMoE() if moe else ColumnParallelLinear(), name)
    out[name] = [c.bits, c.group_size, c.sym]
print("RESULT " + json.dumps(out))
"""


def vllm_inc_check(python: str, cfg_path: str) -> dict:
    """Resolve the draft layers with the INSTALLED vLLM's INCConfigParser (subprocess in the vLLM venv)."""
    import subprocess
    r = subprocess.run([python, "-c", VLLM_INC_CHECK, cfg_path, json.dumps(DRAFT_LAYERS),
                        json.dumps(MTP_PACKED_MODULES), str(RUNTIME_MTP_LAYER)],
                       capture_output=True, text=True, timeout=600)
    line = next((x for x in r.stdout.splitlines() if x.startswith("RESULT ")), None)
    if r.returncode or line is None:
        raise BuildError(f"vLLM check failed (rc {r.returncode}): {(r.stderr or r.stdout).strip()[-800:]}")
    return {k: tuple(v) for k, v in json.loads(line[7:]).items()}


def build(args) -> int:
    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    cache = os.path.abspath(args.cache)
    parent = os.path.dirname(dst)
    staging = os.path.join(parent, f".{os.path.basename(dst)}.staging-{os.getpid()}")
    url = lambda f: f"{args.base_url.rstrip('/')}/{DELTA_REPO}/resolve/{DELTA_REV}/{f}"  # noqa: E731

    log(f"== intel MTP INT4 install\n   src  {src}\n   dst  {dst}\n   cache {cache}\n   delta {DELTA_REPO}@{DELTA_REV[:8]}")
    if os.path.lexists(dst):
        raise BuildError(f"{dst} exists - refusing to overwrite (remove it yourself or pick another --dst)")
    for f in REPLACED:
        if not os.path.isfile(os.path.join(src, f)):
            raise BuildError(f"{src}/{f} missing")
    if not os.path.isdir(parent):
        raise BuildError(f"{parent} does not exist")
    if os.stat(src).st_dev != os.stat(parent).st_dev:
        raise BuildError(f"{src} and {parent} are on different filesystems: hardlinks impossible")

    # 1. source identity ------------------------------------------------------------------------
    log("-- 1. source checkpoint vs Intel@861536dd (sizes; sha256 per --verify-base)")
    bad = []
    for name, (size, sha) in BASE_FILES.items():
        p = os.path.join(src, name)
        if name == INDEX:
            continue
        if not os.path.exists(p):
            bad.append(f"{name} missing")
        elif os.path.getsize(p) != size:
            bad.append(f"{name} size {os.path.getsize(p)} != {size}")
    if bad:
        raise BuildError("source is not the pinned Intel revision: " + "; ".join(bad))
    log(f"    all {len(BASE_FILES)} safetensors files have the pinned sizes")
    to_hash = [] if args.verify_base == "sizes" else [SIDECAR] if args.verify_base == "sidecar" else list(BASE_FILES)
    for name in to_hash:
        if name == INDEX:
            continue
        got = sha256_file(os.path.join(src, name))
        if got != BASE_FILES[name][1]:
            raise BuildError(f"{name}: sha256 {got} != pinned {BASE_FILES[name][1]}")
        log(f"    sha256 ok  {name}")
    pristine = {
        CONFIG: git_blob_sha1(os.path.join(src, CONFIG)) == BASE_GIT_BLOBS[CONFIG],
        INDEX: sha256_file(os.path.join(src, INDEX)) == BASE_INDEX_SHA256,
    }
    for k, v in pristine.items():
        log(f"    {k}: {'identical to Intel@861536dd' if v else 'LOCALLY MODIFIED (derivation still uses it)'}")

    # 2. delta download -------------------------------------------------------------------------
    log("-- 2. delta files (urllib, size + sha256 pinned)")
    os.makedirs(cache, exist_ok=True)
    names = [n for n in DELTA_FILES if not (args.dry_run and n == SIDECAR)]
    for n in names:
        size, sha = DELTA_FILES[n]
        st = download(url(n), os.path.join(cache, n), size, sha, offline=args.offline)
        log(f"    {st:10s} {n} ({size} B)")
    # cross-check the pins with the repo's own SHA256SUMS
    sums = {}
    for line in open(os.path.join(cache, "SHA256SUMS")):
        h, _, fn = line.strip().partition("  ")
        sums[fn.lstrip("./")] = h
    for n, (_, sha) in DELTA_FILES.items():
        if n in sums and sums[n] != sha:
            raise BuildError(f"SHA256SUMS disagrees with pin for {n}")

    # 3. config / index derivation + checks -------------------------------------------------------
    log("-- 3. derive config.json and the index from the source")
    src_cfg = json.load(open(os.path.join(src, CONFIG)))
    new_cfg, removed, added = derive_config(src_cfg)
    log(f"    extra_config: removed {removed} per-expert 16-bit exclusions, added {added} = {INT4_ENTRY}")
    if removed != 3 * NUM_EXPERTS:
        raise BuildError(f"expected 1536 MTP expert exclusions, found {removed}")
    delta_cfg = json.load(open(os.path.join(cache, CONFIG)))
    expect = copy.deepcopy(delta_cfg)
    expect["quantization_config"]["extra_config"]["mtp.layers.0.mlp.experts"] = dict(INT4_ENTRY)
    if new_cfg == expect:
        log("    derived config == hampsonw config.json + the extra 'mtp.layers.0.mlp.experts' key")
    elif pristine[CONFIG]:
        raise BuildError("derived config differs from hampsonw's although the source config is pristine")
    else:
        log("    NOTE: derived config differs from hampsonw's (source config is locally modified)")
    log("    draft-model resolution in OUR tree (extra_config keys remapped 0->48 by mtp.py):")
    check_draft_resolution(new_cfg)
    exp_name = f"mtp.layers.{RUNTIME_MTP_LAYER}.mlp.experts"
    if args.vllm_python:
        tmp = os.path.join(cache, "derived-config.json")
        json_dump(new_cfg, tmp)
        got = vllm_inc_check(args.vllm_python, tmp)
        want = draft_resolution_table(new_cfg, remap=True)
        # group size is meaningless for 16-bit (unquantized) layers; INCConfigParser fills in its default 128
        norm = lambda t: {k: (b, g if b < 16 else None, sy) for k, (b, g, sy) in t.items()}
        if norm(got) != norm(want):
            raise BuildError(f"installed vLLM resolves the draft differently: {got} vs {want}")
        log(f"    installed vLLM ({args.vllm_python}) INCConfigParser agrees: "
            f"{exp_name} -> {got[exp_name]}, all other draft layers 16 bit")
    hw_tab = draft_resolution_table(delta_cfg, remap=True)
    log(f"    (hampsonw's config.json unchanged would give {exp_name} -> bits={hw_tab[exp_name][0]} in our tree"
        f"{' = BF16 MoE, load failure on qweight' if hw_tab[exp_name][0] >= 16 else ''})")

    cached_side = os.path.join(cache, SIDECAR)
    if args.dry_run and not os.path.isfile(cached_side):
        if args.offline:
            raise BuildError(f"--offline --dry-run needs {cached_side}")
        log("    (dry run: sidecar not downloaded; reading only its header over HTTP)")
        delta_hdr = json.loads(_fetch_header(url(SIDECAR)))
    else:
        delta_hdr, _ = read_st_header(cached_side)
    src_side_hdr, _ = read_st_header(os.path.join(src, SIDECAR))
    src_index = json.load(open(os.path.join(src, INDEX)))
    new_index, replaced, saved = derive_index(src_index, src_side_hdr, delta_hdr)
    log(f"    index: {replaced} BF16 expert weights -> {3 * replaced} GPTQ tensors; total_size -{saved} B")
    delta_index = json.load(open(os.path.join(cache, INDEX)))
    if new_index == delta_index:
        log("    derived index == hampsonw portable index (exactly)")
    elif pristine[INDEX]:
        raise BuildError("derived index differs from hampsonw's although the source index is pristine")
    else:
        log("    NOTE: derived index differs from hampsonw's (source index is locally modified); using derived")
    ple = [k for k in new_index["weight_map"] if ".ngram_embedding.shard_" in k]
    log(f"    index keeps {len(ple)} PLE shard entries (PLE-SSD resolves them through the index)")
    for shard in sorted(set(new_index["weight_map"].values())):
        if shard != SIDECAR and not os.path.exists(os.path.join(src, shard)):
            raise BuildError(f"index references {shard}, missing in {src}")

    if args.dry_run:
        log("== dry run OK: nothing built. Run again without --dry-run to download the 1.49 GB sidecar and build.")
        return 0

    # 4. validate the downloaded sidecar ---------------------------------------------------------
    log("-- 4. validate the delta sidecar against the source sidecar")
    v = validate_delta_sidecar(os.path.join(src, SIDECAR), os.path.join(cache, SIDECAR))
    log(f"    {v['non_expert_tensors']} non-expert MTP tensors byte-identical ({v['non_expert_bytes_identical']} B), "
        f"{v['expert_tensors']} expert tensors in Intel's GPTQ layout, qzeros = 0x77777777 (sym)")

    # 5. build ----------------------------------------------------------------------------------
    log(f"-- 5. build {staging}")
    if os.path.lexists(staging):
        raise BuildError(f"{staging} exists")
    os.makedirs(staging)
    try:
        linked = link_tree(src, staging, set(REPLACED))
        json_dump(new_cfg, os.path.join(staging, CONFIG))
        json_dump(new_index, os.path.join(staging, INDEX))
        side_src, side_dst = os.path.join(cache, SIDECAR), os.path.join(staging, SIDECAR)
        if os.stat(cache).st_dev == os.stat(staging).st_dev:
            os.link(side_src, side_dst)
            how = "hardlink of cache"
        else:
            shutil.copyfile(side_src, side_dst)
            how = "copy of cache"
        with open(os.path.join(staging, PROVENANCE), "w") as f:
            f.write(
                f"built={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by intel_mtp4_install.py\n"
                f"source={src} (Intel {BASE_REPO}@{BASE_REV}; sidecar sha256 {BASE_FILES[SIDECAR][1]})\n"
                f"delta={DELTA_REPO}@{DELTA_REV}\n"
                f"{SIDECAR} sha256={DELTA_FILES[SIDECAR][1]} ({how})\n"
                f"{CONFIG}: source minus {removed} per-expert exclusions, plus {added} = {json.dumps(INT4_ENTRY)}\n"
                f"{INDEX}: source with {replaced} BF16 MTP expert weights replaced by qweight/scales/qzeros\n"
                f"all other files: hardlinks of {src} (same inode)\n")
        # 6. verify ----------------------------------------------------------------------------
        log("-- 6. verify the staging dir")
        n_hl = 0
        for rel, kind in linked:
            if kind != "hardlink":
                continue
            a, b = os.stat(os.path.join(src, rel)), os.stat(os.path.join(staging, rel))
            if (a.st_dev, a.st_ino) != (b.st_dev, b.st_ino):
                raise BuildError(f"{rel} is not a hardlink of the source")
            n_hl += 1
        for shard in set(new_index["weight_map"].values()):
            if not os.path.isfile(os.path.join(staging, shard)):
                raise BuildError(f"index references missing {shard}")
        if sha256_file(side_dst) != DELTA_FILES[SIDECAR][1]:
            raise BuildError("sidecar in staging has the wrong sha256")
        for name in BASE_FILES:
            if name in REPLACED:
                continue
            a, b = os.stat(os.path.join(src, name)), os.stat(os.path.join(staging, name))
            if a.st_ino != b.st_ino:
                raise BuildError(f"{name} not shared with the source")
        ple_files = sorted({new_index["weight_map"][k] for k in ple})
        for f in ple_files:
            if os.stat(os.path.join(src, f)).st_ino != os.stat(os.path.join(staging, f)).st_ino:
                raise BuildError(f"PLE shard {f} is not shared with the source")
        log(f"    {n_hl} hardlinks share the source inode (PLE shard(s) {', '.join(ple_files)} included), "
            f"{sum(1 for _, k in linked if k == 'symlink')} symlinks re-created")
        os.rename(staging, dst)
    except BaseException:
        if args.keep_staging:
            log(f"!! build failed, staging kept at {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    old = BASE_FILES[SIDECAR][0]
    new = DELTA_FILES[SIDECAR][0]
    log(f"== done: {dst}\n   MTP sidecar {old / 1e9:.2f} GB -> {new / 1e9:.2f} GB "
        f"(MTP weights in VRAM -{saved / 2**30:.2f} GiB on the last PP stage)\n"
        f"   new disk usage: {(new + os.path.getsize(os.path.join(dst, INDEX)) + os.path.getsize(os.path.join(dst, CONFIG))) / 1e9:.2f} GB "
        f"(everything else is hardlinked)")
    return 0


def _fetch_header(u: str) -> bytes:
    req = urllib.request.Request(u, headers={"User-Agent": "mtp4-install/1.0", "Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=60) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    req = urllib.request.Request(u, headers={"User-Agent": "mtp4-install/1.0", "Range": f"bytes=8-{7 + n}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    if len(b) != n:
        raise BuildError("short header read")
    return b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/opt/vllm-qfn/models/intel-autoround")
    ap.add_argument("--dst", default="/opt/vllm-qfn/models/intel-autoround-mtp4")
    ap.add_argument("--cache", default="/opt/vllm-qfn/models/.mtp4-cache/hampsonw-" + DELTA_REV[:8],
                    help="download dir for the delta (kept for re-runs; same filesystem -> sidecar is hardlinked)")
    ap.add_argument("--base-url", default="https://huggingface.co")
    ap.add_argument("--verify-base", choices=("sizes", "sidecar", "all"), default="sidecar",
                    help="sizes only / + sha256 of model_extra_tensors (default) / + sha256 of every shard")
    ap.add_argument("--offline", action="store_true", help="use only files already in --cache")
    ap.add_argument("--dry-run", action="store_true", help="all checks + small downloads, no sidecar, no build")
    ap.add_argument("--keep-staging", action="store_true", help="keep the staging dir if the build fails")
    ap.add_argument("--vllm-python", metavar="PY",
                    help="python of the vLLM venv: also resolve the draft layers with the installed INCConfigParser")
    args = ap.parse_args(argv)
    try:
        return build(args)
    except BuildError as e:
        log(f"!! {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
