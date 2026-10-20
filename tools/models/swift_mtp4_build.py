#!/usr/bin/env python3
"""Build swift15-awq-mtp4: Swift 1.5 W4A16 AWQ (compressed-tensors) with the MTP draft layer's 512 routed
experts quantized to INT4 in exactly the format of Swift's own routed experts.

Format (read from config.json and checked against a routed expert of the main model):
  compressed-tensors "pack-quantized", W4A16_ASYM, group 128 along the input dim, i.e. per expert/proj:
    weight_packed      I32  [N, K/8]     code+8 (0..15), element j of each 8 at bits 4j..4j+3
    weight_scale       BF16 [N, K/128]
    weight_zero_point  I32  [N/8, K/128] zp+8, packed along the OUTPUT dim (packed_dim=0)   (asym only)
    weight_shape       I64  [2]          [N, K]
  dequant: w = (code - zp) * scale          (N = out, K = in; gate/up: 640x2560, down: 2560x640)
Quantization: RTN with min/max per group (the "memoryless_minmax" observer Swift's recipe names; zero always
representable, as compressed-tensors does), optional per-group MSE clip search (--clip-grid N).
Source: mtp.layers.0.mlp.experts.gate_up_proj [512, 1280, 2560] (gate = rows 0..639) and down_proj
[512, 2560, 640] in model_mtp.safetensors (also accepts per-expert ...experts.E.{gate,up,down}_proj.weight).

Output dir (built in a staging dir, renamed at the end; refuses to overwrite):
  every file of SRC hardlinked except config.json, model.safetensors.index.json and the old MTP shard
  + model_mtp_int4.safetensors   all mtp.* tensors: the non-expert ones byte-identical, experts INT4
  + model.safetensors.index.json mtp.* -> new shard, everything else unchanged
  + config.json                  ignore "re:^mtp.*"  ->  "re:^mtp\\.(?!layers\\.\\d+\\.mlp\\.experts\\.).*"
                                 (everything in the MTP except its routed experts stays BF16)

usage (server, vLLM venv python; CPU only, < 2 GB RAM):
  python swift_mtp4_build.py --selftest                 # torch vs reference implementation, no checkpoint
  python swift_mtp4_build.py --dry-run                  # plan + config/index checks, writes nothing
  nice -n 10 python swift_mtp4_build.py [--threads 8] [--clip-grid 1]
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import re
import shutil
import struct
import sys
import time

OLD_IGNORE = "re:^mtp.*"
NEW_IGNORE = r"re:^mtp\.(?!layers\.\d+\.mlp\.experts\.).*"
CONFIG = "config.json"
INDEX = "model.safetensors.index.json"
DEFAULT_SHARD = "model_mtp_int4.safetensors"
PROVENANCE = "MTP4_PROVENANCE.txt"
DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I32": 4, "I64": 8, "I16": 2, "I8": 1, "U8": 1,
              "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "U16": 2, "U32": 4, "U64": 8}
PROJS = ("gate_proj", "up_proj", "down_proj")
# Qwen4ExpMTP.packed_modules_mapping (vllm/models/qwen4_exp/nvidia/mtp.py)
MTP_PACKED = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"],
}
MiB = 1024 * 1024


def log(msg: str) -> None:
    print(msg, flush=True)


class BuildError(RuntimeError):
    pass


# ==================================================================================================
# safetensors helpers (pure Python)
# ==================================================================================================
def read_st_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise BuildError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n > 100_000_000:
            raise BuildError(f"{path}: implausible header length {n}")
        return json.loads(f.read(n)), 8 + n


def nbytes_of(dtype: str, shape) -> int:
    n = DTYPE_SIZE[dtype]
    for s in shape:
        n *= s
    return n


def check_st_layout(path: str) -> dict:
    hdr, start = read_st_header(path)
    pos = 0
    for (b, e), k, v in sorted(((v["data_offsets"], k, v) for k, v in hdr.items() if k != "__metadata__"),
                               key=lambda x: x[0]):
        if b != pos or e - b != nbytes_of(v["dtype"], v["shape"]):
            raise BuildError(f"{path}: bad data_offsets for {k}")
        if (start + b) % DTYPE_SIZE[v["dtype"]]:
            raise BuildError(f"{path}: {k} is not aligned")
        pos = e
    if start + pos != os.path.getsize(path):
        raise BuildError(f"{path}: size {os.path.getsize(path)} != header + data {start + pos}")
    return hdr


def pread(path: str, offset: int, n: int) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        out = bytearray()
        while len(out) < n:
            b = os.pread(fd, n - len(out), offset + len(out))
            if not b:
                raise BuildError(f"{path}: short read at {offset + len(out)}")
            out += b
        return bytes(out)
    finally:
        os.close(fd)


class Entry:
    __slots__ = ("name", "dtype", "shape", "nbytes", "offset")

    def __init__(self, name, dtype, shape):
        self.name, self.dtype, self.shape = name, dtype, list(shape)
        self.nbytes = nbytes_of(dtype, shape)
        self.offset = None


def layout(entries: list[Entry], metadata: dict | None = None) -> tuple[bytes, int]:
    """Assign offsets (largest element size first, like the safetensors writer -> every tensor aligned),
    return (8-byte length + JSON header padded to a multiple of 8, total data bytes)."""
    order = sorted(range(len(entries)), key=lambda i: -DTYPE_SIZE[entries[i].dtype])
    off = 0
    for i in order:
        entries[i].offset = off
        off += entries[i].nbytes
    hdr = {"__metadata__": metadata} if metadata else {}
    for i in order:
        e = entries[i]
        hdr[e.name] = {"dtype": e.dtype, "shape": e.shape, "data_offsets": [e.offset, e.offset + e.nbytes]}
    js = json.dumps(hdr, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 8)
    return struct.pack("<Q", len(js)) + js, off


# ==================================================================================================
# vLLM matching logic, ported (compressed-tensors ignore / targets) - used for verification
# ==================================================================================================
def ct_match(value: str, target: str) -> bool:  # config_utils.is_equal_or_regex_match
    if target.startswith("re:"):
        return re.match(target[3:], value) is not None
    return target == value


def ct_ignored(layer_name: str, ignore: list[str], fused: dict = MTP_PACKED) -> bool:
    """compressed_tensors/utils.should_ignore_layer + config_utils.find_matching_patterns"""
    if any(ct_match(layer_name, p) for p in ignore):
        return True
    proj = layer_name.split(".")[-1]
    if proj not in fused:
        return False
    per = [any(ct_match(layer_name.replace(proj, s), p) for p in ignore) for s in fused[proj]]
    if any(per) and not all(per):
        raise BuildError(f"{layer_name}: shards would get different schemes")
    return all(per)


def ct_target_matches(layer_name: str, class_name: str, targets: list[str]) -> bool:
    """find_matched_target, incl. _add_fused_moe_to_target_scheme_map ('Linear' also targets RoutedExperts)"""
    ts = list(targets)
    if "Linear" in ts and "RoutedExperts" not in ts:
        ts.append("RoutedExperts")
    if any(ct_match(layer_name, t) for t in ts):
        return True
    fused = next((k for k in MTP_PACKED if layer_name.endswith(k)), None)
    if fused and all(any(ct_match(layer_name.replace(fused, u), t) for t in ts) for u in MTP_PACKED[fused]):
        return True
    return any(t.lower() in class_name.lower() for t in ts)


def mtp_module_names(layer: int) -> tuple[list[tuple[str, str]], str]:
    """(quantizable linear prefixes of the draft, class name), routed experts prefix"""
    p = f"mtp.layers.{layer}"
    lin = [("mtp.fc_embedding", "ColumnParallelLinear"), ("mtp.fc_hidden", "ColumnParallelLinear"),
           (f"{p}.self_attn.qkv_proj", "QKVParallelLinear"), (f"{p}.self_attn.o_proj", "RowParallelLinear"),
           (f"{p}.self_attn.indexer.index_qk_proj", "ReplicatedLinear"),
           (f"{p}.mlp.shared_expert.gate_up_proj", "MergedColumnParallelLinear"),
           (f"{p}.mlp.shared_expert.down_proj", "RowParallelLinear"),
           # built with quant_config=None today; must stay ignored anyway
           (f"{p}.mlp.gate", "ReplicatedLinear"), (f"{p}.mlp.shared_expert_gate", "ReplicatedLinear"),
           (f"{p}.attn_hyper_connection.input_mix_weight_down_block_inject", "MergedColumnParallelLinear"),
           (f"{p}.attn_hyper_connection.input_mix_weight_up", "ReplicatedLinear"),
           (f"{p}.mlp_hyper_connection.input_mix_weight_down_block_inject", "MergedColumnParallelLinear"),
           (f"{p}.mlp_hyper_connection.input_mix_weight_up", "ReplicatedLinear"),
           ("mtp.hyper_connection_mixer.input_mix_weight_down_block_inject", "MergedColumnParallelLinear"),
           ("mtp.hyper_connection_mixer.input_mix_weight_up", "ReplicatedLinear")]
    return lin, f"{p}.mlp.experts"


def ct_draft_table(qc: dict, layer: int) -> list[tuple[str, bool]]:
    """[(name, quantized?)] for the draft model's modules under this quantization_config."""
    ignore = qc.get("ignore") or []
    targets = [t for g in qc["config_groups"].values() for t in g.get("targets", [])]
    lin, experts = mtp_module_names(layer)
    out = []
    for name, cls in lin:
        out.append((name, (not ct_ignored(name, ignore)) and ct_target_matches(name, cls, targets)))
    # CompressedTensorsMoEMethod.get_moe_method checks the unfused names of expert 0
    q = [(not ct_ignored(experts + s, ignore)) and ct_target_matches(experts + s, "RoutedExperts", targets)
         for s in (".0.gate_proj", ".0.up_proj", ".0.down_proj")]
    if len(set(q)) != 1:
        raise BuildError("MoE projections would get different schemes")
    out.append((experts, q[0]))
    return out


def derive_ignore(ignore: list[str], runtime_layer: int, ckpt_layer: int) -> tuple[list[str], list[str]]:
    expert_names = [f"mtp.layers.{L}.mlp.experts.0.{p}" for L in (runtime_layer, ckpt_layer) for p in PROJS]
    new, changes = [], []
    for p in ignore:
        if not any(ct_match(n, p) for n in expert_names):
            new.append(p)
        elif p == OLD_IGNORE:
            new.append(NEW_IGNORE)
            changes.append(f"{p!r} -> {NEW_IGNORE!r}")
        elif not p.startswith("re:"):
            changes.append(f"removed {p!r}")
        else:
            raise BuildError(f"ignore entry {p!r} matches the MTP experts; don't know how to rewrite it")
    if not changes:
        raise BuildError("no ignore entry covers the MTP experts (already converted?)")
    return new, changes


# ==================================================================================================
# quantization backends: identical math, torch (server) and pure Python (reference/tests)
#   asym: lo=min(min,0)*r, hi=max(max,0)*r; s=bf16((hi-lo)/15) (0 -> eps); zp=clamp(round(-8-lo/s),-8,7)
#         q=clamp(round(w/s)+zp,-8,7)           sym: s=bf16(max(-lo,hi)/7.5), zp=0, q=clamp(round(w/s),-8,7)
#   all arithmetic in fp32 (each op correctly rounded), round = half-to-even; r from --clip-grid
# ==================================================================================================
def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bf16_round(x: float) -> float:
    b = struct.unpack("<I", struct.pack("<f", x))[0]
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", b))[0]


def clip_ratios(n: int) -> list[float]:
    if n <= 1:
        return [1.0]
    return [f32(1.0 - i * 0.30 / (n - 1)) for i in range(n)]


SCALE_EPS = {"BF16": 2.0 ** -7, "F16": 2.0 ** -10, "F32": 2.0 ** -23}  # torch.finfo(dtype).eps


class RefBackend:
    """Pure-Python reference. Slow: tests and --selftest only."""
    name = "python"

    def __init__(self, bits=4, group=128, symmetric=False, scale_dtype="BF16", clip_grid=1):
        if bits != 4:
            raise BuildError("only 4-bit is implemented")
        self.group, self.sym, self.sdt = group, symmetric, scale_dtype
        self.ratios = clip_ratios(clip_grid)

    @staticmethod
    def decode(raw: bytes, dtype: str) -> list[float]:
        if dtype == "BF16":
            hs = struct.unpack(f"<{len(raw) // 2}H", raw)
            return [struct.unpack("<f", struct.pack("<I", h << 16))[0] for h in hs]
        if dtype == "F16":
            return list(struct.unpack(f"<{len(raw) // 2}e", raw))
        if dtype == "F32":
            return list(struct.unpack(f"<{len(raw) // 4}f", raw))
        raise BuildError(f"unsupported dtype {dtype}")

    def _scale_store(self, x: float) -> float:
        if self.sdt == "BF16":
            return bf16_round(x)
        if self.sdt == "F16":
            return struct.unpack("<e", struct.pack("<e", x))[0]
        return x

    def _group(self, vals):
        mn, mx = min(min(vals), 0.0), max(max(vals), 0.0)
        best = None
        for r in self.ratios:
            lo, hi = (mn, mx) if r == 1.0 else (f32(mn * r), f32(mx * r))
            if self.sym:
                s = self._scale_store(f32(max(-lo, hi) / 7.5))
                s = s or SCALE_EPS[self.sdt]
                zp = 0
            else:
                s = self._scale_store(f32(f32(hi - lo) / 15.0))
                s = s or SCALE_EPS[self.sdt]
                zp = min(7, max(-8, round(f32(-8.0 - f32(lo / s)))))
            q = [min(7, max(-8, round(f32(v / s)) + zp)) for v in vals]
            if len(self.ratios) == 1:
                return q, s, zp
            err = 0.0
            for qi, v in zip(q, vals):
                d = f32(f32((qi - zp) * s) - v)
                err += d * d
            if best is None or err < best[3]:
                best = (q, s, zp, err)
        return best[:3]

    def quantize(self, raw: bytes, dtype: str, N: int, K: int):
        w = self.decode(raw, dtype)
        G, gs = K // self.group, self.group
        codes = [0] * (N * K)
        scales, zps = [0.0] * (N * G), [0] * (N * G)
        sse = ssw = 0.0
        for n in range(N):
            for g in range(G):
                vals = w[n * K + g * gs: n * K + (g + 1) * gs]
                q, s, zp = self._group(vals)
                scales[n * G + g], zps[n * G + g] = s, zp
                for j, (qi, v) in enumerate(zip(q, vals)):
                    codes[n * K + g * gs + j] = qi + 8
                    d = (qi - zp) * s - v
                    sse += d * d
                    ssw += v * v
        packed = ref_pack_rows(codes, N, K)
        if self.sdt == "BF16":
            sb = struct.pack(f"<{len(scales)}H", *[struct.unpack("<I", struct.pack("<f", s))[0] >> 16 for s in scales])
        elif self.sdt == "F16":
            sb = struct.pack(f"<{len(scales)}e", *scales)
        else:
            sb = struct.pack(f"<{len(scales)}f", *scales)
        out = {"weight_packed": packed, "weight_scale": sb, "weight_shape": struct.pack("<qq", N, K)}
        if not self.sym:
            uz = [z + 8 for z in zps]  # [N, G] -> transpose -> pack rows -> transpose back
            t = [uz[n * G + g] for g in range(G) for n in range(N)]
            pz = struct.unpack(f"<{G * N // 8}i", ref_pack_rows(t, G, N))
            out["weight_zero_point"] = struct.pack(f"<{G * N // 8}i",
                                                   *[pz[g * (N // 8) + r] for r in range(N // 8) for g in range(G)])
        return out, sse, ssw


def ref_pack_rows(u: list[int], R: int, C: int) -> bytes:
    """u: R*C values in 0..15 (row-major) -> int32 [R, C/8], element j of each 8 at bits 4j (LE)."""
    if C % 8:
        raise BuildError("pack: columns not a multiple of 8")
    words = []
    for i in range(0, R * C, 8):
        v = 0
        for j in range(8):
            x = u[i + j]
            if not 0 <= x <= 15:
                raise BuildError("pack: value out of range")
            v |= x << (4 * j)
        words.append(v - (1 << 32) if v >= 1 << 31 else v)
    return struct.pack(f"<{len(words)}i", *words)


def ref_unpack_rows(b: bytes, R: int, C: int) -> list[int]:
    words = struct.unpack(f"<{R * C // 8}I", b)
    return [(w >> (4 * j)) & 0xF for w in words for j in range(8)]


def ref_dequant(packed: bytes, scale: bytes, zp: bytes | None, N: int, K: int, group: int, sdt: str) -> list[float]:
    """Independent decoder (used by the self-check of the python backend and the tests)."""
    G = K // group
    u = ref_unpack_rows(packed, N, K)
    s = RefBackend.decode(scale, sdt)
    if zp is None:
        z = [8] * (N * G)
    else:
        words = struct.unpack(f"<{N // 8 * G}I", zp)  # [N/8, G]
        z = [0] * (N * G)
        for r in range(N // 8):
            for g in range(G):
                for j in range(8):
                    z[(8 * r + j) * G + g] = (words[r * G + g] >> (4 * j)) & 0xF
    return [(u[n * K + k] - z[n * G + k // group]) * s[n * G + k // group] for n in range(N) for k in range(K)]


class TorchBackend:
    name = "torch"

    def __init__(self, bits=4, group=128, symmetric=False, scale_dtype="BF16", clip_grid=1, threads=None):
        import torch  # noqa: F401
        import numpy  # noqa: F401  (tensor -> bytes)
        if bits != 4:
            raise BuildError("only 4-bit is implemented")
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        self.group, self.sym, self.sdt = group, symmetric, scale_dtype
        self.ratios = clip_ratios(clip_grid)
        self.tdt = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def tensor(self, raw: bytes, dtype: str, shape):
        return self.torch.frombuffer(bytearray(raw), dtype=self.tdt[dtype]).reshape(*shape)

    def pack_rows(self, u):
        torch = self.torch
        R, C = u.shape
        if C % 8:
            raise BuildError("pack: columns not a multiple of 8")
        v = (u.to(torch.int64).reshape(R, C // 8, 8) << torch.arange(0, 32, 4, dtype=torch.int64)).sum(-1)
        return torch.where(v >= 2 ** 31, v - 2 ** 32, v).to(torch.int32)

    def _bytes(self, t) -> bytes:
        t = t.contiguous()
        if t.dtype in (self.torch.bfloat16, self.torch.float16):
            t = t.view(self.torch.int16)
        return t.numpy().tobytes()

    def quantize(self, raw: bytes, dtype: str, N: int, K: int):
        torch = self.torch
        gs, G = self.group, K // self.group
        g = self.tensor(raw, dtype, (N, K)).to(torch.float32).reshape(N, G, gs)
        mn = g.amin(-1).clamp(max=0.0)
        mx = g.amax(-1).clamp(min=0.0)
        eps = SCALE_EPS[self.sdt]
        best = None
        for r in self.ratios:
            lo, hi = (mn, mx) if r == 1.0 else (mn * torch.full_like(mn, r), mx * torch.full_like(mx, r))
            # tensor / tensor (not tensor / scalar): an IEEE division on every backend, like the reference
            if self.sym:
                s = (torch.maximum(-lo, hi) / torch.full_like(hi, 7.5)).to(self.tdt[self.sdt]).to(torch.float32)
                s = torch.where(s == 0, torch.full_like(s, eps), s)
                zp = torch.zeros_like(s)
            else:
                s = ((hi - lo) / torch.full_like(hi, 15.0)).to(self.tdt[self.sdt]).to(torch.float32)
                s = torch.where(s == 0, torch.full_like(s, eps), s)
                zp = (lo / s).neg_().sub_(8.0).round_().clamp_(-8, 7)
            q = (torch.round(g / s[..., None]) + zp[..., None]).clamp_(-8, 7)
            if len(self.ratios) == 1:
                best = (q, s, zp)
                break
            err = (((q - zp[..., None]) * s[..., None] - g) ** 2).sum(-1)
            if best is None:
                best = (q, s, zp, err)
            else:
                m = err < best[3]
                best = (torch.where(m[..., None], q, best[0]), torch.where(m, s, best[1]),
                        torch.where(m, zp, best[2]), torch.where(m, err, best[3]))
        q, s, zp = best[:3]
        deq = (q - zp[..., None]).double() * s[..., None].double()
        sse = float(((deq - g.double()) ** 2).sum())
        ssw = float((g.double() ** 2).sum())
        out = {"weight_packed": self._bytes(self.pack_rows((q + 8).reshape(N, K))),
               "weight_scale": self._bytes(s.to(self.tdt[self.sdt])),
               "weight_shape": struct.pack("<qq", N, K)}
        if not self.sym:
            out["weight_zero_point"] = self._bytes(self.pack_rows((zp + 8).t().contiguous()).t())
        return out, sse, ssw

    # ---- independent decoder for the self-check (bit ops only, no shared code with the packer) ----
    def unpack_rows(self, b: bytes, R: int, C: int):
        torch = self.torch
        v = torch.frombuffer(bytearray(b), dtype=torch.int32).reshape(R, C // 8).to(torch.int64) & 0xFFFFFFFF
        return torch.stack([(v >> (4 * j)) & 0xF for j in range(8)], dim=-1).reshape(R, C)

    def dequant(self, packed: bytes, scale: bytes, zp: bytes | None, N: int, K: int):
        torch = self.torch
        G = K // self.group
        u = self.unpack_rows(packed, N, K).reshape(N, G, self.group)
        s = self.tensor(scale, self.sdt, (N, G)).to(torch.float32)
        if zp is None:
            z = torch.full((N, G), 8, dtype=torch.int64)
        else:
            v = torch.frombuffer(bytearray(zp), dtype=torch.int32).reshape(N // 8, G).to(torch.int64) & 0xFFFFFFFF
            z = torch.stack([(v >> (4 * j)) & 0xF for j in range(8)], dim=1).reshape(N, G)
        return ((u - z[..., None]).to(torch.float32) * s[..., None]).reshape(N, K), u.reshape(N, K), z, s

    def compare(self, raw_src: bytes, dtype: str, out: dict, N: int, K: int) -> dict:
        torch = self.torch
        w = self.tensor(raw_src, dtype, (N, K)).to(torch.float64)
        deq, u, z, s = self.dequant(out["weight_packed"], out["weight_scale"], out.get("weight_zero_point"), N, K)
        d = deq.double() - w
        sg = s.double().repeat_interleave(self.group, dim=1)
        return {"rel": float(d.norm() / w.norm().clamp_min(1e-30)),
                "cos": float((deq.double() * w).sum() / (deq.double().norm() * w.norm()).clamp_min(1e-30)),
                "max_abs": float(d.abs().max()), "max_err_in_steps": float((d.abs() / sg).max())}


def ref_compare(raw_src: bytes, dtype: str, out: dict, N: int, K: int, group: int, sdt: str) -> dict:
    w = RefBackend.decode(raw_src, dtype)
    deq = ref_dequant(out["weight_packed"], out["weight_scale"], out.get("weight_zero_point"), N, K, group, sdt)
    s = RefBackend.decode(out["weight_scale"], sdt)
    G = K // group
    sse = sum((a - b) ** 2 for a, b in zip(deq, w))
    ssw = sum(b * b for b in w)
    dot = sum(a * b for a, b in zip(deq, w))
    ssd = sum(a * a for a in deq)
    steps = max(abs(deq[i] - w[i]) / s[(i // K) * G + (i % K) // group] for i in range(N * K))
    return {"rel": math.sqrt(sse / ssw) if ssw else 0.0, "cos": dot / math.sqrt(ssd * ssw) if ssd and ssw else 1.0,
            "max_abs": max(abs(a - b) for a, b in zip(deq, w)), "max_err_in_steps": steps}


# ==================================================================================================
# checkpoint analysis
# ==================================================================================================
class Matrix:
    """One expert projection: contiguous bytes [N, K] somewhere in a shard."""
    __slots__ = ("layer", "expert", "proj", "path", "offset", "N", "K", "dtype")

    def __init__(self, layer, expert, proj, path, offset, N, K, dtype):
        self.layer, self.expert, self.proj = layer, expert, proj
        self.path, self.offset, self.N, self.K, self.dtype = path, offset, N, K, dtype

    def read(self) -> bytes:
        return pread(self.path, self.offset, self.N * self.K * DTYPE_SIZE[self.dtype])

    def out_name(self, suffix: str) -> str:
        return f"mtp.layers.{self.layer}.mlp.experts.{self.expert}.{self.proj}.{suffix}"


def analyse(src: str, shard_name: str) -> dict:
    cfg = json.load(open(os.path.join(src, CONFIG)))
    tc = cfg.get("text_config", cfg)
    H, I, E = tc["hidden_size"], tc["moe_intermediate_size"], tc["num_experts"]
    L_main = tc["num_hidden_layers"]
    L_mtp = tc.get("mtp_num_hidden_layers", 1)
    qc = cfg.get("quantization_config")
    if not qc or qc.get("quant_method") != "compressed-tensors":
        raise BuildError("config.json: not a compressed-tensors checkpoint")
    if qc.get("format") != "pack-quantized":
        raise BuildError(f"format {qc.get('format')!r}, expected pack-quantized")
    groups = qc["config_groups"]
    # the scheme the main routed experts use = the group that targets them
    sample = f"model.language_model.layers.0.mlp.experts.0.gate_proj"
    gsel = [g for g in groups.values()
            if ct_target_matches(sample, "RoutedExperts", g.get("targets", []))]
    if len(gsel) != 1:
        raise BuildError(f"{len(gsel)} config groups target the routed experts, expected 1")
    wq = gsel[0]["weights"]
    if (wq.get("type"), wq.get("strategy"), wq.get("num_bits"), wq.get("actorder"), wq.get("dynamic")) != \
            ("int", "group", 4, None, False) or gsel[0].get("input_activations") is not None:
        raise BuildError(f"unsupported weight scheme {wq}")
    group, sym = int(wq["group_size"]), bool(wq["symmetric"])
    if gsel[0].get("format", qc["format"]) != "pack-quantized":
        raise BuildError("group format is not pack-quantized")

    index = json.load(open(os.path.join(src, INDEX)))
    wm = index["weight_map"]
    mtp_names = [k for k in wm if k.startswith("mtp.")]
    if not mtp_names:
        raise BuildError("index has no mtp.* tensors")
    mtp_files = sorted({wm[k] for k in mtp_names})
    if shard_name in set(wm.values()) or os.path.exists(os.path.join(src, shard_name)):
        raise BuildError(f"{shard_name} already exists in the source")
    headers = {}
    for f in mtp_files:
        h, start = read_st_header(os.path.join(src, f))
        names = [k for k in h if k != "__metadata__"]
        foreign = [k for k in names if not k.startswith("mtp.")]
        if foreign:
            raise BuildError(f"{f} also holds non-MTP tensors ({foreign[:2]}); this script drops whole MTP shards")
        headers[f] = (h, start)
    in_files = {k for f, (h, _) in headers.items() for k in h if k != "__metadata__"}
    if in_files != set(mtp_names):
        raise BuildError("mtp.* tensors in the index and in the MTP shards differ")

    # expert matrices
    mats, expert_src = [], set()
    for L in range(L_mtp):
        pre = f"mtp.layers.{L}.mlp.experts"
        fused_gu, fused_dn = f"{pre}.gate_up_proj", f"{pre}.down_proj"
        if fused_gu in in_files or fused_dn in in_files:
            fgu, fdn = wm[fused_gu], wm[fused_dn]
            hgu, sgu = headers[fgu]
            hdn, sdn = headers[fdn]
            igu, idn = hgu[fused_gu], hdn[fused_dn]
            if igu["shape"] != [E, 2 * I, H] or idn["shape"] != [E, H, I]:
                raise BuildError(f"fused experts {igu['shape']} / {idn['shape']}, expected {[E, 2 * I, H]} / {[E, H, I]}"
                                 " (transposed layout is not handled)")
            dt = igu["dtype"]
            es = DTYPE_SIZE[dt]
            for e in range(E):
                b = sgu + igu["data_offsets"][0] + e * 2 * I * H * es
                mats.append(Matrix(L, e, "gate_proj", os.path.join(src, fgu), b, I, H, dt))
                mats.append(Matrix(L, e, "up_proj", os.path.join(src, fgu), b + I * H * es, I, H, dt))
                mats.append(Matrix(L, e, "down_proj", os.path.join(src, fdn),
                                   sdn + idn["data_offsets"][0] + e * H * I * DTYPE_SIZE[idn["dtype"]], H, I, idn["dtype"]))
            expert_src |= {fused_gu, fused_dn}
            layout_kind = "fused 3D (HF v5: gate_up_proj [E, 2I, H], gate = first I rows)"
        else:
            for e in range(E):
                for p in PROJS:
                    n = f"{pre}.{e}.{p}.weight"
                    if n not in in_files:
                        raise BuildError(f"missing MTP expert tensor {n}")
                    h, st = headers[wm[n]]
                    want = [H, I] if p == "down_proj" else [I, H]
                    if h[n]["shape"] != want:
                        raise BuildError(f"{n}: shape {h[n]['shape']}, expected {want}")
                    mats.append(Matrix(L, e, p, os.path.join(src, wm[n]), st + h[n]["data_offsets"][0],
                                       want[0], want[1], h[n]["dtype"]))
                    expert_src.add(n)
            layout_kind = "per expert (experts.E.{gate,up,down}_proj.weight)"
    for m in mats:
        if m.dtype not in ("BF16", "F16", "F32"):
            raise BuildError(f"MTP experts are {m.dtype}, expected a float dtype (already quantized?)")
        if m.K % group or m.N % 8:
            raise BuildError(f"{m.proj}: [{m.N}, {m.K}] incompatible with group {group} / int32 packing")

    # template: a routed expert of the main model
    tmpl = {}
    pat = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.0\.(gate_proj|up_proj|down_proj)\."
                     r"(weight_packed|weight_scale|weight_zero_point|weight_shape)$")
    tmpl_layer = None
    for k, f in wm.items():
        m = pat.match(k)
        if m and (tmpl_layer is None or m.group(1) == tmpl_layer):
            tmpl_layer = m.group(1)
            if f not in headers:
                headers[f] = read_st_header(os.path.join(src, f))
            h, st = headers[f]
            tmpl[(m.group(2), m.group(3))] = (h[k]["dtype"], h[k]["shape"], os.path.join(src, f), st + h[k]["data_offsets"][0])
    if not tmpl:
        raise BuildError("no routed expert of the main model found in the index (template)")
    scale_dtype = tmpl[("gate_proj", "weight_scale")][0]
    if ("gate_proj", "weight_zero_point") in tmpl and sym:
        raise BuildError("main experts have zero points but the config says symmetric")
    if not sym and ("gate_proj", "weight_zero_point") not in tmpl:
        raise BuildError("config says asymmetric but the main experts have no weight_zero_point")

    non_expert = sorted(in_files - expert_src)
    return {"cfg": cfg, "qc": qc, "index": index, "wm": wm, "H": H, "I": I, "E": E, "L_main": L_main,
            "L_mtp": L_mtp, "group": group, "sym": sym, "scale_dtype": scale_dtype, "mtp_files": mtp_files,
            "headers": headers, "mats": mats, "non_expert": non_expert, "expert_src": sorted(expert_src),
            "tmpl": tmpl, "tmpl_layer": tmpl_layer, "layout_kind": layout_kind, "mtp_names": mtp_names}


def expected_shapes(N: int, K: int, group: int, sym: bool, sdt: str) -> dict:
    d = {"weight_packed": ("I32", [N, K // 8]), "weight_scale": (sdt, [N, K // group]), "weight_shape": ("I64", [2])}
    if not sym:
        d["weight_zero_point"] = ("I32", [N // 8, K // group])
    return d


def plan_entries(a: dict) -> tuple[list[Entry], list[tuple], dict]:
    """entries of the new shard, copy jobs [(entry_idx, path, abs_offset)], expert entries {(mat_idx, suffix): entry_idx}"""
    entries, copies, ex = [], [], {}
    wm, headers, src_files = a["wm"], a["headers"], {}
    for n in a["non_expert"]:
        f = wm[n]
        h, st = headers[f]
        entries.append(Entry(n, h[n]["dtype"], h[n]["shape"]))
        copies.append((len(entries) - 1, f, st + h[n]["data_offsets"][0]))
    for mi, m in enumerate(a["mats"]):
        for suf, (dt, shp) in expected_shapes(m.N, m.K, a["group"], a["sym"], a["scale_dtype"]).items():
            t = a["tmpl"].get((m.proj, suf))
            if t is None or (t[0], t[1]) != (dt, shp):
                raise BuildError(f"{m.proj}.{suf}: planned {dt} {shp}, main-model expert has {t and t[:2]}")
            entries.append(Entry(m.out_name(suf), dt, shp))
            ex[(mi, suf)] = len(entries) - 1
    return entries, copies, ex


# ==================================================================================================
# build
# ==================================================================================================
def link_tree(src: str, dst: str, skip_top: set[str]) -> list[tuple[str, str]]:
    done = []
    src = os.path.abspath(src)
    for root, dirs, files in os.walk(src, followlinks=False):
        rel_root = os.path.relpath(root, src)
        out_root = dst if rel_root == "." else os.path.join(dst, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for name in sorted(dirs + files):
            rel = name if rel_root == "." else os.path.join(rel_root, name)
            if rel_root == "." and name in skip_top:
                if name in dirs:
                    dirs.remove(name)
                continue
            sp, dp = os.path.join(root, name), os.path.join(out_root, name)
            if os.path.islink(sp):
                target = os.readlink(sp)
                if not os.path.isabs(target):
                    resolved = os.path.normpath(os.path.join(root, target))
                    if not (resolved + os.sep).startswith(src + os.sep):
                        target = resolved
                os.symlink(target, dp)
                if name in dirs:
                    dirs.remove(name)
                done.append((rel, "symlink"))
            elif os.path.isdir(sp):
                continue
            elif os.path.isfile(sp):
                os.link(sp, dp)
                done.append((rel, "hardlink"))
            else:
                log(f"    skipping special file {rel}")
    return done


def json_dump(obj, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def write_shard(path: str, a: dict, backend, entries, copies, ex, progress=True) -> dict:
    header, data_len = layout(entries, {"format": "pt"})
    start = len(header)
    stats = {p: [0, 0.0, 0.0, 0.0] for p in PROJS}  # count, sum rel, max rel, sse/ssw (for global)
    tot_sse = tot_ssw = 0.0
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.ftruncate(fd, start + data_len)
        os.pwrite(fd, header, 0)
        for ei, f, off in copies:
            e = entries[ei]
            os.pwrite(fd, pread(os.path.join(a["src"], f), off, e.nbytes), start + e.offset)
        t0, last = time.time(), 0.0
        for mi, m in enumerate(a["mats"]):
            out, sse, ssw = backend.quantize(m.read(), m.dtype, m.N, m.K)
            for suf, b in out.items():
                e = entries[ex[(mi, suf)]]
                if len(b) != e.nbytes:
                    raise BuildError(f"{e.name}: produced {len(b)} bytes, planned {e.nbytes}")
                os.pwrite(fd, b, start + e.offset)
            rel = math.sqrt(sse / ssw) if ssw else 0.0
            st = stats[m.proj]
            st[0] += 1
            st[1] += rel
            st[2] = max(st[2], rel)
            tot_sse += sse
            tot_ssw += ssw
            if progress and time.time() - last > 20:
                last = time.time()
                done = (mi + 1) / len(a["mats"])
                eta = (time.time() - t0) / done * (1 - done)
                log(f"    {mi + 1}/{len(a['mats'])} expert matrices ({done:.0%}, ETA {eta / 60:.1f} min)")
        os.fsync(fd)
    finally:
        os.close(fd)
    return {"per_proj": {p: {"n": s[0], "mean_rel": s[1] / max(s[0], 1), "max_rel": s[2]} for p, s in stats.items()},
            "global_rel": math.sqrt(tot_sse / tot_ssw) if tot_ssw else 0.0, "data_bytes": data_len,
            "file_bytes": start + data_len}


def self_check(a: dict, shard_path: str, backend, sample_experts: list[int]) -> list[str]:
    """Re-read the written shard and verify it. Returns a list of failures (empty = OK)."""
    fails = []
    hdr = check_st_layout(shard_path)
    _, start = read_st_header(shard_path)
    names = {k for k in hdr if k != "__metadata__"}
    want = set(a["non_expert"]) | {m.out_name(s) for m in a["mats"]
                                   for s in expected_shapes(m.N, m.K, a["group"], a["sym"], a["scale_dtype"])}
    if names != want:
        fails.append(f"tensor names differ: {sorted(names ^ want)[:4]}")
    # non-expert tensors byte-identical
    for n in a["non_expert"]:
        f = a["wm"][n]
        h, st = a["headers"][f]
        b0 = pread(os.path.join(a["src"], f), st + h[n]["data_offsets"][0], nbytes_of(h[n]["dtype"], h[n]["shape"]))
        b1 = pread(shard_path, start + hdr[n]["data_offsets"][0], nbytes_of(hdr[n]["dtype"], hdr[n]["shape"]))
        if b0 != b1:
            fails.append(f"{n}: bytes differ from the source")
    log(f"    {len(a['non_expert'])} non-expert MTP tensors byte-identical to the source: "
        f"{'yes' if not any('bytes differ' in x for x in fails) else 'NO'}")
    # sampled experts: decode from the FILE and compare with the BF16 source
    rows = []
    for m in a["mats"]:
        if m.expert not in sample_experts:
            continue
        out = {}
        for suf in expected_shapes(m.N, m.K, a["group"], a["sym"], a["scale_dtype"]):
            i = hdr[m.out_name(suf)]
            out[suf] = pread(shard_path, start + i["data_offsets"][0], nbytes_of(i["dtype"], i["shape"]))
        if struct.unpack("<qq", out["weight_shape"]) != (m.N, m.K):
            fails.append(f"{m.out_name('weight_shape')} != [{m.N}, {m.K}]")
        if isinstance(backend, TorchBackend):
            r = backend.compare(m.read(), m.dtype, out, m.N, m.K)
        else:
            r = ref_compare(m.read(), m.dtype, out, m.N, m.K, a["group"], a["scale_dtype"])
        rows.append((m.layer, m.expert, m.proj, r))
        lim_steps = 0.6 if len(backend.ratios) == 1 else 8.0
        if r["rel"] > 0.25 or r["cos"] < 0.96 or r["max_err_in_steps"] > lim_steps:
            fails.append(f"expert {m.expert} {m.proj}: {r}")
    log("    sampled experts, dequantized from the written file vs BF16 source:")
    log("      layer expert proj        rel.err  cosine    max|err|  max err/step")
    for L, e, p, r in rows:
        log(f"      {L:5d} {e:6d} {p:10s} {r['rel']:8.4f} {r['cos']:9.6f} {r['max_abs']:9.5f} {r['max_err_in_steps']:8.3f}")
    # cross-check the bit layout with compressed-tensors' own packer (if it is importable here)
    if isinstance(backend, TorchBackend):
        msg = ct_library_check(backend, shard_path, hdr, start, a)
        log(f"    compressed-tensors library cross-check: {msg}")
        if msg.startswith("FAIL"):
            fails.append(msg)
        msg = safetensors_library_check(shard_path, hdr, start)
        log(f"    safetensors library read-back: {msg}")
        if msg.startswith("FAIL"):
            fails.append(msg)
    return fails


def _import_ct():
    paths = ["compressed_tensors.compressors.quantized_compressors.pack_quantized",
             "compressed_tensors.compressors.pack_quantized", "compressed_tensors.compressors",
             "compressed_tensors.utils"]
    for p in paths:
        try:
            mod = __import__(p, fromlist=["pack_to_int32", "unpack_from_int32"])
            if hasattr(mod, "pack_to_int32") and hasattr(mod, "unpack_from_int32"):
                return mod.pack_to_int32, mod.unpack_from_int32, p
        except Exception:  # noqa: BLE001
            continue
    try:  # moved in a newer release? search the package
        import importlib
        import pkgutil
        import compressed_tensors
        for info in pkgutil.walk_packages(compressed_tensors.__path__, "compressed_tensors."):
            if "pack" not in info.name and "compress" not in info.name:
                continue
            try:
                mod = importlib.import_module(info.name)
            except Exception:  # noqa: BLE001
                continue
            if hasattr(mod, "pack_to_int32") and hasattr(mod, "unpack_from_int32"):
                return mod.pack_to_int32, mod.unpack_from_int32, info.name
    except Exception:  # noqa: BLE001
        pass
    return None


def ct_library_check(backend, shard_path, hdr, start, a) -> str:
    imp = _import_ct()
    if imp is None:
        return "skipped (compressed_tensors pack_to_int32/unpack_from_int32 not importable)"
    pack, unpack, where = imp
    torch = backend.torch
    try:
        m = next(x for x in a["mats"] if x.expert == 0 and x.proj == "gate_proj")
        rd = lambda n: pread(shard_path, start + hdr[n]["data_offsets"][0],  # noqa: E731
                             nbytes_of(hdr[n]["dtype"], hdr[n]["shape"]))
        pw = torch.frombuffer(bytearray(rd(m.out_name("weight_packed"))), dtype=torch.int32).reshape(m.N, m.K // 8)
        q = backend.unpack_rows(rd(m.out_name("weight_packed")), m.N, m.K).to(torch.int8) - 8
        if not torch.equal(pack(q, 4), pw):
            return f"FAIL: pack_to_int32 ({where}) packs our codes differently"
        if not torch.equal(unpack(pw, 4, torch.Size([m.N, m.K])).to(torch.int8), q):
            return f"FAIL: unpack_from_int32 ({where}) decodes our weight_packed differently"
        if not a["sym"]:
            G = m.K // a["group"]
            pz = torch.frombuffer(bytearray(rd(m.out_name("weight_zero_point"))), dtype=torch.int32).reshape(m.N // 8, G)
            z = unpack(pz, 4, torch.Size([m.N, G]), packed_dim=0).to(torch.int8)
            if not torch.equal(pack(z, 4, packed_dim=0), pz):
                return "FAIL: zero point packing differs from pack_to_int32(packed_dim=0)"
            if int(z.min()) < -8 or int(z.max()) > 7:
                return "FAIL: zero points out of range"
        return f"OK (bit-exact with {where}.pack_to_int32/unpack_from_int32, weights and zero points)"
    except Exception as e:  # noqa: BLE001
        return f"skipped ({type(e).__name__}: {e})"


def safetensors_library_check(shard_path, hdr, start) -> str:
    try:
        from safetensors import safe_open
    except Exception:  # noqa: BLE001
        return "skipped (safetensors not importable)"
    try:
        names = [k for k in hdr if k != "__metadata__"]
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            if keys != set(names):
                return "FAIL: key set differs"
            for n in names[:: max(1, len(names) // 40)]:
                t = f.get_tensor(n)
                b = pread(shard_path, start + hdr[n]["data_offsets"][0], nbytes_of(hdr[n]["dtype"], hdr[n]["shape"]))
                tt = t.contiguous()
                if tt.dtype.is_floating_point and tt.element_size() == 2:
                    import torch
                    tt = tt.view(torch.int16)
                if tt.numpy().tobytes() != b:
                    return f"FAIL: {n} differs when read by safetensors"
        return f"OK ({len(keys)} tensors, sampled values identical)"
    except Exception as e:  # noqa: BLE001
        return f"FAIL ({type(e).__name__}: {e})"


def vllm_scheme_check(new_qc: dict, old_qc: dict, runtime_layer: int) -> str:
    """Ask the installed vLLM's own CompressedTensorsConfig which modules of the draft get quantized."""
    try:
        import torch
        from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
    except Exception as e:  # noqa: BLE001
        return f"skipped (vLLM not importable: {type(e).__name__})"
    try:
        def scheme_table(qcd):
            qc = CompressedTensorsConfig.from_config(copy.deepcopy(qcd))
            qc.packed_modules_mapping = dict(MTP_PACKED)
            qc._add_fused_moe_to_target_scheme_map()
            res = {}
            lin, experts = mtp_module_names(runtime_layer)
            for name, cls in lin:
                mod = type(cls, (torch.nn.Module,), {})()
                res[name] = qc.get_scheme_dict(mod, name) is not None
            mod = type("RoutedExperts", (torch.nn.Module,), {})()
            sd = [qc.get_scheme_dict(mod, experts + s) for s in (".0.gate_proj", ".0.up_proj", ".0.down_proj")]
            res[experts] = sd[0] is not None and all(x == sd[0] for x in sd)
            w = sd[0]["weights"] if sd[0] else None
            return res, w
        new, w = scheme_table(new_qc)
        old, _ = scheme_table(old_qc)
        _, experts = mtp_module_names(runtime_layer)
        bad = [n for n, q in new.items() if q != (n == experts)]
        if bad or old[experts]:
            return f"FAIL: quantized={ {n: new[n] for n in bad} }, old config experts quantized={old[experts]}"
        return (f"OK (installed vLLM: {experts} -> W{w.num_bits} group {w.group_size} symmetric={w.symmetric}; "
                f"{len(new) - 1} other draft linears stay unquantized; old config: experts unquantized)")
    except Exception as e:  # noqa: BLE001
        return f"skipped ({type(e).__name__}: {e})"


def build(args) -> int:
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    parent = os.path.dirname(dst)
    staging = os.path.join(parent, f".{os.path.basename(dst)}.staging-{os.getpid()}")
    log(f"== swift MTP INT4 build\n   src {src}\n   dst {dst}")
    if os.path.lexists(dst):
        raise BuildError(f"{dst} exists - refusing to overwrite")
    if not os.path.isdir(parent):
        raise BuildError(f"{parent} does not exist")
    if os.stat(src).st_dev != os.stat(parent).st_dev:
        raise BuildError(f"{src} and {parent} are on different filesystems: hardlinks impossible")

    log("-- 1. analyse the source")
    a = analyse(src, args.shard_name)
    a["src"] = src
    runtime_layer = a["L_main"]  # vLLM builds draft layer idx at prefix mtp.layers.<num_hidden_layers + idx>
    log(f"    scheme: W4 group {a['group']} {'symmetric' if a['sym'] else 'ASYMMETRIC (zero points)'}, "
        f"scale {a['scale_dtype']}; template = model.language_model.layers.{a['tmpl_layer']}.mlp.experts.0.*")
    log(f"    MTP shard(s) {a['mtp_files']}: {len(a['mtp_names'])} tensors; experts {a['layout_kind']}")
    log(f"    {len(a['mats'])} expert matrices ({a['E']} experts x 3, layers {a['L_mtp']}), "
        f"{len(a['non_expert'])} non-expert tensors copied byte-for-byte")
    if a["mats"]:
        m0 = a["tmpl"][("gate_proj", "weight_shape")]
        shp = struct.unpack("<qq", pread(m0[2], m0[3], 16))
        log(f"    main expert gate_proj weight_shape = {list(shp)} (N=out, K=in) - ours writes [{a['I']}, {a['H']}]")
        if list(shp) != [a["I"], a["H"]]:
            raise BuildError("main expert weight_shape is not [out, in]")
        if not a["sym"]:
            zt = a["tmpl"][("gate_proj", "weight_zero_point")]
            words = struct.unpack(f"<{nbytes_of(zt[0], zt[1]) // 4}I", pread(zt[2], zt[3], nbytes_of(zt[0], zt[1])))
            nib = [(w >> (4 * j)) & 0xF for w in words for j in range(8)]
            log(f"    main expert zero points (stored, +8 offset): mean {sum(nib) / len(nib):.2f}, "
                f"min {min(nib)}, max {max(nib)}")

    entries, copies, ex = plan_entries(a)
    header, data_len = layout(list(entries), {"format": "pt"})
    old_bytes = sum(nbytes_of(a["headers"][a["wm"][n]][0][n]["dtype"], a["headers"][a["wm"][n]][0][n]["shape"])
                    for n in a["mtp_names"])
    log(f"    MTP tensors: {old_bytes / 2**30:.3f} GiB -> {data_len / 2**30:.3f} GiB "
        f"(saves {(old_bytes - data_len) / 2**30:.3f} GiB of weights on the GPU holding the drafter)")

    log("-- 2. config.json (ignore list) and index")
    new_cfg = copy.deepcopy(a["cfg"])
    nqc = new_cfg["quantization_config"]
    nqc["ignore"], changes = derive_ignore(nqc.get("ignore") or [], runtime_layer, 0)
    for c in changes:
        log(f"    ignore: {c}")
    for L in (runtime_layer, 0):
        old_t = dict(ct_draft_table(a["qc"], L))
        new_t = dict(ct_draft_table(nqc, L))
        _, experts = mtp_module_names(L)
        bad = [n for n, q in new_t.items() if q != (n == experts)]
        if bad or old_t[experts]:
            raise BuildError(f"ignore simulation (layer {L}): wrong modules quantized: {bad}")
    log(f"    simulated vLLM matching: draft prefix mtp.layers.{runtime_layer}.mlp.experts -> quantized "
        f"(was ignored); fc_embedding/fc_hidden/qkv/o/indexer/shared_expert/HC -> still ignored")
    for p in a["qc"].get("ignore") or []:  # main-model entries untouched
        if p != OLD_IGNORE and p not in nqc["ignore"]:
            raise BuildError(f"ignore entry {p} lost")
    msg = vllm_scheme_check(nqc, a["qc"], runtime_layer) if not args.no_vllm_check else "skipped (--no-vllm-check)"
    log(f"    installed-vLLM check: {msg}")
    if msg.startswith("FAIL"):
        raise BuildError(msg)

    wm = a["wm"]
    new_wm = {k: v for k, v in wm.items() if not k.startswith("mtp.")}
    for e in entries:
        new_wm[e.name] = args.shard_name
    new_index = copy.deepcopy(a["index"])
    new_index["weight_map"] = new_wm
    if "total_size" in new_index.get("metadata", {}):
        new_index["metadata"]["total_size"] = int(new_index["metadata"]["total_size"]) - old_bytes + data_len
    ple = [k for k in new_wm if ".ngram_embedding.shard_" in k]
    log(f"    index: {len(a['mtp_names'])} mtp entries -> {len(entries)} in {args.shard_name}; "
        f"{len(ple)} PLE entries kept; {a['mtp_files']} no longer referenced (not linked)")
    for f in sorted(set(new_wm.values())):
        if f != args.shard_name and not os.path.isfile(os.path.join(src, f)):
            raise BuildError(f"index references {f}, missing in source")

    if args.dry_run:
        log("== dry run OK: nothing written")
        return 0

    if args.crosscheck_intel:
        crosscheck_intel(a, args.crosscheck_intel)

    backend = (TorchBackend if args.backend == "torch" else RefBackend)(
        4, a["group"], a["sym"], a["scale_dtype"], args.clip_grid,
        **({"threads": args.threads} if args.backend == "torch" else {}))
    log(f"-- 3. build {staging}  (backend {backend.name}, clip grid {args.clip_grid}"
        f"{', threads %d' % backend.torch.get_num_threads() if args.backend == 'torch' else ''})")
    os.makedirs(staging)
    try:
        skip = {CONFIG, INDEX} | set(a["mtp_files"])
        linked = link_tree(src, staging, skip)
        t0 = time.time()
        st = write_shard(os.path.join(staging, args.shard_name), a, backend, entries, copies, ex)
        log(f"    quantized {len(a['mats'])} matrices in {time.time() - t0:.0f} s; relative error (Frobenius) "
            f"overall {st['global_rel']:.4f}; " + ", ".join(
                f"{p} mean {v['mean_rel']:.4f} max {v['max_rel']:.4f}" for p, v in st["per_proj"].items()))
        json_dump(new_cfg, os.path.join(staging, CONFIG))
        json_dump(new_index, os.path.join(staging, INDEX))
        with open(os.path.join(staging, PROVENANCE), "w") as f:
            f.write(f"built={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} by swift_mtp4_build.py\n"
                    f"source={src}\n"
                    f"{args.shard_name}: all mtp.* tensors of {a['mtp_files']}; {len(a['mats'])} routed-expert "
                    f"matrices RTN W4 g{a['group']} {'sym' if a['sym'] else 'asym'} (compressed-tensors pack-quantized,"
                    f" scale {a['scale_dtype']}), clip grid {args.clip_grid}; relative error {st['global_rel']:.4f}\n"
                    f"{CONFIG}: ignore {'; '.join(changes)}\n"
                    f"{INDEX}: mtp.* -> {args.shard_name}; total_size {old_bytes - data_len:+d} B removed\n"
                    f"all other files: hardlinks of {src} (same inode); {a['mtp_files']} not carried over\n")
        log("-- 4. self-check (reads the written files)")
        fails = self_check(a, os.path.join(staging, args.shard_name), backend, args.sample_experts)
        n_hl = 0
        for rel, kind in linked:
            if kind == "hardlink":
                s0, s1 = os.stat(os.path.join(src, rel)), os.stat(os.path.join(staging, rel))
                if (s0.st_dev, s0.st_ino) != (s1.st_dev, s1.st_ino):
                    fails.append(f"{rel} is not a hardlink of the source")
                n_hl += 1
        for f in set(new_wm.values()):
            if not os.path.isfile(os.path.join(staging, f)):
                fails.append(f"index references missing {f}")
        for f in a["mtp_files"]:
            if os.path.exists(os.path.join(staging, f)):
                fails.append(f"old MTP shard {f} is still in the output dir")
        plef = sorted({new_wm[k] for k in ple})
        log(f"    {n_hl} files hardlinked (same inode), incl. {len(plef)} PLE shard files: "
            f"{plef[0] if plef else '-'} ... {plef[-1] if plef else ''}")
        if fails:
            raise BuildError("self-check failed:\n      " + "\n      ".join(fails))
        log("    self-check OK")
        os.rename(staging, dst)
    except BaseException:
        if args.keep_staging:
            log(f"!! failed; staging kept at {staging}")
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    log(f"== done: {dst}\n   {args.shard_name}: {st['file_bytes'] / 1e9:.3f} GB "
        f"(was {sum(os.path.getsize(os.path.join(src, f)) for f in a['mtp_files']) / 1e9:.3f} GB); "
        f"drafter weights -{(old_bytes - data_len) / 2**30:.2f} GiB on the last PP stage")
    return 0


def crosscheck_intel(a: dict, intel_dir: str) -> None:
    """Optional: compare Swift's fused gate/up split with Intel's per-expert BF16 MTP experts."""
    import torch
    p = os.path.join(intel_dir, "model_extra_tensors.safetensors")
    h, st = read_st_header(p)
    log(f"-- cross-check expert layout against {p}")
    for e in (0, 1, a["E"] - 1):
        mats = {m.proj: m for m in a["mats"] if m.layer == 0 and m.expert == e}
        for proj in PROJS:
            n = f"mtp.layers.0.mlp.experts.{e}.{proj}.weight"
            if n not in h:
                log(f"    {n} not in Intel sidecar - skipped")
                return
        sw = {k: torch.frombuffer(bytearray(m.read()), dtype=torch.bfloat16).float().flatten() for k, m in mats.items()}
        iw = {k: torch.frombuffer(bytearray(pread(p, st + h[f"mtp.layers.0.mlp.experts.{e}.{k}.weight"]["data_offsets"][0],
                                                  mats[k].N * mats[k].K * 2)), dtype=torch.bfloat16).float().flatten()
              for k in PROJS}
        cos = lambda x, y: float((x @ y) / (x.norm() * y.norm()))  # noqa: E731
        same = all(torch.equal(sw[k], iw[k]) for k in PROJS)
        log(f"    expert {e}: cos(swift gate, intel gate)={cos(sw['gate_proj'], iw['gate_proj']):.4f} "
            f"cos(swift gate, intel up)={cos(sw['gate_proj'], iw['up_proj']):.4f} "
            f"cos(down)={cos(sw['down_proj'], iw['down_proj']):.4f}{'  (byte-identical: MTP not retrained)' if same else ''}")


# ==================================================================================================
# selftest (no checkpoint): torch backend vs pure-Python reference, library cross-checks
# ==================================================================================================
def _bf16_bytes(vals: list[float]) -> bytes:
    return struct.pack(f"<{len(vals)}H", *[struct.unpack("<I", struct.pack("<f", bf16_round(f32(v))))[0] >> 16
                                           for v in vals])


def synthetic_matrix(N: int, K: int, seed: int, special: bool = True) -> bytes:
    rnd = random.Random(seed)
    vals = [rnd.gauss(0.0, 0.02) for _ in range(N * K)]
    if special and N >= 6 and K >= 256:
        for k in range(128):
            vals[0 * K + k] = 0.0                       # all-zero group
            vals[1 * K + k] = 0.013                     # constant positive group
            vals[2 * K + k] = -abs(vals[2 * K + k])     # all-negative group
            vals[3 * K + k] = abs(vals[3 * K + k])      # all-positive group
            vals[4 * K + k] = vals[4 * K + k] * 1e-30    # subnormal-ish tiny values
        vals[5 * K + 7] = 0.9                           # outlier
    return _bf16_bytes(vals)


def selftest(args) -> int:
    ok = True
    log("== selftest")
    shapes = [(16, 256), (8, 512), (64, 128)]
    for sym in (False, True):
        for grid in (1, 4):
            ref = RefBackend(4, 128, sym, "BF16", grid)
            try:
                tb = TorchBackend(4, 128, sym, "BF16", grid, threads=args.threads)
            except ImportError as e:
                log(f"   torch/numpy not importable ({e}); only the reference backend is tested")
                tb = None
            for i, (N, K) in enumerate(shapes):
                raw = synthetic_matrix(N, K, 1000 + i)
                ro, rsse, rssw = ref.quantize(raw, "BF16", N, K)
                rc = ref_compare(raw, "BF16", ro, N, K, 128, "BF16")
                line = f"   sym={sym!s:5} grid={grid} [{N}x{K}] ref rel={rc['rel']:.4f} steps={rc['max_err_in_steps']:.2f}"
                if grid == 1 and rc["max_err_in_steps"] > 0.6:
                    ok = False
                    line += "  FAIL(ref bound)"
                if tb is not None:
                    to, tsse, tssw = tb.quantize(raw, "BF16", N, K)
                    tc = tb.compare(raw, "BF16", to, N, K)
                    same = all(to[k] == ro[k] for k in ro) and set(to) == set(ro)
                    line += f" | torch rel={tc['rel']:.4f}"
                    if grid == 1:
                        line += " bit-exact" if same else "  FAIL(torch != reference)"
                        ok &= same
                    else:
                        mism = sum(a != b for a, b in zip(to["weight_packed"], ro["weight_packed"]))
                        line += f" ({mism} packed bytes differ: near-tie clip choices allowed)"
                        if tc["rel"] > rc["rel"] * 1.02 + 1e-6:
                            ok = False
                            line += " FAIL(torch error worse)"
                    if abs(tc["rel"] - rc["rel"]) > 1e-3 and grid == 1:
                        ok = False
                log(line)
    # error must not grow with clipping, must shrink on average
    N, K = 64, 256
    raw = synthetic_matrix(N, K, 7, special=False)
    r1 = ref_compare(raw, "BF16", RefBackend(4, 128, False, "BF16", 1).quantize(raw, "BF16", N, K)[0], N, K, 128, "BF16")
    r8 = ref_compare(raw, "BF16", RefBackend(4, 128, False, "BF16", 8).quantize(raw, "BF16", N, K)[0], N, K, 128, "BF16")
    log(f"   clip search: minmax rel={r1['rel']:.4f} -> grid 8 rel={r8['rel']:.4f}")
    ok &= r8["rel"] <= r1["rel"] + 1e-9
    try:
        tb = TorchBackend(4, 128, False, "BF16", 1, threads=args.threads)
    except ImportError:
        tb = None
    if tb is not None:
        torch = tb.torch
        imp = _import_ct()
        if imp is None:
            log("   compressed_tensors pack_to_int32/unpack_from_int32 not importable: library cross-check skipped")
        else:
            pack, unpack, where = imp
            g = torch.Generator().manual_seed(0)
            q = torch.randint(-8, 8, (32, 256), generator=g, dtype=torch.int8)
            a1 = tb.pack_rows((q.to(torch.int64) + 8))
            a2 = pack(q, 4)
            z1 = tb.pack_rows((q.to(torch.int64) + 8).t().contiguous()).t()
            z2 = pack(q, 4, packed_dim=0)
            e1, e2 = torch.equal(a1, a2), torch.equal(z1.contiguous(), z2.contiguous())
            e3 = torch.equal(unpack(a2, 4, torch.Size([32, 256])).to(torch.int8), q)
            log(f"   {where}: pack rows {'==' if e1 else '!='}, pack dim0 {'==' if e2 else '!='}, "
                f"unpack {'==' if e3 else '!='}")
            ok &= e1 and e2 and e3
        try:
            from vllm.model_executor.layers.quantization.utils.quant_utils import unpack_quantized_values_into_int32
            from vllm.scalar_type import scalar_types
            q = torch.randint(0, 16, (16, 64), dtype=torch.int64)
            p = tb.pack_rows(q)
            u = unpack_quantized_values_into_int32(p, scalar_types.uint4, packed_dim=1)
            e4 = torch.equal(u.to(torch.int64), q)
            log(f"   vLLM unpack_quantized_values_into_int32(uint4, packed_dim=1) {'==' if e4 else '!='} our packing")
            ok &= e4
        except Exception as e:  # noqa: BLE001
            log(f"   vLLM unpack check skipped ({type(e).__name__}: {e})")
    log("== selftest " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="/opt/vllm-qfn/models/swift15-awq")
    ap.add_argument("--dst", default="/opt/vllm-qfn/models/swift15-awq-mtp4")
    ap.add_argument("--shard-name", default=DEFAULT_SHARD)
    ap.add_argument("--clip-grid", type=int, default=1,
                    help="1 = plain min/max RTN (default); N>1 = per-group MSE search over N clip ratios 1.0..0.70")
    ap.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--sample-experts", default="0,1,127,255,511",
                    help="experts dequantized from the written file in the self-check")
    ap.add_argument("--backend", choices=("torch", "python"), default="torch",
                    help="python = slow pure-Python reference (tests only)")
    ap.add_argument("--crosscheck-intel", metavar="DIR",
                    help="also compare the gate/up split with Intel's per-expert MTP weights in DIR")
    ap.add_argument("--no-vllm-check", action="store_true", help="skip the installed-vLLM matching check")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--keep-staging", action="store_true")
    args = ap.parse_args(argv)
    args.sample_experts = [int(x) for x in args.sample_experts.split(",") if x.strip()]
    try:
        return selftest(args) if args.selftest else build(args)
    except BuildError as e:
        log(f"!! {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
