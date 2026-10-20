#!/usr/bin/env python3
"""Q6-class Swift build: routed experts of chosen layers as INT8 (W8A16, symmetric, group 128, round-to-nearest from the
BF16 release), every other tensor unchanged from SRC (Swift AWQ INT4 experts elsewhere; attention as in SRC, i.e. INT4
in swift15-awq, BF16 in swift15-a16). sm_80 has Marlin MoE kernels for 4 and 8 bits only, so "Q6" = a 4/8-bit mix.

The AWQ recipe only smooths inside each expert (up_proj -> down_proj) and uses plain RTN for attention, so replacing a
whole expert (gate, up, down) by a fresh quantization of the BF16 weights is self-consistent.

  check  SRC                     source identity (preserved BF16 tensors byte-equal between SRC and the BF16 release)
                                 and the BF16 expert layout (gate/up order, orientation) against the AWQ experts
  prep   SRC LAYERS              download + quantize the experts of LAYERS (e.g. 0-7,40-47) into CACHE/layer-L.safetensors
  build  SRC OUT LAYERS          assemble OUT from SRC (hardlinks for untouched files) + the cached INT8 layers
  verify OUT                     index/shard consistency, INT8 dequant vs BF16 on a sample, vLLM scheme resolution
Run with the n4 venv python (torch, safetensors, compressed_tensors, vllm)."""
import json, os, re, sys, time, urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.helpers import pack_to_int32, unpack_from_int32

CACHE = Path(os.environ.get("Q6X_CACHE", "/opt/vllm-qfn/models/.q6x-cache"))
LM = "model.language_model.layers"
PROJ = ("gate_proj", "up_proj", "down_proj")
GS = 128


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def layers_arg(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return sorted(set(out))


# ---------------------------------------------------------------- BF16 release over HTTP (range reads, no full download)
class Remote:
    def __init__(self, src):
        m = json.load(open(Path(src) / "QUANTIZATION_MANIFEST.json"))
        self.repo = os.environ.get("Q6X_REPO", m["source_repo"])
        self.rev = os.environ.get("Q6X_REV", m["source_revision"])
        self.urls, self.hdrs = {}, {}
        self.index = json.loads(self._get(self._url("model.safetensors.index.json")))["weight_map"]
        log(f"BF16 release {self.repo}@{self.rev[:10]}: {len(self.index)} tensors")

    def _url(self, f):
        if f in self.urls:
            return self.urls[f]
        url = f"https://huggingface.co/{self.repo}/resolve/{self.rev}/{f}"

        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None

        op = urllib.request.build_opener(NoRedirect)
        for _ in range(6):  # renamed repos redirect within huggingface.co first, then to the CDN
            try:
                op.open(urllib.request.Request(url, method="HEAD"), timeout=60)
                break
            except urllib.error.HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    raise
                url = urllib.parse.urljoin(url, e.headers["Location"])
                if "/resolve/" not in urllib.parse.urlparse(url).path:
                    break
        self.urls[f] = url
        return url

    @staticmethod
    def _get(url, start=None, end=None):
        for attempt in range(8):
            try:
                h = {} if start is None else {"Range": f"bytes={start}-{end}"}
                with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=180) as r:
                    data = r.read()
                if start is not None and len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} of {end - start + 1}")
                return data
            except Exception as e:  # noqa: BLE001
                if attempt == 7:
                    raise
                log(f"  retry {attempt + 1}: {e}")
                time.sleep(2 ** attempt)

    def header(self, f):
        if f not in self.hdrs:
            u = self._url(f)
            n = int.from_bytes(self._get(u, 0, 7), "little")
            self.hdrs[f] = (8 + n, json.loads(self._get(u, 8, 8 + n - 1)))
        return self.hdrs[f]

    def meta(self, name):
        f = self.index[name]
        base, h = self.header(f)
        t = h[name]
        return f, base + t["data_offsets"][0], base + t["data_offsets"][1], t["shape"], t["dtype"]

    def tensor(self, name):
        f, a, b, shape, dt = self.meta(name)
        assert dt == "BF16", (name, dt)
        return torch.frombuffer(bytearray(self._get(self._url(f), a, b - 1)), dtype=torch.bfloat16).view(shape)

    def rows(self, name, i0, i1, pool=None):
        """slice [i0:i1] of the first dim of a BF16 tensor, fetched in parallel 64 MiB pieces"""
        f, a, b, shape, dt = self.meta(name)
        per = (b - a) // shape[0]
        lo, hi = a + i0 * per, a + i1 * per
        step = 64 << 20
        parts = [(s, min(s + step, hi) - 1) for s in range(lo, hi, step)]
        u = self._url(f)
        get = (lambda p: self._get(u, *p))
        data = b"".join(pool.map(get, parts) if pool else map(get, parts))
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).view([i1 - i0] + shape[1:])


# ---------------------------------------------------------------- quantization helpers
def q8(w):
    """w [out, in] -> (int32 packed [out, in/4], bf16 scale [out, in/128]); symmetric, scale from the group absmax"""
    o, i = w.shape
    g = w.float().view(o, i // GS, GS)
    s = (g.abs().amax(-1) / 127.0).clamp(min=1e-12).to(torch.bfloat16)
    q = torch.round(g / s.float().unsqueeze(-1)).clamp(-128, 127).to(torch.int8).view(o, i)
    return pack_to_int32(q, 8), s


def dq8(packed, scale, shape):
    q = unpack_from_int32(packed, 8, torch.Size(shape), packed_dim=1).float()
    o, i = shape
    return (q.view(o, i // GS, GS) * scale.float().unsqueeze(-1)).view(o, i)


def dq4_asym(packed, scale, zp, shape):
    o, i = shape
    q = unpack_from_int32(packed, 4, torch.Size(shape), packed_dim=1).float()
    z = unpack_from_int32(zp, 4, torch.Size([o, i // GS]), packed_dim=0).float()
    return ((q.view(o, i // GS, GS) - z.unsqueeze(-1)) * scale.float().unsqueeze(-1)).view(o, i)


def rel(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm())


def cos_rows(a, b):
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=1).mean())


class Src:
    def __init__(self, d):
        self.d = Path(d)
        self.index = json.load(open(self.d / "model.safetensors.index.json"))["weight_map"]

    def get(self, name):
        with safe_open(self.d / self.index[name], "pt") as f:
            return f.get_tensor(name)


def expert_views(remote, L, e0, e1, pool=None, layout=None):
    """BF16 gate, up, down for experts e0..e1 of layer L as [n, out, in]"""
    gu = remote.rows(f"{LM}.{L}.mlp.experts.gate_up_proj", e0, e1, pool)
    dn = remote.rows(f"{LM}.{L}.mlp.experts.down_proj", e0, e1, pool)
    lay = layout or json.load(open(CACHE / "layout.json"))
    if lay["gate_up_transposed"]:
        gu = gu.transpose(1, 2)
    if lay["down_transposed"]:
        dn = dn.transpose(1, 2)
    inter = gu.shape[1] // 2
    gate, up = (gu[:, :inter], gu[:, inter:]) if lay["gate_first"] else (gu[:, inter:], gu[:, :inter])
    return gate, up, dn


# ---------------------------------------------------------------- commands
def check(src):
    S, R = Src(src), Remote(src)
    ok = True
    for name in [f"{LM}.0.mlp.shared_expert.gate_proj.weight", f"{LM}.0.mlp.gate.weight",
                 f"{LM}.47.mlp.shared_expert.down_proj.weight", f"{LM}.23.mlp.shared_expert_gate.weight"]:
        a, b = S.get(name), R.tensor(name)
        same = a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))
        ok &= same
        log(f"identity {name}: {'byte-equal' if same else 'DIFFERENT'}")
    if not ok:
        sys.exit("BF16 release is not the source of SRC - set Q6X_REPO/Q6X_REV")
    H, I = 2560, 640
    _, _, _, gshape, _ = R.meta(f"{LM}.0.mlp.experts.gate_up_proj")
    _, _, _, dshape, _ = R.meta(f"{LM}.0.mlp.experts.down_proj")
    log(f"BF16 fused shapes: gate_up {gshape}, down {dshape}")
    lay = {"gate_up_transposed": gshape[1:] == [H, 2 * I], "down_transposed": dshape[1:] == [I, H]}
    assert gshape[1:] in ([2 * I, H], [H, 2 * I]) and dshape[1:] in ([H, I], [I, H]), "unexpected fused layout"
    awq = {p: dq4_asym(*(S.get(f"{LM}.0.mlp.experts.3.{p}.{k}") for k in ("weight_packed", "weight_scale", "weight_zero_point")),
                       S.get(f"{LM}.0.mlp.experts.3.{p}.weight_shape").tolist()) for p in PROJ}
    lay["gate_first"] = True
    g, u, d = (x[0] for x in expert_views(R, 0, 3, 4, layout=lay))
    e_first, e_second = rel(awq["gate_proj"], g), rel(awq["gate_proj"], u)
    log(f"gate_proj (unsmoothed) AWQ vs BF16 first half {e_first:.3f}, second half {e_second:.3f}")
    lay["gate_first"] = e_first < e_second
    if lay["gate_first"] is False:
        g, u = u, g
    best, other = min(e_first, e_second), max(e_first, e_second)
    cu = cos_rows(awq["up_proj"], u)
    cd = cos_rows(awq["down_proj"].T, d.T)  # down is smoothed per input column -> compare columns
    log(f"up_proj row cosine {cu:.4f}; down_proj column cosine {cd:.4f}")
    if not (best < 0.2 and other > 0.6 and cu > 0.97 and cd > 0.97):
        sys.exit(f"layout check FAILED: {lay}")
    CACHE.mkdir(parents=True, exist_ok=True)
    json.dump(lay | {"repo": R.repo, "rev": R.rev}, open(CACHE / "layout.json", "w"))
    w8 = dq8(*q8(g), list(g.shape))
    log(f"layout OK {lay}; INT8 RTN error on that gate_proj {rel(w8, g):.4f} vs AWQ INT4 {best:.3f}")


def prep(src, layers):
    R = Remote(src)
    lay = json.load(open(CACHE / "layout.json"))
    assert lay["repo"] == R.repo and lay["rev"] == R.rev, "run check first"
    E, CH = 512, 32
    with ThreadPoolExecutor(8) as pool:
        for L in layers:
            out = CACHE / f"layer-{L}.safetensors"
            if out.exists():
                log(f"layer {L}: cached"); continue
            t0 = time.time(); ts = {}
            for e0 in range(0, E, CH):
                gate, up, dn = expert_views(R, L, e0, e0 + CH, pool, lay)
                for j in range(CH):
                    for p, w in zip(PROJ, (gate[j], up[j], dn[j])):
                        pk, s = q8(w)
                        n = f"{LM}.{L}.mlp.experts.{e0 + j}.{p}"
                        ts[n + ".weight_packed"] = pk
                        ts[n + ".weight_scale"] = s
                        ts[n + ".weight_shape"] = torch.tensor(list(w.shape), dtype=torch.int64)
            save_file(ts, str(out) + ".tmp", metadata={"format": "pt"})
            os.replace(str(out) + ".tmp", out)
            gib = sum(t.numel() * t.element_size() for t in ts.values()) / 2**30
            log(f"layer {L}: {gib:.2f} GiB INT8 in {time.time() - t0:.0f} s")


def build(src, out, layers):
    src, out = Path(src), Path(out)
    idx = json.load(open(src / "model.safetensors.index.json"))
    wm = idx["weight_map"]
    pat = re.compile(rf"^{re.escape(LM)}\.({'|'.join(map(str, layers))})\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.")
    drop = {n for n in wm if pat.match(n)}
    shards = sorted({wm[n] for n in drop})
    log(f"{len(drop)} INT4 expert tensors of {len(layers)} layers in {len(shards)} shards to rewrite")
    for L in layers:
        assert (CACHE / f"layer-{L}.safetensors").exists(), f"layer {L} not prepared"
    out.mkdir(parents=True, exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.name not in shards and f.name not in ("config.json", "model.safetensors.index.json"):
            os.link(f, out / f.name)
    new_wm = {n: v for n, v in wm.items() if n not in drop}
    for sh in shards:
        layers_here = sorted({int(pat.match(n).group(1)) for n in drop if wm[n] == sh})
        ts = {}
        with safe_open(src / sh, "pt") as f:
            for k in f.keys():
                if k not in drop:
                    ts[k] = f.get_tensor(k)
        for L in layers_here:
            with safe_open(CACHE / f"layer-{L}.safetensors", "pt") as f:
                for k in f.keys():
                    # an expert goes to the shard that held its INT4 weight_packed
                    if wm.get(k.rsplit(".", 1)[0] + ".weight_packed") == sh:
                        ts[k] = f.get_tensor(k); new_wm[k] = sh
        save_file(ts, str(out / sh) + ".tmp", metadata={"format": "pt"})
        os.replace(str(out / sh) + ".tmp", out / sh)
        os.sync()
        log(f"wrote {sh} (layers {layers_here})")
    total = 0
    for sh in sorted(set(new_wm.values())):
        with open(out / sh, "rb") as fh:
            n = int.from_bytes(fh.read(8), "little"); h = json.loads(fh.read(n))
        total += sum(v["data_offsets"][1] - v["data_offsets"][0] for k, v in h.items() if k != "__metadata__")
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_wm.items()))},
              open(out / "model.safetensors.index.json", "w"), indent=2)
    cfg = json.load(open(src / "config.json"))
    qc = cfg["quantization_config"]
    qc["config_groups"]["group_q6x_int8"] = {
        "format": "pack-quantized", "input_activations": None, "output_activations": None,
        # prefix-agnostic (vLLM may rename model.language_model.*); MTP stays excluded by the ignore entry re:^mtp.*
        "targets": [rf"re:.*\.layers\.({'|'.join(map(str, layers))})\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$"],
        "weights": {"actorder": None, "block_structure": None, "dynamic": False, "group_size": GS, "num_bits": 8,
                    "observer": "minmax", "observer_kwargs": {}, "strategy": "group", "symmetric": True, "type": "int"}}
    json.dump(cfg, open(out / "config.json", "w"), indent=2)
    lay = json.load(open(CACHE / "layout.json"))
    json.dump({"base": str(src), "int8_expert_layers": layers, "scheme": "W8A16 symmetric group 128 RTN (absmax)",
               "bf16_source": f"{lay['repo']}@{lay['rev']}", "built": time.strftime("%F %T"), "tool": "build_q6x.py"},
              open(out / "Q6X_MANIFEST.json", "w"), indent=2)
    log(f"built {out}: {total / 2**30:.1f} GiB indexed")


def verify(out):
    out = Path(out)
    man = json.load(open(out / "Q6X_MANIFEST.json")); layers = man["int8_expert_layers"]
    wm = json.load(open(out / "model.safetensors.index.json"))["weight_map"]
    seen = {}
    for sh in sorted(set(wm.values())):
        with safe_open(out / sh, "pt") as f:
            for k in f.keys():
                seen[k] = seen.get(k, 0) + 1
                assert wm.get(k) == sh, f"{k} in {sh} but index says {wm.get(k)}"
    assert set(seen) == set(wm) and max(seen.values()) == 1, "index and shards disagree"
    log(f"index OK: {len(wm)} tensors, each in exactly one shard")
    S = Src(out)
    R = Remote(out)
    lay = json.load(open(CACHE / "layout.json"))
    for L in (layers[0], layers[len(layers) // 2], layers[-1]):
        for e in (0, 257, 511):
            g, u, d = (x[0] for x in expert_views(R, L, e, e + 1, layout=lay))
            for p, w in zip(PROJ, (g, u, d)):
                n = f"{LM}.{L}.mlp.experts.{e}.{p}"
                assert f"{n}.weight_zero_point" not in wm
                w8 = dq8(S.get(n + ".weight_packed"), S.get(n + ".weight_scale"), S.get(n + ".weight_shape").tolist())
                r = rel(w8, w)
                assert r < 0.01, f"{n}: INT8 error {r}"
        log(f"layer {L}: INT8 experts match BF16 (relative error < 1%)")
    other = next(L for L in range(48) if L not in layers)
    assert f"{LM}.{other}.mlp.experts.0.gate_proj.weight_zero_point" in wm, "untouched layer lost its INT4 tensors"
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    cfg = CompressedTensorsConfig.from_config(json.load(open(out / "config.json"))["quantization_config"])
    cfg._add_fused_moe_to_target_scheme_map()
    mod = type("RoutedExperts", (torch.nn.Module,), {})()
    for prefix in ("model.language_model.layers", "language_model.model.layers"):
        for L in range(48):
            sd = cfg.get_scheme_dict(mod, f"{prefix}.{L}.mlp.experts.0.gate_proj")
            bits = sd["weights"].num_bits if sd else None
            assert bits == (8 if L in layers else 4), f"{prefix}.{L}: scheme bits {bits}"
    sd = cfg.get_scheme_dict(mod, "mtp.layers.48.mlp.experts.0.gate_proj")
    assert sd is None, "MTP experts would be quantized"
    log("vLLM scheme resolution OK: INT8 on the chosen layers, INT4 elsewhere, MTP ignored")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "check":
        check(sys.argv[2])
    elif cmd == "prep":
        prep(sys.argv[2], layers_arg(sys.argv[3]))
    elif cmd == "build":
        build(sys.argv[2], sys.argv[3], layers_arg(sys.argv[4]))
    elif cmd == "verify":
        verify(sys.argv[2])
    else:
        sys.exit(__doc__)
