#!/usr/bin/env python3
"""Which scheme will OUR vLLM pick for each Swift-A16 linear module?  (no GPU)

Replays vLLM's compressed-tensors `ignore` matching on a model's config.json,
exactly as CompressedTensorsConfig.get_scheme_dict does at load time:
  * ignore entries / layer names are renamed with the model class's
    hf_to_vllm_mapper (model.language_model. -> language_model.model., ...),
  * fused modules are expanded with packed_modules_mapping (qkv_proj -> q/k/v,
    in_proj_qkvz -> in_proj_qkv/in_proj_z, ...) and must be all-ignored or
    none-ignored (vLLM raises otherwise),
  * routed experts are checked the way CompressedTensorsMoEMethod does
    (<prefix>.experts.0.{gate,up,down}_proj).

Run it with the vLLM venv's python on the server:
    /opt/vllm-qfn/.venv/bin/python vllm_scheme_check.py /opt/vllm-qfn/models/swift15-a16
It uses vLLM's own should_ignore_layer and the real Qwen4Exp class mappings
when they import; otherwise it loads vLLM's pure-python config_utils.py
directly from the installed package (or --vllm-src) with a hard-coded copy
of the mappings.  Exit code 0 = attention unquantized, experts INT4.
"""
import argparse
import importlib.util
import json
import os
import re
import sys

FALLBACK_PACKED = {  # Qwen4ExpForConditionalGeneration.packed_modules_mapping (model.py)
    "qkv_proj": ["q_proj", "k_proj", "v_proj"], "gate_up_proj": ["gate_proj", "up_proj"],
    "qkv": ["qkv"], "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"], "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "kv_proj": ["key_proj", "value_proj"],
    "input_mix_weight_down_block_inject": ["input_mix_weight_down", "block_inject_weight", "_input_mix_padding"],
}
FALLBACK_PREFIX = {"model.visual.": "visual.", "lm_head.": "language_model.lm_head.",
                   "model.language_model.": "language_model.model."}  # Qwen3VL hf_to_vllm_mapper


def load_matcher(vllm_src):
    """Return (should_ignore(layer, ignore, packed), rename(target), packed, source-description)."""
    packed, rename, how = FALLBACK_PACKED, None, []
    try:
        from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpForConditionalGeneration as M
        packed = dict(M.packed_modules_mapping)
        mapper = M.hf_to_vllm_mapper.get_rename_mapper()
        rename = lambda t: t if (t.startswith("re:") or "." not in t) else mapper._map_name(t)
        how.append("mappings from vllm Qwen4ExpForConditionalGeneration")
    except Exception as e:  # noqa: BLE001
        how.append("hard-coded mappings (model import failed: %s)" % type(e).__name__)
    if rename is None:
        def rename(t):
            if t.startswith("re:") or "." not in t:
                return t
            for a, b in FALLBACK_PREFIX.items():
                if t.startswith(a):
                    return b + t[len(a):]
            return t
    try:
        from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer
        how.append("vllm compressed_tensors.utils.should_ignore_layer")
        return (lambda l, ign, pk: should_ignore_layer(l, ignore=ign, fused_mapping=pk)), rename, packed, how
    except Exception as e:  # noqa: BLE001
        how.append("standalone config_utils.py (%s)" % type(e).__name__)
    root = vllm_src
    if root is None:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.origin:
            raise SystemExit("vLLM not found: pass --vllm-src /path/to/vllm (the package dir)")
        root = os.path.dirname(spec.origin)
    path = os.path.join(root, "model_executor/layers/quantization/utils/config_utils.py")
    if importlib.util.find_spec("regex") is None:
        sys.modules["regex"] = re
    spec = importlib.util.spec_from_file_location("_vllm_config_utils", path)
    cu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cu)

    def should_ignore(layer, ignore, fused):  # == compressed_tensors/utils.should_ignore_layer
        per = cu.find_matching_patterns(layer, ignore, fused)
        s = [len(m) > 0 for m in per]
        if any(s) and not all(s):
            raise ValueError("Found different quantization schemes for the shards of " + layer)
        return all(s)
    how.append(path)
    return should_ignore, rename, packed, how


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model dir (config.json)")
    ap.add_argument("--vllm-src", help="vllm package dir, only for the standalone fallback")
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model, "config.json")))
    qc = cfg.get("quantization_config") or {}
    should_ignore, rename, packed, how = load_matcher(args.vllm_src)
    print("matcher:", "; ".join(how))
    ign = [rename(t) for t in qc.get("ignore", [])]
    tc = cfg.get("text_config", cfg)
    rows = {}

    def add(kind, name):
        try:
            r = "BF16 (ignored)" if should_ignore(name, ign, packed) else "INT4 (quantized)"
        except ValueError as e:
            r = "ERROR: %s" % e
        rows.setdefault(kind, {}).setdefault(r, []).append(name)

    for L, t in enumerate(tc["layer_types"]):
        p = "language_model.model.layers.%d." % L
        if t == "full_attention":
            add("QSA self_attn.qkv_proj", p + "self_attn.qkv_proj")
            add("QSA self_attn.o_proj", p + "self_attn.o_proj")
            add("QSA indexer.index_qk_proj", p + "self_attn.indexer.index_qk_proj")
        else:
            add("GDN linear_attn.in_proj_qkvz", p + "linear_attn.in_proj_qkvz")
            add("GDN linear_attn.out_proj", p + "linear_attn.out_proj")
            add("GDN linear_attn.in_proj_ba", p + "linear_attn.in_proj_ba")
        add("shared_expert.gate_up_proj", p + "mlp.shared_expert.gate_up_proj")
        add("shared_expert.down_proj", p + "mlp.shared_expert.down_proj")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            add("routed experts", p + "mlp.experts.0." + proj)
    ok = True
    for kind, res in rows.items():
        for r, names in res.items():
            print("  %-30s %-18s %3d modules" % (kind, r, len(names)))
            if r.startswith("ERROR"):
                ok = False
    attn_kinds = [k for k in rows if k.startswith(("QSA self_attn", "GDN linear_attn.in_proj_qkvz", "GDN linear_attn.out_proj"))]
    if any(set(rows[k]) != {"BF16 (ignored)"} for k in attn_kinds):
        print("NOTE: attention projections are not all BF16 in this config")
        ok = False
    if set(rows["routed experts"]) != {"INT4 (quantized)"}:
        print("ERROR: routed experts are not all INT4")
        ok = False
    print("RESULT:", "OK" if ok else "CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
