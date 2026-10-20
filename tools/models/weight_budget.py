#!/usr/bin/env python3
"""Weight budget of a checkpoint from safetensors headers only (no tensor data read).
Groups bytes by component: routed experts, shared expert, attention (QSA / GDN), router, embeddings, lm_head, PLE,
MTP, norms/other; and reports the dtypes per group. usage: weight_budget.py MODEL_DIR..."""
import json, re, struct, sys
from collections import defaultdict
from pathlib import Path

SZ = {"BF16": 2, "F16": 2, "F32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "F8_E4M3": 1, "BOOL": 1, "U32": 4, "I16": 2}


def group(name):
    n = name.lower()
    if "mtp" in n:
        return "mtp"
    if "ngram" in n or "ple" in n.split(".") or ".ple." in n or "per_layer" in n:
        return "ple"
    if "visual" in n or "vision" in n:
        return "vision"
    if n.startswith("lm_head"):
        return "lm_head"
    if "embed_tokens" in n:
        return "embed"
    if "shared_expert" in n:
        return "shared_expert"
    if ".experts." in n:
        return "routed_experts"
    if re.search(r"\.(gate|router)\.(weight|bias)$", n) or "shared_expert_gate" in n:
        return "router"
    if "self_attn" in n or "linear_attn" in n or ".attn." in n or "qsa" in n or "indexer" in n:
        return "attention"
    return "other"


for d in map(Path, sys.argv[1:]):
    tot = defaultdict(int); dt = defaultdict(lambda: defaultdict(int))
    for f in sorted(d.glob("*.safetensors")):
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            b = SZ.get(v["dtype"], 2)
            for s in v["shape"]:
                b *= s
            g = group(k); tot[g] += b; dt[g][v["dtype"]] += b
    print(f"== {d.name}: total {sum(tot.values()) / 2**30:.1f} GiB")
    for g, b in sorted(tot.items(), key=lambda x: -x[1]):
        print(f"   {g:15} {b / 2**30:7.2f} GiB   " + ", ".join(f"{k} {v / 2**30:.2f}" for k, v in sorted(dt[g].items(), key=lambda x: -x[1])))
    cfg = json.load(open(d / "config.json")); t = cfg.get("text_config", cfg)
    print("   config:", {k: t.get(k) for k in ("num_experts", "num_experts_per_tok", "moe_intermediate_size", "shared_expert_intermediate_size",
                                             "intermediate_size", "num_hidden_layers", "hidden_size")})
