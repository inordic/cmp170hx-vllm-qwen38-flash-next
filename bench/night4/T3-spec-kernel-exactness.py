#!/usr/bin/env python3
"""T3: offline GPU test that probabilistic MTP drafting + rejection sampling in THIS tree is unbiased.

No model, no server: it drives the tree's own Triton kernels (gumbel_sample for the draft, rejection_sample for the
verification) on synthetic target/draft distributions, k = 1, and compares the distribution of the first emitted
token with the target distribution p (total variation + chi-square).

Cases (adversarial p/q pairs from vektorprime mtp-spec-decode tests, where shared noise gave TV ~0.06):
  shared   draft drawn with is_drafting=False -> same Gumbel key as the residual resample (pre-vllm#54282 behaviour).
           Expected: biased (TV well above the noise floor). Shows the test can see the bug.
  tree     draft drawn exactly as sample_draft does it in this tree (is_drafting=True, salted key).
           Expected: TV at the noise floor.
  c3       draft logits shaped by Asgard patch C3 (top-k 2 / top-p 0.9, tau 0.8) before sampling and caching.
           Expected: TV at the noise floor (lossless). Skipped if C3 is not applied.
  c1       unit check of Asgard patch C1 gather_draft_sampled (placeholder rows -> -1). Skipped if not applied.

usage (on a CUDA box with the patched tree importable, e.g. inside the server venv; nothing else needs to run):
  python3 T3-spec-kernel-exactness.py [--samples 2000000] [--batch 262144]
Noise floor for TV with 4 tokens and 2M samples is ~0.001; the script flags TV > 0.005 or chi-square p < 1e-4.
"""
import argparse
import math
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--samples", type=int, default=2_000_000)
ap.add_argument("--batch", type=int, default=262_144)
a = ap.parse_args()

from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample  # noqa: E402
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import rejection_sample  # noqa: E402

dev = torch.device("cuda")
V = 8  # vocab; the unused tail has -inf logits in both p and q
PAIRS = [
    ([0.1, 0.45, 0.45], [0.8, 0.19, 0.01]),
    ([0.2, 0.4, 0.4], [0.6, 0.3, 0.1]),
    ([0.05, 0.5, 0.3, 0.15], [0.7, 0.25, 0.04, 0.01]),
]


def logits_of(probs):
    x = torch.full((V,), float("-inf"))
    x[: len(probs)] = torch.tensor(probs).log()
    return x


def chi2_sf(x, k):
    # Wilson-Hilferty approximation of the chi-square survival function.
    if k <= 0:
        return 1.0
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))


def run(p, q, mode, shape=None):
    p_l, q_l = logits_of(p).to(dev), logits_of(q).to(dev)
    counts = torch.zeros(V, dtype=torch.int64, device=dev)
    done, it = 0, 0
    while done < a.samples:
        n = min(a.batch, a.samples - done)
        g = torch.Generator(device="cpu").manual_seed(1234 + it)
        seeds = torch.randint(0, 2**62, (n,), generator=g, dtype=torch.int64).to(dev)
        temp = torch.ones(n, dtype=torch.float32, device=dev)
        base_pos = torch.randint(100, 100_000, (n,), generator=g, dtype=torch.int64).to(dev)
        idx = torch.arange(n, dtype=torch.int32, device=dev)
        draft_cache = torch.zeros(n, 1, V, dtype=torch.float32, device=dev)
        q_rows = q_l.expand(n, V).contiguous()
        if shape is not None:
            q_rows = shape(q_rows, temp)
        # Draft for the token at P+1 is keyed by P, the position of the row that verifies it.
        d = gumbel_sample(
            q_rows, idx, temp, seeds, base_pos, apply_temperature=True,
            is_drafting=(mode != "shared"), logits_cache=draft_cache,
            logits_cache_col=torch.zeros((), dtype=torch.int64, device=dev),
        )
        target = p_l.expand(2 * n, V).contiguous()
        draft_sampled = torch.zeros(2 * n, dtype=torch.int32, device=dev)
        draft_sampled[1::2] = d.to(torch.int32)
        pos = torch.stack([base_pos, base_pos + 1], dim=1).reshape(-1).contiguous()
        cu = torch.arange(0, 2 * n + 1, 2, dtype=torch.int32, device=dev)
        exp_idx = torch.repeat_interleave(idx, 2)
        local = torch.tensor([0, 1], dtype=torch.int32, device=dev).repeat(n)
        sampled, _ = rejection_sample(target, draft_cache, draft_sampled, cu, pos, idx, exp_idx, local,
                                      temp, seeds, 1)
        counts += torch.bincount(sampled[:, 0].long(), minlength=V)[:V]
        done += n
        it += 1
    emp = counts.double().cpu() / done
    tgt = torch.zeros(V, dtype=torch.float64)
    tgt[: len(p)] = torch.tensor(p, dtype=torch.float64)
    tv = 0.5 * (emp - tgt).abs().sum().item()
    mask = tgt > 0
    chi = (((counts.double().cpu() - done * tgt) ** 2)[mask] / (done * tgt[mask])).sum().item()
    outside = counts.cpu()[~mask].sum().item()
    return tv, chi2_sf(chi, int(mask.sum()) - 1), outside


def c3_shaper():
    try:
        from vllm.v1.worker.gpu.spec_decode.speculator import c3_shape_draft_logits
    except ImportError:
        return None

    def shape(rows, temp):
        n = rows.shape[0]
        top_k = torch.full((n,), 2, dtype=torch.int32, device=dev)
        top_p = torch.full((n,), 0.9, dtype=torch.float32, device=dev)
        return c3_shape_draft_logits(rows, top_k, top_p, temp, 64, 0.8, torch.float32)

    return shape


bad = 0
shape = c3_shaper()
for p, q in PAIRS:
    for mode in ("shared", "tree", "c3"):
        if mode == "c3" and shape is None:
            print(f"p={p} q={q} c3: skipped (Asgard patch C3 not applied)")
            continue
        tv, pval, outside = run(p, q, mode, shape if mode == "c3" else None)
        biased = tv > 0.005 or pval < 1e-4 or outside > 0
        expect_bias = mode == "shared"
        verdict = ("biased (expected: shows the test detects shared noise)" if biased and expect_bias
                   else "UNEXPECTED: not biased" if expect_bias
                   else "BIASED - FAIL" if biased else "ok")
        if (biased and not expect_bias):
            bad += 1
        print(f"p={p} q={q} {mode:6}: TV={tv:.4f} chi2 p={pval:.3g} outside={outside}  {verdict}")

try:
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler import gather_draft_sampled
except ImportError:
    gather_draft_sampled = None
if gather_draft_sampled is None:
    print("c1: skipped (Asgard patch C1 not applied)")
else:
    # Req 0: decode step (positions 10..13, prefill_len 10) -> drafts kept.
    # Req 1: padded one-token prompt tail (positions 4..7, prefill_len 5) -> drafts become -1.
    input_ids = torch.tensor([5, 6, 7, 8, 9, 0, 0, 0], dtype=torch.int32, device=dev)
    positions = torch.tensor([10, 11, 12, 13, 4, 5, 6, 7], dtype=torch.int64, device=dev)
    logits_idx = torch.arange(8, dtype=torch.int64, device=dev)
    exp_idx = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=dev)
    local = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32, device=dev)
    prefill_len = torch.tensor([10, 5], dtype=torch.int32, device=dev)
    ds, ps = gather_draft_sampled(input_ids, positions, logits_idx, exp_idx, local, prefill_len)
    ok = ds.tolist() == [5, 6, 7, 8, 9, -1, -1, -1] and ps.tolist() == positions.tolist()
    bad += not ok
    print(f"c1: gather_draft_sampled -> {ds.tolist()}  {'ok' if ok else 'FAIL'}")

print("RESULT:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
