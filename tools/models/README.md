# Model build tools

The scripts that built the checkpoints and the draft head tested in night 4 (2026-10-19/20). None of them ships weights: each one reads a checkpoint you already have, or fetches tensors from Hugging Face, and writes a new model directory next to it. The source directories are never modified. Unchanged files are hardlinked, so the PLE-SSD loader still finds its shards at the same `data_offsets`.

The paths in the examples are ours (`/opt/vllm-qfn/models/...`, the n4 venv `/opt/vllm-qfn-n4/.venv`). The night-4 results are in [`../../results/`](../../results/) and in the [night-4 report](../../research/2026-10-20-night4-drafters-and-quality-builds.md).

| Script | Builds | Runs with | Used in production? |
|---|---|---|---|
| `quantize_lm_head.py` + `make-heads.sh` | the int4 (W4A16, group 128, symmetric) MTP draft head for patch 08 (D1), from the checkpoint's BF16 `lm_head`. CPU only, about 15 s. Swift 1.5 AWQ and Intel AutoRound share the same BF16 `lm_head`, so both heads came out byte-identical (sha256 `5c5461d7…`) | the n4 venv's python | **yes**: `VLLM_MTP_DRAFT_HEAD_PATH` in [`../../config/qfn-prod.env`](../../config/qfn-prod.env) |
| `swift_mtp4_build.py` | `swift15-awq-mtp4`: the MTP layer's 512 routed experts quantized to INT4 in Swift's own compressed-tensors format; the drafter shrinks from 4.86 to 1.39 GiB | the vLLM venv's python (torch CPU) | no: −1.1% decode score for +26% KV, rejected by the 1% rule |
| `intel_mtp4_install.py` | `intel-autoround-mtp4`: Intel AutoRound with hampsonw's INT4 MTP-expert delta, downloaded and verified, with a config that works in our tree | any python3 (standard library) | no (Intel speed test only) |
| `fetch_bf16_attention.py` | downloads only the 156 BF16 attention tensors (4.98 GiB) of Swift 1.5's BF16 release with HTTP Range requests; resumable | any python3 (standard library) | no |
| `build_swift_a16.py` | `swift15-a16`: Swift 1.5 AWQ with those BF16 attention projections in place of the INT4 ones; the experts stay INT4. Verifies the result | any python3 (standard library) | no: the quality option. Night 4: +2.6 BFCL, +1.1 long context, −10% decode. Night 5: long context 100%, +1.6 BFCL in total but lower multi-turn tool use (57.0% against 61.3%). It was live for about a quarter of an hour on 2026-10-21 as part of C1T, then rolled back |
| `vllm_scheme_check.py` | nothing: replays vLLM's compressed-tensors ignore matching on a `config.json`, to show which modules load as BF16 and which as INT4. No GPU | the vLLM venv's python | |
| `build_q6x.py` | `swift15-q6x`: A16 plus INT8 (W8A16, group 128, RTN from the BF16 release) routed experts in chosen layers (night 4: 0–10 and 43–47) | the n4 venv's python | no: no quality gain over A16, and −4% decode and −39% KV against A16 |
| `weight_budget.py` | nothing: bytes per component (experts, attention, PLE, MTP, …) and dtypes, from the safetensors headers only | any python3 | |

Detailed notes, written before the builds:
- [`swift-a16.md`](swift-a16.md): Swift-A16 (`fetch_bf16_attention.py`, `build_swift_a16.py`, `vllm_scheme_check.py`), with the provenance checks, the vLLM loading path and the risks;
- [`mtp4-int4-experts.md`](mtp4-int4-experts.md): the INT4 MTP experts for Swift and Intel, including why hampsonw's `config.json` does not apply in our tree as published.

## The draft head (production)

```bash
# make-heads.sh: one head per checkpoint into /opt/vllm-qfn/models/draft-heads/
/opt/vllm-qfn-n4/.venv/bin/python quantize_lm_head.py --model-dir /opt/vllm-qfn/models/swift15-awq \
    --out /opt/vllm-qfn/models/draft-heads/swift15-awq-lm_head-w4a16.safetensors --mse-scale-search
```

It reads only `lm_head.weight` (BF16, 248,320 × 2,560) and writes `lm_head.qweight` / `scales` / `qzeros` in the GPTQ layout that vLLM's Marlin W4A16 path loads (330 MB). `--mse-scale-search` picks each group's scale from a small grid by reconstruction error instead of plain round-to-nearest. Only draft acceptance depends on the head: the target still verifies every token with its BF16 head.

## Credits

- `quantize_lm_head.py` is from [AlexBrandes/cmp170hx-qwen38-flash-next](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next) (`tools/`), MIT; licence in [`quantize_lm_head.LICENSE`](quantize_lm_head.LICENSE). Its docstring refers to that repository's `download.sh`.
- The INT4 MTP-expert delta for Intel is hampsonw's (`hampsonw/Qwen3.8-Flash-Next-W4A16-AutoRound-MTP-INT4-Experts`); `intel_mtp4_install.py` downloads it and does not include it.
- The BF16 attention comes from ukisai's BF16 release of Swift 1.5 (`ukisai/Swift1.5-Qwen3.8-Flash-Next`) and the INT8 experts of `build_q6x.py` from the same release; neither is included.
- The other scripts are ours (MIT, like the rest of this repository).
