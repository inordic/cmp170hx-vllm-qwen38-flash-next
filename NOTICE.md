# Notice

This repository (scripts, configs, documents, the quality harness) is MIT-licensed; see [`LICENSE`](LICENSE). It builds on, and runs, the following work under their own terms. No model weights, datasets or firmware are included.

## Engine

- **vLLM** ([vllm-project/vllm](https://github.com/vllm-project/vllm)), Apache-2.0, at commit `a5a30471ff2bb7f0824f2da10e358af98d304472`. Not redistributed.
  - The patches in `patches/` modify vLLM source and contain short excerpts of it as anchors. They are provided under Apache-2.0.
  - Patches 01–04 are backports or ports of vllm-project/vllm PRs #55601, #60210, #56734 and #60615.
- **The PLE-SSD recipe**: iIIusi0n, [qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx), Apache-2.0.
- **The base build**: 2dameneko, [vllm-Qwen3.8-Flash-Next-2xCMP170HX](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX), Apache-2.0, commit `56e1347`. Their own NOTICE lists further work, including vllm#54709 and DominikBucko's overlays.

## Night-4 patches (08–18; 14 is not published)

- **08 (D1) and 09 (D2)** are ports of AlexBrandes/vllm commits `14457f89e` (MTP W4A16 draft head) and `31208d203` (bounded prefill beside decodes), Apache-2.0 like vLLM.
- **11 (C1)** is a backport of vllm-project/vllm#58784, Apache-2.0.
- **13 (C5)** follows xris99's Mamba block-alignment fix (qwen3.8-flash-next-cmp170hx, commit `c407051`, Apache-2.0).
- **12 (C2) and 15 (C4)** are our code, after vektorprime's analysis in qwen38-flash-next-pp2 (mtp-spec-decode, "bug 3", and the PP align-skew notes).
- **14 (C3) is not included.** Our production tree also contains it: draft shaping adapted from vektorprime's `VLLM_SPEC_DRAFT_TOPKP` / `VLLM_SPEC_DRAFT_TAU` work in qwen38-flash-next-pp2 (mtp-spec-decode). That repository states no licence, so we do not redistribute it. It is off in production. The same idea is in the open vllm#56724.
- **10 (K1), 16–18 (D3, D3b, D3c)** are ours.

## Tools

- `tools/models/quantize_lm_head.py`: from AlexBrandes, [cmp170hx-qwen38-flash-next](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next), MIT, copyright (c) 2026 Alex Brandes. The licence text is in `tools/models/quantize_lm_head.LICENSE`.
- `bench/night4/T1-mtp-drafter-check.py` is an independent implementation of the idea of azampatti's `mtp_drafter_check.sh` ([Qwen3.8-Flash-Next-Int4-FAST](https://github.com/azampatti/Qwen3.8-Flash-Next-Int4-FAST), MIT). `T3-spec-kernel-exactness.py` uses test distributions from vektorprime's mtp-spec-decode tests.
- Not included, only downloaded by the scripts: hampsonw's INT4 MTP-expert delta for Intel AutoRound, ukisai's BF16 release of Swift 1.5, and the incoai DFlash2 drafter.

## Model and template

- **Qwen3.8-Flash-Next** by the Qwen team, under the Qwen Community License 1.0.
- **Swift 1.5** (ukisai), the W4A16 AWQ checkpoint served here, under its own licence.
- **The Qwen Asgard chat template**: [leonbollerup/qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja), Qwen Community License 1.0, with credits to nazeshinjite, froggeric and chromix.

## Hardware tools

- **polyuij42-del**: the CMP 170HX VBIOS tooling (`check_rom.py`, `verify_170hx.sh`), called by the scripts in `vbios/`. Not included.
- **NVIDIA `nvflash`**: not included.
- **cachenetics/170tune**: the HBM clock research.
- **The CMP unlocker** open-driver project: the 64 GiB and SM unlock.

## Benchmarks

- **BFCL v4** (gorilla project, Apache-2.0): checker code vendored in `qeval/qeval/bfcl_vendor/` with its licence.
- **EvalPlus** (HumanEval+, MBPP+), **LiveCodeBench**: their data is fetched by `qeval/tools/prepare_data.py` and is not included.

Not affiliated with NVIDIA, the Qwen team, ukisai, Intel or the vLLM project.
