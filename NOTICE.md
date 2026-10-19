# Notice

This repository (scripts, configs, documents, the quality harness) is MIT-licensed; see [`LICENSE`](LICENSE). It builds on, and runs, the following work under their own terms. No model weights, datasets or firmware are included.

## Engine

- **vLLM** ([vllm-project/vllm](https://github.com/vllm-project/vllm)), Apache-2.0, at commit `a5a30471ff2bb7f0824f2da10e358af98d304472`. Not redistributed.
  - The patches in `patches/` modify vLLM source and contain short excerpts of it as anchors. They are provided under Apache-2.0.
  - Patches 01–04 are backports or ports of vllm-project/vllm PRs #55601, #60210, #56734 and #60615.
- **The PLE-SSD recipe**: iIIusi0n, [qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx), Apache-2.0.
- **The base build**: 2dameneko, [vllm-Qwen3.8-Flash-Next-2xCMP170HX](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX), Apache-2.0, commit `56e1347`. Their own NOTICE lists further work, including vllm#54709 and DominikBucko's overlays.

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
