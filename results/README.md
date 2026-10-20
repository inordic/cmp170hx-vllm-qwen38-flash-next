# Results

Everything here was measured on the box described in the [main README](../README.md), from 13 to 20 October 2026.

**Clocks.** Nights 1–3 and the production checks after a deploy (`*-N3-prod`, `*-N4-prod`) ran at the cards' default clocks; under load they hold about 1455–1485 MHz. The night-4 A/B runs and the night-5 reference `R` ran with both cards **locked at 1440 MHz**, so that configs measured hours apart compare fairly. Compare locked numbers with locked numbers.

## Layout

| Path | Contents |
|---|---|
| `speed/exact-<config>.json` | `perf_exact.py` runs: one cold request per size, real text at exact lengths. Each row holds `size`, `prompt`, `ttft`, `pp` (prompt / TTFT), `gen`, `tg`. `*-long.json` covers 64K and 128K. Night 4 adds `exact-n4-<config>.json` and `exact-N4-prod.json`; night 5 adds `exact-n5-R.json` and `exact-n5-Rmatrix.json` (the full 1K–128K matrix) |
| `speed/decode-<config>.json` | `decode_code.py` runs (nights 4 and 5): 12 code-generation answers of 768 tokens at temperature 0.6. `rows` holds `gen` and `tg` per answer, `acceptance_length` the server's mean acceptance length. Earlier nights' code-decode numbers are in their night logs |
| `speed/night4/`, `speed/night5/` | per config: `res-<config>.json` (the summary `select.py` reads: median prefill and decode per size, code decode, acceptance, code-edit decode, func/multiturn pass) and `edit-<config>.txt` (code-edit decode with thinking off and low, temperature 0.6, the only record of it); `select.txt` and `pick-*.txt` are the night-4 deploy and batch/split picks |
| `concurrency/night{1,2,4,5}-conc.jsonl` | `conc_stagger.py` runs. Each row is one scenario × config: total tok/s, TTFT median / p95 / max, per-stream decode, errors. Night 4: the bounded-prefill A/B (`n4-BP_*`); night 5: the reference `n5-R-*` with 8 and 16 users |
| `quality/<tag>/<effort>/<bench>.jsonl.gz` | per-item quality scores, **without model texts**: id, subset, repeat, seed, correct/score, tokens, latency, runaway, tool calls. `*.summary.json` and `*.meta.json` hold the run's summary and settings |
| `quality/compare-medium.md` | night 2: Swift 1.5 vs Intel AutoRound, paired |
| `quality/compare-templates.md` | night 3: chat-template variants vs chromix, paired |
| `tool-adherence/` | 100 tool-call cases, with and without n-gram lookup |
| `hardware/` | GPU self-test logs per card, and clocks/temperatures during a run |
| `profiles/night1/`, `profiles/night2/`, `profiles/night4/`, `profiles/night5/` | the exact serve profile behind every config name. The night-4/5 profiles were made by [`mkprof.py`](../bench/night4/mkprof.py); its arguments are in each file's first line |
| `checks/night4/`, `checks/night5/` | the correctness checks: T1 dead drafter, T2 fresh-prompt probe, T3 kernel exactness, T4 prefix-bleed soak, the K1 INT8-KV checks (`K8*.txt` / `.json`), greedy identity (`greedy-*.json`), the output-distribution samples (`dist-*.json.gz`; gunzip them into `/root/tune/night4` before `dist_test.py compare`), the deploy's feature check, and `func-*` / `multi-*` per config |
| `logs/night{1,2,3,4}-NIGHT-LOG.md` | the overnight logs, step by step, with timestamps |
| `logs/night5-NIGHT-LOG.md` | **a snapshot** of the night-5 log taken at 21:15 UTC on 2026-10-20, while the night was still running. It covers the reference `R` only |

## Config names

| Name | What changed | Notes |
|---|---|---|
| `cyankiwi`, `swift`, `intel` | night 1 model comparison on one config | Swift won on speed at equal code quality |
| `L0`–`L5`, `O0`/`O1` | night 1 ladder: sequences/capture sizes, chunk size, KV offload | **L1** (24 sequences + extended capture sizes) won; KV offload was rejected |
| `postflash`, `swift-300w` | after the VBIOS cross-flash / with 300 W on both cards | |
| `split2424`, `s2424bf16`, `swap` | pipeline split, KV dtype and card order | 24/24 + BF16 KV + order `1,0` |
| `E0` | the experimental tree: backports 01–04, n-gram off | passed the full gate and soak |
| `E1`, `E1b`, **`E1c`** | PLE read-ahead: cache 4/16 GB, prefetch, then 32 workers / depth 512 / prefetch 131,072 / 16 GB | E1c was the best PLE setting |
| `E2` | CUDA-graph capture sizes | rejected (8K slower) |
| `E3` | n-gram lookup before MTP | code edit +16% |
| `E4` | INT8 KV without `--prefix-match-unit 32` / fine-grained | follow-ups correct, which confirms the INT8 scale-copy bug |
| **`E5`** | E1c + n-gram lookup | production from night 2 to night 4 |
| `N3-prod` | production after the night-3 template deploy | a speed check, 98–103% of E5 |

**Night 4** (prefix `n4-` in `speed/` and `concurrency/`; every config on the production n4 tree, i.e. patches 01–18 including the unpublished 14 (C3, off), port 8008, clocks locked at 1440 MHz):

| Name | What changed | Notes |
|---|---|---|
| `T0` | the night-3 production settings (MTP 3, BF16 draft head, `--prefix-match-unit 32`) | the A/B baseline |
| `NOSPEC` | no speculative decoding | reference for the distribution and greedy tests |
| `D3` | int4 draft head (patch 08) | code +6%, edit +4% |
| `D4`, `D5`, `D5p16` | MTP 4 / 5 with the int4 head | did not start: MTP 4 needs `--prefix-match-unit 16`; MTP 5 fails on "QSA ring capacity 12 must divide the attention block size 1616" |
| **`D4p16`** | int4 head + MTP 4 + `--prefix-match-unit 16` | **the new standard**: code +15%, edit +22% against `T0` |
| `MB` | `D4p16` + INT4 MTP experts | −1.1% decode score, +26% KV pool; rejected |
| `PRn` | `D4p16` + `draft_sample_method: probabilistic` | bias test passed |
| `PRpad` | `PRn` with stock spec padding and the PP-skew probe | control for patches C2 and C4 |
| `SW0`, `B3296`, `B4896`, `S26` | `D4p16` reference; batch 3296 / 4896 tokens; PP split 26/22 | prefill sweeps; nothing beat `SW0` |
| `BP_off`, `BP_lpt`, `BP_d2` | prefill beside decodes: as is; stock `--long-prefill-token-threshold 3200`; patch D2 at 1600 | off is best |
| `A16`, `Q6X` | Swift-A16 (BF16 attention); A16 + INT8 experts in layers 0–10 and 43–47 | quality builds, see [`../tools/models/`](../tools/models/) |
| `I_best` | Intel AutoRound + its int4 head + INT4 MTP experts | speed only |
| `K8off`, `K8dbg`, `K8` | `D4p16` + INT8 KV: without the K1 scale copy (control), with debug checks, as production would run it | the control fails, K1 passes |
| `DF3`, `DF4`, `DFI` | DFlash2 drafter with 3 / 4 drafts on Swift, 3 on Intel | report only |
| `N4-prod` | production on port 8080 after the deploy, **default clocks** | code decode 201.6 tok/s |

**Night 5:** `R` (prefix `n5-`) is the production profile on port 8008 with clocks locked at 1440 MHz; `Rmatrix` is its full prefill matrix from 1K to 128K, and `n5-R-*` its concurrency runs.

## Quality tags

| Tag | Run |
|---|---|
| `swift15`, `intel` | night 2 full suite (chromix template), Swift 1.5 vs Intel AutoRound |
| `swift15-t06` | temperature 0.6 ablation |
| `tpl-v0` … `tpl-v4` | Qwen Asgard switch variants (night 3) |
| `tpl-chromix` | the chromix template (night 3) |
| `tpl-v2final` | **final production**: Qwen Asgard + vLLM patches A1/A2 |
| `tpl-v2-nopreserve` | `preserve_thinking: false` |
| `tpl-x-*` | the model's own template, a client that drops reasoning, and effort xhigh |
| `n4-ref` | night 4: the same-night reference on the new standard (`D4p16`), full suite |
| `n4-A16`, `n4-Q6X` | night 4: Swift-A16 and Swift-Q6x with the `D4p16` drafting. Q6x's BFCL stopped at its time budget (1,264 of 2,080 items) |
| `n4-PRn` | night 4: probabilistic drafting, BFCL once (1,040 items) |
| `n4-terse` | night 4: `chat_template_kwargs: {"terse": true}`, BFCL once and EvalPlus |
| `n4-K8` | night 4: INT8 KV with patch K1, long context only |

The night-4 tags also have the suite's `suite.log` (self-tests, probe and progress lines; no benchmark texts).

**Reproducing the tables.** `run_eval.py compare` in [`../qeval/`](../qeval/) builds the comparison tables. [`../bench/night3-template-ab/analyze_templates.py`](../bench/night3-template-ab/analyze_templates.py) gives the item-clustered bootstrap and sign test, but it reads the full uncompressed results, which aren't in this repo. Unzip these per-item scores first and point it at them.
