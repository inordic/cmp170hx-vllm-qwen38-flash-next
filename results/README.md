# Results

Everything here was measured on the box described in the [main README](../README.md), from 13 to 17 October 2026.

## Layout

| Path | Contents |
|---|---|
| `speed/exact-<config>.json` | `perf_exact.py` runs: one cold request per size, real text at exact lengths. Each row holds `size`, `prompt`, `ttft`, `pp` (prompt / TTFT), `gen`, `tg`. `*-long.json` covers 64K and 128K |
| `concurrency/night1-conc.jsonl`, `night2-conc.jsonl` | `conc_stagger.py` runs. Each row is one scenario × config: total tok/s, TTFT median / p95 / max, per-stream decode, errors |
| `quality/<tag>/<effort>/<bench>.jsonl.gz` | per-item quality scores, **without model texts**: id, subset, repeat, seed, correct/score, tokens, latency, runaway, tool calls. `*.summary.json` and `*.meta.json` hold the run's summary and settings |
| `quality/compare-medium.md` | night 2: Swift 1.5 vs Intel AutoRound, paired |
| `quality/compare-templates.md` | night 3: chat-template variants vs chromix, paired |
| `tool-adherence/` | 100 tool-call cases, with and without n-gram lookup |
| `hardware/` | GPU self-test logs per card, and clocks/temperatures during a run |
| `profiles/night1/`, `profiles/night2/` | the exact serve profile behind every config name |
| `logs/night{1,2,3}-NIGHT-LOG.md` | the overnight logs, step by step, with timestamps |

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
| **`E5`** | E1c + n-gram lookup | **production** |
| `N3-prod` | production after the night-3 template deploy | a speed check, 98–103% of E5 |

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

**Reproducing the tables.** `run_eval.py compare` in [`../qeval/`](../qeval/) builds the comparison tables. [`../bench/night3-template-ab/analyze_templates.py`](../bench/night3-template-ab/analyze_templates.py) gives the item-clustered bootstrap and sign test, but it reads the full uncompressed results, which aren't in this repo. Unzip these per-item scores first and point it at them.
