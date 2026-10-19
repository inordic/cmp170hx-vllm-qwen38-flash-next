# Qwen3.8-Flash-Next on 2× CMP 170HX with vLLM: the Asgard build

This build serves Swift 1.5 (Qwen3.8-Flash-Next, W4A16 AWQ) on two NVIDIA CMP 170HX mining cards, pipeline-parallel in vLLM, for agentic coding. The headline numbers:

- **A 128K-token prompt is read in 15.7 s** (8,336 tok/s).
- **Code editing decodes at 251 tok/s.**
- **16 users share 504 tok/s.**
- **The context window is 262,144 tokens.**
- **The quality suite** scores BFCL v4 at 80.3%, HumanEval+ 95.7%, MBPP+ 82.5% and LiveCodeBench v6 58.3%.

Every number in this README comes from a file in [`results/`](results/), measured on the box described below.

Three things set this build apart:
- **The cards are cross-flashed to the 300 W VBIOS.** Both run at 1695/1728 MHz max, on PCIe Gen2 ×8 each.
- **The ~95 GiB PLE n-gram table is streamed from NVMe** with deep prefetch. It never sits in RAM, and the container has only 55 GB.
- **Every configuration choice was A/B-tested** against speed *and* a paired quality suite (BFCL, EvalPlus, LiveCodeBench, long context), overnight.

The build stands on [2dameneko's vLLM build](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX) and [iIIusi0n's PLE-SSD recipe](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx). This repository adds:
- **vLLM backports:** four upstream fixes for hybrid Mamba under PP, MTP and n-gram lookup;
- **a tuned production profile;**
- **the hardware route:** VBIOS, power and LXC passthrough;
- **the quality harness;**
- **the measurements** behind every decision;
- **the [Qwen Asgard](https://github.com/leonbollerup/qwen-asgard-jinja) chat template**, together with two vLLM patches that keep thinking on/off consistent.

## Speed

**Prefill.** One cold request, real text, exact prompt length, on the production profile ([`results/speed/exact-E5*.json`](results/speed/), [`exact-N3-prod.json`](results/speed/exact-N3-prod.json)):

| Prompt | Time to first token | Prefill tok/s |
|---:|---:|---:|
| 1K | 0.33 s | 3,167 |
| 4K | 0.83 s | 4,920 |
| 8K | 1.39 s | 5,914 |
| 16K | 2.34 s | 7,011 |
| 32K | 4.21 s | 7,788 |
| 64K | 8.03 s | 8,168 |
| 128K | **15.7 s** | **8,336** |

**Decode.** One user, with MTP 3 plus n-gram lookup ([`results/logs/night2-NIGHT-LOG.md`](results/logs/night2-NIGHT-LOG.md)):

| Workload | tok/s (median) |
|---|---:|
| Code editing (thinking off) | **251** |
| Code generation | 169 (mean acceptance length 3.12) |
| Prose, thinking on, temperature 0.6 | 137–155 from 1K to 64K of context |

**Concurrency.** 1K-token prompts, 2-minute runs, no errors ([`results/concurrency/`](results/concurrency/)):

| Users | Total tok/s | Median time to first token | p95 time to first token |
|---:|---:|---:|---:|
| 8 | 390 | 0.32 s | 0.68 s |
| 16 | 504 | 0.44 s | 0.94 s |
| 4 chat + 1 agent at 33K context | 163 | agent 4.8 s, chat 0.22 s | |

Multi-turn conversations reuse their prefix, thanks to `--prefix-match-unit 32` plus the fine-grained Mamba prefix cache:
- in the live feature check, the follow-up turn of a ~15K-token conversation completed in 0.8 s, against 4.5 s for the first turn;
- the multi-turn check hit the cache for 80% of the queried tokens.

## Quality

The settings are the model card's: temperature 1.0, top-p 0.95, top-k 20, reasoning effort medium. Results are scored per item; the harness is in [`qeval/`](qeval/) and the per-item scores are in [`results/quality/`](results/quality/).

BFCL, EvalPlus and long context are from the final production run (`tpl-v2final`); LiveCodeBench is from night 2 (`swift15`), with the same model and engine.

| Benchmark | Items | Score |
|---|---:|---:|
| BFCL v4 (11 categories, incl. multi-turn) | 1,040 × 2 | **80.3%** |
| HumanEval+ | 164 | 95.7% |
| MBPP+ | 378 | 82.5% |
| LiveCodeBench v6 (easy / medium / hard) | 175 | 58.3% (97.7 / 67.3 / 31.2) |
| Long context needles at 32K / 64K / 110K | 180 | 100% / 96.7% / 95.0% |

**What the A/B tests showed** (paired per item, bootstrap CIs; details in [`research/`](research/)):

| Question | Answer |
|---|---|
| Which checkpoint? | Swift 1.5 decodes code **13% faster** than Intel AutoRound (161 vs 142 tok/s) and **ties on code** (EvalPlus 86.0 vs 86.2, LCB 58.3 vs 58.3). Intel is **better at tool calls (+3.0 points BFCL, p=0.007) and long context (+3.9, p=0.016)**. cyankiwi AWQ is behind both. |
| Reasoning effort? | **medium.** xhigh costs **−5.2 BFCL points** (CI −7.8 … −2.8) because it makes wrong parallel-call counts, and coding scores stay equal at +46% tokens. |
| Chat template? | [Qwen Asgard](https://github.com/leonbollerup/qwen-asgard-jinja): **+1.4 BFCL points** over the model's own template (replicated), with nine robustness fixes, among them Claude Code tool search and control-token injection. |
| Temperature 0.6 or 1.0? | No measurable difference. |
| INT8 KV cache? | **Broken** together with prefix caching, MTP and `--prefix-match-unit 32`: scale factors are not copied on mid-block prefix hits. BF16 KV it is. |

## What runs

| | |
|---|---|
| Hardware | 2× NVIDIA CMP 170HX (GA100, sm_80), unlocked to 64 GiB and 74 SMs, **VBIOS 92.00.6D.00.0A (300 W)** on both. PCIe Gen2 ×8 each, no P2P. i7-8700K, 64 GB host RAM, NVMe |
| Host | Proxmox VE 9.1.1, kernel 6.17.2, NVIDIA open driver 610.57.04 (CMP unlocker DKMS), `iomem=relaxed` |
| Container | unprivileged LXC with both GPUs passed through, 12 cores, 55 GB RAM ([`config/host/lxc-container.conf`](config/host/lxc-container.conf)) |
| Engine | vLLM `a5a30471f` (0.29.1rc1.dev402) + iIIusi0n PLE-SSD recipe + 2dameneko patches (commit `56e1347`) + [our patches](patches/), torch 2.13 cu130 |
| Model | [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) (revision `598afc96`), 166 GB |
| Parallelism | pipeline parallel 2, split 24/24, card order `1,0` (chosen while card 1 was the slower 250 W card), expert parallel |
| PLE table | BF16 on NVMe through Linux AIO: 32 workers, I/O depth 512, prefetch 131,072 rows, 16 GB row cache |
| Speculative decoding | MTP, 3 tokens, with n-gram lookup before MTP (backport of vllm#60615) |
| Cache | BF16 KV, prefix caching with `--prefix-match-unit 32`, fine-grained Mamba prefix cache |
| Batching | up to 24 sequences, 4,096 batched tokens, async scheduling, extended CUDA-graph capture sizes |
| Chat | Qwen Asgard v1, `{"reasoning_effort":"medium"}`; OpenAI and Anthropic APIs, tool calls (`qwen3_xml`), reasoning parser `qwen3` |

The full profile is in [`config/qfn-prod.env`](config/qfn-prod.env), and the build steps are in [`BUILD.md`](BUILD.md).

## Repository layout

| Path | Contents |
|---|---|
| [`BUILD.md`](BUILD.md) | from bare cards to a running server, step by step |
| [`config/`](config/) | the production profile, systemd unit and logrotate; the host power unit, thermal guard and LXC config |
| [`patches/`](patches/) | our vLLM patches on top of the 2dameneko build, with manifests and the venv script |
| [`vbios/`](vbios/) | how both cards were cross-flashed to the 300 W VBIOS, and the wrapper scripts |
| [`bench/`](bench/) | speed and functional tests, soak tests, GPU self-test, the overnight template A/B |
| [`qeval/`](qeval/) | the quality harness: BFCL v4, EvalPlus, LiveCodeBench, long context; sandboxed and resumable |
| [`results/`](results/) | every measurement: speed, concurrency, quality (per item), tool adherence, hardware, night logs |
| [`research/`](research/) | the night reports, the model and alternatives research, the Strata history |

## Lessons from the way here

Each of these is backed by a measurement in [`research/`](research/) and [`results/`](results/):

- **The PLE read-ahead is the biggest prefill lever on this box.** Prefetch 131,072 with 32 workers, depth 512 and a 16 GB cache took 128K prefill from 5,936 to 8,336 tok/s. With 55 GB of RAM the ~95 GiB table can't live in the page cache, so a dedicated reader with deep prefetch is the way here. Builds with ~93 GB of RAM can mmap it instead.
- **Benchmarks with random tokens understate this build.** Unique n-grams miss the PLE row cache: 32K measures 4,005 tok/s with random tokens against 6,123 with real text, both before the read-ahead tuning.
- **The 250 W card limited the pipeline.** Cross-flashing it to the 300 W VBIOS gave +5% code decode and up to +8% prefill. Both cards now hold about 1455–1485 MHz under load.
- **The n-gram lookup before MTP** gives +15% on code editing (251 vs 219 tok/s) at no quality cost (tool adherence 98/100).
- **More speculative tokens did not help** with this checkpoint: MTP 4 needs `--prefix-match-unit 16` and was no faster than MTP 2.
- **vLLM beat Strata** (the GGUF engine we started with) on prefill and time to first token: 128K in 15.7 s instead of 23.7 s. Strata's patches and report are in [`research/strata/`](research/strata/).

## Credits

This build exists because of the following work:
- **2dameneko**: [vllm-Qwen3.8-Flash-Next-2xCMP170HX](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX), the vLLM build we start from (Apache-2.0).
- **iIIusi0n**: [qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx), the PLE-SSD offload recipe (Apache-2.0).
- **The vLLM project and the authors of the backported PRs:** #55601, #60210, #56734, #60615, and #54709 via 2dameneko.
- **ukisai**, for Swift 1.5. **Intel** and **cyankiwi**, for the alternative checkpoints we compared.
- **polyuij42-del**, for the CMP 170HX VBIOS tooling the flash wrappers call. **cachenetics/170tune**, for the HBM clock research. The **CMP unlocker** project, for the 64 GiB and SM unlock.
- **u/Prudent_Appearance71**, whose Reddit post started this.
- **The chat template:** nazeshinjite, froggeric and chromix (see [qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja)).

See [`NOTICE.md`](NOTICE.md) for licences.

## Licence

Our scripts, configs and documents are under MIT ([`LICENSE`](LICENSE)). The vLLM patches are under Apache-2.0, like vLLM. Third-party components keep their own licences; see [`NOTICE.md`](NOTICE.md). No model weights or firmware are included.

Not affiliated with NVIDIA, the Qwen team, ukisai, Intel or the vLLM project.
