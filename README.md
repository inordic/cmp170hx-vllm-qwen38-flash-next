# Qwen3.8-Flash-Next on 2× CMP 170HX with vLLM: the Asgard build

This build serves Swift 1.5 (Qwen3.8-Flash-Next, W4A16 AWQ) on two NVIDIA CMP 170HX mining cards, pipeline-parallel in vLLM, for agentic coding. Since 2026-10-20 it drafts 4 tokens per step with MTP, n-gram lookup and an int4 copy of the output head (night 4). The headline numbers:

- **Code generation decodes at 202 tok/s** (median 201.6, mean acceptance length 3.62), up from 169 with the previous standard. *New, night 4; production, default clocks* ([`decode-N4-prod.json`](results/speed/decode-N4-prod.json)).
- **Code editing decodes at 304 tok/s**, against 248 for the previous standard in the same A/B. *New, nights 4 and 5; clocks locked at 1440 MHz* ([`edit-D4p16.txt`](results/speed/night4/edit-D4p16.txt), [`edit-R.txt`](results/speed/night5/edit-R.txt)).
- **A 128K-token prompt is read in 15.9 s** (8,231 tok/s). Prefill does not depend on the drafter; night 2 measured 15.7 s at default clocks. *New, night 5; clocks locked at 1440 MHz* ([`exact-n5-Rmatrix.json`](results/speed/exact-n5-Rmatrix.json)).
- **16 users share 511 tok/s.** *New, night 5; clocks locked at 1440 MHz* ([`night5-conc.jsonl`](results/concurrency/night5-conc.jsonl)).
- **The context window is 262,144 tokens.**
- **The quality suite** scores BFCL v4 at 80.3%, HumanEval+ 95.7%, MBPP+ 82.5% and LiveCodeBench v6 58.3% (nights 2 and 3, [`results/quality/`](results/quality/)). Re-run on the new standard, BFCL, EvalPlus and long context stay within the noise of those numbers (nights 4 and 5, see [Quality](#quality)).

Every number in this README comes from a file in [`results/`](results/), measured on the box described below. Night 4 and 5 A/B runs had both cards **locked at 1440 MHz**, so that configs measured hours apart compare fairly; production checks and nights 1–3 ran at default clocks, where the cards hold about 1455–1485 MHz under load. Each number says which applies.

Three things set this build apart:
- **The cards are cross-flashed to the 300 W VBIOS.** Both run at 1695/1728 MHz max, on PCIe Gen2 ×8 each.
- **The ~95 GiB PLE n-gram table is streamed from NVMe** with deep prefetch. It never sits in RAM, and the container has only 55 GB.
- **Every configuration choice was A/B-tested** against speed *and* a paired quality suite (BFCL, EvalPlus, LiveCodeBench, long context), overnight.

The build stands on [2dameneko's vLLM build](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX) and [iIIusi0n's PLE-SSD recipe](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx). This repository adds:
- **vLLM patches:** four upstream fixes for hybrid Mamba under PP, MTP and n-gram lookup; the int4 MTP draft head; correctness fixes for speculative decoding; the INT8 KV scale fix; the DFlash2 feature tap (see [`patches/`](patches/); one inactive production patch, C3, is not redistributed);
- **a tuned production profile;**
- **the hardware route:** VBIOS, power and LXC passthrough;
- **the quality harness;**
- **the measurements** behind every decision, and the tools that made them;
- **the model build tools** for the draft head and the quality builds tested in night 4;
- **the [Qwen Asgard](https://github.com/leonbollerup/qwen-asgard-jinja) chat template**, together with two vLLM patches that keep thinking on/off consistent.

## Speed

**Prefill.** One cold request, real text, exact prompt length. The current standard is from night 5 ([`exact-n5-Rmatrix.json`](results/speed/exact-n5-Rmatrix.json), clocks locked at 1440 MHz). The previous standard is from nights 2 and 3 ([`exact-E5*.json`](results/speed/), [`exact-N3-prod.json`](results/speed/exact-N3-prod.json), default clocks):

| Prompt | Time to first token | Prefill tok/s | Previous standard, tok/s |
|---:|---:|---:|---:|
| 1K | 0.30 s | 3,459 | 3,167 |
| 4K | 0.80 s | 5,168 | 4,920 |
| 8K | 1.32 s | 6,227 | 5,914 |
| 16K | 2.40 s | 6,824 | 7,011 |
| 32K | 4.22 s | 7,787 | 7,788 |
| 64K | 8.04 s | 8,157 | 8,168 |
| 128K | **15.9 s** | **8,231** | 8,336 |

Production after the night-4 deploy, at default clocks, gave 4,944 tok/s at 4K and 7,838 at 32K, 100–101% of night 3 ([`exact-N4-prod.json`](results/speed/exact-N4-prod.json)).

**Decode.** One user. The current standard drafts with MTP 4, n-gram lookup and the int4 draft head; the previous one with MTP 3 and n-gram lookup:

| Workload | Current standard | Previous standard |
|---|---:|---:|
| Code generation, production, default clocks | **201.6** (acceptance 3.62) [`decode-N4-prod.json`](results/speed/decode-N4-prod.json) | 169 (3.12), night 2 [log](results/logs/night2-NIGHT-LOG.md) |
| Code generation, A/B at 1440 MHz | 193.6 (3.51) [`decode-n4-D4p16.json`](results/speed/decode-n4-D4p16.json); 191.1 and 200.8 in night 5 [`decode-n5-R.json`](results/speed/decode-n5-R.json), [`decode-n5-Rmatrix.json`](results/speed/decode-n5-Rmatrix.json) | 168.5 (3.12) [`decode-n4-T0.json`](results/speed/decode-n4-T0.json) |
| Code editing, thinking off, A/B at 1440 MHz | **303.7** [`edit-D4p16.txt`](results/speed/night4/edit-D4p16.txt); 304.2 in night 5 [`edit-R.txt`](results/speed/night5/edit-R.txt) | 248.0 [`edit-T0.txt`](results/speed/night4/edit-T0.txt); 251 at default clocks in night 2 |
| Summarising prose, thinking on, from 1K to 128K of context | 148–176, night 5, 1440 MHz [`exact-n5-Rmatrix.json`](results/speed/exact-n5-Rmatrix.json) | 137–155 from 1K to 64K, night 2 [`exact-E5.json`](results/speed/exact-E5.json) |

All decode tests sample at temperature 0.6. Code editing has not been measured on the current standard at default clocks yet.

**Concurrency.** 1K-token prompts, 2.5-minute runs, no errors. Current standard: night 5, clocks locked at 1440 MHz ([`night5-conc.jsonl`](results/concurrency/night5-conc.jsonl)). Previous standard: night 2, default clocks ([`night2-conc.jsonl`](results/concurrency/night2-conc.jsonl)).

| Users | Total tok/s | Median time to first token | p95 time to first token | Previous standard |
|---:|---:|---:|---:|---|
| 8 | 404 | 0.34 s | 0.62 s | 390 tok/s, 0.32 / 0.68 s |
| 16 | **511** | 0.43 s | 0.79 s | 504 tok/s, 0.44 / 0.94 s |
| 4 chat + 1 agent at 33K context | 168 | agent 4.96 s, chat 0.21 s | chat 2.25 s | 163 tok/s, agent 4.8 s, chat 0.21 s |

The chat p95 in the mixed run varies from run to run: the same profile gave 0.45 s in night 4 ([`night4-conc.jsonl`](results/concurrency/night4-conc.jsonl), `n4-BP_off`).

Multi-turn conversations reuse their prefix, thanks to `--prefix-match-unit 16` plus the fine-grained Mamba prefix cache:
- after the night-4 deploy, the multi-turn check hit the cache for 79% of the queried tokens ([night-4 log](results/logs/night4-NIGHT-LOG.md));
- follow-up turns of ~20K-token conversations start in 0.16–0.20 s, against 2.6–3.0 s for the first turn ([`multi-D4p16.txt`](results/checks/night4/multi-D4p16.txt), 1440 MHz).

## Quality

The settings are the model card's: temperature 1.0, top-p 0.95, top-k 20, reasoning effort medium. Results are scored per item; the harness is in [`qeval/`](qeval/) and the per-item scores are in [`results/quality/`](results/quality/).

These are the full numbers of the previous standard. BFCL, EvalPlus and long context are from the final night-3 run (`tpl-v2final`); LiveCodeBench is from night 2 (`swift15`), with the same model and engine.

| Benchmark | Items | Score |
|---|---:|---:|
| BFCL v4 (11 categories, incl. multi-turn) | 1,040 × 2 | **80.3%** |
| HumanEval+ | 164 | 95.7% |
| MBPP+ | 378 | 82.5% |
| LiveCodeBench v6 (easy / medium / hard) | 175 | 58.3% (97.7 / 67.3 / 31.2) |
| Long context needles at 32K / 64K / 110K | 180 | 100% / 96.7% / 95.0% |

**On the current standard.** The target model verifies every drafted token, so the drafter can change speed but not what the model may answer. We re-ran the suite anyway, the same night as the A/B:

| Benchmark | Night 3 (`tpl-v2final`) | Night 4 (`n4-ref`) | Night 5 (`n5-R`) |
|---|---:|---:|---:|
| BFCL v4 | 80.3% (2,080) | 79.6% (2,080) | 80.5% (3,120) |
| EvalPlus (HumanEval+ / MBPP+) | 86.5% (95.7 / 82.5) | 86.0% (93.9 / 82.5) | 85.8% |
| Long context | 97.2% | 98.3% | 95.0% |

Night 4: [`results/quality/n4-ref/`](results/quality/n4-ref/medium/). Night 5: so far only the summary line in the [night-5 log](results/logs/night5-NIGHT-LOG.md); its per-item files are not published yet. BFCL's 95% interval is about ±1.7 points. Long context ranges from 95.0% to 98.3% across the three runs: 9 against 3 misses out of 180, which is not significant at that size (Fisher's exact test, p = 0.14).

**What the A/B tests showed** (paired per item, bootstrap CIs; details in [`research/`](research/)):

| Question | Answer |
|---|---|
| Which checkpoint? | Swift 1.5 decodes code **13% faster** than Intel AutoRound (161 vs 142 tok/s) and **ties on code** (EvalPlus 86.0 vs 86.2, LCB 58.3 vs 58.3). Intel is **better at tool calls (+3.0 points BFCL, p=0.007) and long context (+3.9, p=0.016)**. cyankiwi AWQ is behind both. |
| Reasoning effort? | **medium.** xhigh costs **−5.2 BFCL points** (CI −7.8 … −2.8) because it makes wrong parallel-call counts, and coding scores stay equal at +46% tokens. |
| Chat template? | [Qwen Asgard](https://github.com/leonbollerup/qwen-asgard-jinja): **+1.4 BFCL points** over the model's own template (replicated), with nine robustness fixes, among them Claude Code tool search and control-token injection. |
| Temperature 0.6 or 1.0? | No measurable difference. |
| INT8 KV cache? | **Fixed in night 4** by patch 10 (K1). The INT8 scales were not copied when a prefix hit ended inside a block, which broke follow-ups together with prefix caching, MTP and a fine `--prefix-match-unit`. With K1 every check passes and the KV pool grows by 71%. Production still runs BF16 KV until night 5's test. |

## Night 4: what changed and what we found

Night 4 (2026-10-19/20) tested drafting, quality builds and the INT8 KV fix. The A/B ran with clocks locked at 1440 MHz; the report is [`research/2026-10-20-night4-drafters-and-quality-builds.md`](research/2026-10-20-night4-drafters-and-quality-builds.md) and the log [`results/logs/night4-NIGHT-LOG.md`](results/logs/night4-NIGHT-LOG.md).

- **Deployed: the int4 draft head + MTP 4 + `--prefix-match-unit 16`.**
  - Code decode 193.6 against 168.5 tok/s (+15%), code edit 303.7 against 248.0 (+22%), acceptance length 3.51 against 3.12 ([`select.txt`](results/speed/night4/select.txt)).
  - In production, at default clocks: 201.6 against 169 tok/s (+19%).
  - The drafter's own W4A16 `lm_head` ([patch 08](patches/)) makes each draft step cheaper, and that is what made a fourth draft token pay. MTP 4 needs `--prefix-match-unit 16`, because with 4 drafts the attention block is 1616 tokens; MTP 5 does not start.
  - The deploy passed the gate (production, functional and multi-turn checks, 29/29 live features) and a quality guard against night 3.
- **Swift-A16 (BF16 attention) is the quality option.** BFCL 82.1% against 79.6% (+2.6 points) and long context 99.4% against 98.3% (+1.1), with EvalPlus 85.8% against 86.0% ([`n4-A16`](results/quality/n4-A16/medium/)). It costs 10% of code decode (173.7 tok/s, [`decode-n4-A16.json`](results/speed/decode-n4-A16.json)) and 14% of the KV pool, and is still faster than Intel AutoRound (160.8 tok/s, [`decode-n4-I_best.json`](results/speed/decode-n4-I_best.json)). Build tools: [`tools/models/`](tools/models/).
- **Q6-class INT8 experts gain nothing over A16.** Swift-Q6x (A16 plus INT8 experts in 16 layers) scored BFCL 81.5% (1,264 items, time budget), EvalPlus 86.3% and long context 100% ([`n4-Q6X`](results/quality/n4-Q6X/medium/)), at 4% less decode and 39% less KV pool than A16. The quality gain comes from attention precision, not expert precision.
- **INT8 KV is fixed by patch K1.** The control without the scale copy failed 10 of 28 cache-hit turns; with K1 every check passed ([`results/checks/night4/`](results/checks/night4/), `K8*`). KV pool 1,433,217 tokens against 838,024 (+71%) for 1.6% less code decode ([`decode-n4-K8.json`](results/speed/decode-n4-K8.json)), long context unchanged at 98.3% ([`n4-K8`](results/quality/n4-K8/medium/)). The churn test only reached 17% KV usage; night 5 tests a full pool.
- **DFlash2 runs on our stack, but loses to MTP 4 + n-gram lookup.** With patches 16–18 the DFlash2 drafter reaches acceptance 3.02 with 3 drafts and 3.31 with 4. Against the deployed setup, 4 drafts are 2% slower on code and 24% slower on code edits ([`decode-n4-DF4.json`](results/speed/decode-n4-DF4.json), [`edit-DF4.txt`](results/speed/night4/edit-DF4.txt)), because vLLM allows n-gram lookup only with MTP.
- **Probabilistic drafting shows no bias, and its speed gain is not settled.**
  - The output-distribution test against no drafting gave TV 0.057 (letters) and 0.041 (digits), the same as greedy drafting (0.054 / 0.039) and inside the noise floor ([`dist-*.json.gz`](results/checks/night4/), night-4 log). BFCL 81.3% on 1,040 items ([`n4-PRn`](results/quality/n4-PRn/medium/)).
  - Decode at 8K and 32K of context rose by 19% and 13% (3 runs of 256 tokens each) ([`exact-n4-PRn.json`](results/speed/exact-n4-PRn.json) against [`exact-n4-D4p16.json`](results/speed/exact-n4-D4p16.json)), but code decode (−3%) and code edits (±0) did not. All of these ran at temperature 0.6. The night-4 report attributes the gain to T=1, the model's recommended sampling; a T=1 speed comparison is still owed.
- **Terse answers keep the score at fewer tokens.** With `terse: true`, BFCL 81.5% and EvalPlus 85.8% ([`n4-terse`](results/quality/n4-terse/medium/)) on 16% fewer tokens for BFCL (354 against 421) and 43% fewer for EvalPlus (536 against 933). EvalPlus's median latency halves (5.0 s against 9.9 s).
- **Not adopted:** batch 3296 or 4896 tokens and the PP split 26/22 (no prefill gain; 26/22 gives +20% KV pool at −3% speed); the bounded prefill of patch 09 (chat p95 5.6 s against 0.45 s under mixed load); INT4 MTP experts (+26% KV pool for −1.1% decode).
- **Correctness checks** on the new tree: no dead drafter in 21 parallel streams, 24/24 fresh-prompt probes, unbiased drafting kernels, 72/72 prefix-bleed answers, no preemption on a 131K prompt ([`results/checks/night4/`](results/checks/night4/), [`patches/README.md`](patches/README.md)).

**Next: night 5** (2026-10-20/21, running). It compares the current standard (`R`, whose performance matrix is above) with three candidates that each add one piece (C1–C3 are night-5 candidate names, not patches):
- C1 = Swift-A16 + INT8 KV, after the K1 checks, this time including a full pool;
- C2 = C1 + probabilistic drafting;
- C3 = C2 + terse as the template default.

Every build gets the full quality suite at T=1 the same night. The speed tools still sample at temperature 0.6, so a decode-speed comparison at T=1 stays open.

## What runs

| | |
|---|---|
| Hardware | 2× NVIDIA CMP 170HX (GA100, sm_80), unlocked to 64 GiB and 74 SMs, **VBIOS 92.00.6D.00.0A (300 W)** on both. PCIe Gen2 ×8 each, no P2P. i7-8700K, 64 GB host RAM, NVMe |
| Host | Proxmox VE 9.1.1, kernel 6.17.2, NVIDIA open driver 610.57.04 (CMP unlocker DKMS), `iomem=relaxed` |
| Container | unprivileged LXC with both GPUs passed through, 12 cores, 55 GB RAM ([`config/host/lxc-container.conf`](config/host/lxc-container.conf)) |
| Engine | vLLM `a5a30471f` (0.29.1rc1.dev402) + iIIusi0n PLE-SSD recipe + 2dameneko patches (commit `56e1347`) + [our patches 01–18](patches/) (14, C3, is in production but not published; it is off), torch 2.13 cu130 |
| Model | [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) (revision `598afc96`), 166 GB |
| Parallelism | pipeline parallel 2, split 24/24, card order `1,0` (chosen while card 1 was the slower 250 W card), expert parallel |
| PLE table | BF16 on NVMe through Linux AIO: 32 workers, I/O depth 512, prefetch 131,072 rows, 16 GB row cache |
| Speculative decoding | MTP, **4 tokens**, with n-gram lookup before MTP (backport of vllm#60615). The drafter uses an int4 (W4A16) copy of the `lm_head` (patch 08); the target verifies with its BF16 head |
| Cache | BF16 KV, prefix caching with `--prefix-match-unit 16`, fine-grained Mamba prefix cache |
| Batching | up to 24 sequences, 4,096 batched tokens, async scheduling, extended CUDA-graph capture sizes |
| Chat | Qwen Asgard v1, `{"reasoning_effort":"medium"}`; OpenAI and Anthropic APIs, tool calls (`qwen3_xml`), reasoning parser `qwen3` |

The full profile is in [`config/qfn-prod.env`](config/qfn-prod.env), and the build steps are in [`BUILD.md`](BUILD.md).

## Repository layout

| Path | Contents |
|---|---|
| [`BUILD.md`](BUILD.md) | from bare cards to a running server, step by step |
| [`config/`](config/) | the production profile, systemd unit and logrotate; the host power unit, thermal guard and LXC config |
| [`patches/`](patches/) | our vLLM patches 01–18 (without 14) on top of the 2dameneko build, with manifests, notes and the venv script |
| [`tools/models/`](tools/models/) | the int4 draft head, the INT4 MTP experts, Swift-A16 and Swift-Q6x builds, and a weight-budget tool |
| [`vbios/`](vbios/) | how both cards were cross-flashed to the 300 W VBIOS, and the wrapper scripts |
| [`bench/`](bench/) | speed and functional tests, soak tests, GPU self-test, the overnight template A/B, the night-4 correctness checks and A/B |
| [`qeval/`](qeval/) | the quality harness: BFCL v4, EvalPlus, LiveCodeBench, long context; sandboxed and resumable |
| [`results/`](results/) | every measurement: speed, concurrency, quality (per item), checks, tool adherence, hardware, profiles, night logs |
| [`research/`](research/) | the night reports, the model and alternatives research, the Strata history |

## Lessons from the way here

Each of these is backed by a measurement in [`research/`](research/) and [`results/`](results/):

- **The PLE read-ahead is the biggest prefill lever on this box.** Prefetch 131,072 with 32 workers, depth 512 and a 16 GB cache took 128K prefill from 5,936 to 8,336 tok/s. With 55 GB of RAM the ~95 GiB table can't live in the page cache, so a dedicated reader with deep prefetch is the way here. Builds with ~93 GB of RAM can mmap it instead.
- **Benchmarks with random tokens understate this build.** Unique n-grams miss the PLE row cache: 32K measures 4,005 tok/s with random tokens against 6,123 with real text, both before the read-ahead tuning.
- **The 250 W card limited the pipeline.** Cross-flashing it to the 300 W VBIOS gave +5% code decode and up to +8% prefill. Both cards now hold about 1455–1485 MHz under load.
- **The n-gram lookup before MTP** gives +15% on code editing (251 vs 219 tok/s) at no quality cost (tool adherence 98/100).
- **More speculative tokens pay only with a cheap draft step.** Before, MTP 4 needed `--prefix-match-unit 16` and was no faster than MTP 2. In night 4, with the drafter's `lm_head` in int4, MTP 4 gave +15% on code and +22% on code edits over MTP 3.
- **On this checkpoint, quality sits in attention precision.** BF16 attention (Swift-A16) gained 2.6 BFCL points; INT8 experts on top added nothing.
- **Lock the clocks for overnight A/B runs.** Configs measured hours apart only compare fairly at a fixed clock; we lock both cards at 1440 MHz, close to what they hold under load.
- **vLLM beat Strata** (the GGUF engine we started with) on prefill and time to first token: 128K in 15.7 s instead of 23.7 s. Strata's patches and report are in [`research/strata/`](research/strata/).

## Credits

This build exists because of the following work:
- **2dameneko**: [vllm-Qwen3.8-Flash-Next-2xCMP170HX](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX), the vLLM build we start from (Apache-2.0).
- **iIIusi0n**: [qwen3.8-flash-next-cmp170hx](https://github.com/iIIusi0n/qwen3.8-flash-next-cmp170hx), the PLE-SSD offload recipe (Apache-2.0).
- **The vLLM project and the authors of the backported PRs:** #55601, #60210, #56734, #60615, #58784, and #54709 via 2dameneko.
- **AlexBrandes**: the int4 MTP draft head and the bounded mixed prefill (patches 08 and 09, from AlexBrandes/vllm, Apache-2.0), and `quantize_lm_head.py` ([cmp170hx-qwen38-flash-next](https://github.com/AlexBrandes/cmp170hx-qwen38-flash-next), MIT).
- **vektorprime** (qwen38-flash-next-pp2): the speculative-decoding analyses behind patches 12 and 15 and the test distributions of T3. **xris99**: the Mamba block-alignment fix behind patch 13 (Apache-2.0). **azampatti**: the dead-drafter check behind T1. **DominikBucko**: the 131K preemption canary.
- **ukisai**, for Swift 1.5 and its BF16 release. **Intel** and **cyankiwi**, for the alternative checkpoints we compared. **incoai**, for the DFlash2 drafter, and **hampsonw**, for the INT4 MTP-expert delta for Intel.
- **polyuij42-del**, for the CMP 170HX VBIOS tooling the flash wrappers call. **cachenetics/170tune**, for the HBM clock research. The **CMP unlocker** project, for the 64 GiB and SM unlock.
- **u/Prudent_Appearance71**, whose Reddit post started this.
- **The chat template:** nazeshinjite, froggeric and chromix (see [qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja)).

See [`NOTICE.md`](NOTICE.md) for licences.

## Licence

Our scripts, configs and documents are under MIT ([`LICENSE`](LICENSE)). The vLLM patches are under Apache-2.0, like vLLM. Third-party components keep their own licences; see [`NOTICE.md`](NOTICE.md). No model weights or firmware are included.

Not affiliated with NVIDIA, the Qwen team, ukisai, Intel or the vLLM project.
