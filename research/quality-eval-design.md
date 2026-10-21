# B. Overnight quality evaluation: Swift 1.5 AWQ vs Intel AutoRound vs cyankiwi AWQ

Prepared 2026-10-14 on the Mac. Nothing was sent to the server.
Folder: `research-tonight/quality/` (this file is in it).

## 1. Summary

- **The suite tonight:** 4 benchmarks, the same for each model.
  - EvalPlus HumanEval+ and MBPP+ (542 tasks).
  - BFCL v4 with native tool calls (1,040 cases, 100 of them multi-turn).
  - LiveCodeBench v6 (175 problems).
  - A long-context code test at 32K, 64K and 110K (180 questions).
- **Time per model:** the phase budgets add up to 126 minutes. Items already running finish after the budget, so expect about 2 h 15 to 2 h 25. The script stops for good at 150 minutes.
- **Settings:**
  - Sampling: temperature 1.0, top_p 0.95, top_k 20 (the Qwen and Swift recommendation).
  - `reasoning_effort=medium`.
  - Streaming, the same item order for every model, and fixed per-item seeds.
- **What it can tell apart.** The test pairs the models on the same items (exact McNemar test plus paired bootstrap):
  - EvalPlus, BFCL and the average over all four benchmarks resolve about **±2–2.5 points**.
  - LiveCodeBench and long context alone resolve only about **±5–7 points**.
  - ±3 % per benchmark is out of reach for LiveCodeBench in one night (see §3).
- **Runaways** (the answer hits the token limit) are reported separately, with confidence intervals, per benchmark and per subset. The report also shows **accuracy as if the token limit had been lower**, so last night's 12K limit can be compared directly.
- **Ready to run:**
  - A harness in plain Python (standard library only, Python 3.12).
  - All datasets already converted, plus checksums.
  - A code sandbox: model code runs as `nobody`, with no network if the container allows it, memory/CPU/file-size limits and a timeout.
  - Scripts for one model, for the effort sweep and for the whole night, plus a paired comparison report.
  - Everything was tested end to end on the Mac against a mock server. The sandbox's Linux-as-root path (setpriv/unshare as `nobody`) could not be tested on the Mac; the selftest checks it on the server first.

## 2. Plan for tonight (ranked)

| # | Benchmark | Items | Settings | Budget | Why |
|---|---|---|---|---|---|
| 1 | **BFCL v4, native tool calling** | 940 single-turn (100 simple, 100 multiple, 100 parallel, 100 parallel_multiple, 100 irrelevance, 100 live_simple, 150 live_multiple, 16 live_parallel, 24 live_parallel_multiple, 150 live_irrelevance) + 100 multi_turn_base | `tools=[...]` through vLLM's own tool parser (`qwen3_xml`), max 8,192 tokens per call, up to 20 steps per turn, earlier reasoning sent back | 38 min, c16 | The agentic path itself: parser, argument types, irrelevance, stateful multi-step use |
| 2 | **LiveCodeBench v6** | 175 (43 easy, 52 medium, 80 hard; Jan–Apr 2025; AtCoder 112, LeetCode 63) | LCB prompt, last code block, hidden tests (about 40 per problem, 6 s each), max 24,576 tokens | 50 min, c16 | The coding benchmark that separates models best, with real execution |
| 3 | **EvalPlus HumanEval+ / MBPP+** | 164 + 378 | EvalPlus prompt, differential testing against the reference solution, max 16,384 tokens | 18 min, c16 | Cheap and precise, links to last night's 41 tasks, gives the runaway rate at scale |
| 4 | **Long-context code test** (synthetic, in the style of RULER) | 180 = 18 code bases × 10 questions (32K / 64K / 110K tokens) | Questions on the same code base share the prompt prefix, and the first one warms the prefix cache. Max 8,192 tokens | 20 min, c6 | 10–130K contexts like the real use case. 4-bit quantization often hurts here first |
| 5 | (optional, another night) IFEval through lm-eval | 541 | see §5 | about 30 min | Instruction following; not core to agentic coding |

**Order:** EvalPlus → BFCL → LiveCodeBench → long context. The quick EvalPlus phase doubles as a health and runaway check.

**Items are shuffled by a hash of the item ID.** The order is identical for every model. When a budget runs out, the completed items are a random sample of the benchmark, and the comparison uses the items every model completed.

**Suggested night order:**
1. **Swift** (production; the reference for the effort sweep).
2. **Intel** (main challenger).
3. **cyankiwi** (already worse last night; it suffers first if time runs out).

**Timeline:** 3 × about 2 h 20 plus swaps (about 4–12 min each) ≈ **7.3 h**.
- The effort sweep on Swift (low and high, about 45 min each) adds about 1.5 h. Run it on a second night if the night is short.

## 3. Candidate benchmarks

Runtime assumes this server at medium effort:
- About 450 tok/s total decode at 16 parallel streams, 350 at 8 and about 150 for a single stream.
- About 6–7K tok/s prefill.

Tokens per item are output tokens including reasoning, and they are estimates. Last night HumanEval at medium had a median of about 800 tokens, and runaways ran to the limit. The Swift card reports **55 % fewer reasoning tokens** than the base model, so Swift will be cheaper per item.

| Benchmark | License | Size | Tokens per item (medium) | Runtime here, full set | n for ±3 % (one model, at typical p) | Sandbox |
|---|---|---|---|---|---|---|
| EvalPlus HumanEval+ | code and data Apache-2.0 (from HumanEval, MIT) | 164 | 1.0–1.5K | 6–9 min | 384 (p≈0.9), so all 164 are not enough alone | yes |
| EvalPlus MBPP+ | Apache-2.0 (from MBPP, CC-BY-4.0) | 378 | 0.8–1.2K | 11–17 min | 683 (p≈0.8) | yes |
| LiveCodeBench (code_generation_lite) | code MIT; dataset card says only "cc"; problems © LeetCode, AtCoder, Codeforces | v6 window 175 (v5+v6 342; all 1,055) | easy 2–4K, medium 6–12K, hard 15–30K+ (cap 24K) | 40–78 min for v6 | 896–1,067 (p≈0.5–0.7): not possible | yes (stdin programs, hidden tests up to 60 MB) |
| BFCL v4 | Apache-2.0 | single-turn 3,548; multi-turn 4×200; web search 100; memory 155 | single 0.3–0.8K; multi-turn 4–8K per entry | 1,040-case mix: 30–55 min | single-turn 384–545 (p≈0.85–0.9); multi-turn 1,067 (p≈0.5) | no (simulated APIs in-process) |
| τ²-bench | MIT | airline 50, retail 114, telecom 114 | 20–60K per task and trial, plus user-simulator tokens | hours per domain with 4 trials | not possible (n ≤ 114) | no, but needs a user-simulator LLM (default GPT-4.1), which confounds a quant comparison |
| RULER (NVIDIA) | Apache-2.0 | 13 tasks × lengths × samples | 0.3–2K output, 32–128K prefill each | full run ≈ 12 h of prefill | about 500 per length | no |
| Our RULER-style code test | haystack = Python stdlib (PSF); needles synthetic | 180 | 0.5–2K output; prefill shared per code base | 15–22 min | about 550 at p≈0.85 (180 gives ±5 %) | no |
| IFEval | Apache-2.0 | 541 | 1–2K | 20–40 min | 545 (p≈0.85): full set | no (checker needs nltk and langdetect) |
| GPQA-Diamond | CC-BY-4.0, **gated** (terms must be accepted on HF) | 198 | 5–10K | 37–74 min | not possible (±6.5 % at n=198) | no |
| MMLU-Pro | MIT | 12,032 | 1.5–3K | 55–110 min for a 1,000 subset | about 1,000 | no |
| Aider polyglot | aider Apache-2.0, Exercism MIT | 225 exercises in 6 languages, 2 tries | 10–20K | 85–170 min plus test time | not possible (±6.5 %) | yes: **Docker with 6 toolchains** |

**Formulas:**
- One model: n = 1.96² · p(1−p) / 0.03². That gives p=0.5 → 1,067; 0.7 → 896; 0.8 → 683; 0.9 → 384; 0.95 → 203.
- **A paired comparison needs far fewer items:** n ≈ 1.96² · d / 0.03², where d is the fraction of items where exactly one of the two models is right. That gives d=0.05 → 213; 0.10 → 427; 0.20 → 854.
- At temperature 1.0 with one sample per item, d is inflated by sampling noise: an item that either model solves with probability q is discordant 2q(1−q) of the time. So:
  - Saturated sets (EvalPlus, BFCL single-turn): d ≈ 0.06–0.12, giving ±2–2.5 points tonight.
  - LiveCodeBench, BFCL multi-turn and long context: d ≈ 0.15–0.3, giving ±5–10 points.

**Expected resolution tonight** (95 % half-width of the paired difference):

| Benchmark | Items | Assumed d | ± points |
|---|---|---|---|
| EvalPlus | 542 | 0.09 | 2.5 |
| BFCL | 1,040 | 0.12 | 2.1 |
| LiveCodeBench v6 | about 150–175 | 0.2 | 6.6–7.2 |
| Long context | 180 | 0.1 | 4.6 (a bit wider: the bootstrap resamples whole code bases) |
| Average of the 4 | | | about 2.2 |

## 4. Methodology

**Sampling: use temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence 0 for all three models.**
- This is the official Qwen3.8-Flash-Next thinking-mode setting (also in `generation_config.json`) and the Swift card's setting. Quantization should be judged where it is deployed.
- Last night's temperature 0.6 is off the recommended setting for this model generation. Lower temperature in thinking models goes with more repetition loops, so some of last night's runaways may come from it.
- `run_effort_sweep.sh T06=1` runs a temperature-0.6 HumanEval+ ablation on Swift (about 8 min) to check this directly.

**Seeds and repeats:**
- Every request gets the seed `seed_base + hash(bench, item, repeat)`, the same for all models. This makes reruns of the *same* model reproducible, apart from vLLM batch nondeterminism.
- Seeds do **not** make different weights comparable, so they are not a variance-reduction tool here.
- Repeats (`--repeats k`) reduce sampling noise per item. Tonight more items is worth more than repeats.
- If two models end up inside each other's CI on LiveCodeBench, give those two `--repeats 2` on a later night. The report then uses per-item mean scores and a sign-flip permutation test.

**Token limits and runaways:**
- Limits per call:

  | Benchmark | Limit |
  |---|---|
  | EvalPlus | 16K |
  | BFCL | 8K per call |
  | LiveCodeBench | 24K (32K in the effort sweep) |
  | Long context | 8K, with a ~111K prompt, under the 131K context |

- A runaway (`finish_reason=length` on any call) counts as a **failure** in the main score.
- Reported separately:
  - the runaway rate with a Wilson CI;
  - `runaway_no_answer` (cut off inside the reasoning);
  - `looping` (the last 8 KB compresses below 0.12 with zlib, i.e. repetition);
  - accuracy excluding runaways;
  - `acc_at_cap`: accuracy if the limit had been 2K/4K/8K/12K/16K/24K/32K. `acc_at_cap["12288"]` reproduces last night's limit.

**Fair comparison:**
- Same server configuration, chat template (chromix), tool parser, prompts, item order, seeds, sampling, token limits and concurrency.
- Results are compared only on items every model completed.
- Per benchmark, for each pair of models:
  - accuracy with Wilson CIs;
  - the difference with a paired bootstrap 95 % CI (10,000 resamples; long context resamples whole code bases);
  - the **exact McNemar p** with the discordant counts (A only right / B only right).
- An equal-weight average over the benchmarks with a combined CI.
- **Multiple comparisons:** 3 pairs × 4 benchmarks = 12 tests. Read p < 0.05 with a Holm correction, or simply require the averaged CI to exclude 0.
- **Decision rule, consistent with last night's deploy rules:**
  - Intel replaces Swift only if Intel's lead's averaged CI excludes 0, or BFCL/LiveCodeBench show p < 0.01.
  - Otherwise "equal within ±X points" is the result, and Swift wins on speed.

**Quality vs reasoning_effort:**
- `run_effort_sweep.sh` runs low and high on one model with a reduced suite: HumanEval+ 164, BFCL 350 (incl. 50 multi-turn), LiveCodeBench 80, long context 60.
- Those items are subsets of the main medium run (deterministic subsampling), so `compare results/swift15/{low,medium,high}` pairs them automatically.
- Every item records:
  - completion tokens and estimated reasoning tokens (completion tokens split by characters, since vLLM doesn't report the split);
  - TTFT;
  - **t_answer**: time to the first answer token, i.e. the end of thinking;
  - total latency.
- Tokens don't depend on load, but latency at c16 does. For honest single-user time-to-answer, `C1=1` runs 20 HumanEval+ items per effort at concurrency 1, or estimate time as TTFT + tokens / 150 tok/s.
- Use `acc_at_cap` to compare efforts at the same limit.
- **Note:** the official template accepts `xhigh|medium|low`; our server's chromix template takes `low|medium|high`. `probe` sends low, medium and high and warns if low doesn't use fewer tokens than high.

## 5. Harnesses

### Our harness (`run_eval.py`)
- Standard library only.
- Streams `/v1/chat/completions` with a custom model name, an API key (`--key` / `KEY` / `--key-file`) and `chat_template_kwargs.reasoning_effort` (`--effort-mode top|both` also sends the top-level field).
- Sends temperature, top_p, top_k, min_p and seed, plus any `--extra-body` JSON.
- Reads both `reasoning` (current vLLM) and `reasoning_content` (older), and falls back to `<think>` splitting.
- Resumable JSONL output. A time budget per phase.
- Scoring follows each benchmark's own logic, ported or vendored (§6).

### Existing harnesses
Checked against PyPI and GitHub on 2026-10-14.

| Harness (version) | Custom endpoint + key + name | Extra body (top_k, effort) | Thinking models | Verdict |
|---|---|---|---|---|
| **evalplus 0.3.1** (unmaintained since 2025-10) | `--backend openai --base-url`, key from `OPENAI_API_KEY` | **no**: hard-codes top_p 0.95 and **max_completion_tokens 768**, no extra_body | reads `content`; 768 tokens cut thinking off | generate with our harness; scoring cross-check: `evalplus.evaluate --samples` |
| **lm-eval 0.4.13** `local-chat-completions` | `base_url=…/v1/chat/completions`, `OPENAI_API_KEY`, `num_concurrent` | **yes**: `--gen_kwargs` takes JSON, including nested `chat_template_kwargs`, and spreads it into the request body | reads `content`; `think_end_token` for servers without a reasoning parser; **set `max_gen_toks` in gen_kwargs** (IFEval's YAML says 1280) | good for IFEval |
| **bfcl-eval 2026.3.23** (V4) | needs a `ModelConfig` entry (OpenAICompletionsHandler, FC), `OPENAI_BASE_URL` in `.env` | **no**: only temperature (default 0.001), no max_tokens or top_k | records `reasoning_content` only (vLLM now sends `reasoning`; affects only the logs) | use our harness; its checker is vendored |
| **LiveCodeBench runner** (last commit 2025-07) | `OPENAI_BASE_URL` via the SDK, model registered in `lm_styles.py` | no top_k, extra_body or seed | reads `content`; defaults max_tokens 2000, n 10, temperature 0.2 | use our harness; its grader is ported |
| **lighteval 0.13.0** | litellm YAML (base_url, api_key) | only temperature, top_p, seed, max_new_tokens | strips `<think>` | not needed |
| **openai/simple-evals** | `OPENAI_BASE_URL` | no | content only, max_tokens 1024 | unmaintained for new models |

**Commands for the external tools** (optional; install them in a venv on the server):
```bash
# IFEval with lm-eval
python3 -m venv /root/venv-lmeval && /root/venv-lmeval/bin/pip install "lm_eval[api,ifeval]==0.4.13"
OPENAI_API_KEY=$(cat /root/llm-api-key) /root/venv-lmeval/bin/lm-eval run --model local-chat-completions \
  --model_args model=qwen3.8-flash-next,base_url=http://127.0.0.1:8080/v1/chat/completions,num_concurrent=16,timeout=3600,max_retries=3 \
  --tasks ifeval --apply_chat_template --output_path results/lmeval-ifeval-$TAG \
  --gen_kwargs '{"max_gen_toks":16384,"temperature":1.0,"top_p":0.95,"top_k":20,"chat_template_kwargs":{"reasoning_effort":"medium"}}'

# Cross-check our EvalPlus scores with the official evaluator (samples = the code our run extracted)
python3 - <<'PY'
import json
recs = [json.loads(l) for l in open("results/swift15/medium/evalplus.jsonl")]
for ds, pref in (("humaneval", "HumanEval/"), ("mbpp", "Mbpp/")):
    with open(f"/tmp/samples-{ds}.jsonl", "w") as f:
        for r in recs:
            if r["id"].startswith(pref):
                f.write(json.dumps({"task_id": r["id"], "solution": r.get("code", "")}) + "\n")
PY
/root/venv-lmeval/bin/pip install evalplus==0.3.1
/root/venv-lmeval/bin/evalplus.evaluate --dataset humaneval --samples /tmp/samples-humaneval.jsonl
/root/venv-lmeval/bin/evalplus.evaluate --dataset mbpp --samples /tmp/samples-mbpp.jsonl
```

## 6. What is in the folder

```
quality/
  B_quality_eval.md        this file
  run_eval.py              CLI: selftest | probe | run | rescore | compare
  run_tonight.sh           one model, full suite, budgets, hard cap (SUITE_MIN=150)
  run_effort_sweep.sh      low/high reduced suite (+ C1=1 latency probe, T06=1 temperature ablation)
  night_quality.sh         swap -> health check -> suite for each model -> restore -> compare (swap commands via env)
  SHA256SUMS               checksums of code + data (without data/lcb/tests_v5)
  qeval/
    common.py              streaming client, runner (thread pool, resume, budget, prefix-cache groups), helpers
    sandbox.py             nobody + unshare -n + setpriv, rlimits, process-group kill, temp dirs
    worker_evalplus.py     EvalPlus differential tester (evalplus 0.3.1 semantics, special oracles, MBPP+ input decoding)
    worker_lcb.py          LiveCodeBench grader (testing_util.py semantics: stdin mocking, Decimal compare, 6 s per test)
    bench_*.py             evalplus, lcb, bfcl, longctx
    bfcl_vendor/           BFCL AST checker (Python), multi-turn checker, 8 simulated API backends (Apache-2.0, LICENSE kept)
    report.py              summaries, paired bootstrap, exact McNemar, sign-flip test, markdown
  tools/prepare_data.py    builds data/ from dl/ (already run)
  tools/mock_server.py     fake OpenAI server for harness tests (not a model)
  data/                    evalplus (1.2 MB), bfcl (9.9 MB), longctx (4.9 MB), lcb v6 (99 MB), lcb v5 (408 MB, optional)
  dl/                      raw downloads (don't copy)
```

**Data provenance:**

| Data | Source |
|---|---|
| EvalPlus | `HumanEvalPlus.jsonl.gz` v0.1.10 and `MbppPlus.jsonl.gz` v0.2.0 (GitHub releases) |
| LiveCodeBench | `livecodebench/code_generation_lite` test5/test6.jsonl (last updated 2025-06-05; no release after v6) |
| BFCL | gorilla repo @ 6ea5797 (2026-03-23): `BFCL_v4_*.json`, possible answers, multi-turn function docs |
| Long-context test | built here from the Python 3.12.15 standard library, token-counted with the `Qwen/Qwen3.8-Flash-Next` tokenizer |

**Long-context set (18 code bases, prompt sizes):**

| Bin | Code bases | Prompt tokens |
|---|---|---|
| 32K | 6 | 30.4–31.7K |
| 64K | 6 | 63.1–64.2K |
| 110K | 6 | 110.7–111.7K |

Each code base carries 10 questions:
- 4 near-duplicate key lookups (`CFG_AMBER_X` vs `CFG_AMBER_Y`).
- 2 "all ports registered for route X" questions (3 values each, plus 6 distractor routes).
- 2 variable-tracking chains (3 hops, plus a distractor chain with a nearby value).
- 2 "compute `quota_for_x(c)`", where the multiplier constant is defined far from the function.

Needles are placed only at top-level statement boundaries (found with the AST), so they are real module-level code. Results are broken down by kind, length and depth.

**Security notes:**
- LiveCodeBench hidden tests are base64 → zlib → **pickle** upstream. They were decoded on the Mac with an unpickler that refuses every global (only plain strings can load) and stored as JSON. The server never unpickles anything.
- BFCL multi-turn calls are evaluated against a namespace of the simulated API methods with empty builtins. Model calls are rebuilt from the JSON arguments with `repr()`, so only literals reach `eval`.
- Model-written code (EvalPlus and LiveCodeBench) runs only in the sandbox:
  - user `nobody:nogroup` via `setpriv --clear-groups --no-new-privs`;
  - `unshare -n` (no network) when the container allows it;
  - RLIMIT_AS 4 GB, CPU, FSIZE 16–64 MB, NPROC 512;
  - a private temp directory owned by nobody, cleaned up afterwards;
  - process-group SIGKILL on timeout;
  - at most 3 at a time (`--exec-workers`), to spare the vLLM CPU threads.
- `selftest` prints what sandboxed code can see (uid, network, write access to the harness directory).

**Validation done on the Mac** (sandbox mode `self+sandbox-exec`, no network, writes only inside the sandbox):
- **EvalPlus:** all 542 reference solutions run through the worker; 541 pass. HumanEval/32 (`find_zero`) fails its own 1e-4 check on one input: evalplus 0.3.1 fails that task for every model anyway (a `continue` bug), so it is a constant across models.
- **LiveCodeBench grader:** hand-written solutions pass 3 problems (stdin and call-based; 42/43/33 tests). A wrong solution fails with "Wrong answer"; TLE and the call-based path also work.
- **BFCL:**
  - Multi-turn ground truth sent through the JSON tool-call round trip scores 200/200 in base, miss_param and long_context.
  - Single-turn ground truth scores 100 % in simple, multiple and parallel; the rest are dataset quirks.
- **Whole pipeline:** the full pipeline ran against the mock server: probe, all 4 benchmarks, budgets, resume, the effort sweep, the night orchestrator and compare.

## 7. Published reference numbers (sanity checks)

No source publishes scores for any of these three 4-bit builds on our benchmarks. Each source ran its own harness, mostly at **xhigh** effort and temperature 1.0, so only compare within a source.

**Official Qwen3.8-Flash-Next, BF16** ([model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)):

| Benchmark | Score |
|---|---|
| LiveCodeBench v6 | 91.9 (window not stated) |
| GPQA-Diamond | 91.7 |
| IFBench | 81.3 |
| SWE-bench Pro | 62.5 |
| Toolathlon Verified | 73.5 |

No official HumanEval/EvalPlus (post-trained), BFCL, IFEval, RULER or MMLU-Pro.

**Other sources:**
- **Tech report, base model, pre-trained** ([arXiv 2608.30320](https://arxiv.org/html/2608.30320)): EvalPlus average 78.76 and MMLU-Pro 73.23 (5-shot). Not comparable with a chat model.
- **Swift 1.5 card** (ukisai; Swift **BF16** vs base BF16, xhigh):

  | Benchmark | Base | Swift 1.5 |
  |---|---|---|
  | LiveCodeBench v6 | 88.40 | 90.39 |
  | GPQA-D | 89.80 | 89.60 |
  | MMLU-Pro | 87.75 | 87.20 |
  | IFBench | 73.20 | 70.13 |
  | Terminal-Bench 2.1 | 67.64 | 69.66 |

  - Swift uses **55.8 % fewer reasoning tokens (mean)**.
  - At medium/low effort, GPQA-D drops 2.6/2.4 points relative to base.
  - Sampling: temperature 1.0, top_p 0.95, top_k 20 (confirmed).
- **Intel AutoRound card:** lm-eval BF16 vs INT4, 99.64 % recovery on average (MMLU 0.8651 → 0.8560, GSM8K 0.9673 → 0.9682). Only the routed experts are quantized.
- **NVIDIA NVFP4 card** (a reference for the size of quantization loss):

  | Benchmark | FP8 | NVFP4 |
  |---|---|---|
  | tau2-Telecom | 90.8 | 90.1 |
  | Terminal-Bench 2.1 | 83.3 | 82.9 |
  | GPQA-D | 92.0 | 91.5 |

  Expect the 4-bit vs 16-bit gap to be about 0.5–1 point on agentic tasks, so differences *between* good 4-bit quants may be below what one night can resolve.
- **Community** (Reddit, Aider at xhigh, Swift Q5 GGUF vs base Q5):

  | Metric | Base | Swift |
  |---|---|---|
  | First-try pass | 40.2 % | 41.1 % |
  | Median tokens per case | 17.6K | 7.0K |

  McNemar p ≈ 0.29, not significant.

**Sanity checks for our medium-effort numbers:**
- LiveCodeBench v6 should land below the xhigh 88–92 (fewer reasoning tokens, a 24K limit).
- EvalPlus HumanEval+ should be about 85–95 %.
- A quant that falls more than about 10 points below the others on any benchmark points to a setup problem (template, parser, effort wiring) rather than quantization. Check `probe-*.json`, the `raw_tool_markup` count (tool-call XML left in the text = parser miss) and `api_error`.

**Discrepancies to check on the server:**
- **cyankiwi's base model:** the public `cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4` card names the **stock** `Qwen/Qwen3.8-Flash-Next` as its base, not an abliterated one. The closest abliterated AWQ build is `calclout/...-Abliterated-AWQ-W4A16`, which has no model card. Check the local model's README/config.
- **Tool parser:** the official vLLM recipe uses `--tool-call-parser qwen3_coder`; we run `qwen3_xml`. Keep it identical for all three models tonight.

## 8. How to run

**1. Copy from the Mac.** About 115 MB without the optional v5 tests. The container is an LXC (reachable over SSH):
```bash
cd /path/to/research-tonight   # the local directory that holds the harness as quality/
rsync -a --exclude dl --exclude 'results*' --exclude data/lcb/tests_v5 quality/ root@<container>:/root/tune/quality/
```

**2. On the server, once:**
```bash
cd /root/tune/quality && sha256sum -c --quiet SHA256SUMS && echo checksums OK
python3 run_eval.py selftest            # must end with SELFTEST PASSED; shows mode=nobody+netns (or nobody)
python3 run_eval.py probe --endpoint http://127.0.0.1:8080 --key-file /root/llm-api-key --model qwen3.8-flash-next --tag swift15
#   -> low < medium < high tokens, the answer "17", one get_weather tool call
```

**3. One model** (it must already be served):
```bash
systemd-run --unit=qeval-swift15 --property=WorkingDirectory=/root/tune/quality \
  --setenv=TAG=swift15 --setenv=KEY_FILE=/root/llm-api-key --setenv=ENDPOINT=http://127.0.0.1:8080 \
  --setenv=MODEL=qwen3.8-flash-next --setenv=REASONING_EFFORT=medium --setenv=CONCURRENCY=16 \
  /root/tune/quality/run_tonight.sh
journalctl -u qeval-swift15 -f          # or: tail -f results/swift15/medium/suite.log
```

**4. Whole night.** Set your own swap commands (for example the existing `qfn-window.sh up <profile>` / `prod`):
```bash
systemd-run --unit=qeval-night --property=WorkingDirectory=/root/tune/quality \
  --setenv=KEY_FILE=/root/llm-api-key --setenv=MODELS="swift15 intel cyankiwi" \
  --setenv=SWAP_swift15='...' --setenv=SWAP_intel='...' --setenv=SWAP_cyankiwi='...' \
  --setenv=RESTORE_CMD='...' [--setenv=SWEEP_TAG=swift15] /root/tune/quality/night_quality.sh
```

**5. Morning:**
```bash
python3 run_eval.py compare results/swift15/medium results/intel/medium results/cyankiwi/medium \
  --labels swift15,intel,cyankiwi --json-out results/compare-medium.json     # writes .json and .md
python3 run_eval.py compare results/swift15/low results/swift15/medium results/swift15/high \
  --labels low,medium,high --json-out results/compare-effort.json
```

**Useful knobs:**
- `--budget-min` / `B_*` env variables: per-phase budgets.
- `--limit N`: deterministic subsample.
- `--repeats k`.
- `--bfcl-spec "multi_turn_base:200,multi_turn_miss_param:100"`.
- `--releases v5,v6` (copy `tests_v5` first).
- `--bins 110k --per-bin 6`.
- `--history-reasoning none`: don't send earlier reasoning back in tool loops.
- `--no-stream`.
- `rescore results/.../lcb.jsonl`: re-execute the stored code without model calls, e.g. if the sandbox was misconfigured.

**Before starting, check:**
- **Context limit:** the longest prompt is ~111.7K tokens + 8K output. It needs `max_model_len` ≥ 120K; production ran 125K prompts, so this should hold.
- **KV room:** cyankiwi has 574K tokens of KV; the long-context phase uses about 111K + 6 × 10K thanks to prefix sharing.
- **CPU:** the sandbox runs ≤ 3 processes next to vLLM on the 6-core i7. If LiveCodeBench shows many "Time Limit Exceeded" results on easy problems, rescore with `--exec-workers 1`.
