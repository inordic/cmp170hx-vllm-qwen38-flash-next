# Speed and functional tests

Every script talks to an OpenAI-compatible endpoint on `127.0.0.1` and takes the port as its first argument. The API key comes from `API_KEY`, or from the file named in `KEY_FILE` (default `/root/llm-api-key`).

## Scripts

| Script | What it measures |
|---|---|
| `perf_exact.py NAME PORT TOKENIZER_DIR [RUNS] [GEN]` | single-stream prefill and decode with **exact** prompt lengths. Each prompt is real text cut to the size with the model's tokenizer, and has a unique first line so the prefix cache never helps. Medians over the runs. `PERF_SIZES` sets the sizes. This is what [`../results/speed/`](../results/speed/) holds |
| `method_compare.py` | real text vs random tokens at the same length. Random tokens understate PLE builds, because unique n-grams miss the row cache |
| `conc_stagger.py` | N users with staggered starts, plus a mixed scenario (chat users and one long-context agent). Reports total tok/s, TTFT median/p95 and per-stream decode |
| `prod_check.py PORT` | model aliases, default and per-request reasoning effort, a tool call, Anthropic `/v1/messages`, streaming TTFT, API key enforcement |
| `func_check.py PORT` | a plain answer, a tool call, and needle-in-a-haystack up to 128K |
| `multiturn_check.py PORT N TOKENS` | multi-turn conversations with prefix-cache follow-ups. Reports TTFT per turn and the cache hit rate |
| `tool_adherence.py` | 100 tool-call cases. Did the model call the right tool with valid arguments? |
| `soak.py`, `soak_mixed.py` | long runs: errors, RSS, and fact checks per answer |
| `ple_disk.py` | PLE table read traffic during prefill |
| `gputest/` | `gpu_selftest.cu` (memory and compute self-test per card), `mem_probe.cu`. Build with `nvcc -arch=sm_80` |
| `night3-template-ab/` | the overnight chat-template A/B: server variants, verify, experiments, a gated deploy, and the item-clustered analysis (`analyze_templates.py`) |

## Night 4 (`night4/`)

The night-4 tests and the overnight A/B (2026-10-19/20) that chose the current production profile. The scripts expect their files in `/root/tune/night4` and a test server on port 8008, like the run did; the paths are at the top of each script. Results are in [`../results/checks/night4/`](../results/checks/night4/), [`../results/speed/`](../results/speed/) and [`../results/logs/night4-NIGHT-LOG.md`](../results/logs/night4-NIGHT-LOG.md).

**Correctness checks** (standard library unless noted). T1, T2 and T4 take the key with `--api-key` (`run.sh` passes the contents of the key file); the others read `API_KEY`, or the file in `KEY_FILE`.

| Script | What it checks |
|---|---|
| `T1-mtp-drafter-check.py` | "dead drafter": does a stream fall to ~1 token per step when 3, 5, 6 or 7 run in parallel? Tokens per streamed chunk = accepted drafts + 1. Idea from azampatti's `mtp_drafter_check.sh`; independent implementation |
| `T2-pad-probe.py` | one-token fresh prompts while other requests decode: non-200 answers, NaN logprobs, or a `!` first token (placeholder drafts accepted). Checks patches C1 and C2; vektorprime measured 11/12 failures on stock code |
| `T3-spec-kernel-exactness.py` | offline, on one GPU, no server (torch + the patched tree): drives the tree's own drafting and rejection kernels on synthetic distributions and checks that probabilistic drafting is unbiased. It must flag the old shared-noise bug, which shows that it can see a bias. Also has a unit check of C1, and a `c3` case for patch C3, which production has but this repository does not publish: without C3 that case prints "skipped". Night 4 ran it on the production tree ([`T3.txt`](../results/checks/night4/T3.txt)) |
| `T4-prefix-bleed-soak.py` | long documents with a planted code, sent cold and then warm twice (prefix hit, and hit + follow-up turn), concurrently. Any wrong or foreign code, loop or HTTP error is a cross-request state bleed |
| `k1_cow_check.py` | INT8 KV + mid-block prefix hits (patch K1): `multiturn` (hit offsets steered via `/tokenize`, with cold controls), `fanout`, `firstbatch`, `fill`, `pressure`, `compare`. Also counts the K1 lines in the server log |
| `canaries.py` | `preempt`: one ~131K-token prompt, preemptions before/after (the Mamba state-block leak DominikBucko saw); `stalerows`: per-request decode at 3/5/6/7 users, an outlier far below the median points at the stale-row drafter bug |
| `dist_test.py` | output-distribution test for drafting: 1,500 random-letter and random-digit answers at T=1 per config, TV distance against no drafting, with the noise floor and a permutation p-value |
| `greedy_identity.py`, `greedy_diff.py` | 16 greedy answers per config, how many are identical between two configs, and where and how they diverge |

**The overnight run.**

| Script | What it does |
|---|---|
| `run.sh` | the night: clocks locked at 1440 MHz for the A/B phases (a helper on the host does the locking), baseline and canaries, no drafting, int4 draft head, MTP depth 4/5, INT4 MTP experts, DFlash2, probabilistic drafting, batch and PP split, bounded prefill under load, the same-night quality reference, Swift-A16 and Swift-Q6x, Intel, INT8 KV with K1, terse; then the deploy |
| `speed.sh NAME` | the same speed set for every config: exact-length prefill/decode (`perf_exact.py`), code decode, code-edit decode, `func_check.py`, `multiturn_check.py`; writes `res-NAME.json` |
| `mkprof.py BASE OUT opt=value...` | derives a test profile from the production profile: venv, model, draft head, MTP depth, speculative-config keys, prefix-match unit, batch, split, KV dtype, extra flags and env switches |
| `select.py`, `accept_mb.py` | the deploy pick: score = geometric mean of code and code-edit decode; a candidate must pass func + multiturn and keep 8K/32K prefill within 3%. `accept_mb.py` takes INT4 MTP experts only if they keep 99% of the score |
| `deploy4.sh NAME` | puts the winner into production with the gate (prod/func/multiturn, else rollback), a quality guard against night 3, the feature check and a speed check |
| `df_hook.sh` | the late DFlash2 block, added during the night after patches D3b/D3c |

`speed.sh` calls `decode_code.py` and `edit_decode.py` (code decode and code-edit decode), and `deploy4.sh` calls night 3's `feature_check.py`. These three are not in this repository yet.

## Night 5 (`night5/`)

The night-5 A/B (2026-10-20/21): the night-4 standard `R` against three candidates that each add one piece, C1 (Swift-A16 + INT8 KV), C2 (+ probabilistic drafting) and C3 (+ terse). Same conventions as `night4/`; the T1/T2/T4, K1, speed and canary tools are the night-4 ones, and its `speed.sh` is [`night4/speed.sh`](night4/speed.sh) with night-5 paths and the `n5-` prefix. Results: [`../results/logs/night5-NIGHT-LOG.md`](../results/logs/night5-NIGHT-LOG.md), [`../results/checks/night5/`](../results/checks/night5/), [`../results/quality/compare-night5.md`](../results/quality/compare-night5.md).

| Script | What it does |
|---|---|
| `run.sh` | the night: clocks locked at 1440 MHz; `R` with the full prefill matrix (1K–128K) and 8/16-user and mixed concurrency; C1 with the K1 checks first (first batch, multi-turn, 16-user fanout, a pool filled to 1.45×, multi-turn again); C2 with T1/T2; C3 with T4 and concurrency; the full quality suite at T=1 for all four (BFCL × 3, EvalPlus, long context); then `decide.py`. With `DEPLOY=0` (the default since 2026-10-20) it only reports |
| `decide.py REF CAND...` | picks the most complete candidate that passes every gate against `REF` on the same night: BFCL ≥ REF − 1.5 points, EvalPlus and long context ≥ REF − 2, mean task latency (the median latencies of BFCL and EvalPlus, averaged) ≤ 1.1 × REF, func + multi-turn pass, and for INT8 KV the K1 checks |
| `deploy5.sh NAME` | puts a candidate into production with the gate (prod/func/multi-turn, else rollback), then the feature check and a speed check. Its "8080 healthy" line greps the first KV-pool line of the service log, which can be from an earlier start (see [`../results/README.md`](../results/README.md)) |
| `mkprof.py` | night 4's [`mkprof.py`](night4/mkprof.py) plus `tplkw=KEY:VALUE`, which sets a key in `--default-chat-template-kwargs` (used for `terse`) |
| `rollback-c1t.sh` | 2026-10-21: put the night-4 standard back into production after C1T, then the gate checks and a code-decode run (the `[rollback]` lines of the night-5 log) |
