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
