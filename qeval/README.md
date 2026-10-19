# qeval: the quality harness

A resumable quality suite for an OpenAI-compatible endpoint, built to compare models, configs and chat templates on the same items. The design notes are in [`../research/quality-eval-design.md`](../research/quality-eval-design.md).

## Benchmarks

| Bench | Items | Scored by |
|---|---|---|
| **BFCL v4** | 11 categories, 1,040 items: simple, multiple, parallel, parallel-multiple, irrelevance, the live variants, multi-turn base | AST checker and multi-turn state checker, vendored from the gorilla repo in `qeval/bfcl_vendor/` (Apache-2.0; commit in `UPSTREAM_COMMIT`) |
| **EvalPlus** | HumanEval+ 164, MBPP+ 378 | the plus tests, run in a sandbox |
| **LiveCodeBench v6** | 175 problems (easy / medium / hard) | hidden tests, run in a sandbox |
| **Long context** | 180 needles at 32K, 64K and 110K | exact match in code haystacks |

## How it runs

**Sandbox.** Generated code runs as `nobody` in its own network namespace: no network, and no writes outside a temp directory. A self-test runs before every suite and aborts if the isolation is not in place.

**Usage:**

```bash
python3 tools/prepare_data.py --dl <downloads> --out data
```

```bash
ENDPOINT=http://127.0.0.1:8080 KEY_FILE=/root/llm-api-key MODEL=qwen3.8-flash-next TAG=mytag REASONING_EFFORT=medium CONCURRENCY=16 ./run_tonight.sh
```

- `prepare_data.py` builds `data/` once from the public releases. Hidden LiveCodeBench tests are decoded with an unpickler that refuses every global.
- `run_tonight.sh` writes results to `results/<tag>/<effort>/<bench>.jsonl`.
- `SHA256SUMS` pins the data files.
- `run_eval.py compare` makes the comparison tables.

**Options:**
- `PHASES`, `BFCL_SPEC` and `BFCL_REPEATS` select benchmarks, categories and repeats.
- `EXTRA_BODY` is deep-merged into every request; use `@file` for a file, e.g. for template A/Bs.
- `EXTRA_ARGS` passes further harness flags, e.g. `--history-reasoning none`.

**Sampling.** The defaults are the model card's: temperature 1.0, top-p 0.95, top-k 20. Every item gets a fixed seed per (item, repeat), so two runs are paired item by item.
