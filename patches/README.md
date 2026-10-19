# Our vLLM patches

These patches go on top of **vLLM `a5a30471` + iIIusi0n `qwen38-ple-ssd.patch` + 2dameneko `56e1347`**. Apply them in order; see [`../BUILD.md`](../BUILD.md).

| Patch | What it does | Origin |
|---|---|---|
| `01-vllm-55601-seed-mamba-state-idx-with-mamba-block-size.patch` | seeds the Mamba state index with the Mamba block size | backport of [vllm#55601](https://github.com/vllm-project/vllm/pull/55601) |
| `02-vllm-60210-mamba-align-copies-by-request-slot.patch` | under pipeline parallelism, a step's Mamba state post-processing runs after later steps have re-gathered the batch tables. The state copies now index the per-request-slot tables, not batch rows that may already belong to another request | backport of [vllm#60210](https://github.com/vllm-project/vllm/pull/60210) |
| `03-vllm-56734-dummy-draft-steps-pad-slot-mapping.patch` | pads the slot mapping in dummy draft steps (MTP + CUDA graphs) | backport of [vllm#56734](https://github.com/vllm-project/vllm/pull/56734) |
| `04-vllm-60615-PORT-ngram-lookup-before-mtp.patch` | n-gram lookup in front of MTP drafting (`"ngram_lookup": true` in the speculative config); off by default | **port** of the open draft [vllm#60615](https://github.com/vllm-project/vllm/pull/60615). Its prerequisites (#40704, #60335) are not in this tree, so the five commits are squashed and adapted |
| `05-auth-x-api-key.patch` | accepts the API key as `x-api-key` (Anthropic clients) as well as `Authorization: Bearer` | ours |
| `06-A1-thinking-off-consistency.py` | vLLM's qwen3 reasoning parser only reads `chat_template_kwargs.enable_thinking`. A1 works out on/off exactly the way Qwen Asgard does (effort none/off, inline `<\|think_*\|>` tags, string values), for the template and the parser together. Without it, the answer can land in `reasoning` with an empty `content`. Checked on 1176 combinations | ours (from [qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja)) |
| `07-A2-history-reasoning-fields.py` | reads history reasoning from a plain-string `thinking` field, and makes `/tokenize` accept `reasoning_content` | ours |

**Manifests.** `manifest/` holds the sha256 of every touched file before patch 01 and after each of patches 01–04, plus every file that differs from `a5a30471` after patch 04. `make-venv.sh` builds the venv for the patched tree from the precompiled wheel.

**Tested on this box.** Patches 01–04 pass the full gate: functional, multi-turn and a 20-minute soak with 896 requests and 0 errors. With n-gram lookup on, tool adherence is 98/100, against 96/100 without.
