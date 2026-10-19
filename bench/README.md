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
