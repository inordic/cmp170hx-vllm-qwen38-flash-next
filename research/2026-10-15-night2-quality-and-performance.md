# Night 2 (2026-10-14/15): quality and performance, Asgard / LXC 232

## In short

- **Long prompts are much faster.** 128K tokens now take **15.7 s** instead of **20.8 s**.
  - About **8,000 tok/s** at 32–128K, roughly the same as the fastest public 2× CMP builds.
  - The fix was reading the n-gram table off the NVMe faster: more readers, more read-ahead, and a 16 GB RAM cache.
- **Faster in production too:**
  - Code editing: **251 tok/s**.
  - 8 users: **+10%** (390 tok/s). 16 users: **+10%** (504 tok/s).
  - Short chats while a long prompt runs: response time (p95) **1.3 s instead of 2.3 s**.
- **The VBIOS flash (yesterday midday) held up.**
  - Both cards passed a full memory and compute test.
  - All functional and load tests passed with the new clocks.
- **Quality: it's a tie between the models for code, but Intel AutoRound is clearly better at tool calls and long context.**
  - The overall difference is −1.8 points (just short of significant), so **Swift stays in production** under the agreed rule.
  - **I still recommend considering Intel for agent work.** See section 3.
- **Production on 8080:** Swift 1.5 with the night's configuration. All checks passed and the 64-minute quality run against production ran without errors.

## 1. GPU stability after the flash

| | GPU 1 (flashed) | GPU 0 |
|---|---|---|
| VBIOS | 92.00.6D.00.0A (300 W) | 92.00.6D.00.0A |
| Full memory sweep (~63 GB) + exact compute check, 3× each | **3/3 PASS, 0 errors** | 3/3 PASS |
| Memory bandwidth (STREAM triad) | **1,549–1,597 GB/s**, same as GPU 0 (before the flash it ran at 1458 instead of 1728 MHz) | 1,597–1,606 GB/s |
| Highest temperature in the night (core / HBM) | 75 / 79 °C | **85 / 84 °C** (at NVIDIA's "max operating temp") |

- The temperature guard (drop to 275 W at 90 °C) **never had to step in**.
- GPU 0 runs at its thermal limit under sustained full load and occasionally lowers its own clock. That is within spec (slowdown at 95 °C, shutdown at 98 °C).

## 2. Performance

All variants were measured on the same server with the same tests. Prefill numbers are tok/s for a single user.

| Variant | 4K | 8K | 32K | 64K | 128K | Code decode | Code editing | Correct? |
|---|---|---|---|---|---|---|---|---|
| Production yesterday (after the flash) | 4,549 | 5,747 | 6,702 | 5,894 | 6,303 | 169.5 | – | ✅ |
| E0: vLLM bug fixes (#55601, #60210, #56734) | 5,112 | 5,603 | 6,123 | 6,393 | 5,936 | 168 | 219 | ✅ incl. 20-min soak |
| E1: table cache 4 GB + read-ahead 64K | 4,443 | 5,344 | 7,708 | 6,506 | 8,041 | 167 | – | – |
| E1b: as E1 with a 16 GB cache | 4,715 | 5,679 | 7,656 | 8,046 | 7,474 | 171 | – | – |
| E1c: 32 readers, queue 512, read-ahead 128K, cache 16 GB | 4,400 | 5,479 | **7,913** | 7,926 | 8,230 | 169 | – | ✅ |
| E2: more CUDA graphs | 4,331 | **2,176** | 6,189 | 5,257 | 6,836 | – | – | **rejected** |
| E3: n-gram before MTP | – | – | – | – | – | 174 | **253** | ✅ |
| **E5 = E1c + n-gram (now in production)** | 4,822 | **5,914** | 7,668 | **8,168** | **8,336** | **171** | **251** | ✅ incl. 15-min soak |

**Several users at once, E5 vs E0:**

| Load | E0 | E5 |
|---|---|---|
| 8 users | 353 tok/s | **390** |
| 16 users | 458 tok/s | **504** |
| Mixed load (short chats, TTFT p95) | 2.3 s | **1.3 s** |

**Why the gap to the fastest builds is gone:** the earlier gap came from reading the table off the NVMe.
- With random tokens, almost every lookup misses the cache. That is the method the other builders measure with.
- Measured that way we were at 4,005 (32K) and 3,080 (128K). With E1c it is 5,410 and **5,935 (+93%)**.
- The NVMe never ran short of bandwidth (~150 MB/s). The problem was waiting for each read: too few parallel reads and too little read-ahead.

**N-gram before MTP and tool calls:** the soak test pointed to slightly fewer tool calls. A direct A/B on 100 identical cases showed no harm: **96/100 without n-gram, 98/100 with it**. Both variants fail the same case (#36: empty answer, no call). That points to the model or the tool parser, not n-gram.

**INT8 KV, cause confirmed:** without our two cache flags, INT8 gives correct follow-up answers. The bug is that the INT8 patch doesn't copy its scale factors when a cache hit lands mid-block. A fix would give **1.53M tokens** of KV room instead of 0.88M (more long sessions at once). That is a later job.

## 3. Quality: Swift 1.5 vs Intel AutoRound

**Setup:** same server configuration (E5), the same items in the same order, temperature 1.0 / top_p 0.95 / top_k 20 (the model's own recommendation), reasoning medium. The comparison is paired on identical items, with significance tests.

| Benchmark | Swift 1.5 | Intel AutoRound | Difference | p | Verdict |
|---|---|---|---|---|---|
| EvalPlus HumanEval+ / MBPP+ (542) | 86.0% | 86.2% | −0.2 pt | 1.0 | **tie** |
| LiveCodeBench v6 (175) | 58.3% | 58.3% | 0 | 1.0 | **tie**; Swift has fewer runaways (23% vs 31%) and 8% fewer tokens |
| **BFCL v4 tool calls (1,040)** | 79.0% | **82.0%** | **−3.0 pt** | **0.007** | **Intel better** |
| **Long context 32K / 64K / 110K (180)** | 96.1% | **100%** | **−3.9 pt** | **0.016** | **Intel better** |
| Average over all four | | | −1.8 pt (CI −3.7…+0.2) | | just short of significant |

Where Swift loses:

| | Swift | Intel |
|---|---|---|
| BFCL `live_simple` | 77% | 88% |
| BFCL `live_parallel_multiple` | 58% | 71% |
| BFCL `simple_python` | 90% | 95% |
| Long context at 110K | 93% | 100% |

Swift's misses on long context are mostly the questions that ask for 4 values at once at 110K.

**Temperature 0.6 vs 1.0 (Swift, HumanEval+):** 96.3% vs 95.7%, 0 runaways in both. No difference, so the default of 1.0 stays.

**My recommendation:** your use case is agentic coding, which means tool calls and long contexts. That is exactly where Intel is significantly better. The cost is decode about **12% slower** (code 142 vs 161 tok/s, measured on night 1) and slightly longer thinking. Under the agreed rule Swift stays, because the overall average isn't quite significant. **I'd switch to Intel for agent work** if you can live with the slower decode. It takes 5 minutes: change `MODEL` in the profile and restart.

## 4. What runs in production now

`/opt/vllm-qfn/qfn-prod.env`, service `vllm-qfn` (enabled), port 8080:

| Setting | Value |
|---|---|
| vLLM | **the bug-fix build** in `/opt/vllm-qfn-exp`: a5a30471 + recipe + 2dameneko + #55601, #60210, #56734, #60615 (ported) |
| Model | Swift 1.5 AWQ |
| Topology | PP2 24/24, GPU 1 first |
| Cache and batching | BF16 KV, max 24 requests, chunk 4096, prefix-match-unit 32 + fine-grained cache |
| Speculative decoding | **MTP3 + n-gram lookup** |
| n-gram table on NVMe | **32 readers, queue 512, read-ahead 131072, 16 GB RAM cache** |
| Template, reasoning, API | chromix template, reasoning medium, same key and model names as before |

**Rolling back:**
- **To yesterday's production** (Swift without tonight's changes):
  ```bash
  cp /root/tune/night2/qfn-prod.env.baseline /opt/vllm-qfn/qfn-prod.env && systemctl restart vllm-qfn
  ```
- **To Intel instead:** in `qfn-prod.env`, set `MODEL=/opt/vllm-qfn/models/intel-autoround`, then `systemctl restart vllm-qfn`.

## 5. Open items and recommendations

1. **Swift or Intel in production?** Intel is better for tool calls and long context, Swift is faster. Your call. I recommend Intel for agent work.
2. **The INT8 KV fix:** copy the scale factors on mid-block cache hits. Gain: 1.5M tokens of KV room instead of 0.88M. Medium effort; it fits a day of preparation plus a night of tests.
3. **GPU 0's temperature (85 °C under full load).** Leave it at 300 W, or cap it at ~275 W for more margin; the cost is a few percent on long prompts. The temperature guard on the host is a transient timer and stops at the next host reboot. If you want it permanent, I'll make it a real service.
4. **The clock offset** (+200–250 MHz, now possible on both cards): not done. The risk is silent calculation errors, and both cards are already at their temperature limit.
5. **Tool-call case #36** (empty answer in both variants): worth looking at the tool parser.
6. **Disk:** cyankiwi (176 GB) is still there as you wanted. We have three models in 223 GB free.

**Raw data in container 232:**

| What | Where |
|---|---|
| Night log | `/root/tune/night2/NIGHT-LOG.md` |
| Per-run logs | `run-*.log` (same folder) |
| Profiles | `profiles/` (same folder) |
| Quality results | `/root/qeval/results/` (`compare-medium.md` has every table and subset) |
| Temperature traces (host) | `/root/gpu-thermal-trace.log` |
