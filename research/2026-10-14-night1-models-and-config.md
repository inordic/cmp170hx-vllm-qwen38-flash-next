# Night job 2026-10-13/14: model comparison and multi-user tuning (Asgard, LXC 232)

## In short

**Production on `http://<server>:8080` now runs Swift 1.5 AWQ (ukisai) with a new configuration.** It replaced yesterday's cyankiwi model.

- **Quality: better.**
  - HumanEval: 40/41 (yesterday 37/41). The code is actually run against the tests.
  - Tool calls: 13/13.
  - Fewer runaways where the model thinks until the token limit and never answers.
- **Decode: faster.**
  - Code: 161 tok/s, against 144 yesterday (+12%).
  - A finished answer arrives 17% sooner (median 5.0 s vs 6.0 s).
- **Prefill: unchanged.** About 4.5K tok/s at 4K, 5.3K at 8K and ~6–6.6K at 32–128K.
- **More users at the same time.** At 16 users the total is 457 tok/s instead of 365, with TTFT 0.5 s instead of 5.9 s. At 24 users it is 494 tok/s instead of 361, with TTFT 0.66 s instead of 11.5 s. There is room for 878K tokens of context, up from 574K.
- **Checks:**
  - All production checks passed.
  - Soak test: 75 min, 4,261 requests, **0 errors**, memory flat, no slowdown.

## 1. Model comparison

All three models ran with the same configuration and the same tests.

| | cyankiwi AWQ (yesterday) | **Swift 1.5 AWQ** | Intel AutoRound |
|---|---|---|---|
| HumanEval subset (41 tasks, code executed) | 37/41 | **40/41** | **41/41** |
| HumanEval failures that were runaways | 4 | 1 | 0 |
| Tool calls (13 scenarios) | 13/13 | 13/13 | 13/13 |
| Tool calls mid-conversation (100 cases) | 100/100 | 99/100 | – |
| Finished answer, 8 coding tasks (median / total) | 6.0 s / 51 s | **5.0 s / 44 s** | 5.6 s / 52 s |
| Decode on code | 144 tok/s | **161 tok/s** | 142 tok/s |
| Prefill 1K / 4K / 8K | 2,810 / 4,675 / 5,745 | 3,168 / 4,716 / 5,504 | 3,232 / 4,619 / 5,907 |
| Prefill 16K / 32K | 6,971 / 6,473 | 6,586 / 6,645 | 6,771 / 6,628 |
| Prefill 64K / 128K (median of 2) | 5,999 / 6,274 | 5,973 / 6,610 | 6,552 / 6,150 |
| KV room | 574K tokens | **901K** | 779K |
| 8 users at once | 351 tok/s | 351 | 344 |

How to read this:
- **Prefill is a tie at every length.** Single runs vary by ±12%.
- **Quality:** Swift and Intel are clearly better than cyankiwi. Intel and Swift are tied within the noise (one task apart).
- **Swift wins on speed:** decode, time to a finished answer, and KV room.
- **Intel is the best alternative if quality comes first.** It had no runaways and the cleanest answers, but decode is 12% slower.

## 2. Multi-user tuning (on Swift)

| Step | 16 users | 24 users | Single user | Multi-turn correct? |
|---|---|---|---|---|
| L0: before (max 8 requests) | 365 tok/s, TTFT 5.9 s | 361 tok/s, TTFT 11.5 s | reference | yes |
| **L1: max 24 requests + CUDA graphs for larger batches** | **457 tok/s, TTFT 0.5 s** | **494 tok/s, TTFT 0.66 s** | unchanged | **yes**, so deployed |
| L2–L4: + INT8 KV (1.5–1.7M tokens of room), GPU memory 0.92, chunk sizes, fairness | 434–444 tok/s | 483–494 tok/s | 4K prefill −15% | **NO**, see below |
| L5: L1 + chunk 4,896 | – | – | no gain | yes, but not adopted |

## 3. Rejected, and why

- **INT8 KV is broken in this combination.**
  - What happens: follow-up turns that hit the prefix cache come back empty or wrong. The model thinks until the token limit, about 62 s.
  - When: it happened in all four INT8 variants and never with BF16.
  - It was also slower for a single user.
  - Never enable INT8 KV in this setup.
- **Parking sessions in RAM (`--kv-offloading-size 16`).**
  - The test: 40 sessions of 32K each, more than the GPU cache holds.
  - None of the 40 follow-ups was restored; with or without offload they took about 5 s.
  - With offload, 2 of 80 answers were also wrong.
  - The offload does not seem to cover this hybrid model's GDN state.
- **The other experiments:**
  - **Chunk size 4,896:** no gain.
  - **MTP2 under load:** not tested, because it would cost single-user decode.
  - **DFlash2:** a dead end. No draft model exists, and vLLM doesn't support it for this model.
- **The n-gram table in RAM won't speed anything up.** Prefill reads only 12–30 KiB per token from the NVMe, 80–190 MB/s, so the disk is nowhere near its limit.

## 4. What production runs

`/opt/vllm-qfn/qfn-prod.env`, under the permanent service `vllm-qfn`:

| Setting | Value |
|---|---|
| Model | `/opt/vllm-qfn/models/swift15-awq` (ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ @ 598afc96) |
| Pipeline | PP2, 24/24 layers, GPU 1 as the first stage |
| KV cache | BF16 |
| Speculative decoding | MTP3 |
| Batching | max 24 requests, extended CUDA graphs, chunk 4096 |
| Prefix cache | 32-token granularity plus the fine-grained cache for MTP |
| Chat template | your chromix template |
| Reasoning | `medium` by default |
| Anthropic API | yes (`x-api-key` works) |
| Model names | `qwen3.8-flash-next` (main), `swift-1.5-awq`, `Qwen3.8-Flash-Next-AWQ-INT4`, `swift-1.5-iq3_xxs` |
| API key | unchanged (`/root/llm-api-key`) |

**Final matrix, single user:**

| Prompt | TTFT | Prefill | Decode |
|---|---|---|---|
| 0.7K | 0.28 s | – | 155 tok/s |
| 3.8K | 0.84 s | 4,531 tok/s | – |
| 7.9K | 1.49 s | 5,313 tok/s | – |
| 27K | 4.1 s | 6,730 tok/s | – |
| 122K | 21.4 s | – | – |

Decode stays at 133–165 tok/s across prompt sizes. Four users at 0.5K produce 310 tok/s in total.

## 5. Recommendations (host changes, your call)

1. **Raise GPU 0's power limit to 300 W.** On long prompts it sits at 235–242 W of its 250 W limit and drops from 1455 to 1380 MHz. Its VBIOS allows 300 W. Expected gain: a few percent on long prefill. Command on the host: `nvidia-smi -i <index of 0000:01:00.0> -pl 300`.
2. **GPU 1's memory clock (1458 → 1728 MHz).** Use 170tune, or flash the 300 W VBIOS. GPU 1 is the first pipeline stage.
3. **Disk.** Three models take 511 GB (224 GB free).
   - Keep **Swift**, which is production.
   - Keep **Intel** as the quality alternative.
   - **cyankiwi** (176 GB) can go once you're happy with Swift. It also works as the rollback model.
4. **For a later session:**
   - backport n-gram lookup in front of MTP (#60615, +17% on code edits);
   - backport the two hybrid fixes (#55601, #60210);
   - report the INT8 KV bug to 2dameneko.

## 6. Operations

| Task | Command |
|---|---|
| Status | `systemctl status vllm-qfn`, `/opt/vllm-qfn/qfn-window.sh status` |
| Restart | `systemctl restart vllm-qfn` (about 3.5 min until 8080 answers) |
| Logs | `/var/log/vllm-qfn/vllm.log` |
| Roll back to yesterday's setup (cyankiwi) | `cp /opt/vllm-qfn/qfn-prod.env.before-night /opt/vllm-qfn/qfn-prod.env && systemctl restart vllm-qfn` |
| Use Intel instead | in `qfn-prod.env`, set `MODEL=/opt/vllm-qfn/models/intel-autoround`, then restart |

**Raw data:** in the container, `/root/tune/night/`. The summary is in `NIGHT-LOG.md`. Also there: one `run-*.log` per test, `quality-*.json` (each HumanEval task with its error), `tools-*.json`, `conc.jsonl`, `soak.log`, `perf-night-final.log` and the profiles in `profiles/`.

**Safety timers:** at 07:15 and 08:00 Swedish time, timers check 8080. If it isn't answering, they restore yesterday's configuration automatically. Production is healthy, so they will do nothing.
