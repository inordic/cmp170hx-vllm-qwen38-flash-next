# Strata on Asgard - overnight tuning report (2026-10-08/09)

Swift 1.5 Qwen3.8-Flash-Next IQ3_XXS on 2x CMP 170HX 64 GB (Gen2 x8), LXC 232, 32 GB RAM.
Production runs as systemd service `strata` (enabled, auto-restart), API on :8080 (key in `/root/strata-api-key`).

## Result: time to first token (TTFT) and prefill

| Prompt | Stock Strata (setup defaults) | Production now | Prefill tok/s stock -> now |
|---|---|---|---|
| ~270 tokens | - | **0.40 s** | - |
| ~1K | 1.1 s | **0.64 s** | 835 -> 1,489 (+78%) |
| ~4K | 2.4 s | **1.5 s** | 1,577 -> 2,557 (+62%) |
| ~8K | - | **2.5 s** | -> 3,140 |
| ~16K | - | **4.4 s** | -> 3,877 |
| ~32K | ~10.5 s | **7.0 s** | 3,068 -> 4,240 (+38%) |
| ~64K | - | **13.1 s** | -> 4,928 |
| ~128K | 27.8 s | **23.6 s** | 4,114 -> 5,045 (+23%) |
| follow-up turn (same chat) | 0.22 s | **0.19 s** | prefix cache |
| switching between chats | full re-read (seconds-minutes) | **restored from RAM** (8 chats, 8 GiB) | |

Generation: ~105-125 tok/s (was ~87-113). Numbers: medians, single stream, fresh prompts, 128 output tokens.

## What production runs (`/opt/Strata/strata-production.json`)

| Setting | Effect |
|---|---|
| `STRATA_PF_FUSED=1` | fused int8 tensor-core prompt kernels (SM80): prefill +14-18% |
| `STRATA_PREFILL_STREAM_MIN=128` | those kernels also for prompts < 1024 tokens: 1K TTFT 1.0 -> 0.64 s |
| `--pipeline-windows 2` | both cards busy during decode: generation +6-17% |
| `--mtp-window 8192` | draft layer reads only the last 8K of a prompt: 16-64K prefill +5-6% |
| `--conversation-cache-mib 8192 --conversation-cache-slots 8` | up to 8 conversations parked in RAM: follow-up cache hits 15% -> 92% when chats interleave |
| `STRATA_PF_SPLIT_PARTS=6` (**local patch 1**) | prompts read in ~6 chunks (2K-8K) so both cards work at once: 4-16K prefill +18-26% |
| `STRATA_PLE_VRAM=0` (**local patch 2**) | 26.8 GiB n-gram table resident on GPU 0 instead of random SSD reads: +2-13% and no more disk stalls (one SSD hiccup had made a 30K prompt 14.7 s instead of 8 s) |
| `STRATA_PARK_SKIP_SAME_PREFIX=1` (**local patch 3**) | a follow-up of the same chat is not parked first (clients drop the model's thinking): follow-ups 0.45-0.7 s -> 0.19 s |
| layer split auto (24/24), int8 KV, `--spec 4`, `--spec-min-p 0.5` | measured best |

The engine is `/opt/Strata/engine/strata-prod` = Strata v0.1.41 (commit fb58e0d) built for SM80 with the three patches
(`/opt/Strata/strata-pf-split-parts.patch`, `strata-ple-vram.patch`, `strata-park-skip.patch`). Every patch is opt-in
by its environment variable; without them the engine behaves like upstream. Previous build: `strata-prod.v1`.

## Verified

- Needle in a haystack: 6/6 at 32K and 128K (depths 10/50/90) - also 6/6 on IQ2_XS
- Tool calls (OpenAI format) incl. tool result round trip, Anthropic `/v1/messages`, streaming (first token 0.3-0.4 s), API key enforced (401)
- Soak: 425 mixed requests / 40 min, 0 errors, GPU memory flat; plus 263 requests / 20 min with the conversation cache, 0 errors
- Engine killed: server restarts it, the waiting request still gets HTTP 200 (~84 s); server killed: systemd back in ~98 s
- GPUs never throttle (temps < 60 C, no power/thermal events) - prefill is not power bound

## Tried and rejected

Larger/smaller fixed chunks (`--prefill 2048/4096/16384/auto:32768`), `--spec 3/5`, `--spec-min-p 0.3`, `--kv k8v4`, `--kv fp16`,
`--ple-io mmap`, bigger PLE row cache / more in-flight reads, `STRATA_GDN_CHUNKED`, GDN kernels, `STRATA_PA_FAST`,
`STRATA_PF_GEMM` (+PAD crashes), layer split 22/25/26 (26 helped only without the 8K MTP window), swapping GPU order,
`--mtp-window 4096` (+1-3% prefill but lower draft acceptance), very small chunks (512), `--calibrate` (tunes only
non-resident experts - all 24,576 experts are resident here). **IQ2_XS is slower at prefill** (-7 to -12%) on these cards.

## On your side

1. **NAT**: <public-ip>:8080 must forward to **<container-ip>:8080** (it is not reachable from outside yet).
2. **RAM upgrade (96 GB)**: give the container ~84 GiB, then raise the conversation cache in
   `/root/tune/make_prod.py` (e.g. 32768 MiB / 16 slots), run `python3 /root/tune/make_prod.py auto 8192` and
   `systemctl restart strata`.
3. Container cores 8 -> 12 (small), `nvidia-smi -pm 1` on the host (faster restarts), GPU 1's HBM clock (1458 vs 1728 MHz).
4. Do not run `./setup.sh --update` blindly: it replaces the engine. Re-apply the three patches after an update.

## Operations

```
systemctl status strata            # state
systemctl restart strata           # restart (~80-90 s until ready; 19 s of it loads the n-gram table into VRAM)
tail -f /var/log/strata/server.log # requests;  engine log: /var/log/strata/engine.log (logrotate daily, 7 kept)
cat /root/strata-api-key           # API key
```
Rollback to stock Strata: `systemctl disable --now strata; cd /opt/Strata && ./setup.sh`.
Tuning data: `/root/tune/` (results.jsonl, tune.log, per-variant logs). IQ2_XS model + pack (~103 GB) are in
`/opt/Strata-data/{models/swift-IQ2_XS,packs/swift-iq2_xs}` - delete them if you need the space.

## Update 2026-10-09 morning

- **Chat template**: `qwen-3.8-27b-chromix+froggerick.v1` (leonbollerup/ai) in `/opt/Strata-data/tokenizer-chromix/chat_template.jinja`
  (the config's `tokenizer`; the Swift original is kept beside it as `chat_template.swift-original.jinja`). Served at `/props`;
  tools, parallel tool calls, tool history and preserved reasoning detected; all functional checks pass.
- **Default reasoning effort: medium** - `/opt/Strata/strata-production.shared-settings.json` = `{"reasoning_effort": "medium"}`
  (used for every request that sets no `reasoning_effort` / `enable_thinking`; a request's own value still wins).
- **Container restart verified**: after `systemctl reboot` of LXC 232 the service started by itself and the API answered
  ~85 s after boot with the production engine, flags and environment.

### Performance (production, 256 output tokens, unique prompts, all requests of a cell sent at once)

| prompt | conc | pp tok/s | tg tok/s per stream | TTFT median / max | total out tok/s |
|---|---|---|---|---|---|
| 457 | c1 | 630 | 121 | 0.77 / 0.93 s | 89 |
| 430 | c2 | 718 | 123 | 1.95 / 3.30 s | 95 |
| 487 | c4 | 769 | 116 | 4.88 / 9.06 s | 91 |
| 4.5K | c1 | 2,461 | 116 | 1.83 / 1.98 s | 63 |
| 3.6K | c2 | 2,209 | 117 | 3.38 / 5.62 s | 67 |
| 3.6K | c4 | 2,334 | 114 | 7.24 / 13.3 s | 66 |
| 15K | c1 | 3,710 | 115 | 4.14 s | 40 |
| 16K | c2 | 3,618 | 108 | 7.71 / 11.1 s | 38 |
| 17K | c4 | 3,526 | 110 | 15.1 / 26.5 s | 36 |
| 30K | c1 | 4,054 | 101 | 7.50 s | 26 |
| 31K | c2 | 4,119 | 103 | 12.3 / 17.6 s | 26 |
| 28K | c4 | 3,897 | 109 | 22.3 / 37.0 s | 26 |
| 51K | c1 | 4,582 | 103 | 11.3 s | 19 |
| 62K | c2 | 4,575 | 103 | 21.3 / 29.9 s | 16 |
| 140K | c1 | 4,899 | 101 | 28.8 s | 8 |

Production serves one request at a time (the others queue). `--batch 4` was measured too: c4 with short prompts
improves (0.5K: TTFT max 9.1 -> 3.3 s, total 91 -> 109 tok/s), but a single stream loses 10-15% decode
(`--pipeline-windows` is off with batch slots) and long prompts gain nothing (prompts are still read one at a time).
Kept serial. To try it: add `"--batch", "4"` to the config's args and restart.

### Launch (as running)

```
# systemd: /etc/systemd/system/strata.service
ExecStart=/opt/Strata/.venv/bin/python /opt/Strata/serve/server.py --engine strata --config /opt/Strata/strata-production.json --port 8080

# the engine server.py starts from that config:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1 \
STRATA_PF_FUSED=1 STRATA_PF_SPLIT_PARTS=6 STRATA_PLE_VRAM=0 STRATA_PREFILL_STREAM_MIN=128 STRATA_PARK_SKIP_SAME_PREFIX=1 \
/opt/Strata/engine/strata-prod --serve \
  --pack /opt/Strata-data/packs/swift-iq3_xxs \
  --native /opt/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --ple-gguf /opt/Strata-data/models/swift-IQ3_XXS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf \
  --expert-profile /opt/Strata/data/expert-profile.bin --expert-cache auto --prefill auto \
  --spec 4 --spec-min-p 0.5 --mtp /opt/Strata-data/mtp/rt \
  --max-context 262144 --kv int8 --resident-experts \
  --pipeline-windows 2 --mtp-window 8192 \
  --conversation-cache-mib 8192 --conversation-cache-slots 8 \
  --layer-split auto
```
