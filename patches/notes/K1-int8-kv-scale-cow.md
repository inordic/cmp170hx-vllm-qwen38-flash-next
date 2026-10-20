# Patch K1: copy INT8 per-token-head KV scales on copy-on-write block copies

The design note for [`../10-K1-int8-kv-scale-cow.py`](../10-K1-int8-kv-scale-cow.py), written before night 4. Line numbers refer to the unpatched night-3 tree (`/opt/vllm-qfn-exp`). "Patch 16" below is **2dameneko's** patch 16 (`kv_int8_qsa`), not ours. Section 6 says what actually ran in night 4.

## 1. Diagnosis (confirmed)

How a follow-up gets stale scales:
1. `_has_partial_local_hit` (single_type_kv_cache_manager.py L168-177) marks a hit as partial when it does not end on a
   block edge; L316-323 record the shared tail block.
2. With MTP and fine-grained hashing, `find_longest_cache_hit` drops one 32-token unit for the drafter (L831-833), so
   even a hit on a block edge ends inside the previous block: nearly every follow-up becomes a copy-on-write (CoW).
   The prompt-tail cache entry (`_cache_partial_tail_block`, L861-889) also sits inside a block. Without
   --prefix-match-unit, hits stay block-aligned and no CoW happens (why E4, INT8 without the prefix-match flags, was
   correct).
3. `allocate_new_blocks` (L384-394) takes a fresh block (zeroed); `_apply_cow` (L451-471) swaps it in and queues
   `(src, dst)`.
4. The pairs reach the worker as `SchedulerOutput.kv_cache_block_copies` (kv_cache_manager.py L888-902,
   scheduler.py L1470-1479).
5. Both runners zero the new blocks, then call `copy_kv_cache_blocks_inplace` (MRV2 gpu/model_runner.py L1202-1216;
   V1 gpu_model_runner.py L1241-1250).
6. `copy_kv_cache_blocks_inplace` (v1/worker/utils.py L704-750) copies whole blocks, but only of the tensors in
   `kv_caches`.

Scale cache layout (2dameneko patch 16, qsa.py `_pth_caches` L205-233): `torch.ones`, `[kernel blocks, block_size, kv_heads]`
fp32 (= `[N*ratio, 3200, 2]`), one K/V pair per QSA layer impl, outside the KV page; written with the INT8 data
through slot_mapping, read by physical page / slot / head. Only teardown touched them (`clear_layer_kv_caches` sets
them to None but leaves `_pth_key`).

Partial-block semantics: copying the same full scale rows for the same pairs (same kernel-block folding) is exactly
right: every destination slot then holds the source's data and scales; later slots are overwritten (data + scales)
before they are read; zeroing runs before the copy; Mamba CoW pairs in the same list are harmless.

All KV block copy paths: CoW MRV2 runner (patched), CoW V1 runner (patched), HiSparse host copy (unused, left alone),
new-block zeroing (KV only), Mamba state copies (not quantized), CPU offload connectors (unused; QSA backend
`supports_kv_connector()` is False).

MTP draft layers have the same bug: full-attention Qwen4ExpDecoderLayer with INT8 cache dtype build
Qwen4ExpQSAAttention, own scale caches, same attention group and copy pairs -> covered by K1. Probably part of the lower
MTP acceptance in the earlier INT8 run (2.50 vs 2.77).

vllm#55112 (partial copy when a narrower Mamba/indexer view shares the buffer address) is separate; BF16 has the same
exposure and passes. K1 includes a fix, off by default (`VLLM_K1_FIX_55112=1`); debug mode detects it.

## 2. The patch

`K1-int8-kv-scale-cow.py`: 4 files, 8 anchors (each asserted exactly once; sources compiled before writing; files
containing "Asgard patch K1" skipped).
- v1/worker/utils.py: `register_kv_block_aux_tensors(owner, *attrs)` (weak refs, attrs read at copy time, None
  skipped); `_k1_copy_aux_block_tensors` after the KV loop (same index tensor, same folding, gather-then-scatter on the
  same stream; non-multiple first dims skipped with a one-time warning; dead owners pruned); keyword-only
  `copy_aux_block_tensors=False` (only the two runners pass True). Logging: one INFO on registration, then
  `Asgard patch K1: CoW step N: ...` on the first copy and every `VLLM_K1_LOG_EVERY` (default 50).
- Test switches: `VLLM_K1_DEBUG=1` (log every copy, count stale rows, verify every KV view dst == src; syncs - tests
  only), `VLLM_K1_DISABLE=1` (skip the scale copy = reproduce the bug, instrumented), `VLLM_K1_FIX_55112=1`.
- Both runners pass `copy_aux_block_tensors=True`.
- qsa.py `_pth_caches`: registers `_k_scale_cache`/`_v_scale_cache` after allocation (INT8/FP8 per-token-head only) and
  re-allocates them if teardown set them to None.
- BF16 unchanged: nothing registers, the copy is byte-identical (checked on 4 layouts); cost one dict check per copy.
- Risks: ~26 small index ops per CoW step; no persistent memory; debug mode syncs; anchors fail closed on rebase.

## 3. Memory

Scale caches sit outside vLLM's KV budget (created on the first QSA forward after the KV cache is bound). 16 bytes per
slot per QSA layer (1/64 of the INT8 KV): ~645 blocks, ~2.07M slots -> ~33 MB per QSA layer, ~0.20 GB rank 0 (6
layers), ~0.23 GB rank 1 (6 + MTP), ~0.43 GB total. No need to lower gpu-memory-utilization (about 6.3 GiB per GPU
unrequested at 0.90; these tensors already existed in all earlier INT8 runs). Expect INT8 used memory ~ BF16 + 0.2 GiB
per GPU, and >= 1 GiB free per GPU after the pinned-pool test.

## 4. Verification plan

This was the plan. Night 4 ran a subset of it under other profile names; see section 6.

Apply: back up the 4 files as .pre-k1, run `K1-int8-kv-scale-cow.py $SRC`, py_compile with the n4 venv python.
Applying K1 does not change the BF16 phases.

| Phase | Profile | Tests | Pass when |
|---|---|---|---|
| K1off | INT8 + DISABLE=1 DEBUG=1 LOG_EVERY=1 | multiturn --cold, multiturn_check 3 20000 | bug reproduces: hit turns fail, cold controls pass, "NOT copied" > 0, stale rows > 0, CoW steps > 0 |
| K1dbg | INT8 + DEBUG=1 | multiturn --cold, fanout 12, multiturn_check | all pass, 0 partial copies (else rerun with FIX_55112=1) |
| K1 | INT8 production-like | firstbatch, multiturn 6x5 --cold, multiturn_check, fanout 16, fill sweep, pressure churn 1.2x, pinned 1.5x, multiturn again, speed, func_check, MTP acceptance | all pass; follow-up TTFT ~0.17 s; acceptance >= ~2.7; peak KV >= 90%; no tracebacks/OOM; >= 1 GiB free per GPU |
| B0 | BF16 on the K1 tree | firstbatch/multiturn --block 1600, multiturn_check, speed, compare | 0 K1 registrations, 0 CoW lines, speed = production |

Coverage of the Gemma-4 stale-scale bug patterns: #51091 firstbatch, #50749 fill sweep, #50702 pressure, #49716
fanout + pressure.

Outcome: if K1off reproduces and the rest passes, INT8 with these flags is correct and gives a 1.53M-token KV pool
instead of 0.88M. Production is a separate decision (single-user decode was 4-8% slower with INT8).

## 5. Local checks

`test_k1_local.py`, `mock_server.py` and `k1-night.sh` are not published; `k1_cow_check.py` is [`../../bench/night4/k1_cow_check.py`](../../bench/night4/k1_cow_check.py).

Patch script on scratch copies (second run: already patched; compiles on Python 3.11-3.14). test_k1_local.py 36/36:
BF16 byte-identical, real `_pth_caches` allocates and registers, copied scale rows match (incl. folding), disable/debug
switches, #55112 reproduced/flagged/fixed. k1_cow_check.py works against mock_server.py (empty follow-ups correctly
fail). k1-night.sh syntax OK; mkprof builds the expected profiles.

## 6. What ran in night 4

All on the deploy candidate's settings (int4 draft head, MTP 4, `--prefix-match-unit 16`) plus `KV_DTYPE=int8_per_token_head`, clocks locked at 1440 MHz. Profiles: [`../../results/profiles/night4/K8off.env`](../../results/profiles/night4/K8off.env), [`K8dbg.env`](../../results/profiles/night4/K8dbg.env), [`K8.env`](../../results/profiles/night4/K8.env). Outputs: [`../../results/checks/night4/`](../../results/checks/night4/) (`K8*.txt`, `K8*.json`); summary lines in [`../../results/logs/night4-NIGHT-LOG.md`](../../results/logs/night4-NIGHT-LOG.md).

| Run | Plan name | Result |
|---|---|---|
| `K8off-mt` (`VLLM_K1_DISABLE=1`, debug) | K1off | **the bug reproduces**: 10 of 28 cache-hit turns failed, all 12 cold controls passed; 74 CoW steps with the scale copy skipped, 598 stale scale rows |
| `K8dbg-mt`, `K8dbg-fan` (debug) | K1dbg | multiturn and a 12-user fanout all pass; "NOT copied" 0, partial copies 0, tracebacks 0 |
| `K8-first`, `K8-mt`, `K8-churn` | K1 | first batch (12 users), multiturn 6 x 5 (22 of 24 follow-up hits mid-block) and pool churn all pass; median follow-up TTFT 0.17 s |
| `K8` speed | K1 | code decode 190.4 tok/s (acceptance 3.49) against 193.6 with BF16 KV; KV pool 1,433,217 tokens against 838,024 (+71%) |
| `n4-K8` quality | | long context 98.3%, the same as the BF16 reference `n4-ref` |

Not run: the fill sweep, the pinned 1.5x pool, fanout 16 and the BF16 control on the K1 tree (B0). The churn test peaked at **17% KV usage**, so eviction with a full pool is still untested.
