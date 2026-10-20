# Swift-A16: Swift 1.5 AWQ with BF16 attention

How the Swift-A16 checkpoint was built, written before night 4 with the scripts in this directory (`fetch_bf16_attention.py`, `build_swift_a16.py`, `vllm_scheme_check.py`). No weights are included.

**Night-4 result** (profile [`A16.env`](../../results/profiles/night4/A16.env), same night, same harness as the reference `n4-ref`):
- quality: BFCL 82.1% ([summary](../../results/quality/n4-A16/medium/bfcl.summary.json)) against 79.6%, long context 99.4% ([summary](../../results/quality/n4-A16/medium/longctx.summary.json)) against 98.3%, EvalPlus 85.8% against 86.0%;
- speed, clocks locked at 1440 MHz: code decode 173.7 tok/s ([`decode-n4-A16.json`](../../results/speed/decode-n4-A16.json)) against 193.6 (−10%); KV pool 723,684 tokens against 838,024.

Swift-A16 is the Swift 1.5 W4A16 AWQ checkpoint (`/opt/vllm-qfn/models/swift15-awq`) with one change. Its INT4 attention projections are replaced by the BF16 originals from Swift's own BF16 release. The routed experts stay INT4 (AWQ). Every other tensor is unchanged, byte for byte. Intel/Qwen3.8-Flash-Next-W4A16-AutoRound uses the same design, and its attention group comes to the same 5.04 GiB.

| | |
|---|---|
| BF16 source | `ukisai/Swift1.5-Qwen3.8-Flash-Next` @ `0bd4fe22431372cdad1979267d3ab45aa7e6150a` (the repo's only commit, "initial release"; the old name `ukisai/Swift-Qwen3.8-Flash-Next` redirects here). Not gated, so no token is needed. |
| What is fetched | 156 tensors from 36 of the 131 BF16 shards, using HTTP Range requests |
| Download | **5,347,737,600 bytes = 4.98 GiB**, plus 173 KB of shard headers and the 171 KB index |
| New disk space | **67.05 GiB** for 17 rewritten shards, plus the 4.98 GiB attention work file |
| Hardlinked (no extra space) | 34 files, 101.72 GiB. 32 of them are PLE-only (95.37 GiB): `model-00002`, `embedding-model-00006..00036`. The other two are `model-00019` and `model_mtp`. |
| Weights | INT4 attention out: 624 tensors, 1.29 GiB. BF16 attention in: 156 tensors, 4.98 GiB. **That is +3.69 GiB of GPU weights, about +1.84 GiB per PP rank at 24/24.** |
| config.json | 156 entries are appended to `quantization_config.ignore` (728 → 884). Nothing else changes. |

## Which tensors make up "attention"

The AWQ stores these modules as compressed-tensors INT4. Each has 4 parts: `weight_packed` I32 `[out, in/8]`, `weight_scale` BF16 `[out, in/128]`, `weight_zero_point` I32 `[out/8, in/128]` and `weight_shape` I64 `[2]`.

| Layers | Module (HF name, no `.weight`) | BF16 shape | vLLM module |
|---|---|---|---|
| 12 QSA layers (3, 7, …, 47) | `model.language_model.layers.L.self_attn.q_proj` | 12288×2560 | `qkv_proj` (fused q, k, v) |
| | `…self_attn.k_proj`, `…self_attn.v_proj` | 512×2560 | `qkv_proj` |
| | `…self_attn.o_proj` | 2560×6144 | `o_proj` |
| 36 GDN layers | `model.language_model.layers.L.linear_attn.in_proj_qkv` | 10240×2560 | `in_proj_qkvz` (fused qkv, z) |
| | `…linear_attn.in_proj_z` | 6144×2560 | `in_proj_qkvz` |
| | `…linear_attn.out_proj` | 2560×6144 | `out_proj` |

Every part of each fused vLLM module is switched together: q/k/v as one group, and in_proj_qkv/in_proj_z as one group. vLLM raises "Found different quantization schemes for the shards" when the shards of one fused module are mixed.

The rest of Intel's BF16 attention group is already BF16 and already in the ignore list in Swift AWQ. That covers `in_proj_a`/`in_proj_b` (fused `in_proj_ba`), `A_log`, `dt_bias`, `conv1d`, `norm`, `q_norm`/`k_norm`, the `indexer.*` modules (including `index_qk_proj`), shared experts, routers, HC, PLE, vision, lm_head and all MTP tensors. Those 0.056 GiB are byte-identical to the BF16 release (checked, see below). Swift-A16's attention group is therefore exactly Intel's 5.036 GiB.

Why swapping in BF16 is self-consistent: the AWQ recipe only smooths *inside each routed expert* (up_proj → down_proj), and attention got plain RTN (`QUANTIZATION_MANIFEST.json`: "RTN for attention projections"). No smoothing scales were folded into attention or the norms, so the BF16 originals are exactly what the INT4 tensors approximate.

### Provenance: the AWQ was quantized from this exact BF16 repo

The checks below were run from here against both HF repos. Only small byte ranges were read.

- AWQ `QUANTIZATION_MANIFEST.json` gives `source_repo: ukisai/Swift-Qwen3.8-Flash-Next` and `source_revision f0892ab…`, which is a pre-publication revision. The model card has `base_model: ukisai/Swift-Qwen3.8-Flash-Next`, which redirects to `ukisai/Swift1.5-Qwen3.8-Flash-Next`.
- All 31 `embedding-model-000NN-of-00131` files of the AWQ have the same sha256 as `model-000NN-of-00131` of the BF16 repo (31/31). The AWQ's `FILE_MANIFEST.json` was compared against the HF LFS oids.
- 30 of 30 preserved BF16 tensors are byte-identical between the two repos. They cover routers, `index_qk_proj`, `conv1d`, `in_proj_a`, `A_log`, `dt_bias` and the norms of layers 0, 1, 3, 7, 20 and 47.
- `config.json` is identical apart from `quantization_config`. The AWQ's `weight_shape` values equal the BF16 shapes.
- `layers.3.self_attn.k_proj` was dequantized (INT4 g128 asym) and compared with the BF16 tensor. The relative RMS error is 0.13, which is typical RTN error for 4-bit. A wrong layout or orientation would give about 1.4.

## Files

- `fetch_bf16_attention.py`: the downloader. Standard library only (urllib).
- `build_swift_a16.py`: the builder and verifier. Standard library only.
- `vllm_scheme_check.py`: optional. It replays vLLM's ignore and fused-module matching on a config, with no GPU needed.

## Commands (on the server, LXC 232)

### 0. Preflight

```bash
df -h /opt/vllm-qfn/models            # need >= 75 GiB really free: 4.98 + 67.05 + margin
# On the Proxmox HOST: the LVM thin pool has gone full twice before (ext4 went emergency_ro, zero-byte files).
# df inside the container does not show thin-pool overcommit:
lvs -o lv_name,data_percent,metadata_percent,lv_size   # pool Data% must leave > 75 GiB
mkdir -p /opt/vllm-qfn/models/.a16-work
# copy fetch_bf16_attention.py, build_swift_a16.py and vllm_scheme_check.py into it
```

The work directory sits on the same filesystem as the models, like `.q6x-cache`. vLLM only globs inside each model directory, so a hidden sibling directory is never loaded.

### 1. Plan only (a few hundred KB, about 1 minute)

```bash
cd /opt/vllm-qfn/models/.a16-work
python3 -I fetch_bf16_attention.py --work-dir /opt/vllm-qfn/models/.a16-work \
    --awq-dir /opt/vllm-qfn/models/swift15-awq --plan-only
```

Expected output:
- `156 attention tensors in 36 of 131 BF16 shards` and `total to download: 5347737600 bytes = 4.980 GiB`.
- `AWQ config.json == BF16 config.json`.
- `AWQ cross-check: all 156 BF16 shapes equal the AWQ weight_shape values`. This reads `weight_shape` from your local files.
- `provenance: 12/12 preserved tensors byte-identical`.

### 2. Fetch (4.98 GiB)

Run it as a transient unit, never with nohup over ssh:

```bash
systemd-run --unit swift-a16-fetch --collect -p WorkingDirectory=/opt/vllm-qfn/models/.a16-work \
    /usr/bin/python3 -I /opt/vllm-qfn/models/.a16-work/fetch_bf16_attention.py \
    --work-dir /opt/vllm-qfn/models/.a16-work --awq-dir /opt/vllm-qfn/models/swift15-awq --jobs 4
journalctl -u swift-a16-fetch -f
```

How the fetch behaves:
- **Resumable.** If it is interrupted, run the same command again. Finished tensors are kept, and a partly downloaded tensor continues from its last 16 MiB checkpoint, which is fsynced and recorded in `*.state.json`.
- The output is preallocated at its final size, so no second copy is ever written.
- Each tensor's request first goes to `huggingface.co/…/resolve/<rev>/<shard>`, which answers with a 302 to a signed CDN URL. Signed URLs expire after about an hour, so every retry resolves again.
- The script accepts only `206` responses, and the `Content-Range` must match exactly. It never downloads a whole shard.
- `HF_TOKEN` is used if set, and is stripped on the cross-host redirect.
- Retries back off exponentially (12 attempts by default) and honour `Retry-After` on 429.

At the end the script verifies the result. It checks that the tensor set matches the plan, that every dtype and shape is right, and that the offsets are contiguous with no holes and the exact file size. It also computes a sha256 for each tensor and samples BF16 values to confirm they are finite and not all zero. Two files are written:
- `swift15-bf16-attention.safetensors` (5,347,759,104 bytes)
- `swift15-bf16-attention.manifest.json`, with the source shard, byte range and sha256 of each tensor.

### 3. Build: dry run, then the real build

```bash
cd /opt/vllm-qfn/models/.a16-work
python3 -I build_swift_a16.py --src /opt/vllm-qfn/models/swift15-awq \
    --attn /opt/vllm-qfn/models/.a16-work/swift15-bf16-attention.safetensors \
    --dst /opt/vllm-qfn/models/swift15-a16 --dry-run --emit-config /tmp/swift15-a16-config.json

systemd-run --unit swift-a16-build --collect -p WorkingDirectory=/opt/vllm-qfn/models/.a16-work \
    /usr/bin/python3 -I /opt/vllm-qfn/models/.a16-work/build_swift_a16.py \
    --src /opt/vllm-qfn/models/swift15-awq \
    --attn /opt/vllm-qfn/models/.a16-work/swift15-bf16-attention.safetensors \
    --dst /opt/vllm-qfn/models/swift15-a16
journalctl -u swift-a16-build -f
```

What the builder does:
- **Checks first.** It verifies the source index against the shard headers: every tensor appears exactly once, sizes match shapes, there are no holes, and each file is exactly as long as its header says. That last check catches the zero-byte or truncated files the full thin pool produced before. It also checks that the attention file covers exactly the 156 modules that `layer_types` requires, that the AWQ `weight_shape` data and packed shapes match the BF16 shapes, that free space (`df`) is enough, and that source and destination are on the same filesystem so hardlinks work.
- **Builds in `swift15-a16.building`** and renames it to `swift15-a16` only after verification passes. It refuses to start if either name already exists.
- **Hardlinks** every file that has no INT4 attention.
- **Rewrites 17 shards** (`model-00001`, `00003`…`00018`). The other tensors are copied raw, the INT4 attention parts are dropped, and the BF16 `.weight` of the same modules goes into the same shard. That keeps the per-layer locality for PP: each rank still reads only its own shards.
  - The tensor order is the same as the Rust safetensors writer uses (dtype size descending, then name), so every tensor stays aligned. The header is padded to 8 bytes.
  - Every 512 MiB written, the builder flushes and drops the page cache (`fadvise DONTNEED`) for both source and destination. 67 GiB of copying therefore doesn't fill the container's memory cgroup next to a running vLLM.
  - The builder never rewrites a shard that holds PLE n-gram tensors; it refuses instead.
- **Regenerates `model.safetensors.index.json`** from the written headers, writes the new `config.json`, and copies the small files. `FILE_MANIFEST.json` is stored as `FILE_MANIFEST.swift15-awq.json` because its sha256 values are stale for the rewritten shards. `SWIFT_A16_BUILD.json` records what was linked, rewritten, dropped and added, with a sha256 for each written shard.
- **Verifies the result**:
  - Every tensor in the index sits in exactly one file and is listed for that file. No tensor is lost or duplicated, and the set equals the source minus the INT4 attention plus the BF16 attention. Dtypes and shapes match their sources.
  - Hardlinks share the inode with the source. All PLE tensors stay in the same file names, at the same `data_offsets`, with the same header length.
  - `config.json` differs from the source only in `quantization_config.ignore`, and it is replayed with vLLM's matching rules.
  - The bytes of every tensor in the rewritten shards are compared with their sources: small tensors in full, large ones by the first and last MiB.

  Optional deep check, which reads 2 × 67 GiB and also checks the sha256 of every BF16 tensor against the fetch manifest:
  ```bash
  python3 -I build_swift_a16.py --src … --attn … --dst /opt/vllm-qfn/models/swift15-a16 --verify-only --verify-bytes full
  ```

`--placement extra` puts all BF16 attention into one extra shard, `model-a16-attention.safetensors`, instead. Both modes are tested. The default is better for PP and for the instanttensor/fastsafetensors loaders, which load whole files.

### 4. Check how vLLM will resolve the schemes (no GPU, the server keeps running)

```bash
/opt/vllm-qfn-n4/.venv/bin/python vllm_scheme_check.py /opt/vllm-qfn/models/swift15-a16
```

Expected: `QSA self_attn.qkv_proj` and `o_proj`, and `GDN in_proj_qkvz` and `out_proj`, are all **BF16 (ignored)**. `routed experts` show 144 entries as **INT4 (quantized)**. The last line is `RESULT: OK`. Run against `swift15-awq` instead, the attention shows INT4.

The script imports vLLM's own `should_ignore_layer` and the Qwen4Exp class mappings when they load. Otherwise it loads vLLM's pure-Python `config_utils.py` from the installed package. Only that fallback path was tested here, against our n4 tree, because there is no torch locally.

### 5. vLLM smoke test

This step stops production for the test (the GPUs move to the test unit). Night-4's P9 in [`run.sh`](../../bench/night4/run.sh) already does this with `A16=/opt/vllm-qfn/models/swift15-a16`. To run it by hand:

```bash
N=/root/tune/night4
python3 $N/mkprof.py $N/qfn-prod.env.baseline $N/profiles/A16.env venv=n4 port=8008 label=A16 \
    model=/opt/vllm-qfn/models/swift15-a16
/opt/vllm-qfn/qfn-window.sh up $N/profiles/A16.env
```

In `/opt/vllm-qfn/logs/vllm.log`, check four things:
1. There is no `Found different quantization schemes`, no unexpected or missing weight error, and no `KeyError ... weight_packed`.
2. `Model loading took X GiB`. Each PP rank should be about 1.8 GiB above the Swift AWQ run.
3. `GPU KV cache size` is a bit smaller than with Swift AWQ. P9 retries with `maxlen=196608` if the server doesn't start.
4. The PLE-SSD table opens with the same row count as before. Its files are the hardlinked originals.

Then:
- **Quick prompts.** `curl` a chat completion and a tool call. Night-4's `greedy_identity.py` can compare greedy outputs with Swift AWQ, which should be close but not identical.
- **Quality.** Run `suite n4-A16 "bfcl evalplus longctx"`. The target is Intel's +3.0 pt BFCL and +3.9 pt long-context over Swift AWQ.
- **Speed.** Run `speed.sh A16`. Decode will be slower (see the risks below), so compare it with `intel-autoround`, which has the same attention size.

For DEBUG-level confirmation, `VLLM_LOGGING_LEVEL=DEBUG` logs `Using scheme: CompressedTensorsWNA16 for …` only for quantized linears. With A16, no `qkv_proj`, `o_proj`, `in_proj_qkvz` or `out_proj` lines should appear.

## Our vLLM tree: what loads this checkpoint, and why it works

| Item | Where (n4tree/vllm) |
|---|---|
| `ignore` matching is exact string, or `re:` + `re.match`. It is applied to the vLLM name after the class's `hf_to_vllm_mapper` renames (`model.language_model.` → `language_model.model.`), and both the ignore entries and the layer names get that rename. So exact HF module names, like the 728 already there, are the correct form. | `model_executor/layers/quantization/utils/config_utils.py:253,301`; `compressed_tensors.py:123-151` (`apply_vllm_mapper`); `model_loader/utils.py:285-305` |
| A fused layer is ignored only if **all** of its shards match. Mixed shards raise an error. | `compressed_tensors/utils.py:53-70` |
| An ignored layer gets no scheme, so it uses `UnquantizedLinearMethod` (BF16 `weight` parameter, weight_loader_v2). | `compressed_tensors.py:153-170, 893-922, 940-958`; `layers/linear.py:49` |
| Fused names used: QSA `qkv_proj` (q, k, v) and `o_proj`; GDN `in_proj_qkvz` (in_proj_qkv, in_proj_z), `in_proj_ba` and `out_proj`. | `models/qwen4_exp/nvidia/qsa.py:412-427`; `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:431-503`; packed mapping in `models/qwen4_exp/nvidia/model.py:662-673,901` |
| Checkpoint → parameter mapping for BF16. `.q_proj/.k_proj/.v_proj` → `.qkv_proj` (shards q, k, v). `.in_proj_qkv` → `.in_proj_qkvz` (tuple shard ids 0, 1, 2) and `.in_proj_z` → shard 3. The unquantized MergedColumn and QKV weight loaders handle tuple shard ids. This is the same path the AWQ already uses for the BF16 `in_proj_ba`, and Intel's BF16 attention loads through it on our vLLM. | `model_executor/models/qwen3_next.py:624-634`; `qwen3_5.py:220-227`; `layers/linear.py:742,772,914` |
| Routed experts stay INT4. `CompressedTensorsMoEMethod` checks `<prefix>.experts.0.{gate,up,down}_proj`, and nothing in the new ignore list matches them. | `compressed_tensors_moe/compressed_tensors_moe.py:27-54` |
| MTP: unchanged. It is BF16 through `re:^mtp.*`, its file is hardlinked, and the new exact names (layers 0–47) cannot match MTP layer 48. | `models/qwen4_exp/nvidia/mtp.py:173-219` |
| No code reads `qkv_proj.weight` or `in_proj_qkvz.weight` directly or assumes `weight_packed`. The low-latency skinny-GEMM swap applies only to unquantized linears on SM90/SM103, so it has no effect on the SM80 CMP 170HX. | `qsa.py`, `qwen_gdn_linear_attn.py`; `models/qwen4_exp/nvidia/low_latency_gemm.py:155-162,226-263` |
| File list: vLLM globs `*.safetensors` and keeps only the files named in the index. A missing indexed file is a hard error. The builder writes the index from the real headers and fails on any stray `*.safetensors`. | `model_loader/weight_utils.py:612-638`; `default_loader.py:212-238` |

### What the PLE-SSD loader (`models/qwen4_exp/nvidia/ple_ssd.py`) needs, and why it all still holds

- **Its own index read.** `PLESSDTable` reads `<model>/model.safetensors.index.json` and selects the keys containing `layers.N.ple.ple_embedding.ngram_embedding.shard_` (`ple_ssd.py:288-293`). It needs shard indices 0..127 contiguous, BF16 `[rows, 160]`, `end-begin == rows*row_bytes`, and the file long enough for `data_start+end`. It then reads rows at `data_start + begin + row*320` with pread (`_read_rows`, `:415`) or O_DIRECT AIO (`PLESSDNativeReader`, `:185-249`). The checks are at `:296-349`. The new index maps all 128 PLE keys to the **same file names**, which are hardlinks of the original inodes, so header lengths and `data_offsets` are unchanged and O_DIRECT alignment is untouched. The builder verifies this.
- **The weights iterator.** `ple_ssd_weights_iterator` (`:34-…`) sorts files by header. All-PLE files are yielded as meta tensors, mixed files go through `safe_open`, and all other files take the normal or direct-I/O path. The rewritten shards contain no PLE tensors, which the builder enforces, so they are ordinary files. `model-00002` and `embedding-model-*` stay all-PLE. `model-00003` (the PLE projections and `ngram_heads_*`, no n-gram shards) was already an ordinary file and stays one.
- **No assumptions about names or positions.** The loader assumes nothing about shard names, the shard count, `-of-000NN`, or the order of files beyond the index. Headers are about 2.5 MB, well under its 100 MB limit.

## Risks

1. **Disk and the thin pool.** About 72 GiB of real new blocks. The script checks `df` with a 2 GiB margin, but it cannot see an overcommitted LVM thin pool on the host. Check `lvs` first (step 0).
2. **Hardlinks share content.** 101.7 GiB is shared between `swift15-awq` and `swift15-a16`. Deleting either directory is safe, but space is only freed when both names are gone. **Never modify a shard in place** in either directory, for example with an in-place PLE FP8 conversion: tools that write new files are fine. `du -sh a b` counts shared files once, while `du` on one directory alone counts them fully.
3. **Decode speed.** Attention weights read per decoded token grow from 1.39 to 5.35 GB, through cuBLAS BF16 GEMV on SM80. Expect decode tok/s noticeably lower than Swift AWQ, in Intel AutoRound's range. Prefill is compute-bound and changes much less. Measure it with `speed.sh`.
4. **VRAM.** About +1.84 GiB per GPU at the 24/24 split, so the KV and mamba pool shrinks. P9 already has the 196K-context fallback.
5. **Source availability.** The revision is pinned. If ukisai gates or deletes the repo, set `HF_TOKEN` or the fetch fails. The fetch is resumable, and nothing is half-written into model directories.
6. **No end-to-end GPU test was possible here.** Steps 4 and 5 are the real load tests. The scripts were tested as listed below.

## How the scripts were tested (locally, no server access)

The test fixtures (`_test/`, the header dumps) are not published.

- **Real HF data.** `--plan-only` ran against `ukisai/Swift1.5-Qwen3.8-Flash-Next`. HTTP Range through the 302 to the HF CDN returns `206` with a correct `Content-Range`. All 36 parsed headers imply exactly the file sizes the HF API reports. The provenance checks above ran against both HF repos.
- **Real Swift AWQ headers.** `build_swift_a16.py --src-dump ckpt-dumps/swift15-awq-dump.json --attn-plan plan.json --dry-run` ran on the header dump (296,942 tensors). It produced the numbers in the table above. The emitted config was replayed with vLLM's real `find_matching_patterns` from the n4 tree: the attention modules come out ignored (12/12 qkv_proj, 12/12 o_proj, 36/36 in_proj_qkvz, 36/36 out_proj), the experts stay quantized, and there is no mixed-scheme error.
- **Synthetic end-to-end** (`_test/`). A tiny Swift-like BF16 repo and AWQ were generated (4 layers, PLE-only shards, an MTP file, a `.cache` directory), plus a fake HF server with a 302 redirect, Range support and fault injection.
  - Fetcher, with dropped connections and 403s injected: it retries, never sends the token to the CDN host, and its output is byte-identical to the source tensors.
  - Hard `kill -9` during the download, then a rerun: it resumes mid-tensor from the checkpoints, and the result is identical to a clean run.
  - Builder: `inplace`, `extra` and `--copy-unchanged` builds pass `--verify-bytes full`. The official `safetensors` library also parsed every output file. Every tensor equals its source bytes, the I32 and I64 data is aligned, and the PLE files are the same inodes.
  - Builder refusals, each of which stops it: an existing dst; an existing `.building`; a truncated or zero-byte source shard; an attention file missing a tensor or with a wrong shape; a stray `*.safetensors` in dst; an already-converted source.
