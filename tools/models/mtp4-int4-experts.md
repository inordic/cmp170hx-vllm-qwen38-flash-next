# INT4 MTP experts for our two Qwen3.8-Flash-Next checkpoints

How the `-mtp4` checkpoints were built, written before night 4 with `intel_mtp4_install.py` and `swift_mtp4_build.py` from this directory. No weights are included.

**Night-4 result.** On the deploy candidate (int4 draft head, MTP 4, `--prefix-match-unit 16`), Swift with INT4 MTP experts (profile [`MB.env`](../../results/profiles/night4/MB.env)) gave:
- code decode 186.8 tok/s against 193.6 and code-edit 307.9 against 303.7, clocks locked at 1440 MHz ([`decode-n4-MB.json`](../../results/speed/decode-n4-MB.json), [`edit-MB.txt`](../../results/speed/night4/edit-MB.txt)); acceptance 3.42 against 3.51;
- KV pool 1,054,153 tokens against 838,024 (+26%).

The combined decode score was 1.1% lower, so the night's 1% rule rejected it. It stays a capacity option. Intel with INT4 MTP experts ran as `I_best` (160.8 tok/s code decode).

Both checkpoints ship the MTP drafter as BF16, 4.856 GiB. 96.5 % of that is the 512 routed experts of the
single MTP layer. Quantizing only those experts to INT4 (RTN, group 128) shrinks the drafter to 1.39 GiB and
**frees 3.47 GiB (3.73 GB) of weights on the GPU that holds the drafter**, which is the last PP stage. The
target model is untouched. Speculative decoding is lossless, because the target still verifies every token, so
the only things that can change are draft acceptance and speed.

| | MTP tensors before | after | file on disk | new disk usage |
|---|---:|---:|---|---:|
| Intel AutoRound (hampsonw delta, GPTQ sym) | 5,214,301,696 B | 1,488,580,096 B | `model_extra_tensors.safetensors` 1,489,128,680 B | ~1.52 GB (+24 MB index) |
| Swift 1.5 AWQ (built here, compressed-tensors asym) | 5,214,301,696 B | 1,488,604,672 B | `model_mtp_int4.safetensors` ~1.49 GB | ~1.52 GB (+index) |

Every other file in the new directories is a **hardlink** of the source (same inode, byte-identical). That
includes the PLE shards that the PLE-SSD loader reads by `data_offsets`. Neither script modifies the source
directories. Each one builds into `.<dst>.staging-<pid>`, checks the result and renames it at the end. Neither
one overwrites an existing destination.

Files in this directory:

| file | runs with | what |
|---|---|---|
| `intel_mtp4_install.py` | any `python3` (stdlib only) | Intel: download + verify the hampsonw delta, build `/opt/vllm-qfn/models/intel-autoround-mtp4` |
| `swift_mtp4_build.py` | vLLM venv python (torch CPU, numpy) | Swift: quantize the MTP experts, build `/opt/vllm-qfn/models/swift15-awq-mtp4` |

The local tests (`test_local.py`, section 6) and the downloaded reference material are not published.

---

## 1. Intel: the hampsonw delta

**Repo:** `hampsonw/Qwen3.8-Flash-Next-W4A16-AutoRound-MTP-INT4-Experts` at revision
`b898827562533ebb0e94b1b0cde0acf8062acc6b` (2026-09-02). The delta was built from
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@861536dd`. Intel `main` (`4c67bf68`) has identical model files; only
the README changed.

| delta file | size | sha256 | use |
|---|---:|---|---|
| `model_extra_tensors.safetensors` | 1,489,128,680 | `674e7102…6fc2db2fa` | **replaces** Intel's 5.21 GB BF16 MTP sidecar |
| `config.json` | 106,641 | `6aec450f…f8be7925` | reference only (see below: unusable as-is in our tree) |
| `model.safetensors.index.json` | 24,063,625 | `6bcb9d76…cb8df7f` | reference: the portable index (keeps the 128 PLE entries) |
| `model.safetensors.index.ple-mmap.json` | 24,047,863 | `71dfe837…0084c` | **do not use**: it drops the PLE entries, and our PLE-SSD resolves PLE through the index |
| `SHA256SUMS`, `DERIVATION.txt`, `README.md`, `tools/quantize_qwen38_mtp_experts.py`, `evidence/*` | small | in `SHA256SUMS` | provenance |

**Contents of the delta:**
- The 1536 BF16 expert weights `mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight` become 4608 GPTQ
  tensors `.qweight` I32 `[K/8, N]`, `.scales` F16 `[K/128, N]` and `.qzeros` I32 `[K/128, N/8]`. The qzeros are
  the constant `0x77777777`, which means symmetric. This is exactly the layout of Intel's own routed experts.
- The other 29 MTP tensors are unchanged. The installer verifies byte-identity after the download.
- `config.json` has the 1536 per-expert exclusions `"mtp.layers.0.mlp.experts.E.X_proj": {"bits": 16}` removed
  and one entry added: `"mtp.layers.48.mlp.experts": {"bits": 4, "group_size": 128, "sym": true, "data_type": "int"}`.
  `block_name_to_quantize` and the `.*mtp.*` regex stay. The exact key wins.
- The index has the 1536 `.weight` entries replaced by the 4608 new names, all in `model_extra_tensors.safetensors`.
  `total_size` drops by 3,725,721,600.

**Does it match our checkpoint?** Yes, verified locally against our header dump:
- All 17 safetensors files have Intel@861536dd's exact sizes.
- All 17 safetensors **headers** match the HF files byte for byte: names, dtypes, shapes and `data_offsets`.
  I range-fetched every header from HF and compared it with `intel-autoround-dump.json`.
- Our `config.json`, `quantization_config.json` and `generation_config.json` equal the revision's files. Our
  README is the `main` one (2418 B), and `main` has the same LFS sha256 for every model file.
- Our index is a prefix match of Intel's index as far as the dump goes (the dump was cut at 200 kB).
- The delta's 29 non-expert MTP tensors have the same names, dtypes and shapes as ours.
- On the server the installer also checks the sha256 of our `model_extra_tensors.safetensors` against
  `e9e4786a…` (the `source_sidecar_sha256` in DERIVATION.txt). `--verify-base all` hashes every shard.

**Key finding: hampsonw's `config.json` does not work in our tree.** `vllm/models/qwen4_exp/nvidia/mtp.py`
(`_make_draft_vllm_config`) renumbers every `mtp.*` key of the draft's INC `extra_config` from the checkpoint
index to the runtime index (`0 -> 48`). hampsonw wrote the runtime name `mtp.layers.48.mlp.experts`, because
the Whamp runtime does not renumber. In our tree that key becomes `mtp.layers.96…` and matches nothing. The
experts then fall back to the `.*mtp.*` 16-bit regex and get a BF16 MoE, and loading fails on `w13_qweight`.
I confirmed this by running the tree's own `INCConfigParser` from `inc/config_parser.py` (test 7).

The installer therefore derives `config.json` from **our** config and writes the INT4 entry under **both**
`mtp.layers.0.mlp.experts` (correct for our tree) and `mtp.layers.48.mlp.experts` (correct for a runtime without
the remap). It checks that the result equals hampsonw's config plus that one extra key. It derives the index from
our index too, and checks that the result is exactly hampsonw's portable index.

**What the installer does:**
1. Checks the source sizes and the sidecar sha256 (`--verify-base sizes|sidecar|all`), and whether config and
   index are pristine.
2. Downloads into `--cache` with urllib. The download resumes over Range, retries on dropped connections and
   pins size + sha256 for every file. It also cross-checks the pins against the repo's `SHA256SUMS`.
3. Derives the config and the index, then simulates our tree's INC resolution for every draft layer: experts W4
   g128 sym, everything else 16 bit. With `--vllm-python PY` it also asks the installed vLLM's
   `INCConfigParser`.
4. Validates the delta sidecar: the 29 non-expert tensors must be byte-identical to ours, and the 4608 expert
   tensors must have the GPTQ shapes and dtypes and symmetric qzeros.
5. Hardlinks the tree, writes `config.json`, `model.safetensors.index.json` and `MTP4_PROVENANCE.txt`, and
   hardlinks the sidecar from the cache (or copies it if the cache is on another filesystem).
6. Checks that every linked file and every PLE shard shares the source inode, then renames the staging dir.

## 2. Swift: `swift_mtp4_build.py`

**Swift's format** (from `config.json` plus a routed expert of the main model in the header dump):
compressed-tensors `pack-quantized`, one config group with `targets: ["Linear"]`, W4A16_ASYM (`symmetric: false`,
group 128, `zp_dtype: torch.int8`, `memoryless_minmax`), BF16 scales. **Swift uses zero points.** Each main
expert has, for N = out and K = in (gate/up 640x2560, down 2560x640):

```
weight_packed      I32  [N, K/8]     (code+8) in 0..15, element j of every 8 at bits 4j..4j+3
weight_scale       BF16 [N, K/128]
weight_zero_point  I32  [N/8, K/128] (zp+8), packed along the OUTPUT dim (pack_to_int32(..., packed_dim=0))
weight_shape       I64  [2]          [N, K]
dequant: w = (code - zp) * scale     (both stored with the +8 offset, so the offset cancels)
```

The MTP experts in `model_mtp.safetensors` are fused in HF v5 layout: `mtp.layers.0.mlp.experts.gate_up_proj`
BF16 `[512, 1280, 2560]` and `down_proj` `[512, 2560, 640]`. The model shard contains only the 31 `mtp.*`
tensors. vLLM splits `gate_up_proj` with `chunk(2, dim=1)`, so gate is the first 640 rows. That is how the BF16
drafter is loaded today.

**Recipes.** I found two community converters:
- **Minachist `make_mtp_int4.py`** (`Minachist/Qwen3.8-Flash-Next-INT4-Mixed-AutoRound`, `vllm-patch/{3x24GB+MTP,4x24GB}/`):
  compressed-tensors, but **symmetric** INT4 with an MSE clip search. It adds a `group_mtp` config group with a
  regex target. That works for their checkpoint because it targets layers explicitly and has no MTP ignore entry.
- **Saren** (`Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN`) and **hampsonw**
  (`tools/quantize_qwen38_mtp_experts.py`): GPTQ sym RTN for the auto-round format.

None of them fits Swift as-is: Swift needs asymmetric codes with zero points, its targets are class-based
(`Linear`, plus `RoutedExperts` added by vLLM) and its ignore list contains `re:^mtp.*`. The script is an
equivalent written for that format:
- **Quantization:** RTN with the compressed-tensors `calculate_qparams` math per group of 128:
  - lo = min(min, 0), hi = max(max, 0); scale = bf16((hi − lo) / 15), with 0 replaced by eps.
  - zp = clamp(round(−8 − lo/scale), −8, 7); code = clamp(round(w/scale) + zp, −8, 7).
  - All arithmetic is fp32 with round-half-even.
  - Optional `--clip-grid N` runs a per-group MSE search over N clip ratios from 1.0 to 0.70, like Minachist's.
    The default is 1 (plain min/max, matching Swift's observer and the community RTN results).
- **Output:** one new shard `model_mtp_int4.safetensors` with:
  - the 29 non-expert MTP tensors, copied raw and byte-identical;
  - per expert and projection, the four tensors above under **per-expert names**
    `mtp.layers.0.mlp.experts.{e}.{gate,up,down}_proj.weight_*`. These are the same names and the same format
    as Swift's main routed experts, so the loader path is the one the 48 main layers already use. Every
    planned dtype and shape is asserted against the main-model template.
- **Index:** every `mtp.*` entry points to the new shard. `model_mtp.safetensors` is neither indexed nor linked.
  vLLM loads every tensor of every indexed file, so the old shard must not stay. The 128 PLE entries and
  everything else are unchanged.
- **Config:** a single change. `ignore: "re:^mtp.*"` becomes `"re:^mtp\.(?!layers\.\d+\.mlp\.experts\.).*"`.
  Everything else in the MTP stays ignored (fc_embedding, fc_hidden, qkv/o_proj, indexer, shared expert, gate,
  hyper-connections). The regex matches no main-model module (checked on all 75,124 module names). vLLM's
  matcher uses the `regex` module, which supports lookahead.
- Reads only the MTP shard (4.86 GB, pread per expert), plus 16 bytes and the zero points of one main expert.
  Peak RAM is far below 1 GB, and the output is written with pwrite. Expect about 1–3 min on CPU with
  `--clip-grid 1`, and roughly ×N with a clip grid.
- **Self-check** (before the rename; any failure aborts and removes the staging dir):
  - re-parses the new shard's layout (contiguous, aligned) and checks the 29 copied tensors byte for byte;
  - dequantizes the sampled experts (default 0, 1, 127, 255, 511; all three projections) **from the written
    file** with an independent bit decoder and reports relative error, cosine, max |err| and max error in
    quantization steps. It fails above 0.25 rel. error, below 0.96 cosine, or above 0.6 steps with the default
    grid;
  - checks that compressed-tensors' own `pack_to_int32`/`unpack_from_int32` reproduce our weights and zero
    points bit for bit, and that `safetensors.safe_open` reads the file back;
  - asks the installed vLLM's `CompressedTensorsConfig.get_scheme_dict` which draft modules get a scheme: only
    `mtp.layers.48.mlp.experts`, with W4 g128 asym;
  - confirms that hardlinks share the source inode, including all PLE shard files.
- `--selftest` (no checkpoint) checks the torch implementation bit for bit against a pure-Python reference
  implementation of the same fp32 math, including edge cases (all-zero, constant, one-signed and tiny groups, an
  outlier, sym and asym). It also runs the compressed-tensors and vLLM unpack cross-checks.
- `--crosscheck-intel /opt/vllm-qfn/models/intel-autoround` is optional. It compares Swift's gate/up split with
  Intel's per-expert BF16 MTP experts by cosine. If they are byte-identical, Swift's MTP was not retrained.

Expected error: synthetic Gaussian weights give about 0.10 relative Frobenius error (asym min/max). hampsonw
measured 0.126 NRMSE for symmetric absmax on the real Intel MTP experts. Asymmetric with a zero point should
come in at or below that.

**How our tree builds the MTP layer and its quant config** (`vllm/models/qwen4_exp/nvidia/mtp.py`):
- `Qwen4ExpMultiTokenPredictor` calls `_make_draft_vllm_config`. That calls `get_draft_quant_config`, which
  builds a fresh quant config from the same `config.json`. The draft keeps the top-level `quantization_config`,
  because `hf_config_override` only renames `model_type` and `architectures`.
- `configure_quant_config(…, Qwen4ExpMTP)` sets `packed_modules_mapping`. It applies no `hf_to_vllm_mapper`,
  because `Qwen4ExpMTP` is not `SupportsQuant`.
- `_remap_ignored_layers` renumbers `ignored_layers`, `exclude_modules`, `quantized_layers` and `extra_config`
  (`mtp.layers.0 -> 48`). **compressed-tensors' `ignore` is not renumbered.**
- The draft layer is `Qwen4ExpDecoderLayer(prefix="mtp.layers.48")`, from `mtp_start_layer_idx = num_hidden_layers`
  (the proposer passes prefix ""). Its MoE is `Qwen3NextSparseMoeBlock`, and `FusedMoEFactory` is built with
  `prefix="mtp.layers.48.mlp.experts"`.
- `CompressedTensorsConfig.get_quant_method` → `CompressedTensorsMoEMethod.get_moe_method` checks
  `mtp.layers.48.mlp.experts.0.{gate,up,down}_proj` against `ignore`:
  - **today** `re:^mtp.*` matches, so `UnquantizedFusedMoEMethod`;
  - **after** the change it is not ignored, so `RoutedExperts` matches target `Linear` (added by
    `_add_fused_moe_to_target_scheme_map`), so `CompressedTensorsWNA16MoEMethod` with zero points: the same
    method and kernel selection as the main layers.
- Loading: `mtp.layers.0.mlp.experts.E.gate_proj.weight_packed` → `model.layers.0.mlp.experts.E…` →
  `RoutedExperts.load_weights` → `w13_weight_packed` (likewise scale, zero_point and shape). This is the
  per-expert path the main layers use. Unexpected `.weight_scale` names are not in
  `_QWEN4_EXP_IGNORED_MISSING_SUFFIXES`, so a wrong method would fail loudly rather than drop weights.
- **Asgard patch D1** (int4 draft `lm_head`, `VLLM_MTP_DRAFT_HEAD_PATH`) is independent of this. It builds the
  head with its own `_QuantDraftHeadConfig` and fills it from its own file. Neither checkpoint's MTP shard
  contains an lm_head. The existing head files in `/opt/vllm-qfn/models/draft-heads/` are derived from the
  unchanged `lm_head` and work with the new directories. D1 and INT4 experts combine.

## 3. Commands (on the server)

Copy the two scripts from `tools/models/` to the server, e.g. `/root/tune/mtp4/`. `VPY` is the venv whose vLLM is the n4 tree;
adjust it if production uses another venv.

```bash
VPY=/opt/vllm-qfn-n4/.venv/bin/python
cd /root/tune/mtp4

# ---- Intel -----------------------------------------------------------------------------------
python3 intel_mtp4_install.py --dry-run --vllm-python $VPY      # small downloads (~24 MB) + all checks, builds nothing
python3 intel_mtp4_install.py --vllm-python $VPY                # 1.49 GB download, verify, build intel-autoround-mtp4
#   --verify-base all   also sha256 every source shard (~280 GB of reads, incl. the 102 GB PLE shard)
#   --offline           build from files already in the cache (/opt/vllm-qfn/models/.mtp4-cache/hampsonw-b8988275)
#   the cache can be deleted after the build: the sidecar in the new dir is a hardlink

# ---- Swift -----------------------------------------------------------------------------------
$VPY swift_mtp4_build.py --selftest                             # torch == reference, library cross-checks
$VPY swift_mtp4_build.py --dry-run                              # plan + ignore/index checks + installed-vLLM check
nice -n 10 ionice -c3 $VPY swift_mtp4_build.py --threads 8 \
     --crosscheck-intel /opt/vllm-qfn/models/intel-autoround    # build swift15-awq-mtp4 (cross-check optional)
#   --clip-grid 16   MSE clip search (lower error, ~16x CPU time)
```

The build is CPU-only and does not touch the GPUs or the running service. It reads 4.9 GB and writes 1.5 GB, so
run it at a quiet time or with `nice`/`ionice` as above.

## 4. Validation after the build

1. **Weights:** `python3 /root/tune/night4/weight_budget.py /opt/vllm-qfn/models/swift15-awq /opt/vllm-qfn/models/swift15-awq-mtp4`
   should show the `mtp` group drop from 4.86 GiB BF16 to 1.39 GiB (I32 1.18 + BF16 0.21). Do the same for the
   Intel pair.
2. **Serve a test profile** on port 8008. This is the same flow as night 4; production on 8080 is untouched:
   ```bash
   N=/root/tune/night4; P=$N/profiles; BASE=$N/qfn-prod.env.baseline
   python3 $N/mkprof.py $BASE $P/M4.env venv=n4 port=8008 label=M4 model=/opt/vllm-qfn/models/swift15-awq-mtp4
   python3 $N/mkprof.py $BASE $P/I4.env venv=n4 port=8008 label=I4 model=/opt/vllm-qfn/models/intel-autoround-mtp4
   /opt/vllm-qfn/qfn-window.sh up $P/M4.env      # (head=... can be added as before for D1)
   ```
   In `/opt/vllm-qfn/logs/vllm.log`, the last PP rank's `Model loading took X GiB` should be about 3.47 GiB lower
   than with the old dir (T0), and `GPU KV cache size: N tokens` should be higher (see the PP note below). A
   wrong quant method would fail at load (`no parameter 'w13_weight_packed'` / `'w13_qweight'`). Run
   `nvidia-smi` per GPU before and after.
3. **Acceptance and speed:** run `bash $N/speed.sh M4` (and `I4`) and compare with T0 and the Intel BF16-MTP
   run. Look at `accept` (mean acceptance length from `decode_code.py`) and code / edit decode tok/s. Community
   numbers: Saren 88.0 % → 88.5 % (MTP 3), hampsonw 62.6 % → 63.5 % (K2); decode +2–4 %.
4. **Losslessness:** run `python3 $N/greedy_identity.py collect M4 8008` and then
   `python3 $N/greedy_identity.py compare T0 M4`. Expect 16/16, or 1–2 flips from near-ties. Then run
   `python3 $N/canaries.py stalerows 8008` and the func/multiturn checks that speed.sh already does.
5. **Rollback:** point `MODEL` back at the old dir. `rm -rf` of a `-mtp4` dir only drops its links and new files;
   the source dirs are never written.

## 5. VRAM, PP and risks

- **VRAM:** −3.47 GiB of weights on the GPU that holds the drafter. Marlin repacking briefly needs about one more
  copy of the INT4 experts (~1.2 GiB) during load, before KV is profiled. In vLLM PP the KV block count is the
  **minimum over the stages**, so the freed memory only becomes KV tokens if the last stage was the binding one.
  It probably was, since it carries the MTP, lm_head and the final mixer. If stage 0 becomes the limit, move
  layers to the last stage with `split=A,B` (`VLLM_PP_LAYER_PARTITION`). One main layer's INT4 experts are about
  1.3 GB. With a fixed `--kv-cache-memory-bytes` you have to raise the value yourself.
- **Acceptance:** RTN on the drafter only. Both community measurements show no drop, but our workload has to
  confirm it (step 4.3). The rollback is a one-line profile change.
- **Intel config:** the published `config.json` silently does not apply in our tree (see the key finding in
  section 1). The installer's config handles both runtimes. If a future tree drops the 0→48 remap, the `48` key
  still applies.
- **PLE-SSD:** unaffected. The PLE shards are hardlinks (same inode, same `data_offsets`), and the index keeps
  all 128 PLE entries. PLE-SSD finds the shards through `model.safetensors.index.json`, which is why the
  ple-mmap index must not be used. The new MTP shard is a regular safetensors file: PLE-SSD's iterator routes it
  to the normal (or fastsafetensors) loader.
- **PP loading:** `mtp.*` tensors are only read on the stage that owns the MTP (`ep_weight_filter.should_skip_weight`).
  The new shard follows the same name rule.
- **Hardlinks:** the new and the source dir share inodes. Never edit a shared file in place. Writing a new file
  (as `hf download` does) is safe. The new dir must be on the same filesystem as the source; both scripts
  refuse otherwise.
- **Swift dir is vLLM-only:** its MTP experts are now per-expert compressed-tensors tensors, not HF v5 fused
  BF16. This does not matter for serving.
- **Not testable locally:** the torch path of `swift_mtp4_build.py` (no torch here). It is covered on the
  server by `--selftest` (bit-exact against the pure-Python reference, which `test_local.py` checks end to end)
  and by the build's own self-check from the written file. There was also no real GPU load test. Section 4
  covers that.

## 6. Local tests (`python3 test_local.py [--online]`, not published)

70 checks pass (71 with `--online`):
- the packing and quantization math;
- a full Swift build of a synthetic checkpoint with the python backend: fused experts, a PLE shard, symlinks,
  subdirectories, every expert re-decoded from the file, refuse-to-overwrite;
- a Swift `--dry-run` on a sparse replica of `swift15-awq`, made from the real headers, config and offsets:
  1536 matrices, 4.856 → 1.386 GiB, ASYM/BF16 detected, the ignore rewrite checked against all 75k module names;
- the Intel derivation on the real metadata, equal to the hampsonw config (+1 key) and index;
- a full Intel install against a local HTTP server: resume from `.part`, a dropped connection, a tampered delta
  rejected, a non-expert byte mismatch rejected, hardlinks, refuse-to-overwrite;
- `--online`: an Intel `--dry-run` on a sparse replica with the real HF delta metadata;
- my ports of vLLM's matching logic against the **real functions** of our tree (`should_ignore_layer`,
  `find_matched_target`, `INCConfigParser._resolve_raw`): identical on 75,160 names and on the original, derived
  and hampsonw configs.
