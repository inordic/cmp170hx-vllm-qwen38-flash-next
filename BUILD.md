# Build: from two CMP 170HX cards to a running server

This is the route we took. The paths are ours: `/opt/vllm-qfn` for the build and the models, and `/opt/vllm-qfn-n4` for the patched tree that production runs since 2026-10-20 (until then `/opt/vllm-qfn-exp`, the tree with patches 01–07). Adjust them to your system. Each step says how to check it before you go on.

## 0. Hardware and host

| Item | Ours | Notes |
|---|---|---|
| Cards | 2× CMP 170HX | unlocked to 64 GiB and 74 SMs with the CMP unlocker driver (open driver 610.57.04 + DKMS). Both run VBIOS 92.00.6D.00.0A (300 W), see [`vbios/`](vbios/) |
| Slots | PCIe Gen2 ×8 / ×8 from the CPU | no P2P is needed: pipeline parallelism sends one activation per step |
| CPU / RAM | i7-8700K / 64 GB | the container gets 55 GB. The PLE table is **not** held in RAM |
| Disk | NVMe | ~170 GB for the checkpoint. The PLE table is read from it on every forward pass |
| Host OS | Proxmox VE 9.1.1, kernel 6.17.2-1-pve | GRUB: `intel_iommu=on iommu=pt iomem=relaxed`. `iomem=relaxed` is only needed for live HBM-clock tools |

On the host:
1. Install the driver.
2. Set the power limit at boot with [`config/host/gpu-power-limit.service`](config/host/gpu-power-limit.service). It sets 300 W on both cards; only do that with the 300 W VBIOS.
3. Optionally run [`config/host/gpu-thermal-guard.sh`](config/host/gpu-thermal-guard.sh) every minute. It lowers a card to 275 W if its core or HBM reaches 90 °C.

**Check:** `nvidia-smi --query-gpu=vbios_version,power.limit,clocks.max.sm,clocks.max.mem --format=csv` shows 92.00.6D.00.0A, 300 W and 1695/1728 MHz on both cards.

## 1. Container

[`config/host/lxc-container.conf`](config/host/lxc-container.conf) is our unprivileged LXC config:
- both GPUs as device nodes;
- the host's `libcuda` and `libnvidia-ml` bind-mounted;
- `lxc.prlimit.memlock: unlimited`, because the engine pins host buffers.

The container runs Ubuntu with Python 3.12.

**Check:** `nvidia-smi` inside the container lists both cards with 64 GiB each.

## 2. The base build (2dameneko)

Follow [2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX](https://github.com/2dameneko/vllm-Qwen3.8-Flash-Next-2xCMP170HX) at commit **`56e1347`**, steps 0–4 of its `INSTALL.md`:
1. Check out upstream vLLM at `a5a30471ff2bb7f0824f2da10e358af98d304472`.
2. Apply iIIusi0n's recipe `qwen38-ple-ssd.patch` (sha256 `09c4e13f…`) and build its native helper `ple_ssd_io.so`.
3. Apply 2dameneko's patches (`patches/apply.sh`).
4. Run `verify.sh`.

**Check:** 2dameneko's `verify.sh` passes (seventeen files match its manifest).

## 3. Our patches

Copy that tree (we keep production and experiments apart), then apply [`patches/`](patches/) in order, 01 to 18 (there is no 14):

```bash
REPO=/path/to/cmp170hx-vllm-qwen38-flash-next     # this repository
cd /opt/vllm-qfn-n4/src                            # the copied, 2dameneko-patched tree
for p in "$REPO"/patches/0[1-4]-*.patch; do git apply -p1 --check "$p" && git apply -p1 "$p"; done
git apply -p1 "$REPO/patches/05-auth-x-api-key.patch"
python3 "$REPO/patches/06-A1-thinking-off-consistency.py" "$PWD"
python3 "$REPO/patches/07-A2-history-reasoning-fields.py" "$PWD"
# night 4 (08-18, no 14): D1, D2, K1, C1, C2, C5, C4, D3, then the two inline fixes D3b and D3c
for p in "$REPO"/patches/0[89]-*.py "$REPO"/patches/1[0-6]-*.py; do python3 "$p" "$PWD" || break; done
for p in "$REPO"/patches/1[78]-*.patch; do git apply -p1 --check "$p" && git apply -p1 "$p"; done
```

[`patches/README.md`](patches/README.md) explains every patch, says which ones are active in production and which only act when their switch is set, and what each was tested with.

**Check:**
- `sha256sum -c patches/manifest/backport-files-after-04.sha256` passes after patch 04 (run in the tree root). Patches 05–07 are checked by their own anchors: each refuses to apply twice or to a different tree.
- `sha256sum -c patches/manifest/n4-files-before-08.sha256` passes before patch 08, and `sha256sum -c patches/manifest/n4-files-after-18.sha256` after patch 18. Our production tree also contains patch C3, which is not published and is off; it differs from yours only in `speculator.py` and `model_runner.py` ([`patches/README.md`](patches/README.md), `n4-production-with-unpublished-C3.sha256`).

Then build the venv with [`patches/make-venv.sh`](patches/make-venv.sh) (`make-venv.sh /opt/vllm-qfn-n4 n4`, the default). It installs the precompiled `a5a30471` wheel and torch 2.13 cu130, then the tree in editable mode, so nothing is compiled.

**Check:** the venv's Python prints `0.29.1rc1.dev402+ga5a30471f…` for `vllm.__version__`.

## 4. Model, template and draft head

1. Download [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) (we run revision `598afc96`, 166 GB) to the NVMe.
2. Check the files with [`scripts/verify_model.py`](scripts/verify_model.py).
3. Get the chat template `qwen-asgard-v1.jinja` from [qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja).
4. Make the int4 draft head for patch 08 with [`tools/models/quantize_lm_head.py`](tools/models/) (CPU only, about 15 s; [`tools/models/make-heads.sh`](tools/models/make-heads.sh) does it for both of our checkpoints):
   ```bash
   /opt/vllm-qfn-n4/.venv/bin/python tools/models/quantize_lm_head.py --model-dir /opt/vllm-qfn/models/swift15-awq \
       --out /opt/vllm-qfn/models/draft-heads/swift15-awq-lm_head-w4a16.safetensors --mse-scale-search
   ```
   **Check:** the script prints the file's size and sha256. Ours is 330,265,872 bytes with sha256 `5c5461d7…`.

**Option, not the standard: Swift-A16.** It is Swift 1.5 AWQ with its attention projections back in BF16. Build it with [`tools/models/`](tools/models/) (see [`swift-a16.md`](tools/models/swift-a16.md)); it needs about 72 GiB of new disk, and the draft head above serves it too.
- **Night 5's results:** better long context (100% against 95.0%) and total BFCL (+1.6 points), but lower multi-turn tool use (57.0% against 61.3%) and 10% slower raw decode.
- **In production:** it served for about a quarter of an hour on 2026-10-21, as part of a build that was rolled back ([night-5 report](research/2026-10-21-night5-max-build.md)).

## 5. Serve

1. Put [`config/qfn-prod.env`](config/qfn-prod.env) next to 2dameneko's `serve/serve.sh`, and set the paths in it: venv, model, `PLE_SSD_LIB`, template, API key file, and the draft head in `VLLM_MTP_DRAFT_HEAD_PATH`.
2. Install [`config/vllm-qfn.service`](config/vllm-qfn.service), which runs `serve.sh` with `RUN_ENV` pointing at the profile.
3. Install [`config/vllm-qfn.logrotate`](config/vllm-qfn.logrotate).

The first start takes several minutes: weights, CUDA-graph capture, and the PLE table opening.

**Check**, in order:
1. `curl -s localhost:8080/health` returns 200.
2. [`bench/prod_check.py`](bench/prod_check.py) passes: aliases, effort, tool call, Anthropic API, streaming, API key.
3. [`bench/func_check.py`](bench/func_check.py) passes: plain answer, tool call, and needles up to 128K.
4. [`bench/multiturn_check.py`](bench/multiturn_check.py) passes: prefix-cache follow-ups.

Then measure. [`bench/perf_exact.py`](bench/perf_exact.py) gives exact-length prompts that you can compare with [`results/speed/`](results/speed/). [`qeval/run_tonight.sh`](qeval/run_tonight.sh) gives the quality suite.

## Settings that mattered

| Setting | Why |
|---|---|
| `PLE_SSD_WORKERS=32`, `PLE_SSD_IO_DEPTH=512`, `PLE_SSD_PREFETCH=131072`, `PLE_SSD_CACHE_MB=16384` | the biggest prefill lever: 128K went from 5.9K to 8.3K tok/s |
| `MTP=0` plus `--speculative-config {"method":"mtp","num_speculative_tokens":4,"ngram_lookup":true}` and `VLLM_MTP_DRAFT_HEAD_PATH` (patch 08) | night 4: +15% code and +22% code-edit decode over MTP 3 with the BF16 head (clocks locked at 1440 MHz). The n-gram lookup alone gave +15% on code editing in night 2 |
| `--prefix-match-unit 16 --enable-mamba-fine-grained-prefix-cache` | multi-turn follow-ups reuse the cache. With 4 draft tokens the attention block is 1616 tokens, which 32 does not divide, so MTP 4 needs 16 |
| `MAX_NUM_SEQS=24` + the extended `CAPTURE_SIZES` | 16 users at 504 tok/s, single-user speed unchanged |
| BF16 KV | INT8 KV was broken with prefix-cache follow-ups + MTP + a fine `--prefix-match-unit`. Patch 10 (K1) fixes it: +71% KV pool for −1.6% decode in night 4, and a pool flooded to 1.45× passed in night 5. Production stays on BF16 KV until INT8 KV has been tested alone for quality (BFCL three times, multi-turn included) |
| `VLLM_PLE_NONPLE_LOADER=instanttensor` | faster engine start: 2dameneko measured 680 s → 150–166 s with their patch 06 |
| `CUDA_DEVICES=1,0`, `VLLM_PP_LAYER_PARTITION=24,24` | 24/24 with BF16 KV was fastest on long prompts |
