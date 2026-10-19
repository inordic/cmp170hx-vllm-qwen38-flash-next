# Build: from two CMP 170HX cards to a running server

This is the route we took. The paths are ours: `/opt/vllm-qfn` for the build, and `/opt/vllm-qfn-exp` for the patched tree that production runs. Adjust them to your system. Each step says how to check it before you go on.

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

Copy that tree (we keep production and experiments apart), then apply [`patches/`](patches/) in order:

```bash
REPO=/path/to/cmp170hx-vllm-qwen38-flash-next     # this repository
cd /opt/vllm-qfn-exp/src                           # the copied, 2dameneko-patched tree
for p in "$REPO"/patches/0[1-4]-*.patch; do git apply -p1 --check "$p" && git apply -p1 "$p"; done
git apply -p1 "$REPO/patches/05-auth-x-api-key.patch"
python3 "$REPO/patches/06-A1-thinking-off-consistency.py" "$PWD"
python3 "$REPO/patches/07-A2-history-reasoning-fields.py" "$PWD"
```

[`patches/README.md`](patches/README.md) explains every patch.

**Check:** `sha256sum -c patches/manifest/backport-files-after-04.sha256` passes after patch 04 (run in the tree root). Patches 05–07 are checked by their own anchors: each refuses to apply twice or to a different tree.

Then build the venv with [`patches/make-venv.sh`](patches/make-venv.sh). It installs the precompiled `a5a30471` wheel and torch 2.13 cu130, then the tree in editable mode, so nothing is compiled.

**Check:** the venv's Python prints `0.29.1rc1.dev402+ga5a30471f…` for `vllm.__version__`.

## 4. Model and template

1. Download [`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ) (we run revision `598afc96`, 166 GB) to the NVMe.
2. Check the files with [`scripts/verify_model.py`](scripts/verify_model.py).
3. Get the chat template `qwen-asgard-v1.jinja` from [qwen-asgard-jinja](https://github.com/leonbollerup/qwen-asgard-jinja).

## 5. Serve

1. Put [`config/qfn-prod.env`](config/qfn-prod.env) next to 2dameneko's `serve/serve.sh`, and set the paths in it: venv, model, `PLE_SSD_LIB`, template, API key file.
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
| `MTP=0` plus `--speculative-config {"method":"mtp","num_speculative_tokens":3,"ngram_lookup":true}` | +15% on code editing from the n-gram lookup |
| `--prefix-match-unit 32 --enable-mamba-fine-grained-prefix-cache` | multi-turn follow-ups reuse the cache |
| `MAX_NUM_SEQS=24` + the extended `CAPTURE_SIZES` | 16 users at 504 tok/s, single-user speed unchanged |
| BF16 KV | INT8 KV is broken with prefix-cache follow-ups + MTP + `--prefix-match-unit 32` |
| `VLLM_PLE_NONPLE_LOADER=instanttensor` | faster engine start: 2dameneko measured 680 s → 150–166 s with their patch 06 |
| `CUDA_DEVICES=1,0`, `VLLM_PP_LAYER_PARTITION=24,24` | 24/24 with BF16 KV was fastest on long prompts |
