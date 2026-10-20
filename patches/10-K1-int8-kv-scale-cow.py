# Patch K1: copy INT8 per-token-head KV scales on copy-on-write block copies (partial prefix-cache hits).
# For vLLM a5a30471 + iIIusi0n PLE-SSD + 2dameneko 56e1347 (patch 16, kv_int8_qsa) + our backports.
#
# Bug: with --kv-cache-dtype int8_per_token_head + --prefix-match-unit 32 + fine-grained mamba cache + MTP, a
# prefix hit that ends inside a block makes the scheduler swap the shared tail block for a private block and
# queue a (src, dst) copy. copy_kv_cache_blocks_inplace() copies only the KV pages in `kv_caches`; patch 16 keeps
# the per-(token, head) fp32 scales outside the page (`_k_scale_cache`/`_v_scale_cache` on the QSA impl, dim 0 =
# kernel blocks), so the private block got correct INT8 bytes with the previous owner's scales -> empty/wrong
# follow-up answers.
# Fix: a small registry of per-block side tensors in vllm/v1/worker/utils.py; the two model runners ask
# copy_kv_cache_blocks_inplace() to copy them with the same (src, dst) pairs (gather-then-scatter, same stream,
# before the forward). The QSA impl registers its two scale caches when it allocates them, which only happens for
# int8/fp8_per_token_head, so BF16 registers nothing and the copy path is unchanged there.
# Also: _pth_caches() re-creates the scales if clear_layer_kv_caches() nulled them (it keeps _pth_key).
# Optional (off by default): vllm#55112 geometry-aware dedup of KV views (VLLM_K1_FIX_55112=1).
# Env: VLLM_K1_DISABLE=1 (skip the side copy, reproduces the bug), VLLM_K1_DEBUG=1 (log every CoW step, count stale
#      side rows before the copy, verify every KV view's dst == src after it; syncs the GPU),
#      VLLM_K1_LOG_EVERY=N (INFO line on the 1st and every N-th side-tensor copy, default 50).
# usage: python3 K1-int8-kv-scale-cow.py <vllm source root>      (root contains vllm/)
import sys
from pathlib import Path

MARK = "Asgard patch K1"
root = Path(sys.argv[1])

UTILS_IMPORT = (
    """import math
from collections import defaultdict
""",
    """import math
import os as _k1_os  # Asgard patch K1
import weakref as _k1_weakref  # Asgard patch K1
from collections import defaultdict
""",
)

UTILS_SIG = (
    """def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
) -> None:
    if not kv_cache_block_copies:
        return
""",
    '''# Asgard patch K1 (INT8 per-token-head KV + sub-block prefix hits): per-block side tensors that must
# follow the copy-on-write block copies below. Some quantized-KV impls keep per-(token, head) scales
# outside the KV page (QSA patch 16: `_k_scale_cache`/`_v_scale_cache`, dim 0 = kernel blocks); the CoW
# copy only moves the pages in `kv_caches`, so the private copy of a partially hit block kept the
# previous owner's scales. Owners register here (weakly); BF16 KV registers nothing, so the copy is
# unchanged there.
#   VLLM_K1_DISABLE=1    keep the registry but skip the side-tensor copy (reproduces the old bug)
#   VLLM_K1_DEBUG=1      log every CoW step, count stale side rows before the copy and check that
#                        every KV view's dst block equals its src block after it (syncs the GPU)
#   VLLM_K1_LOG_EVERY=N  INFO line on the first and every N-th side-tensor copy (default 50)
#   VLLM_K1_FIX_55112=1  vllm#55112: skip only identical aliased KV views (geometry-aware dedup)
def _k1_env_flag(name: str) -> bool:
    value = _k1_os.environ.get(name, "0").strip().lower()
    return value not in ("", "0", "false", "no", "off")


_K1_DISABLE = _k1_env_flag("VLLM_K1_DISABLE")
_K1_DEBUG = _k1_env_flag("VLLM_K1_DEBUG")
_K1_FIX_55112 = _k1_env_flag("VLLM_K1_FIX_55112")
try:
    _K1_LOG_EVERY = max(1, int(_k1_os.environ.get("VLLM_K1_LOG_EVERY", "50") or 50))
except ValueError:
    _K1_LOG_EVERY = 50
# id(owner) -> (weak ref to owner, attribute names)
_K1_AUX_OWNERS: dict[int, tuple[Any, tuple[str, ...]]] = {}
_K1_STATS = {"steps": 0, "pairs": 0, "tensor_copies": 0, "stale_rows": 0, "kv_mismatch": 0}


def register_kv_block_aux_tensors(owner: Any, *attr_names: str) -> None:
    """Asgard patch K1: make CoW block copies also copy ``owner.<attr>`` tensors.

    Each attribute is read at copy time (None or missing is skipped), so a
    re-allocation needs no re-registration. Its dim 0 must count KV blocks
    (scheduler blocks, or kernel blocks folded the same way as the KV views).
    Registration is weak and idempotent.
    """
    key = id(owner)
    attrs = tuple(attr_names)
    entry = _K1_AUX_OWNERS.get(key)
    if entry is not None and entry[0]() is owner and entry[1] == attrs:
        return
    try:
        ref = _k1_weakref.ref(owner)
    except TypeError:
        ref = lambda owner=owner: owner  # noqa: E731
    _K1_AUX_OWNERS[key] = (ref, attrs)
    logger.info_once(
        "Asgard patch K1: per-block KV side tensors %s of %s registered; CoW block "
        "copies %s.",
        attrs,
        type(owner).__name__,
        "will NOT copy them (VLLM_K1_DISABLE=1)" if _K1_DISABLE else "will copy them",
    )


def _k1_copy_aux_block_tensors(
    num_blocks: int, indices_np: np.ndarray, indices: torch.Tensor | None
) -> None:
    """Asgard patch K1: copy registered side tensors with the KV copy's pairs."""
    by_device: dict[torch.device, torch.Tensor] = {}
    if indices is not None:
        by_device[indices.device] = indices
    copied: set[tuple[Any, ...]] = set()
    num_tensors = stale = checked = 0
    for key, (ref, attrs) in list(_K1_AUX_OWNERS.items()):
        owner = ref()
        if owner is None:
            _K1_AUX_OWNERS.pop(key, None)
            continue
        for attr in attrs:
            aux = getattr(owner, attr, None)
            if not isinstance(aux, torch.Tensor) or aux.dim() == 0:
                continue
            per_block, remainder = divmod(aux.shape[0], num_blocks)
            if per_block == 0 or remainder:
                logger.warning_once(
                    "Asgard patch K1: side tensor %s.%s has dim 0 = %d, not a "
                    "multiple of %d KV blocks; it is not copied.",
                    type(owner).__name__,
                    attr,
                    aux.shape[0],
                    num_blocks,
                )
                continue
            aux_key = (aux.device, aux.data_ptr(), tuple(aux.shape), tuple(aux.stride()))
            if aux_key in copied:
                continue
            copied.add(aux_key)
            idx = by_device.get(aux.device)
            if idx is None:
                idx = async_tensor_h2d(indices_np, device=aux.device)
                by_device[aux.device] = idx
            src, dst = idx.unbind(dim=1)
            # Fold kernel blocks into dim 0 like the KV copy; unflatten is a view.
            blocks = aux.unflatten(0, (num_blocks, per_block))
            if _K1_DEBUG:
                differs = (blocks[dst] != blocks[src]).flatten(1).any(dim=1)
                stale += int(differs.sum().item())
                checked += differs.numel()
            if not _K1_DISABLE:
                blocks[dst] = blocks[src]
            num_tensors += 1
    if not num_tensors:
        return
    stats = _K1_STATS
    stats["steps"] += 1
    stats["pairs"] += len(indices_np)
    stats["tensor_copies"] += num_tensors
    stats["stale_rows"] += stale
    if _K1_DEBUG or stats["steps"] == 1 or stats["steps"] % _K1_LOG_EVERY == 0:
        logger.info(
            "Asgard patch K1: CoW step %d: %d block pair(s), %d side tensor(s) %s%s; "
            "totals %d pairs, %d tensor copies",
            stats["steps"],
            len(indices_np),
            num_tensors,
            "NOT copied (VLLM_K1_DISABLE=1)" if _K1_DISABLE else "copied",
            (
                f"; stale dst rows before copy {stale}/{checked} "
                f"(total {stats['stale_rows']}); pairs {indices_np[:8].tolist()}"
                if _K1_DEBUG
                else ""
            ),
            stats["pairs"],
            stats["tensor_copies"],
        )


def _k1_verify_kv_copy(
    kv_caches: Iterable[torch.Tensor], num_blocks: int, indices: torch.Tensor | None
) -> None:
    """Asgard patch K1 (VLLM_K1_DEBUG): after the CoW copy, every distinct KV view
    must hold the same bytes in dst as in src (a mismatch means a partial copy,
    cf. vllm#55112; chained pairs within one step would also show up here)."""
    if indices is None:
        return
    src, dst = indices.unbind(dim=1)
    seen: set[tuple[Any, ...]] = set()
    bad = []
    num_views = 0
    for i, cache in enumerate(kv_caches):
        if cache.device != indices.device or cache.dim() == 0:
            continue
        key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()), cache.dtype)
        if key in seen:
            continue
        seen.add(key)
        per_block, remainder = divmod(cache.shape[0], num_blocks)
        if per_block == 0 or remainder:
            continue
        blocks = cache.unflatten(0, (num_blocks, per_block))
        new, old = blocks[dst], blocks[src]
        try:
            same = torch.equal(new.view(torch.uint8), old.view(torch.uint8))
        except RuntimeError:
            same = torch.equal(new, old)
        num_views += 1
        if not same:
            bad.append((i, tuple(cache.shape), tuple(cache.stride()), str(cache.dtype)))
    _K1_STATS["kv_mismatch"] += len(bad)
    if bad:
        logger.warning(
            "Asgard patch K1 debug: %d of %d distinct KV views differ between dst and "
            "src after the CoW copy (partial copy, cf. vllm#55112): %s",
            len(bad),
            num_views,
            bad[:6],
        )
    else:
        logger.info(
            "Asgard patch K1 debug: all %d distinct KV views match src after the CoW copy",
            num_views,
        )


def copy_kv_cache_blocks_inplace(
    kv_caches: Iterable[torch.Tensor],
    num_blocks: int,
    kv_cache_block_copies: Sequence[KVCacheBlockCopy],
    *,
    copy_aux_block_tensors: bool = False,  # Asgard patch K1: the model runners pass True
) -> None:
    if not kv_cache_block_copies:
        return
    if copy_aux_block_tensors and _K1_DEBUG:
        kv_caches = list(kv_caches)
''',
)

UTILS_KEY = (
    """        key = (cache.device, cache.data_ptr())
        if key in seen:
            continue
        seen.add(key)
""",
    """        key = (cache.device, cache.data_ptr())
        if _K1_FIX_55112:
            # Asgard patch K1 (vllm#55112): hybrid groups overlay views of different
            # geometry at one address; skip only identical aliased views.
            key = (
                cache.device,
                cache.data_ptr(),
                tuple(cache.shape),
                tuple(cache.stride()),
                cache.element_size(),
            )
        if key in seen:
            continue
        seen.add(key)
""",
)

UTILS_TAIL = (
    """            blocks = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))
        blocks[dst] = blocks[src]


def is_uniform_query_len(""",
    """            blocks = cache.unflatten(0, (num_blocks, kernel_blocks_per_block))
        blocks[dst] = blocks[src]
    if copy_aux_block_tensors:  # Asgard patch K1
        if _K1_AUX_OWNERS:
            _k1_copy_aux_block_tensors(num_blocks, indices_np, indices)
        if _K1_DEBUG:
            _k1_verify_kv_copy(kv_caches, num_blocks, indices)


def is_uniform_query_len(""",
)

RUNNER_CALL = (
    """            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
            )
""",
    """            copy_kv_cache_blocks_inplace(
                self.kv_caches,
                self.kv_cache_config.num_blocks,
                scheduler_output.kv_cache_block_copies,
                copy_aux_block_tensors=True,  # Asgard patch K1: + INT8 scale caches
            )
""",
)

QSA_COND = (
    """        ident = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        if self._pth_key != ident:
""",
    """        ident = (kv_cache.data_ptr(), tuple(kv_cache.shape))
        # Asgard patch K1: also re-create the scales after clear_layer_kv_caches() nulled
        # them (it keeps _pth_key, so a same-shape rebind would hand the kernel None).
        if (
            self._pth_key != ident
            or self._k_scale_cache is None
            or self._v_scale_cache is None
        ):
""",
)

QSA_REGISTER = (
    """            self._v_scale_cache = torch.ones_like(self._k_scale_cache)
            self._pth_key = ident
        return key_cache, value_cache, self._k_scale_cache, self._v_scale_cache
""",
    """            self._v_scale_cache = torch.ones_like(self._k_scale_cache)
            self._pth_key = ident
            # Asgard patch K1: the scale rows are indexed by physical KV block id like the
            # quantized page, so the worker's copy-on-write block copies (partial prefix
            # hits with --prefix-match-unit) must copy them too. The attributes are read at
            # copy time; BF16 KV never gets here, so it registers nothing.
            from vllm.v1.worker.utils import register_kv_block_aux_tensors

            register_kv_block_aux_tensors(self, "_k_scale_cache", "_v_scale_cache")
        return key_cache, value_cache, self._k_scale_cache, self._v_scale_cache
""",
)

TARGETS = [
    ("vllm/v1/worker/utils.py", [UTILS_IMPORT, UTILS_SIG, UTILS_KEY, UTILS_TAIL]),
    ("vllm/v1/worker/gpu/model_runner.py", [RUNNER_CALL]),
    ("vllm/v1/worker/gpu_model_runner.py", [RUNNER_CALL]),
    ("vllm/models/qwen4_exp/nvidia/qsa.py", [QSA_COND, QSA_REGISTER]),
]

# Build every new file in memory first (all anchors checked), then write: a failed anchor writes nothing.
out = []
for rel, pairs in TARGETS:
    p = root / rel
    s = p.read_text()
    if MARK in s:
        print(f"{rel}: already patched")
        continue
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{rel}: anchor found {n} times (want 1): {old[:90]!r}"
        s = s.replace(old, new)
    compile(s, str(p), "exec")  # syntax check before anything is written
    out.append((rel, p, s))
for rel, p, s in out:
    p.write_text(s)
    print(f"{rel}: patched (K1)")
