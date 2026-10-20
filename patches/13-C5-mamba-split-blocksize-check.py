# Patch C5: xris99 "Mamba-Block-Alignment" fix, scheduler half (xris99/qwen3.8-flash-next-cmp170hx c407051,
# patches/qwen38-mamba-align.patch), as a check plus an opt-in switch.
#
# xris99's patch has two hunks. The state-seed divisor in mamba_hybrid.py is already in this tree as the upstream
# backport vllm#55601 (mamba_hybrid.py:121-129). The other hunk makes Scheduler._mamba_block_aligned_split end
# prefill chunks on the Mamba block grid (cache_config.mamba_block_size) instead of cache_config.block_size, which
# core.py:350 lowers to the MIN block size of the prefix-cacheable KV groups. On Qwen3.8-Flash-Next in align mode
# every prefix-cacheable group (QSA FullAttentionSpec, QSA compressed MLAAttentionSpec, GDN / PLE MambaSpec) uses
# cache_config.block_size and align mode sets mamba_block_size = block_size (platforms/interface.py:924-925); the
# QSA ring (CircularBufferSpec) is not prefix-cacheable. So both values should be 1600 on our server and the hunk is
# a no-op. Upstream main (2026-10-19) still uses cache_config.block_size here.
#
# Default: no behaviour change; logs a one-time WARNING if the two block sizes ever differ (e.g. after a KV dtype or
# drafter change), which is the case xris99 fixed. VLLM_MAMBA_SPLIT_USE_MAMBA_BS=1 applies xris99's behaviour.
#
# usage: python3 C5-mamba-split-blocksize-check.py <vllm source root>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/v1/core/sched/scheduler.py"
s = p.read_text()
if "Asgard patch C5" in s:
    print("scheduler.py: already patched (C5)")
    sys.exit(0)

pairs = [
    (
        """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
""",
        """        block_size = self.cache_config.block_size
        # Asgard patch C5 (xris99 c407051): the split must follow the Mamba block grid.
        c5_mamba_bs = self.cache_config.mamba_block_size
        if c5_mamba_bs and c5_mamba_bs != block_size:
            if __import__("os").environ.get("VLLM_MAMBA_SPLIT_USE_MAMBA_BS", "0") == "1":
                logger.info_once(
                    "Asgard C5: mamba aligned split uses mamba_block_size=%d "
                    "instead of block_size=%d",
                    c5_mamba_bs,
                    block_size,
                )
                block_size = c5_mamba_bs
            else:
                logger.warning_once(
                    "Asgard C5: block_size=%d != mamba_block_size=%d; prefill chunks "
                    "end off the Mamba block grid (set VLLM_MAMBA_SPLIT_USE_MAMBA_BS=1 "
                    "for xris99's fix)",
                    block_size,
                    c5_mamba_bs,
                )
        # The last block-aligned position whose state can be cached. With
""",
    ),
]

for old, new in pairs:
    n = s.count(old)
    assert n == 1, f"scheduler.py: anchor found {n} times: {old[:80]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("scheduler.py: patched (C5)")
