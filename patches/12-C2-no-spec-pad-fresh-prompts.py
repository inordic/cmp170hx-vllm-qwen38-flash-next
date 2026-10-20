# Patch C2: do not pad a fresh one-token prompt to the 1+k spec-decode shape on hybrid (Mamba/GDN) models.
# Env VLLM_SPEC_PAD_NEW_REQS: unset/"auto" = this fix, "1" = stock padding, "0" = never pad (vektorprime's switch).
#
# Source: vektorprime/qwen38-flash-next-pp2 mtp-spec-decode "bug 3" (VLLM_SPEC_PAD_NEW_REQS=0), re-derived for this tree.
# The scheduler pads a request with exactly one token left to 1+k rows (placeholder drafts) so the step keeps the
# uniform FULL-graph shape. Our mamba_hybrid.py classifies a row as spec-decode purely by
# num_scheduled == num_drafts + 1 (mamba_hybrid.py:285-291), so a padded *fresh* prompt (num_computed_tokens == 0)
# runs the GDN / PLE-conv spec kernels, which always read an initial state from the request's state slot
# (spec rows have no has_initial_state; gdn_attn.py:234, ops/ple.py spec mode reads state when the slot is valid).
# Mamba state blocks are not zeroed on allocation (single_type_kv_cache_manager.py:118-122 records only
# AttentionSpec blocks for zeroing), so that slot holds another request's or uninitialised state -> NaN logprobs
# (HTTP 400) or garbage. vektorprime measured 11/12 failures under load on the same code path.
# "auto" keeps stock padding for prompt tails over prior state (num_computed_tokens > 0), which our classifier runs as
# spec rows from the restored state, and for non-hybrid models. The only cost of not padding is that the step is
# not uniform, so it runs PIECEWISE instead of the FULL decode graph.
#
# Correctness fix; active by default ("auto"), VLLM_SPEC_PAD_NEW_REQS=1 restores stock behaviour.
# Applies with or without Asgard patch D2 (anchors avoid D2's lines).
# usage: python3 C2-no-spec-pad-fresh-prompts.py <vllm source root>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/v1/core/sched/scheduler.py"
s = p.read_text()
if "Asgard patch C2" in s:
    print("scheduler.py: already patched (C2)")
    sys.exit(0)

pairs = [
    (
        """from typing import Any
""",
        """from typing import Any
import os as _c2_os  # Asgard patch C2
""",
    ),
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)

# Asgard patch C2: VLLM_SPEC_PAD_NEW_REQS = auto (default) | 1 (stock) | 0 (never pad).
_C2_PAD_MODE = _c2_os.environ.get("VLLM_SPEC_PAD_NEW_REQS", "auto").strip().lower() or "auto"


def _c2_spec_pad_allowed(num_computed_tokens: int, has_mamba_layers: bool) -> bool:
    \"\"\"Asgard patch C2: whether a request with one token left may be padded to 1 + k.

    A fresh prompt on a hybrid model has no recurrent state; padded, it would run
    the spec-decode kernels from an uninitialised state slot.
    \"\"\"
    if _C2_PAD_MODE == "1":
        return True
    if _C2_PAD_MODE == "0":
        return False
    if has_mamba_layers and num_computed_tokens == 0:
        logger.info_once(
            "Asgard C2: not padding fresh one-token prompts to the spec-decode "
            "shape on this hybrid model (VLLM_SPEC_PAD_NEW_REQS=1 restores stock)."
        )
        return False
    return True
""",
    ),
    (
        """                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                    ):
""",
        """                        and not prefill_scheduled
                        and (scheduled_running_reqs or num_computed_tokens > 0)
                        and _c2_spec_pad_allowed(  # Asgard patch C2
                            num_computed_tokens, self.has_mamba_layers
                        )
                    ):
""",
    ),
]

for old, new in pairs:
    n = s.count(old)
    assert n == 1, f"scheduler.py: anchor found {n} times: {old[:80]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("scheduler.py: patched (C2)")
