# Patch C4: diagnostic probe for the PP "align skew" (vektorprime qwen38-flash-next-pp2 README Part 4, patches-align
# 0011/0012, VLLM_PP_EXACT_MIRROR). No behaviour change; CPU-only bookkeeping on non-last PP ranks.
#
# vektorprime's mechanism: on a non-last PP rank the GPU num_computed_tokens mirror gets +qlen at every
# sample_tokens() (model_runner.py:2034-2037 -> postprocess_num_computed_tokens) and the matching -rejected only when
# the step's result is consumed pp_size engine steps later (update_pp_decode_requests -> postprocess_sampled ->
# postprocess_state(..., num_computed_tokens), where the align save reads the mirror as "new_num_computed").
# The mirror is therefore wrong at that consume iff the same request ran another step between the consumed step and
# the consume. In this tree the V2+PP+async scheduler forbids that for sampling steps (async_scheduler.py:47-49 sets
# next_decode_eligible_step = current_step + pp_size; scheduler.py:666 enforces it), so we expect ZERO skewed rows and
# did not port 0011/0012 (6 files, Triton changes, PP queue protocol).
# This probe proves or refutes that on the live server: at every deferred consume it checks whether any consumed
# request (need-sampled, not freed since) was scheduled again after the consumed step.
#
# Env (default off): VLLM_PP_SKEW_PROBE=1. Logs "Asgard C4 PP skew probe: consumed=N skewed=M" every 60 s on each
# non-last PP rank, and a WARNING with the first offending request slot when M grows.
#
# usage: python3 C4-pp-align-skew-probe.py <vllm source root>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/v1/worker/gpu/model_runner.py"
s = p.read_text()
if "Asgard patch C4" in s:
    print("model_runner.py: already patched (C4)")
    sys.exit(0)

pairs = [
    (
        """import functools
import gc
import time
""",
        """import functools
import gc
import os as _c4_os  # Asgard patch C4
import time
""",
    ),
    (
        """from vllm.v1.worker.gpu.pp_utils import PPHandler
""",
        """from vllm.v1.worker.gpu.pp_utils import PPHandler
from vllm.v1.worker.gpu.pp_utils import (  # Asgard patch C4
    compute_need_sampled_mask as _c4_need_sampled_mask,
)
""",
    ),
    (
        """logger = init_logger(__name__)
""",
        """logger = init_logger(__name__)

# Asgard patch C4: PP align-skew probe (diagnostic only).
_C4_PROBE = _c4_os.environ.get("VLLM_PP_SKEW_PROBE", "0") == "1"
""",
    ),
    (
        """    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
""",
        """    def _c4_probe_state(self) -> dict:
        # Asgard patch C4: per-rank probe state (engine-step counter, per-step rows).
        st = getattr(self, "_c4_state", None)
        if st is None:
            st = self._c4_state = {
                "step": 0,
                "hist": {},
                "last_adv": np.full(self.max_num_reqs, -1, dtype=np.int64),
                "consumed": 0,
                "skewed": 0,
                "warned": 0,
                "t_log": time.monotonic(),
                "pp": int(len(self.pp_handler.queue)) if self.pp_handler else 0,
            }
        return st

    def _c4_on_consume(self) -> None:
        # Asgard patch C4: runs where execute_model pops the entry of step S - pp_size.
        st = self._c4_probe_state()
        st["step"] += 1
        consumed_step = st["step"] - st["pp"]
        hist = st["hist"]
        rec = hist.pop(consumed_step, None)
        for old in [k for k in hist if k < consumed_step]:
            del hist[old]
        if rec is not None:
            idx_np, gen_np, need_np = rec
            alive = need_np & (self.pp_handler.req_idx_gen_np[idx_np] == gen_np)
            rows = idx_np[alive]
            if rows.size:
                st["consumed"] += int(rows.size)
                later = st["last_adv"][rows] > consumed_step
                n_skew = int(later.sum())
                if n_skew:
                    st["skewed"] += n_skew
                    if st["skewed"] > st["warned"] * 2:
                        st["warned"] = st["skewed"]
                        logger.warning(
                            "Asgard C4 PP skew probe: consumed step %d, req slot %d was "
                            "scheduled again at step %d before its deferred align "
                            "postprocess (skewed rows so far: %d)",
                            consumed_step,
                            int(rows[later][0]),
                            int(st["last_adv"][rows[later][0]]),
                            st["skewed"],
                        )
        now = time.monotonic()
        if now - st["t_log"] >= 60.0:
            st["t_log"] = now
            logger.info(
                "Asgard C4 PP skew probe: consumed=%d skewed=%d (pp=%d, step=%d)",
                st["consumed"],
                st["skewed"],
                st["pp"],
                st["step"],
            )

    def _c4_on_receive(self, input_batch: InputBatch) -> None:
        # Asgard patch C4: sample_tokens of step S on a non-last rank (+qlen to the mirror).
        st = self._c4_probe_state()
        step = st["step"]
        idx_np = np.asarray(input_batch.idx_mapping_np[: input_batch.num_reqs]).astype(
            np.int64
        )
        st["last_adv"][idx_np] = step
        need = _c4_need_sampled_mask(input_batch)
        if need is not None:
            st["hist"][step] = (
                idx_np,
                self.pp_handler.req_idx_gen_np[idx_np].copy(),
                np.asarray(need, dtype=bool)[: idx_np.size].copy(),
            )

    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if _C4_PROBE and self.pp_handler is not None and not self.is_last_pp_rank:
            self._c4_on_consume()  # Asgard patch C4
        if self.pp_handler is not None:
""",
    ),
    (
        """            all_decode_next = self.pp_handler.receive(input_batch)
""",
        """            all_decode_next = self.pp_handler.receive(input_batch)
            if _C4_PROBE:
                self._c4_on_receive(input_batch)  # Asgard patch C4
""",
    ),
]

for old, new in pairs:
    n = s.count(old)
    assert n == 1, f"model_runner.py: anchor found {n} times: {old[:80]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("model_runner.py: patched (C4)")
