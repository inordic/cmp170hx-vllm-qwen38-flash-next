# Patch C1: reject draft slots that were never proposed (backport of vllm#58784, merged upstream 2026-09-28,
# commit fedbc3b5, Apache-2.0) to vLLM a5a30471 + our stack.
#
# The scheduler pads a request with exactly one token left to the 1+k spec shape and sends k placeholder drafts.
# MRV2 fills those slots from req_states.draft_tokens (zeroed by add_request), so the rejection sampler verifies k
# drafts of token 0 against a stale draft_logits row. With draft_sample_method="probabilistic" that accepts token 0
# ('!' in the Qwen tokenizer) far too often. The fix rewrites draft rows of a step that starts inside the prefill to -1,
# which every rejection kernel in this tree already treats as "reject".
#
# Correctness fix, active by default (as upstream). VLLM_SPEC_REJECT_UNPROPOSED=0 restores the stock gather.
# VLLM_SPEC_LOG_UNPROPOSED=1 counts rewritten rows on the GPU and logs the running total at most every 30 s
# (one GPU->CPU sync per log line; off by default).
#
# usage: python3 C1-reject-unproposed-drafts.py <vllm source root>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
s = p.read_text()
if "Asgard patch C1" in s:
    print("rejection_sampler.py: already patched (C1)")
    sys.exit(0)

pairs = [
    (
        """from collections.abc import Iterable, Iterator
from typing import Any
""",
        """import os as _c1_os  # Asgard patch C1
import time as _c1_time
from collections.abc import Iterable, Iterator
from typing import Any
""",
    ),
    (
        """from vllm.config.model import PROCESSED_LOGPROBS_MODES
""",
        """from vllm.config.model import PROCESSED_LOGPROBS_MODES
from vllm.logger import init_logger as _c1_init_logger  # Asgard patch C1
""",
    ),
    (
        """class RejectionSampler:
""",
        """# Asgard patch C1 (backport of vllm#58784): reject draft slots that were never proposed.
_c1_logger = _c1_init_logger(__name__)
_C1_ENABLED = _c1_os.environ.get("VLLM_SPEC_REJECT_UNPROPOSED", "1") != "0"
_C1_LOG = _c1_os.environ.get("VLLM_SPEC_LOG_UNPROPOSED", "0") == "1"


@triton.jit
def _c1_gather_draft_sampled_kernel(
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_tokens]
    input_ids_ptr,
    # [num_tokens]
    positions_ptr,
    # [num_logits]
    logits_indices_ptr,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    prefill_len_ptr,
    num_logits,
    BLOCK_SIZE: tl.constexpr,
):
    block = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < num_logits
    token_idx = tl.load(logits_indices_ptr + block, mask=mask, other=0)
    token_id = tl.load(input_ids_ptr + token_idx, mask=mask, other=0)
    pos = tl.load(positions_ptr + token_idx, mask=mask, other=0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + block, mask=mask, other=-1)
    local_pos = tl.load(expanded_local_pos_ptr + block, mask=mask, other=0)
    valid_req = mask & (req_state_idx >= 0)
    prefill_len = tl.load(
        prefill_len_ptr + tl.maximum(req_state_idx, 0), mask=valid_req, other=0
    )
    # Drafts are only proposed after sampling past the prefill, so the draft rows
    # of a step that starts inside the prefill are placeholders (e.g. a one-token
    # prompt tail the scheduler padded to the 1 + k spec shape).
    is_placeholder = valid_req & (local_pos > 0) & (pos - local_pos < prefill_len)
    token_id = tl.where(is_placeholder, -1, token_id)
    tl.store(draft_sampled_ptr + block, token_id, mask=mask)
    tl.store(pos_ptr + block, pos, mask=mask)


def gather_draft_sampled(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    logits_indices: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    expanded_local_pos: torch.Tensor,
    prefill_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Gather the input token and position of each logits row.

    Draft rows of requests that have not yet sampled past their prefill are set
    to -1 so that the rejection kernels reject them (vllm#58784).
    \"\"\"
    num_logits = logits_indices.shape[0]
    draft_sampled = input_ids.new_empty(num_logits)
    pos = positions.new_empty(num_logits)
    if num_logits == 0:
        return draft_sampled, pos
    BLOCK_SIZE = 1024
    _c1_gather_draft_sampled_kernel[(triton.cdiv(num_logits, BLOCK_SIZE),)](
        draft_sampled,
        pos,
        input_ids,
        positions,
        logits_indices,
        expanded_idx_mapping,
        expanded_local_pos,
        prefill_len,
        num_logits,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return draft_sampled, pos


class RejectionSampler:
""",
    ),
    (
        """        draft_sampled = input_batch.input_ids[input_batch.logits_indices]
        pos = input_batch.positions[input_batch.logits_indices]
""",
        """        if _C1_ENABLED:  # Asgard patch C1 (vllm#58784)
            draft_sampled, pos = gather_draft_sampled(
                input_batch.input_ids,
                input_batch.positions,
                input_batch.logits_indices,
                input_batch.expanded_idx_mapping,
                input_batch.expanded_local_pos,
                self.sampler.req_states.prefill_len.gpu,
            )
            if _C1_LOG:
                self._c1_log_placeholders(input_batch, draft_sampled)
        else:
            draft_sampled = input_batch.input_ids[input_batch.logits_indices]
            pos = input_batch.positions[input_batch.logits_indices]
""",
    ),
    (
        """    def _get_logprobs_tensors(
        self,
""",
        """    def _c1_log_placeholders(
        self, input_batch: InputBatch, draft_sampled: torch.Tensor
    ) -> None:
        # Asgard patch C1: count rows rewritten to -1 (the stock gather never yields -1).
        if not hasattr(self, "_c1_count"):
            self._c1_count = torch.zeros((), dtype=torch.int64, device=draft_sampled.device)
            self._c1_last_log = _c1_time.monotonic()
            self._c1_logged = 0
        orig = input_batch.input_ids[input_batch.logits_indices]
        self._c1_count += ((draft_sampled < 0) & (orig >= 0)).sum()
        now = _c1_time.monotonic()
        if now - self._c1_last_log >= 30.0:
            self._c1_last_log = now
            total = int(self._c1_count.item())
            if total != self._c1_logged:
                self._c1_logged = total
                _c1_logger.info(
                    "Asgard C1 (vllm#58784): %d unproposed draft rows rejected so far",
                    total,
                )

    def _get_logprobs_tensors(
        self,
""",
    ),
]

for old, new in pairs:
    n = s.count(old)
    assert n == 1, f"rejection_sampler.py: anchor found {n} times: {old[:80]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("rejection_sampler.py: patched (C1)")
