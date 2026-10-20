# Patch D2: bounded prefill budget beside running decodes (hybrid Mamba), env VLLM_MIXED_PREFILL_CHUNK (0/unset = off).
# Ported from AlexBrandes/vllm commit 31208d203 ("[v1][Scheduler] Bounded prefill budget beside running decodes",
# Apache-2.0) to vLLM a5a30471 + iIIusi0n PLE-SSD + 2dameneko 56e1347 + our backports.
# While any request is decoding, a step's prefill tokens are capped at VLLM_MIXED_PREFILL_CHUNK x (prompts awaiting
# prefill), at most one Mamba block, spent oldest-first, so a new long prompt advances every step and running
# decoders see short steps. With no decoder running, prefill uses the full budget.
# usage: python3 D2-bounded-mixed-prefill.py <vllm source root>
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "vllm/v1/core/sched/scheduler.py"
s = p.read_text()
if "Asgard patch D2" in s:
    print("scheduler.py: already patched"); sys.exit(0)
pairs = [
    ("""import itertools
import time
""", """import itertools
import os as _d2_os  # Asgard patch D2
import time
"""),
    ("""        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
""", """        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        # Asgard patch D2 (from AlexBrandes/vllm 31208d203): bounded prefill beside running decodes.
        self.mixed_prefill_chunk = int(_d2_os.environ.get("VLLM_MIXED_PREFILL_CHUNK", "0") or 0)
        self.step_prefill_cap = 0
        self.step_prefill_left: int | None = None
        if self.mixed_prefill_chunk:
            logger.info("D2 mixed-step prefill cap: %d tokens", self.mixed_prefill_chunk)
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
"""),
    ("""            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            aligned_end = end // block_size * block_size
""", """            if long_prefill_threshold > 0:
                max_prefill_tokens = min(max_prefill_tokens, long_prefill_threshold)
            if self.step_prefill_left is not None:  # Asgard patch D2
                max_prefill_tokens = min(max_prefill_tokens, self.step_prefill_cap)
            aligned_end = end // block_size * block_size
"""),
    ("""        # First, schedule the RUNNING requests.
        req_index = 0
""", """        # Asgard patch D2: budget prefill on every step while a decoder is running. Under PP a decoder skips
        # steps, but the step in between still delays it, so the budget keys on decoders existing.
        self.step_prefill_left = None
        self.step_prefill_cap = 0
        if self.mixed_prefill_chunk and any(not r.is_prefill_chunk for r in self.running):
            awaiting = (
                sum(1 for r in self.running if r.is_prefill_chunk)
                + len(self.waiting)
                + len(self.skipped_waiting)
            )
            self.step_prefill_left = min(
                self.mixed_prefill_chunk * max(1, awaiting),
                max(self.cache_config.block_size, self.mixed_prefill_chunk),
            )

        # First, schedule the RUNNING requests.
        req_index = 0
"""),
    ("""            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
""", """            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            d2_prefill = self.step_prefill_left is not None and request.is_prefill_chunk  # Asgard patch D2
            if d2_prefill:
                self.step_prefill_cap = self.step_prefill_left
                num_new_tokens = min(num_new_tokens, self.step_prefill_left)
            num_new_tokens = min(
                num_new_tokens, token_budget, input_budget - draft_slots
            )
"""),
    ("""            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            req_index += 1
""", """            token_budget -= num_new_tokens
            input_budget -= num_new_tokens + draft_slots
            if d2_prefill:  # Asgard patch D2
                self.step_prefill_left = max(0, self.step_prefill_left - num_new_tokens)
            req_index += 1
"""),
    ("""                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False
""", """                new_encoder_compute_budget = encoder_compute_budget
                pad_spec_decode = False
                d2_prefill = False  # Asgard patch D2
"""),
    ("""                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
""", """                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold
                    d2_prefill = self.step_prefill_left is not None and not pad_spec_decode  # Asgard patch D2
                    if d2_prefill:
                        self.step_prefill_cap = self.step_prefill_left
                        num_new_tokens = min(num_new_tokens, self.step_prefill_left)
                        if num_new_tokens == 0:
                            break
"""),
    ("""                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                request.status = RequestStatus.RUNNING
""", """                token_budget -= num_new_tokens
                input_budget -= num_new_tokens + draft_slots
                if d2_prefill:  # Asgard patch D2
                    self.step_prefill_left = max(0, self.step_prefill_left - num_new_tokens)
                request.status = RequestStatus.RUNNING
"""),
]
for old, new in pairs:
    assert s.count(old) == 1, f"anchor not found exactly once: {old[:80]!r}"
    s = s.replace(old, new)
p.write_text(s)
print("scheduler.py: patched (D2)")
