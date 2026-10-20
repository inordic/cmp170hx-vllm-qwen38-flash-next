# Patch D1: optional W4A16 (int4, group 128, symmetric) draft-only lm_head for the Qwen3.8-Flash-Next MTP drafter.
# Ported from AlexBrandes/vllm commit 14457f89e ("[Qwen4Exp][Spec Decode] Optional W4A16 draft-only lm_head for MTP",
# Apache-2.0) to vLLM a5a30471 + iIIusi0n PLE-SSD + 2dameneko 56e1347 + our backports 01-04.
# Active only when VLLM_MTP_DRAFT_HEAD_PATH names a packed head (see quantize_lm_head.py); unset = unchanged behaviour.
# The verify pass keeps the target's BF16 head, so only draft acceptance can change, never the accepted tokens.
# usage: python3 D1-mtp-int4-draft-head.py <vllm source root>
import sys
from pathlib import Path

root = Path(sys.argv[1])


def patch(rel, pairs, marker):
    p = root / rel
    s = p.read_text()
    if marker in s:
        print(f"{rel}: already patched"); return
    for old, new in pairs:
        assert s.count(old) == 1, f"{rel}: anchor not found exactly once: {old[:70]!r}"
        s = s.replace(old, new)
    p.write_text(s)
    print(f"{rel}: patched")


M = "Asgard patch D1"

patch("vllm/models/qwen4_exp/nvidia/mtp.py", [
    ("""    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
)
""", """    Qwen4ExpMixtureOfExperts,
    Qwen4ExpSparseMoeBlock,
)

# Asgard patch D1: optional W4A16 MTP draft head (VLLM_MTP_DRAFT_HEAD_PATH), from AlexBrandes/vllm 14457f89e.
# Normally the drafter adopts the target's BF16 lm_head after loading, so every draft step pays a full
# 248320x2560 BF16 GEMV (~1.27 GiB read). With VLLM_MTP_DRAFT_HEAD_PATH set, the drafter keeps its own
# Marlin W4A16 head (the quant linear the model's own layers use), filled from a packed GPTQ-layout file.
import os as _d1_os

from vllm.logger import init_logger as _d1_init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig as _D1QuantizationConfig
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQConfig as _D1AutoGPTQConfig,
    AutoGPTQLinearMethod as _D1AutoGPTQLinearMethod,
)

_d1_logger = _d1_init_logger(__name__)
_DRAFT_HEAD_BITS = 4
_DRAFT_HEAD_GROUP = 128


class _QuantDraftHeadConfig(_D1QuantizationConfig):
    \"\"\"Always-quantized config: W4A16 GPTQ-Marlin linear for the draft head.\"\"\"

    def get_name(self) -> str:
        return "auto_gptq"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 75

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict) -> "_QuantDraftHeadConfig":
        raise RuntimeError("_QuantDraftHeadConfig is instantiated directly")

    def get_quant_method(self, layer, prefix) -> "_D1AutoGPTQLinearMethod":
        return _D1AutoGPTQLinearMethod(
            _D1AutoGPTQConfig(
                weight_bits=_DRAFT_HEAD_BITS,
                group_size=_DRAFT_HEAD_GROUP,
                desc_act=False,
                is_sym=True,
                lm_head_quantized=True,
                dynamic={},
                full_config={},
            )
        )
"""),
    ("""        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(""",
     """        self.draft_head_quantized = False
        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            elif _d1_os.environ.get("VLLM_MTP_DRAFT_HEAD_PATH"):
                # Asgard patch D1: the drafter keeps its own Marlin W4A16 head.
                self.lm_head = ColumnParallelLinear(
                    config.hidden_size,
                    config.vocab_size,
                    bias=False,
                    return_bias=False,
                    quant_config=_QuantDraftHeadConfig(),
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
                self.draft_head_quantized = True
                _d1_logger.info("MTP draft lm_head built as W4A16 GPTQ-Marlin head; packed weights from %s",
                                _d1_os.environ["VLLM_MTP_DRAFT_HEAD_PATH"])
            else:
                self.lm_head = ParallelLMHead("""),
    ("""    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight
""", """    def _load_quant_draft_head(self, path: str) -> None:
        \"\"\"Asgard patch D1: fill the W4A16 draft lm_head from a packed GPTQ/AutoGPTQ file.
        Tensors (TP=1): lm_head.qweight int32 [K//8, N], lm_head.scales fp16 [K//128, N],
        lm_head.qzeros int32 [K//128, N//8] (constant zero-point, ignored by the symmetric kernel);
        K = hidden_size, N = vocab_size.\"\"\"
        from safetensors import safe_open

        if getattr(self.lm_head, "tp_size", 1) != 1:
            raise NotImplementedError("VLLM_MTP_DRAFT_HEAD_PATH supports TP=1 only")
        in_size, out_size = self.config.hidden_size, self.config.vocab_size
        expected = {
            "qweight": (in_size // 8, out_size),
            "scales": (in_size // _DRAFT_HEAD_GROUP, out_size),
            "qzeros": (in_size // _DRAFT_HEAD_GROUP, out_size // 8),
        }
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name, shape in expected.items():
                key = f"lm_head.{name}"
                if key not in keys:
                    raise ValueError(f"{path}: missing tensor {key!r}")
                tensor = f.get_tensor(key)
                if tuple(tensor.shape) != tuple(shape):
                    raise ValueError(f"{path}: {key} has shape {tuple(tensor.shape)}, expected {shape}")
                param = getattr(self.lm_head, name)
                param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
                _d1_logger.info("MTP draft head: loaded %s shape=%s dtype=%s", key, tuple(tensor.shape), param.dtype)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        drop_head_weights = self.draft_head_quantized

        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is None:
                    continue
                # Asgard patch D1: the quantized head is filled from VLLM_MTP_DRAFT_HEAD_PATH, not the checkpoint.
                if drop_head_weights and remapped_name.startswith("lm_head."):
                    continue
                yield remapped_name, weight
"""),
    ("""        return loader.load_weights(remap_weight_names(), mapper=mapper)


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]""",
     """        loaded = loader.load_weights(remap_weight_names(), mapper=mapper)
        if drop_head_weights:
            self._load_quant_draft_head(_d1_os.environ["VLLM_MTP_DRAFT_HEAD_PATH"])
        return loaded


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]"""),
], M)

patch("vllm/v1/spec_decode/llm_base_proposer.py", [
    ("""import dataclasses
from importlib.util import find_spec""", """import dataclasses
import os as _d1_os  # Asgard patch D1
from importlib.util import find_spec"""),
    ("""        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )
""", """        elif _d1_os.environ.get("VLLM_MTP_DRAFT_HEAD_PATH") and getattr(self.model, "draft_head_quantized", False):
            # Asgard patch D1: the drafter owns a W4A16 lm_head; keep it.
            share_lm_head = False
            logger.info("MTP draft owns a W4A16 lm_head (VLLM_MTP_DRAFT_HEAD_PATH set); keeping it.")
        else:
            # MTP model
            share_lm_head = True
            logger.info(
                "Detected MTP model. "
                "Sharing target model lm_head weights with the draft model."
            )
"""),
], M)

patch("vllm/v1/worker/gpu/spec_decode/eagle/utils.py", [
    ("""import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace""", """import os as _d1_os  # Asgard patch D1

import torch
import torch.nn as nn

from vllm.config import VllmConfig, replace"""),
    ("""    if target_lm_head is not None and _should_share(
        eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):""", """    if (
        target_lm_head is not None
        # Asgard patch D1: keep the drafter's own W4A16 head (VLLM_MTP_DRAFT_HEAD_PATH).
        and not (_d1_os.environ.get("VLLM_MTP_DRAFT_HEAD_PATH") and getattr(eagle_model, "draft_head_quantized", False))
        and _should_share(eagle_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):"""),
], M)
print("D1 done")
