# Patch D3: DFlash2 target-feature tap for Qwen3.8-Flash-Next (qwen4_exp, NVIDIA model) under MRV2 + PP.
# The drafter incoai/Qwen3.8-Flash-Next-DFlash2 (target_layer_ids [5,15,25,35,45], fc 12800 = 5 x 2560) was
# trained on what its SGLang capture patch records: the COMPLETED output of zero-based layer i, averaged over
# the hc_count hyper-connection streams (hidden_states.unflatten(-1, (hc_count, hidden)).mean(-2)).
#   1. vllm/models/qwen4_exp/nvidia/model.py
#      - Qwen4ExpModel becomes an EagleModelMixin with supports_aux_hidden_states_over_pp = True, so the stock
#        set_eagle3_aux_hidden_state_layers / reserve_aux_intermediate_tensor_slots / PP relay work for it.
#      - Tap after layer L-1 for every aux layer L (vLLM "input to layer L" ids; get_eagle3_aux_layers_from_config
#        already adds +1 to dflash target_layer_ids, so 6,16,26,36,46 = outputs of 5,15,25,35,45). The loop
#        delays each layer's final residual combine into the next layer's fused combine+norm; the tap
#        materializes it on the side with the same hc_combine (same fp32 math + bf16 rounding), the main path
#        is untouched. Earlier-stage taps travel in IntermediateTensors slots aux_hidden_states_<i>; the last
#        stage returns (hidden_states, upstream + local taps). With no aux layers set: no extra op, the return
#        values are exactly as before (MTP path incl. its multi-stream buffer unchanged).
#      - Qwen4ExpForCausalLM also gets SupportsEagle3 (ForConditionalGeneration already has it via Qwen3-VL).
#      - Logs whether the target embed_tokens was loaded on this rank (the drafter shares it on the last stage).
#   2. vllm/model_executor/model_loader/default_loader.py
#      - The PP weight filter (VLLM_PP_LAYER_PARTITION) describes the TARGET checkpoint. It was also armed for
#        the DFlash2 draft checkpoint (5 layers named layers.0-4, loaded on the last stage): with 24,24 set
#        get_pp_indices raises (sum 48 != 5), without it layers 0-2 of the drafter would be skipped silently.
#        It is now off for a draft model from a different checkpoint; MTP (same checkpoint) keeps it.
# usage: python3 D3-qwen4exp-dflash2-tap.py <vllm source root>
import sys
from pathlib import Path

root = Path(sys.argv[1])
M = "Asgard patch D3"


def plan(rel, pairs):
    p = root / rel
    s = p.read_text()
    if M in s:
        print(f"{rel}: already patched")
        return None
    for old, new in pairs:
        assert s.count(old) == 1, f"{rel}: anchor not found exactly once: {old[:70]!r}"
        s = s.replace(old, new)
    compile(s, str(p), "exec")  # every new source compiles before anything is written
    return p, s


MODEL = "vllm/models/qwen4_exp/nvidia/model.py"
LOADER = "vllm/model_executor/model_loader/default_loader.py"

model_pairs = [
    # --- imports + helpers -------------------------------------------------------------------------
    ("""from .qsa import Qwen4ExpQSAAttention
""", """from .qsa import Qwen4ExpQSAAttention

# Asgard patch D3: DFlash2 target-feature tap (aux hidden states for EAGLE3/DFlash drafters).
# incoai/Qwen3.8-Flash-Next-DFlash2 was trained (its SGLang capture patch) on the COMPLETED outputs
# of zero-based layers [5,15,25,35,45], each averaged over the hc_count hyper-connection streams.
# vLLM hands them over as aux layers L = id + 1 ("input to layer L"; get_eagle3_aux_layers_from_config
# adds the +1), so the tap fires after layer L - 1. Inactive - no extra op, unchanged return values -
# unless set_aux_hidden_state_layers() ran (eagle3/dflash/dspark/extract_hidden_states only).
import os as _d3_os

from vllm.logger import init_logger as _d3_init_logger
from vllm.model_executor.models.interfaces import (
    EagleModelMixin as _D3EagleModelMixin,
)
from vllm.model_executor.models.interfaces import (
    SupportsEagle3 as _D3SupportsEagle3,
)

_d3_logger = _d3_init_logger(__name__)
# Diagnostic only: "pending" taps the pre-combine residual the loop carries between layers instead
# of the completed layer output. Default "completed" = the SGLang capture the drafter was trained on.
_D3_TAP_MODE = _d3_os.environ.get("VLLM_D3_TAP", "completed").strip() or "completed"
if _D3_TAP_MODE not in ("completed", "pending"):
    raise ValueError(f"VLLM_D3_TAP must be completed|pending, got {_D3_TAP_MODE!r}")


def _d3_hc_mean(
    hidden_states: torch.Tensor, hc_count: int, hidden_size: int
) -> torch.Tensor:
    \"\"\"[T, hc_count*H] (HC outer, H inner) -> [T, H]; same op as the SGLang capture.\"\"\"
    return hidden_states.unflatten(-1, (hc_count, hidden_size)).mean(dim=-2)
"""),
    # --- Qwen4ExpModel: EagleModelMixin ------------------------------------------------------------
    ("""class Qwen4ExpModel(nn.Module):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER
""", """class Qwen4ExpModel(nn.Module, _D3EagleModelMixin):
    hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER
    # Asgard patch D3: aux hidden states cross PP stages in IntermediateTensors slots.
    supports_aux_hidden_states_over_pp = True
    _d3_embed_loaded = False
"""),
    # --- set_aux_hidden_state_layers bookkeeping + the tap itself ----------------------------------
    ("""    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PATCH (abliteration): embed_tokens rows write into the residual. The projection
""", """    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        # Asgard patch D3: validate, then the stock EagleModelMixin bookkeeping (sorted ids,
        # PP slot base = taps of earlier stages, upstream slot count on the last stage).
        layers = tuple(sorted({int(x) for x in layers}))
        num_layers = self.config.num_hidden_layers
        bad = [x for x in layers if not 0 <= x <= num_layers]
        if bad:
            raise ValueError(
                f"Asgard patch D3: aux hidden state layers {bad} are outside "
                f"[0, {num_layers}] for this {num_layers}-layer Qwen4Exp model"
            )
        _D3EagleModelMixin._set_aux_hidden_state_layers(self, layers)
        pp = get_pp_group()
        local = [
            x
            for x in layers
            if (x == self.start_layer and pp.is_first_rank)
            or self.start_layer < x <= self.end_layer
        ]
        _d3_logger.info(
            "Asgard patch D3: DFlash2 tap active (VLLM_D3_TAP=%s): aux layers %s = "
            "completed outputs of layers %s, mean over hc_count=%d streams -> [T, %d] "
            "each; PP rank %d/%d owns layers %d..%d, taps here %s, upstream slots %d, "
            "slot base %d",
            _D3_TAP_MODE,
            list(layers),
            [x - 1 for x in layers],
            self.config.hc_count,
            self.config.hidden_size,
            pp.rank_in_group,
            pp.world_size,
            self.start_layer,
            self.end_layer - 1,
            local,
            self._aux_upstream_total_cached,
            self._aux_slot_base_cached,
        )
        if pp.world_size > 1 and pp.is_last_rank:
            if self._d3_embed_loaded:
                _d3_logger.info(
                    "Asgard patch D3: target embed_tokens is loaded on the last PP "
                    "rank (the DFlash drafter shares it)"
                )
            else:
                _d3_logger.warning(
                    "Asgard patch D3: could NOT confirm that the target embed_tokens "
                    "was loaded on the last PP rank; a drafter sharing it would read "
                    "uninitialized embeddings (acceptance ~1)"
                )

    def _d3_tap(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        \"\"\"Asgard patch D3: completed output of ``layer``, mean over the HC streams.

        The loop delays each layer's final residual combine into the next layer's
        fused combine+norm, so ``hidden_states`` is the pre-combine residual while
        ``block_output`` is pending. ``hc_combine`` does the same fp32 math and
        bf16 rounding as the residual that fused kernel materializes, so the tap
        sees exactly the tensor the next layer consumes. Side tensor only.
        \"\"\"
        if block_output is not None and _D3_TAP_MODE == "completed":
            hidden_states = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        return _d3_hc_mean(hidden_states, self.config.hc_count, self.config.hidden_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # PATCH (abliteration): embed_tokens rows write into the residual. The projection
"""),
    # --- forward: upstream taps + optional embedding tap --------------------------------------------
    ("""        block_output = None
        injection = None
        last_layer = None
""", """        # Asgard patch D3: aux taps (empty tuple unless set_aux_hidden_state_layers ran).
        d3_layers = self.aux_hidden_state_layers
        d3_aux: list[torch.Tensor] = []
        d3_remote_aux = (
            self.collect_remote_aux_hidden_states(intermediate_tensors)
            if d3_layers
            else []
        )
        if (
            d3_layers
            and get_pp_group().is_first_rank
            and self.start_layer in d3_layers
        ):
            d3_aux.append(
                _d3_hc_mean(hidden_states, self.config.hc_count, self.config.hidden_size)
            )

        block_output = None
        injection = None
        last_layer = None
"""),
    # --- forward: tap after layer L-1, pack into the PP slots ---------------------------------------
    ("""                hidden_states = hidden_states + deepstack_embed

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            return IntermediateTensors({"hidden_states": hidden_states})
""", """                hidden_states = hidden_states + deepstack_embed
            if d3_layers and layer_idx + 1 in d3_layers:
                # Asgard patch D3: completed output of layer_idx (aux id layer_idx + 1).
                d3_aux.append(
                    self._d3_tap(layer, hidden_states, block_output, injection)
                )

        if not get_pp_group().is_last_rank:
            # PP transports one tensor, not the delayed HC tuple. Materialize
            # with the HC module that produced the pending injection.
            if last_layer is not None and block_output is not None:
                hidden_states = last_layer.mlp_hyper_connection.combine(
                    hidden_states, block_output, injection
                )
            # Asgard patch D3: + aux_hidden_states_<base+i> slots ({} when untapped).
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    **self.pack_local_aux_hidden_states(d3_aux),
                }
            )
"""),
    # --- forward: last stage returns (hidden_states, aux list) --------------------------------------
    ("""            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        return sample_hidden_states
""", """            self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
        if d3_layers:
            # Asgard patch D3: the MRV2 runner expects (hidden_states, aux list)
            # whenever aux outputs are on; earlier stages' taps first (layer order).
            return sample_hidden_states, d3_remote_aux + d3_aux
        return sample_hidden_states
"""),
    # --- load_weights: remember whether embed_tokens was loaded on this rank ------------------------
    ("""        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        return loaded
""", """        loaded = loader.load_weights(
            weights,
            mapper=mapper,
        )
        # Asgard patch D3: the last PP rank shares this embedding with a DFlash drafter.
        self._d3_embed_loaded = self._d3_embed_loaded or "embed_tokens.weight" in loaded
        return loaded
"""),
    # --- text-only class: SupportsEagle3 --------------------------------------------------------------
    ("""    SupportsPP,
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
""", """    SupportsPP,
    _D3SupportsEagle3,  # Asgard patch D3
    Qwen4ExpMixtureOfExperts,
    IsHybrid,
):
"""),
]

loader_pairs = [
    ("""        parallel_config = get_current_vllm_config().parallel_config
        pp_size = parallel_config.pipeline_parallel_size
        if pp_size <= 1:
            return

""", """        parallel_config = get_current_vllm_config().parallel_config
        pp_size = parallel_config.pipeline_parallel_size
        if pp_size <= 1:
            return

        # Asgard patch D3: the partition describes the TARGET checkpoint. A draft
        # model from its own checkpoint (DFlash2: 5 layers named layers.0-4, built
        # on the last stage only) must not be filtered with it: with
        # VLLM_PP_LAYER_PARTITION=24,24 get_pp_indices raises (sum 48 != 5), without
        # it the last stage would silently skip the drafter's layers 0-2. MTP drafts
        # live in the target checkpoint and keep the filter.
        _d3_target = get_current_vllm_config().model_config
        if (
            _d3_target is not None
            and model_config is not _d3_target
            and model_config.model != _d3_target.model
        ):
            self.local_layers = None
            logger.info(
                "Asgard patch D3: PP weight filter off for %s (not the target "
                "checkpoint %s)",
                model_config.model,
                _d3_target.model,
            )
            return

"""),
]

todo = [x for x in (plan(MODEL, model_pairs), plan(LOADER, loader_pairs)) if x is not None]
for p, s in todo:
    p.write_text(s)
    print(f"{p.relative_to(root)}: patched")
