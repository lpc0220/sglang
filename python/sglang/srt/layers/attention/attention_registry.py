import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # evade circular imports
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.model_executor.model_runner import ModelRunner

ATTENTION_BACKENDS = {}


def register_attention_backend(name):
    def decorator(fn):
        ATTENTION_BACKENDS[name] = fn
        return fn

    return decorator


@register_attention_backend("flashinfer")
def create_flashinfer_backend(runner):
    import torch

    if not runner.use_mla_backend:
        from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend

        # Init streams
        if runner.server_args.speculative_algorithm == "EAGLE":
            if (
                not hasattr(runner, "plan_stream_for_flashinfer")
                or not runner.plan_stream_for_flashinfer
            ):
                runner.plan_stream_for_flashinfer = torch.cuda.Stream()
        return FlashInferAttnBackend(
            runner, init_new_workspace=runner.init_new_workspace
        )
    else:
        from sglang.srt.layers.attention.flashinfer_mla_backend import (
            FlashInferMLAAttnBackend,
        )

        return FlashInferMLAAttnBackend(runner)


@register_attention_backend("trtllm_mla")
def create_trtllm_mla_backend(runner):
    if not runner.use_mla_backend:
        raise ValueError("trtllm_mla backend can only be used with MLA models.")
    from sglang.srt.layers.attention.trtllm_mla_backend import TRTLLMMLABackend

    return TRTLLMMLABackend(runner)


@register_attention_backend("nsa")
def create_nsa_backend(runner):
    from sglang.srt.layers.attention.nsa_backend import NativeSparseAttnBackend

    return NativeSparseAttnBackend(runner)


@register_attention_backend("triton")
def create_triton_backend(runner):
    assert not runner.model_config.is_encoder_decoder, (
        "Cross attention is not supported in the triton attention backend. "
        "Please use `--attention-backend flashinfer`."
    )
    if runner.server_args.enable_double_sparsity:
        from sglang.srt.layers.attention.double_sparsity_backend import (
            DoubleSparseAttnBackend,
        )

        return DoubleSparseAttnBackend(runner)
    else:
        from sglang.srt.layers.attention.triton_backend import TritonAttnBackend

        return TritonAttnBackend(runner)


@register_attention_backend("torch_native")
def create_torch_native_backend(runner):
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend(runner)


@register_attention_backend("flex_attention")
def create_flex_attention_backend(runner):
    from sglang.srt.layers.attention.torch_flex_backend import TorchFlexAttnBackend

    return TorchFlexAttnBackend(runner)


@register_attention_backend("cutlass_mla")
def create_cutlass_mla_backend(runner):
    from sglang.srt.layers.attention.cutlass_mla_backend import CutlassMLABackend

    return CutlassMLABackend(runner)


@register_attention_backend("trtllm_mha")
def create_trtllm_mha_backend(runner):
    if runner.use_mla_backend:
        raise ValueError("trtllm_mha backend can only be used with non-MLA models.")
    from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend

    return TRTLLMHAAttnBackend(runner)


def attn_backend_wrapper(runner: "ModelRunner", full_attn_backend: "AttentionBackend"):
    """
    Wrapper for special models like hybrid GDN, so we don't
    need to change the code of the original attention backend.
    """
    assert not (
        runner.hybrid_gdn_config is not None and runner.use_mla_backend
    ), "hybrid_gdn can only be used with non-MLA models."

    if cfg := runner.mambaish_config:
        from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
            GDNAttnBackend,
            HybridLinearAttnBackend,
            KimiLinearAttnBackend,
            Mamba2AttnBackend,
        )
        from sglang.srt.utils import is_blackwell

        # Note: Removed check_environments() call as FLA module was deleted
        # This function only checked Triton/Python versions and issued warnings
        if runner.hybrid_gdn_config is not None:
            if is_blackwell():
                assert (
                    runner.server_args.attention_backend == "triton"
                    or runner.server_args.attention_backend == "trtllm_mha"
                ), "triton or trtllm_mha backend are the only supported backends on Blackwell GPUs for hybrid GDN models, use --attention-backend triton or --attention-backend trtllm_mha to specify the backend."
            logger.info(f"Using hybrid linear attention backend for hybrid GDN models.")
            linear_attn_backend = GDNAttnBackend(runner)
        elif runner.mamba2_config is not None:
            linear_attn_backend = Mamba2AttnBackend(runner)
        elif runner.kimi_linear_config is not None:
            linear_attn_backend = KimiLinearAttnBackend(runner)
        else:
            raise ValueError(
                "Expected hybrid GDN or NemotronH models, but got unknown model."
            )
        full_attn_layers = cfg.full_attention_layer_ids
        return HybridLinearAttnBackend(
            full_attn_backend, linear_attn_backend, full_attn_layers
        )

    return full_attn_backend
