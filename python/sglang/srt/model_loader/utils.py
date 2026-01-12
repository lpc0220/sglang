# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/model_executor/model_loader/utils.py

"""Utilities for selecting and loading models."""
import concurrent.futures
import contextlib
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from torch import nn

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.layers import deep_gemm_wrapper

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def set_default_torch_dtype(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(old_dtype)


def get_model_architecture(model_config: ModelConfig) -> Tuple[Type[nn.Module], str]:
    from sglang.srt.models.registry import ModelRegistry

    architectures = getattr(model_config.hf_config, "architectures", [])
    # Special handling for quantized Mixtral.
    # FIXME(woosuk): This is a temporary hack.
    mixtral_supported = ["fp8", "compressed-tensors", "gptq_marlin", "awq_marlin"]

    if (
        model_config.quantization is not None
        and model_config.quantization not in mixtral_supported
        and "MixtralForCausalLM" in architectures
    ):
        architectures = ["QuantMixtralForCausalLM"]

    return ModelRegistry.resolve_model_cls(architectures)


def get_architecture_class_name(model_config: ModelConfig) -> str:
    return get_model_architecture(model_config)[1]


def post_load_weights(model: nn.Module, model_config: ModelConfig):
    # Model weight loading consists of two stages:
    # 1. Initial weight loading.
    # 2. Post-processing of weights, including assigning specific member variables.
    # For `dummy_init`, only the second stage is required.
    if hasattr(model, "post_load_weights"):
        if model_config.hf_config.architectures[0] == "DeepseekV3ForCausalLMNextN":
            model.post_load_weights(is_nextn=True)
        else:
            model.post_load_weights()


def should_deepgemm_weight_requant_ue8m0(weight_block_size):
    """Should we requant fp8 weights into UE8M0 format when loading the model"""
    return (
        deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
        and deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0
        and weight_block_size is not None
    )


def should_async_load(weight: torch.Tensor) -> bool:
    """Return True if we should load the given weight asynchronously.

    For host (CPU) tensors, using a threadpool can overlap H2D copies
    and improve throughput. For device tensors, threading often adds overhead
    (e.g., GIL contention) without benefit, so we do it synchronously.
    """
    device = getattr(weight, "device", None)
    if device is None:
        return False
    return device.type == "cpu"


def maybe_executor_submit(
    *,
    executor: concurrent.futures.ThreadPoolExecutor,
    futures: List[concurrent.futures.Future],
    use_async: bool,
    func: Callable[..., Any],
    func_args: Iterable[Any] = (),
    func_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """Submit a task to the executor if async loading is enabled.

    Parameters (keyword-only):
    - executor: ThreadPoolExecutor used to submit background tasks
    - futures: a list collecting the submitted Future objects
    - use_async: whether to submit to executor or run inline
    - func: the callable to run
    - func_args: positional args for the callable (defaults to empty tuple)
    - func_kwargs: keyword args for the callable (defaults to empty dict)
    """
    if func_kwargs is None:
        func_kwargs = {}
    if use_async:
        futures.append(executor.submit(func, *func_args, **func_kwargs))
    else:
        func(*func_args, **func_kwargs)
