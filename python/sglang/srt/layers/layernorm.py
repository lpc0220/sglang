# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Fused operators for normalization layers."""

import logging
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.batch_invariant_ops import (
    is_batch_invariant_mode_enabled,
    rms_norm_batch_invariant)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    is_cuda,
    is_flashinfer_available)

_is_cuda = is_cuda()
_is_flashinfer_available = is_flashinfer_available()
_flashinfer_layernorm_available = False

if _is_cuda:
    if _is_flashinfer_available:
        try:
            from flashinfer.norm import layernorm

            _flashinfer_layernorm_available = True
        except (ImportError, AttributeError):
            _flashinfer_layernorm_available = False
    else:
        _flashinfer_layernorm_available = False

    from sgl_kernel import (
        fused_add_rmsnorm,
        rmsnorm)
    # DeepSeek-only build: gemma_fused_add_rmsnorm and gemma_rmsnorm removed

logger = logging.getLogger(__name__)

class RMSNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
        cast_x_before_out_mul: bool = False,
        fp32_residual: bool = False,
        weight_dtype: Optional = None,
        override_orig_dtype: Optional = None) -> None:
        super().__init__()
        self.cast_x_before_out_mul = cast_x_before_out_mul
        self.fp32_residual = fp32_residual
        self.override_orig_dtype = override_orig_dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=weight_dtype))
        self.variance_epsilon = eps
        self.hidden_size = hidden_size
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )

    def forward_cuda(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.variance_size_override is not None:
            return self.forward_native(x, residual, **kwargs)
        if is_batch_invariant_mode_enabled():
            if (
                residual is not None
                or get_global_server_args().rl_on_policy_target == "fsdp"
            ):
                return self.forward_native(x, residual, **kwargs)
            return rms_norm_batch_invariant(
                x,
                self.weight.data,
                self.variance_epsilon)
        if residual is not None:
            # TODO: Ideally we want to have (hidden_states+residual)+post_residual_addition.
            # but right now we can only have hidden_states+(residual+post_residual_addition).
            # (hidden_states+residual)+post_residual_addition != hidden_states+(residual+post_residual_addition),
            # we probably need to add another parameter to fused_add_rmsnorm
            post_residual_addition = kwargs.get("post_residual_addition")
            if post_residual_addition is not None:
                residual = residual + post_residual_addition
            fused_add_rmsnorm(x, residual, self.weight.data, self.variance_epsilon)
            return x, residual
        out = rmsnorm(x, self.weight.data, self.variance_epsilon)
        return out

    def forward_native(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
        **kwargs) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if not x.is_contiguous():
            x = x.contiguous()
        orig_dtype = self.override_orig_dtype or x.dtype
        post_residual_addition = kwargs.get("post_residual_addition")
        x = x.to(torch.float32)
        if residual is not None:
            x = (
                x
                + residual.to(torch.float32)
                + (
                    post_residual_addition.to(torch.float32)
                    if post_residual_addition is not None
                    else 0.0
                )
            )
            if self.fp32_residual:
                residual = x.clone()
            else:
                residual = x.to(orig_dtype)

        hidden_size = x.shape[-1]
        if hidden_size != self.hidden_size:
            raise ValueError(
                "Expected hidden_size to be "
                f"{self.hidden_size}, but found: {hidden_size}"
            )

        if self.variance_size_override is None:
            x_var = x
        else:
            if hidden_size < self.variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{self.variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[..., : self.variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)

        if self.cast_x_before_out_mul:
            x = self.weight * x.to(orig_dtype)
        else:
            x = (x * self.weight).to(orig_dtype)

        if residual is None:
            return x
        else:
            return x, residual

    def forward_with_allreduce_fusion(
        self,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward method with allreduce fusion, prioritizing flashinfer fused operations
        """
        if residual is not None:
            from sglang.srt.distributed import get_tensor_model_parallel_world_size
            from sglang.srt.layers.flashinfer_comm_fusion import (
                flashinfer_allreduce_residual_rmsnorm)

            if get_tensor_model_parallel_world_size() > 1:
                fused_result = flashinfer_allreduce_residual_rmsnorm(
                    input_tensor=x,
                    residual=residual,
                    weight=self.weight,
                    eps=self.variance_epsilon)
                if fused_result[0] is not None:
                    return fused_result

        return self.forward(x, residual)


class LayerNorm(MultiPlatformOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.elementwise_affine = elementwise_affine
        self.use_bias = bias
        self.dtype = dtype

        self.bias = nn.Parameter(torch.zeros(hidden_size, dtype=self.dtype))
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=self.dtype))

    def forward_cuda(
        self,
        x: torch.Tensor,
        **kwargs) -> torch.Tensor:
        if (
            _flashinfer_layernorm_available
            and x.dtype == torch.bfloat16
            and self.dtype == torch.float32
        ):
            return layernorm(x, self.weight, self.bias, self.variance_epsilon)
        else:
            return self.forward_native(x, **kwargs)

    def forward_native(
        self,
        x: torch.Tensor,
        **kwargs) -> torch.Tensor:
        weight = self.weight if self.elementwise_affine else None
        bias = self.bias if self.use_bias else None
        orig_dtype = x.dtype
        x = x.to(self.dtype)
        return F.layer_norm(
            x,
            (self.hidden_size),
            weight=self.weight,
            bias=bias,
            eps=self.variance_epsilon).to(orig_dtype)



# DeepSeek-only build: GemmaRMSNorm and Gemma3RMSNorm classes removed
