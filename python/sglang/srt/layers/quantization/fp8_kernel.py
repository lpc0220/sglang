# Copyright 2024 SGLang Team
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

import functools
import json
import logging
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.utils import (
    ceil_align,
    get_bool_env_var,
    get_device_core_count,
    get_device_name,
    is_cuda,
    log_info_on_rank0,
)
from sglang.srt.utils.custom_op import register_custom_op

_is_cuda = is_cuda()

if _is_cuda:
    from sgl_kernel import sgl_per_token_quant_fp8

    from sglang.jit_kernel.per_tensor_quant_fp8 import (
        per_tensor_quant_fp8 as sgl_per_tensor_quant_fp8,
    )

    # Temporary
    try:
        from sgl_kernel import sgl_per_token_group_quant_8bit

        enable_sgl_per_token_group_quant_8bit = True
    except ImportError:
        from sgl_kernel import sgl_per_token_group_quant_fp8

        enable_sgl_per_token_group_quant_8bit = False


    def scaled_fp8_quant(
        input: torch.Tensor,
        scale: Optional[torch.Tensor] = None,
        num_token_padding: Optional[int] = None,
        use_per_token_if_dynamic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert input.ndim == 2, f"Expected 2D input tensor, got {input.ndim}D"
        shape = input.shape
        if num_token_padding:
            shape = (max(num_token_padding, input.shape[0]), shape[1])
        output = torch.empty(shape, device=input.device, dtype=fp8_dtype)

        if scale is None:
            # Dynamic scaling
            if use_per_token_if_dynamic:
                scale = torch.empty(
                    (shape[0], 1), device=input.device, dtype=torch.float32
                )
                sgl_per_token_quant_fp8(input, output, scale)
            else:
                scale = torch.zeros(1, device=input.device, dtype=torch.float32)
                sgl_per_tensor_quant_fp8(
                    input, output, scale, is_static=False
                )  # False for dynamic
        else:
            # Static scaling
            assert (
                scale.numel() == 1
            ), f"Expected scalar scale, got numel={scale.numel()}"
            sgl_per_tensor_quant_fp8(
                input, output, scale, is_static=True
            )  # True for static

        return output, scale


fp8_autotune = triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": block_m}, num_warps=num_warps)
        for block_m in [16, 32, 64, 128]
        for num_warps in [2, 4, 8]
    ],
    key=["K", "BLOCK_K", "M_ALIGNMENT"],
)


@triton.jit
def _per_token_group_quant_fp8_hopper_moe_mn_major(
    a,  # (M, K):(K, 1)
    expert_offsets,  # (num_experts,)
    problem_sizes,  # (num_experts, 3)
    a_fp8,  # (M, K):(K, 1)
    sfa,  # (M, k)
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_M: tl.constexpr,  # tune
):
    k_offset = tl.program_id(0)
    expert_id = tl.program_id(1)

    m = tl.load(problem_sizes + expert_id * 3)
    current_expert_offset = tl.load(expert_offsets + expert_id).to(tl.int64)
    tl.multiple_of(m, M_ALIGNMENT)
    tl.multiple_of(current_expert_offset, M_ALIGNMENT)

    coord_k = k_offset * BLOCK_K + tl.arange(0, BLOCK_K)
    for i in tl.range(tl.cdiv(m, BLOCK_M)):
        coord_m = i * BLOCK_M + tl.arange(0, BLOCK_M)
        a_ptrs = a + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
        a_mask = (coord_m < m)[:, None] & (coord_k < K)[None, :]

        inp = tl.load(a_ptrs, mask=a_mask).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        inp_amax = tl.max(tl.abs(inp), axis=1)  # [BLOCK_M,]
        inp_amax = tl.clamp(inp_amax, min=1e-4, max=float("inf"))
        inp_fp8 = (inp * (448.0 / inp_amax[:, None])).to(tl.float8e4nv)

        # Store fp8
        a_fp8_ptrs = (
            a_fp8 + current_expert_offset * K + coord_m[:, None] * K + coord_k[None, :]
        )
        tl.store(a_fp8_ptrs, inp_fp8, mask=a_mask)

        # Store sfa
        k = tl.cdiv(K, BLOCK_K)
        sfa_ptrs = (
            sfa + current_expert_offset * k + k_offset * m + coord_m
        )  # MN-Major with sfa
        tl.store(sfa_ptrs, inp_amax / 448.0, mask=coord_m < m)


if not _is_cpu:
    _per_token_group_quant_fp8_hopper_moe_mn_major = fp8_autotune(
        _per_token_group_quant_fp8_hopper_moe_mn_major
    )


def per_token_group_quant_fp8_hopper_moe_mn_major(
    A: torch.Tensor,
    expert_offsets: torch.Tensor,
    problem_sizes: torch.Tensor,
    group_size: int,
    expert_tokens_alignment: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert A.dim() == 2
    assert A.is_contiguous(), "`A` is not contiguous"
    assert (
        A.shape[-1] % group_size == 0
    ), "the last dimension of `A` cannot be divisible by `group_size`"

    a_q = torch.empty_like(A, device=A.device, dtype=fp8_dtype)
    M, K = A.shape[0], A.shape[1]
    k = K // group_size
    sfa = torch.empty((M, k), device=A.device, dtype=torch.float32)
    num_experts = problem_sizes.shape[0]
    grid = (k, num_experts)
    _per_token_group_quant_fp8_hopper_moe_mn_major[grid](
        A,
        expert_offsets,
        problem_sizes,
        a_q,
        sfa,
        K,
        group_size,
        expert_tokens_alignment,
    )
    return a_q, sfa


@triton.jit
def _per_group_transpose(
    data_ptr: torch.Tensor,
    trans_data_ptr: torch.Tensor,
    expert_offsets: torch.Tensor,
    k: int,
    M_ALIGNMENT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    expert_id = tl.program_id(0)
    m_id = tl.program_id(1)
    k_id = tl.program_id(2)

    curr_expert_offset = tl.load(expert_offsets + expert_id)
    next_expert_offset = tl.load(expert_offsets + expert_id + 1)
    num_tokens_of_expert = next_expert_offset - curr_expert_offset
    tl.multiple_of(curr_expert_offset, M_ALIGNMENT)
    tl.multiple_of(next_expert_offset, M_ALIGNMENT)

    data_start_ptr = data_ptr + curr_expert_offset * k
    trans_data_start_ptr = trans_data_ptr + curr_expert_offset * k

    k_coord = k_id * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_coord < k
    for start_m in tl.range(0, num_tokens_of_expert, BLOCK_SIZE_M * tl.num_programs(1)):
        m_coord = start_m + m_id * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        m_mask = m_coord < num_tokens_of_expert
        off = m_coord[:, None] * k + k_coord[None, :]
        trans_off = m_coord[:, None] + k_coord[None, :] * num_tokens_of_expert
        mask = m_mask[:, None] & k_mask[None, :]

        data = tl.load(data_start_ptr + off, mask=mask)
        tl.store(trans_data_start_ptr + trans_off, data, mask=mask)


def per_group_transpose(
    a: torch.Tensor,
    expert_offsets: torch.Tensor,
    M_ALIGNMENT: int = 1,
) -> torch.Tensor:
    assert a.dim() == 2
    assert a.is_contiguous(), "`a` is not contiguous"

    m, k = a.size()
    trans_a = torch.empty_like(a)
    num_experts = expert_offsets.size(0) - 1

    grid = lambda META: (
        num_experts,
        triton.cdiv((m + num_experts - 1) // num_experts, META["BLOCK_SIZE_M"]),
        triton.cdiv(k, META["BLOCK_SIZE_K"]),
    )
    _per_group_transpose[grid](
        a, trans_a, expert_offsets, k, M_ALIGNMENT, BLOCK_SIZE_M=16, BLOCK_SIZE_K=8
    )
    return trans_a


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@triton.jit
def scaled_mm_kernel(
    a_ptr,
    b_ptr,
    scale_a_ptr,
    scale_b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    ACCUMULATOR_DTYPE: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_SCALE_A: tl.constexpr,
    BLOCK_SIZE_SCALE_B: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    accumulator_dtype = ACCUMULATOR_DTYPE
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=accumulator_dtype)

    # NOTE: Some tensor inputs are so large, they will cause int32 overflow
    # so it is necessary to use tl.int64 for all the offsets, else SEGV will
    # eventually occur.

    # Offsets and masks.
    offsets_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    masks_am = offsets_am < M

    offsets_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    masks_bn = offsets_bn < N

    offsets_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    offsets_a = stride_am * offsets_am[:, None] + stride_ak * offsets_k[None, :]
    offsets_b = stride_bk * offsets_k[:, None] + stride_bn * offsets_bn[None, :]

    # NOTE: BLOCK_SIZE_SCALE_A could be 1 or BLOCK_SIZE_M, so need to create
    # appropriate offsets and masks for each case. Same goes for
    # BLOCK_SIZE_SCALE_B.
    offsets_scale_am = (
        tl.arange(0, BLOCK_SIZE_SCALE_A)
        + (BLOCK_SIZE_SCALE_A > 1) * pid_m * BLOCK_SIZE_M
    )
    masks_scale_am = offsets_scale_am < M

    offsets_scale_bn = (
        tl.arange(0, BLOCK_SIZE_SCALE_B)
        + (BLOCK_SIZE_SCALE_B > 1) * pid_n * BLOCK_SIZE_N
    )
    masks_scale_bn = offsets_scale_bn < N

    a_ptrs = a_ptr + offsets_a
    b_ptrs = b_ptr + offsets_b

    scale_a_ptrs = scale_a_ptr + offsets_scale_am
    scale_b_ptrs = scale_b_ptr + offsets_scale_bn

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        masks_k = offsets_k < K
        masks_a = masks_am[:, None] & masks_k[None, :]
        a = tl.load(a_ptrs, mask=masks_a)

        masks_b = masks_k[:, None] & masks_bn[None, :]
        b = tl.load(b_ptrs, mask=masks_b)

        # Accumulate results.
        accumulator = tl.dot(a, b, accumulator, out_dtype=accumulator_dtype)

        offsets_k += BLOCK_SIZE_K
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Apply scale at end.
    masks_scale_a = masks_scale_am[:, None] & (tl.arange(0, 1) < 1)[:, None]
    scale_a = tl.load(scale_a_ptrs[:, None], masks_scale_a)
    # Need to broadcast to the appropriate size, if scale_a is already
    # (BLOCK_SIZE_M, 1) then it will broadcast to its own shape. Same goes
    # for scale_b below.
    scale_a = scale_a.broadcast_to((BLOCK_SIZE_M, 1))
    accumulator = scale_a * accumulator.to(tl.float32)

    masks_scale_b = masks_scale_bn[:, None] & (tl.arange(0, 1) < 1)[None, :]
    scale_b = tl.load(scale_b_ptrs[:, None], masks_scale_b)
    scale_b = scale_b.broadcast_to((BLOCK_SIZE_N, 1))
    accumulator = scale_b.T * accumulator.to(tl.float32)

    # Convert to output format.
    c = accumulator.to(c_ptr.type.element_ty)

    # Add bias, it's already in output format, so add it after conversion.
    if bias_ptr:
        offsets_bias = offsets_bn
        bias_ptrs = bias_ptr + offsets_bias
        bias_mask = offsets_bias < N
        bias = tl.load(bias_ptrs, bias_mask)
        c += bias

    # Save output
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    offs_cm = offs_cm.to(tl.int64)
    offs_cn = offs_cn.to(tl.int64)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    tl.store(c_ptrs, c, mask=c_mask)


# input  - [M, K]
# weight - [K, N]
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/compressed_tensors/triton_scaled_mm.py
def triton_scaled_mm(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: type[torch.dtype],
    bias: Optional[torch.Tensor] = None,
    block_size_m: int = 32,
    block_size_n: int = 32,
    block_size_k: int = 32,
    use_heuristic=True,
) -> torch.Tensor:
    M, K = input.shape
    N = weight.shape[1]

    assert N > 0 and K > 0 and M > 0
    assert weight.shape[0] == K
    assert input.dtype == weight.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert bias is None or bias.is_floating_point()
    assert is_weak_contiguous(input)
    assert is_weak_contiguous(weight)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    result = torch.empty((M, N), dtype=out_dtype, device=input.device)

    has_scalar = lambda x: x.shape[0] == 1 and x.shape[1] == 1

    if use_heuristic:
        is_small_N = N < 8192
        next_power_of_2_M = max(32, triton.next_power_of_2(M))
        if next_power_of_2_M <= 32:
            tile_shape = (64, 64, 256) if is_small_N else (64, 128, 256)
        elif next_power_of_2_M <= 64:
            tile_shape = (64, 64, 256)
        elif next_power_of_2_M <= 128:
            tile_shape = (64, 128, 128)
        else:
            tile_shape = (128, 128, 128)

    block_size_m, block_size_n, block_size_k = tile_shape

    block_size_sa = 1 if has_scalar(scale_a) else block_size_m
    block_size_sb = 1 if has_scalar(scale_b) else block_size_n

    accumulator_dtype = tl.float32 if input.is_floating_point() else tl.int32

    # A = input, B = weight, C = result
    # A = M x K, B = K x N, C = M x N
    scaled_mm_kernel[grid](
        input,
        weight,
        scale_a,
        scale_b,
        result,
        bias,
        M,
        N,
        K,
        input.stride(0),
        input.stride(1),
        weight.stride(0),
        weight.stride(1),
        result.stride(0),
        result.stride(1),
        accumulator_dtype,
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=block_size_k,
        BLOCK_SIZE_SCALE_A=block_size_sa,
        BLOCK_SIZE_SCALE_B=block_size_sb,
    )

    return result.to(out_dtype)


if _is_cuda:
    if enable_sgl_per_token_group_quant_8bit:

        @torch.library.register_fake("sgl_kernel::sgl_per_token_group_quant_8bit")
        def _(
            input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        ):
            return

    else:

        @torch.library.register_fake("sgl_kernel::sgl_per_token_group_quant_fp8")
        def _(
            input, output_q, output_s, group_size, eps, fp8_min, fp8_max, scale_ue8m0
        ):
            return

    # FIXME: for some models, this fake registration will cause NaN outputs.
    # So we gate the fake registration with an environment variable for them.
    if not get_bool_env_var("SGLANG_DISABLE_SGL_KERNEL_FAKE_REGISTER"):

        @torch.library.register_fake("sgl_kernel::sgl_per_token_quant_fp8")
        def _(input, output_q, output_s):
            return
