from typing import List, Optional, Tuple

import torch

# HIP/ROCM allreduce branch removed - NVIDIA CUDA-only build
# Only CUDA implementations are kept below


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    return torch.ops.sgl_kernel.init_custom_ar.default(
        ipc_tensors, rank_data, rank, full_nvlink
    )


def dispose(fa: int) -> None:
    torch.ops.sgl_kernel.dispose.default(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
) -> None:
    torch.ops.sgl_kernel.all_reduce.default(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes
    )


def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
    return torch.ops.sgl_kernel.get_graph_buffer_ipc_meta.default(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    return torch.ops.sgl_kernel.register_buffer.default(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    torch.ops.sgl_kernel.register_graph_buffers.default(fa, handles, offsets)


def meta_size() -> int:
    return torch.ops.sgl_kernel.meta_size.default()


def mscclpp_generate_unique_id() -> torch.Tensor:
    return torch.ops.sgl_kernel.mscclpp_generate_unique_id.default()


def mscclpp_init_context(
    unique_id: torch.Tensor,
    rank: int,
    world_size: int,
    scratch: torch.Tensor,
    put_buffer: torch.Tensor,
    nranks_per_node: int,
    rank_to_node: List[int],
    rank_to_ib: List[int],
    context_selection: int,
) -> int:
    return torch.ops.sgl_kernel.mscclpp_init_context.default(
        unique_id,
        rank,
        world_size,
        scratch,
        put_buffer,
        nranks_per_node,
        rank_to_node,
        rank_to_ib,
        context_selection,
    )


def mscclpp_allreduce(
    context: int, inp: torch.Tensor, out: torch.Tensor, nthreads: int, nblocks: int
) -> None:
    torch.ops.sgl_kernel.mscclpp_allreduce.default(
        context, inp, out, nthreads, nblocks
    )
