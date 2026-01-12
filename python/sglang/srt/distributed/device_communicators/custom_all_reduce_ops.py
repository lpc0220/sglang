# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/_custom_ops.py
import logging
from typing import List, Optional, Tuple

import torch

from sglang.srt.utils import is_cuda

logger = logging.getLogger(__name__)

_is_cuda = is_cuda()

IS_CUSTOM_AR_AVAILABLE = _is_cuda
IS_QUICK_AR_AVAILABLE = False
# TODO(zyksir): mscclpp is untested on AMD and therefore disabled.
IS_MSCCLPP_AR_AVAILABLE = _is_cuda

try:
    import sgl_kernel.allreduce as _custom_ar
except ImportError as e:
    if _is_cuda:
        logger.warning("Failed to import from custom_ar with %r", e)
    IS_CUSTOM_AR_AVAILABLE = False
    IS_QUICK_AR_AVAILABLE = False
    IS_MSCCLPP_AR_AVAILABLE = False

# region IS_CUSTOM_AR_AVAILABLE

if not IS_CUSTOM_AR_AVAILABLE:
    pass

elif _is_cuda:
    # CUDA custom allreduce

    def init_custom_ar(
        ipc_tensors: List[torch.Tensor],
        rank_data: torch.Tensor,
        rank: int,
        full_nvlink: bool,
    ) -> int:
        return _custom_ar.init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)

    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
    ) -> None:
        _custom_ar.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes)

    def dispose(fa: int) -> None:
        _custom_ar.dispose(fa)

    def meta_size() -> int:
        return _custom_ar.meta_size()

    def register_buffer(fa: int, ipc_tensors: List[int]) -> None:
        return _custom_ar.register_buffer(fa, ipc_tensors)

    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return _custom_ar.get_graph_buffer_ipc_meta(fa)

    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        _custom_ar.register_graph_buffers(fa, handles, offsets)