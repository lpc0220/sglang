# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/custom_all_reduce.py

import ctypes
import logging
import os
from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import sglang.srt.distributed.device_communicators.custom_all_reduce_ops as ops
from sglang.srt.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    gpu_p2p_access_check,
    is_full_nvlink,
    is_weak_contiguous,
)
from sglang.srt.distributed.parallel_state import in_the_same_node_as
from sglang.srt.environ import envs
from sglang.srt.utils import get_bool_env_var, is_cuda, is_hip, log_info_on_rank0

_is_cuda = is_cuda()

logger = logging.getLogger(__name__)


def _can_p2p(rank: int, world_size: int) -> bool:
    # SGLANG_SKIP_P2P_CHECK can be set to False in sglang
    SGLANG_SKIP_P2P_CHECK = os.getenv("SGLANG_SKIP_P2P_CHECK", "0") == "1"
    for i in range(world_size):
        if i == rank:
            continue
        if SGLANG_SKIP_P2P_CHECK:
            logger.info("Skipping P2P check and trusting the driver's P2P report.")
            return torch.cuda.can_device_access_peer(rank, i)
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


class CustomAllreduce:
    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
    _MAX_CAR_SIZE = 8192 * 1024
            logger.debug("[AR] All-reduce: default")

    # Check if 1-stage AR should be used
    if envs.SGLANG_USE_1STAGE_ALLREDUCE.is_set():
        use_1stage = envs.SGLANG_USE_1STAGE_ALLREDUCE.get()
    else:
        use_1stage = envs.SGLANG_ENABLE_DETERMINISTIC_INFERENCE.get()

    # On AMD with 1-stage AR, use sglang's CustomAllreduce
    # (AiterCustomAllreduce doesn't have deterministic_all_reduce method)
    if use_1stage:
        return CustomAllreduce

    if get_bool_env_var("SGLANG_USE_AITER_AR", default="true"):
        try:
            from aiter.dist.device_communicators.custom_all_reduce import (
                CustomAllreduce as AiterCustomAllreduce,
            )

            logger.info("[AR] Using AiterCustomAllreduce (AMD default)")
            return AiterCustomAllreduce
        except ImportError as e:
            logger.warning(
                "[AR] Aiter custom all-reduce not available; "
                "falling back to sglang CustomAllreduce. Details: %s",
                e,
            )
            return CustomAllreduce

    return CustomAllreduce
