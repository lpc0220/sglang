from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import (
    get_device_sm,
    is_cuda,
)

_is_cuda = is_cuda()
_is_fp8_fnuz = is_fp8_fnuz()
_device_sm = get_device_sm()
