# Phase 3B Completion Summary: CPU/XPU/HPU/NPU Removal

## Objective
Remove all CPU/XPU/HPU/NPU conditional code and backends from the SGLang codebase, targeting NVIDIA GPU only.

## Scope of Changes
**Total Lines Removed**: 201 lines across 12 files
**Files Modified**: 12 files

## Key Changes

### 1. Core Utility Functions Removed (`sglang/srt/utils/common.py`)
**Lines Removed**: 108 lines

Removed Functions:
- `is_hpu()` - HPU (Habana) detection
- `is_xpu()` - XPU (Intel GPU) detection
- `is_npu()` - NPU (Ascend) detection
- `is_cpu()` - CPU backend detection
- `is_host_cpu_x86()` - x86 CPU detection
- `is_host_cpu_arm64()` - ARM64 CPU detection
- `cpu_has_amx_support()` - Intel AMX support detection
- `use_intel_amx_backend()` - Intel AMX backend selector
- `xpu_has_xmx_support()` - XPU matrix extension support
- `is_intel_amx_backend_available` - Intel AMX availability flag
- `is_amx_tile_supported` - AMX tile support flag

Modified Functions:
- `device_context()` - Removed CPU branch, kept only GPU logic
- `get_device()` - Removed CPU/XPU/NPU branches, kept only CUDA path

### 2. DeepSeek-Specific Files Cleaned

#### `/sglang/srt/models/deepseek_common/utils.py`
**Lines Removed**: 6 lines
- Removed imports: `cpu_has_amx_support`, `is_cpu`, `is_npu`
- Removed variables: `_is_npu`, `_is_cpu_amx_available`, `_is_cpu`

#### `/sglang/srt/models/deepseek_common/attention_backend_handler.py`
**Lines Removed**: 25 lines
- Removed import: `use_intel_amx_backend`
- Removed function: `handle_attention_ascend()` (entire NPU backend handler)
- Removed from `_dispatch_mla_subtype()`: Intel AMX conditional (lines 30-31)
- Removed backend registration: `AttentionBackendRegistry.register("ascend", handle_attention_ascend)`

### 3. Torch Multiprocessing Patch (`sglang/srt/utils/patch_torch.py`)
**Lines Removed**: 42 lines
- Removed NPU-specific tensor rebuilding logic
- Removed `torch_npu.multiprocessing.reductions` imports
- Removed `_rebuild_npu_tensor_modified()` function
- Removed `npu_verl_to_sglang()` function
- Simplified `monkey_patch_torch_reductions()` to CUDA-only path

### 4. Layer Infrastructure Files

#### `/sglang/srt/layers/activation.py`
**Lines Removed**: 3 lines
- Removed: `if is_npu(): import torch_npu`

#### `/sglang/srt/layers/sampler.py`
**Lines Removed**: 5 lines
- Removed import: `is_npu`
- Removed: `if is_npu(): import torch_npu`

#### `/sglang/srt/layers/quantization/__init__.py`
**Lines Removed**: 9 lines
- Removed NPU ModelSlim quantization config import
- Removed ModelSlim registration from BASE_QUANTIZATION_METHODS

#### `/sglang/srt/layers/attention/attention_registry.py`
**Lines Removed**: 4 lines
- Removed NPU assertion for hybrid GDN models
- Removed Ascend backend requirement check

### 5. Runtime Files

#### `/sglang/srt/model_executor/model_runner.py`
**Lines Removed**: 2 lines
- Removed: `if is_npu(): register_sgl_tp_rank(self.gpu_id)`

#### `/sglang/srt/compilation/weak_ref_tensor.py`
**Lines Removed**: 6 lines
- Removed import: `is_npu`
- Removed NPU weak_ref_tensor import path
- Updated error message to CUDA-only

#### `/sglang/srt/elastic_ep/elastic_ep.py`
**Lines Removed**: 4 lines
- Removed CPU device selection branch
- Updated error message: "Only CUDA supports elastic ep"

#### `/sglang/srt/disaggregation/utils.py`
**Lines Removed**: 5 lines
- Removed NPU device assignment for ascend backend

## Validation Results

### Syntax Validation
✓ All 12 modified files have valid Python syntax
✓ No syntax errors introduced

### Key Files Validated
- `/sglang/srt/utils/common.py` ✓
- `/sglang/srt/models/deepseek_common/utils.py` ✓
- `/sglang/srt/models/deepseek_common/attention_backend_handler.py` ✓
- `/sglang/srt/utils/patch_torch.py` ✓
- All layer files ✓

## Known Remaining Issues

### Module-Level is_cpu() Calls (17 occurrences)
The following files cache is_cpu() at module level and will need updates:
1. `sglang/srt/layers/layernorm.py` - Line 46
2. `sglang/srt/layers/linear.py` - Line 71
3. `sglang/srt/layers/vocab_parallel_embedding.py` - Line 41
4. `sglang/srt/layers/utils/multi_platform.py` - Line 16
5. `sglang/srt/layers/quantization/w8a8_int8.py` - Line 38
6. `sglang/srt/layers/quantization/fp8.py` - Line 89
7. `sglang/srt/layers/quantization/unquant.py` - Line 42
8. `sglang/srt/layers/quantization/fp8_kernel.py` - Line 41
9. `sglang/srt/layers/activation.py` - Line 47
10. `sglang/srt/layers/rotary_embedding.py` - Line 32
11. `sglang/srt/layers/parameter.py` - Line 27
12. `sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py` - Line 41
13. `sglang/srt/layers/moe/fused_moe_triton/fused_moe.py` - Line 41
14. `sglang/srt/layers/moe/fused_moe_triton/layer.py` - Line 85
15. `sglang/srt/layers/moe/moe_runner/triton.py` - Line 34
16. `sglang/srt/layers/moe/topk.py` - Line 70
17. `sglang/srt/distributed/parallel_state.py` - Line 59

**Impact**: These will cause ImportError when the modules are loaded.

**Solution**: These lines should be removed and any usage of `_is_cpu` or `_is_cpu_amx_available` should be removed from conditional branches.

### NPU-Specific Functions (3 occurrences)
1. `sglang/srt/layers/rotary_embedding.py` - Contains NPU-specific rotary embedding implementation
2. `sglang/srt/layers/attention/nsa/nsa_indexer.py` - NPU-specific imports
3. `sglang/srt/lora/backend/ascend_backend.py` - Entire file is Ascend/NPU specific

**Solution**: Remove these NPU-specific code blocks or entire files.

## Target Platform After Changes
- **Supported**: NVIDIA GPU (CUDA) only
- **Removed**: CPU, XPU (Intel GPU), HPU (Habana), NPU (Ascend)
- **Kept**: HIP/ROCm support (for AMD GPUs - to be addressed in Phase 3C)

## Next Steps
1. **Phase 3B Continuation**: Remove remaining module-level `is_cpu()` calls and associated conditional code
2. **Phase 3C**: Remove AMD GPU (HIP/ROCm) support to target NVIDIA-only
3. **Phase 4**: Full testing and validation

## Git Stats
```
12 files changed, 18 insertions(+), 201 deletions(-)
```

---
**Date**: 2026-01-11
**Phase**: 3B - CPU/XPU/HPU/NPU Removal
**Status**: Partially Complete (core functions removed, cleanup needed)
