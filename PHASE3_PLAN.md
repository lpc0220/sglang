# Phase 3: Deep Cleanup Plan

**Status:** Planning
**Last Updated:** 2026-01-11

## Overview

Phase 2 achieved ~18% code reduction (469 files, ~118,894 lines). Phase 3 targets deeper cleanup requiring careful dependency analysis and conditional code removal.

## Current State After Phase 2

✅ **Completed:**
- All non-DeepSeek models removed (117 files)
- NPU/Ascend hardware backends removed (32 files + 29 test files)
- AMD test infrastructure removed (8 test files)
- Multimodal variants removed (8 files)
- Orphaned configs and processors removed (47 files)
- Non-DeepSeek benchmarks removed (156 files)
- Test files for removed models (21 files)
- Platform documentation removed (5 files)

**Total:** 469 files, ~118,894 lines removed

❌ **Remaining (Phase 3 Targets):**
- AMD/ROCm conditional code in source files
- NPU conditional code in source files
- Unused layers and CUDA kernels
- Dead imports and references
- Hardware-specific optimizations

---

## Phase 3 Targets

### 1. AMD/ROCm Conditional Code Removal

**Complexity:** HIGH - Requires careful analysis to avoid breaking NVIDIA path

#### A. ROCm-Specific Files (4 files, ~613 lines)
Standalone files that can be safely deleted:

```
✅ python/sglang/srt/layers/attention/triton_ops/rocm_mla_decode_rope.py (439 lines)
✅ python/sglang/srt/layers/moe/rocm_moe_utils.py (116 lines)
✅ python/sglang/srt/layers/quantization/rocm_mxfp4_utils.py (13 lines)
✅ python/sglang/srt/layers/rocm_linear_utils.py (45 lines)
```

**Dependencies:**
- Imported in `deepseek_v2.py` under `if _use_aiter_gfx95:` (AMD GFX95 GPUs)
- Imported in `deepseek_v2.py` under `elif _is_hip:` (AMD ROCm)
- Imported in `layers/quantization/quark/quark_moe.py`
- Imported in `layers/communicator.py`

**Action Plan:**
1. Remove conditional import blocks in deepseek_v2.py:
   - Lines 175-195: `if _use_aiter_gfx95:` block
   - Lines 208-214: `elif _is_hip:` block
2. Remove all `elif _is_hip:` code branches throughout deepseek_v2.py
3. Remove ROCm-specific function calls
4. Delete the 4 ROCm files
5. Run validation tests

**Risk:** MEDIUM
- Need to ensure NVIDIA code paths remain intact
- Many conditional branches to remove (~20+ locations in deepseek_v2.py)

#### B. ROCm Conditional Code in Other Files (~72 files)

From original analysis, ~72 files contain conditional ROCm/HIP code. Need to:

1. Scan for `_is_hip`, `_use_aiter_gfx95`, `rocm`, `hip` references
2. Remove conditional branches
3. Simplify logic where possible

**Estimated Impact:** ~5,000-8,000 lines (conditional blocks + imports)

---

### 2. NPU Conditional Code Removal

**Complexity:** MEDIUM

#### A. NPU-Specific Files (2+ files, ~148 lines)

```
✅ python/sglang/srt/distributed/device_communicators/npu_communicator.py (39 lines)
✅ python/sglang/srt/compilation/npu_piecewise_backend.py (109 lines)
```

**Dependencies:**
- Imported in `distributed/parallel_state.py` (conditional)
- Imported in `compilation/backend.py`
- Used when `use_npu_communicator=True` flag is set

**Action Plan:**
1. Remove NPU imports from parallel_state.py and backend.py
2. Remove `use_npu_communicator` parameter and logic
3. Remove `elif _is_npu:` branches in deepseek_v2.py and other files
4. Delete NPU communicator and backend files

**Risk:** LOW
- Already removed NPU backend in Batch 3
- Remaining code is just communicator infrastructure

#### B. NPU Conditional Code in Source Files

Files containing `_is_npu` references:
- deepseek_v2.py: Lines 215-227 (NPU-specific imports and logic)
- Other model files and utilities

**Estimated Impact:** ~2,000-3,000 lines

---

### 3. Unused Layers and Components

**Complexity:** VERY HIGH - Requires dependency graph analysis

#### Strategy:
1. Build call graph for DeepSeek models
2. Identify which layers are actually used by DeepSeek
3. Mark unused layers for removal
4. Validate with tests

#### Potential Targets:
- Attention mechanisms not used by DeepSeek MLA
- Unused normalization layers
- Unused activation functions
- Model-specific utilities (for removed models)

**Estimated Impact:** ~10,000-20,000 lines (very rough estimate)

**Risk:** VERY HIGH
- Need accurate dependency tracking
- Could break DeepSeek if wrong layers removed
- Should be done last after all other cleanup

---

### 4. Dead Imports and References

**Complexity:** LOW-MEDIUM

#### Action Plan:
1. Run import analysis on remaining files
2. Remove unused imports
3. Remove dead code (unreachable branches, commented code)
4. Clean up string references to removed models

**Tools to use:**
- Python import analyzer
- grep for removed model names
- Code coverage analysis

**Estimated Impact:** ~1,000-2,000 lines

---

### 5. Unused CUDA Kernels

**Complexity:** VERY HIGH

#### Strategy:
1. Map CUDA kernel usage in DeepSeek models
2. Identify kernels only used by removed models
3. Safe to remove: kernels with no callers
4. Keep: all kernels called by DeepSeek code

**Estimated Impact:** Unknown - need analysis first

**Risk:** VERY HIGH
- CUDA kernels are critical for performance
- Incorrect removal could break inference
- Should be done after comprehensive testing

---

## Phase 3 Execution Plan

### Recommended Order:

1. **Batch 7: Dead Imports and References** (LOW risk)
   - Clean up imports
   - Remove commented code
   - Low impact, easy validation

2. **Batch 8: NPU Conditional Code** (LOW-MEDIUM risk)
   - Remove NPU communicator files
   - Remove NPU conditional branches
   - Already removed NPU backend in Phase 2

3. **Batch 9: AMD/ROCm File Deletion** (MEDIUM risk)
   - Remove 4 ROCm-specific files
   - Remove conditional import blocks
   - Test on NVIDIA GPU required

4. **Batch 10: AMD/ROCm Conditional Code** (HIGH risk)
   - Remove all `elif _is_hip:` branches
   - Remove ROCm function calls
   - Extensive testing required

5. **Batch 11: Unused Layers** (VERY HIGH risk)
   - Requires comprehensive dependency analysis
   - Should be last major cleanup
   - Conservative approach

6. **Batch 12: CUDA Kernel Cleanup** (VERY HIGH risk)
   - Only if confident in call graph analysis
   - Optional - kernels don't hurt if unused
   - Can defer to future

---

## Testing Strategy for Phase 3

### After Each Batch:
1. Run syntax validation (tests/test_syntax.py)
2. Run broken reference detection (tests/test_no_broken_refs.py)
3. Check for import errors

### Before Final Validation:
1. Test on NVIDIA GPU with actual DeepSeek model
2. Run inference tests
3. Validate performance (throughput, latency)
4. Compare outputs with original SGLang

### Success Criteria:
✅ All syntax tests pass
✅ No broken references
✅ DeepSeek models load successfully
✅ Inference produces correct outputs
✅ Performance within 5% of original

---

## Estimated Final Impact

| Phase | Files Removed | Lines Removed | Reduction |
|-------|--------------|---------------|-----------|
| Phase 2 (Complete) | 469 | ~118,894 | 18% |
| Phase 3 (Estimated) | 100-150 | ~20,000-35,000 | 3-5% |
| **Total** | **569-619** | **~138,894-153,894** | **21-23%** |

**Note:** Original estimate of 60-70% reduction may have been too aggressive. A realistic target is 20-25% with Phase 2+3, focusing on safe removals that preserve DeepSeek functionality.

---

## Risk Assessment

| Batch | Risk Level | Mitigation |
|-------|-----------|------------|
| Batch 7 (Dead imports) | LOW | Easy to validate, no logic changes |
| Batch 8 (NPU conditional) | LOW-MEDIUM | Already removed backend, just cleanup |
| Batch 9 (ROCm files) | MEDIUM | Test on NVIDIA GPU after removal |
| Batch 10 (ROCm conditional) | HIGH | Extensive code review + GPU testing |
| Batch 11 (Unused layers) | VERY HIGH | Requires dependency analysis tool |
| Batch 12 (CUDA kernels) | VERY HIGH | Optional, defer if uncertain |

---

## Next Steps

**Recommended Immediate Action:**
Start with **Batch 7** (Dead Imports and References) - low risk, immediate value.

**User Decision Required:**
1. Should we proceed with Phase 3?
2. What risk level is acceptable?
3. Do we have access to NVIDIA GPU for testing?
4. Should we stop at Phase 2 (18% reduction) and validate first?

---

## Notes

- Phase 2 achieved significant cleanup with zero risk to DeepSeek functionality
- Phase 3 involves deeper changes with higher risk
- Conservative approach recommended: test after each batch
- Can stop at any point if risk becomes too high
- Final validation on NVIDIA GPU cluster is critical before deployment
