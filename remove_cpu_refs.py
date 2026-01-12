#!/usr/bin/env python3
"""
Script to remove all _is_cpu and _is_cpu_amx_available module-level variable usages.
This is part of Phase 3B - removing CPU/XPU/HPU support.
"""

import re
import sys
from pathlib import Path

def remove_cpu_imports_and_vars(file_path):
    """Remove _is_cpu and _is_cpu_amx_available from a file."""
    with open(file_path, 'r') as f:
        content = f.read()

    original_content = content
    changes = []

    # Remove imports
    import_patterns = [
        (r',\s*is_cpu\s*,', ',', 'Remove is_cpu from import list'),
        (r',\s*cpu_has_amx_support\s*,', ',', 'Remove cpu_has_amx_support from import list'),
        (r',\s*is_cpu\s*\)', ')', 'Remove is_cpu from end of import'),
        (r',\s*cpu_has_amx_support\s*\)', ')', 'Remove cpu_has_amx_support from end of import'),
    ]

    for pattern, replacement, desc in import_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append(desc)
            content = new_content

    # Remove module-level variables
    var_patterns = [
        (r'_is_cpu\s*=\s*is_cpu\(\)\n', ''),
        (r'_is_cpu_amx_available\s*=\s*cpu_has_amx_support\(\)\n', ''),
    ]

    for pattern, replacement in var_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append(f'Remove {pattern.split("=")[0].strip()}')
            content = new_content

    # Write back if changed
    if content != original_content:
        with open(file_path, 'w') as f:
            f.write(content)
        return True, changes
    return False, []

# Files to process
files_to_process = [
    "python/sglang/srt/layers/moe/fused_moe_triton/fused_moe_triton_kernels.py",
    "python/sglang/srt/layers/moe/fused_moe_triton/fused_moe.py",
    "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
    "python/sglang/srt/layers/moe/moe_runner/triton.py",
    "python/sglang/srt/layers/moe/topk.py",
    "python/sglang/srt/layers/quantization/w8a8_int8.py",
    "python/sglang/srt/layers/quantization/fp8.py",
    "python/sglang/srt/layers/quantization/unquant.py",
    "python/sglang/srt/layers/quantization/fp8_kernel.py",
    "python/sglang/srt/layers/utils/multi_platform.py",
    "python/sglang/srt/distributed/parallel_state.py",
    "python/sglang/srt/model_executor/model_runner.py",
]

if __name__ == "__main__":
    processed = 0
    for file_rel in files_to_process:
        file_path = Path(file_rel)
        if file_path.exists():
            changed, changes = remove_cpu_imports_and_vars(file_path)
            if changed:
                print(f"✅ {file_rel}: {', '.join(changes)}")
                processed += 1
            else:
                print(f"⚠️  {file_rel}: No changes needed")
        else:
            print(f"❌ {file_rel}: File not found")

    print(f"\n✅ Processed {processed} files")
