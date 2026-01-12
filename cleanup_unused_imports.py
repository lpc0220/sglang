#!/usr/bin/env python3
"""
Phase 3D: Remove unused platform detection imports.
This script removes imports of is_npu, is_hip, is_xpu, is_cpu, use_intel_amx_backend
that are no longer used after platform code removal.
"""

import re
import sys
from pathlib import Path

def cleanup_imports(file_path):
    """Remove unused platform detection imports from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Remove specific platform imports from import lines
    import_patterns = [
        # Remove is_npu
        (r',\s*is_npu\s*,', ',', 'Remove is_npu from import'),
        (r',\s*is_npu\s*\)', ')', 'Remove is_npu from end of import'),
        (r'\(\s*is_npu\s*,', '(', 'Remove is_npu from start of import'),

        # Remove is_hip
        (r',\s*is_hip\s*,', ',', 'Remove is_hip from import'),
        (r',\s*is_hip\s*\)', ')', 'Remove is_hip from end of import'),
        (r'\(\s*is_hip\s*,', '(', 'Remove is_hip from start of import'),

        # Remove is_xpu
        (r',\s*is_xpu\s*,', ',', 'Remove is_xpu from import'),
        (r',\s*is_xpu\s*\)', ')', 'Remove is_xpu from end of import'),
        (r'\(\s*is_xpu\s*,', '(', 'Remove is_xpu from start of import'),

        # Remove is_cpu
        (r',\s*is_cpu\s*,', ',', 'Remove is_cpu from import'),
        (r',\s*is_cpu\s*\)', ')', 'Remove is_cpu from end of import'),
        (r'\(\s*is_cpu\s*,', '(', 'Remove is_cpu from start of import'),

        # Remove use_intel_amx_backend
        (r',\s*use_intel_amx_backend\s*,', ',', 'Remove use_intel_amx_backend from import'),
        (r',\s*use_intel_amx_backend\s*\)', ')', 'Remove use_intel_amx_backend from end'),

        # Remove mxfp_supported (ROCm-specific)
        (r',\s*mxfp_supported\s*,', ',', 'Remove mxfp_supported from import'),
        (r',\s*mxfp_supported\s*\)', ')', 'Remove mxfp_supported from end'),
    ]

    for pattern, replacement, desc in import_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append(desc)
            content = new_content

    # Clean up empty imports like "from x import ()"
    content = re.sub(r'from\s+[\w\.]+\s+import\s+\(\s*\)\n', '', content)

    # Clean up double commas that might result from removals
    content = re.sub(r',\s*,', ',', content)

    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    return False, []

# Files identified with unused imports
files_to_process = [
    "python/sglang/srt/disaggregation/utils.py",
    "python/sglang/srt/layers/logits_processor.py",
    "python/sglang/srt/layers/attention/triton_ops/prefill_attention.py",
    "python/sglang/srt/layers/attention/triton_ops/decode_attention.py",
    "python/sglang/srt/layers/attention/triton_ops/extend_attention.py",
    "python/sglang/srt/layers/attention/triton_ops/double_sparsity_attention.py",
    "python/sglang/srt/layers/attention/base_attn_backend.py",
    "python/sglang/srt/layers/attention/nsa/tilelang_kernel.py",
    "python/sglang/srt/layers/attention/nsa/nsa_indexer.py",
    "python/sglang/srt/layers/attention/attention_registry.py",
    "python/sglang/srt/layers/attention/nsa_backend.py",
    "python/sglang/srt/layers/linear.py",
    "python/sglang/srt/layers/quantization/gguf.py",
    "python/sglang/srt/layers/quantization/__init__.py",
    "python/sglang/srt/layers/quantization/awq.py",
    "python/sglang/srt/layers/quantization/quark/schemes/quark_w4a4_mxfp4.py",
    "python/sglang/srt/layers/quantization/petit.py",
    "python/sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_fp8.py",
    "python/sglang/srt/layers/dp_attention.py",
    "python/sglang/srt/layers/moe/moe_runner/deep_gemm.py",
]

if __name__ == "__main__":
    processed = 0
    for file_rel in files_to_process:
        file_path = Path(file_rel)
        if file_path.exists():
            changed, changes = cleanup_imports(file_path)
            if changed:
                print(f"✅ {file_rel}: {', '.join(changes)}")
                processed += 1
            else:
                print(f"⚠️  {file_rel}: No changes needed")
        else:
            print(f"❌ {file_rel}: File not found")

    print(f"\n✅ Processed {processed} files")
