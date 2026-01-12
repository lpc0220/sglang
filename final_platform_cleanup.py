#!/usr/bin/env python3
"""
Final cleanup: Remove remaining NPU/HIP platform conditionals from core infrastructure files.
These are the last remaining platform checks that need to be resolved.
"""

import re
from pathlib import Path

# Files with remaining platform checks
files_to_fix = {
    "python/sglang/srt/server_args.py": [
        {
            "description": "Remove NPU checks in server args",
            "find": r'if not is_npu\(\):  # CUDA GPU\n',
            "replace": ""
        },
        {
            "description": "Remove MindSpore NPU assertion",
            "find": r'\s+assert is_npu\(\), "MindSpore model impl is only supported on Ascend npu\."\n',
            "replace": ""
        },
    ],
    "python/sglang/srt/layers/rotary_embedding.py": [
        {
            "description": "Remove NPU import conditional",
            "find": r'if is_npu\(\):\n[^\n]*\n',
            "replace": ""
        },
    ],
    "python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py": [
        {
            "description": "Remove NPU dtype conditional - use CUDA dtype",
            "find": r'torch\.int64 if not is_npu\(\) else torch\.int32',
            "replace": "torch.int64"
        },
    ],
    "python/sglang/srt/utils/common.py": [
        {
            "description": "Remove NPU backend check - use CUDA",
            "find": r'"CUDA" if not is_npu\(\) else "PrivateUse1"',
            "replace": '"CUDA"'
        },
    ],
    "python/sglang/srt/compilation/backend.py": [
        {
            "description": "Remove NPU backend conditional",
            "find": r'backend_cls = CUDAPiecewiseBackend if not is_npu\(\) else NPUPiecewiseBackend',
            "replace": "backend_cls = CUDAPiecewiseBackend"
        },
    ],
    "python/sglang/srt/layers/quantization/__init__.py": [
        {
            "description": "Remove HIP from quantization check",
            "find": r'if is_cuda\(\) or \(_is_mxfp_supported and is_hip\(\)\):',
            "replace": "if is_cuda():"
        },
    ],
    "python/sglang/srt/model_executor/forward_batch_info.py": [
        {
            "description": "Remove HIP dtype conditional - use CUDA dtype",
            "find": r'positions_dtype = torch\.int64 if is_hip\(\) else torch\.int32',
            "replace": "positions_dtype = torch.int32"
        },
    ],
    "python/sglang/srt/configs/model_config.py": [
        {
            "description": "Remove ROCm quantization check",
            "find": r'\s+if is_hip\(\) and self\.quantization not in rocm_supported_quantization:\n[^\n]*\n[^\n]*\n',
            "replace": ""
        },
    ],
    "python/sglang/srt/configs/load_config.py": [
        {
            "description": "Remove ROCm load format check",
            "find": r'\s+if is_hip\(\) and load_format in rocm_not_supported_load_format:\n[^\n]*\n',
            "replace": ""
        },
    ],
    "python/sglang/srt/speculative/ngram_info.py": [
        {
            "description": "Remove HIP branch",
            "find": r'elif is_hip\(\):\n[^\n]*\n',
            "replace": ""
        },
    ],
}

def apply_fixes(file_path, fixes):
    """Apply regex fixes to a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    for fix in fixes:
        new_content = re.sub(fix["find"], fix["replace"], content, flags=re.MULTILINE)
        if new_content != content:
            changes.append(fix["description"])
            content = new_content

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    return False, []

if __name__ == "__main__":
    processed = 0
    for file_rel, fixes in files_to_fix.items():
        file_path = Path(file_rel)
        if file_path.exists():
            changed, changes = apply_fixes(file_path, fixes)
            if changed:
                print(f"✅ {file_rel}: {', '.join(changes)}")
                processed += 1
            else:
                print(f"⚠️  {file_rel}: No changes made")
        else:
            print(f"❌ {file_rel}: File not found")

    print(f"\n✅ Processed {processed} files")
