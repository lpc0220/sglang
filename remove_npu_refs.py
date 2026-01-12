#!/usr/bin/env python3
"""
Remove all NPU/Ascend references from SGLang codebase.
This script removes _is_npu variables and NPU conditional branches.
"""

import re
import sys
from pathlib import Path

def remove_npu_refs(file_path):
    """Remove NPU references from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content
    changes = []

    # Remove imports
    import_patterns = [
        (r',\s*is_npu\s*,', ','),
        (r',\s*is_npu\s*\)', ')'),
        (r'from sglang\.srt\.utils import is_npu\n', ''),
    ]

    for pattern, replacement in import_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append(f'Removed is_npu import')
            content = new_content

    # Remove module-level variables
    var_patterns = [
        (r'_is_npu\s*=\s*is_npu\(\)\n', ''),
    ]

    for pattern, replacement in var_patterns:
        new_content = re.sub(pattern, replacement, content)
        if new_content != content:
            changes.append('Removed _is_npu variable')
            content = new_content

    # Write back if changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, changes
    return False, []

# Files with NPU references
files_to_process = [
    "python/sglang/srt/utils/common.py",
    "python/sglang/srt/models/deepseek_common/utils.py",
]

if __name__ == "__main__":
    processed = 0
    for file_rel in files_to_process:
        file_path = Path(file_rel)
        if file_path.exists():
            changed, changes = remove_npu_refs(file_path)
            if changed:
                print(f"✅ {file_rel}: {', '.join(changes)}")
                processed += 1
            else:
                print(f"⚠️  {file_rel}: No changes needed")
        else:
            print(f"❌ {file_rel}: File not found")

    print(f"\n✅ Processed {processed} files")
