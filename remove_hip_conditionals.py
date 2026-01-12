#!/usr/bin/env python3
"""
Script to remove HIP/AMD/aiter conditionals from deepseek_v2.py
Preserves CUDA code paths only.
"""

import re


def remove_hip_from_deepseek_v2(input_file):
    """Remove all HIP/AMD conditional blocks from deepseek_v2.py"""

    with open(input_file, 'r') as f:
        content = f.read()

    original_content = content

    # Pattern 1: Remove standalone if _use_aiter_gfx95 blocks that have else branches
    # These are conditional feature enables that we want to skip

    # Pattern 2: Remove elif _use_aiter_gfx95 branches (keep the if/else structure)
    pattern_elif_aiter = r'(\s*)elif\s+_use_aiter_gfx95[^\n]*\n((?:(?!\1(?:elif|else|if\s+\w|\w+\s*=))[^\n]*\n)*)'
    content = re.sub(pattern_elif_aiter, '', content)

    # Pattern 3: Remove elif _is_hip branches
    pattern_elif_hip = r'(\s*)elif\s+_is_hip[^\n]*\n((?:(?!\1(?:elif|else|if\s+\w|\w+\s*=))[^\n]*\n)*)'
    content = re.sub(pattern_elif_hip, '', content)

    # Pattern 4: if _is_hip: blocks before else:
    pattern_if_hip_else = r'(\s*)if\s+_is_hip[^\n]*\n((?:(?!\1(?:else|elif|if\s+\w|\w+\s*=))[^\n]*\n)*)(\s*)else:\n'
    content = re.sub(pattern_if_hip_else, '', content)

    # Pattern 5: Standalone if _use_aiter_gfx95 blocks (remove entire block)
    pattern_if_aiter_standalone = r'(\s*)if\s+_use_aiter_gfx95[^\n]*\n((?:(?!\1(?:if\s+\w|\w+\s*=))[^\n]*\n)*)'
    content = re.sub(pattern_if_aiter_standalone, '', content)

    # Pattern 6: Complex conditionals with "and _use_aiter"
    content = re.sub(r'\s+and\s+_use_aiter(?!_)', '', content)
    content = re.sub(r'\s+and\s+_use_aiter_gfx95', '', content)
    content = re.sub(r'\s+and\s+not\s+_use_aiter', '', content)

    # Pattern 7: OR conditions with _is_hip
    content = re.sub(r'or\s+_use_aiter_gfx95', '', content)
    content = re.sub(r'\s+or\s+not\s+_is_gfx95_supported', '', content)

    # Pattern 8: elif _is_hip and <condition>: blocks
    pattern_elif_hip_and = r'(\s*)elif\s+_is_hip\s+and[^\n]*\n((?:(?!\1(?:elif|else|if\s+\w|\w+\s*=))[^\n]*\n)*)'
    content = re.sub(pattern_elif_hip_and, '', content)

    # Write back
    with open(input_file, 'w') as f:
        f.write(content)

    # Count changes
    lines_removed = original_content.count('\n') - content.count('\n')

    return lines_removed


if __name__ == "__main__":
    input_file = "python/sglang/srt/models/deepseek_v2.py"
    lines_removed = remove_hip_from_deepseek_v2(input_file)
    print(f"Removed {lines_removed} lines from {input_file}")
    print("Note: This is a first pass. Manual review may be needed for complex nested conditionals.")
