# DeepSeek-only build: Tokenizer patching removed
# Only Kimi tokenizers required patching, which are not supported in this build.


def patch_tokenizer(tokenizer):
    """No-op for DeepSeek-only build - tokenizer patching not required."""
    return tokenizer


def unpatch_tokenizer(tokenizer):
    """No-op for DeepSeek-only build - tokenizer patching not required."""
    return tokenizer
