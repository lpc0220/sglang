"""
Pooler layer stub for DeepSeek-only build.

Note: Full embedding/pooler functionality removed. This stub provides
the EmbeddingPoolerOutput class for type hints compatibility.
DeepSeek models are generation-only and don't use pooler layers.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class EmbeddingPoolerOutput:
    """Stub for embedding pooler output (DeepSeek-only build)."""
    embeddings: Optional[torch.Tensor] = None
