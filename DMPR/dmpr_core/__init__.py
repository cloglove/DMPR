"""Dense mask processing modules."""

from .augmentation import mask_aware_bipo
from .masks import ATIAConfig, Candidate, DensePromptResult, generate_dense_prompts

__all__ = [
    "ATIAConfig",
    "Candidate",
    "DensePromptResult",
    "generate_dense_prompts",
    "mask_aware_bipo",
]
