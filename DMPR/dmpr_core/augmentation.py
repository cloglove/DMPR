"""Synchronized prompt and body-part mask updates during occlusion."""

import numpy as np


def mask_aware_bipo(
    prompt: np.ndarray,
    body_parts: np.ndarray,
    occluder_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Update the target, negative and body-part masks using an occluder mask."""
    prompt = np.asarray(prompt, dtype=np.float32)
    body_parts = np.asarray(body_parts, dtype=np.float32)
    occluder = np.asarray(occluder_mask, dtype=bool)

    if prompt.ndim != 3 or prompt.shape[0] != 2:
        raise ValueError("prompt must have shape (2, H, W).")
    if body_parts.ndim != 3 or body_parts.shape[0] < 2:
        raise ValueError("body_parts must have shape (K+1, H, W).")
    if body_parts.shape[1:] != prompt.shape[1:]:
        raise ValueError("Prompt and body-part mask sizes must match.")
    if occluder.shape != prompt.shape[1:]:
        raise ValueError("Occluder mask size must match the prompts.")

    visible = ~occluder
    updated_prompt = prompt.copy()
    updated_prompt[0] *= visible
    updated_prompt[1] = np.maximum(prompt[1], occluder) * (1.0 - updated_prompt[0])

    updated_parts = body_parts.copy()
    updated_parts[1:] *= visible[None, :, :]
    updated_parts[0] = 1.0 - np.max(updated_parts[1:], axis=0)

    return updated_prompt, updated_parts
