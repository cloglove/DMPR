"""Human instance assignment and dense mask generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Candidate:
    mask: np.ndarray
    parts: np.ndarray
    confidence: float


@dataclass(frozen=True)
class ATIAConfig:
    confidence_threshold: float = 0.5
    area_ratio_threshold: float = 0.01
    center_weight: float = 0.7
    margin_threshold: float = 0.1
    center_width_fraction: float = 0.50
    upper_center_width_fraction: float = 0.60
    upper_center_height_fraction: float = 0.55
    upper_body_labels: tuple[int, ...] = ()
    eps: float = 1e-8


@dataclass
class DensePromptResult:
    prompt: np.ndarray
    body_parts: np.ndarray
    target_index: int | None
    valid_indices: tuple[int, ...]
    confident: bool
    margin: float | None
    scores: dict[int, float]
    status: str


def _region_mask(
    height: int,
    width: int,
    width_fraction: float,
    height_fraction: float = 1.0,
) -> np.ndarray:
    """Return a centered region extending from the top of the image."""
    if not 0.0 < width_fraction <= 1.0 or not 0.0 < height_fraction <= 1.0:
        raise ValueError("Region fractions must lie in (0, 1].")

    x0 = int(round((1.0 - width_fraction) * width / 2.0))
    x1 = int(round((1.0 + width_fraction) * width / 2.0))
    y1 = max(1, int(round(height_fraction * height)))

    region = np.zeros((height, width), dtype=bool)
    region[:y1, x0:x1] = True
    return region


def _upper_body_mask(candidate: Candidate, labels: tuple[int, ...]) -> np.ndarray:
    """Extract the visible upper-body region from part labels."""
    return np.asarray(candidate.mask, dtype=bool) & np.isin(candidate.parts, labels)


def generate_dense_prompts(
    candidates: Sequence[Candidate],
    image_shape: tuple[int, int],
    part_mapping: Mapping[int, int],
    num_parts: int,
    cfg: ATIAConfig = ATIAConfig(),
) -> DensePromptResult:
    """Assign the target and construct instance prompts and body-part masks."""
    height, width = image_shape
    if height <= 0 or width <= 0 or num_parts < 1:
        raise ValueError("Invalid image size or body-part count.")
    if not 0.0 <= cfg.center_weight <= 1.0:
        raise ValueError("center_weight must lie in [0, 1].")
    if cfg.eps <= 0.0:
        raise ValueError("eps must be positive.")

    for original, mapped in part_mapping.items():
        if not isinstance(mapped, (int, np.integer)) or not 0 <= mapped <= num_parts:
            raise ValueError(f"Invalid part mapping for label {original}.")

    for index, candidate in enumerate(candidates):
        if np.asarray(candidate.mask).shape != (height, width):
            raise ValueError(f"Candidate {index} has an invalid mask shape.")
        if np.asarray(candidate.parts).shape != (height, width):
            raise ValueError(f"Candidate {index} has an invalid part-label shape.")

    valid = tuple(
        index
        for index, candidate in enumerate(candidates)
        if candidate.confidence >= cfg.confidence_threshold
        and np.count_nonzero(candidate.mask) / (height * width) >= cfg.area_ratio_threshold
    )

    prompt = np.zeros((2, height, width), dtype=np.float32)
    body_parts = np.zeros((num_parts + 1, height, width), dtype=np.float32)
    body_parts[0] = 1.0

    if not valid:
        return DensePromptResult(
            prompt, body_parts, None, valid, False, None, {}, "no_valid_candidate"
        )

    scores: dict[int, float] = {}
    margin: float | None = None
    confident = True
    status = "single_candidate"

    if len(valid) == 1:
        target_index = valid[0]
    else:
        if not cfg.upper_body_labels:
            raise ValueError("upper_body_labels must be set for target assignment.")

        center = _region_mask(height, width, cfg.center_width_fraction)
        upper_center = _region_mask(
            height,
            width,
            cfg.upper_center_width_fraction,
            cfg.upper_center_height_fraction,
        )

        for index in valid:
            candidate = candidates[index]
            mask = np.asarray(candidate.mask, dtype=bool)
            upper = _upper_body_mask(candidate, cfg.upper_body_labels)
            center_score = np.count_nonzero(mask & center) / (mask.sum() + cfg.eps)
            upper_score = np.count_nonzero(upper & upper_center) / (
                upper.sum() + cfg.eps
            )
            scores[index] = float(
                cfg.center_weight * center_score
                + (1.0 - cfg.center_weight) * upper_score
            )

        ranked = sorted(valid, key=lambda index: (-scores[index], index))
        target_index = ranked[0]
        margin = scores[ranked[0]] - scores[ranked[1]]
        confident = margin >= cfg.margin_threshold
        status = "assigned" if confident else "ambiguous_zero_prompt"

    target = np.asarray(candidates[target_index].mask, dtype=bool)
    labels = np.asarray(candidates[target_index].parts)

    for original, mapped in part_mapping.items():
        if mapped == 0:
            continue
        body_parts[mapped] = np.maximum(
            body_parts[mapped], ((labels == original) & target).astype(np.float32)
        )
    body_parts[0] = 1.0 - np.max(body_parts[1:], axis=0)

    if confident:
        negative = np.zeros((height, width), dtype=bool)
        for index in valid:
            if index != target_index:
                negative |= np.asarray(candidates[index].mask, dtype=bool)
        negative &= ~target
        prompt[0] = target.astype(np.float32)
        prompt[1] = negative.astype(np.float32)

    return DensePromptResult(
        prompt,
        body_parts,
        target_index,
        valid,
        confident,
        margin,
        scores,
        status,
    )
