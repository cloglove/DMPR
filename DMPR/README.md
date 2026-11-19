# DMPR

**Dense Mask Promptable Re-Identification for Occluded Person Re-Identification**

DMPR uses DensePose instance masks and body-part labels to generate target-aware dense prompts for the KPR framework. The mask-processing pipeline includes automatic target instance assignment (ATIA), target/negative prompt construction, body-part supervision masks, and mask-aware updates during inter-person occlusion augmentation.

## Project Structure

```text
DMPR/
├── README.md
├── requirements-core.txt
├── dmpr_core/
│   ├── __init__.py
│   ├── masks.py
│   └── augmentation.py
└── docs/
    ├── install_densepose.md
    ├── install_densepose_legacy.md
    └── install_kpr.md
```

## Dependencies

| Project | Role | Repository |
| --- | --- | --- |
| DensePose | Human-instance segmentation and body-part labels | [facebookresearch/DensePose](https://github.com/facebookresearch/DensePose) |
| Detectron2 DensePose | Alternative DensePose implementation | [detectron2/projects/DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) |
| KPR | Image and prompt tokenization, feature encoding, part-based learning, training and evaluation | [VlSomers/keypoint_promptable_reidentification](https://github.com/VlSomers/keypoint_promptable_reidentification) |

## Environment Setup

DensePose preprocessing and KPR training use separate Python environments. Intermediate masks and labels are stored on disk and loaded by the KPR/DMPR pipeline.

**DensePose environment**

- [Detectron2 DensePose installation](docs/install_densepose.md)
- [Legacy DensePose installation](docs/install_densepose_legacy.md)

**KPR / DMPR environment**

- [KPR installation](docs/install_kpr.md)

Install the DMPR mask-processing dependency in the KPR environment:

```bash
python -m pip install -r requirements-core.txt
```

## Data Flow

```text
Pedestrian crops
      |
      v
DensePose preprocessing
      |
      v
Instance masks + body-part labels + confidence scores
      |
      v
ATIA target assignment
      |
      v
Target Prompt + Negative Prompt + Body-part Target Masks
      |
      v
KPR image/prompt encoding and part-based feature learning
```

DensePose runs offline. Its predictions are converted to per-instance masks and part-label maps aligned with each pedestrian crop before being passed to `dmpr_core`.

## Input Format

Each DensePose candidate is represented by `dmpr_core.Candidate`:

| Field | Type / shape | Description |
| --- | --- | --- |
| `mask` | `ndarray`, `(H, W)` | Visible human-instance mask |
| `parts` | `ndarray`, `(H, W)` | DensePose body-part label map |
| `confidence` | `float` | Detection confidence |

`mask` and `parts` must use the same pixel coordinates as the associated pedestrian image. DensePose ROI predictions must be projected back to crop coordinates using their detected bounding boxes. Use nearest-neighbor interpolation for discrete part labels.

## ATIA Configuration

`ATIAConfig` controls candidate filtering and spatial-prior scoring:

| Parameter | Default | Function |
| --- | --- | --- |
| `confidence_threshold` | `0.5` | Minimum candidate confidence |
| `area_ratio_threshold` | `0.01` | Minimum visible-area ratio |
| `center_weight` | `0.7` | Weight of center-region consistency |
| `margin_threshold` | `0.1` | Minimum Top-1/Top-2 score margin |
| `center_width_fraction` | `0.50` | Width of the central image region |
| `upper_center_width_fraction` | `0.60` | Width of the upper-central image region |
| `upper_center_height_fraction` | `0.55` | Height of the upper-central image region |
| `upper_body_labels` | `()` | DensePose labels assigned to upper-body regions |

Assign `upper_body_labels` using the label definitions of the selected DensePose model. For body-part supervision, provide `part_mapping` from DensePose labels to KPR part IDs (`0` denotes background) and set `num_parts` to the number of foreground part classes.

## Dense Prompt Generation

`dmpr_core.generate_dense_prompts(candidates, image_shape, part_mapping, num_parts, cfg)` selects the target instance and returns:

| Result field | Shape / type | Content |
| --- | --- | --- |
| `prompt` | `(2, H, W)` | Target Prompt and Negative Prompt |
| `body_parts` | `(K+1, H, W)` | Background and target body-part supervision channels |
| `target_index` | `int` or `None` | Selected candidate index |
| `valid_indices` | `tuple[int, ...]` | Candidate indices passing filtering |
| `scores` | `dict[int, float]` | Spatial-prior assignment scores |
| `margin` | `float` or `None` | Difference between the highest two scores |
| `confident` | `bool` | Assignment confidence flag |
| `status` | `str` | Assignment status |

Target and negative masks are mutually exclusive. If multiple candidates produce a score margin below `margin_threshold`, both prompt channels are set to zero.

## Mask-aware BIPO

`dmpr_core.mask_aware_bipo(prompt, body_parts, occluder_mask)` updates the masks after KPR's inter-person occlusion composition.

Inputs:

- `prompt`: two-channel target/negative prompt, shape `(2, H, W)`.
- `body_parts`: body-part supervision, shape `(K+1, H, W)`.
- `occluder_mask`: aligned binary occluder region, shape `(H, W)`.

Updates:

1. Remove occluded pixels from the Target Prompt.
2. Add the occluder region to the Negative Prompt while preserving target/negative exclusivity.
3. Remove occluded pixels from foreground body-part channels.
4. Recompute the background channel from the updated body-part masks.

The function returns `(updated_prompt, updated_parts)`.

## Processing Sequence

1. Run DensePose inference on pedestrian crops in the DensePose environment.
2. Export instance masks, pixel-level body-part labels, and confidence scores in crop coordinates.
3. Switch to the KPR environment and load the preprocessed data.
4. Generate target/negative dense prompts and target body-part supervision with ATIA.
5. Update the masks during KPR's inter-person occlusion augmentation.
6. Pass the resulting prompt and supervision tensors through the KPR training and evaluation pipeline.
