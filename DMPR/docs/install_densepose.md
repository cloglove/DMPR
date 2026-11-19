# DensePose Environment (Detectron2)

## Repositories

- [DensePose](https://github.com/facebookresearch/DensePose)
- [DensePose in Detectron2](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose)
- [Detectron2 installation](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md)

This setup uses the Detectron2 implementation of DensePose for offline human-instance and body-part prediction.

## 1. Create the Environment

```bash
conda create -n densepose python=3.10 -y
conda activate densepose
```

Install PyTorch and Torchvision builds compatible with the GPU driver and CUDA toolkit, following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## 2. Install Detectron2 and DensePose

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install -e .
python -m pip install -e projects/DensePose
```

Detectron2 extensions are built against the installed PyTorch and CUDA toolchain. Use compatible compiler, CUDA toolkit, and GPU architecture settings.

## 3. Export DensePose Predictions

From the Detectron2 repository root:

```bash
cd projects/DensePose
python apply_net.py dump \
  configs/densepose_rcnn_R_50_FPN_s1x.yaml \
  https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl \
  /path/to/pedestrian_images \
  --output /path/to/preprocessed/densepose_dump.pkl
```

The dump contains detection boxes, instance confidence scores, and DensePose predictions. See the [DensePose apply_net documentation](https://github.com/facebookresearch/detectron2/blob/main/projects/DensePose/doc/TOOL_APPLY_NET.md) for output fields and command options.

## 4. Prepare Instance Masks

Convert the predictions for each image into the following fields:

| Field | Shape / type | Description |
| --- | --- | --- |
| `mask` | `(H, W)` | Visible human-region binary mask |
| `parts` | `(H, W)` | Pixel-level body-part label map |
| `confidence` | `float` | Instance detection confidence |

For IUV predictions, map the ROI-level body-part labels into the pedestrian crop with the corresponding `pred_boxes_XYXY` coordinates. Preserve discrete labels using nearest-neighbor interpolation.

Store all detected instances for a crop under the corresponding image identifier, including their aligned masks and confidence scores.

## 5. Activate the KPR Environment

```bash
conda deactivate
conda activate kpr
```

The KPR/DMPR process reads the stored masks and labels without invoking DensePose at training time.
