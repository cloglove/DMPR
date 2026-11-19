# KPR / DMPR Environment

- [KPR repository](https://github.com/VlSomers/keypoint_promptable_reidentification)
- [KPR installation instructions](https://github.com/VlSomers/keypoint_promptable_reidentification#installation-instructions)

The following setup uses the KPR Linux/CUDA environment with Python 3.10, PyTorch 1.13.0, Torchvision 0.14.0, and CUDA 11.7.

## 1. Clone KPR

```bash
git clone https://github.com/VlSomers/keypoint_promptable_reidentification.git
cd keypoint_promptable_reidentification
```

## 2. Create the Environment

```bash
conda create --name kpr python=3.10 pytorch==1.13.0 torchvision==0.14.0 pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda activate kpr
```

## 3. Install KPR Dependencies

```bash
python -m pip install -r requirements.txt
python setup.py develop
```

## 4. Install DMPR Dependencies

From the DMPR repository root:

```bash
cd /path/to/DMPR
python -m pip install -r requirements-core.txt
```

## 5. Prepare DMPR Inputs

Load the stored DensePose outputs for each pedestrian image and construct a `Candidate` for every detected human instance. Input fields are:

- `mask`: aligned visible-region binary mask `(H, W)`.
- `parts`: aligned body-part labels `(H, W)`.
- `confidence`: instance detection score.

Use `generate_dense_prompts` to obtain:

- Target/Negative Prompt tensor `(2, H, W)`.
- Body-part Target Masks `(K+1, H, W)`.

During KPR's occlusion augmentation, call `mask_aware_bipo` with the resulting prompt tensor, body-part masks, and the aligned occluder mask. The updated tensors are passed to KPR's prompt input and part-based supervision paths.

## 6. Deactivate the Environment

```bash
conda deactivate
```
