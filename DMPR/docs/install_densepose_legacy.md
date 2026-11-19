# DensePose Environment (Legacy Caffe2)

Source: [facebookresearch/DensePose](https://github.com/facebookresearch/DensePose)

The legacy DensePose repository uses the original Detectron/Caffe2 stack with Python 2 and a Linux NVIDIA GPU environment. Keep this environment separate from the KPR environment.

## 1. Create the Environment

```bash
conda create -n densepose-legacy python=2.7 -y
conda activate densepose-legacy
```

Install the CUDA-enabled Caffe2 build, Detectron operators, CUDA/cuDNN dependencies, and COCO API required by the source repository. Follow its [installation instructions](https://github.com/facebookresearch/DensePose/blob/main/INSTALL.md) for the corresponding historical software versions.

## 2. Check Caffe2

```bash
python2 -c 'from caffe2.python import core'
python2 -c 'from caffe2.python import workspace; print(workspace.NumCudaDevices())'
```

## 3. Install DensePose

```bash
git clone https://github.com/facebookresearch/DensePose.git
cd DensePose
python2 -m pip install -r requirements.txt
make
make ops
```

For model configuration, weights, and inference commands, see the repository's [Getting Started guide](https://github.com/facebookresearch/DensePose/blob/main/GETTING_STARTED.md).

## 4. Export Preprocessed Data

Save the visible human masks, body-part labels, and confidence scores for each pedestrian crop. Map ROI-level predictions to the corresponding crop coordinates before using them in DMPR.

After preprocessing, activate the KPR environment:

```bash
conda deactivate
conda activate kpr
```

The Detectron2-based DensePose setup is documented separately in [install_densepose.md](install_densepose.md).
