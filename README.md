# NightMono

**Improving Nighttime Monocular Depth Estimation via Scene Priors from Foundation Models**

NightMono is a PyTorch implementation for robust monocular depth estimation under challenging illumination, with a particular focus on nighttime driving scenes. It extends the self-supervised [md4all](https://github.com/md4all/md4all) pipeline with texture-aware feature extraction, attention-based feature injection, dense pseudo-depth supervision, and a texture consistency objective.

> **Research-code notice:** this repository currently contains experiment-specific paths and expects locally prepared datasets, pseudo-depth maps, sky masks, and model checkpoints. Please read [Configuration](#configuration) before running training or evaluation.

## Highlights

- **Texture-aware depth network** built on a Monodepth2-style ResNet encoder and depth decoder.
- **Multi-scale and directional texture extraction** using 3x3, 5x5, horizontal, and vertical convolutional branches.
- **Attention-guided feature injection** with channel attention, spatial attention, and residual fusion at shallow and intermediate encoder stages.
- **Scene-prior supervision** using dense pseudo-depth maps, including special handling of sky regions.
- **Texture consistency learning** that aligns texture-feature boundaries with structural cues from teacher depth.
- **Self-supervised geometry losses**, including multi-view photometric reconstruction, edge-aware smoothness, and velocity supervision.
- Support for **Oxford RobotCar** and **nuScenes**, plus inference on individual images or folders.

## Method Overview

NightMono augments the original depth-estimation pipeline with an explicit texture branch:

1. An RGB image is dynamically normalized according to its capture condition.
2. A texture extractor produces multi-scale, direction-aware features.
3. Texture features are injected into selected ResNet encoder stages through channel and spatial attention.
4. A Monodepth2 decoder predicts inverse depth at multiple scales.
5. Training combines self-supervised geometric losses with pseudo-depth and texture-consistency objectives when the required priors are available.

The main implementation is located in:

- `models/depth_net_texture.py` — texture extraction and attention-based feature injection.
- `losses/SupervisedLoss_2.py` — pseudo-depth supervision.
- `losses/texture_consistency_loss.py` — texture/structure consistency objective.
- `losses/TotalLoss.py` — weighted composition of all active losses.
- `trainer.py` — PyTorch Lightning training, validation, testing, and prediction logic.

## Repository Structure

```text
NightMono/
├── config/                 # Training, evaluation, and inference configurations
├── data/
│   ├── nuscenes_dataset.py
│   ├── custom_dataset.py
│   └── robotcar/           # RobotCar loader and preprocessing scripts
├── evaluation/             # Depth metrics and evaluation utilities
├── losses/                 # Photometric, smoothness, prior, and texture losses
├── models/
│   ├── depth_net_texture.py
│   ├── pose_net.py
│   ├── md2/                # Monodepth2 components
│   └── ForkGAN/            # Image-translation components
├── utils/                  # Geometry, camera, pose, image, and depth utilities
├── visualization/          # Training and prediction visualization
├── train.py                # Training entry point
├── evaluate_depth.py       # Dataset evaluation entry point
├── test_simple.py          # Inference on images or folders
├── translate_simple.py     # Day-to-adverse image translation
├── Dockerfile
└── requirements_w_version.txt
```

## Installation

The original development environment uses Python 3, CUDA 11.3, PyTorch 1.13.1, and PyTorch Lightning 1.9.0. An NVIDIA GPU is recommended.

### Conda

```bash
git clone https://github.com/Yang-void/NightMono.git
cd NightMono

conda create -n nightmono python=3.8 -y
conda activate nightmono
pip install -r requirements_w_version.txt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

For packages without fixed versions, use:

```bash
pip install -r requirements.txt
```

### Docker

Before building the image, replace the placeholders in `Makefile`:

- `<USER_ID>:<GROUP_ID>`
- `<PATH_TO_DATAROOT>`
- `<PATH_TO_MD4ALL>`

Then run:

```bash
make docker-build NAME=build
make docker-start-interactive NAME=dev
```

The Docker setup requires the NVIDIA Container Toolkit and a compatible NVIDIA driver.

## Data Preparation

### Oxford RobotCar

The RobotCar pipeline expects images from the left stereo camera and point clouds from the front LMS sensor. Refer to the original [Oxford RobotCar Dataset](https://robotcar-dataset.robots.ox.ac.uk/) and the [md4all data instructions](https://github.com/md4all/md4all#datasets) for the base preparation procedure.

Precompute undistorted RGB images:

```bash
python data/robotcar/precompute_rgb_images.py \
  --dataroot /path/to/robotcar \
  --scenes 2014-12-09-13-21-02 2014-12-16-18-44-24 \
  --camera_sensor stereo/left \
  --out_dir /path/to/robotcar
```

Precompute LiDAR depth for validation and testing:

```bash
python data/robotcar/precompute_depth_gt.py \
  --dataroot /path/to/robotcar \
  --scenes 2014-12-09-13-21-02 2014-12-16-18-44-24 \
  --mode val test
```

For NightMono training, the current RobotCar loader additionally expects the following files for each training frame:

```text
<scene>/disp1/<timestamp>_depth_pred.npy   # dense pseudo/teacher depth
<scene>/sky_mask/<timestamp>.png           # binary sky mask
```

Split files and filtered-sample files must also be provided through `LOAD.SPLIT_FILES_PATH` and `LOAD.FILTER_FILES_PATH` in the YAML configuration.

### nuScenes

Download the nuScenes train/validation data and metadata from the official [nuScenes website](https://www.nuscenes.org/). Set `DATASET.DATAROOT` and `DATASET.VERSION` in the selected configuration file. The repository retains the original md4all nuScenes training and evaluation pipeline.

## Configuration

Experiments are controlled by YAML files under `config/`. Important fields include:

```yaml
SYSTEM:
  ACCELERATOR: 'gpu'
  DEVICES: 1

SAVE:
  CHECKPOINT_PATH: '/path/to/output/checkpoints'
  LOG_DIR: '/path/to/output/logs'

LOAD:
  CHECKPOINT_PATH: '/path/to/model.ckpt'
  SPLIT_FILES_PATH: '/path/to/splits'
  FILTER_FILES_PATH: '/path/to/filter/files'

DATASET:
  DATAROOT: '/path/to/dataset'

LOSS:
  PHOTOMETRIC:
    WEIGHT: 1.0
  PSOSUPERVISED:
    WEIGHT: 0.02
  TEXTURE:
    WEIGHT: 0.001
```

The command line can override configuration values after the YAML path. For example:

```bash
python train.py \
  --config config/train_md4allDDa_robotcar.yaml \
  DATASET.DATAROOT /data/robotcar \
  SYSTEM.DEVICES 1
```

### Important: checkpoint initialization

The current `train.py` initializes the model by combining two checkpoints specified by the `depth_ckpt` and `texture_ckpt` variables. Replace these hard-coded paths with your own files before training. The loader uses `strict=True`, so the combined state dictionary must contain all parameters expected by the model.

Several YAML files also contain machine-specific absolute paths. Replace every `/root/...` or `/mnt/...` path with a valid path on your system.

## Training

Train the NightMono/md4allDDa configuration on RobotCar:

```bash
python train.py --config config/train_md4allDDa_robotcar.yaml
```

Train the corresponding baseline:

```bash
python train.py --config config/train_baseline_robotcar.yaml
```

The nuScenes configurations can be launched similarly:

```bash
python train.py --config config/train_md4allDDa_nuscenes.yaml
python train.py --config config/train_baseline_nuscenes.yaml
```

Training logs are written through TensorBoard:

```bash
tensorboard --logdir /path/to/logs
```

## Evaluation

First set `LOAD.CHECKPOINT_PATH`, dataset paths, and result directories in the selected YAML file. Then run:

```bash
# Oxford RobotCar, 50 m evaluation range
python evaluate_depth.py --config config/eval_md4allDDa_50m_robotcar_test.yaml

# nuScenes, 80 m evaluation range
python evaluate_depth.py --config config/eval_md4allDDa_80m_nuscenes_val.yaml
```

Quantitative metrics are saved as CSV files when `EVALUATION.SAVE.QUANTITATIVE_RES_PATH` is set. Qualitative RGB, predicted-depth, and ground-truth visualizations can be enabled through the `EVALUATION.SAVE` options.

## Inference on Custom Images

Set `LOAD.CHECKPOINT_PATH` in the appropriate `test_simple` configuration, then run inference on one or more images:

```bash
python test_simple.py \
  --config config/test_simple_md4allDDa_robotcar.yaml \
  --image_paths /path/to/image.png \
  --output_path output
```

The script also accepts a directory containing `.jpg`, `.jpeg`, or `.png` files:

```bash
python test_simple.py \
  --config config/test_simple_md4allDDa_robotcar.yaml \
  --image_paths /path/to/image_folder \
  --output_path output
```

If daytime-dependent normalization is enabled, pass one condition per input image through `--daytimes`. If capture conditions are unknown, use image-wise normalization (`DATASET.AUGMENTATION.NORMALIZE.MODE: 'Image'`).

Inference stores raw predictions as NumPy arrays and, when enabled, writes colorized depth visualizations to the output directory.

## Image Translation

The included ForkGAN components can generate day-to-night or day-to-rain images when the corresponding pretrained translation checkpoints are available:

```bash
python translate_simple.py \
  --image_path /path/to/input.png \
  --checkpoint_dir /path/to/forkgan_checkpoint \
  --model_name forkgan_robotcar_day_night \
  --crop_height 768 \
  --crop_width 1280 \
  --resize_height 320 \
  --resize_width 544 \
  --output_dir output
```

## Current Limitations

- Pretrained NightMono checkpoints and generated scene priors are not included in this repository.
- Some configurations contain experiment-specific absolute paths and must be edited before use.
- The dense pseudo-depth and sky-mask loading path is currently implemented for RobotCar training.
- `trainer.py` currently enables the texture-aware depth model unconditionally through `self.enhance_texture = True`.
- The repository is a research snapshot and may require adaptation for a new dataset or environment.

## Acknowledgements

This project builds upon the following open-source research projects:

- [md4all](https://github.com/md4all/md4all) — robust monocular depth estimation under challenging conditions.
- [Monodepth2](https://github.com/nianticlabs/monodepth2) — self-supervised monocular depth estimation components.
- [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm) — training structure and geometry losses used by md4all.
- [ForkGAN](https://github.com/zhengziqiang/ForkGAN) — day-to-adverse image translation components.

Please also cite the original md4all work when using this code:

```bibtex
@inproceedings{gasperini_morbitzer2023md4all,
  title     = {Robust Monocular Depth Estimation under Challenging Conditions},
  author    = {Gasperini, Stefano and Morbitzer, Nils and Jung, HyunJun and Navab, Nassir and Tombari, Federico},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year      = {2023},
  pages     = {8177--8186}
}
```

If NightMono accompanies a paper, add its BibTeX entry here once the bibliographic information is finalized.

## License

This repository includes code derived from md4all and is distributed for non-commercial use under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. See [`LICENSE`](LICENSE) for details. Components under `models/ForkGAN/` retain their accompanying license notice.
