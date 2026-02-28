<div align="center">

# RadarPillars: Efficient Object Detection from 4D Radar Point Clouds

**OpenPCDet-based implementation for View-of-Delft (VoD) & Astyx datasets**

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

> **This work is currently under review.**
> Pre-trained model weights and full reproduction details will be released upon paper acceptance.
> Please do not use or redistribute without written permission from the authors.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Key Contributions](#key-contributions)
- [Results](#results)
  - [SOTA Comparison (VoD)](#sota-comparison-on-vod)
  - [Ablation Studies](#ablation-studies)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training & Evaluation](#training--evaluation)
- [Visualization Tools](#visualization-tools)
- [Changelog](#changelog)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

---

## Overview

This repository implements the **RadarPillars** architecture ([Gillen et al., IROS 2024](https://arxiv.org/abs/2408.05020)) for **radar-only 3D object detection**. Built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), it removes LiDAR/image dependencies and adds radar-specific physics features including Doppler velocity decomposition and RCS normalization.

**Supported Datasets:**
| Dataset | Classes | Radar Features | Frames |
|---|---|---|---|
| **View-of-Delft (VoD)** | Car, Pedestrian, Cyclist | x, y, z, RCS, v_r, v_r_comp, time | 5-frame accumulation |
| **Astyx HiRes2019** | Car, Pedestrian | x, y, z, RCS, v_r, v_x, v_y | Single frame |

---

## Architecture

<div align="center">

```
Input: Radar Point Cloud (N, 7)
         │
         ▼
┌─────────────────────┐
│  PillarVFE          │  Voxelization + Velocity Decomposition
│  vr → vx, vy        │  φ = atan2(y, x), vx = vr·cos(φ), vy = vr·sin(φ)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  PillarAttention     │  Global Self-Attention (C=32, H=1)
│  + LayerNorm + FFN   │  with key padding mask for sparse radar
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  PointPillarScatter  │  Sparse-to-Dense BEV projection
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  BaseBEVBackbone     │  Multi-scale 2D CNN (3 layers, 32 channels)
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  AnchorHeadSingle    │  Anchor-based detection + Direction classifier
└─────────┬───────────┘
          ▼
    3D Bounding Boxes
```

</div>

<p align="center">
  <img src="docs/model_framework.png" width="80%" alt="OpenPCDet Framework">
  <br><em>OpenPCDet modular framework architecture</em>
</p>

---

## Key Contributions

### 1. Doppler Velocity Decomposition

Radar measures only **radial velocity** (v_r). We decompose it into Cartesian components in the VFE layer for directional awareness:

```
φ = atan2(y, x + 1e-6)
vx = v_r_comp · cos(φ)
vy = v_r_comp · sin(φ)
```

### 2. Physics-Consistent Augmentation

Fixed a critical bug in `augmentor_utils.py` where `random_flip` and `global_rotation` were incorrectly transforming time values instead of velocity vectors. Velocity is a physical vector and must be rotated/flipped alongside point coordinates.

### 3. PillarAttention for Sparse Radar

Masked multi-head self-attention that handles the inherent sparsity of radar point clouds via key padding masks, preventing empty pillar regions from corrupting attention scores.

### 4. Dual Cyclist Anchor Strategy

VoD's Cyclist class contains diverse sub-types (bicycle, rider, motor, moped). A dual-anchor approach captures both small (bicycle) and large (motorcycle) vehicles separately.

---

## Results

### SOTA Comparison on VoD

**Entire Annotated Area (EAA)** — 3D AP (%) at IoU: Car=0.50, Ped/Cyc=0.25

| Rank | Method | Year | Car | Ped | Cyc | mAP |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | MAFF-Net | 2025 RA-L | 42.3 | **46.8** | **74.7** | **54.6** |
| 2 | SCKD | 2025 AAAI | 41.89 | 43.51 | 70.83 | 52.08 |
| 3 | RadarGaussianDet3D | 2025 | 40.7 | 42.4 | 73.0 | 52.0 |
| 5 | SMURF | 2023 TIV | **42.31** | 39.09 | 71.50 | 50.97 |
| 6 | RadarPillars (paper) | 2024 IROS | 41.1 | 38.6 | 72.6 | 50.70 |
| **10** | **Ours (default, e58)** | **--** | **36.29** | **41.09** | **68.90** | **48.76** |
| **11** | **Ours (vel. decomp, e56)** | **--** | **35.43** | **39.96** | **70.76** | **48.72** |
| 12 | CenterPoint (baseline) | -- | 33.87 | 39.01 | 66.85 | 46.58 |
| 13 | PointPillars (baseline) | -- | 37.92 | 31.24 | 65.66 | 44.94 |

### Our Results vs. Paper

| Configuration | Car | Ped | Cyc | mAP |
|---|:---:|:---:|:---:|:---:|
| RadarPillars paper (5-frame) | **41.1** | 38.6 | **72.6** | **50.7** |
| Ours — default (e58) | 36.29 | **41.09** (+2.5) | 68.90 | 48.76 |
| Ours — vel. decomp (e56) | 35.43 | 39.96 (+1.4) | 70.76 | 48.72 |

**Key observations:**
- Pedestrian detection **exceeds** the paper by +1.4 to +2.5 AP
- Velocity decomposition boosts Cyclist AP significantly: 68.90 → **70.76** (+1.86)
- Overall mAP gap is **-1.9** from the original paper
- Cyclist detection shows the largest gap (-1.8 to -3.7 AP)

### 3D AP Evolution (Epoch 30-40)

<p align="center">
  <img src="docs/visualizations/3d_ap_evolution_2peakcyclist.png" width="60%" alt="3D AP Evolution">
  <br><em>Training is highly stable: Cyclist AP stays in 19.5-20.4 range with minimal oscillation</em>
</p>

---

### Ablation Studies

#### Augmentor Bug Fix

| Experiment | Config | Car 3D | Ped 3D | Cyclist 3D |
|---|---|:---:|:---:|:---:|
| boxq_v7 (flip off, buggy) | Single anchor, NMS=0.05 | 38.58 | 0.60 | 0.00 |
| **return_v5** (flip on, bug fixed) | Single anchor, NMS=0.01 | 35.35 | **31.99** | **17.65** |

> Pedestrian: 0.60 → 32.00 | Cyclist: 0.00 → 17.65

#### Dual Cyclist Anchor

| Experiment | Car 3D | Ped 3D | Cyclist 3D | Weighted Mean |
|---|:---:|:---:|:---:|:---:|
| return_v5_epoch80 (single anchor) | 34.31 | 34.32 | 18.08 | 26.20 |
| **2peakcyclist** (dual anchor) | 33.60 | **35.99** | **20.30** | **27.67** |

> Cyclist: 18.08 → 20.30 (+2.22 AP, +12.3%) | Recall@0.3: 0.40 → 0.47

#### Velocity Normalization Analysis

The `v_r_comp` value is decomposed into vx, vy in the VFE layer. Normalization scales these via `(value - μ) / σ`. However, the config's mean/std values didn't match the actual data distribution:

| Parameter | Config (old) | Actual Data | Ratio |
|---|:---:|:---:|:---:|
| vx std | 0.891 | **1.847** | 0.48x |
| vy std | 0.453 | **0.944** | 0.48x |

Since config std was half the actual std, normalization was **amplifying** the distribution instead of compressing it.

<p align="center">
  <img src="docs/visualizations/velocity_norm_comparison.png" width="90%" alt="Velocity Normalization Comparison">
  <br><em>Config normalization increases outliers (5.8%) compared to raw (4.4%). Correct normalization reduces them to 2.3%</em>
</p>

<p align="center">
  <img src="docs/visualizations/velocity_norm_2d_comparison.png" width="90%" alt="2D Velocity Distribution">
  <br><em>Top: vy histogram. Bottom: vx-vy heatmap (log-scale), cyan dashed circle = 3σ boundary</em>
</p>

| | σ (std) | Outlier ratio (\|v\|>3) |
|---|:---:|:---:|
| Raw | 2.075 | 4.1% |
| Config Norm (σ=0.89) | 2.328 | **4.7% (increased)** |
| Correct Norm (σ=2.08) | 1.000 | **1.4% (decreased)** |

#### Normalization ON vs OFF Training Results

Even with incorrect std values, normalization ON outperforms OFF (vx/vy balance effect):

| Class | Norm ON (e128) | Norm OFF (e128_tek_norm) | Diff |
|---|:---:|:---:|:---:|
| Car | 35.27 | 35.00 | +0.27 |
| Ped | 34.27 | 33.17 | **+1.10** |
| Cyc | 19.48 | 18.76 | **+0.72** |

---

## Installation

**Requirements:** Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# Install OpenPCDet with CUDA extensions
python setup.py develop

# Install WandB for experiment tracking (optional)
pip install wandb
```

See [docs/INSTALL.md](docs/INSTALL.md) for detailed instructions.

---

## Dataset Preparation

### View-of-Delft (VoD)

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── training/
│   ├── velodyne/          # Radar point clouds (.bin)
│   ├── label_2/           # 3D annotations
│   ├── calib/             # Calibration files
│   └── image_2/           # Camera images (optional)
└── testing/
    └── velodyne/
```

```bash
# Generate info files and GT database
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

### Astyx HiRes2019

```
data/astyx/
├── ImageSets/
│   ├── train.txt
│   ├── val.txt
│   └── test.txt
├── training/
│   └── radar/             # Radar point clouds (.bin)
└── testing/
```

```bash
python -m pcdet.datasets.astyx.astyx_dataset create_astyx_infos \
    tools/cfgs/dataset_configs/astyx_dataset_radar.yaml
```

---

## Training & Evaluation

### VoD Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16

# With WandB experiment tracking
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --batch_size 16 --use_wandb
```

### Astyx Training

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --cfg_file tools/cfgs/astyx_models/astyx_radarpillar.yaml \
    --batch_size 4
```

### Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
    --cfg_file tools/cfgs/vod_models/vod_radarpillar.yaml \
    --ckpt <checkpoint_path>
```

### Key Hyperparameters

| Parameter | VoD | Astyx |
|---|---|---|
| Voxel Size | 0.16 x 0.16 x 5.0 m | 0.2 x 0.2 x 4.0 m |
| Max Points/Voxel | 16 | 32 |
| Epochs | 60 | 160 |
| Learning Rate | 0.01 | 0.003 |
| Optimizer | adam_onecycle | adam_onecycle |
| Early Stopping | 30 epoch patience | -- |
| NMS Threshold | 0.1 | 0.01 |

---

## Visualization Tools

### BEV (Bird's Eye View) Visualization

Visualize model predictions overlaid on radar point clouds. GT boxes are solid lines, predictions are dashed. Points are colored by RCS value.

```bash
python tools/visualize_bev.py \
    --pred_dir output/cfgs/vod_models/vod_radarpillar/<exp>/eval/epoch_<N>/val/default/final_result/data \
    --samples 00315 00107 \
    --score_thresh 0.15 \
    --output_dir output_bev
```

<p align="center">
  <img src="docs/visualizations/bev_00315.png" width="90%" alt="BEV Sample 00315">
  <br><em>Sample 00315 — Dense urban scene (cars + cyclists + pedestrians)</em>
</p>

<p align="center">
  <img src="docs/visualizations/bev_00107.png" width="90%" alt="BEV Sample 00107">
  <br><em>Sample 00107 — Close-range cyclist cluster</em>
</p>

### Anchor Verification

Analyze dataset object size distributions and verify anchor box alignment.

```bash
python tools/visualize_anchors.py    # Dimension scatter plot with anchors
python tools/plot_cyclist_dist.py    # Cyclist length histogram
```

<p align="center">
  <img src="docs/visualizations/anchor_verification.png" width="80%" alt="Anchor Verification">
  <br><em>Black cross = Baseline anchor (1.59m, centered on data). Blue diamond = Master anchor (1.94m, shifted from center)</em>
</p>

<p align="center">
  <img src="docs/visualizations/cyclist_dist.png" width="50%" alt="Cyclist Distribution">
  <br><em>Bimodal cyclist distribution: stationary bicycles vs. moving riders</em>
</p>

### AP Evolution Plots

```bash
python visualize_radar_logs.py \
    --logs output/cfgs/vod_models/vod_radarpillar/<exp>/eval/epoch_*/val/default/log_eval_*.txt \
    --output output_plots
```

### Velocity Normalization Analysis

```bash
python tools/generate_velocity_norm_plots.py
```

---

## Changelog

| Date | Description |
|---|---|
| 2026-02 | Velocity decomposition: vr_comp → vx, vy in VFE layer |
| 2026-02 | Dual Cyclist anchor strategy for diverse sub-types |
| 2026-02 | Augmentor bug fix: correct velocity index handling in flip/rotation |
| 2026-02 | BEV visualization tool (`tools/visualize_bev.py`) |
| 2026-02 | WandB integration with `--use_wandb` flag |
| 2026-02 | VoD radar pipeline: dataset config, info generation |
| 2026-01 | Astyx radar pipeline: 7-feature point loader, velocity-aware augmentations |

---

## Citation

```bibtex
@inproceedings{gillen2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Gillen, Julius and Bieder, Manuel and Stiller, Christoph},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}
```

```bibtex
@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  howpublished = {\url{https://github.com/open-mmlab/OpenPCDet}}
}
```

---

## Acknowledgement

This project is built upon [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), an open-source 3D object detection framework. We thank the OpenPCDet team for the original codebase and supported methods.

---

## License

`OpenPCDet` is released under the [Apache 2.0 license](LICENSE).
