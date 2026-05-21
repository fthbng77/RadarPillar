<div align="center">

# RadarPillars: Reproduction on View-of-Delft

**Radar-only 3D object detection — OpenPCDet-based reproduction of [Musiat et al., IROS 2024](https://arxiv.org/abs/2408.05020)**

</div>

**RadarPillars** reproduction for **4D mmWave radar** 3D object detection on the **View-of-Delft (VoD)** dataset, built on **OpenPCDet**. This repo reproduces and **beats the published RadarPillars paper by +1.86 mAP_3D** (52.56 vs 50.70, R11) using radar point clouds only — no camera, no LiDAR. **Pretrained weights included** (Git LFS) so you can evaluate without retraining. Keywords: View-of-Delft, VoD, 4D radar, automotive radar perception, autonomous driving, PointPillars, 3D detection.

---

## Headline

| Method | Car | Ped | Cyc | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **Ours — best seed** | **41.6** | **44.8** | 71.3 | **52.56** |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| **RadarPillars (paper)** | 41.1 | 38.6 | 72.6 | **50.70** |
| CenterPoint baseline | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars baseline | 37.9 | 31.2 | 65.7 | 45.0 |

Best checkpoint (mAP_3D 52.56, seed s3 @ epoch 60): [`weights/radarpillar_vod_best_map52.56.pth`](weights/radarpillar_vod_best_map52.56.pth) — tracked via [Git LFS](https://git-lfs.github.com/). After `git clone`, run `git lfs pull` to fetch it.
Full ablation, per-seed logs, hyperparameter tables → [`experiments/RESULTS.md`](experiments/RESULTS.md).

---

## Architecture

```
Radar pcd (N,7)
  → PillarVFE (voxelize + Doppler decomp: vx, vy via atan2)
  → PillarAttention (masked self-attention, C=E=32)
  → PointPillarScatter (320×320×32 BEV)
  → BaseBEVBackbone (3-block 2D CNN, uniform C=32)
  → AnchorHeadSingle (Car / Pedestrian / Cyclist)
```

Key implementation details:
- **Velocity decomposition** in VFE: `vx = v_r_comp·cos(φ)`, `vy = v_r_comp·sin(φ)`, `φ = atan2(y, x)`
- **Physics-consistent augmentation**: velocity vectors rotated/flipped with point coordinates (fixes a bug in OpenPCDet that assumed nuScenes column layout)
- **PillarAttention** with key-padding mask so empty pillars don't poison attention scores
- **`FFN_CHANNELS` config-driven** in `pillar_attention.py` (was hardcoded `*2` before)

---

## Demo

Qualitative results on View-of-Delft validation frames using the **v1.0 checkpoint** (mAP_3D 52.56). Left: ground truth (solid). Right: GT + model predictions (dashed, with confidence). Radar points are colored by RCS.

<p align="center">
  <img src="docs/visualizations/bev_00373.png" width="100%" alt="BEV GT vs predictions, VoD sample 00373"><br>
  <img src="docs/visualizations/bev_00360.png" width="100%" alt="BEV GT vs predictions, VoD sample 00360">
</p>

Reproduce these from a checkpoint — run inference:

```bash
# Inference → writes result.pkl under output/.../eval/
python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt weights/radarpillar_vod_best_map52.56.pth
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

Requirements: Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6.

---

## Data

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

Generate info pkl + GT db:
```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

---

## Train

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <run_name> --workers 4
```

3-seed multi-run (matches the headline number):
```bash
bash experiments/chain_scripts/multiseed_v2.sh
```

---

## Eval

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt weights/radarpillar_vod_best_map52.56.pth
```

---

## Configs

| File | Purpose |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | paper-faithful baseline (no rotation) |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **rotation-augmented variant — produced the headline result** |

---

## Citation

```bibtex
@inproceedings{musiat2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Musiat, Alexander and Reichardt, Laurenz and Schulze, Michael and Wasenm{\"u}ller, Oliver},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  url    = {https://github.com/open-mmlab/OpenPCDet}
}
```

---

## License

Released under the Apache 2.0 License — see [LICENSE](LICENSE). This project is built on top of [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), which is itself Apache 2.0 licensed.
