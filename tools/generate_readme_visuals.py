"""
Generate updated README visuals from current experiment results.
1. Stable-region AP evolution plot (epoch 48-60) for default experiment
2. Export result.pkl to KITTI-format txt for BEV visualization
3. Generate BEV images from the best epoch predictions
"""
import os
import sys
import re
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / 'data' / 'VoD' / 'view_of_delft_PUBLIC' / 'radar_5frames'
OUTPUT_VIS = ROOT / 'docs' / 'visualizations'

# ── 1. AP Evolution Plot (stable region) ────────────────────────────
def parse_train_log(log_path):
    """Parse a training log for 3D AP R40 moderate values per epoch."""
    epoch_pattern = re.compile(r"\*+ EPOCH (\d+) EVALUATION \*+")
    class_pattern = re.compile(r"(Car|Pedestrian|Cyclist) AP_R40@")
    metric_pattern = re.compile(r"3d\s+AP:([\d\.]+), ([\d\.]+), ([\d\.]+)")

    data = {}
    current_epoch = None
    current_class = None

    with open(log_path, 'r') as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                current_epoch = int(m.group(1))
                current_class = None
                data.setdefault(current_epoch, {})
                continue
            if current_epoch is not None:
                m = class_pattern.search(line)
                if m:
                    current_class = m.group(1)
                    continue
                if current_class is not None:
                    m = metric_pattern.search(line)
                    if m:
                        # moderate = second value
                        data[current_epoch][current_class] = float(m.group(2))
                        current_class = None
    return data


def plot_stable_ap_evolution(out_path):
    """Plot AP evolution for default experiment, epoch 48-60 (stable region)."""
    log_dir = ROOT / 'output' / 'cfgs' / 'vod_models' / 'vod_radarpillar' / 'default'
    logs = sorted(log_dir.glob('log_train_*.txt'))
    if not logs:
        print("No default training log found!")
        return

    data = parse_train_log(str(logs[-1]))

    # Filter to stable epochs
    epochs = sorted([e for e in data.keys() if 48 <= e <= 60])
    if not epochs:
        print("No epochs in 48-60 range found!")
        return

    classes = ['Car', 'Pedestrian', 'Cyclist']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(10, 6))
    for cls, color in zip(classes, colors):
        values = [data[e].get(cls, np.nan) for e in epochs]
        ax.plot(epochs, values, marker='o', color=color, label=cls, linewidth=2)

    ax.set_title('RadarPillar 3D AP (R40) Evolution — Default Experiment', fontsize=13,
                 fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('3D AP (%)', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epochs)

    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


# ── 2. Export result.pkl → KITTI txt ────────────────────────────────
def export_pkl_to_kitti_txt(pkl_path, output_dir):
    """Convert result.pkl to per-frame KITTI-format txt files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    for det in results:
        frame_id = det['frame_id']
        out_file = output_dir / f'{frame_id}.txt'

        lines = []
        n = len(det['name'])
        for i in range(n):
            name = det['name'][i]
            trunc = det['truncated'][i]
            occ = det['occluded'][i]
            alpha = det['alpha'][i]
            bbox = det['bbox'][i]  # 2D bbox
            dims = det['dimensions'][i]  # h, w, l (camera frame)
            loc = det['location'][i]  # x, y, z (camera frame)
            ry = det['rotation_y'][i]
            score = det['score'][i]

            line = (f"{name} {trunc:.2f} {int(occ)} {alpha:.2f} "
                    f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
                    f"{dims[0]:.2f} {dims[1]:.2f} {dims[2]:.2f} "
                    f"{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} "
                    f"{ry:.2f} {score:.4f}")
            lines.append(line)

        with open(out_file, 'w') as f:
            f.write('\n'.join(lines) + '\n' if lines else '')

    print(f"Exported {len(results)} frames to {output_dir}")


# ── 3. BEV Visualization (inline, no txt dependency) ────────────────
# Import BEV functions from visualize_bev.py
sys.path.insert(0, str(ROOT / 'tools'))
from visualize_bev import (parse_label_line, draw_box, get_box_corners,
                            CLASS_COLORS_GT, CLASS_COLORS_PRED, CLASS_MAPPING)
from matplotlib.lines import Line2D


def visualize_bev_from_pkl(sample_id, pkl_path, epoch_num, output_dir,
                            score_thresh=0.15, xlim=(0, 52), ylim=(-26, 26)):
    """Generate BEV vis directly from result.pkl for a given sample."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load radar points
    pc_file = DATA_ROOT / 'training' / 'velodyne' / f'{sample_id}.bin'
    points = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 7)

    # Load GT labels
    gt_file = DATA_ROOT / 'training' / 'label_2' / f'{sample_id}.txt'
    gt_objects = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15 or parts[0] == 'DontCare':
                continue
            obj = parse_label_line(line, is_pred=False)
            if obj['class'] in ('Car', 'Pedestrian', 'Cyclist'):
                gt_objects.append(obj)

    # Load predictions from pkl
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    # Find matching frame
    pred_objects = []
    for det in results:
        if det['frame_id'] == sample_id:
            n = len(det['name'])
            for i in range(n):
                score = det['score'][i]
                if score < score_thresh:
                    continue
                # Reconstruct KITTI-format line for reuse of parse_label_line
                name = det['name'][i]
                trunc = det['truncated'][i]
                occ = int(det['occluded'][i])
                alpha = det['alpha'][i]
                bbox = det['bbox'][i]
                dims = det['dimensions'][i]
                loc = det['location'][i]
                ry = det['rotation_y'][i]
                line = (f"{name} {trunc:.2f} {occ} {alpha:.2f} "
                        f"{bbox[0]:.2f} {bbox[1]:.2f} {bbox[2]:.2f} {bbox[3]:.2f} "
                        f"{dims[0]:.2f} {dims[1]:.2f} {dims[2]:.2f} "
                        f"{loc[0]:.2f} {loc[1]:.2f} {loc[2]:.2f} "
                        f"{ry:.2f} {score:.4f}")
                obj = parse_label_line(line, is_pred=True)
                if obj['class'] in ('Car', 'Pedestrian', 'Cyclist'):
                    pred_objects.append(obj)
            break

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for ax, title, show_pred in [
        (ax1, f'Ground Truth — Sample {sample_id}', False),
        (ax2, f'GT + Predictions (Epoch {epoch_num}) — Sample {sample_id}', True)
    ]:
        rcs = points[:, 3]
        scatter = ax.scatter(points[:, 0], points[:, 1], c=rcs, cmap='viridis',
                             s=3, alpha=0.6, zorder=1)

        for obj in gt_objects:
            color = CLASS_COLORS_GT.get(obj['class'], '#95a5a6')
            draw_box(ax, obj, color, linestyle='-', linewidth=2, alpha=0.9)

        if show_pred:
            for obj in pred_objects:
                color = CLASS_COLORS_PRED.get(obj['class'], '#7f8c8d')
                lbl = f"{obj['score']:.2f}"
                draw_box(ax, obj, color, linestyle='--', linewidth=1.5,
                         alpha=0.7, label_text=lbl)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X (forward) [m]')
        ax.set_ylabel('Y (left) [m]')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.plot(0, 0, marker='^', color='white', markersize=10,
                markeredgecolor='black', zorder=5)

    legend_elements = [
        Line2D([0], [0], color='#2ecc71', linewidth=2, label='GT Car'),
        Line2D([0], [0], color='#3498db', linewidth=2, label='GT Pedestrian'),
        Line2D([0], [0], color='#e74c3c', linewidth=2, label='GT Cyclist'),
        Line2D([0], [0], color='#27ae60', linewidth=2, linestyle='--', label='Pred Car'),
        Line2D([0], [0], color='#2980b9', linewidth=2, linestyle='--', label='Pred Pedestrian'),
        Line2D([0], [0], color='#c0392b', linewidth=2, linestyle='--', label='Pred Cyclist'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.colorbar(scatter, ax=[ax1, ax2], label='RCS [dBsm]', shrink=0.6, pad=0.02)
    plt.tight_layout()

    out_path = output_dir / f'bev_{sample_id}.png'
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    OUTPUT_VIS.mkdir(parents=True, exist_ok=True)

    # 1. AP Evolution (stable region)
    print("=== Generating AP Evolution Plot ===")
    plot_stable_ap_evolution(OUTPUT_VIS / '3d_ap_evolution_default.png')

    # 2. BEV Visualizations from default experiment, epoch 58 (best)
    print("\n=== Generating BEV Visualizations ===")
    pkl_path = (ROOT / 'output' / 'cfgs' / 'vod_models' / 'vod_radarpillar' /
                'default' / 'eval' / 'eval_with_train' / 'epoch_58' / 'val' / 'result.pkl')

    if pkl_path.exists():
        for sample in ['00315', '00107']:
            visualize_bev_from_pkl(sample, str(pkl_path), epoch_num=58,
                                   output_dir=OUTPUT_VIS, score_thresh=0.15)
    else:
        print(f"WARNING: {pkl_path} not found. Skipping BEV generation.")

    print("\nDone.")
