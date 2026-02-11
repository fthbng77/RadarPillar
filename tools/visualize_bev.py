"""
BEV (Bird's Eye View) visualization for RadarPillar predictions.
Draws radar point cloud, GT boxes and predicted boxes on a top-down view.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path
import argparse


CLASS_COLORS_GT = {'Car': '#2ecc71', 'Pedestrian': '#3498db', 'Cyclist': '#e74c3c',
                   'bicycle': '#e74c3c', 'rider': '#e74c3c', 'motor': '#e74c3c',
                   'moped_scooter': '#e74c3c', 'ride_other': '#e74c3c',
                   'ride_uncertain': '#e74c3c', 'bicycle_rack': '#95a5a6'}
CLASS_COLORS_PRED = {'Car': '#27ae60', 'Pedestrian': '#2980b9', 'Cyclist': '#c0392b'}
CLASS_MAPPING = {'bicycle': 'Cyclist', 'rider': 'Cyclist', 'motor': 'Cyclist',
                 'moped_scooter': 'Cyclist', 'ride_other': 'Cyclist',
                 'ride_uncertain': 'Cyclist'}


def kitti_cam_to_lidar(x_cam, y_cam, z_cam):
    """Convert KITTI camera coords to LiDAR/radar coords (x=forward, y=left)."""
    return z_cam, -x_cam, -y_cam


def parse_label_line(line, is_pred=False):
    """Parse a KITTI-format label line."""
    parts = line.strip().split()
    cls = parts[0]
    h, w, l = float(parts[8]), float(parts[9]), float(parts[10])
    x_cam, y_cam, z_cam = float(parts[11]), float(parts[12]), float(parts[13])
    ry = float(parts[14])
    score = float(parts[15]) if is_pred else 1.0

    x_lid, y_lid, _ = kitti_cam_to_lidar(x_cam, y_cam, z_cam)
    heading = -(ry + np.pi / 2)

    mapped_cls = CLASS_MAPPING.get(cls, cls)
    return {'class': mapped_cls, 'orig_class': cls, 'x': x_lid, 'y': y_lid,
            'l': l, 'w': w, 'h': h, 'heading': heading, 'score': score}


def get_box_corners(obj):
    """Get 4 BEV corners of a rotated box."""
    cx, cy, l, w, heading = obj['x'], obj['y'], obj['l'], obj['w'], obj['heading']
    cos_h, sin_h = np.cos(heading), np.sin(heading)
    corners = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated = corners @ rot.T + np.array([cx, cy])
    return rotated


def draw_box(ax, obj, color, linestyle='-', linewidth=1.5, alpha=1.0, label_text=None):
    """Draw a rotated BEV bounding box."""
    corners = get_box_corners(obj)
    polygon = plt.Polygon(corners, fill=False, edgecolor=color,
                          linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(polygon)
    # Draw heading direction
    cx, cy = obj['x'], obj['y']
    dx = np.cos(obj['heading']) * obj['l'] * 0.5
    dy = np.sin(obj['heading']) * obj['l'] * 0.5
    ax.plot([cx, cx + dx], [cy, cy + dy], color=color, linewidth=1, alpha=alpha)
    if label_text:
        ax.text(cx, cy + obj['w']/2 + 0.5, label_text, fontsize=6,
                color=color, ha='center', va='bottom', alpha=alpha)


def visualize_bev(sample_id, data_root, pred_dir, output_dir, score_thresh=0.1,
                  xlim=(0, 52), ylim=(-26, 26)):
    """Generate BEV visualization for a single sample."""
    data_root = Path(data_root)
    pred_dir = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load radar point cloud
    pc_file = data_root / 'training' / 'velodyne' / f'{sample_id}.bin'
    points = np.fromfile(str(pc_file), dtype=np.float32).reshape(-1, 7)

    # Load GT labels
    gt_file = data_root / 'training' / 'label_2' / f'{sample_id}.txt'
    gt_objects = []
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 15:
                continue
            cls = parts[0]
            if cls in ('DontCare',):
                continue
            obj = parse_label_line(line, is_pred=False)
            if obj['class'] in ('Car', 'Pedestrian', 'Cyclist'):
                gt_objects.append(obj)

    # Load predictions
    pred_file = pred_dir / f'{sample_id}.txt'
    pred_objects = []
    if pred_file.exists():
        with open(pred_file, 'r') as f:
            for line in f:
                obj = parse_label_line(line, is_pred=True)
                if obj['score'] >= score_thresh:
                    pred_objects.append(obj)

    # Create figure with 2 subplots: GT only (left), GT + Pred (right)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    for ax, title, show_pred in [(ax1, 'Ground Truth', False),
                                  (ax2, 'GT + Predictions (Epoch 34)', True)]:
        # Plot radar points colored by RCS
        rcs = points[:, 3]
        scatter = ax.scatter(points[:, 0], points[:, 1], c=rcs, cmap='viridis',
                            s=3, alpha=0.6, zorder=1)

        # Draw GT boxes (solid)
        for obj in gt_objects:
            color = CLASS_COLORS_GT.get(obj['class'], '#95a5a6')
            draw_box(ax, obj, color, linestyle='-', linewidth=2, alpha=0.9)

        # Draw Pred boxes (dashed) if applicable
        if show_pred:
            for obj in pred_objects:
                color = CLASS_COLORS_PRED.get(obj['class'], '#7f8c8d')
                lbl = f"{obj['score']:.2f}"
                draw_box(ax, obj, color, linestyle='--', linewidth=1.5, alpha=0.7,
                         label_text=lbl)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.set_xlabel('X (forward) [m]')
        ax.set_ylabel('Y (left) [m]')
        ax.set_title(f'{title} - Sample {sample_id}')
        ax.grid(True, alpha=0.3)

        # Ego vehicle marker
        ax.plot(0, 0, marker='^', color='white', markersize=10,
                markeredgecolor='black', zorder=5)

    # Legend
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
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description='BEV Visualization for RadarPillar')
    parser.add_argument('--data_root', type=str,
                        default='data/VoD/view_of_delft_PUBLIC/radar_5frames')
    parser.add_argument('--pred_dir', type=str, required=True,
                        help='Path to prediction txt files')
    parser.add_argument('--output_dir', type=str, default='output_bev')
    parser.add_argument('--samples', nargs='+', default=['00315'],
                        help='Sample IDs to visualize')
    parser.add_argument('--score_thresh', type=float, default=0.1)
    args = parser.parse_args()

    for sid in args.samples:
        visualize_bev(sid, args.data_root, args.pred_dir, args.output_dir,
                      score_thresh=args.score_thresh)


if __name__ == '__main__':
    main()
