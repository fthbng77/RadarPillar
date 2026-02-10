"""
Verify proposed anchor sizes for the Cyclist class in VoD radar dataset.
Computes axis-aligned BEV IoU between cyclist GT boxes and anchor sets.
"""

import pickle
import numpy as np
import os

# ---------- Configuration ----------
PKL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "VoD", "view_of_delft_PUBLIC", "radar_5frames", "vod_infos_train.pkl"
)

# Names that map to 'Cyclist'
CYCLIST_NAMES = {
    'bicycle', 'rider', 'Cyclist', 'moped_scooter',
    'motor', 'ride_other', 'ride_uncertain'
}

# Anchor sets: each anchor is [dx, dy, dz]
CURRENT_ANCHORS = [[0.85, 0.60, 1.20], [1.95, 0.80, 1.60]]
PROPOSED_ANCHORS = [[0.82, 0.76, 1.54], [1.89, 0.68, 1.38]]

# Anchor rotations
ANCHOR_ROTATIONS = [0, np.pi / 2]

# Small/large cyclist threshold on dx
SMALL_LARGE_THRESH = 1.2  # meters


def compute_axis_aligned_bev_iou(gt_dx, gt_dy, gt_heading, anchor_l, anchor_w):
    """
    Compute axis-aligned BEV IoU between a GT box and an anchor.

    For the GT box, we align based on heading:
      - If |heading| closer to 0 or pi: aligned_l = dx, aligned_w = dy
      - If |heading| closer to pi/2:    aligned_l = dy, aligned_w = dx

    Both boxes are centered at origin.
    """
    # Normalize heading to [0, pi)
    h = np.abs(gt_heading) % np.pi

    # Determine if heading is closer to 0/pi or pi/2
    if h <= np.pi / 4 or h >= 3 * np.pi / 4:
        # Closer to 0 or pi
        aligned_l = gt_dx
        aligned_w = gt_dy
    else:
        # Closer to pi/2
        aligned_l = gt_dy
        aligned_w = gt_dx

    intersection = min(aligned_l, anchor_l) * min(aligned_w, anchor_w)
    union = aligned_l * aligned_w + anchor_l * anchor_w - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def compute_max_iou_for_gt(gt_dx, gt_dy, gt_heading, anchors):
    """
    For a single GT box, compute the max BEV IoU across all anchors
    and both anchor rotations (0 and pi/2).
    """
    max_iou = 0.0
    for anchor in anchors:
        al, aw, ah = anchor
        for rot in ANCHOR_ROTATIONS:
            if rot == 0:
                a_l, a_w = al, aw
            else:
                # pi/2 rotation swaps l and w
                a_l, a_w = aw, al
            iou = compute_axis_aligned_bev_iou(gt_dx, gt_dy, gt_heading, a_l, a_w)
            if iou > max_iou:
                max_iou = iou
        # end rotations
    return max_iou


def print_statistics(ious, label):
    """Print IoU statistics for a set of IoU values."""
    ious = np.array(ious)
    n = len(ious)
    if n == 0:
        print(f"  [{label}] No samples.")
        return

    pct_ge_05 = np.mean(ious >= 0.5) * 100
    pct_ge_035 = np.mean(ious >= 0.35) * 100
    mean_iou = np.mean(ious)
    median_iou = np.median(ious)

    p10 = np.percentile(ious, 10)
    p25 = np.percentile(ious, 25)
    p50 = np.percentile(ious, 50)
    p75 = np.percentile(ious, 75)
    p90 = np.percentile(ious, 90)

    print(f"  [{label}]  N={n}")
    print(f"    IoU >= 0.50 : {pct_ge_05:6.2f}%")
    print(f"    IoU >= 0.35 : {pct_ge_035:6.2f}%")
    print(f"    Mean IoU    : {mean_iou:.4f}")
    print(f"    Median IoU  : {median_iou:.4f}")
    print(f"    Percentiles : p10={p10:.4f}  p25={p25:.4f}  p50={p50:.4f}  p75={p75:.4f}  p90={p90:.4f}")


def main():
    print("=" * 70)
    print("Anchor Verification for Cyclist Class - VoD Radar Dataset")
    print("=" * 70)

    # Load pickle
    print(f"\nLoading: {PKL_PATH}")
    with open(PKL_PATH, 'rb') as f:
        infos = pickle.load(f)
    print(f"Total frames in train set: {len(infos)}")

    # Extract cyclist GT boxes
    cyclist_boxes = []      # (dx, dy, dz, heading)
    cyclist_names_found = []
    name_counts = {}

    for info in infos:
        annos = info.get('annos', None)
        if annos is None:
            continue
        names = annos.get('name', [])
        gt_boxes = annos.get('gt_boxes_lidar', None)
        if gt_boxes is None:
            continue

        for i, name in enumerate(names):
            if name in CYCLIST_NAMES:
                if i < len(gt_boxes):
                    box = gt_boxes[i]
                    # box format: [x, y, z, dx, dy, dz, heading]
                    dx, dy, dz, heading = box[3], box[4], box[5], box[6]
                    cyclist_boxes.append((dx, dy, dz, heading))
                    cyclist_names_found.append(name)
                    name_counts[name] = name_counts.get(name, 0) + 1

    print(f"\nTotal cyclist GT boxes found: {len(cyclist_boxes)}")
    print("Breakdown by original annotation name:")
    for name, count in sorted(name_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:20s}: {count}")

    if len(cyclist_boxes) == 0:
        print("No cyclist boxes found. Exiting.")
        return

    # Show GT box statistics
    dxs = np.array([b[0] for b in cyclist_boxes])
    dys = np.array([b[1] for b in cyclist_boxes])
    dzs = np.array([b[2] for b in cyclist_boxes])
    headings = np.array([b[3] for b in cyclist_boxes])

    print("\nGT Box Size Statistics (dx, dy, dz):")
    print(f"  dx (length): mean={np.mean(dxs):.3f}  std={np.std(dxs):.3f}  "
          f"min={np.min(dxs):.3f}  max={np.max(dxs):.3f}  median={np.median(dxs):.3f}")
    print(f"  dy (width) : mean={np.mean(dys):.3f}  std={np.std(dys):.3f}  "
          f"min={np.min(dys):.3f}  max={np.max(dys):.3f}  median={np.median(dys):.3f}")
    print(f"  dz (height): mean={np.mean(dzs):.3f}  std={np.std(dzs):.3f}  "
          f"min={np.min(dzs):.3f}  max={np.max(dzs):.3f}  median={np.median(dzs):.3f}")

    n_small = np.sum(dxs < SMALL_LARGE_THRESH)
    n_large = np.sum(dxs >= SMALL_LARGE_THRESH)
    print(f"\nSmall cyclists (dx < {SMALL_LARGE_THRESH}m): {n_small}")
    print(f"Large cyclists (dx >= {SMALL_LARGE_THRESH}m): {n_large}")

    # Compute IoU for each anchor set
    current_ious = []
    proposed_ious = []

    for (dx, dy, dz, heading) in cyclist_boxes:
        iou_current = compute_max_iou_for_gt(dx, dy, heading, CURRENT_ANCHORS)
        iou_proposed = compute_max_iou_for_gt(dx, dy, heading, PROPOSED_ANCHORS)
        current_ious.append(iou_current)
        proposed_ious.append(iou_proposed)

    current_ious = np.array(current_ious)
    proposed_ious = np.array(proposed_ious)

    # ---------- Overall Statistics ----------
    print("\n" + "=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)
    print("\n--- Current Anchors: [[0.85, 0.60, 1.20], [1.95, 0.80, 1.60]] ---")
    print_statistics(current_ious, "Current - All")

    print("\n--- Proposed Anchors: [[0.82, 0.76, 1.54], [1.89, 0.68, 1.38]] ---")
    print_statistics(proposed_ious, "Proposed - All")

    # ---------- Small vs Large ----------
    small_mask = dxs < SMALL_LARGE_THRESH
    large_mask = dxs >= SMALL_LARGE_THRESH

    print("\n" + "=" * 70)
    print(f"SMALL CYCLISTS (dx < {SMALL_LARGE_THRESH}m)")
    print("=" * 70)
    print("\n--- Current Anchors ---")
    print_statistics(current_ious[small_mask], "Current - Small")
    print("\n--- Proposed Anchors ---")
    print_statistics(proposed_ious[small_mask], "Proposed - Small")

    print("\n" + "=" * 70)
    print(f"LARGE CYCLISTS (dx >= {SMALL_LARGE_THRESH}m)")
    print("=" * 70)
    print("\n--- Current Anchors ---")
    print_statistics(current_ious[large_mask], "Current - Large")
    print("\n--- Proposed Anchors ---")
    print_statistics(proposed_ious[large_mask], "Proposed - Large")

    # ---------- Comparison Summary ----------
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    delta_mean = np.mean(proposed_ious) - np.mean(current_ious)
    delta_median = np.median(proposed_ious) - np.median(current_ious)
    delta_pct50 = (np.mean(proposed_ious >= 0.5) - np.mean(current_ious >= 0.5)) * 100
    delta_pct35 = (np.mean(proposed_ious >= 0.35) - np.mean(current_ious >= 0.35)) * 100

    print(f"  Mean IoU change    : {delta_mean:+.4f}  ({'better' if delta_mean > 0 else 'worse'})")
    print(f"  Median IoU change  : {delta_median:+.4f}  ({'better' if delta_median > 0 else 'worse'})")
    print(f"  IoU>=0.50 change   : {delta_pct50:+.2f}%")
    print(f"  IoU>=0.35 change   : {delta_pct35:+.2f}%")

    # Per-box comparison
    improved = np.sum(proposed_ious > current_ious)
    degraded = np.sum(proposed_ious < current_ious)
    same = np.sum(proposed_ious == current_ious)
    print(f"\n  Per-box: {improved} improved, {degraded} degraded, {same} unchanged")

    # Show worst cases for proposed
    print("\n--- Worst 10 IoU values (Proposed Anchors) ---")
    worst_indices = np.argsort(proposed_ious)[:10]
    for idx in worst_indices:
        dx, dy, dz, heading = cyclist_boxes[idx]
        print(f"  IoU={proposed_ious[idx]:.4f}  dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} "
              f"heading={heading:.3f}  name={cyclist_names_found[idx]}")

    print("\n--- Worst 10 IoU values (Current Anchors) ---")
    worst_indices = np.argsort(current_ious)[:10]
    for idx in worst_indices:
        dx, dy, dz, heading = cyclist_boxes[idx]
        print(f"  IoU={current_ious[idx]:.4f}  dx={dx:.3f} dy={dy:.3f} dz={dz:.3f} "
              f"heading={heading:.3f}  name={cyclist_names_found[idx]}")


if __name__ == '__main__':
    main()
