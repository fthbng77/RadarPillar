import numpy as np

from ...utils import common_utils


def random_flip_along_x(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    Note: Point velocity handling is gated by gt_boxes having velocity columns (>7).
          For radar data (gt_boxes=7 cols, points=[x,y,z,rcs,v_r,v_r_comp,time]),
          indices 5-6 are NOT Cartesian velocity, so velocity flip must be skipped.
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
            if points.shape[1] >= 7:
                points[:, 6] = -points[:, 6]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    Note: Point velocity handling is gated by gt_boxes having velocity columns (>7).
          For radar data (gt_boxes=7 cols, points=[x,y,z,rcs,v_r,v_r_comp,time]),
          indices 5-6 are NOT Cartesian velocity, so velocity flip must be skipped.
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
            if points.shape[1] >= 7:
                points[:, 5] = -points[:, 5]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    Note: rotate_points_along_z already only rotates x,y,z (indices 0:3) and
          preserves all other features. The additional velocity rotation below
          is only needed when gt_boxes carry velocity columns (shape > 7).
          For radar data (gt_boxes=7 cols), points[:, 5:7] are [v_r_comp, time],
          NOT [vx, vy], so the velocity rotation must be skipped.
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
        if points.shape[1] >= 7:
            velocity = np.hstack((points[:, 5:7], np.zeros((points.shape[0], 1), dtype=points.dtype)))
            velocity = common_utils.rotate_points_along_z(velocity[np.newaxis, :, :], np.array([noise_rotation]))[0]
            points[:, 5:7] = velocity[:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale
    return gt_boxes, points
