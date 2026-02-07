import numpy as np

from ...utils import common_utils


def _get_valid_point_velocity_indices(points, point_velocity_indices):
    """
    Args:
        points: (M, C)
        point_velocity_indices: [vx_idx, vy_idx] or None

    Returns:
        tuple(vx_idx, vy_idx) or None
    """
    if point_velocity_indices is None:
        return None
    if not isinstance(point_velocity_indices, (list, tuple)) or len(point_velocity_indices) != 2:
        raise ValueError('POINT_VELOCITY_INDICES must be [vx_idx, vy_idx] when provided')

    vx_idx, vy_idx = int(point_velocity_indices[0]), int(point_velocity_indices[1])
    if vx_idx < 0 or vy_idx < 0 or vx_idx >= points.shape[1] or vy_idx >= points.shape[1]:
        return None
    return vx_idx, vy_idx


def random_flip_along_x(gt_boxes, points, point_velocity_indices=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        point_velocity_indices: optional [vx_idx, vy_idx] for point-wise velocity channels
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

        velocity_indices = _get_valid_point_velocity_indices(points, point_velocity_indices)
        if velocity_indices is not None:
            _, vy_idx = velocity_indices
            points[:, vy_idx] = -points[:, vy_idx]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, point_velocity_indices=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        point_velocity_indices: optional [vx_idx, vy_idx] for point-wise velocity channels
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

        velocity_indices = _get_valid_point_velocity_indices(points, point_velocity_indices)
        if velocity_indices is not None:
            vx_idx, _ = velocity_indices
            points[:, vx_idx] = -points[:, vx_idx]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, point_velocity_indices=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
        point_velocity_indices: optional [vx_idx, vy_idx] for point-wise velocity channels
    Returns:
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

    velocity_indices = _get_valid_point_velocity_indices(points, point_velocity_indices)
    if velocity_indices is not None:
        vx_idx, vy_idx = velocity_indices
        velocity = np.hstack((
            points[:, [vx_idx, vy_idx]],
            np.zeros((points.shape[0], 1), dtype=points.dtype)
        ))
        velocity = common_utils.rotate_points_along_z(
            velocity[np.newaxis, :, :], np.array([noise_rotation])
        )[0]
        points[:, vx_idx] = velocity[:, 0]
        points[:, vy_idx] = velocity[:, 1]

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
