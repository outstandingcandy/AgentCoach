"""BroadTrack-style curve utilities for arc-length parameterized line constraints.

Implements the core math from BroadTrack's camera model and curve optimization:
- Rodrigues rotation (matching Ceres AngleAxisRotatePoint)
- BroadTrack projection: world -> centered image coordinates
- Arc-length parameterized polyline interpolation
- Ray casting from image to ground plane
"""

from __future__ import annotations

import numpy as np


def angle_axis_rotate(angle_axis: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Rotate a 3D point by an angle-axis vector (Rodrigues formula).

    Matches Ceres AngleAxisRotatePoint behavior.

    Args:
        angle_axis: (3,) rotation vector (direction = axis, magnitude = angle).
        point: (3,) or (N, 3) points to rotate.

    Returns:
        Rotated point(s), same shape as input.
    """
    theta2 = np.dot(angle_axis, angle_axis)
    single = point.ndim == 1
    pts = point.reshape(-1, 3)

    if theta2 > np.finfo(float).eps:
        theta = np.sqrt(theta2)
        k = angle_axis / theta  # unit axis
        # Rodrigues: v' = v*cos(t) + (k x v)*sin(t) + k*(k.v)*(1-cos(t))
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        k_cross_v = np.cross(k, pts)
        k_dot_v = pts @ k
        result = pts * cos_t + k_cross_v * sin_t + np.outer(k_dot_v, k) * (1 - cos_t)
    else:
        # Small angle: v' ≈ v + angle_axis x v
        result = pts + np.cross(angle_axis, pts)

    return result[0] if single else result


def angle_axis_to_rotation_matrix(angle_axis: np.ndarray) -> np.ndarray:
    """Convert angle-axis to 3x3 rotation matrix."""
    import cv2
    R, _ = cv2.Rodrigues(angle_axis.reshape(3, 1))
    return R


def broadtrack_project(
    angle_axis: np.ndarray,
    position: np.ndarray,
    f: float,
    k1: float,
    world_points: np.ndarray,
    k2: float = 0.0,
) -> np.ndarray:
    """BroadTrack projection: world points -> centered image coordinates.

    Extended from BroadTrack to support k1+k2 distortion for wide-angle cameras:
        P_cam = R @ (P_world - position)
        p_norm = P_cam[:2] / P_cam[2]
        distortion = 1 + k1 * r² + k2 * r⁴
        p_image = f * distortion * p_norm

    Args:
        angle_axis: (3,) rotation vector.
        position: (3,) camera position in world coordinates.
        f: Focal length in pixels.
        k1: First radial distortion coefficient.
        world_points: (N, 3) world coordinates.
        k2: Second radial distortion coefficient.

    Returns:
        (N, 2) centered image coordinates.
    """
    pts = world_points.reshape(-1, 3)
    # Translate to camera-centered coords, then rotate
    translated = pts - position.reshape(1, 3)
    cam_pts = angle_axis_rotate(angle_axis, translated)

    # Perspective division
    z = cam_pts[:, 2:3]
    z = np.where(np.abs(z) < 1e-10, 1e-10, z)
    p_norm = cam_pts[:, :2] / z

    # Radial distortion: 1 + k1*r² + k2*r⁴
    r2 = np.sum(p_norm ** 2, axis=1, keepdims=True)
    distortion = 1.0 + k1 * r2 + k2 * (r2 * r2)

    # Apply focal length and distortion
    p_image = f * distortion * p_norm
    return p_image


def broadtrack_project_single(
    angle_axis: np.ndarray,
    position: np.ndarray,
    f: float,
    k1: float,
    world_point: np.ndarray,
    k2: float = 0.0,
) -> np.ndarray:
    """Project a single world point. Returns (2,) centered image coords."""
    return broadtrack_project(angle_axis, position, f, k1, world_point.reshape(1, 3), k2=k2)[0]


def compute_cumulated_lengths(polyline_3d: np.ndarray) -> np.ndarray:
    """Compute cumulated segment lengths along a 3D polyline.

    Args:
        polyline_3d: (M, 3) polyline vertices.

    Returns:
        (M,) cumulated lengths starting from 0.
    """
    pts = np.asarray(polyline_3d)
    diffs = np.diff(pts, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def interpolate_on_polyline(
    polyline_3d: np.ndarray,
    cumulated_lengths: np.ndarray,
    t: float,
) -> np.ndarray:
    """Interpolate a 3D point on a polyline at arc-length parameter t.

    Matches BroadTrack CurvePointReprojectionError::getWorldPoint().

    Args:
        polyline_3d: (M, 3) polyline vertices.
        cumulated_lengths: (M,) cumulated lengths from compute_cumulated_lengths.
        t: Arc-length parameter in [0, total_length].

    Returns:
        (3,) interpolated 3D point.
    """
    t = np.clip(t, cumulated_lengths[0], cumulated_lengths[-1])

    # Binary search for the segment containing t
    idx = np.searchsorted(cumulated_lengths, t, side='right') - 1
    idx = np.clip(idx, 0, len(polyline_3d) - 2)

    seg_start = cumulated_lengths[idx]
    seg_end = cumulated_lengths[idx + 1]
    seg_len = seg_end - seg_start

    if seg_len < 1e-12:
        return polyline_3d[idx].copy()

    alpha = (t - seg_start) / seg_len
    return (1.0 - alpha) * polyline_3d[idx] + alpha * polyline_3d[idx + 1]


def find_closest_arc_param(
    world_pt: np.ndarray,
    polyline_3d: np.ndarray,
    cumulated_lengths: np.ndarray,
) -> float:
    """Find the arc-length parameter of the closest point on a polyline.

    Matches BroadTrack getCurveParameter().

    Args:
        world_pt: (3,) query point.
        polyline_3d: (M, 3) polyline vertices.
        cumulated_lengths: (M,) cumulated lengths.

    Returns:
        Arc-length parameter t.
    """
    min_dist2 = np.inf
    best_t = 0.0

    for i in range(len(polyline_3d) - 1):
        a = polyline_3d[i]
        b = polyline_3d[i + 1]
        ab = b - a
        ap = world_pt - a
        ab_len2 = np.dot(ab, ab)

        if ab_len2 < 1e-12:
            t_seg = 0.0
        else:
            t_seg = np.clip(np.dot(ap, ab) / ab_len2, 0.0, 1.0)

        closest = a + t_seg * ab
        dist2 = np.sum((world_pt - closest) ** 2)

        if dist2 < min_dist2:
            min_dist2 = dist2
            best_t = cumulated_lengths[i] + t_seg * (cumulated_lengths[i + 1] - cumulated_lengths[i])

    return best_t


def ray_cast_to_ground(
    pt_2d_centered: np.ndarray,
    f: float,
    k1: float,
    angle_axis: np.ndarray,
    position: np.ndarray,
) -> np.ndarray | None:
    """Cast a ray from a centered image point to the ground plane (z=0).

    Matches BroadTrack Camera::getRay() + Surface::intersection().

    Args:
        pt_2d_centered: (2,) centered image coordinates (u - cx, v - cy).
        f: Focal length.
        k1: Radial distortion coefficient.
        angle_axis: (3,) rotation vector.
        position: (3,) camera position in world.

    Returns:
        (3,) world point on ground plane, or None if ray doesn't hit ground.
    """
    # Normalized image coordinates
    p_norm = pt_2d_centered / f

    # Approximate undistortion (first-order inverse)
    r2 = np.dot(p_norm, p_norm)
    p_undist = p_norm / (1.0 + k1 * r2)

    # Build ray direction in camera frame, then rotate to world
    ray_cam = np.array([p_undist[0], p_undist[1], 1.0])
    R = angle_axis_to_rotation_matrix(angle_axis)
    # Camera orientation: R rotates world->camera, so R.T rotates camera->world
    ray_world = R.T @ ray_cam

    # Intersect with z=0 plane
    if abs(ray_world[2]) < 1e-10:
        return None

    t = -position[2] / ray_world[2]
    if t < 0:
        return None

    return position + t * ray_world


def sample_points_on_image_line(
    x1: float, y1: float, x2: float, y2: float, n_points: int = 20,
) -> np.ndarray:
    """Sample N evenly spaced points along an image line segment.

    Args:
        x1, y1, x2, y2: Line endpoints in image coordinates.
        n_points: Number of points to sample.

    Returns:
        (N, 2) sampled image points.
    """
    t = np.linspace(0, 1, n_points)
    xs = x1 + t * (x2 - x1)
    ys = y1 + t * (y2 - y1)
    return np.column_stack([xs, ys])
