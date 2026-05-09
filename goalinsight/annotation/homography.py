"""Homography and PnP helpers for manual annotations."""

import cv2
import numpy as np


def line_intersection(
    line1: tuple[tuple[float, float], tuple[float, float]],
    line2: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[float, float] | None:
    """Intersection of two lines, or None if parallel."""
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def compute_homography_with_pnp(
    pixel_pts: list[tuple[float, float]],
    world_pts: list[tuple[float, float]],
    img_size: tuple[int, int] = (1920, 1080),
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Compute image->world homography via PnP (coplanar Z=0)."""
    if len(pixel_pts) < 4:
        return None, None, float("inf")

    fx = fy = float(img_size[0])
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    world_3d = np.array([[x, y, 0] for x, y in world_pts], dtype=np.float64)
    pixel_2d = np.array(pixel_pts, dtype=np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            world_3d, pixel_2d, K, None, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None, None, float("inf")

    if not success:
        return None, None, float("inf")

    projected_px, _ = cv2.projectPoints(world_3d, rvec, tvec, K, None)
    pixel_errors = np.linalg.norm(projected_px.reshape(-1, 2) - pixel_2d, axis=1)

    R, _ = cv2.Rodrigues(rvec)
    H_world_to_image = K @ np.column_stack([R[:, 0], R[:, 1], tvec.flatten()])
    H_world_to_image = H_world_to_image / H_world_to_image[2, 2]

    try:
        H_image_to_world = np.linalg.inv(H_world_to_image)
    except np.linalg.LinAlgError:
        return None, None, float("inf")

    src_pts = np.array(pixel_pts, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src_pts, H_image_to_world.astype(np.float32)).reshape(-1, 2)
    dst_pts = np.array(world_pts, dtype=np.float32)
    mean_error = float(np.mean(np.linalg.norm(projected - dst_pts, axis=1)))

    mask = (pixel_errors < 30.0).astype(np.uint8)
    return H_image_to_world.astype(np.float32), mask, mean_error


def compute_homography_ls(
    pixel_pts: list[tuple[float, float]],
    world_pts: list[tuple[float, float]],
) -> tuple[np.ndarray | None, np.ndarray | None, float]:
    """Compute image->world homography via OLS (no outlier rejection)."""
    if len(pixel_pts) < 4:
        return None, None, float("inf")

    src_pts = np.array(pixel_pts, dtype=np.float32)
    dst_pts = np.array(world_pts, dtype=np.float32)

    H, mask = cv2.findHomography(src_pts, dst_pts, 0)
    if H is None:
        return None, None, float("inf")

    projected = cv2.perspectiveTransform(src_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
    mean_error = float(np.mean(np.linalg.norm(projected - dst_pts, axis=1)))

    if mask is None:
        mask = np.ones(len(src_pts), dtype=np.uint8)

    return H, mask, mean_error


def project_world_to_image(
    world_2d: tuple[float, float],
    H_image_to_world: np.ndarray,
) -> tuple[float, float] | None:
    """Project a ground-plane (z=0) world point to image coordinates."""
    try:
        H_inv = np.linalg.inv(H_image_to_world)
    except np.linalg.LinAlgError:
        return None

    world_pt = np.array([[world_2d[0], world_2d[1]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(
        world_pt.reshape(-1, 1, 2), H_inv.astype(np.float32)
    )
    return tuple(projected[0, 0])


def project_3d_with_pnp(
    pt_3d: tuple[float, float, float],
    pixel_pts: list[tuple[float, float]],
    world_pts: list[tuple[float, float]],
    img_size: tuple[int, int],
) -> tuple[float, float] | None:
    """Project a 3D point (z != 0 allowed) using PnP on z=0 correspondences."""
    if len(pixel_pts) < 4:
        return None

    fx = fy = float(img_size[0])
    cx, cy = img_size[0] / 2.0, img_size[1] / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    world_3d = np.array([[x, y, 0] for x, y in world_pts], dtype=np.float64)
    pixel_2d = np.array(pixel_pts, dtype=np.float64)

    try:
        success, rvec, tvec = cv2.solvePnP(
            world_3d, pixel_2d, K, None, flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None

    if not success:
        return None

    pt_3d_arr = np.array([[pt_3d[0], pt_3d[1], pt_3d[2]]], dtype=np.float64)
    projected, _ = cv2.projectPoints(pt_3d_arr, rvec, tvec, K, None)
    return tuple(projected[0, 0])
