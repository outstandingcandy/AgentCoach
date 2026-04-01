"""Undistortion-aware pitch projection helpers (for physical camera backend)."""

import bisect

import cv2
import numpy as np


# FIFA pitch half-dimensions (meters)
_PITCH_HALF_LENGTH = 52.5
_PITCH_HALF_WIDTH = 34.0


def _undistort_and_project_to_pitch(pts_2d: np.ndarray, pose: dict) -> list[float] | None:
    """Project a single distorted image point to pitch coordinates via undistortion.

    Args:
        pts_2d: (1, 2) or (N, 2) array of distorted pixel coordinates.
        pose: dict with K, dist_coeffs, rvec, tvec from camera_poses.pkl.

    Returns:
        [x_world, y_world] on the ground plane, or None if projection fails.
    """
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)

    # Step 1: Undistort pixel coords -> ideal pixel coords (P=K keeps in pixel space)
    pts = np.array(pts_2d, dtype=np.float64).reshape(-1, 1, 2)
    pts_undist = cv2.undistortPoints(pts, K, dist, P=K)

    # Step 2: Apply H_inv (image->world) on undistorted points
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    pt = pts_undist.reshape(-1, 2)[0]
    ph = H_inv @ np.array([pt[0], pt[1], 1.0])
    if abs(ph[2]) > 1e-6:
        return [float(ph[0] / ph[2]), float(ph[1] / ph[2])]
    return None


def _filter_by_pitch_undistorted(
    detections: list[dict],
    pose: dict,
    margin: float = 5.0,
    use_center: bool = False,
) -> list[dict]:
    """Filter detections by pitch boundary using undistorted projection.

    Args:
        detections: List of detection dicts with 'bbox' key.
        pose: Physical camera pose dict.
        margin: Extra meters beyond pitch boundary to allow.
        use_center: If True, use bbox center instead of foot point (for ball).

    Returns:
        Filtered detection list.
    """
    if not detections:
        return detections

    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return detections

    # Extract projection points from all detections
    proj_pts = []
    for det in detections:
        bbox = det["bbox"]
        px = (bbox[0] + bbox[2]) / 2
        if use_center:
            py = (bbox[1] + bbox[3]) / 2
        else:
            py = bbox[3]  # foot point
        proj_pts.append([px, py])

    proj_pts = np.array(proj_pts, dtype=np.float64).reshape(-1, 1, 2)
    proj_undist = cv2.undistortPoints(proj_pts, K, dist, P=K).reshape(-1, 2)

    # Project to world and filter by pitch boundary
    filtered = []
    x_lim = _PITCH_HALF_LENGTH + margin
    y_lim = _PITCH_HALF_WIDTH + margin

    for i, det in enumerate(detections):
        pt = proj_undist[i]
        ph = H_inv @ np.array([pt[0], pt[1], 1.0])
        if abs(ph[2]) < 1e-6:
            continue
        wx, wy = ph[0] / ph[2], ph[1] / ph[2]
        if -x_lim <= wx <= x_lim and -y_lim <= wy <= y_lim:
            filtered.append(det)

    return filtered


def _interpolate_camera_poses(
    camera_poses: dict[int, dict],
    target_frame_indices: list[int],
) -> dict[int, dict]:
    """Interpolate camera poses for frames not calibrated in stage1.

    Uses linear interpolation on rvec, tvec, and K matrix between the two
    nearest calibrated frames.  Falls back to nearest-neighbor when the
    target frame is before the first or after the last calibrated frame.

    Args:
        camera_poses: Dict of frame_idx -> pose from stage1.
        target_frame_indices: Frame indices needed by stage2.

    Returns:
        Expanded camera_poses dict covering all target frames.
    """
    if not camera_poses:
        return camera_poses

    calibrated_indices = sorted(camera_poses.keys())
    expanded = dict(camera_poses)  # Keep originals

    for fidx in target_frame_indices:
        if fidx in expanded:
            continue

        pos = bisect.bisect_left(calibrated_indices, fidx)

        if pos == 0:
            # Before first calibrated frame -- use nearest
            expanded[fidx] = camera_poses[calibrated_indices[0]]
        elif pos >= len(calibrated_indices):
            # After last calibrated frame -- use nearest
            expanded[fidx] = camera_poses[calibrated_indices[-1]]
        else:
            # Between two calibrated frames -- linear interpolation
            left_idx = calibrated_indices[pos - 1]
            right_idx = calibrated_indices[pos]
            t = (fidx - left_idx) / (right_idx - left_idx)

            left_pose = camera_poses[left_idx]
            right_pose = camera_poses[right_idx]

            interp_pose = {}
            for key in ("rvec", "tvec"):
                lv = np.array(left_pose[key], dtype=np.float64)
                rv = np.array(right_pose[key], dtype=np.float64)
                interp_pose[key] = (lv + t * (rv - lv)).tolist()

            # Interpolate K (focal length changes with Veo digital zoom)
            lK = np.array(left_pose["K"], dtype=np.float64)
            rK = np.array(right_pose["K"], dtype=np.float64)
            interp_pose["K"] = (lK + t * (rK - lK)).tolist()

            # Interpolate dist_coeffs
            ld = np.array(left_pose["dist_coeffs"], dtype=np.float64)
            rd = np.array(right_pose["dist_coeffs"], dtype=np.float64)
            interp_pose["dist_coeffs"] = (ld + t * (rd - ld)).tolist()

            expanded[fidx] = interp_pose

    return expanded
