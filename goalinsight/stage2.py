#!/usr/bin/env python3
"""Stage 2: Tracking and Identification - Player detection, tracking, and attribute recognition.

According to the paper, this stage:
1. Detection: YOLOv8 for players/goalkeepers/referees
2. Tracking: StrongSORT with ReID features
3. Attribute recognition: Role and jersey number (via VLM, optional)
4. Team assignment: ReID feature clustering

Supports configurable backends via factory functions:
- ReID: OSNet (default) or PRTReID
- Team classification: KMeans (default) or Tracklet clustering
"""

import bisect
import json
import pickle
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

from .utils.config import (
    get_default_config,
    get_process_fps_from_config,
    FrameSampler,
    get_reid_extractor,
    get_team_classifier,
)
from .tracking import (
    PlayerDetector,
    StrongSORTTracker,
    GoalkeeperDetector,
    BallDetector,
    BallTracker,
)
from .tracking.ball_trajectory import BallTrajectory3D


# ---------------------------------------------------------------------------
# Undistortion-aware pitch projection helpers (for physical camera backend)
# ---------------------------------------------------------------------------

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

    # Step 1: Undistort pixel coords → ideal pixel coords (P=K keeps in pixel space)
    pts = np.array(pts_2d, dtype=np.float64).reshape(-1, 1, 2)
    pts_undist = cv2.undistortPoints(pts, K, dist, P=K)

    # Step 2: Apply H_inv (image→world) on undistorted points
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


# ---------------------------------------------------------------------------
# Ball trajectory filtering helpers
# ---------------------------------------------------------------------------

def _find_nearest_detection_center(
    frame_idx: int,
    all_ball_detections: dict[int, list[dict]],
    max_gap: int = 30,
) -> tuple[float, float] | None:
    """Find the nearest frame with detections and return the best detection center.

    Searches forward and backward from frame_idx within max_gap frames.

    Returns:
        (cx, cy) of the highest-confidence detection in the nearest frame, or None.
    """
    best_dist = max_gap + 1
    best_center = None

    for fidx, dets in all_ball_detections.items():
        if not dets:
            continue
        dist = abs(fidx - frame_idx)
        if dist < best_dist:
            best_dist = dist
            best_det = max(dets, key=lambda d: d.get("confidence", 0))
            best_center = tuple(best_det["center"])

    return best_center if best_dist <= max_gap else None


def _interpolate_tracked_position(
    frame_idx: int,
    tracked_positions: dict[int, tuple[float, float]],
    max_gap: int = 30,
) -> tuple[float, float] | None:
    """Interpolate ball position from the nearest tracked positions.

    Finds the closest tracked frame before and after *frame_idx* and linearly
    interpolates.  Falls back to the single nearest tracked position if the
    frame is at the start/end of a tracked segment.

    Returns:
        (cx, cy) interpolated pixel position, or None if no tracked frame
        is within *max_gap*.
    """
    before_fidx: int | None = None
    after_fidx: int | None = None

    for fidx in tracked_positions:
        dist = abs(fidx - frame_idx)
        if dist > max_gap:
            continue
        if fidx < frame_idx:
            if before_fidx is None or fidx > before_fidx:
                before_fidx = fidx
        elif fidx > frame_idx:
            if after_fidx is None or fidx < after_fidx:
                after_fidx = fidx

    if before_fidx is not None and after_fidx is not None:
        # Linear interpolation
        t = (frame_idx - before_fidx) / (after_fidx - before_fidx)
        bx, by = tracked_positions[before_fidx]
        ax, ay = tracked_positions[after_fidx]
        return (bx + t * (ax - bx), by + t * (ay - by))

    # Only one side available — use nearest
    nearest = before_fidx if before_fidx is not None else after_fidx
    if nearest is not None:
        return tracked_positions[nearest]

    return None


def _filter_trajectories_field_space(
    trajectory_field_data: dict[int, list[tuple]],
    fps: float,
    config: dict,
) -> dict[int, list[tuple]]:
    """Filter trajectories based on physical constraints in field space.

    Each entry in trajectory_field_data is:
        track_id -> list of (frame_idx, (wx, wy), (px, py), confidence)

    Filters:
    1. Minimum length: discard short tracks
    2. Position bounds: discard if majority of points outside pitch + margin
    3. Speed: discard if too many inter-frame speeds exceed max
    4. Smoothness: discard if direction changes are too erratic

    Returns:
        Filtered dict with invalid trajectories removed.
    """
    max_speed = config.get("max_speed_ms", 40.0)
    margin = config.get("position_margin", 8.0)
    min_len = config.get("min_trajectory_length", 3)
    smooth_thresh = config.get("smoothness_threshold", 0.6)
    speed_viol_ratio = config.get("speed_violation_ratio", 0.2)

    pitch_half_length = _PITCH_HALF_LENGTH + margin
    pitch_half_width = _PITCH_HALF_WIDTH + margin

    filtered = {}

    for tid, points in trajectory_field_data.items():
        # Filter 1: minimum length
        if len(points) < min_len:
            continue

        # Filter 2: position bounds
        in_bounds = sum(
            1 for _, (wx, wy), _, _ in points
            if abs(wx) <= pitch_half_length and abs(wy) <= pitch_half_width
        )
        bounds_ok = in_bounds / len(points) >= 0.5

        # Filter 3: speed constraint
        violation_count = 0
        pair_count = 0
        for i in range(1, len(points)):
            fidx_prev, (wx_prev, wy_prev), _, _ = points[i - 1]
            fidx_curr, (wx_curr, wy_curr), _, _ = points[i]
            dt = (fidx_curr - fidx_prev) / fps if fps > 0 else 0
            if dt < 1e-6:
                continue
            dist = ((wx_curr - wx_prev) ** 2 + (wy_curr - wy_prev) ** 2) ** 0.5
            speed = dist / dt
            pair_count += 1
            if speed > max_speed:
                violation_count += 1

        speed_ok = pair_count == 0 or violation_count / pair_count <= speed_viol_ratio

        # Filter 4: smoothness in field space (direction consistency)
        angles = []
        for i in range(2, len(points)):
            v1x = points[i - 1][1][0] - points[i - 2][1][0]
            v1y = points[i - 1][1][1] - points[i - 2][1][1]
            v2x = points[i][1][0] - points[i - 1][1][0]
            v2y = points[i][1][1] - points[i - 1][1][1]
            mag1 = (v1x ** 2 + v1y ** 2) ** 0.5
            mag2 = (v2x ** 2 + v2y ** 2) ** 0.5
            if mag1 < 0.1 or mag2 < 0.1:
                continue
            cos_angle = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angles.append(cos_angle)

        field_smooth_ok = not angles or sum(angles) / len(angles) >= smooth_thresh

        if bounds_ok and speed_ok and field_smooth_ok:
            filtered[tid] = points
            continue

        # Pixel-space fallback: when the ball is airborne, ground-plane
        # projection is unreliable (positions overshoot, speeds explode).
        # Accept the trajectory if it is smooth and consistent in pixel space.
        px_angles = []
        for i in range(2, len(points)):
            v1x = points[i - 1][2][0] - points[i - 2][2][0]
            v1y = points[i - 1][2][1] - points[i - 2][2][1]
            v2x = points[i][2][0] - points[i - 1][2][0]
            v2y = points[i][2][1] - points[i - 1][2][1]
            mag1 = (v1x ** 2 + v1y ** 2) ** 0.5
            mag2 = (v2x ** 2 + v2y ** 2) ** 0.5
            if mag1 < 1.0 or mag2 < 1.0:
                continue
            cos_a = (v1x * v2x + v1y * v2y) / (mag1 * mag2)
            cos_a = max(-1.0, min(1.0, cos_a))
            px_angles.append(cos_a)

        px_smooth = (not px_angles or
                     sum(px_angles) / len(px_angles) >= smooth_thresh)

        # Also check pixel-space speed is reasonable (not teleporting)
        max_px_speed = 300.0  # pixels per frame at process_fps
        px_speed_ok = True
        for i in range(1, len(points)):
            fidx_prev = points[i - 1][0]
            fidx_curr = points[i][0]
            dt_frames = (fidx_curr - fidx_prev) / max(fps / 10.0, 1.0)
            dx = points[i][2][0] - points[i - 1][2][0]
            dy = points[i][2][1] - points[i - 1][2][1]
            px_dist = (dx ** 2 + dy ** 2) ** 0.5
            if dt_frames > 0 and px_dist / dt_frames > max_px_speed:
                px_speed_ok = False
                break

        if px_smooth and px_speed_ok and len(points) >= min_len:
            filtered[tid] = points

    return filtered


def _remove_merge_outliers(
    all_ball_tracks: dict[int, dict],
    fps: float,
    max_deviation: float = 5.0,
) -> int:
    """Remove spatial outliers from the merged ball track.

    After merging multiple trajectories, some frames may come from a secondary
    trajectory that is spatially inconsistent with the surrounding frames
    (e.g., a fence detection filling a 1-frame gap in the main ball track).

    For each frame, we check deviation from the linearly interpolated position
    between the nearest previous and next frames.  If the deviation exceeds
    *max_deviation* meters **and** the neighbors are mutually consistent
    (their interpolated speed < 30 m/s), the frame is an outlier and removed.

    Returns:
        Number of outlier frames removed.
    """
    sorted_frames = sorted(all_ball_tracks.keys())
    if len(sorted_frames) < 3:
        return 0

    to_remove: list[int] = []

    for i in range(1, len(sorted_frames) - 1):
        f_prev = sorted_frames[i - 1]
        f_curr = sorted_frames[i]
        f_next = sorted_frames[i + 1]

        pp_prev = all_ball_tracks[f_prev].get("pitch_position")
        pp_curr = all_ball_tracks[f_curr].get("pitch_position")
        pp_next = all_ball_tracks[f_next].get("pitch_position")

        if not pp_prev or not pp_curr or not pp_next:
            continue

        # Check that neighbors are mutually consistent (low speed)
        dt_pn = (f_next - f_prev) / fps if fps > 0 else 0
        if dt_pn < 1e-6:
            continue
        dist_pn = ((pp_next[0] - pp_prev[0]) ** 2 + (pp_next[1] - pp_prev[1]) ** 2) ** 0.5
        speed_pn = dist_pn / dt_pn
        if speed_pn > 30.0:
            # Neighbors themselves are inconsistent — can't judge the middle point
            continue

        # Interpolate expected position at f_curr
        t_ratio = (f_curr - f_prev) / (f_next - f_prev)
        interp_x = pp_prev[0] + t_ratio * (pp_next[0] - pp_prev[0])
        interp_y = pp_prev[1] + t_ratio * (pp_next[1] - pp_prev[1])

        deviation = ((pp_curr[0] - interp_x) ** 2 + (pp_curr[1] - interp_y) ** 2) ** 0.5

        if deviation > max_deviation:
            to_remove.append(f_curr)

    for fidx in to_remove:
        del all_ball_tracks[fidx]

    return len(to_remove)


def _select_best_trajectory(
    filtered_trajectories: dict[int, list[tuple]],
) -> int | None:
    """Select the single best trajectory from filtered candidates.

    Scoring: 60% length + 25% mean confidence + 15% coverage density.

    Returns:
        track_id of the best trajectory, or None.
    """
    if not filtered_trajectories:
        return None

    max_len = max(len(pts) for pts in filtered_trajectories.values())

    scores = {}
    for tid, points in filtered_trajectories.items():
        length_score = len(points) / max(max_len, 1)
        conf_score = sum(p[3] for p in points) / len(points)
        span = points[-1][0] - points[0][0] + 1
        density_score = len(points) / max(span, 1)
        scores[tid] = length_score * 0.6 + conf_score * 0.25 + density_score * 0.15

    return max(scores, key=scores.get)


# Ball detection diagnostic visualization
# ---------------------------------------------------------------------------

_BALL_DIAG_COLORS = {
    "pass1": (0, 220, 0),       # green
    "pass2": (255, 180, 0),     # cyan-ish (BGR)
    "rejected": (0, 0, 255),    # red (trajectory rejected by field filter)
}
_BALL_DIAG_LABELS = {
    "pass1": "Pass1",
    "pass2": "Pass2 (crop)",
    "rejected": "Rejected trajectory",
}


def _render_ball_detection_diag(
    video_path: Path | str,
    sampler,
    all_ball_dets_diag: dict[int, list[dict]],
    output_dir: Path,
) -> None:
    """Render per-frame diagnostic images showing ball detection sources.

    Each detection is drawn as a circle with color indicating its source:
      green = pass1, blue = pass2 (crop+enlarge),
      red = rejected trajectory.

    Uses a 3-stage pipeline: reader thread -> annotate -> writer thread pool
    """
    import threading
    import queue as _queue
    from concurrent.futures import ThreadPoolExecutor

    diag_dir = output_dir / "ball_detection_diag"
    diag_dir.mkdir(exist_ok=True)

    frame_indices = list(sampler)

    # Count stats
    counts: dict[str, int] = {}
    for dets in all_ball_dets_diag.values():
        for d in dets:
            src = d.get("source", "pass1")
            counts[src] = counts.get(src, 0) + 1

    print(f"\nGenerating ball detection diagnostic visualization...")
    print(f"  " + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # --- Stage 1: reader thread ---
    read_q: _queue.Queue = _queue.Queue(maxsize=8)

    def _reader() -> None:
        cap = cv2.VideoCapture(str(video_path))
        prev_fidx = -1
        for fidx in frame_indices:
            gap = fidx - prev_fidx - 1
            if prev_fidx >= 0 and 0 <= gap <= 8:
                for _ in range(gap):
                    cap.grab()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            prev_fidx = fidx
            if ret:
                read_q.put((fidx, frame))
        read_q.put(None)
        cap.release()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    # --- Stage 3: writer thread pool (imwrite is I/O-bound) ---
    writer_pool = ThreadPoolExecutor(max_workers=4)
    write_futures = []

    # --- Stage 2: annotate on main thread ---
    for _ in tqdm(frame_indices, desc="  Ball diag"):
        item = read_q.get()
        if item is None:
            break
        fidx, frame = item

        dets = all_ball_dets_diag.get(fidx, [])

        # Draw each detection
        for d in dets:
            src = d.get("source", "pass1")
            color = _BALL_DIAG_COLORS.get(src, (255, 255, 255))
            cx, cy = int(d["center"][0]), int(d["center"][1])
            conf = d["confidence"]
            bbox = d["bbox"]
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            thickness = 2 if src in ("pass1", "pass2") else 1
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness)

            radius = 12 if src in ("pass1", "pass2") else 8
            cv2.circle(frame, (cx, cy), radius, color, 2)
            if src in ("pass1", "pass2"):
                cv2.circle(frame, (cx, cy), 3, color, -1)

            label = f"{_BALL_DIAG_LABELS.get(src, src)} {conf:.2f}"
            label_y = max(by1 - 8, 15)
            cv2.putText(frame, label, (bx1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        n_dets = len(dets)
        sources = [d.get("source", "?") for d in dets]
        info = f"Frame {fidx}  |  {n_dets} det(s): {', '.join(sources) if sources else 'none'}"
        cv2.rectangle(frame, (0, 0), (len(info) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(frame, info, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        legend_y = frame.shape[0] - 80
        cv2.rectangle(frame, (0, legend_y - 5), (220, frame.shape[0]), (0, 0, 0), -1)
        for i, (src, color) in enumerate(_BALL_DIAG_COLORS.items()):
            ly = legend_y + i * 22 + 15
            cv2.circle(frame, (15, ly - 4), 6, color, -1)
            cv2.putText(frame, _BALL_DIAG_LABELS[src], (30, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Submit write to thread pool (JPEG encoding + disk I/O)
        out_path = str(diag_dir / f"frame_{fidx:05d}.jpg")
        write_futures.append(writer_pool.submit(cv2.imwrite, out_path, frame))

    reader.join()
    # Wait for all writes to finish
    for fut in write_futures:
        fut.result()
    writer_pool.shutdown(wait=False)
    print(f"  Saved to {diag_dir}/")


# Camera pose interpolation (for stage2 running at higher fps than stage1)
# ---------------------------------------------------------------------------

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
            # Before first calibrated frame — use nearest
            expanded[fidx] = camera_poses[calibrated_indices[0]]
        elif pos >= len(calibrated_indices):
            # After last calibrated frame — use nearest
            expanded[fidx] = camera_poses[calibrated_indices[-1]]
        else:
            # Between two calibrated frames — linear interpolation
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


# ---------------------------------------------------------------------------
# Jersey color histogram extraction for team classification
# ---------------------------------------------------------------------------

def _extract_jersey_color_hist(crop: np.ndarray, h_bins: int = 30, s_bins: int = 32) -> np.ndarray | None:
    """Extract HSV color histogram from upper body (jersey) region.

    Takes the top 50% of the crop (torso area), converts to HSV, and computes
    a normalized H+S histogram. Ignores V channel for illumination invariance.

    Returns:
        Flattened, L2-normalized histogram vector, or None if crop is too small.
    """
    h, w = crop.shape[:2]
    if h < 10 or w < 5:
        return None

    # Upper 50% = jersey/torso area (skip head ~top 15%)
    y_top = max(0, int(h * 0.15))
    y_bot = int(h * 0.65)
    jersey = crop[y_top:y_bot, :]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

    # 2D histogram on H and S channels
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins],
                        [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist /= norm
    return hist


def _extract_jersey_mean_saturation(crop: np.ndarray) -> float | None:
    """Extract mean saturation of jersey region. Low saturation = achromatic (black/white/grey).

    Returns:
        Mean saturation (0-255), or None if crop is too small.
    """
    h, w = crop.shape[:2]
    if h < 10 or w < 5:
        return None

    y_top = max(0, int(h * 0.15))
    y_bot = int(h * 0.65)
    jersey = crop[y_top:y_bot, :]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


# Team colors for visualization
TEAM_COLORS = {
    "team_A": (0, 0, 255),     # Red
    "team_B": (255, 0, 0),     # Blue
    "referee": (0, 255, 255),  # Yellow
    "unknown": (128, 128, 128),  # Gray
    "ball": (0, 165, 255),     # Orange
}


def get_color_for_track(track_id: int, team: str | None = None) -> tuple[int, int, int]:
    """Get color for a track based on team assignment."""
    if team and team in TEAM_COLORS:
        return TEAM_COLORS[team]
    # Fallback to ID-based color
    import random
    random.seed(track_id * 7)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_topdown_pitch(
    height: int,
    tracks: list[dict],
    team_assignments: dict[int, str],
    ball_track: dict | None = None,
    ball_trajectory_world: list[list[float]] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> np.ndarray:
    """Draw top-down pitch diagram with player and ball positions.

    Args:
        height: Output image height (matches camera frame height).
        tracks: List of track dicts with pitch_position, track_id, role.
        team_assignments: Dict of track_id -> team label.
        ball_track: Ball track dict with pitch_position (optional).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        Top-down pitch image (BGR).
    """
    width = int(height * 0.75)  # 3:4 aspect for pitch
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)

    half_l = pitch_length / 2
    half_w = pitch_width / 2
    margin = 6
    scale = min(
        (width - 2) / (pitch_length + 2 * margin),
        (height - 2) / (pitch_width + 2 * margin),
    )
    ox = width / 2
    oy = height / 2

    def w2p(wx, wy):
        px = int(ox + wx * scale)
        py = int(oy - wy * scale)
        return (px, py)

    lc = (255, 255, 255)
    lw = max(1, int(scale * 0.25))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Pitch markings
    cv2.rectangle(pitch, w2p(-half_l, half_w), w2p(half_l, -half_w), lc, lw)
    cv2.line(pitch, w2p(0, half_w), w2p(0, -half_w), lc, lw)
    cv2.circle(pitch, w2p(0, 0), int(9.15 * scale), lc, lw)
    cv2.circle(pitch, w2p(0, 0), max(2, int(0.3 * scale)), lc, -1)
    # Penalty areas
    pa_w = min(20.16, half_w)
    pa_d = min(16.5, half_l * 0.35)
    cv2.rectangle(pitch, w2p(-half_l, pa_w), w2p(-half_l + pa_d, -pa_w), lc, lw)
    cv2.rectangle(pitch, w2p(half_l - pa_d, pa_w), w2p(half_l, -pa_w), lc, lw)
    # Goal areas
    ga_w = min(9.16, half_w * 0.5)
    ga_d = min(5.5, half_l * 0.12)
    cv2.rectangle(pitch, w2p(-half_l, ga_w), w2p(-half_l + ga_d, -ga_w), lc, lw)
    cv2.rectangle(pitch, w2p(half_l - ga_d, ga_w), w2p(half_l, -ga_w), lc, lw)

    # Role-specific markers
    ROLE_SHAPES = {
        "goalkeeper": "diamond",
        "referee": "triangle",
        "player": "circle",
    }

    r = max(5, int(1.2 * scale))
    for track in tracks:
        pos = track.get("pitch_position")
        if pos is None:
            continue

        tid = track["track_id"]
        team = team_assignments.get(tid, track.get("team", "unknown"))
        role = track.get("role", "player")
        color = get_color_for_track(tid, team)

        px, py = w2p(pos[0], pos[1])
        if not (-50 < px < width + 50 and -50 < py < height + 50):
            continue

        shape = ROLE_SHAPES.get(role, "circle")
        if shape == "diamond":
            pts = np.array([
                [px, py - r], [px + r, py], [px, py + r], [px - r, py]
            ], dtype=np.int32)
            cv2.fillPoly(pitch, [pts], color)
            cv2.polylines(pitch, [pts], True, (255, 255, 255), 1)
        elif shape == "triangle":
            pts = np.array([
                [px, py - r], [px + r, py + r], [px - r, py + r]
            ], dtype=np.int32)
            cv2.fillPoly(pitch, [pts], color)
            cv2.polylines(pitch, [pts], True, (255, 255, 255), 1)
        else:
            cv2.circle(pitch, (px, py), r, color, -1)
            cv2.circle(pitch, (px, py), r, (255, 255, 255), 1)

        # Label
        label = f"{tid}"
        if role == "goalkeeper":
            label = f"GK{tid}"
        cv2.putText(pitch, label, (px + r + 2, py + 3), font, 0.35, (255, 255, 255), 1)

    # Ball trajectory trail on pitch
    if ball_trajectory_world and len(ball_trajectory_world) > 1:
        n = len(ball_trajectory_world)
        for i in range(1, n):
            alpha = i / n
            color = (0, int(100 + 65 * alpha), int(180 + 75 * alpha))
            thickness = max(1, int(2 * alpha))
            p1 = w2p(ball_trajectory_world[i - 1][0], ball_trajectory_world[i - 1][1])
            p2 = w2p(ball_trajectory_world[i][0], ball_trajectory_world[i][1])
            cv2.line(pitch, p1, p2, color, thickness)

    # Ball current position
    if ball_track and ball_track.get("pitch_position"):
        bx, by = ball_track["pitch_position"]
        bpx, bpy = w2p(bx, by)
        ball_r = max(6, int(1.0 * scale))
        cv2.circle(pitch, (bpx, bpy), ball_r, TEAM_COLORS["ball"], -1)
        cv2.circle(pitch, (bpx, bpy), ball_r, (255, 255, 255), 2)

    # Legend
    y_leg = 20
    for label, col in [("team_A", TEAM_COLORS["team_A"]),
                        ("team_B", TEAM_COLORS["team_B"]),
                        ("referee", TEAM_COLORS["referee"]),
                        ("GK", TEAM_COLORS["team_A"])]:
        cv2.rectangle(pitch, (5, y_leg - 10), (15, y_leg), col, -1)
        cv2.putText(pitch, label, (18, y_leg), font, 0.35, (255, 255, 255), 1)
        y_leg += 18

    return pitch


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict],
    track_history: dict,
    team_assignments: dict[int, str],
) -> np.ndarray:
    """Draw tracked bounding boxes and trajectories."""
    vis = frame.copy()

    for track in tracks:
        track_id = track["track_id"]
        bbox = track["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        team = team_assignments.get(track_id, track.get("team", "unknown"))
        color = get_color_for_track(track_id, team)

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label
        role = track.get("role", "player")
        jersey = track.get("jersey_number")
        label = f"ID:{track_id}"
        if team and team != "unknown":
            label = f"{team[5:]}-{track_id}"  # e.g., "A-5"
        if jersey:
            label += f" #{jersey}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Track history (kept for potential future use, not drawn on camera view)
        center = ((x1 + x2) // 2, y2)
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(center)
        if len(track_history[track_id]) > 15:
            track_history[track_id] = track_history[track_id][-15:]

    # Draw legend
    y_offset = 30
    for team_name, color in TEAM_COLORS.items():
        if team_name in team_assignments.values():
            cv2.rectangle(vis, (10, y_offset - 15), (25, y_offset), color, -1)
            cv2.putText(vis, team_name, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

    cv2.putText(vis, f"Tracks: {len(tracks)}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def draw_ball_track(
    frame: np.ndarray,
    ball_track: dict | None,
    ball_trajectory: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw ball position and trajectory on frame."""
    if ball_track is None:
        return frame

    vis = frame.copy()
    center = ball_track.get("center", [0, 0])
    cx, cy = int(center[0]), int(center[1])

    # Draw trajectory trail (fading from old to new)
    if ball_trajectory and len(ball_trajectory) > 1:
        n = len(ball_trajectory)
        for i in range(1, n):
            alpha = i / n  # 0→1 as we approach current frame
            color = (0, int(100 + 65 * alpha), int(180 + 75 * alpha))  # Fading orange
            thickness = max(1, int(2 * alpha))
            pt1 = (int(ball_trajectory[i - 1][0]), int(ball_trajectory[i - 1][1]))
            pt2 = (int(ball_trajectory[i][0]), int(ball_trajectory[i][1]))
            cv2.line(vis, pt1, pt2, color, thickness)

    # Draw ball with outline
    radius = 10
    cv2.circle(vis, (cx, cy), radius, TEAM_COLORS["ball"], -1)
    cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2)

    # Draw velocity arrow
    if "velocity" in ball_track:
        vx, vy = ball_track["velocity"]
        speed = (vx**2 + vy**2) ** 0.5
        if speed > 2.0:
            max_len = 60.0
            arrow_scale = min(4.0, max_len / speed)
            end_x = int(cx + vx * arrow_scale)
            end_y = int(cy + vy * arrow_scale)
            cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.25)

    # Draw label with height info
    conf = ball_track.get("confidence", 0.0)
    status = ball_track.get("status", "unknown")
    label = f"Ball {conf:.2f}"
    if status == "confirmed":
        label += " [OK]"
    height_m = ball_track.get("height", 0.0)
    if height_m and height_m > 0.3:
        label += f" h={height_m:.1f}m"
    cv2.putText(vis, label, (cx + 15, cy + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEAM_COLORS["ball"], 1)

    return vis


class _FramePrefetcher:
    """Read video frames in a background thread to overlap IO with GPU inference."""

    def __init__(self, video_path: str | Path, frame_indices: list[int], prefetch_size: int = 4):
        self._cap = cv2.VideoCapture(str(video_path))
        self._frame_indices = frame_indices
        self._buffer: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._prefetch_size = prefetch_size
        self._done = False

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        for fidx in self._frame_indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = self._cap.read()
            if not ret:
                frame = None

            with self._cond:
                # Wait if buffer is full
                while len(self._buffer) >= self._prefetch_size and not self._done:
                    self._cond.wait(timeout=1.0)
                if self._done:
                    break
                self._buffer[fidx] = frame
                self._cond.notify_all()

        with self._cond:
            self._done = True
            self._cond.notify_all()
        self._cap.release()

    def get(self, frame_idx: int, timeout: float = 30.0) -> np.ndarray | None:
        with self._cond:
            while frame_idx not in self._buffer:
                if self._done and frame_idx not in self._buffer:
                    return None
                self._cond.wait(timeout=timeout)
            frame = self._buffer.pop(frame_idx)
            self._cond.notify_all()  # Unblock reader if waiting on full buffer
            return frame

    def shutdown(self):
        with self._cond:
            self._done = True
            self._cond.notify_all()
        self._thread.join(timeout=5)


def run_stage2(
    video_path: Path,
    output_dir: Path,
    calibration_dir: Path | None = None,
    config: dict | None = None,
):
    """Run Stage 2 tracking and identification.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        calibration_dir: Path to Stage 1 calibration results
        config: Optional configuration dict

    Returns:
        Dict with tracking statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    if config is None:
        config = get_default_config()
    # Stage 2 uses tracking_fps if set, otherwise falls back to process_fps
    tracking_fps = config.get("video", {}).get("tracking_fps")
    process_fps = tracking_fps or get_process_fps_from_config(config)

    # Load calibration data (supports both physical camera poses and legacy homographies)
    homographies = {}
    camera_poses = {}
    if calibration_dir:
        if (calibration_dir / "camera_poses.pkl").exists():
            print("Loading Stage 1 physical camera poses...")
            with open(calibration_dir / "camera_poses.pkl", "rb") as f:
                camera_poses = pickle.load(f)
            print(f"  Loaded {len(camera_poses)} camera poses")
            # Pre-compute homographies from physical params for legacy code paths
            for fidx, pose in camera_poses.items():
                R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
                K = np.array(pose["K"], dtype=np.float64)
                tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
                H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
                if abs(H[2, 2]) > 1e-10:
                    H = H / H[2, 2]
                homographies[fidx] = H
        elif (calibration_dir / "homographies.pkl").exists():
            print("Loading Stage 1 calibration results...")
            with open(calibration_dir / "homographies.pkl", "rb") as f:
                homographies = pickle.load(f)
            print(f"  Loaded {len(homographies)} homographies")

    # Initialize detector
    print("Stage 2: Initializing YOLOv8 detector...")
    detector = PlayerDetector({
        "model": "yolov8x",
        "confidence_threshold": 0.5,
        "iou_threshold": 0.45,
        "classes": [0],  # Person class
        "imgsz": 1280,
    })
    detector.load_model()

    # Open video first to get dimensions
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize StrongSORT tracker
    print("Stage 2: Initializing StrongSORT tracker...")
    # Frame interval: native_fps / process_fps (e.g., 30/10 = 3.0)
    # Scales Kalman noise so gating works correctly at lower fps
    effective_fps = process_fps if process_fps and process_fps < fps else fps
    frame_interval = fps / effective_fps if effective_fps > 0 else 1.0
    tracker = StrongSORTTracker({
        "max_age": 50,
        "n_init": 3,
        "max_iou_distance": 0.7,
        "max_cosine_distance": 0.3,  # Tighter cosine distance
        "feature_alpha": 0.9,
        "frame_interval": frame_interval,
    })
    tracker.img_w = width
    tracker.img_h = height

    # Initialize ReID extractor via factory function
    reid_backend = config.get("reid", {}).get("backend", "osnet")
    print(f"Stage 2: Initializing ReID extractor ({reid_backend})...")
    reid_extractor = get_reid_extractor(config)
    reid_extractor.load_model()

    # Initialize team classifier via factory function and goalkeeper detector
    tc_backend = config.get("team_classification", {}).get("backend", "kmeans")
    print(f"Stage 2: Initializing team classifier ({tc_backend})...")
    team_classifier = get_team_classifier(config)
    fr_config = config.get("field_registration", {})
    phys_config = fr_config.get("physical", {})
    pitch_length = phys_config.get("pitch_length", 105.0)
    pitch_width = phys_config.get("pitch_width", 68.0)
    gk_detector = GoalkeeperDetector(pitch_length=pitch_length, pitch_width=pitch_width)

    # Update module-level pitch dimensions for filtering
    global _PITCH_HALF_LENGTH, _PITCH_HALF_WIDTH
    _PITCH_HALF_LENGTH = pitch_length / 2
    _PITCH_HALF_WIDTH = pitch_width / 2

    # Initialize ball detection and tracking
    ball_config = config.get("ball_detection", {})
    ball_tracking_config = config.get("ball_tracking", {})
    ball_enabled = ball_config.get("enabled", True)

    ball_detector = None
    ball_tracker = None
    ball_trajectory_3d = None
    if ball_enabled:
        print("Stage 2: Initializing ball detector and tracker...")
        ball_detector = BallDetector(ball_config)
        ball_detector.load_model()
        # Inject runtime parameters for fps-aware max_age and bounds checking
        ball_tracking_config["fps"] = effective_fps
        ball_tracking_config["frame_width"] = width
        ball_tracking_config["frame_height"] = height
        ball_tracker = BallTracker(ball_tracking_config)
        # 3D trajectory estimator (physics-based)
        traj3d_config = ball_tracking_config.get("trajectory_3d", {})
        traj3d_config["pitch_half_length"] = pitch_length / 2
        traj3d_config["pitch_half_width"] = pitch_width / 2
        if traj3d_config.get("enabled", True):
            ball_trajectory_3d = BallTrajectory3D(traj3d_config)

    # Limit to 1 minute of video for faster testing
    max_duration_sec = 60
    max_frames = int(max_duration_sec * fps)
    effective_frames = min(total_frames, max_frames)

    sampler = FrameSampler(effective_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames ({max_duration_sec}s) at {process_fps or fps} fps")

    # Interpolate camera poses for frames not in stage1 (when tracking_fps > calibration_fps)
    if camera_poses and any(f not in camera_poses for f in sampler):
        camera_poses = _interpolate_camera_poses(camera_poses, list(sampler))
        # Re-compute homographies for interpolated poses
        homographies = {}
        for fidx, pose in camera_poses.items():
            R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
            K = np.array(pose["K"], dtype=np.float64)
            tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
            H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
            if abs(H[2, 2]) > 1e-10:
                H = H / H[2, 2]
            homographies[fidx] = H
        print(f"  Interpolated camera poses: {len(camera_poses)} (from stage1 calibration)")

    # Output video (camera view + top-down pitch side-by-side)
    output_fps = process_fps if process_fps and process_fps < fps else fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    topdown_w = int(height * 0.75)
    combined_w = width + topdown_w
    out = cv2.VideoWriter(str(output_dir / "tracking.mp4"), fourcc, output_fps, (combined_w, height))

    # Storage
    all_tracks = {}
    track_features = {}  # track_id -> list of features
    track_color_hists = {}  # track_id -> list of jersey color histograms
    track_saturations = {}  # track_id -> list of mean saturation values
    track_positions = {}  # track_id -> list of positions
    track_history = {}  # For visualization
    team_assignments = {}  # track_id -> team
    tentative_buffer = {}  # track_id -> list of (frame_idx, track_dict) for backfill

    # Ball tracking storage
    all_ball_tracks = {}  # frame_idx -> ball track dict (final selected trajectory)
    all_ball_track_candidates = {}  # frame_idx -> list of all track dicts
    ball_track_histories = {}  # track_id -> list of (frame_idx, track_dict)
    ball_debug_log: dict[int, dict] = {}  # frame_idx -> debug info
    all_ball_dets_diag: dict[int, list[dict]] = {}  # diagnostic visualization data
    ball_trajectory_history = []  # For visualization

    # === Two-pass ball detection ===
    # Pre-scan entire video for ball positions, then crop+enlarge missed frames.
    # Results stored in all_ball_detections: {frame_idx: list[detection_dict]}
    all_ball_detections = {}
    two_pass_enabled = ball_config.get("two_pass", False) and ball_detector is not None
    if two_pass_enabled:
        crop_size = ball_config.get("crop_size", 300)
        crop_enlarge_to = ball_config.get("crop_enlarge_to", 640)
        max_gap = ball_config.get("max_interpolation_gap", 30)
        frame_indices = list(sampler)

        # ---- Helper: threaded frame reader for pipelining I/O with GPU ----
        import threading
        import queue as _queue

        def _batch_reader_full(
            video_path: str | Path,
            tasks: list[int],
            batch_size: int,
            out_queue: _queue.Queue,
        ) -> None:
            """Read full frames in batches and push to queue."""
            cap = cv2.VideoCapture(str(video_path))
            prev_fidx = -1
            batch_fidxs: list[int] = []
            batch_frames: list[np.ndarray] = []
            for fidx in tasks:
                # Sequential grab is much faster than seek for nearby frames
                gap = fidx - prev_fidx - 1
                if prev_fidx >= 0 and 0 <= gap <= 8:
                    for _ in range(gap):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                prev_fidx = fidx
                if not ret:
                    continue
                batch_fidxs.append(fidx)
                batch_frames.append(frame)
                if len(batch_fidxs) >= batch_size:
                    out_queue.put((batch_fidxs, batch_frames))
                    batch_fidxs, batch_frames = [], []
            if batch_fidxs:
                out_queue.put((batch_fidxs, batch_frames))
            out_queue.put(None)  # sentinel
            cap.release()

        def _batch_reader_crop(
            video_path: str | Path,
            tasks: list[tuple[int, tuple[float, float]]],
            batch_size: int,
            detector: object,
            crop_sz: int,
            enlarge_to: int,
            out_queue: _queue.Queue,
        ) -> None:
            """Read frames, crop+enlarge, and push batches to queue."""
            cap = cv2.VideoCapture(str(video_path))
            prev_fidx = -1
            batch_fidxs: list[int] = []
            batch_images: list[np.ndarray] = []
            batch_metas: list[dict] = []
            for fidx, center in tasks:
                gap = fidx - prev_fidx - 1
                if prev_fidx >= 0 and 0 <= gap <= 8:
                    for _ in range(gap):
                        cap.grab()
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                prev_fidx = fidx
                if not ret:
                    continue
                img, meta = detector.prepare_crop(frame, center, crop_sz, enlarge_to)
                if img is None:
                    continue
                batch_fidxs.append(fidx)
                batch_images.append(img)
                batch_metas.append(meta)
                if len(batch_fidxs) >= batch_size:
                    out_queue.put((batch_fidxs, batch_images, batch_metas))
                    batch_fidxs, batch_images, batch_metas = [], [], []
            if batch_fidxs:
                out_queue.put((batch_fidxs, batch_images, batch_metas))
            out_queue.put(None)  # sentinel
            cap.release()

        # Pass 1: scan all frames, keep ALL detections (no anchor filtering)
        print("\nBall detection pass 1: scanning all frames...")
        pass1_batch_size = ball_config.get("pass1_batch_size", 8)
        num_batches = (len(frame_indices) + pass1_batch_size - 1) // pass1_batch_size

        read_q: _queue.Queue = _queue.Queue(maxsize=2)
        reader = threading.Thread(
            target=_batch_reader_full,
            args=(video_path, frame_indices, pass1_batch_size, read_q),
            daemon=True,
        )
        reader.start()

        pbar = tqdm(total=num_batches, desc="  Ball pass 1")
        while True:
            item = read_q.get()
            if item is None:
                break
            batch_fidxs, batch_frames = item

            if ball_detector.use_sahi:
                batch_dets_list = [ball_detector.detect(f) for f in batch_frames]
            else:
                batch_dets_list = ball_detector.detect_batch(batch_frames)

            for fidx, dets in zip(batch_fidxs, batch_dets_list):
                dets = ball_detector.filter_by_size(dets)
                for d in dets:
                    d["source"] = "pass1"
                all_ball_detections[fidx] = dets
            pbar.update(1)
        pbar.close()
        reader.join()

        pass1_det_frames = sum(1 for d in all_ball_detections.values() if d)
        print(f"  Pass 1: {pass1_det_frames} frames with detections out of {len(frame_indices)}")

        # Pass 2: crop+enlarge using preliminary tracker predictions
        # Run a preliminary tracker pass on Pass 1 detections to get
        # tracked positions, then use those for crop centers.
        prelim_tracker = BallTracker(ball_tracking_config)
        prelim_positions: dict[int, tuple[float, float]] = {}

        for fidx in sorted(frame_indices):
            dets = all_ball_detections.get(fidx, [])
            tracks = prelim_tracker.update(dets)
            for t in tracks:
                if not t.get("predicted", False):
                    prelim_positions[fidx] = tuple(t["center"])
                    break  # use first active track

        pass2_tasks: list[tuple[int, tuple[float, float]]] = []
        for fidx in frame_indices:
            if fidx in prelim_positions:
                continue  # tracker successfully tracked ball at this frame
            # Tracker lost ball — interpolate crop center from nearest tracked frames
            center = _interpolate_tracked_position(fidx, prelim_positions, max_gap)
            if center is not None:
                pass2_tasks.append((fidx, center))

        if pass2_tasks:
            print(f"Ball detection pass 2: crop+enlarge on {len(pass2_tasks)} frames...")
            pass2_count = 0
            pass2_batch_size = ball_config.get("pass2_batch_size", 32)

            num_batches2 = (len(pass2_tasks) + pass2_batch_size - 1) // pass2_batch_size
            read_q2: _queue.Queue = _queue.Queue(maxsize=2)
            reader2 = threading.Thread(
                target=_batch_reader_crop,
                args=(video_path, pass2_tasks, pass2_batch_size,
                      ball_detector, crop_size, crop_enlarge_to, read_q2),
                daemon=True,
            )
            reader2.start()

            pbar2 = tqdm(total=num_batches2, desc="  Ball pass 2")
            while True:
                item = read_q2.get()
                if item is None:
                    break
                batch_fidxs, batch_images, batch_metas = item

                batch_results = ball_detector.detect_crop_batch(
                    batch_images, batch_metas, enlarge_to=crop_enlarge_to)

                for fidx, crop_dets in zip(batch_fidxs, batch_results):
                    crop_dets = ball_detector.filter_by_size(crop_dets)
                    if crop_dets:
                        for d in crop_dets:
                            d["source"] = "pass2"
                        existing = all_ball_detections.get(fidx, [])
                        all_ball_detections[fidx] = existing + crop_dets
                        pass2_count += 1
                pbar2.update(1)
            pbar2.close()
            reader2.join()

            print(f"  Pass 2: found ball in {pass2_count} additional frames")

        # Save diagnostic data (all detections from both passes)
        all_ball_dets_diag: dict[int, list[dict]] = {}
        for fidx, dets in all_ball_detections.items():
            all_ball_dets_diag[fidx] = [
                {
                    "center": list(d["center"]),
                    "bbox": list(d["bbox"]),
                    "confidence": d["confidence"],
                    "source": d.get("source", "pass1"),
                }
                for d in dets
            ]

        total_det_frames = sum(1 for d in all_ball_detections.values() if d)
        print(f"  Total frames with ball detections: {total_det_frames}/{len(frame_indices)}")

    # Process frames
    print("\nStage 2: Processing frames...")
    # Team classification will be done AFTER all tracking is complete
    # This ensures we use trajectory-averaged features for robust clustering

    # Start background frame reader to overlap IO with GPU inference
    prefetcher = _FramePrefetcher(video_path, list(sampler), prefetch_size=4)
    cap.release()  # Prefetcher manages its own VideoCapture

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 2: Tracking")):
        frame = prefetcher.get(frame_idx)
        if frame is None:
            break

        # Get homography for this frame (Stage 1 stores world->image, we need image->world)
        H_world2img = homographies.get(frame_idx)
        H = None
        if H_world2img is not None:
            try:
                H = np.linalg.inv(H_world2img)
            except np.linalg.LinAlgError:
                H = None  # Singular matrix, skip pitch filtering

        # Detect players
        detections = detector.detect(frame)

        # Filter by size (min_height=25 to include distant players)
        detections = detector.filter_by_size(
            detections,
            min_height=25,
            max_height=350,
            min_aspect_ratio=0.25,
            max_aspect_ratio=1.0,
        )

        # Filter by pitch boundary if calibration available
        if frame_idx in camera_poses:
            detections = _filter_by_pitch_undistorted(detections, camera_poses[frame_idx], margin=5.0)
        elif H is not None:
            detections = detector.filter_by_pitch(detections, H, margin=5.0)

        # Extract ReID features
        embeddings = None
        if detections:
            crops = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    crops.append(frame[y1:y2, x1:x2])
                else:
                    crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            embeddings = reid_extractor.extract(crops)

        # Update tracker (returns both confirmed and tentative tracks)
        tracks = tracker.update(detections, embeddings)
        confirmed_tracks = [t for t in tracks if t.get("confirmed", True)]
        tentative_tracks_now = [t for t in tracks if not t.get("confirmed", True)]

        # Buffer tentative tracks for potential backfill
        for t in tentative_tracks_now:
            tid = t["track_id"]
            tentative_buffer.setdefault(tid, []).append((frame_idx, t))

        # Backfill: when a track just got confirmed, insert its buffered frames
        for t in confirmed_tracks:
            tid = t["track_id"]
            if tid in tentative_buffer:
                for buf_fidx, buf_track in tentative_buffer.pop(tid):
                    # Compute pitch position from bbox + camera pose
                    buf_pitch_pos = None
                    buf_H = homographies.get(buf_fidx)
                    if buf_fidx in camera_poses:
                        foot_x = (buf_track["bbox"][0] + buf_track["bbox"][2]) / 2
                        foot_y = buf_track["bbox"][3]
                        buf_pitch_pos = _undistort_and_project_to_pitch(
                            np.array([[foot_x, foot_y]]), camera_poses[buf_fidx]
                        )
                    elif buf_H is not None:
                        foot_x = (buf_track["bbox"][0] + buf_track["bbox"][2]) / 2
                        foot_y = buf_track["bbox"][3]
                        try:
                            H_inv = np.linalg.inv(buf_H)
                            ph = H_inv @ np.array([foot_x, foot_y, 1.0])
                            if abs(ph[2]) > 1e-6:
                                buf_pitch_pos = [float(ph[0] / ph[2]), float(ph[1] / ph[2])]
                        except np.linalg.LinAlgError:
                            pass

                    buf_entry = {
                        "track_id": tid,
                        "bbox": buf_track["bbox"],
                        "confidence": buf_track.get("confidence", 1.0),
                        "pitch_position": buf_pitch_pos,
                        "team": "unknown",  # Will be updated in final assignment pass
                        "role": "player",
                    }
                    all_tracks.setdefault(buf_fidx, []).append(buf_entry)

                    if buf_pitch_pos:
                        track_positions.setdefault(tid, []).append(buf_pitch_pos)

        # Store confirmed track info
        frame_tracks = []
        for i, track in enumerate(confirmed_tracks):
            track_id = track["track_id"]

            # Compute pitch position
            pitch_pos = None
            if frame_idx in camera_poses:
                foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
                foot_y = track["bbox"][3]
                pitch_pos = _undistort_and_project_to_pitch(
                    np.array([[foot_x, foot_y]]), camera_poses[frame_idx]
                )
            elif H is not None:
                foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
                foot_y = track["bbox"][3]
                pt_h = np.array([foot_x, foot_y, 1.0])
                world_h = H @ pt_h
                if abs(world_h[2]) > 1e-6:
                    pitch_pos = [float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2])]

            # Store features and positions for team classification
            if track_id not in track_features:
                track_features[track_id] = []
                track_color_hists[track_id] = []
                track_positions.setdefault(track_id, [])

            # Get feature from tracker
            tracker_features = tracker.get_track_features()
            if track_id in tracker_features and tracker_features[track_id] is not None:
                track_features[track_id].append(tracker_features[track_id])

            # Extract jersey color histogram for team classification
            bx1, by1, bx2, by2 = map(int, track["bbox"])
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(width, bx2), min(height, by2)
            if bx2 > bx1 and by2 > by1:
                crop = frame[by1:by2, bx1:bx2]
                color_hist = _extract_jersey_color_hist(crop)
                if color_hist is not None:
                    track_color_hists[track_id].append(color_hist)
                mean_sat = _extract_jersey_mean_saturation(crop)
                if mean_sat is not None:
                    track_saturations.setdefault(track_id, []).append(mean_sat)

            if pitch_pos:
                track_positions[track_id].append(pitch_pos)

            # Determine role
            team = team_assignments.get(track_id, "unknown")
            role = "player"
            if team == "referee":
                role = "referee"
            elif pitch_pos:
                role = gk_detector.classify_role(pitch_pos, team)

            track_info = {
                "track_id": track_id,
                "bbox": track["bbox"],
                "confidence": track.get("confidence", 1.0),
                "pitch_position": pitch_pos,
                "team": team,
                "role": role,
            }
            frame_tracks.append(track_info)

        all_tracks[frame_idx] = frame_tracks

        # Ball detection and tracking — collect ALL tracks for post-loop filtering
        if ball_detector and ball_tracker:
            if two_pass_enabled:
                ball_detections = all_ball_detections.get(frame_idx, [])
            else:
                ball_detections = ball_detector.detect(frame)
                ball_detections = ball_detector.filter_by_size(ball_detections)

            ball_tracks = ball_tracker.update(ball_detections)

            # Record debug info for every frame
            ball_debug_log[frame_idx] = {
                "detections": [
                    {"center": list(d["center"]), "confidence": d.get("confidence", 0),
                     "bbox": list(d.get("bbox", [])), "source": d.get("source", "pass1")}
                    for d in ball_detections
                ],
                "tracker_output": [
                    {"track_id": t["track_id"], "center": list(t["center"]),
                     "confidence": t["confidence"], "predicted": t.get("predicted", False)}
                    for t in ball_tracks
                ],
            }

            # Store all active tracks per frame; accumulate per-track histories
            all_ball_track_candidates[frame_idx] = ball_tracks
            for t in ball_tracks:
                tid = t["track_id"]
                ball_track_histories.setdefault(tid, []).append((frame_idx, t))

        # Note: Team classification is deferred until all tracking is complete
        # This allows using trajectory-averaged features for better clustering

    prefetcher.shutdown()

    # Save ball tracking debug log
    if ball_debug_log:
        debug_path = output_dir / "ball_debug_log.json"
        with open(debug_path, "w") as f:
            json.dump({str(k): v for k, v in sorted(ball_debug_log.items())}, f, indent=1)
        print(f"  Ball debug log saved to {debug_path}")

    # === Field-space trajectory projection and filtering ===
    if ball_detector and ball_tracker and ball_track_histories:
        print("\nProjecting ball trajectories to field space...")
        trajectory_field_data: dict[int, list[tuple]] = {}

        for tid, history in ball_track_histories.items():
            field_points = []
            for fidx, track_dict in history:
                if track_dict.get("predicted", False):
                    continue
                pixel_center = track_dict["center"]
                conf = track_dict.get("confidence", 0.0)

                if fidx not in camera_poses:
                    continue
                field_pos = _undistort_and_project_to_pitch(
                    np.array([[pixel_center[0], pixel_center[1]]]),
                    camera_poses[fidx],
                )
                if field_pos is None:
                    raise RuntimeError(
                        f"Failed to project ball at frame {fidx} "
                        f"pixel=({pixel_center[0]:.1f}, {pixel_center[1]:.1f}) "
                        f"using camera pose"
                    )

                field_points.append((fidx, tuple(field_pos), tuple(pixel_center), conf))

            if field_points:
                trajectory_field_data[tid] = field_points

        print(f"  Total trajectories: {len(trajectory_field_data)}")
        for tid, pts in trajectory_field_data.items():
            print(f"    Track {tid}: {len(pts)} projected points "
                  f"(frames {pts[0][0]}-{pts[-1][0]})")

        # Apply field-space filtering
        field_filter_config = ball_config.get("field_filter", {})
        if field_filter_config.get("enabled", True):
            filtered_trajectories = _filter_trajectories_field_space(
                trajectory_field_data, fps, field_filter_config,
            )
            n_rejected = len(trajectory_field_data) - len(filtered_trajectories)
            print(f"  Field filter: {len(filtered_trajectories)} trajectories pass, "
                  f"{n_rejected} rejected")
        else:
            filtered_trajectories = trajectory_field_data

        # Use ALL valid trajectories (they cover different time ranges)
        # For frames with overlapping tracks, pick the one with higher confidence
        if filtered_trajectories:
            # Build field_pos lookup from all valid trajectories
            field_lookup: dict[int, tuple] = {}
            for tid, points in filtered_trajectories.items():
                for fidx, field_pos, _, _ in points:
                    field_lookup[fidx] = field_pos

            # Populate all_ball_tracks from all valid trajectories
            # Sort by trajectory score so higher-scoring tracks take priority on overlap
            scored_tids = []
            max_len = max(len(pts) for pts in filtered_trajectories.values())
            for tid, points in filtered_trajectories.items():
                score = len(points) / max(max_len, 1) * 0.6 + \
                    sum(p[3] for p in points) / len(points) * 0.25 + \
                    len(points) / max(points[-1][0] - points[0][0] + 1, 1) * 0.15
                scored_tids.append((score, tid))
            scored_tids.sort(reverse=True)

            for _, tid in scored_tids:
                for fidx, track_dict in ball_track_histories[tid]:
                    if track_dict.get("predicted", False):
                        continue
                    if fidx in all_ball_tracks:
                        continue  # Higher-scored track already owns this frame
                    entry = dict(track_dict)
                    if fidx in field_lookup:
                        entry["pitch_position"] = list(field_lookup[fidx])
                        entry["height"] = 0.0
                        entry["on_ground"] = True
                    all_ball_tracks[fidx] = entry
                    ball_trajectory_history.append(tuple(track_dict["center"]))

            # Run BallTrajectory3D on all valid trajectories for 3D height estimation
            if ball_trajectory_3d:
                for _, tid in scored_tids:
                    ball_trajectory_3d.clear()
                    for fidx, track_dict in ball_track_histories[tid]:
                        if track_dict.get("predicted", False):
                            continue
                        if fidx not in camera_poses:
                            continue
                        time_sec = fidx / fps if fps > 0 else 0.0
                        ball_bbox = tuple(track_dict["bbox"]) if "bbox" in track_dict else None
                        ball_trajectory_3d.add_observation(
                            time_sec, tuple(track_dict["center"]),
                            camera_poses[fidx], bbox=ball_bbox,
                        )
                        traj_result = ball_trajectory_3d.estimate()
                        if traj_result and not traj_result.get("rejected"):
                            if fidx in all_ball_tracks:
                                all_ball_tracks[fidx]["pitch_position"] = traj_result["pitch_position"]
                                all_ball_tracks[fidx]["height"] = traj_result["height"]
                                all_ball_tracks[fidx]["position_3d"] = traj_result["position_3d"]
                                all_ball_tracks[fidx]["on_ground"] = traj_result["on_ground"]

            # Remove spatial outliers from the merged track
            n_outliers = _remove_merge_outliers(all_ball_tracks, fps)
            if n_outliers > 0:
                print(f"  Removed {n_outliers} merge outlier(s)")

            print(f"  Ball tracked in {len(all_ball_tracks)}/{len(frame_indices)} frames "
                  f"(from {len(filtered_trajectories)} valid trajectories)")

            # Tag rejected trajectory detections in diagnostic data
            rejected_tids = set(trajectory_field_data.keys()) - set(filtered_trajectories.keys())
            for tid in rejected_tids:
                for fidx, track_dict in ball_track_histories.get(tid, []):
                    if fidx in all_ball_dets_diag:
                        for d in all_ball_dets_diag[fidx]:
                            tc = track_dict["center"]
                            dc = d["center"]
                            if abs(dc[0] - tc[0]) < 2 and abs(dc[1] - tc[1]) < 2:
                                d["source"] = "rejected"
        else:
            print("  No valid ball trajectory found after filtering")

        # Render diagnostic visualization
        if all_ball_dets_diag:
            _render_ball_detection_diag(
                video_path, sampler, all_ball_dets_diag, output_dir,
            )

    # Final team classification using jersey color histograms
    print("\nFinalizing team assignments using jersey color histograms...")

    # Compute trajectory-averaged color histograms
    mean_color_features = {}
    for tid, hists in track_color_hists.items():
        if hists:
            mean_color_features[tid] = np.mean(hists, axis=0)

    # Compute median pitch positions (robust to projection outliers)
    mean_positions = {}
    for tid, poss in track_positions.items():
        if poss:
            mean_positions[tid] = np.median(poss, axis=0).tolist()

    print(f"  Tracks with color features: {len(mean_color_features)}")
    print(f"  Tracks with positions: {len(mean_positions)}")

    # Compute median bbox heights per track (for filtering noisy small tracks)
    track_bbox_heights_all = {}
    for frame_idx in all_tracks:
        for track in all_tracks[frame_idx]:
            tid = track["track_id"]
            bbox = track["bbox"]
            h = bbox[3] - bbox[1]
            track_bbox_heights_all.setdefault(tid, []).append(h)
    median_bbox_heights = {
        tid: float(np.median(hs)) for tid, hs in track_bbox_heights_all.items()
    }

    # Frame counts per track (for filtering short-lived tracks in referee detection)
    track_frame_counts = {}
    for frame_idx in all_tracks:
        for track in all_tracks[frame_idx]:
            tid = track["track_id"]
            track_frame_counts[tid] = track_frame_counts.get(tid, 0) + 1

    # Compute mean saturation per track (for achromatic referee detection)
    mean_saturations = {}
    for tid, sats in track_saturations.items():
        if sats:
            mean_saturations[tid] = float(np.mean(sats))

    if len(mean_color_features) >= 6:
        team_assignments = team_classifier.fit(
            mean_color_features, mean_positions,
            track_bbox_heights=median_bbox_heights,
            track_frame_counts=track_frame_counts,
            track_mean_saturations=mean_saturations,
        )
        print(f"  Team assignments: {len(team_assignments)} tracks classified")
    else:
        print("  Warning: Not enough tracks for team classification")

    # Role refinement: linesman + goalkeeper detection by position
    goalkeeper_tracks = set()
    if mean_positions:
        print("  Refining roles by position...")
        team_assignments, goalkeeper_tracks = gk_detector.refine_roles(
            team_assignments, mean_positions, all_positions=track_positions
        )

    # Filter out off-field tracks (substitutes, coaches, spectators)
    # A track is off-field if its median position is outside the pitch boundary
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    off_field_margin = 0.5  # meters beyond pitch boundary to still include
    off_field_tracks = set()
    for tid, pos in mean_positions.items():
        if abs(pos[0]) > half_l + off_field_margin or abs(pos[1]) > half_w + off_field_margin:
            off_field_tracks.add(tid)
    if off_field_tracks:
        print(f"  Off-field tracks removed: {sorted(off_field_tracks)}")
        for tid in off_field_tracks:
            team_assignments.pop(tid, None)

    # Update all tracks with final team assignments and roles
    for frame_idx in all_tracks:
        all_tracks[frame_idx] = [
            track for track in all_tracks[frame_idx]
            if track["track_id"] not in off_field_tracks
        ]
        for track in all_tracks[frame_idx]:
            tid = track["track_id"]
            track["team"] = team_assignments.get(tid, "unknown")
            if tid in goalkeeper_tracks:
                track["role"] = "goalkeeper"
            elif track["team"] == "referee":
                track["role"] = "referee"
            else:
                track["role"] = "player"

    # Generate visualization with final team assignments
    print("\nGenerating visualization with final team assignments...")
    render_prefetcher = _FramePrefetcher(video_path, list(sampler), prefetch_size=4)
    track_history = {}  # Reset for visualization

    # Create per-frame output directories
    vis_frames_dir = output_dir / "frames"
    vis_frames_dir.mkdir(exist_ok=True)

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 2: Rendering")):
        frame = render_prefetcher.get(frame_idx)
        if frame is None:
            break

        # Get tracks for this frame with updated team assignments
        frame_tracks = all_tracks.get(frame_idx, [])

        # Draw visualization
        vis = draw_tracks(frame, frame_tracks, track_history, team_assignments)

        # Draw ball if available
        ball_track_this = None
        ball_traj_world = None
        if ball_enabled and frame_idx in all_ball_tracks:
            ball_track_this = all_ball_tracks[frame_idx]
            # Get pixel trajectory up to this frame (for camera view)
            traj = [all_ball_tracks[fi]["center"] for fi in sorted(all_ball_tracks.keys()) if fi <= frame_idx]
            traj = traj[-30:]
            vis = draw_ball_track(vis, ball_track_this, traj)
            # Get world trajectory (for top-down view)
            ball_traj_world = [
                all_ball_tracks[fi]["pitch_position"]
                for fi in sorted(all_ball_tracks.keys())
                if fi <= frame_idx and all_ball_tracks[fi].get("pitch_position")
            ]
            ball_traj_world = ball_traj_world[-30:]

        # Draw top-down pitch diagram
        topdown = draw_topdown_pitch(
            height, frame_tracks, team_assignments,
            ball_track=ball_track_this,
            ball_trajectory_world=ball_traj_world,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        combined = np.hstack([vis, topdown])

        out.write(combined)

        # Save per-frame visualization image and JSON
        fname = f"frame_{frame_idx:05d}"
        cv2.imwrite(str(vis_frames_dir / f"{fname}.jpg"), combined)

        frame_json = {
            "frame_idx": int(frame_idx),
            "timestamp_sec": round(frame_idx / fps, 3),
            "num_tracks": len(frame_tracks),
            "tracks": frame_tracks,
            "ball": ball_track_this,
        }
        with open(vis_frames_dir / f"{fname}.json", "w") as jf:
            json.dump(frame_json, jf, indent=2, default=_json_default)

    render_prefetcher.shutdown()
    out.release()

    # Save results
    with open(output_dir / "tracks.json", "w") as f:
        json.dump({str(k): v for k, v in all_tracks.items()}, f)

    with open(output_dir / "team_assignments.json", "w") as f:
        json.dump(team_assignments, f, indent=2)

    # Save track features for Stage 3
    serializable_features = {
        str(tid): [f.tolist() if isinstance(f, np.ndarray) else f for f in feats]
        for tid, feats in track_features.items()
    }
    with open(output_dir / "track_features.json", "w") as f:
        json.dump(serializable_features, f)

    # Save ball tracks
    if ball_enabled and all_ball_tracks:
        def _sanitize(obj):
            """Convert numpy types to native Python for JSON serialization."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_ball_tracks = {str(k): _sanitize(v) for k, v in all_ball_tracks.items()}
        with open(output_dir / "ball_tracks.json", "w") as f:
            json.dump(serializable_ball_tracks, f, indent=2)
        print(f"  Saved {len(all_ball_tracks)} ball detections")

    # Statistics
    unique_tracks = set()
    for frame_tracks in all_tracks.values():
        for t in frame_tracks:
            unique_tracks.add(t["track_id"])

    team_counts = {}
    for team in team_assignments.values():
        team_counts[team] = team_counts.get(team, 0) + 1

    stats = {
        "total_frames": total_frames,
        "processed_frames": len(sampler),
        "unique_tracks": len(unique_tracks),
        "team_counts": team_counts,
        "avg_tracks_per_frame": np.mean([len(t) for t in all_tracks.values()]) if all_tracks else 0,
    }

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStage 2 Complete:")
    print(f"  Processed frames: {len(sampler)}")
    print(f"  Unique tracks: {len(unique_tracks)}")
    print(f"  Team distribution: {team_counts}")
    print(f"  Output: {output_dir}")

    return stats


def _json_default(obj):
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage2_tracking")
    calibration_dir = Path("data/processed/stage1_field_registration")

    run_stage2(video_path, output_dir, calibration_dir)


if __name__ == "__main__":
    main()
