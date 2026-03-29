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
# Ball anchor filtering (for two-pass ball detection)
# ---------------------------------------------------------------------------

def _find_static_positions(
    anchors: dict[int, tuple[float, float]],
    static_radius: float = 40.0,
    min_cluster_size: int = 5,
    max_displacement_per_frame: float = 0.5,
) -> list[tuple[float, float]]:
    """Find positions of static objects (spare balls, watermarks) in anchors.

    Groups anchors by spatial proximity and identifies clusters with low
    average displacement per frame (static objects).

    Returns:
        List of (cx, cy) centroid positions of static object clusters.
    """
    if len(anchors) < min_cluster_size:
        return []

    sorted_frames = sorted(anchors.keys())

    # Cluster anchors by spatial proximity
    clusters: list[list[int]] = []
    visited = set()

    for fidx in sorted_frames:
        if fidx in visited:
            continue
        cluster = [fidx]
        visited.add(fidx)
        cx, cy = anchors[fidx]

        for fidx2 in sorted_frames:
            if fidx2 in visited:
                continue
            cx2, cy2 = anchors[fidx2]
            dist = ((cx2 - cx) ** 2 + (cy2 - cy) ** 2) ** 0.5
            if dist <= static_radius:
                cluster.append(fidx2)
                visited.add(fidx2)

        clusters.append(cluster)

    # Identify static clusters
    static_centroids = []
    for cluster in clusters:
        if len(cluster) < min_cluster_size:
            continue
        cluster_sorted = sorted(cluster)
        total_disp = 0.0
        total_gap = 0
        for i in range(1, len(cluster_sorted)):
            f1, f2 = cluster_sorted[i - 1], cluster_sorted[i]
            dx = anchors[f2][0] - anchors[f1][0]
            dy = anchors[f2][1] - anchors[f1][1]
            total_disp += (dx ** 2 + dy ** 2) ** 0.5
            total_gap += f2 - f1

        avg_disp = total_disp / max(total_gap, 1)
        if avg_disp < max_displacement_per_frame:
            # Compute centroid of this static cluster
            mean_x = sum(anchors[f][0] for f in cluster) / len(cluster)
            mean_y = sum(anchors[f][1] for f in cluster) / len(cluster)
            static_centroids.append((mean_x, mean_y))

    return static_centroids


def _near_any_static(
    center: tuple[float, float],
    static_positions: list[tuple[float, float]],
    radius: float = 60.0,
) -> bool:
    """Check if a detection center is near any known static object position."""
    for sx, sy in static_positions:
        dist = ((center[0] - sx) ** 2 + (center[1] - sy) ** 2) ** 0.5
        if dist <= radius:
            return True
    return False


def _filter_anchor_outliers(
    anchors: dict[int, tuple[float, float]],
    max_displacement: float = 600.0,
    min_segment_size: int = 3,
) -> list[int]:
    """Find anchor frames that belong to false trajectory segments.

    Splits anchors into segments at edges where total displacement exceeds
    max_displacement pixels. Removes segments smaller than min_segment_size.

    Returns:
        List of outlier frame indices to remove.
    """
    if len(anchors) < 5:
        return []

    sorted_frames = sorted(anchors.keys())
    n = len(sorted_frames)

    # Compute total displacement for each consecutive pair
    displacements = []
    for i in range(n - 1):
        f1, f2 = sorted_frames[i], sorted_frames[i + 1]
        dx = anchors[f2][0] - anchors[f1][0]
        dy = anchors[f2][1] - anchors[f1][1]
        displacements.append((dx ** 2 + dy ** 2) ** 0.5)

    # Split into segments at large jumps
    segments: list[list[int]] = [[sorted_frames[0]]]
    for i in range(n - 1):
        if displacements[i] > max_displacement:
            segments.append([])
        segments[-1].append(sorted_frames[i + 1])

    if len(segments) <= 1:
        return []

    # Keep the largest segment plus any segment with >= min_segment_size
    # that is at least 20% of total anchors. Remove the rest.
    largest_size = max(len(s) for s in segments)
    size_threshold = max(min_segment_size, int(0.2 * len(anchors)))
    outliers = []
    for seg in segments:
        if len(seg) < size_threshold and len(seg) < largest_size:
            outliers.extend(seg)

    return outliers


# Ball position interpolation (for two-pass ball detection)
# ---------------------------------------------------------------------------

def _interpolate_ball_position(
    frame_idx: int,
    anchor_indices: list[int],
    anchors: dict[int, tuple[float, float]],
    max_gap: int = 30,
) -> tuple[float, float] | None:
    """Interpolate ball position from nearest anchor frames.

    Args:
        frame_idx: Target frame index.
        anchor_indices: Sorted list of frame indices with anchor detections.
        anchors: {frame_idx: (cx, cy)} anchor positions.
        max_gap: Maximum frame gap for interpolation.

    Returns:
        Interpolated (cx, cy) or None if no anchors within range.
    """
    import bisect

    pos = bisect.bisect_left(anchor_indices, frame_idx)

    before_idx = anchor_indices[pos - 1] if pos > 0 else None
    after_idx = anchor_indices[pos] if pos < len(anchor_indices) else None

    # Check gap constraints
    if before_idx is not None and frame_idx - before_idx > max_gap:
        before_idx = None
    if after_idx is not None and after_idx - frame_idx > max_gap:
        after_idx = None

    if before_idx is not None and after_idx is not None:
        # Linear interpolation
        t = (frame_idx - before_idx) / (after_idx - before_idx)
        bx, by = anchors[before_idx]
        ax, ay = anchors[after_idx]
        return (bx + t * (ax - bx), by + t * (ay - by))
    elif before_idx is not None:
        return anchors[before_idx]
    elif after_idx is not None:
        return anchors[after_idx]
    else:
        return None


# Ball detection diagnostic visualization
# ---------------------------------------------------------------------------

# Color scheme:  pass1=green, pass2=blue, static=red, outlier=orange, filtered=gray
_BALL_DIAG_COLORS = {
    "pass1": (0, 220, 0),       # green
    "pass2": (255, 180, 0),     # cyan-ish (BGR)
    "static": (0, 0, 255),      # red
    "outlier": (0, 140, 255),   # orange (BGR)
    "filtered": (160, 160, 160),  # gray
}
_BALL_DIAG_LABELS = {
    "pass1": "Pass1",
    "pass2": "Pass2 (crop)",
    "static": "Static (removed)",
    "outlier": "Outlier (removed)",
    "filtered": "Trajectory filter",
}


def _render_ball_detection_diag(
    video_path: Path | str,
    sampler,
    all_ball_dets_diag: dict[int, list[dict]],
    pass1_anchors: dict[int, tuple[float, float]],
    static_positions: list[tuple[float, float]],
    output_dir: Path,
) -> None:
    """Render per-frame diagnostic images showing ball detection sources.

    Each detection is drawn as a circle with color indicating its source:
      green = pass1 (SAHI), blue = pass2 (crop+enlarge),
      red = static object (removed), gray = trajectory-filtered.
    """
    diag_dir = output_dir / "ball_detection_diag"
    diag_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    frame_indices = list(sampler)

    # Count stats
    counts = {"pass1": 0, "pass2": 0, "static": 0, "outlier": 0, "filtered": 0}
    for dets in all_ball_dets_diag.values():
        for d in dets:
            src = d.get("source", "pass1")
            counts[src] = counts.get(src, 0) + 1

    print(f"\nGenerating ball detection diagnostic visualization...")
    print(f"  pass1={counts['pass1']}  pass2={counts['pass2']}  "
          f"static={counts['static']}  outlier={counts['outlier']}  "
          f"filtered={counts['filtered']}")

    for fidx in tqdm(frame_indices, desc="  Ball diag"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        if not ret:
            continue

        dets = all_ball_dets_diag.get(fidx, [])
        is_anchor = fidx in pass1_anchors

        # Draw each detection
        for d in dets:
            src = d.get("source", "pass1")
            color = _BALL_DIAG_COLORS.get(src, (255, 255, 255))
            cx, cy = int(d["center"][0]), int(d["center"][1])
            conf = d["confidence"]
            bbox = d["bbox"]
            bx1, by1, bx2, by2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # Draw bbox
            thickness = 2 if src in ("pass1", "pass2") else 1
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, thickness)

            # Draw center circle
            radius = 12 if src in ("pass1", "pass2") else 8
            cv2.circle(frame, (cx, cy), radius, color, 2)
            if src in ("pass1", "pass2"):
                cv2.circle(frame, (cx, cy), 3, color, -1)

            # Label
            label = f"{_BALL_DIAG_LABELS.get(src, src)} {conf:.2f}"
            label_y = max(by1 - 8, 15)
            cv2.putText(frame, label, (bx1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw anchor marker if this frame is an anchor
        if is_anchor:
            ax, ay = int(pass1_anchors[fidx][0]), int(pass1_anchors[fidx][1])
            # Diamond marker for anchor
            pts = np.array([
                [ax, ay - 18], [ax + 12, ay], [ax, ay + 18], [ax - 12, ay]
            ], dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 2)  # yellow diamond

        # Draw static zone circles (dashed-style with thin line)
        for sx, sy in static_positions:
            cv2.circle(frame, (int(sx), int(sy)), 60, (0, 0, 200), 1)
            cv2.putText(frame, "STATIC", (int(sx) - 25, int(sy) - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1, cv2.LINE_AA)

        # Frame info overlay (top-left)
        n_dets = len(dets)
        sources = [d.get("source", "?") for d in dets]
        info = f"Frame {fidx}  |  {n_dets} det(s): {', '.join(sources) if sources else 'none'}"
        if is_anchor:
            info += "  [ANCHOR]"
        cv2.rectangle(frame, (0, 0), (len(info) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(frame, info, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Legend (bottom-left)
        legend_y = frame.shape[0] - 100
        cv2.rectangle(frame, (0, legend_y - 5), (220, frame.shape[0]), (0, 0, 0), -1)
        for i, (src, color) in enumerate(_BALL_DIAG_COLORS.items()):
            ly = legend_y + i * 22 + 15
            cv2.circle(frame, (15, ly - 4), 6, color, -1)
            cv2.putText(frame, _BALL_DIAG_LABELS[src], (30, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        cv2.imwrite(str(diag_dir / f"frame_{fidx:05d}.jpg"), frame)

    cap.release()
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
    all_ball_tracks = {}  # frame_idx -> ball track dict
    ball_trajectory_history = []  # For visualization

    # === Two-pass ball detection ===
    # Pre-scan entire video for ball positions, then crop+enlarge missed frames.
    # Results stored in all_ball_detections: {frame_idx: list[detection_dict]}
    all_ball_detections = {}
    two_pass_enabled = ball_config.get("two_pass", False) and ball_detector is not None
    if two_pass_enabled:
        pass1_conf = ball_config.get("pass1_confidence_threshold", 0.5)
        crop_size = ball_config.get("crop_size", 300)
        crop_enlarge_to = ball_config.get("crop_enlarge_to", 640)
        max_gap = ball_config.get("max_interpolation_gap", 30)
        frame_indices = list(sampler)

        # Pass 1: scan all frames, collect high-confidence anchors
        print("\nBall detection pass 1: scanning all frames...")
        pass1_anchors = {}  # {frame_idx: (cx, cy)}
        ball_cap = cv2.VideoCapture(str(video_path))
        for fidx in tqdm(frame_indices, desc="  Ball pass 1"):
            ball_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = ball_cap.read()
            if not ret:
                continue
            dets = ball_detector.detect(frame)
            dets = ball_detector.filter_by_size(dets)
            best = ball_detector.get_best_detection(dets)
            if best and best["confidence"] >= pass1_conf:
                pass1_anchors[fidx] = tuple(best["center"])
            # Always store all detections from pass 1 (even low-conf ones)
            for d in dets:
                d["source"] = "pass1"
            all_ball_detections[fidx] = dets

        print(f"  Pass 1: {len(pass1_anchors)} raw anchor frames out of {len(frame_indices)}")

        static_positions: list[tuple[float, float]] = []
        static_ball_dets: dict[int, list[dict]] = {}

        # Pass 2: crop+enlarge on ALL frames (supplements pass 1)
        if pass1_anchors:
            print(f"Ball detection pass 2: crop+enlarge on {len(frame_indices)} frames...")
            anchor_indices = sorted(pass1_anchors.keys())
            pass2_count = 0
            for fidx in tqdm(frame_indices, desc="  Ball pass 2"):
                # Find nearest anchors before and after
                center = _interpolate_ball_position(fidx, anchor_indices, pass1_anchors, max_gap)
                if center is None:
                    continue
                ball_cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = ball_cap.read()
                if not ret:
                    continue
                crop_dets = ball_detector.detect_crop(frame, center, crop_size, crop_enlarge_to)
                crop_dets = ball_detector.filter_by_size(crop_dets)
                if crop_dets:
                    for d in crop_dets:
                        d["source"] = "pass2"
                    # Merge with any pass 1 detections for this frame
                    existing = all_ball_detections.get(fidx, [])
                    all_ball_detections[fidx] = existing + crop_dets
                    pass2_count += 1

            print(f"  Pass 2: found ball in {pass2_count} additional frames")

        ball_cap.release()

        # ---- Post-hoc filtering (after both passes complete) ----
        raw_anchor_count = len(pass1_anchors)

        # 1. Static object filter
        if len(pass1_anchors) >= 3:
            static_positions = _find_static_positions(pass1_anchors)
            if static_positions:
                pass1_anchors = {f: c for f, c in pass1_anchors.items()
                                 if not _near_any_static(c, static_positions)}
                for fidx in list(all_ball_detections.keys()):
                    kept = []
                    for d in all_ball_detections[fidx]:
                        if _near_any_static(tuple(d["center"]), static_positions):
                            d["source"] = "static"
                            static_ball_dets.setdefault(fidx, []).append(d)
                        else:
                            kept.append(d)
                    all_ball_detections[fidx] = kept
                print(f"  Post-filter: removed {len(static_positions)} static object region(s)")

        # 2. Trajectory outlier filter
        if len(pass1_anchors) >= 3:
            outlier_frames = _filter_anchor_outliers(pass1_anchors)
            if outlier_frames:
                for fidx in outlier_frames:
                    if fidx in all_ball_detections:
                        for d in all_ball_detections[fidx]:
                            d["source"] = "outlier"
                            static_ball_dets.setdefault(fidx, []).append(d)
                        all_ball_detections[fidx] = []
                    if fidx in pass1_anchors:
                        del pass1_anchors[fidx]
                print(f"  Post-filter: removed {len(outlier_frames)} trajectory outlier anchor(s)")

        print(f"  Post-filter: {len(pass1_anchors)} anchors remain (from {raw_anchor_count} raw)")

        # Save all detections (before trajectory filter) for diagnostic vis.
        # Includes pass1, pass2, and static-tagged detections.
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
        # Merge removed detections (static + outlier) back in for visualization
        for fidx, dets in static_ball_dets.items():
            diag_list = all_ball_dets_diag.setdefault(fidx, [])
            for d in dets:
                diag_list.append({
                    "center": list(d["center"]),
                    "bbox": list(d["bbox"]),
                    "confidence": d["confidence"],
                    "source": d.get("source", "static"),
                })

        # Filter: for each frame, keep only the detection closest to the
        # anchor-interpolated trajectory.  This prevents far-off false
        # positives (watermark, spare ball) from being fed to the tracker.
        if pass1_anchors:
            anchor_indices = sorted(pass1_anchors.keys())
            for fidx in list(all_ball_detections.keys()):
                dets = all_ball_detections[fidx]
                if not dets:
                    continue
                # Get expected position from anchors
                expected = _interpolate_ball_position(
                    fidx, anchor_indices, pass1_anchors, max_gap
                )
                if expected is None:
                    # No nearby anchor — discard all detections for this frame
                    all_ball_detections[fidx] = []
                    continue
                # Keep only the detection closest to expected position
                def _dist(d: dict) -> float:
                    c = d["center"]
                    return ((c[0] - expected[0]) ** 2 + (c[1] - expected[1]) ** 2) ** 0.5
                best_det = min(dets, key=_dist)
                # Only keep if reasonably close (within crop_size radius)
                if _dist(best_det) <= crop_size:
                    all_ball_detections[fidx] = [best_det]
                else:
                    all_ball_detections[fidx] = []

        # Build set of kept detection centers for tagging filtered ones
        kept_centers: set[tuple[float, float]] = set()
        for dets in all_ball_detections.values():
            for d in dets:
                kept_centers.add((round(d["center"][0], 2), round(d["center"][1], 2)))

        # Tag filtered detections in diagnostic data
        for fidx, dets in all_ball_dets_diag.items():
            for d in dets:
                key = (round(d["center"][0], 2), round(d["center"][1], 2))
                if d["source"] not in ("static", "outlier") and key not in kept_centers:
                    d["source"] = "filtered"

        print(f"  Total frames with ball detections: "
              f"{sum(1 for d in all_ball_detections.values() if d)}/{len(frame_indices)}")

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

        # Ball detection and tracking
        if ball_detector and ball_tracker:
            if two_pass_enabled:
                # Use pre-computed detections from two-pass scan
                ball_detections = all_ball_detections.get(frame_idx, [])
            else:
                ball_detections = ball_detector.detect(frame)
                ball_detections = ball_detector.filter_by_size(ball_detections)

            # Update ball tracker
            ball_tracks = ball_tracker.update(ball_detections)
            primary_ball = ball_tracker.get_primary_ball()

            if primary_ball and not primary_ball.get("predicted", False):
                # Compute ball world position
                ball_center = primary_ball["center"]
                time_sec = frame_idx / fps if fps > 0 else 0.0

                if ball_trajectory_3d and frame_idx in camera_poses:
                    # Feed observation to 3D trajectory estimator
                    ball_bbox = tuple(primary_ball["bbox"]) if "bbox" in primary_ball else None
                    ball_trajectory_3d.add_observation(
                        time_sec, tuple(ball_center), camera_poses[frame_idx],
                        bbox=ball_bbox,
                    )
                    traj_result = ball_trajectory_3d.estimate()
                    if traj_result:
                        if traj_result.get("rejected"):
                            # False positive detected — suppress this ball entirely
                            primary_ball = None
                        else:
                            primary_ball["pitch_position"] = traj_result["pitch_position"]
                            primary_ball["height"] = traj_result["height"]
                            primary_ball["position_3d"] = traj_result["position_3d"]
                            primary_ball["on_ground"] = traj_result["on_ground"]

                # Fallback to ground plane projection
                if primary_ball is not None and primary_ball.get("pitch_position") is None:
                    if frame_idx in camera_poses:
                        from .tracking.ball_trajectory import project_to_ground
                        ground = project_to_ground(tuple(ball_center), camera_poses[frame_idx])
                        if ground:
                            primary_ball["pitch_position"] = ground
                            primary_ball["height"] = 0.0
                            primary_ball["on_ground"] = True
                    elif H is not None:
                        pt_h = np.array([ball_center[0], ball_center[1], 1.0])
                        world_h = H @ pt_h
                        if abs(world_h[2]) > 1e-6:
                            primary_ball["pitch_position"] = [
                                float(world_h[0] / world_h[2]),
                                float(world_h[1] / world_h[2]),
                            ]
                            primary_ball["height"] = 0.0
                            primary_ball["on_ground"] = True

                if primary_ball is not None:
                    all_ball_tracks[frame_idx] = primary_ball
                    ball_trajectory_history.append(tuple(ball_center))
                    max_traj_len = ball_tracking_config.get("trajectory_length", 30)
                    if len(ball_trajectory_history) > max_traj_len:
                        ball_trajectory_history = ball_trajectory_history[-max_traj_len:]

        # Note: Team classification is deferred until all tracking is complete
        # This allows using trajectory-averaged features for better clustering

    prefetcher.shutdown()

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

    # Diagnostic visualization: ball detection two-pass results
    if two_pass_enabled and all_ball_dets_diag:
        _render_ball_detection_diag(
            video_path, sampler, all_ball_dets_diag,
            pass1_anchors, static_positions, output_dir,
        )

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
