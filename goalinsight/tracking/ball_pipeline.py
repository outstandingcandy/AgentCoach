"""Ball detection, trajectory filtering, and diagnostic visualization."""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm



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

    # Only one side available -- use nearest
    nearest = before_fidx if before_fidx is not None else after_fidx
    if nearest is not None:
        return tracked_positions[nearest]

    return None


def _filter_trajectories_field_space(
    trajectory_field_data: dict[int, list[tuple]],
    fps: float,
    config: dict,
    pitch_half_length: float = 52.5,
    pitch_half_width: float = 34.0,
) -> dict[int, list[tuple]]:
    """Filter trajectories based on physical constraints in field space.

    Each entry in trajectory_field_data is:
        track_id -> list of (frame_idx, (wx, wy), (px, py), confidence)

    Filters:
    1. Minimum length: discard short tracks
    2. Position bounds: discard if majority of points outside pitch + margin
    3. Speed: discard if too many inter-frame speeds exceed max
    4. Smoothness: discard if direction changes are too erratic

    Args:
        pitch_half_length: Half of the pitch length in meters.
        pitch_half_width: Half of the pitch width in meters.

    Returns:
        Filtered dict with invalid trajectories removed.
    """
    max_speed = config.get("max_speed_ms", 40.0)
    margin = config.get("position_margin", 8.0)
    min_len = config.get("min_trajectory_length", 3)
    smooth_thresh = config.get("smoothness_threshold", 0.6)
    speed_viol_ratio = config.get("speed_violation_ratio", 0.2)
    min_displacement = config.get("min_displacement", 1.0)

    pitch_half_length = pitch_half_length + margin
    pitch_half_width = pitch_half_width + margin

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

        # Filter 5: minimum displacement (reject near-stationary false positives)
        first_pos = points[0][1]
        last_pos = points[-1][1]
        displacement = ((last_pos[0] - first_pos[0]) ** 2 + (last_pos[1] - first_pos[1]) ** 2) ** 0.5
        displacement_ok = displacement >= min_displacement

        if bounds_ok and speed_ok and field_smooth_ok and displacement_ok:
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

        if px_smooth and px_speed_ok and displacement_ok and len(points) >= min_len:
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
            # Neighbors themselves are inconsistent -- can't judge the middle point
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
