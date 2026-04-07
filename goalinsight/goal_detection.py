"""Goal detection from ball tracking and camera calibration data.

Detects goals by analyzing ball 3D trajectory crossing the goal line
within the goal frame (width and height).
"""

import json
import logging
import argparse
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# FIFA fixed goal dimensions (do not change with pitch size)
GOAL_HALF_WIDTH = 3.66   # meters (7.32m total)
GOAL_HEIGHT = 2.44        # meters


def detect_goals(
    ball_tracks: dict,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 30.0,
    min_confidence: float = 0.15,
    camera_poses: dict | None = None,
) -> list[dict]:
    """Detect goal events from ball tracking data.

    Scans the ball's 3D trajectory for moments where it crosses a goal line
    within the goal frame (width ≤ 7.32m, height ≤ 2.44m).

    Args:
        ball_tracks: Dict keyed by frame index (str), each value has
            'pitch_position' [x, y], 'height', 'confidence', 'center', etc.
        pitch_length: Pitch length in meters (goal lines at ±pitch_length/2).
        pitch_width: Pitch width in meters.
        fps: Video frame rate for timestamp calculation.
        min_confidence: Minimum ball detection confidence to consider.
        camera_poses: Optional dict keyed by frame index with camera params
            (K, rvec, tvec, dist_coeffs) for crossbar projection validation.

    Returns:
        List of goal event dicts, each containing:
            - frame: Frame index of goal
            - timestamp: Time in seconds
            - goal_side: "left" (x < 0) or "right" (x > 0)
            - ball_position: [x, y, z] at crossing
            - ball_pixel: [u, v] pixel coordinates
            - confidence: Detection confidence
    """
    half_length = pitch_length / 2.0

    # Parse and sort ball observations
    observations = _parse_ball_tracks(ball_tracks, min_confidence)
    if len(observations) < 2:
        return []

    # Smooth trajectory to reduce noise
    observations = _smooth_trajectory(observations)

    # Detect goal-line crossings
    crossings = _detect_crossings(observations, half_length)

    # Validate each crossing
    goals = []
    for crossing in crossings:
        event = _validate_crossing(
            crossing, observations, half_length, fps, camera_poses
        )
        if event is not None:
            goals.append(event)

    # Deduplicate goals within 2 seconds of each other
    goals = _deduplicate_goals(goals, fps, min_gap_frames=int(2 * fps))

    return goals


def _parse_ball_tracks(ball_tracks: dict, min_confidence: float) -> list[dict]:
    """Parse ball_tracks.json into sorted list of observations."""
    obs = []
    for frame_str, data in ball_tracks.items():
        pp = data.get("pitch_position")
        if pp is None:
            continue
        conf = data.get("confidence", 0)
        if conf < min_confidence:
            continue
        obs.append({
            "frame": int(frame_str),
            "x": pp[0],
            "y": pp[1],
            "height": data.get("height", 0.0),
            "confidence": conf,
            "center": data.get("center"),
            "position_3d": data.get("position_3d"),
            "predicted": data.get("predicted", False),
        })
    obs.sort(key=lambda o: o["frame"])
    return obs


def _smooth_trajectory(observations: list[dict], window: int = 3) -> list[dict]:
    """Apply median filter to x, y positions to reduce tracking noise.

    Only smooths when we have enough points. Does not modify height
    (which is already fragile near goals).
    """
    n = len(observations)
    if n < window:
        return observations

    xs = np.array([o["x"] for o in observations])
    ys = np.array([o["y"] for o in observations])

    # Median filter with edge padding
    half_w = window // 2
    xs_smooth = np.copy(xs)
    ys_smooth = np.copy(ys)
    for i in range(half_w, n - half_w):
        xs_smooth[i] = np.median(xs[i - half_w : i + half_w + 1])
        ys_smooth[i] = np.median(ys[i - half_w : i + half_w + 1])

    for i, o in enumerate(observations):
        o["x_raw"] = o["x"]
        o["y_raw"] = o["y"]
        o["x"] = float(xs_smooth[i])
        o["y"] = float(ys_smooth[i])

    return observations


def _detect_crossings(
    observations: list[dict], half_length: float
) -> list[dict]:
    """Find frames where ball crosses a goal line.

    A crossing occurs when the ball's x-coordinate transitions from inside
    the pitch to beyond the goal line, moving toward that goal.
    """
    crossings = []
    goal_margin = 1.0  # must be within 1m of goal line before crossing

    for i in range(1, len(observations)):
        prev = observations[i - 1]
        curr = observations[i]

        # Skip if frames are too far apart (tracking gap > 1 second at 30fps)
        frame_gap = curr["frame"] - prev["frame"]
        if frame_gap > 30:
            continue

        # Speed sanity check
        dt = frame_gap / 30.0  # approximate, using 30fps
        dx = curr["x"] - prev["x"]
        dy = curr["y"] - prev["y"]
        speed = (dx**2 + dy**2) ** 0.5 / max(dt, 1e-6)
        if speed > 50.0:  # > 50 m/s is physically impossible
            continue

        # Check right goal (x = +half_length)
        if prev["x"] <= half_length and curr["x"] > half_length:
            if prev["x"] > half_length - 10:  # was approaching from within 10m
                # Interpolate crossing position
                t = (half_length - prev["x"]) / max(dx, 1e-6)
                t = np.clip(t, 0, 1)
                cross_y = prev["y"] + t * (curr["y"] - prev["y"])
                cross_h = prev["height"] + t * (curr["height"] - prev["height"])
                cross_frame = prev["frame"] + t * frame_gap
                crossings.append({
                    "frame": int(round(cross_frame)),
                    "goal_side": "right",
                    "cross_y": cross_y,
                    "cross_height": max(cross_h, 0),
                    "prev_obs": prev,
                    "curr_obs": curr,
                    "speed": speed,
                })

        # Check left goal (x = -half_length)
        if prev["x"] >= -half_length and curr["x"] < -half_length:
            if prev["x"] < -half_length + 10:  # approaching from within 10m
                t = (-half_length - prev["x"]) / min(dx, -1e-6)
                t = np.clip(t, 0, 1)
                cross_y = prev["y"] + t * (curr["y"] - prev["y"])
                cross_h = prev["height"] + t * (curr["height"] - prev["height"])
                cross_frame = prev["frame"] + t * frame_gap
                crossings.append({
                    "frame": int(round(cross_frame)),
                    "goal_side": "left",
                    "cross_y": cross_y,
                    "cross_height": max(cross_h, 0),
                    "prev_obs": prev,
                    "curr_obs": curr,
                    "speed": speed,
                })

    return crossings


def _validate_crossing(
    crossing: dict,
    observations: list[dict],
    half_length: float,
    fps: float,
    camera_poses: dict | None,
) -> dict | None:
    """Validate a goal-line crossing as an actual goal.

    Checks:
    1. Ball is within goal width (|y| <= 3.66m)
    2. Ball height is within goal height (<= 2.44m)
    3. Ball was approaching the goal (not bouncing back)
    4. Optional: pixel position vs projected crossbar validation
    """
    cross_y = crossing["cross_y"]
    cross_h = crossing["cross_height"]

    # Check goal frame bounds
    if abs(cross_y) > GOAL_HALF_WIDTH:
        logger.debug(
            f"Frame {crossing['frame']}: crossing rejected, "
            f"y={cross_y:.1f}m outside goal width ±{GOAL_HALF_WIDTH}m"
        )
        return None

    if cross_h > GOAL_HEIGHT + 0.5:  # small tolerance for estimation error
        logger.debug(
            f"Frame {crossing['frame']}: crossing rejected, "
            f"height={cross_h:.1f}m above goal {GOAL_HEIGHT}m"
        )
        return None

    # Check approach direction: need >=2 frames within 10m approaching the goal
    frame = crossing["frame"]
    goal_x = half_length if crossing["goal_side"] == "right" else -half_length
    sign = 1 if crossing["goal_side"] == "right" else -1

    approach_count = 0
    for obs in observations:
        if obs["frame"] > frame:
            break
        if obs["frame"] < frame - 30:  # look back ~1 second
            continue
        dist_to_goal = sign * (goal_x - obs["x"])
        if 0 < dist_to_goal < 10:
            approach_count += 1

    if approach_count < 2:
        logger.debug(
            f"Frame {crossing['frame']}: crossing rejected, "
            f"insufficient approach frames ({approach_count})"
        )
        return None

    # Optional: crossbar projection validation
    crossbar_valid = None
    if camera_poses is not None:
        crossbar_valid = _validate_with_crossbar(
            crossing, half_length, camera_poses
        )

    # Build goal event
    prev = crossing["prev_obs"]
    curr = crossing["curr_obs"]
    avg_conf = (prev["confidence"] + curr["confidence"]) / 2

    return {
        "frame": crossing["frame"],
        "timestamp": crossing["frame"] / fps,
        "goal_side": crossing["goal_side"],
        "ball_position": [
            float(goal_x),
            float(cross_y),
            float(cross_h),
        ],
        "ball_pixel": curr["center"],
        "confidence": float(avg_conf),
        "ball_speed_mps": float(crossing["speed"]),
        "crossbar_validation": crossbar_valid,
    }


def _validate_with_crossbar(
    crossing: dict,
    half_length: float,
    camera_poses: dict,
) -> dict | None:
    """Validate ball pixel is within projected goal frame in image space.

    Projects the 4 goal corners (3D) to image pixels using camera params,
    then checks if the ball pixel falls within the projected quadrilateral.
    """
    try:
        import cv2
    except ImportError:
        return None

    frame_str = str(crossing["frame"])
    # Find nearest calibrated frame
    pose_frames = sorted(camera_poses.keys(), key=lambda f: int(f))
    nearest = min(pose_frames, key=lambda f: abs(int(f) - crossing["frame"]))
    if abs(int(nearest) - crossing["frame"]) > 15:
        return None

    pose = camera_poses[nearest]
    if not pose.get("rvec"):
        return None

    rvec = np.array(pose["rvec"], dtype=np.float64)
    tvec = np.array(pose["tvec"], dtype=np.float64)
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)

    goal_x = half_length if crossing["goal_side"] == "right" else -half_length
    goal_3d = np.array([
        [goal_x, GOAL_HALF_WIDTH, 0],
        [goal_x, GOAL_HALF_WIDTH, -GOAL_HEIGHT],
        [goal_x, -GOAL_HALF_WIDTH, 0],
        [goal_x, -GOAL_HALF_WIDTH, -GOAL_HEIGHT],
    ], dtype=np.float64)

    img_pts, _ = cv2.projectPoints(goal_3d, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    ball_pixel = crossing["curr_obs"]["center"]
    if ball_pixel is None:
        return None

    bx, by = ball_pixel
    # Simple bbox check (goal frame is roughly rectangular in image)
    gx_min, gx_max = img_pts[:, 0].min(), img_pts[:, 0].max()
    gy_min, gy_max = img_pts[:, 1].min(), img_pts[:, 1].max()

    # Expand bbox slightly for tolerance
    margin = 20  # pixels
    inside = (gx_min - margin <= bx <= gx_max + margin and
              gy_min - margin <= by <= gy_max + margin)

    return {
        "ball_pixel": [float(bx), float(by)],
        "goal_bbox": [float(gx_min), float(gy_min), float(gx_max), float(gy_max)],
        "goal_corners_px": img_pts.tolist(),
        "inside_goal_frame": inside,
    }


def _deduplicate_goals(
    goals: list[dict], fps: float, min_gap_frames: int
) -> list[dict]:
    """Remove duplicate goal detections within min_gap_frames of each other."""
    if not goals:
        return []

    goals.sort(key=lambda g: g["frame"])
    deduped = [goals[0]]
    for g in goals[1:]:
        if g["frame"] - deduped[-1]["frame"] > min_gap_frames:
            deduped.append(g)
        elif g["confidence"] > deduped[-1]["confidence"]:
            deduped[-1] = g  # keep higher confidence detection

    return deduped


def detect_goals_from_output(
    output_dir: str | Path,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float | None = None,
    min_confidence: float = 0.15,
) -> list[dict]:
    """Detect goals from a pipeline output directory.

    Args:
        output_dir: Path to pipeline output (e.g., output/pipeline_007_2).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.
        fps: Video FPS. If None, read from calibration_metadata.json.

    Returns:
        List of goal events.
    """
    output_dir = Path(output_dir)

    # Load ball tracks (try new name first, fallback to legacy)
    ball_path = output_dir / "tracking" / "ball_tracks.json"
    if not ball_path.exists():
        ball_path = output_dir / "stage2" / "ball_tracks.json"
    if not ball_path.exists():
        logger.error(f"Ball tracks not found in {output_dir}/tracking/ or {output_dir}/stage2/")
        return []
    with open(ball_path) as f:
        ball_tracks = json.load(f)

    # Load camera poses (optional, for crossbar validation)
    camera_poses = None
    poses_path = output_dir / "field_registration" / "camera_poses.json"
    if not poses_path.exists():
        poses_path = output_dir / "stage1" / "camera_poses.json"
    if poses_path.exists():
        with open(poses_path) as f:
            camera_poses = json.load(f)

    # Read fps from metadata if not provided
    if fps is None:
        meta_path = output_dir / "field_registration" / "calibration_metadata.json"
        if not meta_path.exists():
            meta_path = output_dir / "stage1" / "calibration_metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            fps = meta.get("video_info", {}).get("fps", 30.0)
        else:
            fps = 30.0

    return detect_goals(
        ball_tracks=ball_tracks,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        fps=fps,
        min_confidence=min_confidence,
        camera_poses=camera_poses,
    )


def main():
    parser = argparse.ArgumentParser(description="Detect goals from pipeline output")
    parser.add_argument("output_dir", help="Pipeline output directory")
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    goals = detect_goals_from_output(
        output_dir=args.output_dir,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        fps=args.fps,
        min_confidence=args.min_confidence,
    )

    if not goals:
        print("No goals detected.")
        return

    print(f"\nDetected {len(goals)} goal(s):\n")
    for i, g in enumerate(goals, 1):
        ts = g["timestamp"]
        minutes = int(ts // 60)
        seconds = ts % 60
        print(f"  Goal {i}:")
        print(f"    Time:     {minutes}:{seconds:05.2f} (frame {g['frame']})")
        print(f"    Side:     {g['goal_side']} goal")
        pos = g["ball_position"]
        print(f"    Position: x={pos[0]:.1f}m, y={pos[1]:.1f}m, h={pos[2]:.2f}m")
        print(f"    Speed:    {g['ball_speed_mps']:.1f} m/s")
        print(f"    Confidence: {g['confidence']:.2f}")
        if g.get("crossbar_validation"):
            cv = g["crossbar_validation"]
            print(f"    Crossbar check: {'PASS' if cv['inside_goal_frame'] else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
