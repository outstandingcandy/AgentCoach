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

import json
import pickle
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
) -> list[dict]:
    """Filter detections by pitch boundary using undistorted projection.

    Args:
        detections: List of detection dicts with 'bbox' key.
        pose: Physical camera pose dict.
        margin: Extra meters beyond pitch boundary to allow.

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

    # Extract foot points from all detections
    foot_pts = []
    for det in detections:
        bbox = det["bbox"]
        foot_x = (bbox[0] + bbox[2]) / 2
        foot_y = bbox[3]
        foot_pts.append([foot_x, foot_y])

    foot_pts = np.array(foot_pts, dtype=np.float64).reshape(-1, 1, 2)
    foot_undist = cv2.undistortPoints(foot_pts, K, dist, P=K).reshape(-1, 2)

    # Project to world and filter by pitch boundary
    filtered = []
    x_lim = _PITCH_HALF_LENGTH + margin
    y_lim = _PITCH_HALF_WIDTH + margin

    for i, det in enumerate(detections):
        pt = foot_undist[i]
        ph = H_inv @ np.array([pt[0], pt[1], 1.0])
        if abs(ph[2]) < 1e-6:
            continue
        wx, wy = ph[0] / ph[2], ph[1] / ph[2]
        if -x_lim <= wx <= x_lim and -y_lim <= wy <= y_lim:
            filtered.append(det)

    return filtered


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

        # Draw trajectory (shorter for cleaner visualization)
        center = ((x1 + x2) // 2, y2)
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(center)
        if len(track_history[track_id]) > 15:  # Shorter trajectory
            track_history[track_id] = track_history[track_id][-15:]

        for i in range(1, len(track_history[track_id])):
            alpha = i / len(track_history[track_id])  # Fade effect
            cv2.line(vis, track_history[track_id][i-1], track_history[track_id][i], color, 2)

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

    # Draw trajectory with fade effect
    if ball_trajectory and len(ball_trajectory) > 1:
        for i in range(1, len(ball_trajectory)):
            alpha = i / len(ball_trajectory)
            color = (int(128 + 127 * alpha), int(200 - 35 * alpha), 255)
            pt1 = (int(ball_trajectory[i-1][0]), int(ball_trajectory[i-1][1]))
            pt2 = (int(ball_trajectory[i][0]), int(ball_trajectory[i][1]))
            thickness = max(1, int(3 * alpha))
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
            scale = 4.0
            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)
            cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.25)

    # Draw label
    conf = ball_track.get("confidence", 0.0)
    status = ball_track.get("status", "unknown")
    label = f"Ball {conf:.2f}"
    if status == "confirmed":
        label += " [OK]"
    cv2.putText(vis, label, (cx + 15, cy + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEAM_COLORS["ball"], 1)

    return vis


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
    process_fps = get_process_fps_from_config(config)

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
    tracker = StrongSORTTracker({
        "max_age": 50,
        "n_init": 3,
        "max_iou_distance": 0.7,
        "max_cosine_distance": 0.3,  # Tighter cosine distance
        "feature_alpha": 0.9,
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
    gk_detector = GoalkeeperDetector()

    # Initialize ball detection and tracking
    ball_config = config.get("ball_detection", {})
    ball_tracking_config = config.get("ball_tracking", {})
    ball_enabled = ball_config.get("enabled", True)

    ball_detector = None
    ball_tracker = None
    if ball_enabled:
        print("Stage 2: Initializing ball detector and tracker...")
        ball_detector = BallDetector(ball_config)
        ball_detector.load_model()
        ball_tracker = BallTracker(ball_tracking_config)

    # Limit to 1 minute of video for faster testing
    max_duration_sec = 60
    max_frames = int(max_duration_sec * fps)
    effective_frames = min(total_frames, max_frames)

    sampler = FrameSampler(effective_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames ({max_duration_sec}s) at {process_fps or fps} fps")

    # Output video
    output_fps = process_fps if process_fps and process_fps < fps else fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_dir / "tracking.mp4"), fourcc, output_fps, (width, height))

    # Storage
    all_tracks = {}
    track_features = {}  # track_id -> list of features
    track_positions = {}  # track_id -> list of positions
    track_history = {}  # For visualization
    team_assignments = {}  # track_id -> team

    # Ball tracking storage
    all_ball_tracks = {}  # frame_idx -> ball track dict
    ball_trajectory_history = []  # For visualization

    # Process frames
    print("\nStage 2: Processing frames...")
    # Team classification will be done AFTER all tracking is complete
    # This ensures we use trajectory-averaged features for robust clustering

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 2: Tracking")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
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

        # Update tracker
        tracks = tracker.update(detections, embeddings)

        # Store track info
        frame_tracks = []
        for i, track in enumerate(tracks):
            track_id = track["track_id"]

            # Compute pitch position
            pitch_pos = None
            if frame_idx in camera_poses:
                # Physical camera path: undistort pixel coords before projection
                foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
                foot_y = track["bbox"][3]
                pitch_pos = _undistort_and_project_to_pitch(
                    np.array([[foot_x, foot_y]]), camera_poses[frame_idx]
                )
            elif H is not None:
                # Legacy homography path (no undistortion)
                foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
                foot_y = track["bbox"][3]
                pt_h = np.array([foot_x, foot_y, 1.0])
                world_h = H @ pt_h
                if abs(world_h[2]) > 1e-6:
                    pitch_pos = [float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2])]

            # Store features and positions for team classification
            if track_id not in track_features:
                track_features[track_id] = []
                track_positions[track_id] = []

            # Get feature from tracker (keep all features for trajectory averaging)
            tracker_features = tracker.get_track_features()
            if track_id in tracker_features and tracker_features[track_id] is not None:
                track_features[track_id].append(tracker_features[track_id])

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
            ball_detections = ball_detector.detect(frame)
            ball_detections = ball_detector.filter_by_size(ball_detections)

            # Filter by pitch if calibration available
            if frame_idx in camera_poses:
                ball_detections = _filter_by_pitch_undistorted(
                    ball_detections, camera_poses[frame_idx], margin=5.0
                )
            elif H is not None:
                ball_detections = ball_detector.filter_by_pitch(ball_detections, H)

            # Update ball tracker
            ball_tracks = ball_tracker.update(ball_detections)
            primary_ball = ball_tracker.get_primary_ball()

            if primary_ball:
                all_ball_tracks[frame_idx] = primary_ball
                ball_trajectory_history.append(tuple(primary_ball["center"]))
                # Limit trajectory history length
                max_traj_len = ball_tracking_config.get("trajectory_length", 30)
                if len(ball_trajectory_history) > max_traj_len:
                    ball_trajectory_history = ball_trajectory_history[-max_traj_len:]

        # Note: Team classification is deferred until all tracking is complete
        # This allows using trajectory-averaged features for better clustering

    cap.release()

    # Final team classification using trajectory-averaged features
    print("\nFinalizing team assignments using trajectory-averaged features...")
    print(f"  Computing mean features for {len(track_features)} tracks...")

    # Compute trajectory-averaged ReID features
    mean_features = {}
    for tid, feats in track_features.items():
        if feats:
            mean_features[tid] = np.mean(feats, axis=0)

    # Compute mean pitch positions
    mean_positions = {}
    for tid, poss in track_positions.items():
        if poss:
            mean_positions[tid] = np.mean(poss, axis=0).tolist()

    print(f"  Tracks with features: {len(mean_features)}")
    print(f"  Tracks with positions: {len(mean_positions)}")

    if len(mean_features) >= 6:
        team_assignments = team_classifier.fit(mean_features, mean_positions)
        print(f"  Team assignments: {len(team_assignments)} tracks classified")
    else:
        print("  Warning: Not enough tracks for team classification")

    # Update all tracks with final team assignments
    for frame_idx in all_tracks:
        for track in all_tracks[frame_idx]:
            track["team"] = team_assignments.get(track["track_id"], "unknown")

    # Generate visualization with final team assignments
    print("\nGenerating visualization with final team assignments...")
    cap = cv2.VideoCapture(str(video_path))
    track_history = {}  # Reset for visualization

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 2: Rendering")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Get tracks for this frame with updated team assignments
        frame_tracks = all_tracks.get(frame_idx, [])

        # Draw visualization
        vis = draw_tracks(frame, frame_tracks, track_history, team_assignments)

        # Draw ball if available
        if ball_enabled and frame_idx in all_ball_tracks:
            ball_track = all_ball_tracks[frame_idx]
            # Get trajectory up to this frame
            traj = [all_ball_tracks[fi]["center"] for fi in sorted(all_ball_tracks.keys()) if fi <= frame_idx]
            traj = traj[-30:]  # Last 30 frames
            vis = draw_ball_track(vis, ball_track, traj)

        out.write(vis)

        # Save sample frames
        if idx % max(1, len(sampler) // 50) == 0:
            cv2.imwrite(str(output_dir / f"frame_{frame_idx:05d}.jpg"), vis)

    cap.release()
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
        serializable_ball_tracks = {str(k): v for k, v in all_ball_tracks.items()}
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


def main():
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage2_tracking")
    calibration_dir = Path("data/processed/stage1_field_registration")

    run_stage2(video_path, output_dir, calibration_dir)


if __name__ == "__main__":
    main()
