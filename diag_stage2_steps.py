#!/usr/bin/env python3
"""Diagnostic: visualize each step of Stage 2 processing per frame.

Outputs per-frame images showing:
  Step 1: Raw YOLOv8 detections
  Step 2: After size filtering
  Step 3: After pitch boundary filtering
  Step 4: ReID features (embedding norms)
  Step 5: Tracker output (track IDs)
  Step 6: Pitch position projection
  Step 7: Final team classification summary
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from goalinsight.utils.config import (
    get_default_config,
    get_process_fps_from_config,
    FrameSampler,
    get_reid_extractor,
    get_team_classifier,
    load_config,
)
from goalinsight.tracking import (
    PlayerDetector,
    StrongSORTTracker,
    GoalkeeperDetector,
    BallDetector,
    BallTracker,
)
from goalinsight.stage2 import (
    _undistort_and_project_to_pitch,
    _filter_by_pitch_undistorted,
)


def draw_detections(frame, detections, title, color=(0, 255, 0)):
    """Draw bounding boxes with a title."""
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        conf = det.get("confidence", 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.putText(vis, f"{title} ({len(detections)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def draw_tracks_step(frame, tracks, tracker_features, title):
    """Draw tracker output with track IDs and feature info."""
    vis = frame.copy()
    for track in tracks:
        tid = track["track_id"]
        x1, y1, x2, y2 = map(int, track["bbox"])
        color = (0, 200, 0)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"T{tid}"
        if tid in tracker_features and tracker_features[tid] is not None:
            feat = tracker_features[tid]
            norm = np.linalg.norm(feat)
            label += f" |f|={norm:.1f}"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(vis, f"{title} ({len(tracks)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis


def draw_pitch_positions(frame, tracks, camera_poses, frame_idx, homographies):
    """Draw pitch positions projected from foot points."""
    vis = frame.copy()
    H_world2img = homographies.get(frame_idx)
    H = None
    if H_world2img is not None:
        try:
            H = np.linalg.inv(H_world2img)
        except np.linalg.LinAlgError:
            pass

    positions = []
    for track in tracks:
        tid = track["track_id"]
        x1, y1, x2, y2 = map(int, track["bbox"])
        foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
        foot_y = track["bbox"][3]

        pitch_pos = None
        if frame_idx in camera_poses:
            pitch_pos = _undistort_and_project_to_pitch(
                np.array([[foot_x, foot_y]]), camera_poses[frame_idx]
            )
        elif H is not None:
            pt_h = np.array([foot_x, foot_y, 1.0])
            world_h = H @ pt_h
            if abs(world_h[2]) > 1e-6:
                pitch_pos = [float(world_h[0] / world_h[2]), float(world_h[1] / world_h[2])]

        color = (0, 200, 0) if pitch_pos else (0, 0, 255)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        if pitch_pos:
            label = f"T{tid} ({pitch_pos[0]:.1f},{pitch_pos[1]:.1f})"
            positions.append((tid, pitch_pos))
        else:
            label = f"T{tid} NO_POS"
        cv2.putText(vis, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.putText(vis, f"Pitch Positions ({len(positions)}/{len(tracks)})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    return vis, positions


def main():
    config = load_config("configs/clip_000_physical.yaml")
    config["video"]["process_fps"] = 5
    video_path = Path("data/raw_videos/football_sunday_output_000.mp4")
    output_dir = Path("output/diag_stage2_steps")
    output_dir.mkdir(parents=True, exist_ok=True)

    process_fps = get_process_fps_from_config(config)

    # Find latest stage1 output
    calibration_dir = None
    pipeline_dir = Path("output/pipeline_physical")
    if pipeline_dir.exists():
        runs = sorted(pipeline_dir.glob("run_*/stage1"))
        if runs:
            calibration_dir = runs[-1]
            print(f"Using calibration from: {calibration_dir}")

    # Load calibration
    homographies = {}
    camera_poses = {}
    if calibration_dir:
        if (calibration_dir / "camera_poses.pkl").exists():
            with open(calibration_dir / "camera_poses.pkl", "rb") as f:
                camera_poses = pickle.load(f)
            print(f"Loaded {len(camera_poses)} camera poses")
            for fidx, pose in camera_poses.items():
                R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
                K = np.array(pose["K"], dtype=np.float64)
                tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
                H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
                if abs(H[2, 2]) > 1e-10:
                    H = H / H[2, 2]
                homographies[fidx] = H

    # Initialize components
    detector = PlayerDetector({
        "model": "yolov8x", "confidence_threshold": 0.5,
        "iou_threshold": 0.45, "classes": [0], "imgsz": 1280,
    })
    detector.load_model()

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracker = StrongSORTTracker({
        "max_age": 50, "n_init": 3,
        "max_iou_distance": 0.7, "max_cosine_distance": 0.3,
        "feature_alpha": 0.9,
    })
    tracker.img_w = width
    tracker.img_h = height

    reid_extractor = get_reid_extractor(config)
    reid_extractor.load_model()

    team_classifier = get_team_classifier(config)

    max_duration_sec = 60
    max_frames = int(max_duration_sec * fps)
    effective_frames = min(total_frames, max_frames)
    sampler = FrameSampler(effective_frames, fps, process_fps)

    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames")

    # Accumulate features/positions for final team classification
    all_track_features = {}
    all_track_positions = {}

    for idx, frame_idx in enumerate(sampler):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        print(f"\n=== Frame {frame_idx} (#{idx}) ===")
        frame_dir = output_dir / f"frame_{frame_idx:05d}"
        frame_dir.mkdir(exist_ok=True)

        # Step 1: Raw detections
        raw_detections = detector.detect(frame)
        vis1 = draw_detections(frame, raw_detections, "Step1: Raw YOLOv8")
        cv2.imwrite(str(frame_dir / "step1_raw_detections.jpg"), vis1)
        print(f"  Step 1 - Raw detections: {len(raw_detections)}")

        # Step 2: Size filtering
        size_filtered = detector.filter_by_size(
            raw_detections, min_height=25, max_height=350,
            min_aspect_ratio=0.25, max_aspect_ratio=1.0,
        )
        vis2 = draw_detections(frame, size_filtered, "Step2: Size Filtered", (255, 200, 0))
        cv2.imwrite(str(frame_dir / "step2_size_filtered.jpg"), vis2)
        removed_size = len(raw_detections) - len(size_filtered)
        print(f"  Step 2 - Size filtered: {len(size_filtered)} (removed {removed_size})")

        # Step 3: Pitch boundary filtering
        if frame_idx in camera_poses:
            pitch_filtered = _filter_by_pitch_undistorted(
                size_filtered, camera_poses[frame_idx], margin=5.0
            )
            filter_method = "undistort"
        else:
            H_world2img = homographies.get(frame_idx)
            if H_world2img is not None:
                try:
                    H_inv = np.linalg.inv(H_world2img)
                    pitch_filtered = detector.filter_by_pitch(size_filtered, H_inv, margin=5.0)
                    filter_method = "homography"
                except np.linalg.LinAlgError:
                    pitch_filtered = size_filtered
                    filter_method = "none (singular H)"
            else:
                pitch_filtered = size_filtered
                filter_method = "none (no calibration)"

        vis3 = draw_detections(frame, pitch_filtered, f"Step3: Pitch Filter ({filter_method})", (0, 200, 255))
        cv2.imwrite(str(frame_dir / "step3_pitch_filtered.jpg"), vis3)
        removed_pitch = len(size_filtered) - len(pitch_filtered)
        print(f"  Step 3 - Pitch filtered: {len(pitch_filtered)} (removed {removed_pitch}, method={filter_method})")

        # Step 4: ReID feature extraction
        embeddings = None
        if pitch_filtered:
            crops = []
            for det in pitch_filtered:
                x1, y1, x2, y2 = map(int, det["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    crops.append(frame[y1:y2, x1:x2])
                else:
                    crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            embeddings = reid_extractor.extract(crops)

        vis4 = frame.copy()
        for i, det in enumerate(pitch_filtered):
            x1, y1, x2, y2 = map(int, det["bbox"])
            cv2.rectangle(vis4, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if embeddings is not None and i < len(embeddings):
                norm = np.linalg.norm(embeddings[i])
                cv2.putText(vis4, f"|e|={norm:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        emb_dim = embeddings.shape[1] if embeddings is not None and len(embeddings) > 0 else 0
        cv2.putText(vis4, f"Step4: ReID Features (dim={emb_dim})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imwrite(str(frame_dir / "step4_reid_features.jpg"), vis4)
        print(f"  Step 4 - ReID: {len(pitch_filtered)} crops, embedding dim={emb_dim}")

        # Step 5: Tracker update
        tracks = tracker.update(pitch_filtered, embeddings)
        tracker_features = tracker.get_track_features()
        vis5 = draw_tracks_step(frame, tracks, tracker_features, "Step5: Tracker Output")
        cv2.imwrite(str(frame_dir / "step5_tracker_output.jpg"), vis5)
        print(f"  Step 5 - Tracks: {len(tracks)}, IDs: {[t['track_id'] for t in tracks]}")

        # Step 6: Pitch position projection
        vis6, positions = draw_pitch_positions(frame, tracks, camera_poses, frame_idx, homographies)
        cv2.imwrite(str(frame_dir / "step6_pitch_positions.jpg"), vis6)
        print(f"  Step 6 - Positions: {len(positions)}/{len(tracks)}")
        for tid, pos in positions:
            print(f"    T{tid}: ({pos[0]:.1f}, {pos[1]:.1f})")

        # Accumulate for team classification
        for track in tracks:
            tid = track["track_id"]
            if tid not in all_track_features:
                all_track_features[tid] = []
                all_track_positions[tid] = []
            if tid in tracker_features and tracker_features[tid] is not None:
                all_track_features[tid].append(tracker_features[tid])
            # Get pitch pos
            foot_x = (track["bbox"][0] + track["bbox"][2]) / 2
            foot_y = track["bbox"][3]
            if frame_idx in camera_poses:
                pp = _undistort_and_project_to_pitch(
                    np.array([[foot_x, foot_y]]), camera_poses[frame_idx]
                )
                if pp:
                    all_track_positions[tid].append(pp)

    cap.release()

    # Step 7: Team classification diagnosis
    print("\n" + "=" * 60)
    print("Step 7: Team Classification Diagnosis")
    print("=" * 60)

    mean_features = {}
    for tid, feats in all_track_features.items():
        if feats:
            mean_features[tid] = np.mean(feats, axis=0)
            print(f"  T{tid}: {len(feats)} features, mean norm={np.linalg.norm(mean_features[tid]):.3f}")
        else:
            print(f"  T{tid}: NO features")

    mean_positions = {}
    for tid, poss in all_track_positions.items():
        if poss:
            mean_positions[tid] = np.mean(poss, axis=0).tolist()
            print(f"  T{tid}: mean pos=({mean_positions[tid][0]:.1f}, {mean_positions[tid][1]:.1f})")

    print(f"\n  Total tracks with features: {len(mean_features)}")
    print(f"  Total tracks with positions: {len(mean_positions)}")
    print(f"  min_samples_per_team (from classifier): {team_classifier.min_samples}")
    print(f"  Threshold for fit: {team_classifier.min_samples * 2} tracks needed")
    print(f"  Have {len(mean_features)} tracks -> {'PASS' if len(mean_features) >= team_classifier.min_samples * 2 else 'FAIL'}")

    if len(mean_features) >= team_classifier.min_samples * 2:
        assignments = team_classifier.fit(mean_features, mean_positions)
        print(f"\n  Team assignments: {assignments}")
    else:
        print(f"\n  SKIPPED: need >= {team_classifier.min_samples * 2} tracks, have {len(mean_features)}")

    # Also try with lower threshold
    print("\n  --- Trying with min_samples=2 ---")
    from goalinsight.tracking.team.kmeans_classifier import KMeansTeamClassifier
    tc2 = KMeansTeamClassifier({"min_samples_per_team": 2})
    if len(mean_features) >= 4:
        assignments2 = tc2.fit(mean_features, mean_positions)
        print(f"  Team assignments (min=2): {assignments2}")

    # Feature similarity matrix
    if mean_features:
        tids = sorted(mean_features.keys())
        feats = np.array([mean_features[t] for t in tids])
        feats_norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-8)
        sim = feats_norm @ feats_norm.T
        print(f"\n  Cosine similarity matrix (tracks {tids}):")
        header = "       " + "  ".join(f"T{t:>3}" for t in tids)
        print(header)
        for i, tid in enumerate(tids):
            row = f"  T{tid:>3}: " + "  ".join(f"{sim[i, j]:.2f}" for j in range(len(tids)))
            print(row)

    print(f"\nDiagnostic output saved to: {output_dir}")


if __name__ == "__main__":
    main()
