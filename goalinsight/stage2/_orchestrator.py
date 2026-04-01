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

from ..utils.config import (
    get_default_config,
    get_process_fps_from_config,
    FrameSampler,
    get_reid_extractor,
    get_team_classifier,
)
from ..utils.serialization import json_default as _json_default_impl, sanitize_for_json
from ..utils.prefetcher import FramePrefetcher
from ..tracking import (
    PlayerDetector,
    StrongSORTTracker,
    GoalkeeperDetector,
    BallDetector,
    BallTracker,
)
from ..tracking.ball_trajectory import BallTrajectory3D

from ._pitch_projection import (
    _PITCH_HALF_LENGTH,
    _PITCH_HALF_WIDTH,
    _undistort_and_project_to_pitch,
    _filter_by_pitch_undistorted,
    _interpolate_camera_poses,
)
from ._ball_pipeline import (
    _find_nearest_detection_center,
    _interpolate_tracked_position,
    _filter_trajectories_field_space,
    _remove_merge_outliers,
    _select_best_trajectory,
    _render_ball_detection_diag,
)
from ._feature_extraction import (
    _extract_jersey_color_hist,
    _extract_jersey_mean_saturation,
)
from ._visualization import (
    TEAM_COLORS,
    get_color_for_track,
    draw_topdown_pitch,
    draw_tracks,
    draw_ball_track,
)


_FramePrefetcher = FramePrefetcher  # backward compat alias


def _batch_reader_full(
    video_path,
    tasks: list[int],
    batch_size: int,
    out_queue,
) -> None:
    """Read full frames in batches and push to queue (sequential grab for speed)."""
    cap = cv2.VideoCapture(str(video_path))
    prev_fidx = -1
    batch_fidxs: list[int] = []
    batch_frames: list[np.ndarray] = []
    for fidx in tasks:
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
    import goalinsight.stage2._pitch_projection as _pp_mod
    _pp_mod._PITCH_HALF_LENGTH = pitch_length / 2
    _pp_mod._PITCH_HALF_WIDTH = pitch_width / 2

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

    sampler = FrameSampler(total_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

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
        import queue as _queue

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
            # Tracker lost ball -- interpolate crop center from nearest tracked frames
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

    # Pre-batch YOLO detection: read all frames + run batched GPU inference,
    # so the tracking loop only does ReID (small crops) on GPU.
    import queue as _queue2
    yolo_batch_size = config.get("detection", {}).get("batch_size", 32)
    _frame_indices_list = list(sampler)

    print(f"  Pre-detecting players: {len(_frame_indices_list)} frames, batch_size={yolo_batch_size}")
    _det_read_q: _queue2.Queue = _queue2.Queue(maxsize=3)
    _det_reader = threading.Thread(
        target=_batch_reader_full,
        args=(video_path, _frame_indices_list, yolo_batch_size, _det_read_q),
        daemon=True,
    )
    _det_reader.start()

    _precomputed_frames: dict[int, np.ndarray] = {}
    _precomputed_dets: dict[int, list] = {}
    _det_pbar = tqdm(total=len(_frame_indices_list), desc="  YOLO batch detect")
    while True:
        _det_item = _det_read_q.get()
        if _det_item is None:
            break
        _batch_fidxs, _batch_frames = _det_item
        _batch_dets_list = detector.detect_batch(_batch_frames)
        for _i, _fidx in enumerate(_batch_fidxs):
            _precomputed_frames[_fidx] = _batch_frames[_i]
            _precomputed_dets[_fidx] = _batch_dets_list[_i]
        _det_pbar.update(len(_batch_fidxs))
    _det_pbar.close()
    _det_reader.join()
    cap.release()

    print(f"  Pre-detected {len(_precomputed_dets)} frames")

    # Pre-compute filtered detections + ReID embeddings for all frames (batched GPU)
    print("  Pre-computing ReID embeddings...")
    _filtered_dets: dict[int, list] = {}
    _precomputed_embeds: dict[int, Any] = {}

    # Step 1: filter detections and collect crops (CPU)
    _all_crops: list[np.ndarray] = []
    _crop_map: list[tuple[int, int]] = []  # (frame_idx, crop_index_in_frame)
    _crops_per_frame: dict[int, int] = {}

    for frame_idx in _frame_indices_list:
        frame = _precomputed_frames.get(frame_idx)
        if frame is None:
            continue

        detections = _precomputed_dets.get(frame_idx, [])
        detections = detector.filter_by_size(
            detections, min_height=25, max_height=350,
            min_aspect_ratio=0.25, max_aspect_ratio=1.0,
        )

        H_world2img = homographies.get(frame_idx)
        H = None
        if H_world2img is not None:
            try:
                H = np.linalg.inv(H_world2img)
            except np.linalg.LinAlgError:
                H = None

        if frame_idx in camera_poses:
            detections = _filter_by_pitch_undistorted(detections, camera_poses[frame_idx], margin=5.0)
        elif H is not None:
            detections = detector.filter_by_pitch(detections, H, margin=5.0)

        _filtered_dets[frame_idx] = detections

        n_crops = 0
        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 > x1 and y2 > y1:
                _all_crops.append(frame[y1:y2, x1:x2])
            else:
                _all_crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
            _crop_map.append((frame_idx, n_crops))
            n_crops += 1
        _crops_per_frame[frame_idx] = n_crops

    # Step 2: batch ReID extraction (GPU-saturated)
    if _all_crops:
        print(f"  ReID batch extract: {len(_all_crops)} crops")
        all_embeddings = reid_extractor.extract(_all_crops)

        # Split embeddings back per frame
        offset = 0
        for frame_idx in _frame_indices_list:
            n = _crops_per_frame.get(frame_idx, 0)
            if n > 0:
                _precomputed_embeds[frame_idx] = all_embeddings[offset : offset + n]
                offset += n

    del _all_crops, _crop_map, _crops_per_frame
    print(f"  ReID done: {len(_precomputed_embeds)} frames with embeddings")

    # Tracking loop — now pure CPU
    for idx, frame_idx in enumerate(tqdm(_frame_indices_list, desc="Stage 2: Tracking")):
        frame = _precomputed_frames.get(frame_idx)
        if frame is None:
            break

        H_world2img = homographies.get(frame_idx)
        H = None
        if H_world2img is not None:
            try:
                H = np.linalg.inv(H_world2img)
            except np.linalg.LinAlgError:
                H = None

        detections = _filtered_dets.get(frame_idx, [])
        embeddings = _precomputed_embeds.get(frame_idx)

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

        # Ball detection and tracking -- collect ALL tracks for post-loop filtering
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

    # Free pre-computed detection cache
    del _precomputed_dets

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

            merge_max_dist = ball_config.get("field_filter", {}).get(
                "merge_max_distance", 15.0
            )  # meters — max gap between new frame and nearest existing frame

            for _, tid in scored_tids:
                for fidx, track_dict in ball_track_histories[tid]:
                    if track_dict.get("predicted", False):
                        continue
                    if fidx in all_ball_tracks:
                        continue  # Higher-scored track already owns this frame

                    # Spatial continuity check: reject if too far from nearest
                    # temporally-close existing frame (within ~1s)
                    field_pos = field_lookup.get(fidx)
                    if field_pos is not None and all_ball_tracks:
                        max_frame_gap = int(fps * 1.5)  # ~1.5 seconds
                        nearby = [
                            f for f in all_ball_tracks
                            if abs(f - fidx) <= max_frame_gap
                        ]
                        if nearby:
                            nearest_fidx = min(nearby, key=lambda f: abs(f - fidx))
                            nearest_pp = all_ball_tracks[nearest_fidx].get("pitch_position")
                            if nearest_pp is not None:
                                dx = field_pos[0] - nearest_pp[0]
                                dy = field_pos[1] - nearest_pp[1]
                                dist = (dx ** 2 + dy ** 2) ** 0.5
                                if dist > merge_max_dist:
                                    continue

                    entry = dict(track_dict)
                    if field_pos is not None:
                        entry["pitch_position"] = list(field_pos)
                        entry["height"] = 0.0
                        entry["on_ground"] = True
                    all_ball_tracks[fidx] = entry
                    ball_trajectory_history.append(tuple(track_dict["center"]))

            # Remove spatial outliers BEFORE 3D estimation (uses raw projection coords)
            n_outliers = _remove_merge_outliers(all_ball_tracks, fps)
            if n_outliers > 0:
                print(f"  Removed {n_outliers} merge outlier(s)")

            # Run BallTrajectory3D on all valid trajectories for 3D height estimation
            if ball_trajectory_3d:
                for _, tid in scored_tids:
                    ball_trajectory_3d.clear()
                    for fidx, track_dict in ball_track_histories[tid]:
                        if track_dict.get("predicted", False):
                            continue
                        if fidx not in camera_poses:
                            continue
                        if fidx not in all_ball_tracks:
                            continue  # removed by outlier filter
                        time_sec = fidx / fps if fps > 0 else 0.0
                        ball_bbox = tuple(track_dict["bbox"]) if "bbox" in track_dict else None
                        ball_trajectory_3d.add_observation(
                            time_sec, tuple(track_dict["center"]),
                            camera_poses[fidx], bbox=ball_bbox,
                        )
                        traj_result = ball_trajectory_3d.estimate()
                        if traj_result and not traj_result.get("rejected"):
                            # Only update frames owned by this tid (avoid cross-track overwrites)
                            if fidx in all_ball_tracks and all_ball_tracks[fidx].get("track_id") == tid:
                                all_ball_tracks[fidx]["pitch_position"] = traj_result["pitch_position"]
                                all_ball_tracks[fidx]["height"] = traj_result["height"]
                                all_ball_tracks[fidx]["position_3d"] = traj_result["position_3d"]
                                all_ball_tracks[fidx]["on_ground"] = traj_result["on_ground"]

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

    # Reuse pre-computed frames if available, otherwise read from video
    _have_cached_frames = bool(_precomputed_frames)
    if not _have_cached_frames:
        render_prefetcher = _FramePrefetcher(video_path, list(sampler), prefetch_size=4)

    track_history = {}  # Reset for visualization

    # Create per-frame output directories
    vis_frames_dir = output_dir / "frames"
    vis_frames_dir.mkdir(exist_ok=True)

    # Pre-sort ball track keys for fast slicing
    _ball_sorted_keys = sorted(all_ball_tracks.keys()) if ball_enabled else []

    # Background writer thread for file I/O
    import queue as _wq
    _write_queue: _wq.Queue = _wq.Queue(maxsize=32)

    def _writer_thread():
        while True:
            item = _write_queue.get()
            if item is None:
                break
            task_type = item[0]
            if task_type == "video":
                out.write(item[1])
            elif task_type == "image":
                cv2.imwrite(item[1], item[2])
            elif task_type == "json":
                with open(item[1], "w") as jf:
                    json.dump(item[2], jf, indent=2, default=_json_default)

    _writer = threading.Thread(target=_writer_thread, daemon=True)
    _writer.start()

    for idx, frame_idx in enumerate(tqdm(list(sampler), desc="Stage 2: Rendering")):
        if _have_cached_frames:
            frame = _precomputed_frames.get(frame_idx)
        else:
            frame = render_prefetcher.get(frame_idx)
        if frame is None:
            break

        # Get tracks for this frame with updated team assignments
        frame_tracks = all_tracks.get(frame_idx, [])

        # Draw visualization
        vis = draw_tracks(frame, frame_tracks, track_history, team_assignments)

        # Draw ball: raw detections + tracked ball
        ball_track_this = None
        ball_traj_world = None
        ball_dets_this = None
        if ball_enabled:
            ball_dets_this = ball_debug_log.get(frame_idx, {}).get("detections")
        traj = []
        traj_by_tid = {}
        if ball_enabled and frame_idx in all_ball_tracks:
            ball_track_this = all_ball_tracks[frame_idx]
            end = bisect.bisect_right(_ball_sorted_keys, frame_idx)
            keys_up_to = _ball_sorted_keys[:end]
            traj = [all_ball_tracks[fi]["center"] for fi in keys_up_to]
            traj = traj[-30:]
            # Build per-track_id trajectories
            for fi in keys_up_to:
                tid = all_ball_tracks[fi].get("track_id")
                if tid is not None:
                    traj_by_tid.setdefault(tid, []).append(all_ball_tracks[fi]["center"])
            # Keep only last 30 points per track
            traj_by_tid = {tid: pts[-30:] for tid, pts in traj_by_tid.items()}
            # Get world trajectory (for top-down view)
            ball_traj_world = [
                all_ball_tracks[fi]["pitch_position"]
                for fi in keys_up_to
                if all_ball_tracks[fi].get("pitch_position")
            ]
            ball_traj_world = ball_traj_world[-30:]
        vis = draw_ball_track(vis, ball_track_this, traj,
                              ball_detections=ball_dets_this,
                              ball_trajectories_by_tid=traj_by_tid)

        # Draw top-down pitch diagram
        topdown = draw_topdown_pitch(
            height, frame_tracks, team_assignments,
            ball_track=ball_track_this,
            ball_trajectory_world=ball_traj_world,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        combined = np.hstack([vis, topdown])

        # Offload all I/O to background writer
        fname = f"frame_{frame_idx:05d}"
        _write_queue.put(("video", combined))
        _write_queue.put(("image", str(vis_frames_dir / f"{fname}.jpg"), combined))
        _write_queue.put(("json", str(vis_frames_dir / f"{fname}.json"), {
            "frame_idx": int(frame_idx),
            "timestamp_sec": round(frame_idx / fps, 3),
            "num_tracks": len(frame_tracks),
            "tracks": frame_tracks,
            "ball": ball_track_this,
        }))

    # Wait for writer to finish
    _write_queue.put(None)
    _writer.join()

    if not _have_cached_frames:
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
        serializable_ball_tracks = {str(k): sanitize_for_json(v) for k, v in all_ball_tracks.items()}
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


_json_default = _json_default_impl


def main():
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage2_tracking")
    calibration_dir = Path("data/processed/stage1_field_registration")

    run_stage2(video_path, output_dir, calibration_dir)


if __name__ == "__main__":
    main()
