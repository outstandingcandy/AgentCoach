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
import logging
import pickle
import threading
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..utils.config import (
    get_default_config,
    get_process_fps_from_config,
    FrameSampler,
)
from ..utils.factories import get_reid_extractor, get_team_classifier, get_jersey_recognizer
from ..utils.serialization import json_default as _json_default_impl, sanitize_for_json
from ..utils.prefetcher import FramePrefetcher
from .detector import PlayerDetector
from .strongsort_tracker import StrongSORTTracker
from .team.kmeans_classifier import GoalkeeperDetector
from .ball_detector import BallDetector
from .ball_tracker import BallTracker
from .unified_detector import UnifiedDetector
from .ball_trajectory import BallTrajectory3D

from .pitch_projection import (
    _undistort_and_project_to_pitch,
    _filter_by_pitch_undistorted,
    _interpolate_camera_poses,
)
from .ball_pipeline import (
    _find_nearest_detection_center,
    _interpolate_tracked_position,
    _filter_trajectories_field_space,
    _remove_merge_outliers,
    _select_best_trajectory,
    _render_ball_detection_diag,
)
from .feature_extraction import (
    _extract_jersey_color_hist,
    _extract_jersey_mean_saturation,
)
from .tracking_visualization import (
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


def run_tracking(
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
            logger.info("Loading Stage 1 physical camera poses...")
            with open(calibration_dir / "camera_poses.pkl", "rb") as f:
                camera_poses = pickle.load(f)
            logger.info(f"  Loaded {len(camera_poses)} camera poses")
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
            logger.info("Loading Stage 1 calibration results...")
            with open(calibration_dir / "homographies.pkl", "rb") as f:
                homographies = pickle.load(f)
            logger.info(f"  Loaded {len(homographies)} homographies")

    # Initialize detectors — unified mode fuses player + ball into one YOLO pass
    det_config = config.get("detection", {})
    ball_config = config.get("ball_detection", {})
    ball_tracking_config = config.get("ball_tracking", {})
    ball_enabled = ball_config.get("enabled", True)
    unified_config = config.get("unified_detection", {})

    # Determine if unified detection can be used
    unified_enabled = unified_config.get("enabled", True) and ball_enabled
    if unified_enabled and ball_config.get("use_sahi", False):
        unified_enabled = False  # SAHI needs a different inference path
    if unified_enabled:
        det_model = det_config.get("model_path") or det_config.get("model", "yolov8x")
        ball_model = ball_config.get("model_path") or ball_config.get("model", "yolov8x")
        if det_model != ball_model:
            unified_enabled = False  # Different models, can't fuse

    unified_det = None
    if unified_enabled:
        logger.info("Stage 2: Initializing unified detector (player + ball)...")
        unified_det = UnifiedDetector({
            "model": unified_config.get("model", det_config.get("model", "yolov8x")),
            "model_path": unified_config.get("model_path", det_config.get("model_path")),
            "confidence_threshold": unified_config.get("confidence_threshold", 0.3),
            "iou_threshold": unified_config.get("iou_threshold", 0.45),
            "imgsz": unified_config.get("imgsz", 1920),
        })
        unified_det.load_model()
        # PlayerDetector still needed for filter methods but does not load its own model
        detector = PlayerDetector({
            "model": "yolov8x",
            "confidence_threshold": det_config.get("confidence_threshold", 0.5),
            "iou_threshold": 0.45,
            "classes": [0],
            "imgsz": 1280,
        })
    else:
        logger.info("Stage 2: Initializing YOLOv8 detector...")
        detector = PlayerDetector({
            "model": "yolov8x",
            "confidence_threshold": det_config.get("confidence_threshold", 0.5),
            "iou_threshold": 0.45,
            "classes": [0],
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
    logger.info("Stage 2: Initializing StrongSORT tracker...")
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
    logger.info(f"Stage 2: Initializing ReID extractor ({reid_backend})...")
    reid_extractor = get_reid_extractor(config)
    reid_extractor.load_model()

    # Initialize team classifier via factory function and goalkeeper detector
    tc_backend = config.get("team_classification", {}).get("backend", "kmeans")
    logger.info(f"Stage 2: Initializing team classifier ({tc_backend})...")
    team_classifier = get_team_classifier(config)
    fr_config = config.get("field_registration", {})
    phys_config = fr_config.get("physical", {})
    pitch_length = phys_config.get("pitch_length", 105.0)
    pitch_width = phys_config.get("pitch_width", 68.0)
    gk_detector = GoalkeeperDetector(pitch_length=pitch_length, pitch_width=pitch_width)

    # Initialize jersey recognizer (if enabled)
    jr_config = config.get("jersey_recognition", {})
    jersey_recognition_enabled = jr_config.get("enabled", False)
    jersey_recognizer = None
    jersey_aggregator = None
    if jersey_recognition_enabled:
        jr_backend = jr_config.get("backend", "qwen_vl")
        logger.info(f"Stage 2: Initializing jersey recognizer ({jr_backend})...")
        jersey_recognizer = get_jersey_recognizer(config)
        from ..jersey.qwen_recognizer import JerseyNumberAggregator
        jersey_aggregator = JerseyNumberAggregator(
            window_size=jr_config.get("aggregation_window", 10)
        )

    pitch_half_length = pitch_length / 2
    pitch_half_width = pitch_width / 2

    # Initialize ball detection and tracking
    ball_detector = None
    ball_tracker = None
    ball_trajectory_3d = None
    if ball_enabled:
        logger.info("Stage 2: Initializing ball detector and tracker...")
        ball_detector = BallDetector(ball_config)
        if unified_det is not None:
            # Share model weights — no second GPU load
            ball_detector.model = unified_det.model
        else:
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
    logger.info(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    logger.info(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

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
        logger.info(f"  Interpolated camera poses: {len(camera_poses)} (from stage1 calibration)")

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
    jersey_detections = {}  # track_id -> list of (jersey_number, confidence)
    tentative_buffer = {}  # track_id -> list of (frame_idx, track_dict) for backfill

    # Ball tracking storage
    all_ball_tracks = {}  # frame_idx -> ball track dict (final selected trajectory)
    all_ball_track_candidates = {}  # frame_idx -> list of all track dicts
    ball_track_histories = {}  # track_id -> list of (frame_idx, track_dict)
    ball_debug_log: dict[int, dict] = {}  # frame_idx -> debug info
    all_ball_dets_diag: dict[int, list[dict]] = {}  # diagnostic visualization data
    ball_trajectory_history = []  # For visualization

    # === Detection: unified (fused) or legacy (separate) ===
    all_ball_detections: dict[int, list] = {}
    two_pass_enabled = ball_config.get("two_pass", False) and ball_detector is not None
    import queue as _queue

    # Helper for ball pass 2 crop+enlarge (used by both unified and legacy paths)
    def _batch_reader_crop(
        video_path: str | Path,
        tasks: list[tuple[int, tuple[float, float]]],
        batch_size: int,
        det: object,
        crop_sz: int,
        enlarge_to: int,
        out_queue: _queue.Queue,
    ) -> None:
        """Read frames, crop+enlarge, and push batches to queue."""
        _cap = cv2.VideoCapture(str(video_path))
        prev_fidx = -1
        batch_fidxs: list[int] = []
        batch_images: list[np.ndarray] = []
        batch_metas: list[dict] = []
        for fidx, center in tasks:
            gap = fidx - prev_fidx - 1
            if prev_fidx >= 0 and 0 <= gap <= 8:
                for _ in range(gap):
                    _cap.grab()
            else:
                _cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = _cap.read()
            prev_fidx = fidx
            if not ret:
                continue
            img, meta = det.prepare_crop(frame, center, crop_sz, enlarge_to)
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
        _cap.release()

    _frame_indices_list = list(sampler)
    _precomputed_dets: dict[int, list] = {}

    if unified_det is not None:
        # ---- Fused detection pass: player + ball in one YOLO call ----
        fused_batch_size = unified_config.get("batch_size", 8)
        player_conf_thresh = unified_config.get(
            "player_confidence_threshold",
            det_config.get("confidence_threshold", 0.5),
        )
        num_batches = (len(_frame_indices_list) + fused_batch_size - 1) // fused_batch_size

        logger.info(f"Fused detection pass: {len(_frame_indices_list)} frames, "
                    f"batch_size={fused_batch_size}, imgsz={unified_det.imgsz}")

        read_q: _queue.Queue = _queue.Queue(maxsize=2)
        reader = threading.Thread(
            target=_batch_reader_full,
            args=(video_path, _frame_indices_list, fused_batch_size, read_q),
            daemon=True,
        )
        reader.start()

        pbar = tqdm(total=num_batches, desc="  Fused detect")
        while True:
            item = read_q.get()
            if item is None:
                break
            batch_fidxs, batch_frames = item

            batch_all_dets = unified_det.detect_batch(batch_frames)

            for i, fidx in enumerate(batch_fidxs):
                players, balls = UnifiedDetector.split_by_class(batch_all_dets[i])

                # Player pipeline: apply conf threshold post-hoc
                players = [d for d in players if d["confidence"] >= player_conf_thresh]
                _precomputed_dets[fidx] = players

                # Ball pipeline: apply size filter
                if ball_detector is not None:
                    balls = ball_detector.filter_by_size(balls)
                    for d in balls:
                        d["source"] = "pass1"
                    all_ball_detections[fidx] = balls

            pbar.update(1)
        pbar.close()
        reader.join()

        logger.info(f"  Pre-detected {len(_precomputed_dets)} frames (player + ball)")
        if ball_detector is not None:
            pass1_det_frames = sum(1 for d in all_ball_detections.values() if d)
            logger.info(f"  Ball pass 1: {pass1_det_frames} frames with detections")
            # Unified pass pre-computes ball detections, so treat as two-pass
            two_pass_enabled = True

    else:
        # ---- Legacy: separate ball pass 1 + player pre-detection ----
        if two_pass_enabled:
            frame_indices = list(sampler)
            logger.info("Ball detection pass 1: scanning all frames...")
            pass1_batch_size = ball_config.get("pass1_batch_size", 8)
            num_batches = (len(frame_indices) + pass1_batch_size - 1) // pass1_batch_size

            read_q_ball: _queue.Queue = _queue.Queue(maxsize=2)
            reader_ball = threading.Thread(
                target=_batch_reader_full,
                args=(video_path, frame_indices, pass1_batch_size, read_q_ball),
                daemon=True,
            )
            reader_ball.start()

            pbar = tqdm(total=num_batches, desc="  Ball pass 1")
            while True:
                item = read_q_ball.get()
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
            reader_ball.join()

            pass1_det_frames = sum(1 for d in all_ball_detections.values() if d)
            logger.info(f"  Pass 1: {pass1_det_frames} frames with detections out of {len(frame_indices)}")

        # Legacy player pre-detection
        logger.info("Stage 2: Processing frames...")
        yolo_batch_size = det_config.get("batch_size", 32)
        logger.info(f"  Pre-detecting players: {len(_frame_indices_list)} frames, batch_size={yolo_batch_size}")
        _det_read_q: _queue.Queue = _queue.Queue(maxsize=3)
        _det_reader = threading.Thread(
            target=_batch_reader_full,
            args=(video_path, _frame_indices_list, yolo_batch_size, _det_read_q),
            daemon=True,
        )
        _det_reader.start()

        _det_pbar = tqdm(total=len(_frame_indices_list), desc="  YOLO batch detect")
        while True:
            _det_item = _det_read_q.get()
            if _det_item is None:
                break
            _batch_fidxs, _batch_frames = _det_item
            _batch_dets_list = detector.detect_batch(_batch_frames)
            for _i, _fidx in enumerate(_batch_fidxs):
                _precomputed_dets[_fidx] = _batch_dets_list[_i]
            _det_pbar.update(len(_batch_fidxs))
        _det_pbar.close()
        _det_reader.join()
        logger.info(f"  Pre-detected {len(_precomputed_dets)} frames")

    cap.release()

    # Ball pass 2: crop+enlarge on frames where pass 1 missed the ball
    if two_pass_enabled and ball_detector is not None:
        crop_size = ball_config.get("crop_size", 300)
        crop_enlarge_to = ball_config.get("crop_enlarge_to", 640)
        max_gap = ball_config.get("max_interpolation_gap", 30)

        prelim_tracker = BallTracker(ball_tracking_config)
        prelim_positions: dict[int, tuple[float, float]] = {}

        for fidx in sorted(_frame_indices_list):
            dets = all_ball_detections.get(fidx, [])
            tracks = prelim_tracker.update(dets)
            for t in tracks:
                if not t.get("predicted", False):
                    prelim_positions[fidx] = tuple(t["center"])
                    break  # use first active track

        pass2_tasks: list[tuple[int, tuple[float, float]]] = []
        for fidx in _frame_indices_list:
            if fidx in prelim_positions:
                continue
            center = _interpolate_tracked_position(fidx, prelim_positions, max_gap)
            if center is not None:
                pass2_tasks.append((fidx, center))

        if pass2_tasks:
            logger.info(f"Ball detection pass 2: crop+enlarge on {len(pass2_tasks)} frames...")
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

            logger.info(f"  Pass 2: found ball in {pass2_count} additional frames")

    # Save ball diagnostic data (all detections from both passes)
    if all_ball_detections:
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
        logger.info(f"  Total frames with ball detections: {total_det_frames}/{len(_frame_indices_list)}")

    # Process frames
    logger.info("Stage 2: Processing frames...")
    # Team classification will be done AFTER all tracking is complete
    # This ensures we use trajectory-averaged features for robust clustering

    # Crop extraction + feature computation.
    # When fused detection was used, reuse _precomputed_dets (skip redundant YOLO).
    # Frames are NOT cached (saves ~40GB for long videos); rendering re-reads via FramePrefetcher.
    import queue as _queue2
    _has_precomputed = len(_precomputed_dets) > 0
    yolo_batch_size = config.get("detection", {}).get("batch_size", 32)
    _frame_indices_list = list(sampler)

    if _has_precomputed:
        logger.info(f"  Extracting crops from {len(_frame_indices_list)} frames "
                    f"(reusing {len(_precomputed_dets)} precomputed detections)")
    else:
        logger.info(f"  Pre-detecting players: {len(_frame_indices_list)} frames, batch_size={yolo_batch_size}")
    _det_read_q: _queue2.Queue = _queue2.Queue(maxsize=3)
    _det_reader = threading.Thread(
        target=_batch_reader_full,
        args=(video_path, _frame_indices_list, yolo_batch_size, _det_read_q),
        daemon=True,
    )
    _det_reader.start()

    _filtered_dets: dict[int, list] = {}
    _all_crops: list[np.ndarray] = []
    _crops_per_frame: dict[int, int] = {}
    _det_color_hists: dict[int, list] = {}   # frame_idx -> per-detection color hists
    _det_saturations: dict[int, list] = {}   # frame_idx -> per-detection saturations
    _det_jersey_numbers: dict[int, list] = {}  # frame_idx -> per-detection (number, conf)
    _precomputed_ball_dets: dict[int, list] = {}  # non-two-pass ball detections

    _det_pbar = tqdm(total=len(_frame_indices_list),
                     desc="  Crop extract" if _has_precomputed else "  YOLO batch detect")
    while True:
        _det_item = _det_read_q.get()
        if _det_item is None:
            break
        _batch_fidxs, _batch_frames = _det_item

        # Run YOLO only when we don't have precomputed detections
        _batch_dets_list = None
        if not _has_precomputed:
            _batch_dets_list = detector.detect_batch(_batch_frames)

        for _i, _fidx in enumerate(_batch_fidxs):
            _frame = _batch_frames[_i]

            # Get raw detections: from precomputed (fused pass) or fresh YOLO
            if _has_precomputed:
                raw_dets = _precomputed_dets.get(_fidx, [])
            else:
                raw_dets = _batch_dets_list[_i]

            # Filter detections by size and pitch
            detections = detector.filter_by_size(
                raw_dets, min_height=25, max_height=350,
                min_aspect_ratio=0.25, max_aspect_ratio=1.0,
            )
            _H_world2img = homographies.get(_fidx)
            _H_inv = None
            if _H_world2img is not None:
                try:
                    _H_inv = np.linalg.inv(_H_world2img)
                except np.linalg.LinAlgError:
                    pass
            if _fidx in camera_poses:
                detections = _filter_by_pitch_undistorted(
                    detections, camera_poses[_fidx], margin=5.0,
                    pitch_half_length=pitch_half_length,
                    pitch_half_width=pitch_half_width,
                )
            elif _H_inv is not None:
                detections = detector.filter_by_pitch(detections, _H_inv, margin=5.0)
            _filtered_dets[_fidx] = detections

            # Extract ReID crops + jersey color features from each detection
            n_crops = 0
            frame_hists = []
            frame_sats = []
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 > x1 and y2 > y1:
                    crop = _frame[y1:y2, x1:x2]
                    _all_crops.append(crop)
                    frame_hists.append(_extract_jersey_color_hist(crop))
                    frame_sats.append(_extract_jersey_mean_saturation(crop))
                else:
                    _all_crops.append(np.zeros((64, 32, 3), dtype=np.uint8))
                    frame_hists.append(None)
                    frame_sats.append(None)
                n_crops += 1
            _crops_per_frame[_fidx] = n_crops
            _det_color_hists[_fidx] = frame_hists
            _det_saturations[_fidx] = frame_sats

            # Jersey number recognition (batched per frame)
            if jersey_recognizer and n_crops > 0:
                frame_crops = _all_crops[-n_crops:]
                jr_results = jersey_recognizer.recognize_batch(frame_crops)
                _det_jersey_numbers[_fidx] = jr_results
            elif jersey_recognizer:
                _det_jersey_numbers[_fidx] = []

            # Ball detection (non-two-pass mode)
            if ball_detector and not two_pass_enabled:
                _ball_dets = ball_detector.detect(_frame)
                _ball_dets = ball_detector.filter_by_size(_ball_dets)
                _precomputed_ball_dets[_fidx] = _ball_dets

        _det_pbar.update(len(_batch_fidxs))
    _det_pbar.close()
    _det_reader.join()
    # Free precomputed detections (already filtered and stored in _filtered_dets)
    del _precomputed_dets

    logger.info(f"  Pre-detected {len(_filtered_dets)} frames")

    # Batch ReID extraction (GPU-saturated)
    _precomputed_embeds: dict[int, Any] = {}
    if _all_crops:
        reid_chunk_size = 2048
        n_total_crops = len(_all_crops)
        logger.info(f"  ReID streaming extract: {n_total_crops} crops (chunks of {reid_chunk_size})")
        embed_chunks = []
        for chunk_start in range(0, n_total_crops, reid_chunk_size):
            chunk = _all_crops[chunk_start : chunk_start + reid_chunk_size]
            embed_chunks.append(reid_extractor.extract(chunk))
        all_embeddings = np.concatenate(embed_chunks, axis=0)
        del embed_chunks

        offset = 0
        for frame_idx in _frame_indices_list:
            n = _crops_per_frame.get(frame_idx, 0)
            if n > 0:
                _precomputed_embeds[frame_idx] = all_embeddings[offset : offset + n]
                offset += n

    del _all_crops, _crops_per_frame
    logger.info(f"  ReID done: {len(_precomputed_embeds)} frames with embeddings")

    # Helper: match a track bbox to its source detection by IoU
    def _match_track_to_det(track_bbox, dets):
        best_idx, best_iou = -1, 0.3
        tx1, ty1, tx2, ty2 = track_bbox
        for i, det in enumerate(dets):
            dx1, dy1, dx2, dy2 = det["bbox"]
            ix1, iy1 = max(tx1, dx1), max(ty1, dy1)
            ix2, iy2 = min(tx2, dx2), min(ty2, dy2)
            if ix2 > ix1 and iy2 > iy1:
                inter = (ix2 - ix1) * (iy2 - iy1)
                union = (tx2 - tx1) * (ty2 - ty1) + (dx2 - dx1) * (dy2 - dy1) - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
        return best_idx

    # Tracking loop — pure CPU, no frame data needed
    for idx, frame_idx in enumerate(tqdm(_frame_indices_list, desc="Stage 2: Tracking")):

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

            # Use pre-computed jersey color features (extracted during YOLO batch phase)
            _det_idx = _match_track_to_det(track["bbox"], detections)
            if _det_idx >= 0:
                _hists = _det_color_hists.get(frame_idx, [])
                _sats = _det_saturations.get(frame_idx, [])
                if _det_idx < len(_hists) and _hists[_det_idx] is not None:
                    track_color_hists[track_id].append(_hists[_det_idx])
                if _det_idx < len(_sats) and _sats[_det_idx] is not None:
                    track_saturations.setdefault(track_id, []).append(_sats[_det_idx])
                # Accumulate jersey number predictions per track
                if jersey_aggregator:
                    _jrs = _det_jersey_numbers.get(frame_idx, [])
                    if _det_idx < len(_jrs):
                        jnum, jconf = _jrs[_det_idx]
                        jersey_aggregator.add_prediction(track_id, jnum, jconf)
                        jersey_detections.setdefault(track_id, []).append(
                            {"number": jnum, "confidence": jconf}
                        )

            if pitch_pos:
                track_positions[track_id].append(pitch_pos)

            # Determine role
            team = team_assignments.get(track_id, "unknown")
            role = "player"
            if team == "referee":
                role = "referee"
            elif pitch_pos:
                role = gk_detector.classify_role(pitch_pos, team)

            # Get current jersey number consensus for this track
            jersey_number = None
            jersey_conf = 0.0
            if jersey_aggregator:
                jersey_number, jersey_conf = jersey_aggregator.get_consensus(track_id)

            track_info = {
                "track_id": track_id,
                "bbox": track["bbox"],
                "confidence": track.get("confidence", 1.0),
                "pitch_position": pitch_pos,
                "team": team,
                "role": role,
                "jersey_number": jersey_number,
                "jersey_confidence": round(jersey_conf, 3),
            }
            frame_tracks.append(track_info)

        all_tracks[frame_idx] = frame_tracks

        # Ball detection and tracking -- collect ALL tracks for post-loop filtering
        if ball_detector and ball_tracker:
            if two_pass_enabled:
                ball_detections = all_ball_detections.get(frame_idx, [])
            else:
                ball_detections = _precomputed_ball_dets.get(frame_idx, [])

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

    # Free pre-computed detection/feature caches
    del _det_color_hists, _det_saturations, _det_jersey_numbers, _precomputed_ball_dets
    del _filtered_dets, _precomputed_embeds, all_ball_detections

    # Save ball tracking debug log and free memory
    if ball_debug_log:
        debug_path = output_dir / "ball_debug_log.json"
        with open(debug_path, "w") as f:
            json.dump({str(k): v for k, v in sorted(ball_debug_log.items())}, f, indent=1)
        logger.info(f"  Ball debug log saved to {debug_path}")
    ball_debug_log.clear()

    # === Field-space trajectory projection and filtering ===
    if ball_detector and ball_tracker and ball_track_histories:
        logger.info("Projecting ball trajectories to field space...")
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

        logger.info(f"  Total trajectories: {len(trajectory_field_data)}")
        for tid, pts in trajectory_field_data.items():
            logger.info(f"    Track {tid}: {len(pts)} projected points "
                        f"(frames {pts[0][0]}-{pts[-1][0]})")

        # Apply field-space filtering
        field_filter_config = ball_config.get("field_filter", {})
        if field_filter_config.get("enabled", True):
            filtered_trajectories = _filter_trajectories_field_space(
                trajectory_field_data, fps, field_filter_config,
                pitch_half_length=pitch_half_length,
                pitch_half_width=pitch_half_width,
            )
            n_rejected = len(trajectory_field_data) - len(filtered_trajectories)
            logger.info(f"  Field filter: {len(filtered_trajectories)} trajectories pass, "
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
                "merge_max_distance", 40.0
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
                logger.info(f"  Removed {n_outliers} merge outlier(s)")

            # Run BallTrajectory3D on all valid trajectories for 3D height estimation
            # Uses batch fit_track() API: collects all observations per track,
            # segments at kicks, classifies ground/airborne, fits one parabola
            # per airborne segment with ground-contact anchors at boundaries.
            if ball_trajectory_3d:
                for _, tid in scored_tids:
                    # Collect all valid observations for this track
                    track_obs = []
                    for fidx, track_dict in ball_track_histories[tid]:
                        if track_dict.get("predicted", False):
                            continue
                        if fidx not in camera_poses:
                            continue
                        if fidx not in all_ball_tracks:
                            continue  # removed by outlier filter
                        track_obs.append((fidx, tuple(track_dict["center"]), camera_poses[fidx]))

                    if not track_obs:
                        continue

                    # Batch fit: segments at kicks, classifies ground/airborne
                    traj_results = ball_trajectory_3d.fit_track(track_obs, fps)

                    for fidx, traj_result in traj_results.items():
                        if fidx in all_ball_tracks and all_ball_tracks[fidx].get("track_id") == tid:
                            all_ball_tracks[fidx]["pitch_position"] = traj_result["pitch_position"]
                            all_ball_tracks[fidx]["height"] = traj_result["height"]
                            all_ball_tracks[fidx]["position_3d"] = traj_result["position_3d"]
                            all_ball_tracks[fidx]["on_ground"] = traj_result["on_ground"]

            logger.info(f"  Ball tracked in {len(all_ball_tracks)}/{len(_frame_indices_list)} frames "
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
            logger.info("  No valid ball trajectory found after filtering")

        # Render diagnostic visualization
        if all_ball_dets_diag:
            _render_ball_detection_diag(
                video_path, sampler, all_ball_dets_diag, output_dir,
            )

    # Free ball processing intermediates before rendering
    del ball_track_histories, all_ball_track_candidates, all_ball_dets_diag

    team_assignments, jersey_assignments = _classify_teams(
        all_tracks=all_tracks,
        track_color_hists=track_color_hists,
        track_positions=track_positions,
        track_saturations=track_saturations,
        team_classifier=team_classifier,
        gk_detector=gk_detector,
        jersey_aggregator=jersey_aggregator,
        jersey_detections=jersey_detections,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    if config.get("output", {}).get("save_visualizations", True):
        _render_tracking_video(
            video_path=video_path,
            output_dir=output_dir,
            sampler=sampler,
            all_tracks=all_tracks,
            team_assignments=team_assignments,
            all_ball_tracks=all_ball_tracks,
            ball_debug_log=ball_debug_log,
            ball_enabled=ball_enabled,
            out=out,
            fps=fps,
            height=height,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
    else:
        out.release()
        logger.info("Skipping tracking visualization (save_visualizations=false)")

    return _save_tracking_outputs(
        output_dir=output_dir,
        all_tracks=all_tracks,
        team_assignments=team_assignments,
        jersey_assignments=jersey_assignments,
        track_features=track_features,
        all_ball_tracks=all_ball_tracks,
        ball_enabled=ball_enabled,
        total_frames=total_frames,
        sampler=sampler,
    )


_json_default = _json_default_impl


# ---------------------------------------------------------------------------
# Extracted sub-functions for run_tracking
# ---------------------------------------------------------------------------


def _classify_teams(
    *,
    all_tracks: dict,
    track_color_hists: dict,
    track_positions: dict,
    track_saturations: dict,
    team_classifier,
    gk_detector,
    jersey_aggregator,
    jersey_detections: dict,
    pitch_length: float,
    pitch_width: float,
) -> tuple[dict, dict]:
    """Team classification, role refinement, off-field filtering, jersey voting.

    Returns:
        (team_assignments, jersey_assignments)
    """
    logger.info("Finalizing team assignments using jersey color histograms...")

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

    logger.info(f"  Tracks with color features: {len(mean_color_features)}")
    logger.info(f"  Tracks with positions: {len(mean_positions)}")

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

    team_assignments = {}
    if len(mean_color_features) >= 6:
        team_assignments = team_classifier.fit(
            mean_color_features, mean_positions,
            track_bbox_heights=median_bbox_heights,
            track_frame_counts=track_frame_counts,
            track_mean_saturations=mean_saturations,
        )
        logger.info(f"  Team assignments: {len(team_assignments)} tracks classified")
    else:
        logger.warning("Not enough tracks for team classification")

    # Role refinement: linesman + goalkeeper detection by position
    goalkeeper_tracks = set()
    if mean_positions:
        logger.info("  Refining roles by position...")
        team_assignments, goalkeeper_tracks = gk_detector.refine_roles(
            team_assignments, mean_positions, all_positions=track_positions
        )

    # Filter out off-field tracks (substitutes, coaches, spectators)
    half_l = pitch_length / 2
    half_w = pitch_width / 2
    off_field_margin = 0.5
    off_field_tracks = set()
    for tid, pos in mean_positions.items():
        if abs(pos[0]) > half_l + off_field_margin or abs(pos[1]) > half_w + off_field_margin:
            off_field_tracks.add(tid)
    if off_field_tracks:
        logger.info(f"  Off-field tracks removed: {sorted(off_field_tracks)}")
        for tid in off_field_tracks:
            team_assignments.pop(tid, None)

    # Compute final jersey number assignments via voting
    jersey_assignments = {}
    if jersey_aggregator and jersey_detections:
        logger.info("  Computing jersey number consensus...")
        for tid in jersey_detections:
            jnum, jconf = jersey_aggregator.get_consensus(tid)
            if jnum is not None:
                jersey_assignments[tid] = {
                    "number": jnum,
                    "confidence": round(jconf, 3),
                }
        logger.info(f"  Jersey numbers identified: {len(jersey_assignments)} tracks")

    # Update all tracks with final team assignments, roles, and jersey numbers
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
            jr = jersey_assignments.get(tid)
            track["jersey_number"] = jr["number"] if jr else None
            track["jersey_confidence"] = jr["confidence"] if jr else 0.0

    return team_assignments, jersey_assignments


def _render_tracking_video(
    *,
    video_path: Path,
    output_dir: Path,
    sampler,
    all_tracks: dict,
    team_assignments: dict,
    all_ball_tracks: dict,
    ball_debug_log: dict,
    ball_enabled: bool,
    out: cv2.VideoWriter,
    fps: float,
    height: int,
    pitch_length: float,
    pitch_width: float,
) -> None:
    """Render the tracking visualization video with team + ball overlays."""
    logger.info("Generating visualization with final team assignments...")

    render_prefetcher = _FramePrefetcher(video_path, list(sampler), prefetch_size=4)
    track_history = {}

    vis_frames_dir = output_dir / "frames"
    vis_frames_dir.mkdir(exist_ok=True)

    _ball_sorted_keys = sorted(all_ball_tracks.keys()) if ball_enabled else []

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
        frame = render_prefetcher.get(frame_idx)
        if frame is None:
            break

        frame_tracks = all_tracks.get(frame_idx, [])
        vis = draw_tracks(frame, frame_tracks, track_history, team_assignments)

        # Draw ball
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
            for fi in keys_up_to:
                tid = all_ball_tracks[fi].get("track_id")
                if tid is not None:
                    traj_by_tid.setdefault(tid, []).append(all_ball_tracks[fi]["center"])
            traj_by_tid = {tid: pts[-30:] for tid, pts in traj_by_tid.items()}
            ball_traj_world = [
                all_ball_tracks[fi]["pitch_position"]
                for fi in keys_up_to
                if all_ball_tracks[fi].get("pitch_position")
            ]
            ball_traj_world = ball_traj_world[-30:]
        vis = draw_ball_track(vis, ball_track_this, traj,
                              ball_detections=ball_dets_this,
                              ball_trajectories_by_tid=traj_by_tid)

        topdown = draw_topdown_pitch(
            height, frame_tracks, team_assignments,
            ball_track=ball_track_this,
            ball_trajectory_world=ball_traj_world,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
        )
        combined = np.hstack([vis, topdown])

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

    _write_queue.put(None)
    _writer.join()

    render_prefetcher.shutdown()
    out.release()


def _save_tracking_outputs(
    *,
    output_dir: Path,
    all_tracks: dict,
    team_assignments: dict,
    jersey_assignments: dict,
    track_features: dict,
    all_ball_tracks: dict,
    ball_enabled: bool,
    total_frames: int,
    sampler,
) -> dict:
    """Write tracking output files and compute statistics."""
    with open(output_dir / "tracks.json", "w") as f:
        json.dump({str(k): v for k, v in all_tracks.items()}, f)

    with open(output_dir / "team_assignments.json", "w") as f:
        json.dump(team_assignments, f, indent=2)

    if jersey_assignments:
        with open(output_dir / "jersey_assignments.json", "w") as f:
            json.dump({str(k): v for k, v in jersey_assignments.items()}, f, indent=2)
        logger.info(f"  Saved jersey assignments for {len(jersey_assignments)} tracks")

    serializable_features = {
        str(tid): [f.tolist() if isinstance(f, np.ndarray) else f for f in feats]
        for tid, feats in track_features.items()
    }
    with open(output_dir / "track_features.json", "w") as f:
        json.dump(serializable_features, f)

    if ball_enabled and all_ball_tracks:
        serializable_ball_tracks = {str(k): sanitize_for_json(v) for k, v in all_ball_tracks.items()}
        with open(output_dir / "ball_tracks.json", "w") as f:
            json.dump(serializable_ball_tracks, f, indent=2)
        logger.info(f"  Saved {len(all_ball_tracks)} ball detections")

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

    logger.info(f"Stage 2 Complete:")
    logger.info(f"  Processed frames: {len(sampler)}")
    logger.info(f"  Unique tracks: {len(unique_tracks)}")
    logger.info(f"  Team distribution: {team_counts}")
    logger.info(f"  Output: {output_dir}")

    return stats


def main():
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage2_tracking")
    calibration_dir = Path("data/processed/stage1_field_registration")

    run_tracking(video_path, output_dir, calibration_dir)


if __name__ == "__main__":
    main()
