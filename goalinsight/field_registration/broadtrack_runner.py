"""Stage 1 BroadTrack backend implementation.

BroadTrack-style calibration:
- 8-parameter camera model with Cauchy robust loss
- Joint keypoint + arc-length parameterized line curve constraints
- Centered image coordinates with simple radial distortion (k1 only)

Selected via config: field_registration.backend = "broadtrack"
"""

import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..utils.config import get_default_config, get_process_fps_from_config, FrameSampler
from ..utils.serialization import json_default as _json_default
from ..utils.pitch import get_pitch_template_points, project_pitch_to_image, _draw_topdown_pitch
from .shared_vis import draw_vis_keypoints, draw_vis_lines, draw_vis_calibration
from ._runner_base import (
    open_video,
    make_sampler,
    init_calibration_results,
    save_calibration_outputs,
    compute_calibration_stats,
    print_calibration_summary,
)


def run_stage1_broadtrack(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using BroadTrack-style calibration.

    Args:
        video_path: Path to input video.
        output_dir: Directory for output files.
        vis_dir: Directory for visualizations.
        config: Configuration dict.
        process_fps: Processing frame rate.

    Returns:
        Dict with calibration statistics.
    """
    from . import KeypointDetector
    from .pnlcalib import KeypointMapper, LineMapper
    from .pnlcalib.broadtrack_calibrator import BroadTrackCalibrator

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    bt_config = fr_config.get("broadtrack", {})

    # Initialize keypoint detector (reuse existing PnLCalib detector)
    logger.info("Stage 1 (BroadTrack): Initializing keypoint detector...")
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_config.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_config.get("keypoint_model_path"),
            "confidence_threshold": pnl_config.get("keypoint_threshold", 0.3434),
        }
    }
    kp_detector = KeypointDetector(kp_config)
    kp_detector.load_model()
    keypoint_mapper = KeypointMapper()

    # Line detector is optional — lines can be derived from keypoints
    use_line_model = bt_config.get("use_line_model", False)
    line_detector = None
    line_mapper = LineMapper()

    if use_line_model:
        from . import LineDetector
        logger.info("Stage 1 (BroadTrack): Initializing line detector...")
        line_config = {
            "backend": "pnlcalib",
            "pnlcalib": {
                "weights": pnl_config.get("line_weights", "SV_lines"),
                "confidence_threshold": pnl_config.get("line_threshold", 0.15),
            }
        }
        line_detector = LineDetector(line_config)
        line_detector.load_model()
    else:
        logger.info("Stage 1 (BroadTrack): Deriving lines from keypoints (no line model)")

    ransac_thresh = pnl_config.get("ransac_threshold", 30.0)
    pitch_template = get_pitch_template_points()

    # Open video
    video = open_video(video_path)
    cap = video.cap
    width, height = video.width, video.height
    sampler = make_sampler(video, process_fps)

    # Initialize BroadTrack calibrator
    calibrator = BroadTrackCalibrator(
        image_size=(width, height),
        cauchy_f_scale=bt_config.get("cauchy_f_scale", 5.0),
        line_sample_points=bt_config.get("line_sample_points", 20),
        line_weight=bt_config.get("line_weight", 1.0),
        focal_candidates=bt_config.get("focal_candidates"),
        max_nfev=bt_config.get("max_nfev", 500),
        ransac_threshold=ransac_thresh,
    )

    # Results storage
    calibration_results = init_calibration_results(video_path, video, process_fps)
    homographies = {}

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_line_dir = vis_dir / "lines"
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_line_dir, vis_calib_dir]:
        d.mkdir(exist_ok=True)

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1 (BroadTrack): Calibrating")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints (and lines if line model enabled)
        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        lines = line_detector.detect(frame) if line_detector is not None else []

        # Update calibrator with detections
        calibrator.update(keypoints, lines)

        # Run BroadTrack calibration
        result = calibrator.calibrate(
            keypoint_mapper, line_mapper if use_line_model else None,
            min_confidence=pnl_config.get("keypoint_threshold", 0.3434),
        )

        if result is not None:
            homographies[frame_idx] = result["homography"]
            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": result["num_keypoints"],
                "num_lines": result["num_lines"],
                "num_intersections": result["num_intersections"],
                "total_points": result["total_points"],
                "inliers": result["inliers"],
                "reprojection_error": result["final_error"],
            }
            calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}

        # Save visualizations periodically
        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            json_fname = f"frame_{frame_idx:05d}.json"
            cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))
            cv2.imwrite(str(vis_line_dir / fname), draw_vis_lines(frame, lines))
            # Draw calibration visualization
            vis = _draw_broadtrack_calibration(
                frame, keypoints, lines, result, pitch_template,
                keypoint_mapper, calibrator,
            )
            cv2.imwrite(str(vis_calib_dir / fname), vis)

            # Save detailed per-frame JSON
            frame_info = _build_frame_json(frame_idx, keypoints, lines, result)
            json_str = json.dumps(frame_info, indent=2, default=_json_default)
            with open(vis_calib_dir / json_fname, "w") as jf:
                jf.write(json_str)

    cap.release()

    save_calibration_outputs(output_dir, calibration_results, homographies)
    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
    )
    print_calibration_summary(stats, label="Stage 1 (BroadTrack)")
    return stats


def _build_frame_json(frame_idx, keypoints, lines, result):
    """Build detailed JSON dict for a single frame's calibration data."""
    info = {
        "frame_idx": int(frame_idx),
        "calibrated": result is not None,
        "detected_keypoints": [
            {
                "id": int(kp.get("id", -1)),
                "x": float(kp.get("x", 0)),
                "y": float(kp.get("y", 0)),
                "confidence": float(kp.get("confidence", 0)),
                "class_name": kp.get("class_name", ""),
            }
            for kp in keypoints
        ],
        "detected_lines": [
            {
                "id": int(ln.get("id", -1)),
                "x1": float(ln.get("x1", 0)),
                "y1": float(ln.get("y1", 0)),
                "x2": float(ln.get("x2", 0)),
                "y2": float(ln.get("y2", 0)),
                "confidence": float(ln.get("confidence", 0)),
                "class_name": ln.get("class_name", ""),
            }
            for ln in lines
        ],
    }

    if result is not None:
        info["reprojection_error"] = result.get("final_error", None)
        info["num_keypoints"] = result.get("num_keypoints", 0)
        info["num_lines"] = result.get("num_lines", 0)
        info["total_points"] = result.get("total_points", 0)
        info["inliers"] = result.get("inliers", 0)

        cam = result.get("camera_params", {})
        if cam:
            info["camera"] = {
                "focal_length": float(cam.get("focal_length", 0)),
                "K": cam["K"].tolist() if "K" in cam else None,
                "rvec": cam["rvec"].flatten().tolist() if "rvec" in cam else None,
                "tvec": cam["tvec"].flatten().tolist() if "tvec" in cam else None,
                "dist_coeffs": cam["dist_coeffs"].tolist() if "dist_coeffs" in cam else None,
            }

        if "keypoint_details" in result:
            info["keypoint_details"] = result["keypoint_details"]
        if "line_details" in result:
            info["line_details"] = result["line_details"]

    return info


def _draw_broadtrack_calibration(
    frame, keypoints, lines, result, pitch_template, keypoint_mapper, calibrator,
):
    """Draw BroadTrack calibration visualization: camera view + top-down pitch."""
    h, w = frame.shape[:2]
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if result is not None and result.get("camera_params"):
        camera_params = result["camera_params"]

        # Project pitch template onto image
        projected = project_pitch_to_image(
            result["homography"], pitch_template, camera_params
        )

        # Draw projected pitch lines
        for name, pts in projected.items():
            color = (0, 255, 255)
            for i in range(len(pts) - 1):
                if pts[i] is not None and pts[i + 1] is not None:
                    p1 = tuple(int(c) for c in pts[i])
                    p2 = tuple(int(c) for c in pts[i + 1])
                    margin = 200
                    p1_in = -margin < p1[0] < w + margin and -margin < p1[1] < h + margin
                    p2_in = -margin < p2[0] < w + margin and -margin < p2[1] < h + margin
                    if p1_in and p2_in:
                        cv2.line(vis, p1, p2, color, 2)

        # Draw detected keypoints (green=inlier, red=outlier)
        if "img_pts" in result and "inlier_mask" in result:
            for pt, is_inlier in zip(result["img_pts"], result["inlier_mask"]):
                x, y = int(pt[0]), int(pt[1])
                color = (0, 255, 0) if is_inlier else (0, 0, 255)
                cv2.circle(vis, (x, y), 5, color, -1)

        # Draw detected lines (cyan)
        for line in lines:
            if line.get("confidence", 0) >= 0.15:
                x1, y1 = int(line["x1"]), int(line["y1"])
                x2, y2 = int(line["x2"]), int(line["y2"])
                cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Header
        err = result.get("final_error", 0)
        n_in = result.get("inliers", 0)
        n_total = result.get("total_points", 0)
        n_lines = result.get("num_lines", 0)
        header = f"BT Err: {err:.1f}px | Inliers: {n_in}/{n_total} | Lines: {n_lines}"
    else:
        header = "BroadTrack: Calibration FAILED"

    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (0, 200, 0), 1)

    # Top-down pitch diagram - use a stub calibrator for compatibility
    class _Stub:
        line_intersections = []
        image_size = (w, h)
    pitch_h = h
    pitch_w = int(pitch_h * 1.5)
    pitch = _draw_topdown_pitch(pitch_h, pitch_w, result, keypoints, _Stub(), keypoint_mapper)

    # Combine side-by-side
    combined = np.hstack([vis, pitch])
    return combined
