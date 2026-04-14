"""Stage 1 PnLCalib and NBJW backend implementations."""

import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

from ..utils.config import get_default_config, get_process_fps_from_config, FrameSampler
from ..utils.pitch import get_pitch_template_points, project_pitch_to_image
from .shared_vis import (
    draw_vis_keypoints,
    draw_vis_lines,
    draw_vis_intersections,
    draw_vis_calibration,
)
from ._runner_base import (
    open_video,
    make_sampler,
    init_calibration_results,
    save_calibration_outputs,
    compute_calibration_stats,
    print_calibration_summary,
)


def _run_stage1_pnlcalib(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using PnLCalib backend."""
    from . import KeypointDetector, LineDetector
    from .pnlcalib import (
        FramebyFrameCalib,
        KeypointMapper,
        LineMapper,
    )

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})

    # Initialize detectors
    logger.info("Stage 1: Initializing keypoint detector (HRNet/PnLCalib)...")
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_config.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_config.get("keypoint_model_path"),  # Custom fine-tuned model
            "confidence_threshold": pnl_config.get("keypoint_threshold", 0.3434),
        }
    }
    kp_detector = KeypointDetector(kp_config)
    kp_detector.load_model()
    keypoint_mapper = KeypointMapper()

    # Line detection is disabled by default (use_lines=false in config).
    # When disabled, calibration relies purely on keypoints.
    use_lines = pnl_config.get("use_lines", False)
    line_detector = None
    line_mapper = LineMapper()
    if use_lines:
        logger.info("Stage 1: Initializing line detector (HRNet/PnLCalib)...")
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
        logger.info("Stage 1: Line detection disabled (keypoint-only mode)")
    ransac_thresh = pnl_config.get("ransac_threshold", 30.0)
    calib_method = pnl_config.get("calibration_method", "iterative_pnp")

    pitch_template = get_pitch_template_points()

    # Open video
    video = open_video(video_path)
    cap = video.cap
    sampler = make_sampler(video, process_fps)
    width, height = video.width, video.height

    # Results storage
    calibration_results = init_calibration_results(video_path, video, process_fps)
    homographies = {}

    # Create calibrator once (image_size and alpha don't change per frame)
    calibrator = FramebyFrameCalib(image_size=(width, height), alpha=0.7)

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_line_dir = vis_dir / "lines" if use_lines else None
    vis_inter_dir = vis_dir / "intersections" if use_lines else None
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_line_dir, vis_inter_dir, vis_calib_dir]:
        if d is not None:
            d.mkdir(exist_ok=True)

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)  # ~100 visualizations

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1: Calibrating")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints (and lines if enabled)
        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        lines = line_detector.detect(frame) if line_detector is not None else []
        calibrator.update(keypoints, lines)

        # Run calibration
        result = calibrator.calibrate(
            keypoint_mapper, line_mapper,
            min_confidence=0.3, use_line_refinement=use_lines,
            ransac_threshold=ransac_thresh,
            method=calib_method,
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
            cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))
            if vis_line_dir is not None:
                cv2.imwrite(str(vis_line_dir / fname), draw_vis_lines(frame, lines))
            if vis_inter_dir is not None:
                cv2.imwrite(str(vis_inter_dir / fname), draw_vis_intersections(frame, keypoints, lines, calibrator))
            cv2.imwrite(str(vis_calib_dir / fname), draw_vis_calibration(frame, keypoints, lines, calibrator, result, pitch_template, keypoint_mapper))

    cap.release()

    save_calibration_outputs(output_dir, calibration_results, homographies)
    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
    )
    print_calibration_summary(stats, label="Stage 1")
    return stats


def _run_stage1_nbjw(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using NBJW backend."""
    from .nbjw import NbjwCalibrator

    fr_config = config.get("field_registration", {})
    nbjw_config = fr_config.get("nbjw", {})
    nbjw_config["device"] = config.get("device", "cuda")

    # Initialize NBJW calibrator
    logger.info("Stage 1: Initializing NBJW calibrator...")
    calibrator = NbjwCalibrator(nbjw_config)
    calibrator.load_models()

    pitch_template = get_pitch_template_points()

    # Open video
    video = open_video(video_path)
    cap = video.cap
    width, height = video.width, video.height
    sampler = make_sampler(video, process_fps)

    # Results storage
    calibration_results = init_calibration_results(
        video_path, video, process_fps,
        extra_video_info={"backend": "nbjw"},
    )
    homographies = {}

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)  # ~100 visualizations

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1: Calibrating (NBJW)")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints
        detection_result = calibrator.detect_keypoints(frame)
        keypoints = detection_result.get("keypoints", {})

        # Compute homography
        homography = calibrator.compute_homography(keypoints, width, height)

        if homography is not None:
            homographies[frame_idx] = homography
            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": len(keypoints),
                "num_lines": len(detection_result.get("lines", {})),
            }
            calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}

        # Save visualization periodically
        if idx % vis_interval == 0:
            vis = _draw_visualization_nbjw(
                frame, keypoints, homography, pitch_template, width, height
            )
            cv2.imwrite(str(vis_dir / f"frame_{frame_idx:05d}.jpg"), vis)

    cap.release()

    save_calibration_outputs(output_dir, calibration_results, homographies)
    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
        extra_stats={"backend": "nbjw"},
    )
    print_calibration_summary(stats, label="Stage 1 (NBJW)")
    return stats


def _draw_visualization_nbjw(
    frame: np.ndarray,
    keypoints: dict,
    homography: np.ndarray | None,
    pitch_template: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw visualization for NBJW backend."""
    vis = frame.copy()

    # Draw projected pitch if calibration succeeded
    if homography is not None:
        # Need to invert homography for world -> image projection
        try:
            H_inv = np.linalg.inv(homography)
            H_inv = H_inv / H_inv[-1, -1]
            projected = project_pitch_to_image(H_inv, pitch_template)
            for name, points in projected.items():
                valid_points = [p for p in points if p is not None]
                for i in range(len(valid_points) - 1):
                    pt1, pt2 = valid_points[i], valid_points[i + 1]
                    if all(-1000 < c < 3000 for c in pt1 + pt2):
                        cv2.line(vis, pt1, pt2, (0, 165, 255), 2)
        except np.linalg.LinAlgError:
            pass

    # Draw keypoints (NBJW keypoints are normalized 0-1)
    for idx, kp in keypoints.items():
        if isinstance(idx, int) and idx <= 57:  # Only draw pitch keypoints
            x = int(kp.get("x", 0) * width)
            y = int(kp.get("y", 0) * height)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return vis
