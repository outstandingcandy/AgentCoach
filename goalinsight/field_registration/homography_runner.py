"""Stage 1 (Homography): Field Registration via direct ground-plane homography.

Estimates a 3×3 homography from 2D keypoint correspondences using DLT
(cv2.findHomography), without decomposing into camera intrinsics/extrinsics.
Faster and simpler than the physical backend, needs only 4 non-collinear points.

Selected via config: field_registration.backend = "homography"

Output format is compatible with Stage 2/3. Produces:
  - homographies.pkl: ground-plane homographies per frame
  - calibration_metadata.json: per-frame stats
"""

import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from ..utils.pitch import get_pitch_template_points, project_pitch_to_image, _draw_topdown_pitch
from .shared_vis import draw_vis_keypoints
from ..utils.config import get_default_config, get_process_fps_from_config, FrameSampler
from ..utils.serialization import json_default as _json_default
from ._runner_base import (
    open_video,
    make_sampler,
    init_calibration_results,
    save_calibration_outputs,
    compute_calibration_stats,
    print_calibration_summary,
)

logger = logging.getLogger(__name__)


def run_stage1_homography(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using direct homography estimation.

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
    from ._detector_config import (
        build_keypoint_detector_config,
        get_detection_config,
    )
    from .pnlcalib import KeypointMapper
    from .homography_calibrator import HomographyCalibrator

    fr_config = config.get("field_registration", {})
    det_config = get_detection_config(fr_config)
    homog_config = fr_config.get("homography", {})

    # Initialize keypoint detector
    logger.info("Stage 1 (Homography): Initializing keypoint detector...")
    kp_detector = KeypointDetector(build_keypoint_detector_config(det_config))
    kp_detector.load_model()
    keypoint_mapper = KeypointMapper()

    # Open video
    video = open_video(video_path)
    cap = video.cap
    width, height = video.width, video.height
    sampler = make_sampler(video, process_fps)

    pitch_length = homog_config.get("pitch_length", 105.0)
    pitch_width = homog_config.get("pitch_width", 68.0)
    pitch_template = get_pitch_template_points(pitch_length, pitch_width)
    if pitch_length != 105.0 or pitch_width != 68.0:
        logger.info(f"  Custom pitch dimensions: {pitch_length}x{pitch_width}m")

    calibrator = HomographyCalibrator(
        image_size=(width, height),
        ransac_reproj_error=homog_config.get("ransac_reproj_error", 10.0),
        max_reproj_error=homog_config.get("max_reproj_error", 15.0),
        world_error_threshold=homog_config.get("world_error_threshold", 5.0),
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    # Results storage
    calibration_results = init_calibration_results(
        video_path, video, process_fps,
        extra_top_level={"backend": "homography"},
    )
    homographies = {}

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_calib_dir]:
        d.mkdir(exist_ok=True)

    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1 (Homography)")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        calibrator.update(keypoints, [])

        result = calibrator.calibrate(
            keypoint_mapper,
            min_confidence=det_config.get("keypoint_threshold", 0.3434),
        )

        if result is not None:
            homographies[frame_idx] = result["homography"]
            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": result["num_keypoints"],
                "num_intersections": result["num_intersections"],
                "total_points": result["total_points"],
                "inliers": result["inliers"],
                "reprojection_error": result["final_error"],
                "world_error": result.get("world_error"),
            }
            calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}

        # Visualizations
        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))

            vis = _draw_homography_calibration(
                frame, keypoints, result, pitch_template,
                calibrator, keypoint_mapper,
                pitch_length=pitch_length, pitch_width=pitch_width,
            )
            cv2.imwrite(str(vis_calib_dir / fname), vis)

            # Per-frame JSON
            frame_info = _build_frame_json(frame_idx, keypoints, result)
            json_str = json.dumps(frame_info, default=_json_default)
            with open(vis_calib_dir / f"frame_{frame_idx:05d}.json", "w") as jf:
                jf.write(json_str)

    cap.release()

    save_calibration_outputs(
        output_dir, calibration_results, homographies,
        json_default=_json_default,
    )
    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
    )
    print_calibration_summary(stats, label="Stage 1 (Homography)")
    return stats


def _build_frame_json(frame_idx, keypoints, result):
    """Build per-frame JSON dict."""
    info = {
        "frame_idx": int(frame_idx),
        "calibrated": result is not None,
        "detected_keypoints": [
            {
                "id": int(kp.get("id", -1)),
                "x": float(kp.get("x", 0)),
                "y": float(kp.get("y", 0)),
                "confidence": float(kp.get("confidence", 0)),
            }
            for kp in keypoints
        ],
    }
    if result is not None:
        info["reprojection_error"] = result.get("final_error")
        info["world_error"] = result.get("world_error")
        info["total_points"] = result.get("total_points", 0)
        info["inliers"] = result.get("inliers", 0)
        info["homography"] = result["homography"].tolist()
    return info


def _draw_homography_calibration(frame, keypoints, result, pitch_template,
                                  calibrator, keypoint_mapper,
                                  pitch_length=105.0, pitch_width=68.0):
    """Draw calibration visualization: camera view + top-down pitch diagram."""
    h, w = frame.shape[:2]
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if result is not None and result.get("homography") is not None:
        H = result["homography"]
        projected = project_pitch_to_image(H, pitch_template, camera_params=None)

        margin = 200
        for name, pts in projected.items():
            for i in range(len(pts) - 1):
                if pts[i] is not None and pts[i + 1] is not None:
                    try:
                        p1 = (int(float(pts[i][0])), int(float(pts[i][1])))
                        p2 = (int(float(pts[i + 1][0])), int(float(pts[i + 1][1])))
                        p1_in = -margin < p1[0] < w + margin and -margin < p1[1] < h + margin
                        p2_in = -margin < p2[0] < w + margin and -margin < p2[1] < h + margin
                        if p1_in and p2_in:
                            clamp = w + h
                            p1c = (max(-clamp, min(clamp, p1[0])), max(-clamp, min(clamp, p1[1])))
                            p2c = (max(-clamp, min(clamp, p2[0])), max(-clamp, min(clamp, p2[1])))
                            cv2.line(vis, p1c, p2c, (0, 255, 255), 2)
                    except (ValueError, OverflowError, TypeError):
                        continue

        # Draw detected keypoints (green=inlier, red=outlier)
        if "img_pts" in result and "inlier_mask" in result:
            for pt, is_inlier in zip(result["img_pts"], result["inlier_mask"]):
                x, y = int(pt[0]), int(pt[1])
                color = (0, 255, 0) if is_inlier else (0, 0, 255)
                cv2.circle(vis, (x, y), 5, color, -1)

        # Project template keypoints via homography
        all_world = calibrator._field_world_coords
        for kid, (wx, wy) in enumerate(all_world):
            if kid in {12, 14, 16, 18}:  # non-ground
                continue
            pt_h = H @ np.array([wx, wy, 1.0])
            if abs(pt_h[2]) > 1e-6:
                ix, iy = pt_h[0] / pt_h[2], pt_h[1] / pt_h[2]
                if -50 < ix < w + 50 and -50 < iy < h + 50:
                    pt = (int(ix), int(iy))
                    cv2.circle(vis, pt, 3, (0, 255, 255), -1)
                    cv2.putText(vis, str(kid), (pt[0] + 5, pt[1] - 3),
                                font, 0.3, (0, 255, 255), 1)

        err = result.get("final_error", 0)
        world_err = result.get("world_error", 0)
        n_in = result.get("inliers", 0)
        n_total = result.get("total_points", 0)
        header = f"Homography Err: {err:.1f}px | World: {world_err:.2f}m | Inliers: {n_in}/{n_total}"
    else:
        header = "Homography: Calibration FAILED"

    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (0, 200, 0), 1)

    # Top-down pitch diagram
    class _Stub:
        line_intersections = []
        image_size = (w, h)
    pitch_h = h
    pitch_w = int(pitch_h * 1.5)
    pitch = _draw_topdown_pitch(pitch_h, pitch_w, result, keypoints, _Stub(), keypoint_mapper,
                                pitch_length=pitch_length, pitch_width=pitch_width)

    combined = np.hstack([vis, pitch])
    return combined
