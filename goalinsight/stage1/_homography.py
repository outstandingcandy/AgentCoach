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
from ._shared_vis import draw_vis_keypoints
from ..utils.config import get_default_config, get_process_fps_from_config, FrameSampler
from ..utils.serialization import json_default as _json_default

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
    from ..field_registration import KeypointDetector
    from ..field_registration.pnlcalib import KeypointMapper
    from ..field_registration.homography_calibrator import HomographyCalibrator

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    homog_config = fr_config.get("homography", {})

    # Initialize keypoint detector
    print("Stage 1 (Homography): Initializing keypoint detector...")
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

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampler = FrameSampler(total_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

    pitch_length = homog_config.get("pitch_length", 105.0)
    pitch_width = homog_config.get("pitch_width", 68.0)
    pitch_template = get_pitch_template_points(pitch_length, pitch_width)
    if pitch_length != 105.0 or pitch_width != 68.0:
        print(f"  Custom pitch dimensions: {pitch_length}×{pitch_width}m")

    calibrator = HomographyCalibrator(
        image_size=(width, height),
        ransac_reproj_error=homog_config.get("ransac_reproj_error", 10.0),
        max_reproj_error=homog_config.get("max_reproj_error", 15.0),
        world_error_threshold=homog_config.get("world_error_threshold", 5.0),
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )

    # Results storage
    calibration_results = {
        "video_info": {
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "process_fps": process_fps,
        },
        "backend": "homography",
        "frames": {},
    }
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
            min_confidence=pnl_config.get("keypoint_threshold", 0.3434),
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

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f, default=_json_default)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)

    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": len(sampler),
        "calibrated_frames": calibrated_count,
        "calibration_rate": calibrated_count / len(sampler) if sampler else 0,
    }

    errors = [
        calibration_results["frames"][idx]["reprojection_error"]
        for idx in calibration_results["frames"]
        if calibration_results["frames"][idx].get("calibrated")
    ]
    world_errors = [
        calibration_results["frames"][idx]["world_error"]
        for idx in calibration_results["frames"]
        if calibration_results["frames"][idx].get("calibrated")
        and calibration_results["frames"][idx].get("world_error") is not None
    ]
    if errors:
        stats["mean_error"] = float(np.mean(errors))
        stats["median_error"] = float(np.median(errors))
    if world_errors:
        stats["mean_world_error"] = float(np.mean(world_errors))
        stats["median_world_error"] = float(np.median(world_errors))

    print(f"\nStage 1 (Homography) Complete:")
    print(f"  Calibrated: {calibrated_count}/{len(sampler)} ({stats['calibration_rate']*100:.1f}%)")
    if errors:
        print(f"  Median error: {stats['median_error']:.2f} px")
        print(f"  Mean error: {stats['mean_error']:.2f} px")
    if world_errors:
        print(f"  Median world error: {stats['median_world_error']:.2f} m")

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
                        if p1_in or p2_in:
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
