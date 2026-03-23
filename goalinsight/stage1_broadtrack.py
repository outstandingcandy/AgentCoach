"""Stage 1 (BroadTrack variant): Field Registration using BroadTrack-style calibration.

This is a separate stage1 implementation that follows BroadTrack's algorithm:
- 8-parameter camera model with Cauchy robust loss
- Joint keypoint + arc-length parameterized line curve constraints
- Centered image coordinates with simple radial distortion (k1 only)

The existing stage1 logic is preserved unchanged. This variant is selected
via config: field_registration.backend = "broadtrack"

Output format is fully compatible with the original stage1.
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .stage1 import (
    get_pitch_template_points,
    project_pitch_to_image,
    draw_vis_keypoints,
    draw_vis_lines,
    draw_vis_calibration,
    _draw_topdown_pitch,
)
from .utils.config import get_default_config, get_process_fps_from_config, FrameSampler


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
    from .field_registration import KeypointDetector
    from .field_registration.pnlcalib import KeypointMapper, LineMapper
    from .field_registration.pnlcalib.broadtrack_calibrator import BroadTrackCalibrator

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    bt_config = fr_config.get("broadtrack", {})

    # Initialize keypoint detector (reuse existing PnLCalib detector)
    print("Stage 1 (BroadTrack): Initializing keypoint detector...")
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
        from .field_registration import LineDetector
        print("Stage 1 (BroadTrack): Initializing line detector...")
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
        print("Stage 1 (BroadTrack): Deriving lines from keypoints (no line model)")

    ransac_thresh = pnl_config.get("ransac_threshold", 30.0)
    pitch_template = get_pitch_template_points()

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
    calibration_results = {
        "video_info": {
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "process_fps": process_fps,
        },
        "frames": {}
    }
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

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f)
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
    if errors:
        stats["mean_error"] = float(np.mean(errors))
        stats["median_error"] = float(np.median(errors))

    print(f"\nStage 1 (BroadTrack) Complete:")
    print(f"  Calibrated: {calibrated_count}/{len(sampler)} ({stats['calibration_rate']*100:.1f}%)")
    if errors:
        print(f"  Median error: {stats['median_error']:.2f} px")

    return stats


def _json_default(obj):
    """JSON serializer fallback for numpy types."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


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
                    if all(-5000 < c < 10000 for c in p1 + p2):
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
