"""Stage 1 (Physical): Field Registration with fixed camera intrinsics.

Uses known camera intrinsics from a profile (K, distortion) and only optimizes
6-DOF extrinsics (rvec, tvec) per frame. Includes temporal warm-starting where
the previous frame's pose seeds the next frame's optimization.

Selected via config: field_registration.backend = "physical"

Output format is compatible with Stage 2/3. Produces both:
  - camera_poses.pkl: physical camera parameters per frame
  - homographies.pkl: derived ground-plane homographies for backward compat
"""

import json
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from .stage1 import (
    get_pitch_template_points,
    project_pitch_to_image,
    draw_vis_keypoints,
    draw_vis_lines,
    _draw_topdown_pitch,
)
from .utils.config import get_default_config, get_process_fps_from_config, FrameSampler

logger = logging.getLogger(__name__)


class CameraStateTracker:
    """Temporal state tracker for camera pose warm-starting.

    Stores the previous frame's optimized rvec/tvec and provides it as
    initial guess for the next frame. Resets when reprojection error exceeds
    threshold, forcing a full PnP RANSAC re-initialization.
    """

    def __init__(self, max_reproj_error: float = 15.0):
        self.last_rvec: np.ndarray | None = None
        self.last_tvec: np.ndarray | None = None
        self.max_reproj_error = max_reproj_error
        self.warm_start_count = 0
        self.reinit_count = 0

    def get_initial_guess(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get warm-start pose from previous frame."""
        if self.last_rvec is not None:
            return self.last_rvec.copy(), self.last_tvec.copy()
        return None, None

    def update(self, rvec: np.ndarray, tvec: np.ndarray, reproj_error: float):
        """Update tracker with current frame's result."""
        if reproj_error < self.max_reproj_error:
            self.last_rvec = rvec.copy()
            self.last_tvec = tvec.copy()
            self.warm_start_count += 1
        else:
            # Error too high — force PnP re-init on next frame
            self.last_rvec = None
            self.last_tvec = None
            self.reinit_count += 1

    def needs_reinit(self) -> bool:
        return self.last_rvec is None


def _load_camera_profile(config: dict, video_width: int, video_height: int):
    """Load camera intrinsic profile from YAML file.

    Args:
        config: Pipeline config dict.
        video_width: Actual video width for validation.
        video_height: Actual video height for validation.

    Returns:
        (K, dist_coeffs) as numpy arrays.
    """
    phys_config = config.get("field_registration", {}).get("physical", {})
    profile_name = phys_config.get("camera_profile", "veo_1080p")

    # Find camera_profiles.yaml
    profiles_path = phys_config.get("camera_profiles_path")
    if profiles_path:
        profiles_path = Path(profiles_path)
    else:
        # Auto-detect: look in configs/ relative to project root
        candidates = [
            Path("configs/camera_profiles.yaml"),
            Path(__file__).parent.parent / "configs" / "camera_profiles.yaml",
            Path(__file__).parent.parent.parent / "configs" / "camera_profiles.yaml",
        ]
        for c in candidates:
            if c.exists():
                profiles_path = c
                break

    if profiles_path is None or not profiles_path.exists():
        raise FileNotFoundError(
            f"Camera profiles file not found. Searched: {candidates}. "
            f"Set field_registration.physical.camera_profiles_path in config."
        )

    with open(profiles_path) as f:
        profiles_data = yaml.safe_load(f)

    profiles = profiles_data.get("profiles", {})
    if profile_name not in profiles:
        raise ValueError(
            f"Camera profile '{profile_name}' not found. "
            f"Available: {list(profiles.keys())}"
        )

    profile = profiles[profile_name]
    K = np.array(profile["K"], dtype=np.float64)
    dist_coeffs = np.array(profile["dist_coeffs"], dtype=np.float64).ravel()
    expected_size = profile.get("image_size", [video_width, video_height])

    if expected_size[0] != video_width or expected_size[1] != video_height:
        logger.warning(
            "Video resolution %dx%d doesn't match profile '%s' expected %dx%d. "
            "Scaling intrinsics.",
            video_width, video_height, profile_name,
            expected_size[0], expected_size[1],
        )
        sx = video_width / expected_size[0]
        sy = video_height / expected_size[1]
        K[0, 0] *= sx
        K[0, 2] *= sx
        K[1, 1] *= sy
        K[1, 2] *= sy

    print(f"  Camera profile: {profile_name}")
    print(f"  K: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    print(f"  Distortion: {dist_coeffs.tolist()}")

    return K, dist_coeffs


def run_stage1_physical(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using physical camera calibration with fixed intrinsics.

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
    from .field_registration.physical_calibrator import PhysicalCalibrator

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    phys_config = fr_config.get("physical", {})

    # Initialize keypoint detector (reuse PnLCalib HRNet detector)
    print("Stage 1 (Physical): Initializing keypoint detector...")
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

    # Optional line detector
    use_line_model = phys_config.get("use_line_model", False)
    line_detector = None
    line_mapper = LineMapper()

    if use_line_model:
        from .field_registration import LineDetector
        print("Stage 1 (Physical): Initializing line detector...")
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
        print("Stage 1 (Physical): Deriving lines from keypoints (no line model)")

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

    # Load camera profile with fixed intrinsics
    print("Stage 1 (Physical): Loading camera profile...")
    K, dist_coeffs = _load_camera_profile(config, width, height)

    # Initialize calibrator with fixed intrinsics
    calibrator = PhysicalCalibrator(
        K=K,
        dist_coeffs=dist_coeffs,
        image_size=(width, height),
        ransac_reproj_error=phys_config.get("ransac_reproj_error", 15.0),
        line_weight=phys_config.get("line_weight", 1.0),
        line_sample_points=phys_config.get("line_sample_points", 20),
        focal_bounds=tuple(phys_config.get("focal_bounds", [1200.0, 2200.0])),
        cx_bounds=tuple(phys_config.get("cx_bounds", [-100.0, 100.0])),
        cy_bounds=tuple(phys_config.get("cy_bounds", [-50.0, 50.0])),
        k1_bounds=tuple(phys_config.get("k1_bounds", [-0.35, -0.15])),
        intrinsic_reg_weight=phys_config.get("intrinsic_reg_weight", 0.0),
        world_residual_weight=phys_config.get("world_residual_weight", 0.0),
    )

    # Initialize temporal tracker
    tracker = CameraStateTracker(
        max_reproj_error=phys_config.get("max_reproj_error", 15.0),
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
    camera_poses = {}

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_line_dir = vis_dir / "lines"
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_line_dir, vis_calib_dir]:
        d.mkdir(exist_ok=True)

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1 (Physical): Calibrating")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints (and lines if enabled)
        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        lines = line_detector.detect(frame) if line_detector is not None else []

        # Update calibrator with detections
        calibrator.update(keypoints, lines)

        # Get warm-start from temporal tracker
        init_rvec, init_tvec = tracker.get_initial_guess()

        # Run calibration
        result = calibrator.calibrate(
            keypoint_mapper,
            line_mapper if use_line_model else None,
            min_confidence=pnl_config.get("keypoint_threshold", 0.3434),
            initial_rvec=init_rvec,
            initial_tvec=init_tvec,
        )

        if result is not None:
            homographies[frame_idx] = result["homography"]
            camera_poses[frame_idx] = result["camera_pose"]

            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": result["num_keypoints"],
                "num_lines": result["num_lines"],
                "num_intersections": result["num_intersections"],
                "total_points": result["total_points"],
                "inliers": result["inliers"],
                "reprojection_error": result["final_error"],
                "world_error": result.get("world_error"),
                "world_error_all": result.get("world_error_all"),
                "line_constraints": result["line_constraints_count"],
                "warm_start": init_rvec is not None,
            }
            calibrated_count += 1

            # Update temporal tracker
            tracker.update(
                result["camera_params"]["rvec"],
                result["camera_params"]["tvec"],
                result["final_error"],
            )
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}
            # Reset tracker on failure
            tracker.last_rvec = None
            tracker.last_tvec = None

        # Save visualizations and detailed JSON periodically
        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            json_fname = f"frame_{frame_idx:05d}.json"
            cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))
            cv2.imwrite(str(vis_line_dir / fname), draw_vis_lines(frame, lines))

            vis = _draw_physical_calibration(
                frame, keypoints, lines, result, pitch_template,
                keypoint_mapper, calibrator,
            )
            cv2.imwrite(str(vis_calib_dir / fname), vis)

            # Save detailed per-frame JSON
            frame_info = _build_frame_json(
                frame_idx, keypoints, lines, result,
                warm_start=init_rvec is not None,
                debug_info=calibrator._last_debug if result is None else None,
                image_size=(width, height),
            )
            json_str = json.dumps(frame_info, indent=2, default=_json_default)
            with open(vis_calib_dir / json_fname, "w") as jf:
                jf.write(json_str)

    cap.release()

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f, default=_json_default)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)
    with open(output_dir / "camera_poses.pkl", "wb") as f:
        pickle.dump(camera_poses, f)

    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": len(sampler),
        "calibrated_frames": calibrated_count,
        "calibration_rate": calibrated_count / len(sampler) if sampler else 0,
        "warm_start_frames": tracker.warm_start_count,
        "reinit_frames": tracker.reinit_count,
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
    world_errors_all = [
        calibration_results["frames"][idx]["world_error_all"]
        for idx in calibration_results["frames"]
        if calibration_results["frames"][idx].get("calibrated")
        and calibration_results["frames"][idx].get("world_error_all") is not None
    ]
    if errors:
        stats["mean_error"] = float(np.mean(errors))
        stats["median_error"] = float(np.median(errors))
    if world_errors:
        stats["mean_world_error"] = float(np.mean(world_errors))
        stats["median_world_error"] = float(np.median(world_errors))
    if world_errors_all:
        stats["mean_world_error_all"] = float(np.mean(world_errors_all))
        stats["median_world_error_all"] = float(np.median(world_errors_all))

    print(f"\nStage 1 (Physical) Complete:")
    print(f"  Calibrated: {calibrated_count}/{len(sampler)} ({stats['calibration_rate']*100:.1f}%)")
    print(f"  Warm-starts: {tracker.warm_start_count}, Re-inits: {tracker.reinit_count}")
    if errors:
        print(f"  Median error: {stats['median_error']:.2f} px")
        print(f"  Mean error: {stats['mean_error']:.2f} px")
    if world_errors:
        print(f"  Median world error (detected): {stats['median_world_error']:.2f} m")
    if world_errors_all:
        print(f"  Median world error (all 57):   {stats['median_world_error_all']:.2f} m")

    return stats


def _build_frame_json(frame_idx, keypoints, lines, result, warm_start=False, debug_info=None, image_size=None):
    """Build detailed JSON dict for a single frame's calibration data.

    Includes all detected keypoints, lines, camera parameters, per-point
    reprojection errors, and line constraint details. When calibration fails,
    debug_info provides RANSAC input points and failure reason.
    """
    info = {
        "frame_idx": int(frame_idx),
        "calibrated": result is not None,
        "warm_start": warm_start,
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

    # Include debug info for failed calibrations
    if result is None and debug_info is not None:
        info["failure_reason"] = debug_info.get("failure_reason", "unknown")
        # Input correspondences that were fed to RANSAC
        dbg_img = debug_info.get("img_pts")
        dbg_world = debug_info.get("world_pts")
        dbg_ids = debug_info.get("kp_ids", [])
        if dbg_img is not None and len(dbg_img) > 0:
            info["ransac_input_points"] = [
                {
                    "kp_id": int(dbg_ids[i]) if i < len(dbg_ids) else -1,
                    "img_x": float(dbg_img[i][0]),
                    "img_y": float(dbg_img[i][1]),
                    "world_x": float(dbg_world[i][0]) if dbg_world is not None and i < len(dbg_world) else None,
                    "world_y": float(dbg_world[i][1]) if dbg_world is not None and i < len(dbg_world) else None,
                    "world_z": float(dbg_world[i][2]) if dbg_world is not None and i < len(dbg_world) else None,
                }
                for i in range(len(dbg_img))
            ]
        # RANSAC failure details (if PnP was attempted)
        dbg_ransac = debug_info.get("ransac_info")
        if dbg_ransac is not None:
            info["ransac"] = {
                "success": dbg_ransac.get("success", False),
                "failure_reason": dbg_ransac.get("failure_reason", ""),
                "reprojection_threshold": dbg_ransac.get("reprojection_threshold"),
                "total_points": dbg_ransac.get("total_points"),
                "inlier_count": dbg_ransac.get("inlier_count"),
                "inlier_indices": dbg_ransac.get("inlier_indices", []),
            }

    if result is not None:
        info["reprojection_error"] = result.get("final_error")
        info["world_error"] = result.get("world_error")
        info["world_error_all"] = result.get("world_error_all")
        info["num_keypoints"] = result.get("num_keypoints", 0)
        info["num_lines"] = result.get("num_lines", 0)
        info["total_points"] = result.get("total_points", 0)
        info["inliers"] = result.get("inliers", 0)
        info["line_constraints_count"] = result.get("line_constraints_count", 0)

        # Full camera parameters
        cam = result.get("camera_params", {})
        if cam:
            init = result.get("intrinsics_init", {})
            info["camera"] = {
                "focal_length_init": init.get("f"),
                "focal_length": float(cam.get("focal_length", 0)),
                "cx_init": init.get("cx"),
                "cy_init": init.get("cy"),
                "k1_init": init.get("k1"),
                "K": cam["K"].tolist() if "K" in cam else None,
                "rvec": cam["rvec"].flatten().tolist() if "rvec" in cam else None,
                "tvec": cam["tvec"].flatten().tolist() if "tvec" in cam else None,
                "dist_coeffs": cam["dist_coeffs"].tolist() if "dist_coeffs" in cam else None,
            }

        # RANSAC filtering info
        ransac = result.get("ransac_info")
        if ransac is not None:
            info["ransac"] = {
                "reprojection_threshold": ransac["reprojection_threshold"],
                "total_points": ransac["total_points"],
                "inlier_count": ransac["inlier_count"],
                "outlier_count": ransac["outlier_count"],
                "inlier_indices": ransac["inlier_indices"],
                "mean_inlier_error": ransac["mean_inlier_error"],
                "mean_outlier_error": ransac["mean_outlier_error"],
                "mean_all_error": ransac["mean_all_error"],
                "rvec_init": ransac["rvec_init"].tolist() if hasattr(ransac["rvec_init"], "tolist") else ransac["rvec_init"],
                "tvec_init": ransac["tvec_init"].tolist() if hasattr(ransac["tvec_init"], "tolist") else ransac["tvec_init"],
            }

        # Line constraint details
        lc_list = result.get("line_constraints", [])
        if lc_list and cam and "rvec" in cam and "tvec" in cam:
            import cv2 as _cv2

            line_details = []
            for lc in lc_list:
                ws = np.array(lc["world_samples"])
                imgs = np.array(lc["img_samples"])

                # Project 3D world samples with optimized pose
                proj_raw, _ = _cv2.projectPoints(
                    ws.reshape(-1, 1, 3),
                    cam["rvec"].reshape(3, 1),
                    cam["tvec"].reshape(3, 1),
                    cam["K"], cam["dist_coeffs"],
                )
                proj = proj_raw.reshape(-1, 2)

                # Filter out degenerate projections (behind camera / singularity)
                IMG_BOUND = 10000
                valid_mask = (np.abs(proj[:, 0]) < IMG_BOUND) & (np.abs(proj[:, 1]) < IMG_BOUND)

                # Perpendicular distance from projected to detected 2D line
                p1 = imgs[0]
                p2 = imgs[-1]
                line_dir = p2 - p1
                line_len = float(np.linalg.norm(line_dir))
                if line_len > 1e-6:
                    normal = np.array([-line_dir[1], line_dir[0]]) / line_len
                    perp_dists = ((proj - p1) @ normal).tolist()
                else:
                    perp_dists = [0.0] * len(proj)

                # Stats only from valid projections
                valid_perp = [abs(perp_dists[j]) for j in range(len(proj)) if valid_mask[j]]

                samples = []
                for j in range(len(ws)):
                    sample = {
                        "world": [float(ws[j][0]), float(ws[j][1]), float(ws[j][2])],
                        "img_detected": [float(imgs[j][0]), float(imgs[j][1])],
                        "valid": bool(valid_mask[j]),
                    }
                    if valid_mask[j]:
                        sample["img_projected"] = [float(proj[j][0]), float(proj[j][1])]
                        sample["perp_error"] = float(perp_dists[j])
                    samples.append(sample)

                line_details.append({
                    "line_id": int(lc["line_id"]),
                    "num_samples": len(ws),
                    "num_valid": int(np.sum(valid_mask)),
                    "img_endpoint_1": [float(p1[0]), float(p1[1])],
                    "img_endpoint_2": [float(p2[0]), float(p2[1])],
                    "mean_perp_error": float(np.mean(valid_perp)) if valid_perp else None,
                    "max_perp_error": float(np.max(valid_perp)) if valid_perp else None,
                    "samples": samples,
                })
            info["line_constraints"] = line_details

        # Per-keypoint reprojection details
        if "img_pts" in result and "inlier_mask" in result:
            import cv2 as _cv2

            img_pts = result["img_pts"]
            inlier_mask = result["inlier_mask"]
            world_pts = result.get("world_pts")
            kp_ids = result.get("kp_ids", [])
            cam = result.get("camera_params", {})

            keypoint_details = []
            projected = None
            if (world_pts is not None and len(world_pts) > 0
                    and "rvec" in cam and "tvec" in cam):
                proj_raw, _ = _cv2.projectPoints(
                    np.array(world_pts).reshape(-1, 1, 3),
                    cam["rvec"].reshape(3, 1),
                    cam["tvec"].reshape(3, 1),
                    cam["K"], cam["dist_coeffs"],
                )
                projected = proj_raw.reshape(-1, 2)

            for i in range(len(img_pts)):
                detail = {
                    "kp_id": int(kp_ids[i]) if i < len(kp_ids) else -1,
                    "img_x": float(img_pts[i][0]),
                    "img_y": float(img_pts[i][1]),
                    "is_inlier": bool(inlier_mask[i]),
                }
                if world_pts is not None and i < len(world_pts):
                    detail["world_x"] = float(world_pts[i][0])
                    detail["world_y"] = float(world_pts[i][1])
                    detail["world_z"] = float(world_pts[i][2])
                if projected is not None and i < len(projected):
                    detail["proj_x"] = float(projected[i][0])
                    detail["proj_y"] = float(projected[i][1])
                    detail["error_px"] = float(np.linalg.norm(
                        projected[i] - img_pts[i]
                    ))
                per_world = result.get("per_point_world_errors")
                if per_world is not None and i < len(per_world):
                    detail["error_world_m"] = float(per_world[i])
                keypoint_details.append(detail)
            info["keypoint_details"] = keypoint_details

        # Projection consistency check: project ALL template keypoints and compare
        # with detected keypoints to find phantom/missing projections
        if cam and "rvec" in cam and "tvec" in cam:
            import cv2 as _cv2
            from .field_registration.pnlcalib import KeypointMapper

            all_world = KeypointMapper.PNLCALIB_WORLD_COORDS_2D
            non_ground = KeypointMapper.NON_GROUND_KEYPOINTS
            detected_ids = {kp["id"] for kp in keypoints if kp.get("confidence", 0) >= 0.3}
            img_w, img_h = image_size if image_size else (1920, 1080)
            margin = 50

            projected_in_image = []  # template points that project into image
            phantom_count = 0  # projected in image but not detected
            missing_count = 0  # detected but projected outside image

            for kid, (wx, wy) in enumerate(all_world):
                if kid in non_ground:
                    continue
                obj = np.array([[[wx, wy, 0.0]]], dtype=np.float64)
                proj, _ = _cv2.projectPoints(
                    obj, cam["rvec"].reshape(3, 1), cam["tvec"].reshape(3, 1),
                    cam["K"], cam["dist_coeffs"],
                )
                ix, iy = float(proj.ravel()[0]), float(proj.ravel()[1])
                in_image = -margin < ix < img_w + margin and -margin < iy < img_h + margin
                detected = kid in detected_ids

                if in_image:
                    status = "detected" if detected else "phantom"
                    if not detected:
                        phantom_count += 1
                    projected_in_image.append({
                        "kp_id": kid,
                        "proj_x": round(ix, 1),
                        "proj_y": round(iy, 1),
                        "status": status,
                    })
                elif detected:
                    missing_count += 1
                    projected_in_image.append({
                        "kp_id": kid,
                        "proj_x": round(ix, 1),
                        "proj_y": round(iy, 1),
                        "status": "missing",  # detected but projected outside
                    })

            info["projection_consistency"] = {
                "total_ground_keypoints": len(all_world) - len(non_ground),
                "projected_in_image": len([p for p in projected_in_image if p["status"] != "missing"]),
                "detected_count": len(detected_ids),
                "phantom_count": phantom_count,
                "missing_count": missing_count,
                "details": projected_in_image,
            }

    return info


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


def _draw_physical_calibration(frame, keypoints, lines, result, pitch_template,
                                keypoint_mapper, calibrator):
    """Draw calibration visualization: camera view + top-down pitch diagram."""
    h, w = frame.shape[:2]
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    if result is not None and result.get("camera_params"):
        camera_params = result["camera_params"]

        # Project pitch template onto image using camera model
        projected = project_pitch_to_image(
            result["homography"], pitch_template, camera_params,
        )

        # Draw projected pitch lines (yellow) — only segments within image bounds
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
                            # Clamp for OpenCV safety
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

        # Draw detected lines (cyan)
        for line in lines:
            if line.get("confidence", 0) >= 0.15:
                x1, y1 = int(line["x1"]), int(line["y1"])
                x2, y2 = int(line["x2"]), int(line["y2"])
                cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        # Draw projected template keypoints (yellow, matching pitch lines)
        from .field_registration.pnlcalib import KeypointMapper
        all_world = KeypointMapper.PNLCALIB_WORLD_COORDS_2D
        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS
        for kid, (wx, wy) in enumerate(all_world):
            if kid in non_ground:
                continue
            obj = np.array([[[wx, wy, 0.0]]], dtype=np.float64)
            proj, _ = cv2.projectPoints(
                obj, camera_params["rvec"], camera_params["tvec"],
                camera_params["K"], camera_params["dist_coeffs"],
            )
            ix, iy = proj.ravel()
            if -50 < ix < w + 50 and -50 < iy < h + 50:
                pt = (int(ix), int(iy))
                cv2.circle(vis, pt, 3, (0, 255, 255), -1)
                cv2.putText(vis, str(kid), (pt[0] + 5, pt[1] - 3),
                            font, 0.3, (0, 255, 255), 1)

        # Header
        err = result.get("final_error", 0)
        world_err = result.get("world_error", 0)
        world_err_all = result.get("world_error_all", 0)
        n_in = result.get("inliers", 0)
        n_total = result.get("total_points", 0)
        n_lines = result.get("line_constraints_count", 0)
        header = f"Physical Err: {err:.1f}px | World: {world_err:.2f}m (det) / {world_err_all:.2f}m (all) | Inliers: {n_in}/{n_total} | Lines: {n_lines}"
    else:
        header = "Physical: Calibration FAILED"

    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (0, 200, 0), 1)

    # Top-down pitch diagram
    class _Stub:
        line_intersections = []
        image_size = (w, h)
    pitch_h = h
    pitch_w = int(pitch_h * 1.5)
    pitch = _draw_topdown_pitch(pitch_h, pitch_w, result, keypoints, _Stub(), keypoint_mapper)

    # Combine side-by-side
    combined = np.hstack([vis, pitch])
    return combined
