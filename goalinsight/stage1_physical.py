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
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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


class DetectionPrefetcher:
    """Prefetch frames and run keypoint/line detection in background threads.

    Overlaps I/O + detection with calibration on the main thread.
    Uses a dedicated video reader thread and a pool for detection.
    """

    def __init__(
        self,
        video_path: str,
        frame_indices: list[int],
        kp_detector,
        line_detector,
        num_workers: int = 2,
        prefetch_size: int = 4,
    ):
        self.frame_indices = frame_indices
        self.kp_detector = kp_detector
        self.line_detector = line_detector
        self.prefetch_size = prefetch_size

        # Results queue (ordered dict-like)
        self._results: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._done = False

        # Start background pipeline
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(str(video_path),),
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, video_path: str):
        """Read frames and submit detection jobs."""
        cap = cv2.VideoCapture(video_path)
        pending = deque()  # Track pending futures to limit prefetch

        for frame_idx in self.frame_indices:
            # Limit prefetch depth
            while len(pending) >= self.prefetch_size:
                # Wait for oldest to complete
                oldest_idx, oldest_future = pending[0]
                oldest_future.result()  # Block until done
                pending.popleft()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            future = self._executor.submit(
                self._detect_frame, frame_idx, frame,
            )
            pending.append((frame_idx, future))

        # Wait for all remaining
        for idx, future in pending:
            future.result()

        cap.release()
        with self._lock:
            self._done = True
            self._ready.notify_all()

    def _detect_frame(self, frame_idx: int, frame: np.ndarray):
        """Run detection on a single frame."""
        keypoints = self.kp_detector.detect(frame, convert_to_soccernet=False)
        lines = self.line_detector.detect(frame) if self.line_detector is not None else []

        with self._lock:
            self._results[frame_idx] = (frame, keypoints, lines)
            self._ready.notify_all()

    def get(self, frame_idx: int, timeout: float = 30.0) -> tuple[np.ndarray, list, list] | None:
        """Get detection results for a frame, blocking until ready.

        Returns None if the frame could not be read (e.g. video shorter than
        reported by CAP_PROP_FRAME_COUNT).
        """
        with self._lock:
            while frame_idx not in self._results:
                if self._done and frame_idx not in self._results:
                    return None
                self._ready.wait(timeout=timeout)
                if frame_idx not in self._results and self._done:
                    return None
            result = self._results.pop(frame_idx)
            return result

    def shutdown(self):
        self._reader_thread.join(timeout=5)
        self._executor.shutdown(wait=True)


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
            "Scaling focal length.",
            video_width, video_height, profile_name,
            expected_size[0], expected_size[1],
        )
        sx = video_width / expected_size[0]
        K[0, 0] *= sx
        K[1, 1] *= sx

    # Fix principal point to image geometric center, zero distortion
    K[0, 2] = video_width / 2.0
    K[1, 2] = video_height / 2.0
    dist_coeffs = np.zeros(5, dtype=np.float64)

    print(f"  Camera profile: {profile_name}")
    print(f"  K: f={K[0,0]:.1f}, cx={K[0,2]:.1f} (center), cy={K[1,2]:.1f} (center)")
    print(f"  Distortion: disabled (images assumed undistorted)")

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

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()  # Metadata extracted; prefetcher will open its own handle

    sampler = FrameSampler(total_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

    # Load camera profile with fixed intrinsics
    print("Stage 1 (Physical): Loading camera profile...")
    K, dist_coeffs = _load_camera_profile(config, width, height)

    # Initialize calibrator (7-DOF: rvec, tvec, f)
    do_joint = phys_config.get("joint_optimize", True)
    focal_bounds = tuple(phys_config.get("focal_bounds", [1200.0, 2200.0]))

    if not do_joint:
        print("  Joint optimization disabled — per-frame f optimization only")

    camera_position = phys_config.get("camera_position", None)
    if camera_position is not None:
        camera_position = tuple(camera_position)
        print(f"  Camera position constraint: ({camera_position[0]}, {camera_position[1]}, {camera_position[2]})")

    pitch_length = phys_config.get("pitch_length", 105.0)
    pitch_width = phys_config.get("pitch_width", 68.0)
    if pitch_length != 105.0 or pitch_width != 68.0:
        print(f"  Custom pitch dimensions: {pitch_length}×{pitch_width}m")
    pitch_template = get_pitch_template_points(pitch_length, pitch_width)

    calibrator = PhysicalCalibrator(
        K=K,
        image_size=(width, height),
        ransac_reproj_error=phys_config.get("ransac_reproj_error", 15.0),
        line_weight=phys_config.get("line_weight", 1.0),
        line_sample_points=phys_config.get("line_sample_points", 20),
        focal_bounds=focal_bounds,
        world_residual_weight=phys_config.get("world_residual_weight", 0.0),
        world_error_threshold=phys_config.get("world_error_threshold", 5.0),
        camera_position=camera_position,
        position_weight=phys_config.get("position_weight", 50.0),
        pitch_length=pitch_length,
        pitch_width=pitch_width,
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

    # Process frames — Pass 1: per-frame calibration to collect initial estimates
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)
    joint_frame_data = []  # Collect data for cross-frame joint intrinsic optimization

    # Use prefetcher to overlap frame reading + detection with calibration
    num_workers = phys_config.get("num_workers", 2)
    prefetch_size = phys_config.get("prefetch_size", 4)
    print(f"  Detection prefetch: {num_workers} workers, buffer={prefetch_size}")

    prefetcher = DetectionPrefetcher(
        video_path=str(video_path),
        frame_indices=list(sampler),
        kp_detector=kp_detector,
        line_detector=line_detector,
        num_workers=num_workers,
        prefetch_size=prefetch_size,
    )

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1 (Physical): Pass 1 - Per-frame")):
        # Get pre-detected frame from background workers
        result = prefetcher.get(frame_idx)
        if result is None:
            break  # Video shorter than reported — stop processing
        frame, keypoints, lines = result

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

            # Collect frame data for joint intrinsic optimization
            joint_frame_data.append({
                "frame_idx": frame_idx,
                "rvec": result["camera_params"]["rvec"].ravel(),
                "tvec": result["camera_params"]["tvec"].ravel(),
                "img_pts": result["img_pts"],
                "world_pts": result["world_pts"],
                "line_constraints": result.get("line_constraints", []),
                "f": result["camera_params"]["focal_length"],
            })

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
                pitch_length=pitch_length, pitch_width=pitch_width,
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

    prefetcher.shutdown()

    # === Cross-frame joint intrinsic optimization ===
    joint_result = None
    if do_joint and len(joint_frame_data) >= 2:
        # Filter out high-error frames — they have bad detections that would corrupt joint optimization
        # Use per-frame reprojection error to filter (computed in Pass 1)
        good_frame_data = []
        for fd in joint_frame_data:
            finfo = calibration_results["frames"].get(fd["frame_idx"], {})
            px_err = finfo.get("reprojection_error", 999)
            n_inliers = finfo.get("inliers", 0)
            if px_err < 80 and n_inliers >= 5:
                good_frame_data.append(fd)
        print(f"\n  Joint optimization: {len(good_frame_data)}/{len(joint_frame_data)} frames (filtered by error<80px, inliers>=5)...")
        print(f"  Profile f={K[0,0]:.1f}, cx={K[0,2]:.1f} (center), cy={K[1,2]:.1f} (center)")

        # Seed joint optimization with median focal length from Pass 1
        if good_frame_data:
            med_f = float(np.median([fd["f"] for fd in good_frame_data]))
            calibrator.K[0, 0] = med_f
            calibrator.K[1, 1] = med_f
            print(f"  Median f={med_f:.1f}")

        joint_result = calibrator.joint_optimize_intrinsics(good_frame_data)
        if joint_result is not None:
            print(f"  Joint f={joint_result['f']:.1f}, cost={joint_result['cost']:.2f}")

            # Update calibrator with jointly optimized focal length and lock it
            calibrator.K = joint_result["K"].copy()
            eps = 0.01
            calibrator.focal_bounds = (joint_result["f"] - eps, joint_result["f"] + eps)

            # === Pass 2: Re-run per-frame calibration with fixed joint intrinsics ===
            print(f"\n  Pass 2: Re-calibrating {len(sampler)} frames with joint intrinsics...")
            tracker2 = CameraStateTracker(
                max_reproj_error=phys_config.get("max_reproj_error", 15.0),
            )

            # Build lookup from joint result per-frame extrinsics for warm-starting
            joint_extrinsics = {}
            for i, fd in enumerate(good_frame_data):
                joint_extrinsics[fd["frame_idx"]] = (
                    joint_result["per_frame"][i]["rvec"],
                    joint_result["per_frame"][i]["tvec"],
                )

            calibrated_count = 0
            calibration_results["frames"] = {}
            homographies = {}
            camera_poses = {}

            prefetcher2 = DetectionPrefetcher(
                video_path=str(video_path),
                frame_indices=list(sampler),
                kp_detector=kp_detector,
                line_detector=line_detector,
                num_workers=num_workers,
                prefetch_size=prefetch_size,
            )

            for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1 (Physical): Pass 2 - Joint intrinsics")):
                result = prefetcher2.get(frame_idx)
                if result is None:
                    break
                frame, keypoints, lines = result
                calibrator.update(keypoints, lines)

                # Warm-start: prefer joint extrinsics, then temporal tracker
                if frame_idx in joint_extrinsics:
                    init_rvec = joint_extrinsics[frame_idx][0].copy()
                    init_tvec = joint_extrinsics[frame_idx][1].copy()
                else:
                    init_rvec, init_tvec = tracker2.get_initial_guess()

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
                    tracker2.update(
                        result["camera_params"]["rvec"],
                        result["camera_params"]["tvec"],
                        result["final_error"],
                    )
                else:
                    calibration_results["frames"][frame_idx] = {"calibrated": False}
                    tracker2.last_rvec = None
                    tracker2.last_tvec = None

                # Save visualizations (Pass 2 overwrites Pass 1)
                if idx % vis_interval == 0:
                    fname = f"frame_{frame_idx:05d}.jpg"
                    json_fname = f"frame_{frame_idx:05d}.json"
                    cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))
                    cv2.imwrite(str(vis_line_dir / fname), draw_vis_lines(frame, lines))
                    vis = _draw_physical_calibration(
                        frame, keypoints, lines, result, pitch_template,
                        keypoint_mapper, calibrator,
                        pitch_length=pitch_length, pitch_width=pitch_width,
                    )
                    cv2.imwrite(str(vis_calib_dir / fname), vis)
                    frame_info = _build_frame_json(
                        frame_idx, keypoints, lines, result,
                        warm_start=init_rvec is not None,
                        debug_info=calibrator._last_debug if result is None else None,
                        image_size=(width, height),
                    )
                    json_str = json.dumps(frame_info, indent=2, default=_json_default)
                    with open(vis_calib_dir / json_fname, "w") as jf:
                        jf.write(json_str)

            prefetcher2.shutdown()
            tracker = tracker2
        else:
            print("  Joint optimization failed.")
    else:
        if not do_joint:
            print(f"\n  Joint optimization disabled (joint_optimize=false). Using profile intrinsics directly.")
        else:
            print(f"\n  Skipping joint optimization (only {len(joint_frame_data)} calibrated frames)")

    # Store joint intrinsics in results
    if joint_result is not None:
        calibration_results["joint_intrinsics"] = {
            "f": joint_result["f"],
            "cx": joint_result["cx"],
            "cy": joint_result["cy"],
            "cost": joint_result["cost"],
            "n_frames": joint_result["n_frames"],
        }

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f, default=_json_default)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)
    with open(output_dir / "camera_poses.pkl", "wb") as f:
        pickle.dump(camera_poses, f)

    # Save camera poses as JSON (human-readable counterpart of pkl)
    camera_poses_json = {}
    for fidx, pose in camera_poses.items():
        camera_poses_json[str(fidx)] = {
            "K": np.array(pose["K"]).tolist(),
            "dist_coeffs": np.array(pose["dist_coeffs"]).tolist(),
            "rvec": np.array(pose["rvec"]).flatten().tolist(),
            "tvec": np.array(pose["tvec"]).flatten().tolist(),
        }
        # Add derived fields
        R_mat, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
        cam_pos = -R_mat.T @ np.array(pose["tvec"], dtype=np.float64).ravel()
        camera_poses_json[str(fidx)]["camera_position"] = {
            "x": round(float(cam_pos[0]), 3),
            "y": round(float(cam_pos[1]), 3),
            "z": round(float(cam_pos[2]), 3),
        }
        camera_poses_json[str(fidx)]["focal_length"] = float(np.array(pose["K"])[0, 0])
    with open(output_dir / "camera_poses.json", "w") as f:
        json.dump(camera_poses_json, f, indent=2)

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
    if joint_result is not None:
        print(f"  Joint f={joint_result['f']:.1f}")

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
        cam = result.get("camera_params") or {}
        if cam:
            init = result.get("intrinsics_init", {})
            cam_info = {
                "focal_length_init": init.get("f"),
                "focal_length": float(cam.get("focal_length", 0)),
                "cx_init": init.get("cx"),
                "cy_init": init.get("cy"),
                "K": cam["K"].tolist() if "K" in cam else None,
                "rvec": cam["rvec"].flatten().tolist() if "rvec" in cam else None,
                "tvec": cam["tvec"].flatten().tolist() if "tvec" in cam else None,
                "dist_coeffs": cam["dist_coeffs"].tolist() if "dist_coeffs" in cam else None,
            }

            # Compute camera world position and orientation angles
            if "rvec" in cam and "tvec" in cam:
                R_mat, _ = cv2.Rodrigues(cam["rvec"].reshape(3, 1))
                cam_pos = -R_mat.T @ cam["tvec"].ravel()
                cam_info["camera_height"] = round(float(cam_pos[2]), 2)
                cam_info["camera_position"] = {
                    "x": round(float(cam_pos[0]), 2),
                    "y": round(float(cam_pos[1]), 2),
                    "z": round(float(cam_pos[2]), 2),
                }
                # Camera orientation: pitch (tilt), yaw (pan), roll
                # R maps world→camera; R.T maps camera→world
                # Camera forward direction in world = R.T @ [0,0,1]
                fwd = R_mat.T @ np.array([0, 0, 1.0])
                right = R_mat.T @ np.array([1.0, 0, 0])
                pitch_deg = float(np.degrees(np.arcsin(-fwd[2])))  # angle below horizon
                yaw_deg = float(np.degrees(np.arctan2(fwd[0], fwd[1])))
                roll_deg = float(np.degrees(np.arctan2(right[2], np.sqrt(right[0]**2 + right[1]**2))))
                cam_info["camera_angles"] = {
                    "pitch_deg": round(pitch_deg, 2),
                    "yaw_deg": round(yaw_deg, 2),
                    "roll_deg": round(roll_deg, 2),
                }

            info["camera"] = cam_info

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
            cam = result.get("camera_params") or {}

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
                    "kp_id": kp_ids[i] if i < len(kp_ids) else -1,
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
                                keypoint_mapper, calibrator,
                                pitch_length=105.0, pitch_width=68.0):
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

        # Draw derived line constraints
        lc_list = result.get("line_constraints", [])
        if lc_list:
            for lc in lc_list:
                img_samples = lc["img_samples"]
                # Detected image line (red) — connects detected keypoints on this line
                p1 = (int(img_samples[0][0]), int(img_samples[0][1]))
                p2 = (int(img_samples[-1][0]), int(img_samples[-1][1]))
                cv2.line(vis, p1, p2, (0, 0, 255), 1)

                # Projected world line samples (orange) — where the model projects them
                ws = np.array(lc["world_samples"])
                proj, _ = cv2.projectPoints(
                    ws.reshape(-1, 1, 3),
                    camera_params["rvec"], camera_params["tvec"],
                    camera_params["K"], camera_params["dist_coeffs"],
                )
                proj = proj.reshape(-1, 2)
                for i in range(len(proj) - 1):
                    px1 = (int(proj[i][0]), int(proj[i][1]))
                    px2 = (int(proj[i + 1][0]), int(proj[i + 1][1]))
                    p1_ok = -margin < px1[0] < w + margin and -margin < px1[1] < h + margin
                    p2_ok = -margin < px2[0] < w + margin and -margin < px2[1] < h + margin
                    if p1_ok or p2_ok:
                        clamp = w + h
                        c1 = (max(-clamp, min(clamp, px1[0])), max(-clamp, min(clamp, px1[1])))
                        c2 = (max(-clamp, min(clamp, px2[0])), max(-clamp, min(clamp, px2[1])))
                        cv2.line(vis, c1, c2, (0, 165, 255), 1)

        # Draw projected template keypoints (yellow, matching pitch lines)
        from .field_registration.pnlcalib import KeypointMapper
        all_world = calibrator._field_world_coords
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
    pitch = _draw_topdown_pitch(pitch_h, pitch_w, result, keypoints, _Stub(), keypoint_mapper,
                               pitch_length=pitch_length, pitch_width=pitch_width)

    # Combine side-by-side
    combined = np.hstack([vis, pitch])
    return combined
