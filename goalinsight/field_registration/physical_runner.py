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

from ..utils.pitch import get_pitch_template_points, project_pitch_to_image, _draw_topdown_pitch
from ..utils.projection import project_points_2d
from .shared_vis import draw_vis_keypoints, draw_vis_lines
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


def _stamp_source(vis: np.ndarray, label: str) -> None:
    """Stamp a small bottom-left tag on the calibration vis to identify which
    pipeline pass produced this frame's pose. Lets a reviewer scan the
    calibration/ folder and see at a glance whether each frame came from
    Pass 1 PnP, Pass 2 lock-C refit, joint-focal refit, or chain gap-fill.
    """
    if vis is None or vis.size == 0:
        return
    cv2.putText(
        vis, label, (10, vis.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA,
    )


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

    logger.info(f"  Camera profile: {profile_name}")
    logger.info(f"  K: f={K[0,0]:.1f}, cx={K[0,2]:.1f} (center), cy={K[1,2]:.1f} (center)")
    logger.info(f"  Distortion: disabled (images assumed undistorted)")

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
    from . import KeypointDetector
    from .pnlcalib import KeypointMapper, LineMapper
    from .physical_calibrator import PhysicalCalibrator

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    phys_config = fr_config.get("physical", {})

    # Initialize keypoint detector (reuse PnLCalib HRNet detector)
    logger.info("Stage 1 (Physical): Initializing keypoint detector...")
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
        from . import LineDetector
        logger.info("Stage 1 (Physical): Initializing line detector...")
        line_config = {
            "backend": "pnlcalib",
            "pnlcalib": {
                "weights": pnl_config.get("line_weights", "SV_lines"),
                "confidence_threshold": pnl_config.get("line_threshold", 0.15),
            }
        }
        line_detector = LineDetector(line_config)
        line_detector.load_model(pnl_config.get("line_model_path"))
    else:
        logger.info("Stage 1 (Physical): Deriving lines from keypoints (no line model)")

    # Open video
    video = open_video(video_path)
    video.cap.release()  # Metadata extracted; prefetcher will open its own handle
    width, height = video.width, video.height
    total_frames = video.total_frames
    fps = video.fps
    sampler = make_sampler(video, process_fps)

    # Load camera profile with fixed intrinsics
    logger.info("Stage 1 (Physical): Loading camera profile...")
    K, dist_coeffs = _load_camera_profile(config, width, height)

    # Initialize calibrator (7-DOF: rvec, tvec, f)
    do_joint = phys_config.get("joint_optimize", True)
    focal_bounds = tuple(phys_config.get("focal_bounds", [1200.0, 2200.0]))

    if not do_joint:
        logger.info("  Joint optimization disabled — per-frame f optimization only")

    camera_position = phys_config.get("camera_position", None)
    lock_camera_position = phys_config.get("lock_camera_position", False)
    if camera_position is not None:
        camera_position = tuple(camera_position)
        mode = "HARD-LOCKED" if lock_camera_position else "soft"
        logger.info(
            f"  Camera position constraint ({mode}): "
            f"({camera_position[0]}, {camera_position[1]}, {camera_position[2]})"
        )
    elif lock_camera_position:
        logger.warning("  lock_camera_position=True but no camera_position set; ignoring")
        lock_camera_position = False

    # Pitch geometry — two layered sources:
    # 1. Top-level config.pitch.* dict (matches pnlcalib_orig convention,
    #    covers PA / GA / goal / center circle).
    # 2. Per-stage overrides at field_registration.physical.{pitch_length,
    #    pitch_width} (legacy keys; only set length / width).
    # Anything missing falls back to FIFA defaults inside PhysicalCalibrator.
    top_pitch = config.get("pitch", {}) or {}
    pitch_dims = dict(top_pitch)
    if "pitch_length" in phys_config:
        pitch_dims["pitch_length"] = phys_config["pitch_length"]
    if "pitch_width" in phys_config:
        pitch_dims["pitch_width"] = phys_config["pitch_width"]
    pitch_length = pitch_dims.get("pitch_length", 105.0)
    pitch_width = pitch_dims.get("pitch_width", 68.0)
    non_default = {k: v for k, v in pitch_dims.items()
                   if k not in ("pitch_length", "pitch_width")}
    if pitch_length != 105.0 or pitch_width != 68.0 or non_default:
        if non_default:
            logger.info(f"  Custom pitch geometry: {pitch_length}x{pitch_width}m + "
                        f"markings overridden ({sorted(non_default.keys())})")
        else:
            logger.info(f"  Custom pitch dimensions: {pitch_length}x{pitch_width}m")

    # Push the resolved geometry into the global active pitch state so the
    # downstream visualizers (get_pitch_template_points, _draw_topdown_pitch)
    # use the same PA / GA / CR sizes as the calibrator. Without this they
    # render FIFA boundaries while the calibrator solves against kids dims,
    # and the overlay looks "wrong" even when the calibration is correct.
    # Mirrors the same setup in pnlcalib_runner.
    if non_default or pitch_length != 105.0 or pitch_width != 68.0:
        from ..annotation.pitch.geometry import SoccerPitch as _SP
        from ..annotation import pitch_constants as _pc
        _pc.set_active_pitch(_SP.from_dict(pitch_dims))

    pitch_template = get_pitch_template_points(pitch_length, pitch_width)

    calibrator = PhysicalCalibrator(
        K=K,
        image_size=(width, height),
        ransac_reproj_error=phys_config.get("ransac_reproj_error", 15.0),
        line_weight=phys_config.get("line_weight", 1.0),
        line_sample_points=phys_config.get("line_sample_points", 20),
        min_line_length_px=phys_config.get("min_line_length_px", 30.0),
        focal_bounds=focal_bounds,
        world_residual_weight=phys_config.get("world_residual_weight", 0.0),
        world_error_threshold=phys_config.get("world_error_threshold", 5.0),
        camera_position=camera_position,
        position_weight=phys_config.get("position_weight", 50.0),
        lock_camera_position=lock_camera_position,
        position_bounds_m=(
            tuple(phys_config["position_bounds_m"])
            if phys_config.get("position_bounds_m") is not None
            else None
        ),
        pitch_bounds_deg=(
            tuple(phys_config["pitch_bounds_deg"])
            if phys_config.get("pitch_bounds_deg") is not None
            else None
        ),
        pitch_dims=pitch_dims,
    )

    # Initialize temporal tracker
    tracker = CameraStateTracker(
        max_reproj_error=phys_config.get("max_reproj_error", 15.0),
    )

    # Results storage. Forward goal_length / goal_height too so downstream
    # consumers (events.shot, etc.) can pick non-FIFA goal frames out of the
    # metadata without having to re-parse the original config.
    extra_vi: dict[str, float] = {
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
    }
    for k in ("goal_length", "goal_height"):
        if k in pitch_dims:
            extra_vi[k] = pitch_dims[k]
    calibration_results = init_calibration_results(
        video_path, video, process_fps,
        extra_video_info=extra_vi,
    )
    homographies = {}
    camera_poses = {}

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_line_dir = vis_dir / "lines"
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_line_dir, vis_calib_dir]:
        d.mkdir(exist_ok=True)

    # ===== Two-phase pipeline: GPU detection then CPU calibration =====
    # Phase A: Read all frames and batch-detect keypoints/lines (GPU-saturated)
    # Phase B: Calibrate from cached detections (CPU-only, no GPU contention)

    batch_size = phys_config.get("detection_batch_size", 32)
    calibration_skip = max(1, int(phys_config.get("calibration_skip", 1)))
    logger.info(f"  Batch detection: batch_size={batch_size}")

    frame_indices = list(sampler)

    # Frame skipping: only detect/calibrate every Nth frame, interpolate the rest
    if calibration_skip > 1:
        calib_frame_indices = frame_indices[::calibration_skip]
        # Always include the last frame for better interpolation at boundaries
        if calib_frame_indices[-1] != frame_indices[-1]:
            calib_frame_indices.append(frame_indices[-1])
        logger.info(f"  Calibration skip={calibration_skip}: calibrating {len(calib_frame_indices)}/{len(frame_indices)} frames, interpolating rest")
    else:
        calib_frame_indices = frame_indices

    detections = _batch_detect_all_frames(
        video_path, calib_frame_indices, kp_detector, line_detector,
        batch_size=batch_size,
    )

    # Phase B: Per-frame calibration (CPU only).
    # vis_interval=1 means every frame gets a calibration vis JPG so the
    # calibration/ folder is a complete per-frame audit. Override to a
    # larger number under field_registration.physical.vis_interval to
    # save disk if needed.
    calibrated_count = 0
    vis_interval = int(phys_config.get("vis_interval", 1))
    joint_frame_data = []

    for idx, frame_idx in enumerate(tqdm(calib_frame_indices, desc="Stage 1 (Physical): Pass 1 - Per-frame")):
        det = detections.get(frame_idx)
        if det is None:
            continue
        frame, keypoints, lines = det

        calibrator.update(keypoints, lines)
        init_rvec, init_tvec = tracker.get_initial_guess()

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

            joint_frame_data.append({
                "frame_idx": frame_idx,
                "rvec": result["camera_params"]["rvec"].ravel(),
                "tvec": result["camera_params"]["tvec"].ravel(),
                "img_pts": result["img_pts"],
                "world_pts": result["world_pts"],
                "line_constraints": result.get("line_constraints", []),
                "f": result["camera_params"]["focal_length"],
            })

            tracker.update(
                result["camera_params"]["rvec"],
                result["camera_params"]["tvec"],
                result["final_error"],
            )
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}
            tracker.last_rvec = None
            tracker.last_tvec = None

        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            json_fname = f"frame_{frame_idx:05d}.json"
            kp_thr = pnl_config.get("keypoint_threshold", 0.3434)
            ln_thr = pnl_config.get("line_threshold", 0.15)
            cv2.imwrite(str(vis_kp_dir / fname),
                        draw_vis_keypoints(frame, keypoints, conf_threshold=kp_thr))
            cv2.imwrite(str(vis_line_dir / fname),
                        draw_vis_lines(frame, lines, conf_threshold=ln_thr))

            vis = _draw_physical_calibration(
                frame, keypoints, lines, result, pitch_template,
                keypoint_mapper, calibrator,
                pitch_length=pitch_length, pitch_width=pitch_width,
            )
            _stamp_source(vis, "Pass 1: PnP")
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

    # === Cross-frame camera-position lock (Pass 2, independent of joint f) ===
    # The camera is fixed (zoom/pan/tilt only). Take the median of Pass 1's
    # per-frame solved C, hard-lock it, and re-solve every frame at 4-DOF
    # (rvec + focal). This forces a single shared C without coupling focal
    # across frames, so the field-zoom case still works. Set
    # `field_registration.physical.lock_position_pass2: false` to skip.
    if (
        phys_config.get("lock_position_pass2", True)
        and len(joint_frame_data) >= 2
    ):
        # Filter to "trustworthy" Pass 1 frames before computing the median.
        good = []
        for fd in joint_frame_data:
            finfo = calibration_results["frames"].get(fd["frame_idx"], {})
            if finfo.get("reprojection_error", 9e9) < 50 and finfo.get("inliers", 0) >= 3:
                good.append(fd)
        if len(good) >= 2:
            positions = []
            for fd in good:
                R_i, _ = cv2.Rodrigues(np.asarray(fd["rvec"]).reshape(3, 1))
                positions.append(-R_i.T @ np.asarray(fd["tvec"]).ravel())
            P = np.stack(positions, axis=0)
            C_med = np.median(P, axis=0)
            spread = float(np.max(np.linalg.norm(P - C_med, axis=1)))
            logger.info(
                "  Pass 2 (lock C): median C=(%.3f, %.3f, %.3f) m  spread=%.2f m  (%d frames)",
                C_med[0], C_med[1], C_med[2], spread, len(good),
            )

            calibrator.camera_position = (
                float(C_med[0]), float(C_med[1]), float(C_med[2]),
            )
            calibrator.lock_camera_position = True
            tracker_lp = CameraStateTracker(
                max_reproj_error=phys_config.get("max_reproj_error", 15.0),
            )
            warm = {fd["frame_idx"]: (fd["rvec"], fd["tvec"]) for fd in joint_frame_data}

            calibrated_count = 0
            calibration_results["frames"] = {}
            homographies = {}
            camera_poses = {}
            joint_frame_data = []  # rebuild for the (optional) joint stage that follows

            for idx, frame_idx in enumerate(tqdm(
                calib_frame_indices,
                desc="Stage 1 (Physical): Pass 2 - Locked C",
            )):
                det = detections.get(frame_idx)
                if det is None:
                    continue
                frame, keypoints, lines = det
                calibrator.update(keypoints, lines)
                if frame_idx in warm:
                    init_rvec, init_tvec = warm[frame_idx][0].copy(), warm[frame_idx][1].copy()
                else:
                    init_rvec, init_tvec = tracker_lp.get_initial_guess()
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
                    joint_frame_data.append({
                        "frame_idx": frame_idx,
                        "rvec": result["camera_params"]["rvec"].ravel(),
                        "tvec": result["camera_params"]["tvec"].ravel(),
                        "img_pts": result["img_pts"],
                        "world_pts": result["world_pts"],
                        "line_constraints": result.get("line_constraints", []),
                        "f": result["camera_params"]["focal_length"],
                    })
                    tracker_lp.update(
                        result["camera_params"]["rvec"],
                        result["camera_params"]["tvec"],
                        result["final_error"],
                    )
                else:
                    calibration_results["frames"][frame_idx] = {"calibrated": False}
                    tracker_lp.last_rvec = None
                    tracker_lp.last_tvec = None

                if idx % vis_interval == 0:
                    fname = f"frame_{frame_idx:05d}.jpg"
                    json_fname = f"frame_{frame_idx:05d}.json"
                    vis_lp = _draw_physical_calibration(
                        frame, keypoints, lines, result, pitch_template,
                        keypoint_mapper, calibrator,
                        pitch_length=pitch_length, pitch_width=pitch_width,
                    )
                    _stamp_source(vis_lp, "Pass 2: locked-C")
                    cv2.imwrite(str(vis_calib_dir / fname), vis_lp)
                    info = _build_frame_json(
                        frame_idx, keypoints, lines, result,
                        warm_start=init_rvec is not None,
                        debug_info=calibrator._last_debug if result is None else None,
                        image_size=(width, height),
                    )
                    with open(vis_calib_dir / json_fname, "w") as jf:
                        jf.write(json.dumps(info, indent=2, default=_json_default))
            tracker = tracker_lp

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
        logger.info(f"  Joint optimization: {len(good_frame_data)}/{len(joint_frame_data)} frames (filtered by error<80px, inliers>=5)...")
        logger.info(f"  Profile f={K[0,0]:.1f}, cx={K[0,2]:.1f} (center), cy={K[1,2]:.1f} (center)")

        # Seed joint optimization with median focal length from Pass 1
        if good_frame_data:
            med_f = float(np.median([fd["f"] for fd in good_frame_data]))
            calibrator.K[0, 0] = med_f
            calibrator.K[1, 1] = med_f
            logger.info(f"  Median f={med_f:.1f}")

        joint_result = calibrator.joint_optimize_intrinsics(good_frame_data)
        if joint_result is not None:
            logger.info(f"  Joint f={joint_result['f']:.1f}, cost={joint_result['cost']:.2f}")

            # Update calibrator with jointly optimized focal length and lock it
            calibrator.K = joint_result["K"].copy()
            eps = 0.01
            calibrator.focal_bounds = (joint_result["f"] - eps, joint_result["f"] + eps)

            # === Pass 2: Re-run per-frame calibration with fixed joint intrinsics ===
            logger.info(f"  Pass 2: Re-calibrating {len(sampler)} frames with joint intrinsics...")
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

            # Reuse cached detections from Phase A (same frames, same models)
            for idx, frame_idx in enumerate(tqdm(calib_frame_indices, desc="Stage 1 (Physical): Pass 2 - Joint intrinsics")):
                det = detections.get(frame_idx)
                if det is None:
                    continue
                frame, keypoints, lines = det
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

                # Save visualizations (Pass 2 joint-intrinsics overwrites Pass 1)
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
                    _stamp_source(vis, "Pass 2: joint-f")
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

            tracker = tracker2
        else:
            logger.info("  Joint optimization failed.")
    else:
        if not do_joint:
            logger.info(f"  Joint optimization disabled (joint_optimize=false). Using profile intrinsics directly.")
        else:
            logger.info(f"  Skipping joint optimization (only {len(joint_frame_data)} calibrated frames)")

    # Store joint intrinsics in results
    if joint_result is not None:
        calibration_results["joint_intrinsics"] = {
            "f": joint_result["f"],
            "cx": joint_result["cx"],
            "cy": joint_result["cy"],
            "cost": joint_result["cost"],
            "n_frames": joint_result["n_frames"],
        }

    # === Optional SIFT-based gap-fill via ChainCalibrator ===
    # Replaces (or supplements) the simple linear interpolation below for
    # frames where calibration failed or was skipped. Uses the calibrated
    # frames as anchors and propagates H through SIFT feature matches
    # between consecutive frames — much more accurate than linear
    # interpolation when the camera is panning / zooming, which is what
    # the kids_soccer footage is doing across long stretches.
    chain_cfg = phys_config.get("gap_fill_chain", {})
    if chain_cfg.get("enabled", False) and camera_poses:
        n_filled = _physical_chain_gap_fill(
            chain_cfg, video_path, frame_indices,
            calibration_results, homographies, camera_poses,
            width=width, height=height, calibrator=calibrator,
            vis_calib_dir=vis_calib_dir,
            pitch_template=pitch_template,
            keypoint_mapper=keypoint_mapper,
            pitch_length=pitch_length,
            pitch_width=pitch_width,
            vis_interval=vis_interval,
        )
        if n_filled:
            logger.info("  Chain gap-fill: filled %d frames via SIFT propagation", n_filled)

    # Interpolate poses for skipped frames (linear; fallback for any frames
    # the chain pass didn't cover or wasn't enabled).
    if calibration_skip > 1 and camera_poses:
        from ..tracking.pitch_projection import _interpolate_camera_poses
        skipped_indices = [f for f in frame_indices if f not in camera_poses]
        if skipped_indices:
            camera_poses = _interpolate_camera_poses(camera_poses, skipped_indices)
            # Derive homographies for interpolated frames
            for fidx in skipped_indices:
                if fidx in camera_poses:
                    pose = camera_poses[fidx]
                    K_i = np.array(pose["K"], dtype=np.float64)
                    R_i, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
                    tvec_i = np.array(pose["tvec"], dtype=np.float64).ravel()
                    H_i = K_i @ np.column_stack([R_i[:, 0], R_i[:, 1], tvec_i])
                    if abs(H_i[2, 2]) > 1e-10:
                        H_i = H_i / H_i[2, 2]
                    homographies[fidx] = H_i
                    calibration_results["frames"][fidx] = {"calibrated": True, "interpolated": True}
            logger.info(f"  Interpolated {len(skipped_indices)} skipped frames from {calibrated_count} calibrated anchors")

    # Save results (base: metadata JSON, homographies pkl, camera_poses pkl + json)
    save_calibration_outputs(
        output_dir, calibration_results, homographies,
        camera_poses=camera_poses, json_default=_json_default,
    )
    # Append derived fields to camera_poses.json
    camera_poses_json_path = output_dir / "camera_poses.json"
    with open(camera_poses_json_path) as f:
        camera_poses_json = json.load(f)
    for fidx_str, entry in camera_poses_json.items():
        pose = camera_poses.get(int(fidx_str))
        if pose is None:
            continue
        R_mat, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
        cam_pos = -R_mat.T @ np.array(pose["tvec"], dtype=np.float64).ravel()
        entry["camera_position"] = {
            "x": round(float(cam_pos[0]), 3),
            "y": round(float(cam_pos[1]), 3),
            "z": round(float(cam_pos[2]), 3),
        }
        entry["focal_length"] = float(np.array(pose["K"])[0, 0])
    with open(camera_poses_json_path, "w") as f:
        json.dump(camera_poses_json, f, indent=2)

    # Statistics
    interpolated_count = sum(
        1 for f in calibration_results["frames"].values()
        if f.get("interpolated")
    )
    # Extra world_error_all stat (physical-specific)
    frames = calibration_results["frames"]
    world_errors_all = [
        frames[idx]["world_error_all"]
        for idx in frames
        if frames[idx].get("calibrated")
        and not frames[idx].get("interpolated")
        and frames[idx].get("world_error_all") is not None
    ]
    extra = {
        "interpolated_frames": interpolated_count,
        "warm_start_frames": tracker.warm_start_count,
        "reinit_frames": tracker.reinit_count,
    }
    if world_errors_all:
        extra["mean_world_error_all"] = float(np.mean(world_errors_all))
        extra["median_world_error_all"] = float(np.median(world_errors_all))

    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
        exclude_interpolated=True, extra_stats=extra,
    )

    print_calibration_summary(stats, label="Stage 1 (Physical)")
    if interpolated_count > 0:
        logger.info(f"  Interpolated: {interpolated_count} frames (skip={calibration_skip})")
    logger.info(f"  Warm-starts: {tracker.warm_start_count}, Re-inits: {tracker.reinit_count}")
    if "median_world_error_all" in stats:
        logger.info(f"  Median world error (all 57):   {stats['median_world_error_all']:.2f} m")
    if joint_result is not None:
        logger.info(f"  Joint f={joint_result['f']:.1f}")

    return stats


def _physical_chain_gap_fill(
    chain_cfg: dict,
    video_path: Path,
    frame_indices: list[int],
    calibration_results: dict,
    homographies: dict[int, np.ndarray],
    camera_poses: dict[int, dict],
    width: int,
    height: int,
    calibrator=None,
    vis_calib_dir: Path | None = None,
    pitch_template=None,
    keypoint_mapper=None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    vis_interval: int = 1,
) -> int:
    """SIFT-based per-frame gap fill for physical-backend outputs.

    For each frame that needs filling:

      1. Find the nearest left + right PnP anchor (bracketing fi).
      2. SIFT-match anchor↔target image pairs.
      3. Back-project the anchor's matched pixels to ground (z=0) via the
         anchor's H to get world coords.
      4. Run 4-DOF PnP (rvec + focal, lock C) on the target frame using
         (target image pixels, ground-plane world coords) as the input.
      5. Average left and right solutions weighted by frame-distance.

    This avoids the chain-accumulation drift of ChainCalibrator (which
    multiplied delta_H matrices into a non-rotational H, then we had to
    SVD-project back onto SO(3), losing fidelity). Each chain-filled
    frame here is independently solved with an over-determined PnP on
    >= ~20 SIFT inliers; the locked C eliminates focal/distance ambiguity.
    """
    from .homography_chain.feature_matcher import FeatureMatcher
    from ..tracking.pitch_projection import (
        _interpolate_camera_poses as _lin_interp_poses,
    )

    # Optional GPU backend (LightGlue + SuperPoint) — ~20× faster on
    # 4K, available when ``gap_fill_chain.matcher_backend: lightglue``
    # is set in config.
    matcher_backend = (
        chain_cfg.get("matcher_backend", "sift") or "sift"
    ).lower()
    if matcher_backend == "lightglue":
        try:
            from .homography_chain.lightglue_matcher import (
                LightGlueFeatureMatcher,
            )
            FeatureMatcherCls = LightGlueFeatureMatcher
            logger.info("  Chain gap-fill: using LightGlue+SuperPoint backend")
        except ImportError as exc:
            logger.warning(
                "  Chain gap-fill: lightglue not installed (%s), "
                "falling back to SIFT", exc,
            )
            FeatureMatcherCls = FeatureMatcher
    else:
        FeatureMatcherCls = FeatureMatcher

    ff = calibration_results["frames"]
    anchor_max_err = float(chain_cfg.get("anchor_max_reproj_px", 30.0))
    overwrite_above = float(chain_cfg.get("overwrite_above_reproj_px", 30.0))
    chain_step = int(chain_cfg.get("frame_step", 3))
    min_inliers = int(chain_cfg.get("min_sift_inliers", 12))
    # Anchors farther than this many frames from the target are skipped —
    # SIFT matches across a long camera-pan-and-zoom span lock onto background
    # features (trees, buildings) instead of pitch-plane features, and the
    # solved rvec/focal end up biased away from the local truth. With ~30 frame
    # max distance the bracket covers ~1 second of motion which the camera
    # operator can't pan out of meaningfully.
    max_anchor_dist = int(chain_cfg.get("max_anchor_distance_frames", 30))
    # When only one anchor (left OR right) is available — typically at the
    # video boundaries — relax the distance limit to this. SIFT match quality
    # degrades with distance but a wide-window match is still better than the
    # linear-extrapolation fallback those edge frames would otherwise get.
    max_anchor_dist_oneside = int(
        chain_cfg.get("max_anchor_distance_oneside_frames", max_anchor_dist * 2)
    )
    # When neither a PnP nor previously chain-solved anchor is near enough,
    # promote already-solved chain frames into the anchor pool and try again.
    # Each chain frame stores its hop_count (PnP anchor = 0); a target's
    # solution gets hop = min(neighbour hop) + 1. Cap the depth so errors
    # don't snowball across the video.
    max_chain_hops = int(chain_cfg.get("max_chain_hops", 3))

    # Top of frame typically has a fixed scoreboard / banner overlay (OSD)
    # that doesn't move with the camera, plus distant sky / buildings whose
    # frame-to-frame parallax is sub-pixel. Both produce SIFT inliers with
    # ~0 displacement that drown out the few real on-pitch matches in the
    # RANSAC consensus. Mask off the top fraction so SIFT only sees the
    # pitch + foreground. 0.25 = ignore top 25 % (270 px on 1080p).
    sift_top_skip_frac = float(chain_cfg.get("sift_top_skip_frac", 0.25))
    sift_mask = np.full((height, width), 255, dtype=np.uint8)
    sift_mask_top = int(round(height * sift_top_skip_frac))
    if sift_mask_top > 0:
        sift_mask[:sift_mask_top, :] = 0
    logger.info(
        "  Chain gap-fill: SIFT mask blanks top %d px (top_skip_frac=%.2f) "
        "to skip OSD overlay and far sky.",
        sift_mask_top, sift_top_skip_frac,
    )

    # Collect anchors: calibrated, non-interpolated, low reprojection.
    anchor_frames = sorted(
        fi for fi, rec in ff.items()
        if rec.get("calibrated")
        and not rec.get("interpolated")
        and (rec.get("reprojection_error") or 9e9) <= anchor_max_err
    )
    if len(anchor_frames) < 1:
        logger.warning("  Chain gap-fill: no anchors after filtering — skipped")
        return 0

    # Need C lock for 4-DOF PnP.
    C_target = (
        np.array(calibrator.camera_position, dtype=np.float64)
        if getattr(calibrator, "camera_position", None) is not None
        else None
    )
    if C_target is None:
        logger.warning("  Chain gap-fill: camera_position not set — skipped")
        return 0

    # Determine which target frames need filling. We previously clamped to
    # the anchor span, but that left frames after the last anchor with only
    # linear interpolation (which is meaningless on a one-sided edge — there
    # is no right anchor to interpolate toward). Now we let chain extend
    # past the last anchor; the per-frame anchor selection below uses
    # max_anchor_distance_frames to bail out cleanly when the only available
    # anchor is too far away.
    target_lo = 0
    target_hi = max(frame_indices)
    target_set = list(range(target_lo, target_hi + 1, chain_step))
    targets = []
    for fi in target_set:
        if fi in anchor_frames:
            continue  # don't touch anchors
        existing = ff.get(fi, {})
        is_interp = existing.get("interpolated", False)
        existing_err = existing.get("reprojection_error", 9e9)
        if not is_interp and existing_err <= overwrite_above and existing.get("calibrated"):
            continue
        targets.append(fi)

    logger.info(
        "  Chain gap-fill: %d anchors, %d targets in span [%d, %d], step=%d",
        len(anchor_frames), len(targets), target_lo, target_hi, chain_step,
    )

    # Pre-load anchor frame images and their inverse-H (to back-project
    # pixels -> ground). Anchor count is small (~30) so we keep them in RAM.
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  Chain gap-fill: failed to open video — skipped")
        return 0

    anchor_imgs: dict[int, np.ndarray] = {}
    anchor_H_inv: dict[int, np.ndarray] = {}
    for af in anchor_frames:
        H_a = homographies.get(af)
        if H_a is None:
            continue
        try:
            H_a_inv = np.linalg.inv(np.array(H_a, dtype=np.float64))
        except np.linalg.LinAlgError:
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, af)
        ok, frame = cap.read()
        if not ok:
            continue
        anchor_imgs[af] = frame
        anchor_H_inv[af] = H_a_inv

    matcher = FeatureMatcherCls(
        n_features=2000,
        ratio_threshold=0.75,
        min_matches=min_inliers,
    )

    # Cache extracted SIFT features per anchor so we don't re-extract on
    # every target. Only kp/desc are cached; raw image already in anchor_imgs.
    anchor_feat: dict[int, tuple] = {}
    for af, img in anchor_imgs.items():
        anchor_feat[af] = matcher.extract_features(img, mask=sift_mask)

    def _solve_one_anchor(target_img: np.ndarray, af: int) -> tuple[np.ndarray, float] | None:
        """SIFT match anchor->target; run 4-DOF PnP. Returns (rvec, focal)."""
        kp_a, desc_a = anchor_feat[af]
        kp_t, desc_t = matcher.extract_features(target_img, mask=sift_mask)
        if desc_a is None or desc_t is None:
            return None
        matches = matcher.match_features(kp_a, desc_a, kp_t, desc_t)
        if len(matches) < min_inliers:
            return None
        # Compute homography for inlier mask only.
        _H, mask, meta = matcher.compute_homography(matches)
        if mask is None or int(mask.sum()) < min_inliers:
            return None
        # Anchor pixels (inliers) and target pixels (inliers)
        inlier = mask.flatten().astype(bool)
        pts_a = np.float32([m[0].pt for m in matches])[inlier]
        pts_t = np.float32([m[1].pt for m in matches])[inlier]
        # Back-project anchor pixels to ground via H_a_inv (image -> world).
        H_inv = anchor_H_inv[af]
        ones = np.ones((len(pts_a), 1), dtype=np.float64)
        ph = np.hstack([pts_a, ones])              # (N, 3) homogeneous img coords
        wh = (H_inv @ ph.T).T                      # (N, 3)
        # Filter degenerate w<=0 entries
        good = np.abs(wh[:, 2]) > 1e-9
        if good.sum() < min_inliers:
            return None
        wh = wh[good]; pts_t = pts_t[good]
        world_xy = wh[:, :2] / wh[:, 2:3]          # (N, 2) world ground coords
        # Pitch sanity: keep only matches whose ground projection is on/near pitch.
        # pitch_length / pitch_width are the helper's params (forwarded from
        # the runner) — earlier code used a 60/40 FIFA fallback that on a
        # kids pitch let "ground" points 80–100 m behind the actual goal line
        # count as on-pitch, while still failing to gather ≥min_inliers true
        # on-pitch points. With kids dims (33×22) the buffer is much tighter.
        L_half = float(pitch_length) / 2.0
        W_half = float(pitch_width) / 2.0
        on_pitch = (
            (np.abs(world_xy[:, 0]) <= L_half + 30) &
            (np.abs(world_xy[:, 1]) <= W_half + 30)
        )
        if on_pitch.sum() < min_inliers:
            return None
        world_xy = world_xy[on_pitch]; pts_t = pts_t[on_pitch]
        world_3d = np.hstack([world_xy, np.zeros((len(world_xy), 1))])

        # Use anchor's pose as fallback warm-start.
        pose_a = camera_poses[af]
        rvec_anchor = np.array(pose_a["rvec"], dtype=np.float64)
        f_init = float(pose_a["K"][0][0])
        K_init = np.array(
            [[f_init, 0, width / 2.0], [0, f_init, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )

        # ===== Second RANSAC: PnP-on-ground =====
        # The first RANSAC (inside compute_homography above) verified
        # image-to-image consistency only — it can't tell genuine ground
        # features from off-ground SIFT matches (rooftops, scoreboards
        # behind the goal). Those non-ground points get reprojected to
        # bogus world coords by H_a^-1 and bias the 4-DOF LM into a
        # wrap-around rvec.
        # solvePnPRansac re-checks consistency in 3D: each random 4-pt
        # EPNP draws a full 6-DOF (R, t) and only points on the actual
        # ground plane stay in the inlier set. Then we feed the cleaned
        # subset to the 4-DOF refine, with the RANSAC's own (rvec, tvec)
        # as warm-start (already near target's true pose, not anchor's).
        n_pts = len(world_3d)
        if n_pts < 4:
            return None
        ok, rvec_pnp, tvec_pnp, inlier_idx = cv2.solvePnPRansac(
            objectPoints=world_3d.astype(np.float64),
            imagePoints=pts_t.astype(np.float64),
            cameraMatrix=K_init,
            distCoeffs=np.zeros(5, dtype=np.float64),
            iterationsCount=200,
            reprojectionError=8.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or inlier_idx is None or len(inlier_idx) < min_inliers:
            return None
        inlier_idx = inlier_idx.flatten()
        world_3d = world_3d[inlier_idx]
        pts_t = pts_t[inlier_idx]
        rvec_init = rvec_pnp.ravel()
        # ========================================

        # 4-DOF LM refine (rvec + focal; tvec = -R·C_locked).
        K_save = calibrator.K.copy()
        focal_save = calibrator.focal_bounds
        try:
            calibrator.K = K_init.copy()
            tvec_init = -cv2.Rodrigues(rvec_init)[0] @ C_target
            rvec_opt, _tvec_opt, f_opt = calibrator._refine_7dof(
                rvec_init, tvec_init, pts_t.astype(np.float64), world_3d, [],
            )
        finally:
            calibrator.K = K_save
            calibrator.focal_bounds = focal_save

        # Final reprojection error gate (Plan A complement). The PnP
        # RANSAC already filtered most outliers, but the LM step may
        # have drifted slightly under cauchy loss; if median reproj
        # error is still high after refine, the solution isn't trust-
        # worthy enough to either keep or promote into hop-2 anchor.
        R_chk, _ = cv2.Rodrigues(rvec_opt)
        t_chk = -R_chk @ C_target
        K_chk = np.array(
            [[f_opt, 0, width / 2.0], [0, f_opt, height / 2.0], [0, 0, 1]],
            dtype=np.float64,
        )
        proj, _ = cv2.projectPoints(
            world_3d, rvec_opt, t_chk.reshape(3, 1),
            K_chk, np.zeros(5, dtype=np.float64),
        )
        reproj_err = float(np.median(
            np.linalg.norm(proj.reshape(-1, 2) - pts_t, axis=1)
        ))
        max_reproj = float(chain_cfg.get("solve_max_reproj_px", 30.0))
        if reproj_err > max_reproj:
            return None
        return rvec_opt, float(f_opt)

    # Iterate targets across multiple hops: each hop tries to solve all
    # remaining targets using the current anchor pool. Successfully solved
    # chain frames join the pool (with hop_count = parent hop + 1) so they
    # can serve as bracketing anchors for further-out targets in the next
    # hop. Stop when no progress is made or we hit max_chain_hops.
    #
    # hop_count tracks how many SIFT-PnP steps a frame is removed from a
    # real PnP anchor. PnP anchors = 0; first-hop chain = 1; etc. A frame
    # at hop k can only be used as anchor for new targets if k < max_chain_hops.
    hop_count: dict[int, int] = {af: 0 for af in anchor_frames}
    extended_anchor_imgs: dict[int, np.ndarray] = dict(anchor_imgs)
    extended_anchor_H_inv: dict[int, np.ndarray] = dict(anchor_H_inv)
    extended_anchor_feat: dict[int, tuple] = dict(anchor_feat)
    pending = set(targets)
    n_updated = 0
    updated_frames: list[int] = []

    for hop in range(max_chain_hops):
        # Build the per-hop anchor pool: PnP anchors are always available;
        # chain-promoted anchors are available only if their hop_count < hop+1
        # (we don't allow a chain anchor to seed something at the same hop
        # before all hop-h targets settle, otherwise the order of iteration
        # would non-deterministically affect results).
        usable_anchors = sorted(
            af for af, hc in hop_count.items() if hc <= hop
        )
        if not usable_anchors:
            break

        solved_this_hop: list[int] = []
        for fi in sorted(pending):
            left = max((a for a in usable_anchors if a <= fi), default=None)
            right = min((a for a in usable_anchors if a >= fi), default=None)
            has_two = left is not None and right is not None
            limit = max_anchor_dist if has_two else max_anchor_dist_oneside
            if left is not None and (fi - left) > limit:
                left = None
            if right is not None and (right - fi) > limit:
                right = None
            candidates = [
                a for a in (left, right)
                if a is not None and a in extended_anchor_imgs
            ]
            if not candidates:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, target_img = cap.read()
            if not ok:
                continue

            # Patch the closures so _solve_one_anchor sees the extended pool.
            anchor_imgs.update(extended_anchor_imgs)
            anchor_H_inv.update(extended_anchor_H_inv)
            anchor_feat.update(extended_anchor_feat)

            solutions = []
            for af in candidates:
                sol = _solve_one_anchor(target_img, af)
                if sol is not None:
                    solutions.append((af, sol[0], sol[1]))
            if not solutions:
                continue
            weights = np.array(
                [1.0 / max(abs(fi - af), 1) for af, _, _ in solutions],
                dtype=np.float64,
            )
            weights /= weights.sum()
            rvec_avg = sum(w * r for w, (_, r, _) in zip(weights, solutions))
            f_avg = sum(w * f for w, (_, _, f) in zip(weights, solutions))
            R_avg, _ = cv2.Rodrigues(rvec_avg)
            t_locked = -R_avg @ C_target
            K_out = np.array(
                [[f_avg, 0, width / 2.0], [0, f_avg, height / 2.0], [0, 0, 1]],
                dtype=np.float64,
            )
            H_locked = K_out @ np.column_stack(
                [R_avg[:, 0], R_avg[:, 1], t_locked]
            )
            if abs(H_locked[2, 2]) > 1e-10:
                H_locked = H_locked / H_locked[2, 2]
            homographies[fi] = H_locked
            camera_poses[fi] = {
                "K": K_out.tolist(),
                "rvec": rvec_avg.ravel().tolist(),
                "tvec": t_locked.ravel().tolist(),
                "dist_coeffs": [0.0] * 5,
            }
            # hop_count for fi = min hop of its parent anchors + 1
            parent_hops = [hop_count[af] for af, _, _ in solutions if af in hop_count]
            new_hop = (min(parent_hops) if parent_hops else hop) + 1
            hop_count[fi] = new_hop

            src_label = (
                "sift_pnp_left+right" if len(solutions) == 2
                else "sift_pnp_one_anchor"
            )
            if new_hop > 1:
                src_label += f"_h{new_hop}"
            ff[fi] = {
                "calibrated": True,
                "interpolated": False,
                "chain_filled": True,
                "reprojection_error": None,
                "source": src_label,
            }
            n_updated += 1
            updated_frames.append(fi)
            solved_this_hop.append(fi)

            # Promote into the anchor pool for the next hop. We need the H
            # to back-project SIFT inliers (computed above as H_locked) plus
            # the target image's SIFT features (re-extract once, cached).
            extended_anchor_imgs[fi] = target_img.copy()
            try:
                extended_anchor_H_inv[fi] = np.linalg.inv(H_locked)
            except np.linalg.LinAlgError:
                continue
            kp_fi, desc_fi = matcher.extract_features(target_img, mask=sift_mask)
            extended_anchor_feat[fi] = (kp_fi, desc_fi)

        if not solved_this_hop:
            logger.info("  Chain gap-fill: hop %d converged (no new solves)", hop + 1)
            break
        logger.info(
            "  Chain gap-fill: hop %d solved %d frames; %d remaining",
            hop + 1, len(solved_this_hop), len(pending) - len(solved_this_hop),
        )
        pending -= set(solved_this_hop)
        if not pending:
            break

    cap.release()
    logger.info("  Chain gap-fill: solved %d / %d targets via single-step SIFT+PnP",
                n_updated, len(targets))

    # Re-render calibration vis for chain-filled frames so the JPGs reflect
    # the new (better) projection. Open the video once and seek for each
    # frame that needs a fresh vis.
    if (
        vis_calib_dir is not None
        and pitch_template is not None
        and keypoint_mapper is not None
        and calibrator is not None
        and updated_frames
    ):
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            n_vis = 0
            for idx, fi in enumerate(sorted(updated_frames)):
                # Honour the same vis_interval as Pass 1/2 so we don't
                # explode disk by writing every chain-filled frame.
                if vis_interval > 1 and (idx % vis_interval) != 0:
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ok, frame = cap.read()
                if not ok:
                    continue
                pose = camera_poses[fi]
                rvec_p = np.array(pose["rvec"], dtype=np.float64)
                R_p, _ = cv2.Rodrigues(rvec_p)
                synth_result = {
                    "homography": homographies[fi],
                    "camera_params": {
                        "K": np.array(pose["K"], dtype=np.float64),
                        "rvec": rvec_p,
                        "tvec": np.array(pose["tvec"], dtype=np.float64),
                        "dist_coeffs": np.array(pose["dist_coeffs"], dtype=np.float64),
                        "R": R_p,
                    },
                }
                vis = _draw_physical_calibration(
                    frame, [], [], synth_result, pitch_template,
                    keypoint_mapper, calibrator,
                    pitch_length=pitch_length, pitch_width=pitch_width,
                )
                # Stamp a marker so we know this came from chain
                src_lbl = ff[fi].get("source", "chain")
                _stamp_source(vis, f"Chain: {src_lbl}")
                cv2.imwrite(str(vis_calib_dir / f"frame_{fi:05d}.jpg"), vis)
                n_vis += 1
            cap.release()
            logger.info(
                "  Chain gap-fill: re-rendered %d vis frames in %s",
                n_vis, vis_calib_dir,
            )

    return n_updated


def _batch_detect_all_frames(
    video_path: Path,
    frame_indices: list[int],
    kp_detector,
    line_detector,
    batch_size: int = 8,
) -> dict[int, tuple[np.ndarray, list, list]]:
    """Read frames and batch-detect keypoints/lines with pipelined I/O.

    Uses a thread pool for parallel frame reading + preprocessing (CPU),
    while the main thread runs GPU inference. This keeps the GPU saturated.

    Returns:
        Dict mapping frame_idx -> (frame, keypoints, lines).
    """
    import threading
    import queue
    from concurrent.futures import ThreadPoolExecutor
    import torch

    detections: dict[int, tuple[np.ndarray, list, list]] = {}
    use_lines = line_detector is not None
    n_total = len(frame_indices)
    num_readers = 8  # Parallel video readers

    # Split frame_indices into batch-sized chunks
    batch_chunks = [
        frame_indices[i : i + batch_size]
        for i in range(0, n_total, batch_size)
    ]

    # --- Worker: read frame + preprocess to tensor (CPU-heavy, parallelized) ---
    def _read_and_preprocess(args):
        """Read a single frame and preprocess to tensors. Each call uses its own cap."""
        fidx, vid_path = args
        cap = cv2.VideoCapture(str(vid_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        kp_tensor = kp_detector.preprocess(frame)
        line_tensor = line_detector.preprocess(frame) if use_lines else None
        return (fidx, frame, kp_tensor, line_tensor)

    # Pipeline: thread pool reads+preprocesses frames, delivers ready batches
    # to main thread which only does GPU forward + post-process.
    batch_queue: queue.Queue = queue.Queue(maxsize=3)

    def _loader_thread():
        pool = ThreadPoolExecutor(max_workers=num_readers)
        for chunk_indices in batch_chunks:
            futures = [
                pool.submit(_read_and_preprocess, (fidx, video_path))
                for fidx in chunk_indices
            ]
            results = [f.result() for f in futures]
            # Filter None (failed reads), keep order
            valid = [r for r in results if r is not None]
            batch_queue.put(valid)
        batch_queue.put(None)  # Sentinel
        pool.shutdown(wait=False)

    loader = threading.Thread(target=_loader_thread, daemon=True)
    loader.start()

    # --- Main thread: GPU forward pass only ---
    pbar = tqdm(total=len(batch_chunks), desc="Stage 1 (Physical): Batch detect")
    while True:
        item = batch_queue.get()
        if item is None:
            break
        if not item:
            pbar.update(1)
            continue

        valid_indices = [r[0] for r in item]
        raw_frames = [r[1] for r in item]
        kp_tensors = [r[2] for r in item]
        line_tensors = [r[3] for r in item] if use_lines else None

        # Keypoint detection: stack pre-built tensors → GPU forward → post-process
        kp_batch = torch.stack(kp_tensors, dim=0).to(kp_detector.device)
        with torch.no_grad():
            kp_heatmaps = kp_detector.model(kp_batch)
        batch_kps = _postprocess_kp_batch(kp_detector, kp_heatmaps, raw_frames)

        # Line detection
        if use_lines and line_tensors is not None:
            line_batch = torch.stack(line_tensors, dim=0).to(line_detector.device)
            with torch.no_grad():
                line_heatmaps = line_detector.model(line_batch)
            batch_lines = _postprocess_line_batch(line_detector, line_heatmaps, raw_frames)
        else:
            batch_lines = [[] for _ in raw_frames]

        for i, fidx in enumerate(valid_indices):
            detections[fidx] = (raw_frames[i], batch_kps[i], batch_lines[i])

        pbar.update(1)

    pbar.close()
    loader.join()
    return detections


def _postprocess_kp_batch(kp_detector, heatmaps, frames):
    """Post-process keypoint heatmaps for a batch of frames."""
    from .pnlcalib import get_keypoints_from_heatmap_maxpool

    results = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        hm = heatmaps[i : i + 1]

        if kp_detector.backend == "pnlcalib":
            hm_for_extraction = hm[:, :-1, :, :]
            hm_h, hm_w = hm.shape[2], hm.shape[3]
            scale_x = w / hm_w
            scale_y = h / hm_h

            keypoints = get_keypoints_from_heatmap_maxpool(
                hm_for_extraction, scale=1, max_keypoints=1,
                threshold=kp_detector.confidence_threshold,
            )
            for kp in keypoints:
                kp["x"] = kp["x"] * scale_x
                kp["y"] = kp["y"] * scale_y
        else:
            keypoints = kp_detector._extract_keypoints_from_heatmaps(
                hm[0].cpu().numpy(), original_size=(w, h),
            )
        results.append(keypoints)
    return results


def _postprocess_line_batch(line_detector, heatmaps, frames):
    """Post-process line heatmaps for a batch of frames."""
    from .pnlcalib import get_lines_from_heatmap_maxpool
    from .pnlcalib.hrnet_line import HRNetLineModel

    results = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        hm = heatmaps[i : i + 1]

        hm_for_extraction = hm[:, :-1, :, :]
        hm_h, hm_w = hm.shape[2], hm.shape[3]
        scale_x = w / hm_w
        scale_y = h / hm_h

        lines = get_lines_from_heatmap_maxpool(
            hm_for_extraction, scale=1, threshold=line_detector.confidence_threshold,
        )
        for line in lines:
            line["x1"] *= scale_x
            line["y1"] *= scale_y
            line["x2"] *= scale_x
            line["y2"] *= scale_y
            line["length"] = float(np.sqrt(
                (line["x2"] - line["x1"])**2 + (line["y2"] - line["y1"])**2
            ))
        for line in lines:
            dx = line["x2"] - line["x1"]
            dy = line["y2"] - line["y1"]
            line["angle"] = float(np.arctan2(dy, dx) * 180 / np.pi)
            line["class_name"] = HRNetLineModel.get_line_class_name(line["id"])
        results.append(lines)
    return results


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
                    "kp_id": dbg_ids[i] if i < len(dbg_ids) else -1,
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
            line_details = []
            for lc in lc_list:
                ws = np.array(lc["world_samples"])
                imgs = np.array(lc["img_samples"])

                # Project 3D world samples with optimized pose
                proj = project_points_2d(
                    ws, cam["rvec"], cam["tvec"],
                    cam["K"], cam["dist_coeffs"],
                )

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
                projected = project_points_2d(
                    np.array(world_pts), cam["rvec"], cam["tvec"],
                    cam["K"], cam["dist_coeffs"],
                )

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
            from .pnlcalib import KeypointMapper

            all_world = KeypointMapper.PNLCALIB_WORLD_COORDS_2D
            non_ground = KeypointMapper.NON_GROUND_KEYPOINTS
            detected_ids = {kp["id"] for kp in keypoints if kp.get("confidence", 0) >= 0.3}
            img_w, img_h = image_size if image_size else (1920, 1080)
            margin = 50

            projected_in_image = []  # template points that project into image
            phantom_count = 0  # projected in image but not detected
            missing_count = 0  # detected but projected outside image

            ground_ids = [kid for kid in range(len(all_world)) if kid not in non_ground]
            ground_world = np.array(
                [[all_world[k][0], all_world[k][1], 0.0] for k in ground_ids],
                dtype=np.float64,
            )
            ground_proj = project_points_2d(
                ground_world, cam["rvec"], cam["tvec"],
                cam["K"], cam["dist_coeffs"],
            )
            for proj_idx, kid in enumerate(ground_ids):
                ix, iy = float(ground_proj[proj_idx, 0]), float(ground_proj[proj_idx, 1])
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

        # Draw projected pitch lines (yellow). Skip segments whose BOTH raw
        # endpoints (before clipping) fall outside the image+margin — those
        # are "phantom" segments where two far-side world points project past
        # opposite image edges and the connecting line crosses the visible
        # frame as an artifact (e.g. the opposite-half penalty area drawn
        # under steep perspective). Without this guard, cv2.clipLine
        # faithfully renders that crossing as a phantom stroke through the
        # image even though no such line is visible in the scene.
        margin = 200
        for name, pts in projected.items():
            for i in range(len(pts) - 1):
                if pts[i] is None or pts[i + 1] is None:
                    continue
                try:
                    p1 = (int(float(pts[i][0])), int(float(pts[i][1])))
                    p2 = (int(float(pts[i + 1][0])), int(float(pts[i + 1][1])))
                except (ValueError, OverflowError, TypeError):
                    continue
                p1_in = -margin < p1[0] < w + margin and -margin < p1[1] < h + margin
                p2_in = -margin < p2[0] < w + margin and -margin < p2[1] < h + margin
                if not p1_in and not p2_in:
                    continue  # phantom — both endpoints off-frame
                # cv2.clipLine(rect=(x, y, w, h)) clips to [x, x+w) × [y, y+h).
                ok, c1, c2 = cv2.clipLine(
                    (-margin, -margin, w + 2 * margin, h + 2 * margin),
                    p1, p2,
                )
                if ok:
                    cv2.line(vis, c1, c2, (0, 255, 255), 2)

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
                proj = project_points_2d(
                    ws, camera_params["rvec"], camera_params["tvec"],
                    camera_params["K"], camera_params["dist_coeffs"],
                )
                for i in range(len(proj) - 1):
                    px1 = (int(proj[i][0]), int(proj[i][1]))
                    px2 = (int(proj[i + 1][0]), int(proj[i + 1][1]))
                    p1_ok = -margin < px1[0] < w + margin and -margin < px1[1] < h + margin
                    p2_ok = -margin < px2[0] < w + margin and -margin < px2[1] < h + margin
                    if p1_ok or p2_ok:
                        clamp = w + h
                        c1 = (max(-clamp, min(clamp, px1[0])), max(-clamp, min(clamp, px1[1])))
                        c2 = (max(-clamp, min(clamp, px2[0])), max(-clamp, min(clamp, px2[1])))
                        cv2.line(vis, c1, c2, (255, 0, 0), 1)

        # Draw projected template keypoints (yellow, matching pitch lines)
        from .pnlcalib import KeypointMapper
        all_world = calibrator._field_world_coords
        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS
        ground_ids = [kid for kid in range(len(all_world)) if kid not in non_ground]
        ground_world = np.array(
            [[all_world[k][0], all_world[k][1], 0.0] for k in ground_ids],
            dtype=np.float64,
        )
        ground_proj = project_points_2d(
            ground_world, camera_params["rvec"], camera_params["tvec"],
            camera_params["K"], camera_params["dist_coeffs"],
        )
        for proj_idx, kid in enumerate(ground_ids):
            ix, iy = ground_proj[proj_idx]
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
