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
    """Run Stage 1 using PnLCalib backend.

    When ``field_registration.pnlcalib_orig.enabled`` is set, delegates to
    the upstream-aligned port (``pnlcalib_orig``) instead of the project's
    in-tree refactor. Same output contract — ``homographies.pkl`` and
    ``calibration_metadata.json`` — so downstream tracking is unchanged.
    """
    fr_config = config.get("field_registration", {})
    if fr_config.get("pnlcalib_orig", {}).get("enabled"):
        return _run_stage1_pnlcalib_orig(
            video_path, output_dir, vis_dir, config, process_fps,
        )

    from . import KeypointDetector, LineDetector
    from .pnlcalib import (
        FramebyFrameCalib,
        KeypointMapper,
        LineMapper,
    )

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
    line_mapper = LineMapper(pitch_dims=config.get("pitch") or None)
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
        line_detector.load_model(pnl_config.get("line_model_path"))
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

    # Create calibrator once (image_size and alpha don't change per frame).
    # Forward pitch_dims so non-FIFA youth/7-a-side configs drive the line
    # intersection world coords correctly.
    pitch_cfg_legacy = config.get("pitch", {})
    calibrator = FramebyFrameCalib(
        image_size=(width, height), alpha=0.7,
        pitch_dims=pitch_cfg_legacy or None,
    )

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
            cv2.imwrite(
                str(vis_kp_dir / fname),
                draw_vis_keypoints(
                    frame, keypoints,
                    conf_threshold=pnl_config.get("keypoint_threshold", 0.3434),
                ),
            )
            if vis_line_dir is not None:
                cv2.imwrite(
                    str(vis_line_dir / fname),
                    draw_vis_lines(
                        frame, lines,
                        conf_threshold=pnl_config.get("line_threshold", 0.15),
                    ),
                )
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


def _run_stage1_pnlcalib_orig(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using the upstream-aligned ``pnlcalib_orig`` backend.

    Same I/O contract as ``_run_stage1_pnlcalib`` (writes
    ``homographies.pkl`` + ``calibration_metadata.json``). Differences vs.
    the in-tree ``pnlcalib`` path:

    - Calibration uses ``cv2.calibrateCamera`` (Zhang multi-plane) +
      18-combo ``heuristic_voting`` (3 modes × 6 RANSAC thresholds), per
      upstream ``mguti97/PnLCalib``.
    - Crossbar (z != 0) keypoints are fed in directly via the upstream
      3-plane reparameterization rather than the project's PnP RANSAC.

    Output ``H`` matrices are stored in the project's **y-up centered
    world frame** so downstream tracking (``orchestrator.py`` inverts H
    to map image points to pitch coords) sees the same convention as the
    in-tree ``pnlcalib`` path. Upstream solves in y-down; we negate
    column 1 of R to bridge (equivalent to ``R_up = R_dn @ diag(1,-1,1)``).
    """
    from . import KeypointDetector, LineDetector
    from .pnlcalib_orig import FramebyFrameCalib, PnLCalibIdMap
    from ..annotation.pitch.geometry import SoccerPitch

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})
    pitch_cfg = config.get("pitch", {})

    logger.info("Stage 1: Initializing keypoint detector (HRNet/PnLCalib)...")
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_config.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_config.get("keypoint_model_path"),
            "confidence_threshold": pnl_config.get("keypoint_threshold", 0.3434),
        },
    }
    kp_detector = KeypointDetector(kp_config)
    kp_detector.load_model()

    use_lines = pnl_config.get("use_lines", False)
    line_detector = None
    if use_lines:
        logger.info("Stage 1: Initializing line detector (HRNet/PnLCalib)...")
        line_config = {
            "backend": "pnlcalib",
            "pnlcalib": {
                "weights": pnl_config.get("line_weights", "SV_lines"),
                "confidence_threshold": pnl_config.get("line_threshold", 0.15),
            },
        }
        line_detector = LineDetector(line_config)
        line_detector.load_model(pnl_config.get("line_model_path"))
    else:
        logger.info("Stage 1: Line detection disabled (keypoint-only mode)")

    voting_th = pnl_config.get("voting_threshold", 5.0)

    # Set active pitch so overlay's get_pitch_template_points() reflects
    # the same dims pnlcalib_orig is solving against. Otherwise the
    # diagnostic overlay would project FIFA boundaries through a kids-
    # frame H and look wrong even when calibration is correct.
    if pitch_cfg:
        from ..annotation.pitch.geometry import SoccerPitch as _SP
        from ..annotation import pitch_constants as _pc
        _pc.set_active_pitch(_SP.from_dict(pitch_cfg))

    pitch_template = get_pitch_template_points()

    video = open_video(video_path)
    cap = video.cap
    sampler = make_sampler(video, process_fps)
    width, height = video.width, video.height

    calibration_results = init_calibration_results(video_path, video, process_fps)
    homographies: dict[int, np.ndarray] = {}

    pitch_obj = SoccerPitch.from_dict(pitch_cfg) if pitch_cfg else SoccerPitch()
    id_map = PnLCalibIdMap(pitch=pitch_obj)
    calibrator = FramebyFrameCalib(
        iwidth=width, iheight=height, denormalize=False, pitch_dims=pitch_cfg,
    )

    # Geometric sanity gate. PnLCalib's heuristic_voting + LM optimisation
    # picks the candidate with the lowest rep_err over 18 (mode, ransac)
    # combinations but doesn't sanity-check the resulting intrinsics. On
    # a sparse / cluttered keypoint set the LM happily lands on a
    # "phantom" solution with a wildly wrong fx (e.g. 19 px on a 1920
    # image, equivalent to ~180° hfov, on a clip whose actual hfov is
    # 43°). The phantom solver fits the points, but the resulting H is
    # geometrically meaningless and downstream chain propagation copies
    # it to neighbours. We have a measured fx prior in the config —
    # reject any solve whose fx falls outside [fx_prior * factor_low,
    # fx_prior * factor_high].
    sanity_cfg = pnl_config.get("geometry_sanity", {})
    sanity_enabled = bool(sanity_cfg.get("enabled", True))
    fx_prior = config.get("intrinsics", {}).get("fx")
    fx_factor_low = float(sanity_cfg.get("fx_factor_low", 0.5))
    fx_factor_high = float(sanity_cfg.get("fx_factor_high", 2.0))

    def _is_pose_intrinsic_sane(fx_solved):
        """Reject solves whose fx deviates far from the measured prior.

        Disabled when ``intrinsics.fx`` isn't configured — without a
        prior we have no reference to flag phantom solutions.
        """
        if fx_prior is None:
            return True, None
        lo = fx_prior * fx_factor_low
        hi = fx_prior * fx_factor_high
        if not (lo <= fx_solved <= hi):
            return False, (
                f"fx={fx_solved:.0f} outside [{lo:.0f}, {hi:.0f}] "
                f"(prior={fx_prior:.0f}, factor=[{fx_factor_low}, {fx_factor_high}])"
            )
        return True, None

    # Optional pre-solver outlier filter. The downstream Zhang-LM has 9
    # free params (fx, fy, cx, cy, R, t) which is enough freedom that a
    # single mis-classified keypoint (e.g. detector reporting id 17 on a
    # left-half frame) can be absorbed into a phantom small-fx solution
    # with deceptively low rep_err. We filter geometric outliers BEFORE
    # the solver sees them by fitting a planar homography with RANSAC,
    # which has exactly 8 dof and rejects mis-IDed points cleanly.
    of_cfg = pnl_config.get("geometric_outlier_filter", {})
    of_enabled = bool(of_cfg.get("enabled", False))
    of_threshold = float(of_cfg.get("ransac_threshold_px", 50.0))
    of_min_keypoints = int(of_cfg.get("min_keypoints", 6))

    from .pnlcalib_orig.pitch_template import build_keypoint_table
    from .pnlcalib_orig.id_mapping import (
        project_kp_to_upstream, NON_GROUND_UPSTREAM_IDS,
    )
    _kp_table = build_keypoint_table(pitch_cfg)["keypoint_world_coords_2D"]

    def _filter_keypoint_outliers(keypoints):
        """Drop keypoints whose world<->image correspondence is
        inconsistent with a planar homography fit on the remaining
        ground-plane points.

        Crossbar (NON_GROUND) points can't fit a single plane so they
        bypass the filter entirely — no-op for them. Returns the
        filtered list, unchanged when fewer than ``of_min_keypoints``
        ground points are available or RANSAC can't find an H.
        """
        ground, non_ground = [], []
        for kp in keypoints:
            up_id = project_kp_to_upstream(int(kp["id"]))
            if up_id in NON_GROUND_UPSTREAM_IDS:
                non_ground.append(kp)
                continue
            if not (1 <= up_id <= len(_kp_table)):
                continue
            wx, wy_dn = _kp_table[up_id - 1]
            if not (np.isfinite(wx) and np.isfinite(wy_dn)):
                continue
            ground.append((kp, wx, -float(wy_dn)))  # upstream y-down -> y-up

        if len(ground) < of_min_keypoints:
            return keypoints  # too few to filter reliably

        img_pts = np.array([[kp["x"], kp["y"]] for kp, _, _ in ground], dtype=np.float64)
        world_pts = np.array([[wx, wy] for _, wx, wy in ground], dtype=np.float64)

        H, mask = cv2.findHomography(
            world_pts, img_pts, method=cv2.RANSAC,
            ransacReprojThreshold=of_threshold,
            maxIters=2000, confidence=0.999,
        )
        if H is None or mask is None:
            return keypoints
        mask = mask.ravel().astype(bool)
        kept = [g[0] for g, m in zip(ground, mask) if m] + non_ground
        if len(kept) < of_min_keypoints:
            # RANSAC was too aggressive — better to feed the solver the
            # full set than starve it of geometry.
            return keypoints
        return kept

    vis_kp_dir = vis_dir / "keypoints"
    vis_calib_dir = vis_dir / "calibration"
    vis_line_dir = vis_dir / "lines" if use_lines else None
    for d in [vis_kp_dir, vis_calib_dir, vis_line_dir]:
        if d is not None:
            d.mkdir(exist_ok=True)

    # Reuse the in-tree KeypointMapper for vis (for non-ground id set + world
    # coords lookup in the top-down panel). The actual calibration above has
    # already used the upstream-aligned PnLCalibIdMap.
    from .pnlcalib import KeypointMapper as _ProjKeypointMapper
    keypoint_mapper_vis = _ProjKeypointMapper()

    class _CalibVisShim:
        """Adapter exposing only the attributes ``draw_vis_calibration``
        / ``_draw_topdown_pitch`` reach into. Keeps shared_vis backend-
        agnostic without leaking pnlcalib_orig internals into it."""
        def __init__(self, w, h):
            self.line_intersections = []
            self.image_size = (w, h)

    calib_shim = _CalibVisShim(width, height)

    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)
    sampled_frames: list[int] = []

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1: Calibrating (orig)")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        sampled_frames.append(frame_idx)

        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        lines = line_detector.detect(frame) if line_detector is not None else []

        keypoints_for_solver = (
            _filter_keypoint_outliers(keypoints) if of_enabled else keypoints
        )
        kp_dict = id_map.detector_output_to_upstream_dict(keypoints_for_solver)
        lines_dict = id_map.detector_lines_to_upstream_dict(lines)

        # Detection counts. Recorded on every frame (including solver-fail
        # and sanity-reject branches) so diagnostics can tell apart
        # "HRNet found nothing" from "solver couldn't make use of it".
        detection_info = {
            "num_keypoints_detected": len(keypoints),
            "num_keypoints_solver": len(kp_dict),
            "num_non_ground": len(set(kp_dict) & NON_GROUND_UPSTREAM_IDS),
            "num_lines_detected": len(lines),
            "num_lines_solver": len(lines_dict),
        }

        calibrator.update(kp_dict, lines_dict)
        result = calibrator.heuristic_voting(
            refine=True, refine_lines=use_lines, th=voting_th,
        )

        if result is not None:
            cp = result["cam_params"]
            R_dn = np.asarray(cp["rotation_matrix"], dtype=np.float64).reshape(3, 3)
            pos_dn = np.asarray(cp["position_meters"], dtype=np.float64).reshape(3)
            t_dn = -R_dn @ pos_dn
            # Convert upstream y-down → project y-up: R_up = R_dn @ diag(1,-1,1)
            # which simply negates column 1. Translation is invariant.
            R_up = R_dn.copy()
            R_up[:, 1] = -R_up[:, 1]
            fx = float(cp["x_focal_length"])
            fy = float(cp["y_focal_length"])
            px0, py0 = float(cp["principal_point"][0]), float(cp["principal_point"][1])
            K = np.array([[fx, 0, px0], [0, fy, py0], [0, 0, 1]], dtype=np.float64)
            H = K @ np.column_stack([R_up[:, 0], R_up[:, 1], t_dn])
            if abs(H[2, 2]) > 1e-10:
                H = H / H[2, 2]

            sane, reject_reason = (
                _is_pose_intrinsic_sane(fx) if sanity_enabled else (True, None)
            )
            if not sane:
                calibration_results["frames"][frame_idx] = {
                    "calibrated": False,
                    "rejected": "geometry_sanity",
                    "reason": reject_reason,
                    "reprojection_error": float(result["rep_err"]),
                    **detection_info,
                }
            else:
                homographies[frame_idx] = H
                calibration_results["frames"][frame_idx] = {
                    "calibrated": True,
                    "num_keypoints": len(calibrator.keypoints_dict),
                    "num_lines": len(calibrator.lines_dict),
                    "num_intersections": 0,
                    "total_points": int(sum(
                        len(s) for s in calibrator.subsets.values()
                    )),
                    "inliers": int(len(calibrator.subsets["full"])),
                    "reprojection_error": float(result["rep_err"]),
                    "mode": result.get("mode"),
                    "ransac": result.get("use_ransac"),
                    **detection_info,
                }
                calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {
                "calibrated": False,
                **detection_info,
            }

        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(
                str(vis_kp_dir / fname),
                draw_vis_keypoints(
                    frame, keypoints,
                    conf_threshold=pnl_config.get("keypoint_threshold", 0.3434),
                ),
            )
            if vis_line_dir is not None:
                cv2.imwrite(
                    str(vis_line_dir / fname),
                    draw_vis_lines(
                        frame, lines,
                        conf_threshold=pnl_config.get("line_threshold", 0.15),
                    ),
                )
            # Build an in-tree-shaped result for draw_vis_calibration.
            # No img_pts/inlier_mask (orig solves via cv2.calibrateCamera over
            # 3-plane reparameterization, not RANSAC PnP), so all detected
            # keypoints render as filled green ("inlier set unknown" branch).
            # Frames rejected by the geometry sanity gate are absent from
            # ``homographies`` even though ``result is not None`` — render
            # them as failures so the overlay still shows the keypoints.
            if result is not None and frame_idx in homographies:
                vis_result = {
                    "homography": homographies[frame_idx],
                    "final_error": float(result["rep_err"]),
                    "inliers": int(len(calibrator.subsets["full"])),
                    "total_points": int(sum(
                        len(s) for s in calibrator.subsets.values()
                    )),
                }
            else:
                vis_result = None
            cv2.imwrite(
                str(vis_calib_dir / fname),
                draw_vis_calibration(
                    frame, keypoints, lines, calib_shim, vis_result,
                    pitch_template, keypoint_mapper_vis,
                ),
            )

    cap.release()

    # Optional: SIFT-based homography chaining to fill frames PnLCalib
    # couldn't solve. Mutates ``homographies`` and ``calibration_results``
    # in place so downstream stages see a denser map without any schema
    # change.
    gap_cfg = fr_config.get("gap_filling", {})
    if gap_cfg.get("enabled"):
        from .gap_filling import fill_gaps_with_chain
        fx_for_gap = float(config.get("intrinsics", {}).get("fx", width))
        gap_stats = fill_gaps_with_chain(
            video_path=video_path,
            sampled_frames=sampled_frames,
            image_width=width,
            image_height=height,
            homographies=homographies,
            calibration_results=calibration_results,
            fx=fx_for_gap,
            config=gap_cfg,
        )
        logger.info(
            "Gap-filling: filled %d/%d uncalibrated sampled frames "
            "(skipped %d, failed %d, runs=%d)",
            gap_stats["filled"],
            len(sampled_frames) - calibrated_count,
            gap_stats["skipped_too_long"],
            gap_stats["failed"],
            gap_stats["runs"],
        )

    save_calibration_outputs(output_dir, calibration_results, homographies)
    stats = compute_calibration_stats(
        calibration_results, video, sampler, calibrated_count,
    )
    print_calibration_summary(stats, label="Stage 1 (pnlcalib_orig)")
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
