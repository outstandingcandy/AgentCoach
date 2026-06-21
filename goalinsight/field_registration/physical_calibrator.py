"""Physical camera calibrator with 7-DOF optimization.

The input images are assumed to be already undistorted, so distortion coefficients
are fixed at zero and the principal point is fixed at the image center.
Only 7 parameters are optimized: rvec(3), tvec(3), and focal length f.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..utils.projection import project_points_2d
from .pitch_template import LINE_KEYPOINTS, build_field_template
from .pnlcalib.curve_utils import (
    compute_cumulated_lengths,
    interpolate_on_polyline,
    sample_points_on_image_line,
)
from .pnlcalib.line_mapping import LineMapper

logger = logging.getLogger(__name__)

# Ground-only line IDs (exclude crossbars and goal posts)
GROUND_LINE_IDS = set(range(23)) - {6, 7, 8, 9, 10, 11}

# Symmetric ±roll cap (degrees) for the locked-C look-at LM. Wide enough
# for handheld jitter; tight enough that "camera flipped sideways"
# degenerate solutions can't fit. 15° matches the existing default
# pitch_bounds_deg upper end (``configs/sunday_soccer.yaml``: [2, 15]).
_LOOKAT_ROLL_BOUND_DEG = 15.0


def _lookat_to_R(yaw: float, el: float, roll: float) -> np.ndarray:
    """Build OpenCV ``R`` (world→camera) from (yaw, el, roll).

    Convention:
      - yaw: rotation about world +Z (yaw=0 → optical axis along +Y_world)
      - el:  elevation BELOW horizon in radians (positive = looking down,
             matching the existing ``pitch_bounds_deg`` semantics)
      - roll: rotation of the camera "up" vector about the optical axis,
              positive = counter-clockwise as seen from behind the
              camera looking forward.

    OpenCV camera frame: +z forward, +y down, +x right. The returned
    rows are [x_cam_w; y_cam_w; z_cam_w] — exactly what
    ``cv2.Rodrigues`` would produce from the equivalent rvec.
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    ce, se = np.cos(el), np.sin(el)
    cr, sr = np.cos(roll), np.sin(roll)
    # Forward / no-roll up basis derived from yaw + elevation.
    fwd = np.array([ce * sy, ce * cy, -se])
    up0 = np.array([se * sy, se * cy, ce])
    # Apply roll about fwd: rotate up0 toward (fwd × up0).
    up = up0 * cr + np.cross(fwd, up0) * sr
    z_cam = fwd
    y_cam = -up                          # camera y points DOWN
    x_cam = np.cross(y_cam, z_cam)       # right-handed: x = y × z
    return np.stack([x_cam, y_cam, z_cam], axis=0)


def _R_to_lookat(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`_lookat_to_R`. Returns (yaw, el, roll) radians.

    Uses ``arctan2`` so the extraction is stable across the full sphere.
    Round-trips ``_lookat_to_R`` to machine precision (verified against
    a sample of real-world poses on the Sunday cup clip).
    """
    fwd = R[2]
    y_cam_w = R[1]
    el = float(np.arcsin(-np.clip(fwd[2], -1.0, 1.0)))
    yaw = float(np.arctan2(fwd[0], fwd[1]))
    cy, sy = np.cos(yaw), np.sin(yaw)
    ce, se = np.cos(el), np.sin(el)
    fwd1 = np.array([ce * sy, ce * cy, -se])
    up1 = np.array([se * sy, se * cy, ce])
    minus_y = -y_cam_w
    a = float(np.dot(minus_y, up1))
    b = float(np.dot(minus_y, np.cross(fwd1, up1)))
    roll = float(np.arctan2(b, a))
    return yaw, el, roll

# Re-exported for backward compat with code that imports
# LINE_INTERSECTIONS from this module. Built lazily from FIFA defaults
# since the legacy constant was the FIFA-spec table.
LINE_INTERSECTIONS = build_field_template()[2]


class PhysicalCalibrator:
    """Camera calibrator with 7-DOF optimization: rvec(3), tvec(3), f.

    Distortion is fixed at zero and principal point at image center.
    Only focal length is optimized among intrinsics.
    """

    def __init__(
        self,
        K: np.ndarray,
        image_size: tuple[int, int],
        ransac_reproj_error: float = 15.0,
        line_weight: float = 1.0,
        line_sample_points: int = 20,
        min_line_length_px: float = 30.0,
        focal_bounds: tuple[float, float] = (1200.0, 2200.0),
        world_residual_weight: float = 0.0,
        world_error_threshold: float = 5.0,
        camera_position: tuple[float, float, float] | None = None,
        position_weight: float = 50.0,
        lock_camera_position: bool = False,
        position_bounds_m: tuple[float, float, float] | None = None,
        pitch_bounds_deg: tuple[float, float] | None = None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        pitch_dims: dict | None = None,
    ):
        """Initialize with image size and focal length guess.

        Args:
            K: 3x3 intrinsic matrix (only f=K[0,0] is used as initial guess;
               cx/cy are overridden to image center).
            image_size: (width, height) of video frames.
            ransac_reproj_error: PnP RANSAC reprojection threshold in pixels.
            line_weight: Weight alpha for line residuals vs point residuals.
            line_sample_points: Number of points sampled along each line.
            focal_bounds: (min_f, max_f) absolute bounds for focal length.
            world_residual_weight: Weight for world-space back-projection residuals.
                0 = disabled.
            world_error_threshold: World-space error threshold (meters) for iterative
                outlier rejection. Set to float('inf') to disable.
            min_line_length_px: Reject detected line segments shorter than this
                in pixel space — short detections carry near-zero direction
                information and pollute the LM. PnLCalib upstream uses 50.
            camera_position: Known camera position (x, y, z) in meters. If set, adds
                residuals penalizing deviation from this position. None = no constraint.
                Also enables the few-point (≥3) P3P init branch.
            position_weight: Weight for camera position residuals (pixels-equivalent).
                Only used when lock_camera_position=False (soft constraint).
            lock_camera_position: If True (and camera_position is set), tvec is
                removed from optimization and recomputed every step as -R·C_target.
                Reduces 7-DOF to 4-DOF (rvec, f). Use when position is known
                to ≤cm accuracy. Otherwise keep False for soft constraint.
            pitch_bounds_deg: Optional ``(min_deg, max_deg)`` hard bounds on
                the camera tilt angle below the horizon. Skipped when None.
                Use to keep LM from drifting to near-horizontal poses on
                sparse KP sets (sideline rigs are typically 10–35° below
                horizon).
            pitch_length: Pitch length in meters (default 105 = FIFA standard).
                Only used when ``pitch_dims`` is None.
            pitch_width: Pitch width in meters (default 68 = FIFA standard).
                Only used when ``pitch_dims`` is None.
            pitch_dims: Full pitch geometry override (penalty area, goal area,
                goal frame, center circle, penalty mark distance). Keys not
                present fall back to FIFA defaults. When given, overrides
                ``pitch_length``/``pitch_width`` if those are in the dict.
                Required for non-FIFA pitches (e.g. youth fields).
        """
        self.width, self.height = image_size
        # Fix principal point at image geometric center, zero distortion
        self.K = np.array([
            [K[0, 0], 0, self.width / 2.0],
            [0, K[0, 0], self.height / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.ransac_reproj_error = ransac_reproj_error
        self.line_weight = line_weight
        self.line_sample_points = line_sample_points
        self.min_line_length_px = min_line_length_px
        self.focal_bounds = focal_bounds
        self.world_residual_weight = world_residual_weight
        self.world_error_threshold = world_error_threshold
        self.camera_position = camera_position
        self.position_weight = position_weight
        self.lock_camera_position = lock_camera_position and camera_position is not None
        # Per-axis hard bounds (in metres) around the prior camera position.
        # When set, LM's tvec is constrained so the resulting camera centre
        # stays within ``camera_position ± position_bounds_m``. Use for axes
        # where the prior is well-known (e.g. fixed sideline rig where x is
        # tightly known to ~3m). Skipped when no prior is set or the axis
        # bound is None / non-positive.
        self.position_bounds_m = position_bounds_m
        # Hard bounds (degrees) on the camera tilt angle (pitch_deg = angle
        # below horizon, computed as ``arcsin(-(R^T @ [0,0,1]).z)``). When
        # set, LM gets a barrier residual that grows linearly outside the
        # range. A typical sideline rig sits at 10–35° tilt; values outside
        # that band almost always come from LM running away on a sparse,
        # spatially-clustered keypoint set (e.g. the frame-779 case where
        # 8 KPs all clustered in the right-half far corner let LM drift
        # tilt down to 6° while compensating with a smaller fx).
        # ``None`` skips the constraint entirely.
        self.pitch_bounds_deg = pitch_bounds_deg

        # Resolve pitch dims:
        # - pitch_dims given → full override (FIFA fills missing keys).
        # - else → length/width with FIFA markings.
        if pitch_dims:
            resolved = dict(pitch_dims)
            resolved.setdefault("pitch_length", pitch_length)
            resolved.setdefault("pitch_width", pitch_width)
        else:
            resolved = {"pitch_length": pitch_length, "pitch_width": pitch_width}
        self.pitch_dims = resolved
        self.pitch_length = resolved["pitch_length"]
        self.pitch_width = resolved["pitch_width"]

        # Build field template (world coords, line defs, intersections)
        self._field_world_coords, self._field_line_defs, self._field_intersections = \
            build_field_template(resolved)

        self._keypoints: list[dict] = []
        self._lines: list[dict] = []
        self._last_debug: dict | None = None
        self._last_ransac_debug: dict | None = None

    def update(self, keypoints: list[dict], lines: list[dict]):
        """Update with detected keypoints and lines for current frame."""
        self._keypoints = keypoints
        self._lines = lines

    def calibrate_correspondences(
        self,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
        kp_ids: list | None = None,
        line_constraints: list | None = None,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Calibrate from caller-supplied (pixel, world) correspondences.

        The annotator already knows the (pixel, world) match for each
        keypoint by name and doesn't need to go through the KP detector
        / KeypointMapper path. This method skips Step 1 / 2 of
        :meth:`calibrate` (correspondence preparation, line-from-keypoint
        derivation) and runs the same Step 3-6 cold-start + 7-DOF LM +
        sanity-check pipeline as the pipeline backend.

        Args:
            img_pts: (N, 2) pixel coordinates.
            world_pts: (N, 3) world coordinates (z=0 ground points,
                z=-2.44 crossbar points). Must have len == len(img_pts).
            kp_ids: Optional per-correspondence labels for diagnostics.
                Defaults to range(N).
            line_constraints: Optional pre-built line constraint dicts
                (same shape :meth:`_derive_lines_from_keypoints` outputs).
                Pass [] to skip the line residual term entirely.
            initial_rvec / initial_tvec: Warm-start pose; skips the
                cold-start RANSAC / P3P stage when both are given.

        Returns:
            Same result dict shape as :meth:`calibrate`, or None on
            failure (see ``self._last_debug`` for the reason).
        """
        img_pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
        world_pts = np.asarray(world_pts, dtype=np.float64).reshape(-1, 3)
        if kp_ids is None:
            kp_ids = list(range(len(img_pts)))
        if line_constraints is None:
            line_constraints = []
        return self._calibrate_core(
            img_pts, world_pts, list(kp_ids), line_constraints,
            initial_rvec=initial_rvec, initial_tvec=initial_tvec,
            keypoint_mapper=None, line_mapper=None,
        )

    def calibrate(
        self,
        keypoint_mapper,
        line_mapper=None,
        min_confidence: float = 0.3,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Run 6-DOF calibration on current frame data.

        Args:
            keypoint_mapper: KeypointMapper for 3D correspondences.
            line_mapper: LineMapper for line world coords (or None to derive from keypoints).
            min_confidence: Minimum keypoint confidence threshold.
            initial_rvec: Warm-start rotation vector (from previous frame). If None, uses PnP RANSAC.
            initial_tvec: Warm-start translation vector. If None, uses PnP RANSAC.

        Returns:
            Result dict with camera params, or None if calibration failed.
        """
        # Reset debug info for this frame
        self._last_debug = None

        # Step 1: Prepare keypoint correspondences
        img_pts, world_pts, kp_ids, confidences = self._prepare_correspondences(
            keypoint_mapper, min_confidence
        )
        # With known camera_position we can solve from 3 points (P3P);
        # without it the geometry needs 6 to disambiguate.
        min_pts = 3 if self.camera_position is not None else 6
        if len(world_pts) < min_pts:
            logger.debug("Too few keypoints (%d < %d), skipping", len(world_pts), min_pts)
            self._last_debug = {
                "failure_reason": f"too_few_keypoints ({len(world_pts)} < {min_pts})",
                "img_pts": img_pts,
                "world_pts": world_pts,
                "kp_ids": kp_ids,
                "ransac_info": None,
            }
            return None

        # Step 2: Prepare line constraints
        line_constraints = self._derive_lines_from_keypoints(
            kp_ids, img_pts, keypoint_mapper, line_mapper
        )

        # Step 2b: Add line-line intersection points BEFORE PnP. With sparse
        # detector output (e.g. 3 collinear kp + 5 lines) the intersections
        # often supply the only non-collinear correspondences — without them
        # P3P has no chance.
        isect_img, isect_world, isect_ids = self._compute_line_intersections(
            line_constraints, kp_ids
        )
        n_intersections = len(isect_ids)
        if n_intersections > 0:
            img_pts = np.vstack([img_pts, isect_img])
            world_pts = np.vstack([world_pts, isect_world])
            kp_ids = list(kp_ids) + isect_ids
            logger.debug("Added %d line intersection points pre-PnP (total=%d)",
                         n_intersections, len(kp_ids))

        return self._calibrate_core(
            img_pts, world_pts, kp_ids, line_constraints,
            initial_rvec=initial_rvec, initial_tvec=initial_tvec,
            keypoint_mapper=keypoint_mapper, line_mapper=line_mapper,
        )

    def _calibrate_core(
        self,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
        kp_ids: list,
        line_constraints: list,
        *,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
        keypoint_mapper=None,
        line_mapper=None,
    ) -> dict[str, Any] | None:
        """Step 3-6 of calibrate() — cold-start + 7-DOF LM + sanity check.

        Shared by :meth:`calibrate` (pipeline path) and
        :meth:`calibrate_correspondences` (annotator path). Anything
        before Step 3 — building correspondences from the KP detector
        and stitching line constraints — is the caller's job.

        ``keypoint_mapper`` / ``line_mapper`` are only consulted when
        the world-error outlier rejection loop wants to re-derive line
        constraints after dropping a keypoint. The annotator passes
        None and the loop simply preserves the supplied constraints.
        """
        min_pts = 3 if self.camera_position is not None else 6

        # Step 3: Get initial pose estimate
        ransac_info = None
        if initial_rvec is not None and initial_tvec is not None:
            # Warm-start from previous frame — skip PnP RANSAC
            rvec_init = initial_rvec.copy().ravel()
            tvec_init = initial_tvec.copy().ravel()
            logger.debug("Using warm-start from previous frame")
        else:
            # Cold start. Few-point P3P branch when (a) total points
            # are scarce and we have a position prior, OR (b) the
            # camera position is hard-locked — in that case the prior
            # is fully trusted, so the dense focal sweep + position-
            # based candidate scoring inside ``_pnp_few_points_init``
            # is strictly stronger than the multi-focal RANSAC, even
            # when there are ≥6 points. Otherwise fall back to the
            # standard multi-focal RANSAC (which doesn't use the
            # position prior).
            use_few_pts = (
                self.camera_position is not None and (
                    len(world_pts) < 6 or self.lock_camera_position
                )
            )
            if use_few_pts:
                pnp_result = self._pnp_few_points_init(world_pts, img_pts)
            else:
                pnp_result = self._pnp_ransac_init(world_pts, img_pts)
            if pnp_result is None:
                self._last_debug = {
                    "failure_reason": "pnp_ransac_failed",
                    "img_pts": img_pts,
                    "world_pts": world_pts,
                    "kp_ids": kp_ids,
                    "ransac_info": self._last_ransac_debug,
                }
                return None
            rvec_init, tvec_init, ransac_info = pnp_result

        # Step 4: 7-DOF joint point+line optimization (rvec, tvec, f)
        # with iterative world-error-based outlier rejection
        MAX_WORLD_OUTLIER_ITERS = 3

        cur_img_pts = img_pts
        cur_world_pts = world_pts
        cur_kp_ids = kp_ids
        cur_lc = line_constraints
        cur_rvec = rvec_init
        cur_tvec = tvec_init

        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_fixed = self.dist_coeffs.copy()

        if np.isfinite(self.world_error_threshold):
            for world_iter in range(MAX_WORLD_OUTLIER_ITERS):
                rvec_opt, tvec_opt, f_opt = self._refine_7dof(
                    cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
                )

                # Build optimized intrinsics for this iteration
                K_iter = np.array([
                    [f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]
                ], dtype=np.float64)

                # Compute per-point world error
                _, per_pt_werr = self._compute_world_error(
                    rvec_opt, tvec_opt, cur_img_pts, cur_world_pts,
                    K=K_iter, dist=dist_fixed,
                )

                # Distance-adaptive threshold: use relative error (world_error / distance).
                # A point is an outlier if its relative error exceeds threshold/30.
                # E.g., threshold=5m → max 16.7% relative error; at 30m that's 5m, at 90m that's 15m.
                R_iter, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
                cam_center = -R_iter.T @ tvec_opt.ravel()
                dists = np.linalg.norm(cur_world_pts - cam_center[np.newaxis, :], axis=1)
                dists = np.maximum(dists, 1.0)  # avoid division by zero
                relative_err = per_pt_werr / dists
                relative_thresh = self.world_error_threshold / 30.0  # ~16.7% for threshold=5m

                # Find world-error outliers
                keep_mask = relative_err < relative_thresh
                n_removed = int(np.sum(~keep_mask))
                if n_removed == 0 or world_iter == MAX_WORLD_OUTLIER_ITERS - 1:
                    break

                # Need at least 6 points to continue
                if int(np.sum(keep_mask)) < min_pts:
                    break

                logger.debug("World-error iter %d: removed %d points (threshold=%.1fm)",
                             world_iter, n_removed, self.world_error_threshold)

                cur_img_pts = cur_img_pts[keep_mask]
                cur_world_pts = cur_world_pts[keep_mask]
                cur_kp_ids = [cur_kp_ids[i] for i in range(len(cur_kp_ids)) if keep_mask[i]]
                cur_rvec = rvec_opt
                cur_tvec = tvec_opt

                # Re-derive line constraints from remaining keypoints.
                # Skipped when the caller (e.g. annotator) didn't supply
                # the mappers — the original line_constraints stay.
                if keypoint_mapper is not None:
                    cur_lc = self._derive_lines_from_keypoints(
                        cur_kp_ids, cur_img_pts, keypoint_mapper, line_mapper
                    )
        else:
            # No outlier rejection — use all keypoints, single optimization pass
            rvec_opt, tvec_opt, f_opt = self._refine_7dof(
                cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
            )

        # Use the filtered points for final result
        img_pts = cur_img_pts
        world_pts = cur_world_pts
        kp_ids = cur_kp_ids
        line_constraints = cur_lc

        # Build optimized intrinsics (cx/cy fixed at center, dist=0)
        K_opt = np.array([
            [f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]
        ], dtype=np.float64)
        dist_opt = dist_fixed

        # Step 5: Sanity check — reject catastrophically bad results
        projected_check = project_points_2d(
            world_pts, rvec_opt, tvec_opt, K_opt, dist_opt,
        )
        median_err_check = float(np.median(np.linalg.norm(
            projected_check - img_pts, axis=1
        )))
        if median_err_check > 200:
            logger.warning("Optimization result rejected (median_err=%.0fpx > 200px)", median_err_check)
            self._last_debug = {
                "failure_reason": f"optimization_diverged (median_err={median_err_check:.0f}px)",
                "img_pts": img_pts,
                "world_pts": world_pts,
                "kp_ids": kp_ids,
                "ransac_info": ransac_info,
            }
            return None

        # Step 6: Compute stats and format result
        mean_error, inlier_mask, inlier_count = self._compute_reprojection_stats(
            rvec_opt, tvec_opt, img_pts, world_pts, K=K_opt, dist=dist_opt
        )
        world_error, per_point_world_errors = self._compute_world_error(
            rvec_opt, tvec_opt, img_pts, world_pts, K=K_opt, dist=dist_opt
        )
        world_error_all, per_kp_world_errors = self._compute_world_error_all(
            rvec_opt, tvec_opt, K=K_opt, dist=dist_opt
        )

        # Build result
        R, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
        tvec_col = tvec_opt.reshape(3, 1)

        # Derive ground-plane homography for backward compatibility
        H = K_opt @ np.column_stack([R[:, 0], R[:, 1], tvec_col.ravel()])
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]

        return {
            "homography": H,
            "final_error": float(mean_error),
            "world_error": float(world_error),
            "world_error_all": float(world_error_all),
            "per_point_world_errors": per_point_world_errors,
            "per_kp_world_errors": per_kp_world_errors,
            # Caller-supplied paths (e.g. annotator) bypass the KP /
            # line / intersection prep stages, so these counts only
            # reflect what was passed into _calibrate_core.
            "num_keypoints": len(self._keypoints),
            "num_lines": len(self._lines),
            "num_intersections": len([
                k for k in kp_ids if isinstance(k, str) and k.startswith("isect_")
            ]),
            "total_points": len(img_pts),
            "inliers": int(inlier_count),
            "intrinsics_init": {
                "f": float(self.K[0, 0]),
                "cx": float(self.K[0, 2]),
                "cy": float(self.K[1, 2]),
            },
            "camera_params": {
                "K": K_opt.copy(),
                "rvec": rvec_opt.copy(),
                "tvec": tvec_opt.copy(),
                "R": R,
                "dist_coeffs": dist_opt.copy(),
                "focal_length": float(f_opt),
            },
            "camera_pose": {
                "K": K_opt.tolist(),
                "dist_coeffs": dist_opt.tolist(),
                "rvec": rvec_opt.tolist(),
                "tvec": tvec_opt.tolist(),
                "reprojection_error": float(mean_error),
                "world_error": float(world_error),
                "world_error_all": float(world_error_all),
                "inliers_count": int(inlier_count),
            },
            "img_pts": img_pts,
            "world_pts": world_pts,
            "kp_ids": kp_ids,
            "inlier_mask": inlier_mask,
            "line_constraints_count": len(line_constraints),
            "line_constraints": line_constraints,
            "ransac_info": ransac_info,
        }

    def _prepare_correspondences(self, keypoint_mapper, min_confidence):
        """Extract 3D-2D keypoint correspondences including crossbar points."""
        img_pts, world_pts, kp_ids = keypoint_mapper.build_3d_correspondence_matrix(
            self._keypoints,
            filter_by_confidence=min_confidence,
            exclude_non_ground=False,  # Include crossbar points (z=-2.44)
        )

        # Override world coordinates with field template (supports non-standard pitch sizes)
        if len(world_pts) > 0:
            world_pts = world_pts.copy()
            for idx, kid in enumerate(kp_ids):
                if kid < len(self._field_world_coords):
                    wx, wy = self._field_world_coords[kid]
                    z = world_pts[idx, 2]  # preserve z (0 or -2.44 for crossbar)
                    world_pts[idx] = [wx, wy, z]

        # Build confidence array matching the correspondence order
        conf_map = {kp["id"]: kp.get("confidence", 1.0) for kp in self._keypoints}
        confidences = np.array([conf_map.get(kid, 1.0) for kid in kp_ids], dtype=np.float64)

        return (
            img_pts.astype(np.float64) if len(img_pts) > 0 else np.zeros((0, 2), dtype=np.float64),
            world_pts.astype(np.float64) if len(world_pts) > 0 else np.zeros((0, 3), dtype=np.float64),
            kp_ids,
            confidences,
        )

    def _derive_lines_from_keypoints(self, kp_ids, kp_img, keypoint_mapper, line_mapper):
        """Build line constraints, preferring line detector output over kp-derived ones.

        For each ground line, image endpoints come from (in priority order):
        1. The line detector's own (x1, y1, x2, y2) when that line ID is in
           ``self._lines``. Without this path, line head output is wasted
           whenever <2 of a line's keypoints are detected.
        2. Fallback: connect the two most extreme detected keypoints on
           that line — works when a single keypoint isn't enough.

        Either way we sample N evenly-spaced points along both the 3D world
        line and the 2D image segment. Segments shorter than
        ``self.min_line_length_px`` in image space are dropped — short
        detections carry near-zero direction information.
        """
        # Index detector lines by id so we can look them up cheaply.
        detected_lines_by_id = {}
        for ln in self._lines:
            lid = ln.get("id")
            if lid is None:
                continue
            detected_lines_by_id[lid] = ln

        detected = {}
        for i, kid in enumerate(kp_ids):
            detected[kid] = kp_img[i]

        constraints = []
        non_ground = keypoint_mapper.NON_GROUND_KEYPOINTS

        for line_id in LINE_KEYPOINTS.keys():
            # Skip non-ground lines (crossbars / posts)
            if line_id not in self._field_line_defs:
                continue

            # Source 1: line detector
            img_p1 = img_p2 = None
            if line_id in detected_lines_by_id:
                ln = detected_lines_by_id[line_id]
                img_p1 = (float(ln["x1"]), float(ln["y1"]))
                img_p2 = (float(ln["x2"]), float(ln["y2"]))
            else:
                # Source 2: kp fallback (need ≥2 kp on this line)
                kp_list = LINE_KEYPOINTS.get(line_id, [])
                found = [(kid, detected[kid]) for kid in kp_list
                         if kid in detected and kid not in non_ground]
                if len(found) >= 2:
                    img_p1 = (float(found[0][1][0]), float(found[0][1][1]))
                    img_p2 = (float(found[-1][1][0]), float(found[-1][1][1]))

            if img_p1 is None:
                continue

            # Reject short image segments. A line carries useful direction
            # only when its endpoints are far apart in pixel space — at
            # 8-15 px length, ~1 px endpoint noise rotates the inferred
            # direction by 4-8°, polluting LM with high-variance constraints.
            seg_len = ((img_p1[0] - img_p2[0]) ** 2
                       + (img_p1[1] - img_p2[1]) ** 2) ** 0.5
            if seg_len < self.min_line_length_px:
                continue

            # Get world line endpoints
            ld = self._field_line_defs[line_id]
            p1_w = np.array(ld["p1"], dtype=np.float64)
            p2_w = np.array(ld["p2"], dtype=np.float64)
            polyline_3d = np.array([p1_w, p2_w])
            cum_lengths = compute_cumulated_lengths(polyline_3d)

            total_len = cum_lengths[-1]
            if total_len < 1e-6:
                continue
            t_values = np.linspace(0, total_len, self.line_sample_points)
            world_samples = np.array([
                interpolate_on_polyline(polyline_3d, cum_lengths, t)
                for t in t_values
            ])

            img_samples = sample_points_on_image_line(
                img_p1[0], img_p1[1], img_p2[0], img_p2[1],
                n_points=self.line_sample_points,
            )

            constraints.append({
                "world_samples": world_samples,
                "img_samples": img_samples,
                "line_id": line_id,
            })

        return constraints

    def _compute_line_intersections(self, line_constraints, existing_kp_ids):
        """Compute intersection points of derived image lines.

        For each pair of lines known to intersect on the field, computes the
        2D image intersection and returns it as an additional correspondence.
        Skips intersections whose keypoint is already detected (avoids duplicates).

        Args:
            line_constraints: list of line constraint dicts from _derive_lines_from_keypoints.
            existing_kp_ids: list of already-detected keypoint IDs.

        Returns:
            (img_pts, world_pts, kp_ids) — arrays of intersection correspondences.
        """
        # Build lookup: line_id → (p1_img, p2_img) from line constraints
        line_endpoints = {}
        for lc in line_constraints:
            lid = lc["line_id"]
            img = lc["img_samples"]
            line_endpoints[lid] = (img[0], img[-1])

        existing_set = set(existing_kp_ids)
        # When the camera's FOV cuts a line endpoint off, the line's
        # extrapolated intersection lands well outside the frame yet is
        # still a valid 3D-2D correspondence (the world point is unique;
        # only the pixel location is extrapolated). Allow ±1.5× image
        # extents — far enough to cover useful out-of-frame intersections,
        # tight enough to reject parallel-line numerical blowups.
        margin_x = self.width * 1.5
        margin_y = self.height * 1.5

        int_img = []
        int_world = []
        int_ids = []

        for (lid_a, lid_b), (wx, wy, wz, kp_id) in self._field_intersections.items():
            if lid_a not in line_endpoints or lid_b not in line_endpoints:
                continue

            # Skip if the corresponding keypoint is already detected
            if kp_id in existing_set:
                continue

            # Image line a: p1a → p2a
            p1a, p2a = line_endpoints[lid_a]
            # Image line b: p1b → p2b
            p1b, p2b = line_endpoints[lid_b]

            # Parametric line-line intersection
            x1, y1 = float(p1a[0]), float(p1a[1])
            x2, y2 = float(p2a[0]), float(p2a[1])
            x3, y3 = float(p1b[0]), float(p1b[1])
            x4, y4 = float(p2b[0]), float(p2b[1])

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue  # parallel lines

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)

            # Skip if extrapolation lands absurdly far from the frame —
            # signals near-parallel detector lines whose computed
            # intersection has huge numerical error.
            if ix < -margin_x or ix > self.width + margin_x:
                continue
            if iy < -margin_y or iy > self.height + margin_y:
                continue

            int_img.append([ix, iy])
            int_world.append([wx, wy, wz])
            int_ids.append(f"isect_{lid_a}_{lid_b}")

        if not int_img:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                [],
            )

        return (
            np.array(int_img, dtype=np.float64),
            np.array(int_world, dtype=np.float64),
            int_ids,
        )

    def _pnp_ransac_init(self, world_pts, img_pts):
        """Initialize camera pose via multi-focal multi-solver PnP RANSAC.

        Tries multiple focal lengths × multiple solvers and picks the result
        with the most inliers. This avoids bad initialization when the actual
        focal length differs significantly from the profile value.

        Returns:
            (rvec, tvec, ransac_info) or None on failure.
            ransac_info contains inlier indices, per-point errors, and init pose.
        """
        # Sample focal lengths spanning the allowed range. Densify the
        # candidate set so a wide focal_bounds (e.g. [4000, 12000] for
        # a zooming clip) gets covered with steps small enough that
        # SQPNP/EPNP can produce ≥4-inlier solutions at one of them —
        # the original 5-point sweep left 2-3k gaps and frames whose
        # true focal landed in a gap saw all-trial-failed RANSAC.
        f_min, f_max = self.focal_bounds
        f_profile = float(self.K[0, 0])
        n_steps = max(7, int((f_max - f_min) / 1000) + 1)
        focal_candidates = sorted(set(
            list(np.linspace(f_min, f_max, n_steps).tolist())
            + [f_profile]
        ))
        solvers = [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]
        cx, cy = self.K[0, 2], self.K[1, 2]
        dist = self.dist_coeffs

        best_result = None
        best_inlier_count = -1
        best_f = f_profile

        for f_try in focal_candidates:
            K_try = np.array([[f_try, 0, cx], [0, f_try, cy], [0, 0, 1]], dtype=np.float64)
            for solver in solvers:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=world_pts.reshape(-1, 1, 3),
                    imagePoints=img_pts.reshape(-1, 1, 2),
                    cameraMatrix=K_try,
                    distCoeffs=dist,
                    reprojectionError=self.ransac_reproj_error,
                    iterationsCount=2000,
                    flags=solver,
                )
                if success and inliers is not None and len(inliers) >= 4:
                    if len(inliers) > best_inlier_count:
                        best_inlier_count = len(inliers)
                        best_result = (rvec, tvec, inliers)
                        best_f = f_try

        if best_result is None:
            logger.debug("PnP RANSAC failed (all trials)")
            self._last_ransac_debug = {
                "success": False,
                "failure_reason": "all_trials_failed",
                "reprojection_threshold": float(self.ransac_reproj_error),
                "total_points": int(len(img_pts)),
                "inlier_count": 0,
                "inlier_indices": [],
            }
            return None

        rvec, tvec, inliers = best_result

        # Update K with the best focal length for reprojection error computation
        K_best = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
        # Also update self.K so _refine_7dof starts from this f
        self.K[0, 0] = best_f
        self.K[1, 1] = best_f

        # Compute per-point reprojection errors for RANSAC result
        projected = project_points_2d(world_pts, rvec, tvec, K_best, dist)
        per_point_errors = np.linalg.norm(projected - img_pts, axis=1)
        logger.debug("PnP RANSAC best: f=%.0f, %d/%d inliers",
                      best_f, best_inlier_count, len(img_pts))

        inlier_indices = inliers.ravel().tolist()
        inlier_mask = np.zeros(len(img_pts), dtype=bool)
        inlier_mask[inlier_indices] = True

        ransac_info = {
            "success": True,
            "reprojection_threshold": float(self.ransac_reproj_error),
            "total_points": int(len(img_pts)),
            "inlier_count": int(len(inlier_indices)),
            "outlier_count": int(len(img_pts) - len(inlier_indices)),
            "inlier_indices": inlier_indices,
            "inlier_mask": inlier_mask,
            "per_point_errors": per_point_errors,
            "mean_inlier_error": float(np.mean(per_point_errors[inlier_mask])),
            "mean_outlier_error": float(np.mean(per_point_errors[~inlier_mask])) if np.any(~inlier_mask) else 0.0,
            "mean_all_error": float(np.mean(per_point_errors)),
            "rvec_init": rvec.ravel().copy(),
            "tvec_init": tvec.ravel().copy(),
        }

        logger.debug("PnP RANSAC: %d/%d inliers, mean_err=%.2fpx",
                      len(inlier_indices), len(img_pts), ransac_info["mean_all_error"])
        return rvec.ravel(), tvec.ravel(), ransac_info

    def _pnp_few_points_init(self, world_pts, img_pts):
        """Initialize pose from 3-5 points using camera_position to disambiguate.

        With ≥6 points the multi-focal RANSAC sweep handles initialization
        robustly. Below that the geometry is under-constrained: P3P returns
        up to 4 candidate poses, EPnP can lock onto a mirror solution. We
        use the known camera position as a tiebreaker: project each
        candidate's camera center, pick the one closest to the prior.

        Returns:
            (rvec, tvec, ransac_info) or None on failure.
        """
        assert self.camera_position is not None, "few-points branch requires camera_position"
        n = len(world_pts)
        if n < 3:
            return None

        cx, cy = self.K[0, 2], self.K[1, 2]
        dist = self.dist_coeffs
        pos_target = np.array(self.camera_position, dtype=np.float64)

        # Sample focal candidates densely — P3P is highly sensitive to f,
        # and the right answer can sit between sparse profile-anchored
        # samples. Walk the allowed range in fixed steps.
        f_min, f_max = self.focal_bounds
        focal_step = max(100.0, (f_max - f_min) / 20.0)
        # ``arange(start, f_max + 1.0, step)`` can produce a candidate
        # one step beyond f_max; clip so a winning candidate never
        # sets self.K[0,0] outside the LM's bounds.
        focal_candidates = [
            min(f, f_max) for f in np.arange(f_min, f_max + 1.0, focal_step)
        ]

        # P3P needs exactly 3 points; for n>3 try multiple subsets so
        # disambiguation has more signal. Cap at 4 subsets for speed.
        obj_for_p3p_indices = [list(range(3))]
        if n >= 4:
            obj_for_p3p_indices += [
                [0, 1, n - 1],
                [0, n - 2, n - 1],
            ]
            if n == 5:
                obj_for_p3p_indices.append([1, 2, 3])

        best = None
        best_score = float("inf")

        for f_try in focal_candidates:
            K_try = np.array(
                [[f_try, 0, cx], [0, f_try, cy], [0, 0, 1]], dtype=np.float64,
            )
            for subset in obj_for_p3p_indices:
                obj_sub = world_pts[subset].astype(np.float64).reshape(-1, 1, 3)
                img_sub = img_pts[subset].astype(np.float64).reshape(-1, 1, 2)

                try:
                    n_sol, rvecs, tvecs = cv2.solveP3P(
                        obj_sub, img_sub, K_try, dist, flags=cv2.SOLVEPNP_P3P,
                    )
                except cv2.error:
                    n_sol = 0

                if n_sol <= 0:
                    continue

                for rv, tv in zip(rvecs, tvecs):
                    R0, _ = cv2.Rodrigues(rv)
                    cc0 = (-R0.T @ tv.ravel()).ravel()
                    if cc0[2] <= 0.1:
                        continue  # below ground

                    # Cheap LM polish on ALL points so we score how well a
                    # candidate explains the full set, not just the P3P
                    # subset. Without this, a mis-focal P3P can luck into
                    # fitting the 3 chosen points perfectly while the other
                    # n-3 points blow up.
                    try:
                        rv_pol, tv_pol = cv2.solvePnPRefineLM(
                            world_pts.astype(np.float64).reshape(-1, 1, 3),
                            img_pts.astype(np.float64).reshape(-1, 1, 2),
                            K_try, dist, rv.copy(), tv.copy(),
                            (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 50, 1e-4),
                        )
                    except cv2.error:
                        rv_pol, tv_pol = rv, tv

                    R, _ = cv2.Rodrigues(rv_pol)
                    cam_center = (-R.T @ tv_pol.ravel()).ravel()
                    if cam_center[2] <= 0.1:
                        continue

                    proj = project_points_2d(world_pts, rv_pol, tv_pol, K_try, dist)
                    reproj_err = float(np.mean(np.linalg.norm(proj - img_pts, axis=1)))
                    pos_err = float(np.linalg.norm(cam_center - pos_target))

                    # Position deviation is in meters, reproj in pixels.
                    # Both matter — weight position heavily since the prior
                    # is the whole reason we're in this branch.
                    score = pos_err * 10.0 + reproj_err

                    if score < best_score:
                        best_score = score
                        best = (
                            rv_pol.ravel(), tv_pol.ravel(), f_try,
                            cam_center, reproj_err, pos_err,
                        )

        if best is None:
            logger.debug("Few-point P3P found no valid candidate")
            self._last_ransac_debug = {
                "success": False,
                "failure_reason": "p3p_no_valid_candidate",
                "reprojection_threshold": float(self.ransac_reproj_error),
                "total_points": int(len(img_pts)),
                "inlier_count": 0,
                "inlier_indices": [],
            }
            return None

        rvec, tvec, best_f, cam_center, reproj_err, pos_err = best

        # Update self.K so _refine_7dof starts from this f
        self.K[0, 0] = best_f
        self.K[1, 1] = best_f

        K_best = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
        projected = project_points_2d(world_pts, rvec, tvec, K_best, dist)
        per_point_errors = np.linalg.norm(projected - img_pts, axis=1)
        inlier_mask = per_point_errors < self.ransac_reproj_error
        inlier_indices = np.where(inlier_mask)[0].tolist()

        logger.debug(
            "Few-point P3P: n=%d, f=%.0f, cam=(%.1f,%.1f,%.1f) [target=(%.1f,%.1f,%.1f)], "
            "pos_err=%.2fm, reproj=%.2fpx",
            n, best_f, cam_center[0], cam_center[1], cam_center[2],
            pos_target[0], pos_target[1], pos_target[2], pos_err, reproj_err,
        )

        n_in = int(np.sum(inlier_mask))
        n_out = int(np.sum(~inlier_mask))
        ransac_info = {
            "success": True,
            "method": "few_points_p3p",
            "reprojection_threshold": float(self.ransac_reproj_error),
            "total_points": int(len(img_pts)),
            "inlier_count": int(len(inlier_indices)),
            "outlier_count": int(len(img_pts) - len(inlier_indices)),
            "inlier_indices": inlier_indices,
            "inlier_mask": inlier_mask,
            "per_point_errors": per_point_errors,
            "mean_inlier_error": float(np.mean(per_point_errors[inlier_mask])) if n_in else 0.0,
            "mean_outlier_error": float(np.mean(per_point_errors[~inlier_mask])) if n_out else 0.0,
            "mean_all_error": float(np.mean(per_point_errors)),
            "rvec_init": rvec.copy(),
            "tvec_init": tvec.copy(),
            "position_prior_error_m": float(pos_err),
        }
        return rvec, tvec, ransac_info

    def _refine_7dof(self, rvec_init, tvec_init, img_pts, world_pts, line_constraints):
        """Bounded LM refinement with point + line + (optional position) residuals.

        Two modes:
        - Standard: state = [rvec(3), tvec(3), f] (7-DOF). camera_position
          is enforced as a soft residual (cauchy + replicated copies).
        - Locked (lock_camera_position=True): state = [yaw, el, roll, f]
          (4-DOF) using a look-at parameterization (yaw around world Z,
          elevation below horizon, roll about optical axis). ``el`` is
          box-bounded by ``pitch_bounds_deg`` and ``roll`` by
          ``_LOOKAT_ROLL_BOUND_DEG`` so the LM literally cannot reach
          camera-pointing-up / heavily tilted poses — Rodrigues rvec
          can't express that as a box bound (its components have no
          single-axis physical meaning). The standard path stays on
          rvec because RANSAC warm starts and the joint-focal stage
          still rely on it; the locked path is where degenerate
          orientations were biting (Pass 2 lock-C frames with ≤5 KP).

        cx/cy fixed at image center, distortion fixed at zero.
        """
        f_init = self.K[0, 0]
        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_zero = np.zeros(5, dtype=np.float64)

        # Defensive: clamp f_init into focal_bounds. The few-point P3P
        # path (``_solve_few_pts_p3p``) walks ``np.arange(f_min, f_max +
        # 1.0, step)`` which can sample one step past f_max; if that
        # candidate wins, ``self.K[0,0]`` ends up just above the bound
        # and scipy's TRF rejects the next frame's ``x0``. Clamp here so
        # the LM always starts inside the box.
        f_lo, f_hi = self.focal_bounds
        f_init = float(np.clip(f_init, f_lo, f_hi))

        locked = self.lock_camera_position
        pos_target = (
            np.array(self.camera_position, dtype=np.float64)
            if self.camera_position is not None else None
        )

        if locked:
            # Convert the rvec warm start to look-at angles (yaw, el, roll)
            # and clamp into the same bounds the LM will see, so scipy's
            # TRF doesn't reject ``x0`` for being on the boundary.
            R_init, _ = cv2.Rodrigues(np.asarray(rvec_init).reshape(3, 1))
            yaw0, el0, roll0 = _R_to_lookat(R_init)
            el_lo_rad = np.radians(self.pitch_bounds_deg[0]) if self.pitch_bounds_deg else -np.pi / 2 + 1e-3
            el_hi_rad = np.radians(self.pitch_bounds_deg[1]) if self.pitch_bounds_deg else +np.pi / 2 - 1e-3
            roll_lim_rad = np.radians(_LOOKAT_ROLL_BOUND_DEG)
            eps = 1e-4
            el0 = float(np.clip(el0, el_lo_rad + eps, el_hi_rad - eps))
            roll0 = float(np.clip(roll0, -roll_lim_rad + eps, roll_lim_rad - eps))
            x0 = np.array([yaw0, el0, roll0, f_init], dtype=np.float64)
        else:
            x0 = np.concatenate([rvec_init.ravel(), tvec_init.ravel(), [f_init]])

        # Pre-compute line normals from detected endpoints (constant during optimization)
        line_normals = []
        line_origins = []
        valid_lc = []
        for lc in line_constraints:
            img_line = lc["img_samples"]
            p1 = img_line[0]
            p2 = img_line[-1]
            line_dir = p2 - p1
            line_len = np.linalg.norm(line_dir)
            if line_len < 1e-6:
                continue
            normal = np.array([-line_dir[1], line_dir[0]]) / line_len
            line_normals.append(normal)
            line_origins.append(p1)
            valid_lc.append(lc)

        n_samples = self.line_sample_points
        per_sample_weight = self.line_weight / np.sqrt(n_samples) if n_samples > 0 else 0.0

        def unpack(x):
            if locked:
                yaw, el, roll = x[0], x[1], x[2]
                f = x[3]
                R = _lookat_to_R(yaw, el, roll)
                rvec, _ = cv2.Rodrigues(R)
                rvec = rvec.ravel()
                tvec = -R @ pos_target
            else:
                rvec = x[:3]
                tvec = x[3:6]
                f = x[6]
            return rvec, tvec, f

        def cost_fn(x):
            rvec, tvec, f = unpack(x)
            K_cur = np.array([[f, 0, cx_fixed], [0, f, cy_fixed], [0, 0, 1]], dtype=np.float64)

            # Point residuals
            projected = project_points_2d(
                world_pts, rvec, tvec, K_cur, dist_zero,
            )
            point_residuals = (projected - img_pts).ravel()

            all_residuals = [point_residuals]

            # Line residuals (zero out projections far outside image bounds)
            if valid_lc:
                img_bound = max(self.width, self.height) * 3
                for i, lc in enumerate(valid_lc):
                    proj_line = project_points_2d(
                        lc["world_samples"], rvec, tvec, K_cur, dist_zero,
                    )
                    diffs = proj_line - line_origins[i]
                    distances = diffs @ line_normals[i]
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

            # World-space back-projection residuals
            if self.world_residual_weight > 0:
                R_cur, _ = cv2.Rodrigues(rvec)
                cam_center = -R_cur.T @ tvec.ravel()
                pts_undist = cv2.undistortPoints(
                    img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_zero
                ).reshape(-1, 2)
                ones = np.ones((len(pts_undist), 1))
                rays_cam = np.hstack([pts_undist, ones])
                rays_world = (R_cur.T @ rays_cam.T).T
                t_params = -cam_center[2] / rays_world[:, 2]
                wp = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world
                diff_xy = (wp[:, :2] - world_pts[:, :2]) * self.world_residual_weight
                invalid = (np.abs(rays_world[:, 2]) < 1e-10) | (t_params < 0)
                diff_xy[invalid] = 0.0
                all_residuals.append(diff_xy.ravel())

            # Camera position SOFT constraint (skipped when hard-locked).
            # Replicate into many small residuals so cauchy loss doesn't
            # suppress them — each copy stays near the quadratic regime.
            if pos_target is not None and not locked:
                R_cur, _ = cv2.Rodrigues(rvec)
                cam_center = -R_cur.T @ tvec.ravel()
                n_copies = 50
                w_per_copy = self.position_weight / np.sqrt(n_copies)
                pos_single = (cam_center - pos_target) * w_per_copy
                all_residuals.append(np.tile(pos_single, n_copies))

                # Per-axis HARD bounds. tvec lives in cam coords; the
                # bound is on the world-frame camera centre, so we can't
                # express it in scipy's `bounds=` argument directly.
                # Instead emit a barrier-style residual that's 0 inside
                # the box and grows linearly outside, weighted high
                # enough that Cauchy loss can't damp it away.
                # IMPORTANT: ``least_squares`` requires the residual
                # vector to have a fixed length across calls, so we
                # ALWAYS append the 3-element vector (zeros when the
                # camera centre is inside the box).
                if self.position_bounds_m is not None:
                    excess = np.zeros(3)
                    for axis in range(3):
                        bnd = self.position_bounds_m[axis]
                        if bnd is None or bnd <= 0:
                            continue
                        delta = cam_center[axis] - pos_target[axis]
                        if delta > bnd:
                            excess[axis] = delta - bnd
                        elif delta < -bnd:
                            excess[axis] = delta + bnd
                    # Heavy weight (1000) — a 1m violation produces a
                    # 1000 px-equivalent residual, far above any
                    # legitimate reprojection cost.
                    all_residuals.append(excess * 1000.0)

            # Camera tilt (pitch_deg) barrier.
            # In the locked path, ``el`` is a box-bounded LM variable so
            # the constraint is enforced exactly by scipy's TRF — no
            # residual needed. In the standard (rvec-parameterised) path,
            # we keep the soft barrier as before because rvec components
            # don't have a single-axis physical meaning.
            if self.pitch_bounds_deg is not None and not locked:
                R_tilt, _ = cv2.Rodrigues(rvec)
                fwd_z = float((R_tilt.T @ np.array([0.0, 0.0, 1.0]))[2])
                # Clamp domain for arcsin stability; sin range is [-1, 1].
                pitch_rad = np.arcsin(-np.clip(fwd_z, -1.0, 1.0))
                pitch_deg_cur = float(np.degrees(pitch_rad))
                lo, hi = self.pitch_bounds_deg
                if pitch_deg_cur < lo:
                    excess_deg = pitch_deg_cur - lo  # negative
                elif pitch_deg_cur > hi:
                    excess_deg = pitch_deg_cur - hi  # positive
                else:
                    excess_deg = 0.0
                # Weight 100 px per degree of violation — a 5° drift
                # produces 500 px-equivalent residual, swamping the
                # ~20 px reprojection cost the LM otherwise enjoys.
                all_residuals.append(np.array([excess_deg * 100.0]))

            return np.concatenate(all_residuals)

        if locked:
            # Look-at parameterisation: [yaw, el, roll, f].
            # ``el`` (below-horizon angle, positive = looking down) is
            # box-bounded by pitch_bounds_deg — this is the whole reason
            # for the reparametrisation: rvec components can't express
            # "camera not pointing up" as a box.
            # ``roll`` is bounded to a small symmetric window (±15°);
            # legitimate handheld/sideline rigs are roughly level.
            # ``yaw`` is left unbounded modulo 2π — TRF works fine with
            # +inf bounds and the ``arctan2`` extraction returns into
            # (-π, π].
            el_lo_rad = np.radians(self.pitch_bounds_deg[0]) if self.pitch_bounds_deg else -np.pi / 2 + 1e-3
            el_hi_rad = np.radians(self.pitch_bounds_deg[1]) if self.pitch_bounds_deg else +np.pi / 2 - 1e-3
            roll_lim_rad = np.radians(_LOOKAT_ROLL_BOUND_DEG)
            bounds = (
                [-np.inf, el_lo_rad, -roll_lim_rad, self.focal_bounds[0]],
                [+np.inf, el_hi_rad, +roll_lim_rad, self.focal_bounds[1]],
            )
        else:
            bounds = (
                [-np.inf] * 6 + [self.focal_bounds[0]],
                [+np.inf] * 6 + [self.focal_bounds[1]],
            )
        result = least_squares(
            cost_fn, x0, method="trf", bounds=bounds,
            loss="cauchy", f_scale=15.0, max_nfev=300,
        )

        rvec_opt, tvec_opt, f_opt = unpack(result.x)
        return rvec_opt, tvec_opt, f_opt

    def _compute_reprojection_stats(self, rvec, tvec, img_pts, world_pts, K=None, dist=None):
        """Compute reprojection error statistics."""
        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        projected = project_points_2d(world_pts, rvec, tvec, K_use, dist_use)
        errors = np.linalg.norm(projected - img_pts, axis=1)

        mean_error = float(np.mean(errors)) if len(errors) > 0 else float("inf")
        # All points that reach this stage participated in optimization —
        # outlier rejection (if any) already happened in the world-error loop
        inlier_mask = np.ones(len(errors), dtype=bool)
        inlier_count = len(errors)

        return mean_error, inlier_mask, inlier_count

    def _compute_world_error(self, rvec, tvec, img_pts, world_pts, K=None, dist=None):
        """Compute world-space back-projection error for detected keypoints (meters).

        Back-projects detected image pixels to ground plane (z=0) and measures
        distance from true world positions.
        """
        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        errors = self._backproject_errors(rvec, tvec, img_pts, world_pts, K_use, dist_use)
        valid = np.isfinite(errors)
        mean_error = float(np.mean(errors[valid])) if valid.any() else float("inf")
        return mean_error, errors

    def _compute_world_error_all(self, rvec, tvec, K=None, dist=None):
        """Compute world-space round-trip error for ALL 57 ground keypoints (meters).

        For each template keypoint: project world→image, then back-project image→world.
        The round-trip deviation reveals projection model accuracy across the full pitch,
        including points outside the camera FOV.
        """
        from .pnlcalib import KeypointMapper

        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

        all_world = self._field_world_coords
        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS

        # Collect ground keypoints
        kp_ids = []
        world_3d = []
        for kid, (wx, wy) in enumerate(all_world):
            if kid in non_ground:
                continue
            kp_ids.append(kid)
            world_3d.append([wx, wy, 0.0])
        world_3d = np.array(world_3d, dtype=np.float64)

        # Project world→image
        img_pts = project_points_2d(world_3d, rvec, tvec, K_use, dist_use)

        # Back-project image→world and compute error
        errors = self._backproject_errors(rvec, tvec, img_pts, world_3d, K_use, dist_use)

        # Build per-keypoint dict
        per_kp = {}
        valid_errors = []
        for i, kid in enumerate(kp_ids):
            per_kp[kid] = float(errors[i])
            if np.isfinite(errors[i]):
                valid_errors.append(errors[i])

        mean_error = float(np.mean(valid_errors)) if valid_errors else float("inf")
        return mean_error, per_kp

    def _backproject_errors(self, rvec, tvec, img_pts, world_pts, K, dist):
        """Back-project image points to ground plane and compute distance from world_pts."""
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam_center = -R.T @ tvec.ravel()

        pts_undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2).astype(np.float64), K, dist
        ).reshape(-1, 2)

        errors = np.full(len(pts_undist), float("inf"))
        for j in range(len(pts_undist)):
            ray_world = R.T @ np.array([pts_undist[j, 0], pts_undist[j, 1], 1.0])
            if abs(ray_world[2]) < 1e-10:
                continue
            t_param = -cam_center[2] / ray_world[2]
            if t_param < 0:
                continue
            wp = cam_center + t_param * ray_world
            errors[j] = np.sqrt((wp[0] - world_pts[j, 0])**2 + (wp[1] - world_pts[j, 1])**2)
        return errors

    def joint_optimize_intrinsics(self, frame_data_list):
        """Cross-frame joint optimization: shared focal length, per-frame extrinsics.

        Solves for a single f across all frames while each frame keeps its own
        (rvec, tvec). cx/cy fixed at image center, distortion fixed at zero.

        Args:
            frame_data_list: list of dicts, each with:
                - "rvec": (3,) initial rotation vector
                - "tvec": (3,) initial translation vector
                - "img_pts": (N, 2) detected image points
                - "world_pts": (N, 3) corresponding world points
                - "line_constraints": list of line constraint dicts

        Returns:
            dict with optimized focal length and per-frame extrinsics, or None.
        """
        n_frames = len(frame_data_list)
        if n_frames < 2:
            logger.warning("Joint optimization needs ≥2 frames, got %d", n_frames)
            return None

        N_INTRINSICS = 1  # f only
        f_init = self.K[0, 0]
        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_zero = np.zeros(5, dtype=np.float64)

        locked = self.lock_camera_position
        pos_target_arr = (
            np.array(self.camera_position, dtype=np.float64)
            if self.camera_position is not None else None
        )
        per_frame_dof = 3 if locked else 6

        # State vector: [f] + [per-frame extrinsics] × N
        # locked  : per-frame = rvec(3)            → 1 + 3N params
        # default : per-frame = rvec(3), tvec(3)   → 1 + 6N params
        x0_parts = [np.array([f_init])]
        for fd in frame_data_list:
            x0_parts.append(fd["rvec"].ravel())
            if not locked:
                x0_parts.append(fd["tvec"].ravel())
        x0 = np.concatenate(x0_parts)

        # Pre-compute line normals per frame (constant during optimization)
        frame_line_info = []
        n_samples = self.line_sample_points
        per_sample_weight = self.line_weight / np.sqrt(n_samples) if n_samples > 0 else 0.0

        for fd in frame_data_list:
            normals, origins, valid_lcs = [], [], []
            for lc in fd.get("line_constraints", []):
                img_line = lc["img_samples"]
                p1, p2 = img_line[0], img_line[-1]
                line_dir = p2 - p1
                line_len = np.linalg.norm(line_dir)
                if line_len < 1e-6:
                    continue
                normal = np.array([-line_dir[1], line_dir[0]]) / line_len
                normals.append(normal)
                origins.append(p1)
                valid_lcs.append(lc)
            frame_line_info.append((normals, origins, valid_lcs))

        img_bound = max(self.width, self.height) * 3

        def cost_fn(x):
            f = x[0]
            K_cur = np.array([[f, 0, cx_fixed], [0, f, cy_fixed], [0, 0, 1]], dtype=np.float64)

            all_residuals = []

            for i, fd in enumerate(frame_data_list):
                offset = N_INTRINSICS + i * per_frame_dof
                rvec = x[offset:offset + 3].reshape(3, 1)
                if locked:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    tvec = (-R_cur @ pos_target_arr).reshape(3, 1)
                else:
                    tvec = x[offset + 3:offset + 6].reshape(3, 1)
                img_pts = fd["img_pts"]
                world_pts = fd["world_pts"]

                # Point residuals
                projected = project_points_2d(
                    world_pts, rvec, tvec, K_cur, dist_zero,
                )
                all_residuals.append((projected - img_pts).ravel())

                # Line residuals
                normals, origins, valid_lcs = frame_line_info[i]
                for j, lc in enumerate(valid_lcs):
                    proj_line = project_points_2d(
                        lc["world_samples"], rvec, tvec, K_cur, dist_zero,
                    )
                    distances = (proj_line - origins[j]) @ normals[j]
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

                # World-space back-projection residuals (vectorized)
                if self.world_residual_weight > 0:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    cam_center = -R_cur.T @ tvec.ravel()
                    pts_undist = cv2.undistortPoints(
                        img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_zero
                    ).reshape(-1, 2)
                    ones = np.ones((len(pts_undist), 1))
                    rays_cam = np.hstack([pts_undist, ones])
                    rays_world = (R_cur.T @ rays_cam.T).T
                    t_params = -cam_center[2] / rays_world[:, 2]
                    wp = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world
                    diff_xy = (wp[:, :2] - world_pts[:, :2]) * self.world_residual_weight
                    invalid = (np.abs(rays_world[:, 2]) < 1e-10) | (t_params < 0)
                    diff_xy[invalid] = 0.0
                    all_residuals.append(diff_xy.ravel())

                # Camera position SOFT constraint (skipped when hard-locked).
                if pos_target_arr is not None and not locked:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    cam_center = -R_cur.T @ tvec.ravel()
                    n_copies = 200
                    w_per_copy = self.position_weight / np.sqrt(n_copies)
                    pos_single = (cam_center - pos_target_arr) * w_per_copy
                    all_residuals.append(np.tile(pos_single, n_copies))

            return np.concatenate(all_residuals)

        # Bounds: shared f bounded, per-frame extrinsics unbounded
        lb = [self.focal_bounds[0]]
        ub = [self.focal_bounds[1]]
        for _ in range(n_frames):
            lb.extend([-np.inf] * per_frame_dof)
            ub.extend([+np.inf] * per_frame_dof)

        logger.info("Joint optimization: %d frames, %d params", n_frames, len(x0))
        result = least_squares(
            cost_fn, x0, method="trf", bounds=(lb, ub),
            loss="soft_l1", f_scale=15.0, max_nfev=2000,
        )

        # Extract results
        f_opt = float(result.x[0])
        K_opt = np.array([[f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]], dtype=np.float64)

        per_frame = []
        for i in range(n_frames):
            offset = N_INTRINSICS + i * per_frame_dof
            rvec_i = result.x[offset:offset + 3]
            if locked:
                R_i, _ = cv2.Rodrigues(rvec_i)
                tvec_i = -R_i @ pos_target_arr
            else:
                tvec_i = result.x[offset + 3:offset + 6]
            per_frame.append({"rvec": rvec_i, "tvec": tvec_i})

        logger.info("Joint result: f=%.1f, cost=%.2f", f_opt, result.cost)

        return {
            "K": K_opt,
            "dist_coeffs": dist_zero.copy(),
            "f": f_opt,
            "cx": float(cx_fixed),
            "cy": float(cy_fixed),
            "per_frame": per_frame,
            "cost": float(result.cost),
            "n_frames": n_frames,
        }
