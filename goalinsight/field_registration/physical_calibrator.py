"""Physical camera calibrator with fixed intrinsics and 6-DOF extrinsic optimization.

Core philosophy: camera intrinsics (K, distortion) are loaded from a known profile
and NEVER optimized. Only the 6 extrinsic parameters (rvec, tvec) are refined via
joint point+line Levenberg-Marquardt optimization.

This eliminates the brute-force focal/distortion sweep that fails under heavy
barrel distortion (e.g., Veo ~120deg FOV cameras).
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from .pnlcalib.curve_utils import (
    compute_cumulated_lengths,
    interpolate_on_polyline,
    sample_points_on_image_line,
)
from .pnlcalib.line_mapping import LineMapper

logger = logging.getLogger(__name__)

# Ground-only line IDs (exclude crossbars and goal posts)
GROUND_LINE_IDS = set(range(23)) - {6, 7, 8, 9, 10, 11}

# Keypoints that lie on each ground line (ordered by arc-length parameter t).
# Copied from BroadTrackCalibrator to avoid import dependency.
LINE_KEYPOINTS: dict[int, list[int]] = {
    0:  [3, 4],                         # Big rect. left top
    1:  [4, 30, 45, 33, 24],            # Big rect. left side
    2:  [24, 23],                        # Big rect. left bottom
    3:  [6, 5],                          # Big rect. right top
    4:  [5, 32, 55, 35, 25],            # Big rect. right side
    5:  [25, 26],                        # Big rect. right bottom
    12: [28, 34, 50, 31, 1],            # Middle line
    13: [0, 1, 2],                       # Side line top
    14: [27, 23, 19, 15, 11, 7, 3, 0],  # Side line left (goal line)
    15: [29, 26, 22, 17, 13, 10, 6, 2], # Side line right (goal line)
    16: [27, 28, 29],                    # Side line bottom
    17: [7, 8],                          # Small rect. left top
    18: [8, 20],                         # Small rect. left side
    19: [20, 19],                        # Small rect. left bottom
    20: [10, 9],                         # Small rect. right top
    21: [9, 21],                         # Small rect. right side
    22: [21, 22],                        # Small rect. right bottom
}


class PhysicalCalibrator:
    """Camera calibrator with fixed intrinsics and 6-DOF extrinsic optimization.

    Uses known camera intrinsics (K, distortion) from a profile and only optimizes
    the 6 extrinsic parameters (rvec, tvec) via joint point+line LM optimization.
    """

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        image_size: tuple[int, int],
        ransac_reproj_error: float = 15.0,
        line_weight: float = 1.0,
        line_sample_points: int = 20,
        focal_bounds: tuple[float, float] = (1200.0, 2200.0),
        cx_bounds: tuple[float, float] = (-100.0, 100.0),
        cy_bounds: tuple[float, float] = (-50.0, 50.0),
        k1_bounds: tuple[float, float] = (-0.35, -0.15),
        k2_bounds: tuple[float, float] = (-0.05, 0.10),
        k3_bounds: tuple[float, float] = (-0.05, 0.05),
        intrinsic_reg_weight: float = 0.0,
        world_residual_weight: float = 0.0,
        world_error_threshold: float = 5.0,
    ):
        """Initialize with camera intrinsics profile.

        Args:
            K: 3x3 intrinsic matrix (initial guess, f/cx/cy are optimized per-frame).
            dist_coeffs: 5-element distortion coefficients (k1, k2 optimized, rest locked).
            image_size: (width, height) of video frames.
            ransac_reproj_error: PnP RANSAC reprojection threshold in pixels.
            line_weight: Weight alpha for line residuals vs point residuals.
            line_sample_points: Number of points sampled along each line.
            focal_bounds: (min_f, max_f) absolute bounds for focal length.
            cx_bounds: (min_offset, max_offset) from profile cx.
            cy_bounds: (min_offset, max_offset) from profile cy.
            k1_bounds: (min_k1, max_k1) absolute bounds for radial distortion k1.
            k2_bounds: (min_k2, max_k2) absolute bounds for radial distortion k2.
            k3_bounds: (min_k3, max_k3) absolute bounds for radial distortion k3.
            intrinsic_reg_weight: Regularization weight penalizing intrinsic deviation
                from profile. 0 = no regularization, higher = stronger prior.
            world_residual_weight: Weight for world-space back-projection residuals.
                0 = disabled. Adds residuals that minimize deviation of back-projected
                detected pixels from true world positions (meters).
            world_error_threshold: World-space error threshold (meters) for iterative
                outlier rejection. Set to float('inf') to disable outlier rejection
                and use all keypoints.
        """
        self.K = np.array(K, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64).ravel()
        self.width, self.height = image_size
        self.ransac_reproj_error = ransac_reproj_error
        self.line_weight = line_weight
        self.line_sample_points = line_sample_points
        self.focal_bounds = focal_bounds
        self.cx_bounds = cx_bounds
        self.cy_bounds = cy_bounds
        self.k1_bounds = k1_bounds
        self.k2_bounds = k2_bounds
        self.k3_bounds = k3_bounds
        self.intrinsic_reg_weight = intrinsic_reg_weight
        self.world_residual_weight = world_residual_weight
        self.world_error_threshold = world_error_threshold

        self._keypoints: list[dict] = []
        self._lines: list[dict] = []
        self._last_debug: dict | None = None
        self._last_ransac_debug: dict | None = None

    def update(self, keypoints: list[dict], lines: list[dict]):
        """Update with detected keypoints and lines for current frame."""
        self._keypoints = keypoints
        self._lines = lines

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
        if len(world_pts) < 6:
            logger.debug("Too few keypoints (%d < 6), skipping", len(world_pts))
            self._last_debug = {
                "failure_reason": f"too_few_keypoints ({len(world_pts)} < 6)",
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

        # Step 3: Get initial pose estimate
        ransac_info = None
        if initial_rvec is not None and initial_tvec is not None:
            # Warm-start from previous frame — skip PnP RANSAC
            rvec_init = initial_rvec.copy().ravel()
            tvec_init = initial_tvec.copy().ravel()
            logger.debug("Using warm-start from previous frame")
        else:
            # Cold start — PnP RANSAC initialization
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

        # Step 4: 11-DOF joint point+line optimization (rvec, tvec, f, cx, cy, k1, k2)
        # with iterative world-error-based outlier rejection
        MAX_WORLD_OUTLIER_ITERS = 3

        cur_img_pts = img_pts
        cur_world_pts = world_pts
        cur_kp_ids = kp_ids
        cur_lc = line_constraints
        cur_rvec = rvec_init
        cur_tvec = tvec_init

        if np.isfinite(self.world_error_threshold):
            for world_iter in range(MAX_WORLD_OUTLIER_ITERS):
                rvec_opt, tvec_opt, f_opt, cx_opt, cy_opt, dist_opt_5 = self._refine_10dof(
                    cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
                )

                # Build optimized intrinsics for this iteration
                K_iter = self.K.copy()
                K_iter[0, 0] = f_opt
                K_iter[1, 1] = f_opt
                K_iter[0, 2] = cx_opt
                K_iter[1, 2] = cy_opt
                dist_iter = dist_opt_5.copy()

                # Compute per-point world error
                _, per_pt_werr = self._compute_world_error(
                    rvec_opt, tvec_opt, cur_img_pts, cur_world_pts,
                    K=K_iter, dist=dist_iter,
                )

                # Find world-error outliers
                keep_mask = per_pt_werr < self.world_error_threshold
                n_removed = int(np.sum(~keep_mask))
                if n_removed == 0 or world_iter == MAX_WORLD_OUTLIER_ITERS - 1:
                    break

                # Need at least 6 points to continue
                if int(np.sum(keep_mask)) < 6:
                    break

                logger.debug("World-error iter %d: removed %d points (threshold=%.1fm)",
                             world_iter, n_removed, self.world_error_threshold)

                cur_img_pts = cur_img_pts[keep_mask]
                cur_world_pts = cur_world_pts[keep_mask]
                cur_kp_ids = [cur_kp_ids[i] for i in range(len(cur_kp_ids)) if keep_mask[i]]
                cur_rvec = rvec_opt
                cur_tvec = tvec_opt

                # Re-derive line constraints from remaining keypoints
                cur_lc = self._derive_lines_from_keypoints(
                    cur_kp_ids, cur_img_pts, keypoint_mapper, line_mapper
                )
        else:
            # No outlier rejection — use all keypoints, single optimization pass
            rvec_opt, tvec_opt, f_opt, cx_opt, cy_opt, dist_opt_5 = self._refine_10dof(
                cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
            )

        # Use the filtered points for final result
        img_pts = cur_img_pts
        world_pts = cur_world_pts
        kp_ids = cur_kp_ids
        line_constraints = cur_lc

        # Build optimized intrinsics
        K_opt = self.K.copy()
        K_opt[0, 0] = f_opt
        K_opt[1, 1] = f_opt
        K_opt[0, 2] = cx_opt
        K_opt[1, 2] = cy_opt

        dist_opt = dist_opt_5.copy()

        # Step 5: Sanity check — reject catastrophically bad results
        projected_check, _ = cv2.projectPoints(
            world_pts.reshape(-1, 1, 3),
            rvec_opt.reshape(3, 1), tvec_opt.reshape(3, 1),
            K_opt, dist_opt,
        )
        median_err_check = float(np.median(np.linalg.norm(
            projected_check.reshape(-1, 2) - img_pts, axis=1
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
            "num_keypoints": len(self._keypoints),
            "num_lines": len(self._lines),
            "num_intersections": 0,
            "total_points": len(img_pts),
            "inliers": int(inlier_count),
            "intrinsics_init": {
                "f": float(self.K[0, 0]),
                "cx": float(self.K[0, 2]),
                "cy": float(self.K[1, 2]),
                "k1": float(self.dist_coeffs[0]),
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
        """Derive line constraints from detected keypoints on field lines.

        For each ground line, if >=2 of its keypoints are detected, connect the
        two most extreme ones in image space and sample points along that segment.
        """
        detected = {}
        for i, kid in enumerate(kp_ids):
            detected[kid] = kp_img[i]

        constraints = []
        non_ground = keypoint_mapper.NON_GROUND_KEYPOINTS

        for line_id, kp_list in LINE_KEYPOINTS.items():
            found = [(kid, detected[kid]) for kid in kp_list
                     if kid in detected and kid not in non_ground]
            if len(found) < 2:
                continue

            # Get world line endpoints
            ld = LineMapper.LINE_DEFINITIONS[line_id]
            p1_w = np.array(ld["p1"], dtype=np.float64)
            p2_w = np.array(ld["p2"], dtype=np.float64)
            polyline_3d = np.array([p1_w, p2_w])
            cum_lengths = compute_cumulated_lengths(polyline_3d)

            # Sample N evenly spaced 3D points along the world line
            total_len = cum_lengths[-1]
            if total_len < 1e-6:
                continue
            t_values = np.linspace(0, total_len, self.line_sample_points)
            world_samples = np.array([
                interpolate_on_polyline(polyline_3d, cum_lengths, t)
                for t in t_values
            ])

            # Image line: connect first and last detected keypoint
            first_img = found[0][1]
            last_img = found[-1][1]
            img_samples = sample_points_on_image_line(
                float(first_img[0]), float(first_img[1]),
                float(last_img[0]), float(last_img[1]),
                n_points=self.line_sample_points,
            )

            constraints.append({
                "world_samples": world_samples,
                "img_samples": img_samples,
                "line_id": line_id,
            })

        return constraints

    def _pnp_ransac_init(self, world_pts, img_pts):
        """Initialize camera pose via multi-trial PnP RANSAC with fixed intrinsics.

        Runs multiple RANSAC trials with different solvers and picks the result
        with the most inliers to avoid bad local minima from unlucky sampling.

        Returns:
            (rvec, tvec, ransac_info) or None on failure.
            ransac_info contains inlier indices, per-point errors, and init pose.
        """
        solvers = [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_SQPNP]
        best_result = None
        best_inlier_count = -1

        for solver in solvers:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=world_pts.reshape(-1, 1, 3),
                imagePoints=img_pts.reshape(-1, 1, 2),
                cameraMatrix=self.K,
                distCoeffs=self.dist_coeffs,
                reprojectionError=self.ransac_reproj_error,
                iterationsCount=2000,
                flags=solver,
            )
            if success and inliers is not None and len(inliers) >= 4:
                if len(inliers) > best_inlier_count:
                    best_inlier_count = len(inliers)
                    best_result = (rvec, tvec, inliers)

        if best_result is None:
            logger.debug("PnP RANSAC failed (all %d trials)", len(solvers))
            self._last_ransac_debug = {
                "success": False,
                "failure_reason": f"all_{len(solvers)}_trials_failed",
                "reprojection_threshold": float(self.ransac_reproj_error),
                "total_points": int(len(img_pts)),
                "inlier_count": 0,
                "inlier_indices": [],
            }
            return None

        rvec, tvec, inliers = best_result

        # Compute per-point reprojection errors for RANSAC result
        projected, _ = cv2.projectPoints(
            world_pts.reshape(-1, 1, 3),
            rvec, tvec, self.K, self.dist_coeffs,
        )
        per_point_errors = np.linalg.norm(
            projected.reshape(-1, 2) - img_pts, axis=1
        )

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

    def _refine_10dof(self, rvec_init, tvec_init, img_pts, world_pts, line_constraints,
                       confidences=None):
        """12-DOF bounded optimization with point+line residuals.

        State vector: [rvec(3), tvec(3), f, cx, cy, k1, k2, k3] — 12 parameters.
        Tangential distortion (p1, p2) is fixed at 0.

        Line residuals are normalized so each line contributes weight equivalent
        to one point (divided by sqrt(n_samples) to balance against point count).
        """
        f_init = self.K[0, 0]
        cx_init = self.K[0, 2]
        cy_init = self.K[1, 2]
        k1_init = self.dist_coeffs[0]
        k2_init = self.dist_coeffs[1]
        k3_init = self.dist_coeffs[4] if len(self.dist_coeffs) > 4 else 0.0
        x0 = np.concatenate([
            rvec_init.ravel(), tvec_init.ravel(),
            [f_init, cx_init, cy_init, k1_init, k2_init, k3_init],
        ])

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

        def cost_fn(x):
            rvec = x[:3].reshape(3, 1)
            tvec = x[3:6].reshape(3, 1)
            f, cx, cy = x[6], x[7], x[8]
            k1, k2, k3 = x[9], x[10], x[11]
            K_cur = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_cur = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)

            # Point residuals: projected - detected
            projected, _ = cv2.projectPoints(
                world_pts.reshape(-1, 1, 3),
                rvec, tvec, K_cur, dist_cur,
            )
            projected = projected.reshape(-1, 2)
            point_residuals = (projected - img_pts).ravel()

            all_residuals = [point_residuals]

            # Line residuals (zero out degenerate projections outside image bounds)
            if valid_lc:
                img_bound = max(self.width, self.height) * 3
                for i, lc in enumerate(valid_lc):
                    proj_line, _ = cv2.projectPoints(
                        lc["world_samples"].reshape(-1, 1, 3),
                        rvec, tvec, K_cur, dist_cur,
                    )
                    proj_line = proj_line.reshape(-1, 2)
                    diffs = proj_line - line_origins[i]
                    distances = diffs @ line_normals[i]
                    # Zero out degenerate projections (keeps residual vector fixed-size)
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

            # World-space back-projection residuals (vectorized)
            if self.world_residual_weight > 0:
                R_cur, _ = cv2.Rodrigues(rvec)
                cam_center = -R_cur.T @ tvec.ravel()
                pts_undist = cv2.undistortPoints(
                    img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_cur
                ).reshape(-1, 2)
                # Build rays in world space: R^T @ [u, v, 1]^T for all points
                ones = np.ones((len(pts_undist), 1))
                rays_cam = np.hstack([pts_undist, ones])  # (N, 3)
                rays_world = (R_cur.T @ rays_cam.T).T     # (N, 3)
                # Intersect with ground plane z=0
                t_params = -cam_center[2] / rays_world[:, 2]  # (N,)
                wp = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world  # (N, 3)
                diff_xy = (wp[:, :2] - world_pts[:, :2]) * self.world_residual_weight
                # Zero out invalid rays (near-horizontal or behind camera)
                invalid = (np.abs(rays_world[:, 2]) < 1e-10) | (t_params < 0)
                diff_xy[invalid] = 0.0
                all_residuals.append(diff_xy.ravel())

            # Intrinsic regularization: penalize deviation from profile values
            if self.intrinsic_reg_weight > 0:
                reg = np.array([
                    (f - f_init) / f_init * self.intrinsic_reg_weight,
                    (cx - cx_init) / 100.0 * self.intrinsic_reg_weight,
                    (cy - cy_init) / 100.0 * self.intrinsic_reg_weight,
                    (k1 - k1_init) / abs(k1_init) * self.intrinsic_reg_weight,
                    (k2 - k2_init) / max(abs(k2_init), 0.01) * self.intrinsic_reg_weight,
                ])
                all_residuals.append(reg)

            return np.concatenate(all_residuals)

        bounds = (
            [-np.inf] * 6 + [
                self.focal_bounds[0],
                cx_init + self.cx_bounds[0],
                cy_init + self.cy_bounds[0],
                self.k1_bounds[0],
                self.k2_bounds[0],
                self.k3_bounds[0],
            ],
            [+np.inf] * 6 + [
                self.focal_bounds[1],
                cx_init + self.cx_bounds[1],
                cy_init + self.cy_bounds[1],
                self.k1_bounds[1],
                self.k2_bounds[1],
                self.k3_bounds[1],
            ],
        )
        result = least_squares(
            cost_fn, x0, method="trf", bounds=bounds,
            loss="soft_l1", max_nfev=300,
        )

        return (
            result.x[:3], result.x[3:6],
            float(result.x[6]), float(result.x[7]),
            float(result.x[8]),
            np.array([result.x[9], result.x[10], 0.0, 0.0, result.x[11]]),
        )

    def _compute_reprojection_stats(self, rvec, tvec, img_pts, world_pts, K=None, dist=None):
        """Compute reprojection error statistics."""
        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        projected, _ = cv2.projectPoints(
            world_pts.reshape(-1, 1, 3),
            rvec.reshape(3, 1), tvec.reshape(3, 1),
            K_use, dist_use,
        )
        projected = projected.reshape(-1, 2)
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

        all_world = KeypointMapper.PNLCALIB_WORLD_COORDS_2D
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
        projected, _ = cv2.projectPoints(
            world_3d.reshape(-1, 1, 3),
            rvec.reshape(3, 1), tvec.reshape(3, 1),
            K_use, dist_use,
        )
        img_pts = projected.reshape(-1, 2)

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
        """Cross-frame joint optimization: shared intrinsics, per-frame extrinsics.

        Solves for a single (f, cx, cy, k1, k2) across all frames while each frame
        keeps its own (rvec, tvec). This eliminates per-frame intrinsic drift.

        Args:
            frame_data_list: list of dicts, each with:
                - "rvec": (3,) initial rotation vector
                - "tvec": (3,) initial translation vector
                - "img_pts": (N, 2) detected image points
                - "world_pts": (N, 3) corresponding world points
                - "line_constraints": list of line constraint dicts

        Returns:
            dict with optimized intrinsics and per-frame extrinsics, or None.
        """
        n_frames = len(frame_data_list)
        if n_frames < 2:
            logger.warning("Joint optimization needs ≥2 frames, got %d", n_frames)
            return None

        N_INTRINSICS = 6  # f, cx, cy, k1, k2, k3
        f_init = self.K[0, 0]
        cx_init = self.K[0, 2]
        cy_init = self.K[1, 2]
        k1_init = self.dist_coeffs[0]
        k2_init = self.dist_coeffs[1]
        k3_init = self.dist_coeffs[4] if len(self.dist_coeffs) > 4 else 0.0

        # State vector: [shared intrinsics (6)] + [per-frame rvec(3), tvec(3)] × N
        # Total: 6 + 6N parameters
        x0_parts = [np.array([f_init, cx_init, cy_init, k1_init, k2_init, k3_init])]
        for fd in frame_data_list:
            x0_parts.append(fd["rvec"].ravel())
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
            f, cx, cy = x[0], x[1], x[2]
            k1, k2, k3 = x[3], x[4], x[5]
            K_cur = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_cur = np.array([k1, k2, 0.0, 0.0, k3], dtype=np.float64)

            all_residuals = []

            for i, fd in enumerate(frame_data_list):
                offset = N_INTRINSICS + i * 6
                rvec = x[offset:offset + 3].reshape(3, 1)
                tvec = x[offset + 3:offset + 6].reshape(3, 1)
                img_pts = fd["img_pts"]
                world_pts = fd["world_pts"]

                # Point residuals
                projected, _ = cv2.projectPoints(
                    world_pts.reshape(-1, 1, 3), rvec, tvec, K_cur, dist_cur,
                )
                all_residuals.append((projected.reshape(-1, 2) - img_pts).ravel())

                # Line residuals
                normals, origins, valid_lcs = frame_line_info[i]
                for j, lc in enumerate(valid_lcs):
                    proj_line, _ = cv2.projectPoints(
                        lc["world_samples"].reshape(-1, 1, 3),
                        rvec, tvec, K_cur, dist_cur,
                    )
                    proj_line = proj_line.reshape(-1, 2)
                    distances = (proj_line - origins[j]) @ normals[j]
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

                # World-space back-projection residuals (vectorized)
                if self.world_residual_weight > 0:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    cam_center = -R_cur.T @ tvec.ravel()
                    pts_undist = cv2.undistortPoints(
                        img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_cur
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

            return np.concatenate(all_residuals)

        # Bounds: shared intrinsics bounded, per-frame extrinsics unbounded
        lb = [self.focal_bounds[0], cx_init + self.cx_bounds[0],
              cy_init + self.cy_bounds[0], self.k1_bounds[0], self.k2_bounds[0],
              self.k3_bounds[0]]
        ub = [self.focal_bounds[1], cx_init + self.cx_bounds[1],
              cy_init + self.cy_bounds[1], self.k1_bounds[1], self.k2_bounds[1],
              self.k3_bounds[1]]
        for _ in range(n_frames):
            lb.extend([-np.inf] * 6)
            ub.extend([+np.inf] * 6)

        logger.info("Joint optimization: %d frames, %d params", n_frames, len(x0))
        result = least_squares(
            cost_fn, x0, method="trf", bounds=(lb, ub),
            loss="soft_l1", max_nfev=2000,
        )

        # Extract results
        f_opt, cx_opt, cy_opt = result.x[0], result.x[1], result.x[2]
        k1_opt, k2_opt, k3_opt = result.x[3], result.x[4], result.x[5]
        K_opt = np.array([[f_opt, 0, cx_opt], [0, f_opt, cy_opt], [0, 0, 1]], dtype=np.float64)
        dist_opt = np.array([k1_opt, k2_opt, 0.0, 0.0, k3_opt], dtype=np.float64)

        per_frame = []
        for i in range(n_frames):
            offset = N_INTRINSICS + i * 6
            rvec_i = result.x[offset:offset + 3]
            tvec_i = result.x[offset + 3:offset + 6]
            per_frame.append({"rvec": rvec_i, "tvec": tvec_i})

        logger.info("Joint result: f=%.1f, cx=%.1f, cy=%.1f, k1=%.4f, k2=%.4f, k3=%.4f, cost=%.2f",
                     f_opt, cx_opt, cy_opt, k1_opt, k2_opt, k3_opt, result.cost)

        return {
            "K": K_opt,
            "dist_coeffs": dist_opt,
            "f": float(f_opt),
            "cx": float(cx_opt),
            "cy": float(cy_opt),
            "k1": float(k1_opt),
            "k2": float(k2_opt),
            "k3": float(k3_opt),
            "per_frame": per_frame,
            "cost": float(result.cost),
            "n_frames": n_frames,
        }
