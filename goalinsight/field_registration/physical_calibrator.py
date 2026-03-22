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
    ):
        """Initialize with camera intrinsics profile.

        Args:
            K: 3x3 intrinsic matrix (initial guess, f/cx/cy are optimized per-frame).
            dist_coeffs: 5-element distortion coefficients (k1 optimized, rest locked).
            image_size: (width, height) of video frames.
            ransac_reproj_error: PnP RANSAC reprojection threshold in pixels.
            line_weight: Weight alpha for line residuals vs point residuals.
            line_sample_points: Number of points sampled along each line.
            focal_bounds: (min_f, max_f) absolute bounds for focal length.
            cx_bounds: (min_offset, max_offset) from profile cx.
            cy_bounds: (min_offset, max_offset) from profile cy.
            k1_bounds: (min_k1, max_k1) absolute bounds for radial distortion k1.
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
        img_pts, world_pts, kp_ids = self._prepare_correspondences(
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

        # Step 4: 10-DOF joint point+line optimization (rvec, tvec, f, cx, cy, k1)
        rvec_opt, tvec_opt, f_opt, cx_opt, cy_opt, k1_opt = self._refine_10dof(
            rvec_init, tvec_init, img_pts, world_pts, line_constraints
        )

        # Build optimized intrinsics
        K_opt = self.K.copy()
        K_opt[0, 0] = f_opt
        K_opt[1, 1] = f_opt
        K_opt[0, 2] = cx_opt
        K_opt[1, 2] = cy_opt

        dist_opt = self.dist_coeffs.copy()
        dist_opt[0] = k1_opt

        # Step 5: Compute stats and format result
        mean_error, inlier_mask, inlier_count = self._compute_reprojection_stats(
            rvec_opt, tvec_opt, img_pts, world_pts, K=K_opt, dist=dist_opt
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
        return (
            img_pts.astype(np.float64) if len(img_pts) > 0 else np.zeros((0, 2), dtype=np.float64),
            world_pts.astype(np.float64) if len(world_pts) > 0 else np.zeros((0, 3), dtype=np.float64),
            kp_ids,
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
        """Initialize camera pose via PnP RANSAC with fixed intrinsics.

        Single call — no focal/distortion sweep needed since K and dist are fixed.

        Returns:
            (rvec, tvec, ransac_info) or None on failure.
            ransac_info contains inlier indices, per-point errors, and init pose.
        """
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=world_pts.reshape(-1, 1, 3),
            imagePoints=img_pts.reshape(-1, 1, 2),
            cameraMatrix=self.K,
            distCoeffs=self.dist_coeffs,
            reprojectionError=self.ransac_reproj_error,
            iterationsCount=2000,
            flags=cv2.SOLVEPNP_SQPNP,
        )

        if not success or inliers is None or len(inliers) < 4:
            n_inliers = len(inliers) if inliers is not None else 0
            logger.debug("PnP RANSAC failed (success=%s, inliers=%s)",
                         success, n_inliers)
            self._last_ransac_debug = {
                "success": bool(success),
                "failure_reason": "pnp_not_converged" if not success else f"too_few_inliers ({n_inliers} < 4)",
                "reprojection_threshold": float(self.ransac_reproj_error),
                "total_points": int(len(img_pts)),
                "inlier_count": int(n_inliers),
                "inlier_indices": inliers.ravel().tolist() if inliers is not None else [],
            }
            return None

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

    def _refine_10dof(self, rvec_init, tvec_init, img_pts, world_pts, line_constraints):
        """10-DOF bounded optimization with point+line residuals.

        State vector: [rvec(3), tvec(3), f, cx, cy, k1] — 10 parameters.
        f, cx, cy, k1 are bounded; k2/p1/p2/k3 stay locked from profile.

        Line residuals are normalized so each line contributes weight equivalent
        to one point (divided by sqrt(n_samples) to balance against point count).
        """
        f_init = self.K[0, 0]
        cx_init = self.K[0, 2]
        cy_init = self.K[1, 2]
        k1_init = self.dist_coeffs[0]
        x0 = np.concatenate([
            rvec_init.ravel(), tvec_init.ravel(),
            [f_init, cx_init, cy_init, k1_init],
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

        # Base distortion (k2, p1, p2, k3 stay locked)
        dist_base = self.dist_coeffs.copy()

        def cost_fn(x):
            rvec = x[:3].reshape(3, 1)
            tvec = x[3:6].reshape(3, 1)
            f, cx, cy, k1 = x[6], x[7], x[8], x[9]
            K_cur = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
            dist_cur = dist_base.copy()
            dist_cur[0] = k1

            # Point residuals: projected - detected
            projected, _ = cv2.projectPoints(
                world_pts.reshape(-1, 1, 3),
                rvec, tvec, K_cur, dist_cur,
            )
            projected = projected.reshape(-1, 2)
            point_residuals = (projected - img_pts).ravel()

            if not valid_lc:
                return point_residuals

            # Line residuals (zero out degenerate projections outside image bounds)
            img_bound = max(self.width, self.height) * 3
            line_residuals = []
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
                line_residuals.append(distances * per_sample_weight)

            if line_residuals:
                return np.concatenate([point_residuals, np.concatenate(line_residuals)])
            return point_residuals

        bounds = (
            [-np.inf] * 6 + [
                self.focal_bounds[0],
                cx_init + self.cx_bounds[0],
                cy_init + self.cy_bounds[0],
                self.k1_bounds[0],
            ],
            [+np.inf] * 6 + [
                self.focal_bounds[1],
                cx_init + self.cx_bounds[1],
                cy_init + self.cy_bounds[1],
                self.k1_bounds[1],
            ],
        )
        result = least_squares(
            cost_fn, x0, method="trf", bounds=bounds,
            loss="soft_l1", max_nfev=300,
        )

        return (
            result.x[:3], result.x[3:6],
            float(result.x[6]), float(result.x[7]),
            float(result.x[8]), float(result.x[9]),
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
        inlier_mask = errors < self.ransac_reproj_error
        inlier_count = int(np.sum(inlier_mask))

        return mean_error, inlier_mask, inlier_count
