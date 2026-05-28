"""BroadTrack-style camera calibrator for soccer field registration.

Implements the calibration algorithm inspired by BroadTrack (EVS):
- 9-parameter camera model: [angleAxis(3), position(3), focalLength, k1, k2]
- Cauchy robust loss function
- Joint keypoint + arc-length parameterized line curve constraints
- Centered image coordinates (subtract principal point)
- Radial distortion: distortion = 1 + k1 * r² + k2 * r⁴

Extended from BroadTrack's original k1-only model to support k1+k2
for wide-angle cameras (e.g., Veo ~120° FOV).

Reference: BroadTrack/Residuals.h, BroadTrack/CameraTracker.cpp
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from .curve_utils import (
    angle_axis_to_rotation_matrix,
    broadtrack_project,
    compute_cumulated_lengths,
    find_closest_arc_param,
    interpolate_on_polyline,
    ray_cast_to_ground,
    sample_points_on_image_line,
)

logger = logging.getLogger(__name__)

# Camera parameter indices in optimization vector
_AA = slice(0, 3)    # angle-axis rotation
_POS = slice(3, 6)   # camera position
_F = 6               # focal length
_K1 = 7              # first radial distortion
_K2 = 8              # second radial distortion
_N_CAM = 9           # total camera parameters


class BroadTrackCalibrator:
    """Camera calibrator using BroadTrack's optimization approach.

    Uses 9 camera parameters (angleAxis, position, f, k1, k2) +
    per-line-point arc-length parameters, optimized jointly with Cauchy loss.
    """

    # Ground-only line IDs (exclude crossbars and goal posts)
    GROUND_LINE_IDS = set(range(23)) - {6, 7, 8, 9, 10, 11}

    # Keypoints that lie on each ground line (ordered by arc-length parameter t).
    # Derived from pitch geometry: keypoint world coords matched to line segments.
    # Non-ground keypoints (12,14,16,18 = goal post tops) are excluded.
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

    # Distortion priors for wide-angle camera initialization
    # (matching goal-insight's iterative PnP approach)
    DISTORTION_PRIORS = [
        None,                          # no prior
        np.array([-0.15, 0.01]),       # mild barrel
        np.array([-0.30, 0.05]),       # strong barrel
    ]

    def __init__(
        self,
        image_size: tuple[int, int],
        cauchy_f_scale: float = 5.0,
        line_sample_points: int = 20,
        line_weight: float = 1.0,
        focal_candidates: list[float] | None = None,
        max_nfev: int = 500,
        ransac_threshold: float = 30.0,
    ):
        self.width, self.height = image_size
        self.cx = self.width / 2.0
        self.cy = self.height / 2.0
        self.cauchy_f_scale = cauchy_f_scale
        self.line_sample_points = line_sample_points
        self.line_weight = line_weight
        self.focal_candidates = focal_candidates or [600, 800, 1000, 1300, 1600, 2000]
        self.max_nfev = max_nfev
        self.ransac_threshold = ransac_threshold

        self._keypoints = []
        self._lines = []

    def update(self, keypoints: list[dict], lines: list[dict]):
        """Update with detected keypoints and lines for current frame."""
        self._keypoints = keypoints
        self._lines = lines

    def calibrate(
        self,
        keypoint_mapper,
        line_mapper=None,
        min_confidence: float = 0.3,
    ) -> dict[str, Any] | None:
        """Run BroadTrack-style calibration on current frame data.

        Args:
            keypoint_mapper: KeypointMapper for 3D correspondences.
            line_mapper: LineMapper for line world coords. If None, lines are
                derived from detected keypoints (no line model needed).
            min_confidence: Minimum keypoint confidence threshold.
        """
        # Step 1: Prepare keypoint correspondences
        kp_img_centered, kp_world_3d, kp_img_original, kp_ids = self._prepare_keypoints(
            keypoint_mapper, min_confidence
        )
        if len(kp_world_3d) < 6:
            return None

        # Step 2: Prepare line curve constraints
        # Prefer derived lines from keypoints (no extra model needed).
        # Fall back to line detector if line_mapper is provided and lines exist.
        line_constraints = self._derive_lines_from_keypoints(
            kp_ids, kp_img_original, keypoint_mapper, line_mapper,
        )
        if not line_constraints and line_mapper is not None:
            line_constraints = self._prepare_line_constraints(line_mapper, min_confidence)

        # Step 3: PnP initialization with multi-distortion-prior sweep
        init_result = self._initialize_pnp(kp_world_3d, kp_img_original)
        if init_result is None:
            return None

        angle_axis_init, position_init, f_init, k1_init, k2_init = init_result

        # Step 4: Initialize arc-length parameters via ray casting
        arc_params_init = self._init_arc_params(
            angle_axis_init, position_init, f_init, k1_init, line_constraints
        )

        # Step 5: Joint optimization
        opt_result = self._optimize(
            kp_img_centered, kp_world_3d,
            line_constraints, arc_params_init,
            angle_axis_init, position_init, f_init, k1_init, k2_init,
        )
        if opt_result is None:
            return None

        # Step 6: Extract and format results
        return self._format_result(
            opt_result, kp_img_original, kp_world_3d, kp_ids, line_constraints
        )

    def _prepare_keypoints(self, keypoint_mapper, min_confidence):
        """Extract 3D-2D keypoint correspondences."""
        img_pts, world_pts, kp_ids = keypoint_mapper.build_3d_correspondence_matrix(
            self._keypoints,
            filter_by_confidence=min_confidence,
            exclude_non_ground=False,
        )

        if len(img_pts) == 0:
            return np.zeros((0, 2)), np.zeros((0, 3)), np.zeros((0, 2)), []

        img_original = img_pts.astype(np.float64)
        world_3d = world_pts.astype(np.float64)

        img_centered = img_original.copy()
        img_centered[:, 0] -= self.cx
        img_centered[:, 1] -= self.cy

        return img_centered, world_3d, img_original, kp_ids

    def _derive_lines_from_keypoints(
        self, kp_ids, kp_img_original, keypoint_mapper, line_mapper,
    ):
        """Derive line constraints from detected keypoints (no line model needed).

        For each ground line, if >=2 of its keypoints are detected, connect the
        two most extreme ones in image space and sample points along that segment.
        The 3D polyline uses the line's world-coordinate endpoints.

        Args:
            kp_ids: List of detected keypoint IDs.
            kp_img_original: (N, 2) image coordinates of detected keypoints.
            keypoint_mapper: KeypointMapper instance.
            line_mapper: LineMapper for world line endpoints (or None to use
                LINE_DEFINITIONS directly).

        Returns:
            List of line constraint dicts (same format as _prepare_line_constraints).
        """
        from .line_mapping import LineMapper

        detected = {}  # kp_id -> image coords
        for i, kid in enumerate(kp_ids):
            detected[kid] = kp_img_original[i]

        constraints = []
        non_ground = keypoint_mapper.NON_GROUND_KEYPOINTS

        for line_id, kp_list in self.LINE_KEYPOINTS.items():
            # Find detected keypoints on this line (skip non-ground)
            found = [(kid, detected[kid]) for kid in kp_list
                     if kid in detected and kid not in non_ground]
            if len(found) < 2:
                continue

            # Get world line endpoints
            if line_mapper is not None:
                world_line = line_mapper.get_line_world_coords(line_id)
            else:
                ld = LineMapper.LINE_DEFINITIONS[line_id]
                world_line = {
                    "x1": ld["p1"][0], "y1": ld["p1"][1], "z1": ld["p1"][2],
                    "x2": ld["p2"][0], "y2": ld["p2"][1], "z2": ld["p2"][2],
                }
            if world_line is None:
                continue

            p1_w = np.array([world_line["x1"], world_line["y1"], world_line["z1"]])
            p2_w = np.array([world_line["x2"], world_line["y2"], world_line["z2"]])
            polyline_3d = np.array([p1_w, p2_w], dtype=np.float64)
            cum_lengths = compute_cumulated_lengths(polyline_3d)

            # Use the two most extreme keypoints (first and last in ordered list)
            # kp_list is already ordered by arc-length parameter
            first_img = found[0][1]
            last_img = found[-1][1]

            # Sample points along the image line between these two keypoints
            img_pts = sample_points_on_image_line(
                float(first_img[0]), float(first_img[1]),
                float(last_img[0]), float(last_img[1]),
                n_points=self.line_sample_points,
            )
            img_pts_centered = img_pts.copy()
            img_pts_centered[:, 0] -= self.cx
            img_pts_centered[:, 1] -= self.cy

            constraints.append({
                "polyline_3d": polyline_3d,
                "cumulated_lengths": cum_lengths,
                "img_pts_centered": img_pts_centered,
                "line_id": line_id,
                "derived_from_keypoints": [f[0] for f in found],
            })

        return constraints

    def _prepare_line_constraints(self, line_mapper, min_confidence):
        """Prepare line curve constraints from detected lines."""
        constraints = []

        for line in self._lines:
            if line.get("confidence", 1.0) < min_confidence:
                continue

            line_id = line.get("id", -1)
            if line_id < 0 or line_id not in self.GROUND_LINE_IDS:
                continue

            world_line = line_mapper.get_line_world_coords(line_id)
            if world_line is None:
                continue

            p1 = np.array([world_line["x1"], world_line["y1"], world_line["z1"]])
            p2 = np.array([world_line["x2"], world_line["y2"], world_line["z2"]])

            polyline_3d = np.array([p1, p2], dtype=np.float64)
            cum_lengths = compute_cumulated_lengths(polyline_3d)

            img_pts = sample_points_on_image_line(
                line["x1"], line["y1"], line["x2"], line["y2"],
                n_points=self.line_sample_points,
            )
            img_pts_centered = img_pts.copy()
            img_pts_centered[:, 0] -= self.cx
            img_pts_centered[:, 1] -= self.cy

            constraints.append({
                "polyline_3d": polyline_3d,
                "cumulated_lengths": cum_lengths,
                "img_pts_centered": img_pts_centered,
                "line_id": line_id,
            })

        return constraints

    def _initialize_pnp(self, world_pts_3d, img_pts_original):
        """Initialize camera using PnP RANSAC with focal + distortion prior sweep.

        Sweeps focal candidates × distortion priors to find the best initialization.

        Returns:
            (angle_axis, position, f, k1, k2) or None.
        """
        best_error = float("inf")
        best_result = None

        for dist_prior in self.DISTORTION_PRIORS:
            # If we have a distortion prior, undistort points first
            if dist_prior is not None:
                k1_prior, k2_prior = dist_prior
            else:
                k1_prior, k2_prior = 0.0, 0.0

            for f_cand in self.focal_candidates:
                K = np.array([
                    [f_cand, 0, self.cx],
                    [0, f_cand, self.cy],
                    [0, 0, 1],
                ], dtype=np.float64)

                # Undistort image points with the distortion prior
                if dist_prior is not None:
                    dist_coeffs = np.array([k1_prior, k2_prior, 0, 0, 0], dtype=np.float64)
                    pts_for_pnp = cv2.undistortPoints(
                        img_pts_original.reshape(-1, 1, 2).astype(np.float64),
                        K, dist_coeffs, P=K,
                    ).reshape(-1, 2)
                else:
                    pts_for_pnp = img_pts_original
                    dist_coeffs = None

                ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                    world_pts_3d.astype(np.float64),
                    pts_for_pnp.astype(np.float64),
                    K, None,
                    reprojectionError=self.ransac_threshold,
                    iterationsCount=2000,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )

                if not ok or inliers is None or len(inliers) < 6:
                    continue

                # Evaluate reprojection error on original (distorted) points
                from ...utils.projection import project_points_2d
                proj = project_points_2d(
                    world_pts_3d, rvec, tvec, K, dist_coeffs,
                )
                errors = np.linalg.norm(proj - img_pts_original, axis=1)
                median_err = float(np.median(errors))

                if median_err < best_error:
                    best_error = median_err
                    R, _ = cv2.Rodrigues(rvec)
                    position = (-R.T @ tvec).flatten()
                    best_result = (
                        rvec.flatten(), position, f_cand, k1_prior, k2_prior
                    )

        return best_result

    def _init_arc_params(self, angle_axis, position, f, k1, line_constraints):
        """Initialize arc-length parameters via ray casting to ground plane."""
        arc_params_list = []

        for lc in line_constraints:
            polyline = lc["polyline_3d"]
            cum_len = lc["cumulated_lengths"]
            img_pts = lc["img_pts_centered"]
            total_len = cum_len[-1]

            params = []
            for pt in img_pts:
                world_pt = ray_cast_to_ground(pt, f, k1, angle_axis, position)
                if world_pt is not None:
                    t = find_closest_arc_param(world_pt, polyline, cum_len)
                else:
                    t = total_len / 2.0
                params.append(t)

            arc_params_list.append(np.array(params, dtype=np.float64))

        return arc_params_list

    def _optimize(
        self,
        kp_img_centered,
        kp_world_3d,
        line_constraints,
        arc_params_init,
        angle_axis_init,
        position_init,
        f_init,
        k1_init,
        k2_init,
    ):
        """Run joint optimization with Cauchy loss.

        Optimizes: [angleAxis(3), position(3), f, k1, k2, t1, t2, ..., tN]
        """
        n_line_pts = sum(len(lc["img_pts_centered"]) for lc in line_constraints)

        # Parameter vector: [aa(3), pos(3), f, k1, k2, arc_params...]
        x0 = np.concatenate([
            angle_axis_init.flatten(),
            position_init.flatten(),
            [f_init, k1_init, k2_init],
            *arc_params_init,
        ])

        # Bounds
        lower = np.full(len(x0), -np.inf)
        upper = np.full(len(x0), np.inf)
        lower[_F] = 100.0;    upper[_F] = 5000.0
        lower[_K1] = -0.5;    upper[_K1] = 0.5
        lower[_K2] = -0.3;    upper[_K2] = 0.3

        # Arc-length bounds
        idx = _N_CAM
        for lc in line_constraints:
            total_len = lc["cumulated_lengths"][-1]
            n_pts = len(lc["img_pts_centered"])
            lower[idx:idx + n_pts] = 0.0
            upper[idx:idx + n_pts] = total_len
            idx += n_pts

        # Precompute line data for residual function
        line_data = [
            {
                "polyline_3d": lc["polyline_3d"],
                "cumulated_lengths": lc["cumulated_lengths"],
                "img_pts_centered": lc["img_pts_centered"],
                "n_pts": len(lc["img_pts_centered"]),
            }
            for lc in line_constraints
        ]
        lw = self.line_weight

        def residuals(x):
            aa = x[_AA]
            pos = x[_POS]
            f = x[_F]
            k1 = x[_K1]
            k2 = x[_K2]

            all_res = []

            # Keypoint residuals
            if len(kp_world_3d) > 0:
                proj = broadtrack_project(aa, pos, f, k1, kp_world_3d, k2=k2)
                all_res.append((proj - kp_img_centered).ravel())

            # Line curve residuals
            arc_idx = _N_CAM
            for ld in line_data:
                polyline = ld["polyline_3d"]
                cum_len = ld["cumulated_lengths"]
                img_pts = ld["img_pts_centered"]
                n = ld["n_pts"]

                for j in range(n):
                    t = x[arc_idx + j]
                    pt_3d = interpolate_on_polyline(polyline, cum_len, t)
                    proj = broadtrack_project(aa, pos, f, k1, pt_3d.reshape(1, 3), k2=k2)[0]
                    all_res.append((proj - img_pts[j]) * lw)

                arc_idx += n

            return np.concatenate(all_res) if all_res else np.array([])

        try:
            opt = least_squares(
                residuals, x0,
                method="trf",
                loss="cauchy",
                f_scale=self.cauchy_f_scale,
                bounds=(lower, upper),
                max_nfev=self.max_nfev,
            )
        except Exception as e:
            logger.error("Optimization failed: %s", e)
            return None

        return {
            "angle_axis": opt.x[_AA],
            "position": opt.x[_POS],
            "f": opt.x[_F],
            "k1": opt.x[_K1],
            "k2": opt.x[_K2],
            "arc_params": opt.x[_N_CAM:],
            "cost": opt.cost,
        }

    def _format_result(self, opt_result, kp_img_original, kp_world_3d, kp_ids, line_constraints):
        """Convert optimization result to stage1-compatible output format."""
        aa = opt_result["angle_axis"]
        pos = opt_result["position"]
        f = opt_result["f"]
        k1 = opt_result["k1"]
        k2 = opt_result["k2"]

        R = angle_axis_to_rotation_matrix(aa)
        rvec = aa.reshape(3, 1)
        tvec = (-R @ pos).reshape(3, 1)

        K = np.array([
            [f, 0, self.cx],
            [0, f, self.cy],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)

        # Homography for compatibility
        H = K @ np.column_stack([R[:, 0], R[:, 1], tvec.flatten()])
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]

        # Reprojection error using BroadTrack projection
        kp_img_centered = kp_img_original.copy()
        kp_img_centered[:, 0] -= self.cx
        kp_img_centered[:, 1] -= self.cy

        projected = broadtrack_project(aa, pos, f, k1, kp_world_3d, k2=k2)
        errors = np.linalg.norm(projected - kp_img_centered, axis=1)
        inlier_mask = errors < self.ransac_threshold
        median_error = float(np.median(errors)) if len(errors) > 0 else float("inf")

        n_line_pts = sum(len(lc["img_pts_centered"]) for lc in line_constraints)

        camera_params = {
            "K": K,
            "rvec": rvec,
            "tvec": tvec,
            "R": R,
            "dist_coeffs": dist_coeffs,
            "focal_length": f,
        }

        # Detailed per-keypoint information
        projected_original = projected.copy()
        projected_original[:, 0] += self.cx
        projected_original[:, 1] += self.cy

        keypoint_details = []
        for i in range(len(kp_world_3d)):
            keypoint_details.append({
                "kp_id": int(kp_ids[i]) if i < len(kp_ids) else -1,
                "detected": [float(kp_img_original[i, 0]), float(kp_img_original[i, 1])],
                "projected": [float(projected_original[i, 0]), float(projected_original[i, 1])],
                "world_3d": [float(kp_world_3d[i, 0]), float(kp_world_3d[i, 1]), float(kp_world_3d[i, 2])],
                "error_px": float(errors[i]),
                "is_inlier": bool(inlier_mask[i]),
            })

        # Detailed per-line information
        arc_params = opt_result["arc_params"]
        line_details = []
        arc_idx = 0
        for lc in line_constraints:
            n_pts = len(lc["img_pts_centered"])
            line_img_centered = lc["img_pts_centered"]
            line_arc = arc_params[arc_idx:arc_idx + n_pts]

            # Project optimized 3D line points back to image
            line_projected = []
            line_world_pts = []
            line_errors = []
            for j in range(n_pts):
                pt_3d = interpolate_on_polyline(
                    lc["polyline_3d"], lc["cumulated_lengths"], line_arc[j]
                )
                proj_centered = broadtrack_project(aa, pos, f, k1, pt_3d.reshape(1, 3), k2=k2)[0]
                proj_original = [float(proj_centered[0] + self.cx), float(proj_centered[1] + self.cy)]
                detected_original = [
                    float(line_img_centered[j, 0] + self.cx),
                    float(line_img_centered[j, 1] + self.cy),
                ]
                err = float(np.linalg.norm(proj_centered - line_img_centered[j]))
                line_projected.append(proj_original)
                line_world_pts.append([float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2])])
                line_errors.append(err)

            polyline_endpoints = lc["polyline_3d"].tolist()
            line_details.append({
                "line_id": int(lc["line_id"]),
                "polyline_endpoints": polyline_endpoints,
                "num_sample_points": n_pts,
                "sampled_image_pts": (line_img_centered + np.array([[self.cx, self.cy]])).tolist(),
                "projected_pts": line_projected,
                "world_pts": line_world_pts,
                "errors_px": line_errors,
                "mean_error_px": float(np.mean(line_errors)) if line_errors else 0.0,
            })
            arc_idx += n_pts

        return {
            "homography": H,
            "final_error": median_error,
            "num_keypoints": len(kp_world_3d),
            "num_lines": len(line_constraints),
            "num_intersections": 0,
            "total_points": len(kp_world_3d) + n_line_pts,
            "inliers": int(np.sum(inlier_mask)),
            "camera_params": camera_params,
            "img_pts": kp_img_original,
            "inlier_mask": inlier_mask,
            "keypoint_details": keypoint_details,
            "line_details": line_details,
        }
