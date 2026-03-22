"""Frame-by-frame camera calibration following PnLCalib approach.

This module implements the FramebyFrameCalib class that combines:
1. Keypoint detection
2. Line detection
3. Line intersection computation for additional keypoints
4. Heuristic voting for robust calibration
5. Line-based refinement optimization

Reference:
    Gutierrez-Perez & Agudo, "PnLCalib: Single-View Camera-Field Calibration
    using Points and Lines", arXiv 2024.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

from .camera import Camera, is_good_camera

logger = logging.getLogger(__name__)


# Standard FIFA pitch dimensions (in meters)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_L = PITCH_LENGTH / 2  # 52.5
HALF_W = PITCH_WIDTH / 2   # 34.0


class FramebyFrameCalib:
    """Frame-by-frame camera calibration using Points and Lines.

    This class implements the PnLCalib approach:
    1. Detect keypoints and lines
    2. Compute line intersections as additional keypoints
    3. Use heuristic voting to find best calibration
    4. Refine using line constraints

    Attributes:
        alpha: Weight for line error in optimization (default 0.7).
        image_size: Image dimensions (width, height).
    """

    # 57 keypoint world coordinates (ground plane, z=0)
    # Format: [[x, y], ...] in meters, origin at top-left corner of pitch
    # PnLCalib uses origin at corner, we use center - convert as needed
    KEYPOINT_WORLD_COORDS_2D = None  # Will be loaded from KeypointMapper

    def __init__(
        self,
        image_size: tuple[int, int] = (960, 540),
        alpha: float = 0.7,
    ):
        """Initialize calibrator.

        Args:
            image_size: Image dimensions (width, height).
            alpha: Weight for line error vs point error in optimization.
                   Higher alpha = more weight on line constraints.
        """
        self.image_size = image_size
        self.alpha = alpha
        self.w, self.h = image_size

        # Current frame data
        self.keypoints = []
        self.lines = []
        self.line_intersections = []

        # Calibration results
        self.homography = None
        self.camera_params = None

    def update(
        self,
        keypoints: list[dict[str, Any]],
        lines: list[dict[str, Any]],
    ) -> None:
        """Update with new detections.

        Args:
            keypoints: Detected keypoints with id, x, y, confidence.
            lines: Detected lines with id, x1, y1, x2, y2, confidence.
        """
        self.keypoints = keypoints
        self.lines = lines
        self.line_intersections = self._compute_line_intersections(lines)

    def _compute_line_intersections(
        self,
        lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compute intersections between detected lines.

        This augments the keypoint set with line intersection points,
        providing more constraints for calibration.

        Args:
            lines: Detected lines.

        Returns:
            List of intersection points with world coordinates.
        """
        intersections = []

        if len(lines) < 2:
            return intersections

        # Define which line pairs should intersect (based on field geometry)
        # Format: {(line_id1, line_id2): (world_x, world_y)}
        intersection_pairs = self._get_valid_intersection_pairs()

        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i + 1:], start=i + 1):
                id1, id2 = line1.get("id", -1), line2.get("id", -1)

                # Check if this pair should intersect
                pair_key = (min(id1, id2), max(id1, id2))
                if pair_key not in intersection_pairs:
                    continue

                # Compute intersection point
                pt = self._line_line_intersection(
                    (line1["x1"], line1["y1"], line1["x2"], line1["y2"]),
                    (line2["x1"], line2["y1"], line2["x2"], line2["y2"]),
                )

                if pt is None:
                    continue

                # Check if intersection is within image bounds (with margin)
                margin = 50
                if not (-margin < pt[0] < self.w + margin and
                        -margin < pt[1] < self.h + margin):
                    continue

                world_coords = intersection_pairs[pair_key]
                conf = min(line1.get("confidence", 1.0), line2.get("confidence", 1.0))

                intersections.append({
                    "x": pt[0],
                    "y": pt[1],
                    "world_x": world_coords[0],
                    "world_y": world_coords[1],
                    "world_z": 0.0,  # Ground plane
                    "confidence": conf,
                    "source": f"line_{id1}_x_line_{id2}",
                })

        return intersections

    def _get_valid_intersection_pairs(self) -> dict[tuple[int, int], tuple[float, float]]:
        """Get valid line intersection pairs and their world coordinates.

        Returns:
            Dictionary mapping (line_id1, line_id2) to (world_x, world_y).
        """
        # Based on PnLCalib line definitions:
        # 0-2: Left penalty area, 3-5: Right penalty area
        # 12: Center line, 13: Top touchline, 14: Left goal line
        # 15: Right goal line, 16: Bottom touchline
        # 17-19: Left goal area, 20-22: Right goal area

        pairs = {}

        # Penalty area corners (left)
        # Line 0 (Big rect left top) x Line 1 (Big rect left side)
        pairs[(0, 1)] = (-HALF_L + 16.5, 20.16)  # Top-left inner corner
        # Line 1 x Line 2 (Big rect left bottom)
        pairs[(1, 2)] = (-HALF_L + 16.5, -20.16)  # Bottom-left inner corner
        # Line 0 x Line 14 (Left goal line)
        pairs[(0, 14)] = (-HALF_L, 20.16)  # Top-left outer corner
        # Line 2 x Line 14
        pairs[(2, 14)] = (-HALF_L, -20.16)  # Bottom-left outer corner

        # Penalty area corners (right)
        pairs[(3, 4)] = (HALF_L - 16.5, 20.16)
        pairs[(4, 5)] = (HALF_L - 16.5, -20.16)
        pairs[(3, 15)] = (HALF_L, 20.16)
        pairs[(5, 15)] = (HALF_L, -20.16)

        # Center line intersections
        # Line 12 (Center) x Line 13 (Top touchline)
        pairs[(12, 13)] = (0, HALF_W)
        # Line 12 x Line 16 (Bottom touchline)
        pairs[(12, 16)] = (0, -HALF_W)

        # Pitch corners
        # Line 13 (Top) x Line 14 (Left goal line)
        pairs[(13, 14)] = (-HALF_L, HALF_W)
        # Line 13 x Line 15 (Right goal line)
        pairs[(13, 15)] = (HALF_L, HALF_W)
        # Line 14 x Line 16
        pairs[(14, 16)] = (-HALF_L, -HALF_W)
        # Line 15 x Line 16
        pairs[(15, 16)] = (HALF_L, -HALF_W)

        # Goal area corners (left)
        pairs[(17, 18)] = (-HALF_L + 5.5, 9.16)
        pairs[(18, 19)] = (-HALF_L + 5.5, -9.16)
        pairs[(17, 14)] = (-HALF_L, 9.16)
        pairs[(19, 14)] = (-HALF_L, -9.16)

        # Goal area corners (right)
        pairs[(20, 21)] = (HALF_L - 5.5, 9.16)
        pairs[(21, 22)] = (HALF_L - 5.5, -9.16)
        pairs[(20, 15)] = (HALF_L, 9.16)
        pairs[(22, 15)] = (HALF_L, -9.16)

        return pairs

    def _line_line_intersection(
        self,
        line1: tuple[float, float, float, float],
        line2: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """Compute intersection of two lines.

        Args:
            line1: (x1, y1, x2, y2) first line.
            line2: (x1, y1, x2, y2) second line.

        Returns:
            Intersection point (x, y) or None if parallel.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def get_all_correspondences(
        self,
        keypoint_mapper,
        min_confidence: float = 0.3,
        exclude_non_ground: bool = True,
        exclude_edge: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get all 2D-3D correspondences including line intersections.

        Args:
            keypoint_mapper: KeypointMapper instance for keypoint mapping.
            min_confidence: Minimum confidence threshold.
            exclude_non_ground: Whether to exclude non-ground keypoints.
            exclude_edge: Whether to exclude edge keypoints (lens distortion).

        Returns:
            Tuple of (image_points, world_points).
        """
        image_points = []
        world_points = []

        # Add detected keypoints
        kp_img, kp_world, _ = keypoint_mapper.build_correspondence_matrix(
            self.keypoints,
            filter_by_confidence=min_confidence,
            exclude_non_ground=exclude_non_ground,
            exclude_edge=exclude_edge,
        )

        if len(kp_img) > 0:
            image_points.extend(kp_img.tolist())
            world_points.extend(kp_world[:, :2].tolist())  # x, y only

        # Add line intersections
        for inter in self.line_intersections:
            if inter.get("confidence", 0) >= min_confidence:
                image_points.append([inter["x"], inter["y"]])
                world_points.append([inter["world_x"], inter["world_y"]])

        if not image_points:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        return np.array(image_points), np.array(world_points)

    def _filter_inconsistent_sides(
        self,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter keypoints from physically impossible configurations.

        A camera cannot see both ends of a 105m pitch simultaneously.
        If we detect keypoints from both sides, the minority side is likely
        misidentified.

        Args:
            img_pts: Image points (N, 2).
            world_pts: World points (N, 2).

        Returns:
            Filtered (img_pts, world_pts) tuple.
        """
        if len(img_pts) < 6:
            return img_pts, world_pts

        world_x = world_pts[:, 0]
        left_mask = world_x < -20   # Left half of pitch (x < -20m)
        right_mask = world_x > 20   # Right half of pitch (x > 20m)

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        # If one side dominates significantly, filter the minority side.
        # Only filter if:
        # 1. Both sides have detections (to detect the conflict)
        # 2. Minority side has very few points (<=2) - likely misdetection
        # 3. Majority side has 3x more points
        # This avoids filtering when camera can legitimately see both penalty areas.
        if n_left > 0 and n_right > 0:
            if n_right <= 2 and n_left > n_right * 3:  # Left dominates, right is sparse
                keep = ~right_mask
                return img_pts[keep], world_pts[keep]
            elif n_left <= 2 and n_right > n_left * 3:  # Right dominates, left is sparse
                keep = ~left_mask
                return img_pts[keep], world_pts[keep]

        return img_pts, world_pts

    def _filter_spatial_outliers(
        self,
        img_pts: np.ndarray,
        world_pts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter points that violate spatial consistency.

        Image x and world x should have a monotonic relationship (either
        both increase together or both decrease). Points that strongly
        violate this are likely misidentified.

        Args:
            img_pts: Image points (N, 2).
            world_pts: World points (N, 2).

        Returns:
            Filtered (img_pts, world_pts) tuple.
        """
        if len(img_pts) < 6:
            return img_pts, world_pts

        img_x = img_pts[:, 0]
        world_x = world_pts[:, 0]

        # Compute correlation between image x and world x
        # Positive correlation: world x increases as image x increases
        # Negative correlation: world x decreases as image x increases
        correlation = np.corrcoef(img_x, world_x)[0, 1]

        if abs(correlation) < 0.3:
            # Weak correlation - data is too noisy, skip filtering
            return img_pts, world_pts

        # For each point, check if it's consistent with the overall trend
        # Use rank-based check: if image x rank and world x rank differ too much
        img_rank = np.argsort(np.argsort(img_x))
        world_rank = np.argsort(np.argsort(world_x))

        if correlation < 0:
            # Negative correlation: flip world rank
            world_rank = len(world_rank) - 1 - world_rank

        # Points where rank difference is too large are suspicious
        rank_diff = np.abs(img_rank - world_rank)
        threshold = max(3, len(img_pts) * 0.4)  # Allow some tolerance

        # Only filter if there are clear outliers (a few points with large rank diff)
        outlier_mask = rank_diff > threshold
        n_outliers = np.sum(outlier_mask)

        # Filter only if:
        # 1. Few outliers (1-3 points)
        # 2. Most points are consistent
        if 0 < n_outliers <= 3 and n_outliers < len(img_pts) * 0.3:
            keep = ~outlier_mask
            logger.debug(f"Filtered {n_outliers} spatial outliers (rank diff > {threshold:.0f})")
            return img_pts[keep], world_pts[keep]

        return img_pts, world_pts

    def _validate_homography(self, H: np.ndarray) -> bool:
        """Validate homography geometric plausibility.

        Projects pitch center and nearby points to verify the homography
        produces sensible results. Avoids projecting full pitch corners
        since sideline cameras only see part of the pitch.

        Args:
            H: Homography matrix (world -> image).

        Returns:
            True if homography is geometrically plausible.
        """
        # Project a grid of pitch points and check basic sanity
        test_points = np.array([
            [0.0, 0.0, 1.0],          # Center
            [-HALF_L, 0.0, 1.0],      # Left goal line center
            [HALF_L, 0.0, 1.0],       # Right goal line center
            [0.0, HALF_W, 1.0],       # Top touchline center
            [0.0, -HALF_W, 1.0],      # Bottom touchline center
        ])

        valid_projections = 0
        for pt in test_points:
            p = H @ pt
            if abs(p[2]) < 1e-6:
                continue
            px, py = p[0] / p[2], p[1] / p[2]
            # Check if within extended bounds (3x image size)
            if (-2 * self.w < px < 3 * self.w and
                    -2 * self.h < py < 3 * self.h):
                valid_projections += 1

        if valid_projections < 2:
            logger.debug(f"Homography validation failed: only {valid_projections}/5 test points project near image")
            return False

        # Check that the homography preserves local orientation:
        # Moving right in world-x should move roughly consistently in image
        # (not flip back and forth). Test with 3 points along the center line.
        check_pts = np.array([[-10, 0, 1.0], [0, 0, 1.0], [10, 0, 1.0]])
        proj = []
        for pt in check_pts:
            p = H @ pt
            if abs(p[2]) < 1e-6:
                return False
            proj.append(p[:2] / p[2])

        # The 3 projected points should be roughly collinear (not folded)
        v1 = proj[1] - proj[0]
        v2 = proj[2] - proj[1]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        if dot < 0:
            # Vectors point in opposite directions — fold detected
            logger.debug("Homography validation failed: projection fold detected")
            return False

        return True

    def _iterative_pnp_calibrate(
        self,
        keypoint_mapper,
        min_confidence: float = 0.3,
        ransac_threshold: float = 10.0,
    ) -> dict[str, Any] | None:
        """Camera calibration using iterative PnP RANSAC + LM optimization.

        Pipeline:
        1. Build 3D correspondences (ground + crossbar points)
        2. Multi-focal PnP RANSAC sweep with distortion priors
        3. LM optimization of [f, rvec, tvec, k1, k2, p1, p2, k3]
        4. Undistort keypoints, re-run PnP RANSAC, repeat until convergence
        5. Best result across all candidates selected by lowest error

        Args:
            keypoint_mapper: KeypointMapper instance.
            min_confidence: Minimum keypoint confidence.
            ransac_threshold: Reprojection error threshold (pixels).

        Returns:
            Calibration result dict, or None on failure.
        """
        img_pts, world_pts_3d, kp_ids = keypoint_mapper.build_3d_correspondence_matrix(
            self.keypoints, filter_by_confidence=min_confidence, exclude_non_ground=True,
        )
        if len(img_pts) < 6:
            logger.warning(f"Not enough 3D points for PnP: {len(img_pts)}")
            return None

        obj_pts = world_pts_3d.astype(np.float64)
        img_pts_2d = img_pts.astype(np.float64)
        cx, cy = self.w / 2.0, self.h / 2.0
        focal_candidates = [600, 800, 1000, 1300, 1600, 2000]

        lm_lower = [100, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                     -0.5, -0.5, -0.01, -0.01, -0.5]
        lm_upper = [20000, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                     0.5, 0.5, 0.01, 0.01, 0.5]

        # Collect best PnP candidate per distortion prior
        dist_priors = [
            None,
            np.array([-0.15, 0.01, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([-0.30, 0.05, 0.0, 0.0, 0.0], dtype=np.float64),
        ]
        pnp_candidates = []
        for dist_prior in dist_priors:
            best_for_prior = None
            for f_try in focal_candidates:
                K_try = np.array([[f_try, 0, cx], [0, f_try, cy], [0, 0, 1]], dtype=np.float64)
                ok, rv, tv, inl = cv2.solvePnPRansac(
                    obj_pts, img_pts_2d, K_try, dist_prior,
                    reprojectionError=ransac_threshold, iterationsCount=2000,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if ok and inl is not None and len(inl) >= 6:
                    if best_for_prior is None or len(inl) > best_for_prior[0]:
                        d = dist_prior if dist_prior is not None else np.zeros(5, dtype=np.float64)
                        best_for_prior = (len(inl), float(f_try), rv, tv, inl, d)
            if best_for_prior is not None:
                pnp_candidates.append(best_for_prior)

        if not pnp_candidates:
            logger.warning("PnP RANSAC failed for all focal/distortion candidates")
            return None

        # Iterative refinement for each candidate, keep overall best
        max_iters = 5
        overall_best_error = float("inf")
        best_result = None

        def _make_residuals(obj_in, img_in):
            def residuals(x):
                f = x[0]
                rv = x[1:4].reshape(3, 1)
                tv = x[4:7].reshape(3, 1)
                dist = np.array(x[7:12], dtype=np.float64)
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
                projected, _ = cv2.projectPoints(obj_in, rv, tv, K, dist)
                return (projected.reshape(-1, 2) - img_in).ravel()
            return residuals

        for _, focal, rvec, tvec, pnp_inliers, dist_init in pnp_candidates:
            cand_best_error = float("inf")

            for iteration in range(max_iters):
                idx = pnp_inliers.ravel()
                obj_inliers = obj_pts[idx]
                img_inliers = img_pts_2d[idx]

                f_opt, rvec_opt, tvec_opt = focal, rvec, tvec
                dist_cur = dist_init.copy()

                if least_squares is not None:
                    x0 = np.array([
                        focal, rvec[0, 0], rvec[1, 0], rvec[2, 0],
                        tvec[0, 0], tvec[1, 0], tvec[2, 0], *dist_init,
                    ])
                    try:
                        opt = least_squares(
                            _make_residuals(obj_inliers, img_inliers), x0,
                            method="trf", bounds=(lm_lower, lm_upper), max_nfev=300,
                        )
                        f_opt = opt.x[0]
                        rvec_opt = opt.x[1:4].reshape(3, 1)
                        tvec_opt = opt.x[4:7].reshape(3, 1)
                        dist_cur = np.array(opt.x[7:12], dtype=np.float64)
                    except Exception:
                        pass

                # Evaluate on ALL points
                K_opt = np.array([[f_opt, 0, cx], [0, f_opt, cy], [0, 0, 1]], dtype=np.float64)
                all_proj, _ = cv2.projectPoints(obj_pts, rvec_opt, tvec_opt, K_opt, dist_cur)
                all_errors = np.linalg.norm(all_proj.reshape(-1, 2) - img_pts_2d, axis=1)
                inlier_mask = all_errors < ransac_threshold
                n_inl = int(np.sum(inlier_mask))
                error = float(np.median(all_errors))

                R_opt, _ = cv2.Rodrigues(rvec_opt)
                H = K_opt @ np.column_stack([R_opt[:, 0], R_opt[:, 1], tvec_opt.flatten()])
                if abs(H[2, 2]) > 1e-10:
                    H = H / H[2, 2]

                if error < overall_best_error:
                    overall_best_error = error
                    best_result = {
                        "f": f_opt, "rvec": rvec_opt.copy(), "tvec": tvec_opt.copy(),
                        "K": K_opt.copy(), "R": R_opt.copy(), "H": H.copy(),
                        "dist": dist_cur.copy(), "n_inliers": n_inl,
                        "error": error, "inlier_mask": inlier_mask.copy(),
                    }

                if error >= cand_best_error:
                    break
                cand_best_error = error

                if iteration == max_iters - 1:
                    break

                # Undistort and re-run PnP for next iteration
                pts_undist = cv2.undistortPoints(
                    img_pts_2d.reshape(-1, 1, 2), K_opt, dist_cur, P=K_opt,
                ).reshape(-1, 2)
                ok2, rv2, tv2, inl2 = cv2.solvePnPRansac(
                    obj_pts, pts_undist, K_opt, None,
                    reprojectionError=ransac_threshold, iterationsCount=2000,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not ok2 or inl2 is None or len(inl2) < 6:
                    break

                focal, rvec, tvec = f_opt, rv2, tv2
                pnp_inliers = inl2
                dist_init = dist_cur

        r = best_result
        camera_params = {
            "K": r["K"], "rvec": r["rvec"], "tvec": r["tvec"],
            "R": r["R"], "dist_coeffs": r["dist"], "focal_length": r["f"],
        }
        return {
            "homography": r["H"], "error": r["error"],
            "num_points": len(obj_pts), "inliers": r["n_inliers"],
            "camera_params": camera_params,
            "img_pts": img_pts_2d, "inlier_mask": r["inlier_mask"],
        }

    def _h_decompose_calibrate(
        self,
        keypoint_mapper,
        min_confidence: float = 0.3,
        ransac_threshold: float = 10.0,
    ) -> dict[str, Any] | None:
        """Camera calibration using findHomography + Decomposition + LM optimization.

        Pipeline:
        1. findHomography RANSAC (ground-only points)
        2. Decompose H into (f, R, t) via orthogonality constraint
        3. Validate pose geometry (upright camera above ground)
        4. LM optimization with soft_l1 loss (13 params incl. cx, cy)

        Args:
            keypoint_mapper: KeypointMapper instance.
            min_confidence: Minimum keypoint confidence.
            ransac_threshold: Reprojection error threshold (pixels).

        Returns:
            Calibration result dict, or None on failure.
        """
        img_pts, world_pts = self.get_all_correspondences(
            keypoint_mapper, min_confidence, exclude_non_ground=True, exclude_edge=False
        )
        if len(img_pts) < 4:
            logger.warning(f"Not enough points for homography: {len(img_pts)}")
            return None

        img_pts, world_pts = self._filter_inconsistent_sides(img_pts, world_pts)
        img_pts, world_pts = self._filter_spatial_outliers(img_pts, world_pts)
        if len(img_pts) < 4:
            return None

        H_init, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, ransac_threshold)
        if H_init is None:
            logger.warning("Initial Homography estimation failed")
            return None

        inlier_mask = mask.ravel() == 1
        if int(np.sum(inlier_mask)) < 6:
            return None

        cx, cy = self.w / 2.0, self.h / 2.0

        # Decompose H: sweep focal lengths, pick most orthogonal R
        best_f = 1000.0
        best_R = None
        best_t = None
        min_ortho_err = float('inf')
        h1, h2 = H_init[:, 0], H_init[:, 1]

        for f_try in range(500, 3000, 100):
            K_try = np.array([[f_try, 0, cx], [0, f_try, cy], [0, 0, 1]], dtype=np.float64)
            K_inv = np.linalg.inv(K_try)
            L1, L2 = K_inv @ h1, K_inv @ h2
            scale = 2.0 / (np.linalg.norm(L1) + np.linalg.norm(L2))
            r1, r2 = L1 * scale, L2 * scale
            ortho_err = abs(np.dot(r1, r2))

            if ortho_err < min_ortho_err:
                min_ortho_err = ortho_err
                best_f = f_try
                r3 = np.cross(r1, r2)
                U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, r3]))
                R_exact = U @ Vt
                if np.linalg.det(R_exact) < 0:
                    Vt[-1, :] *= -1
                    R_exact = U @ Vt
                best_R = R_exact
                best_t = ((K_inv @ H_init[:, 2]) * scale).reshape(3, 1)

        if best_R is None:
            return None

        # Validate pose geometry
        def _check_pose(R_mat, t_vec):
            c = -R_mat.T @ t_vec.flatten()
            up = -R_mat.T[:, 1]
            return up[2] > 0.4 and 0 < c[2] < 100

        if not _check_pose(best_R, best_t):
            R_alt = best_R.copy()
            R_alt[:, 0] *= -1
            R_alt[:, 1] *= -1
            t_alt = -best_t
            if _check_pose(R_alt, t_alt):
                best_R, best_t = R_alt, t_alt
            else:
                return None

        rvec_init, _ = cv2.Rodrigues(best_R)
        tvec_init = best_t
        f_init = best_f

        # Use ground-only 3D points for optimization
        obj_pts = np.hstack([world_pts, np.zeros((len(world_pts), 1))]).astype(np.float64)

        # LM with soft_l1 loss
        if least_squares is not None:
            def make_residuals(x):
                f = x[0]
                rv = x[1:4].reshape(3, 1)
                tv = x[4:7].reshape(3, 1)
                dist = np.array([x[7], x[8], x[11], x[12], 0.0], dtype=np.float64)
                K = np.array([[f, 0, x[9]], [0, f, x[10]], [0, 0, 1]], dtype=np.float64)
                projected, _ = cv2.projectPoints(obj_pts, rv, tv, K, dist)
                return (projected.reshape(-1, 2) - img_pts).ravel()

            x0 = np.array([
                f_init, rvec_init[0, 0], rvec_init[1, 0], rvec_init[2, 0],
                tvec_init[0, 0], tvec_init[1, 0], tvec_init[2, 0],
                0.0, 0.0, cx, cy, 0.0, 0.0,
            ])
            f_min, f_max = max(300, f_init * 0.7), f_init * 1.3
            bounds = (
                [f_min, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                 -0.15, -0.15, cx - 200, cy - 200, -0.02, -0.02],
                [f_max, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                 0.15, 0.15, cx + 200, cy + 200, 0.02, 0.02],
            )
            try:
                opt = least_squares(make_residuals, x0, method="trf", loss="soft_l1",
                                    f_scale=10.0, bounds=bounds, max_nfev=500)
                f_init = opt.x[0]
                rvec_init = opt.x[1:4].reshape(3, 1)
                tvec_init = opt.x[4:7].reshape(3, 1)
                dist_init = np.array([opt.x[7], opt.x[8], opt.x[11], opt.x[12], 0.0], dtype=np.float64)
                cx, cy = opt.x[9], opt.x[10]
            except Exception as e:
                logger.debug(f"LM failed: {e}")
                dist_init = np.zeros(5, dtype=np.float64)

        K_opt = np.array([[f_init, 0, cx], [0, f_init, cy], [0, 0, 1]], dtype=np.float64)
        all_proj, _ = cv2.projectPoints(obj_pts, rvec_init, tvec_init, K_opt, dist_init)
        all_errors = np.linalg.norm(all_proj.reshape(-1, 2) - img_pts, axis=1)
        final_inlier_mask = all_errors < ransac_threshold
        if int(np.sum(final_inlier_mask)) < 6:
            return None

        error = float(np.median(all_errors))
        R_opt, _ = cv2.Rodrigues(rvec_init)
        H = K_opt @ np.column_stack([R_opt[:, 0], R_opt[:, 1], tvec_init.flatten()])
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]

        camera_params = {
            "K": K_opt, "rvec": rvec_init, "tvec": tvec_init,
            "R": R_opt, "dist_coeffs": dist_init, "focal_length": f_init,
        }
        return {
            "homography": H, "error": error,
            "num_points": len(obj_pts), "inliers": int(np.sum(final_inlier_mask)),
            "camera_params": camera_params,
            "img_pts": img_pts, "inlier_mask": final_inlier_mask,
        }

    def heuristic_voting(
        self,
        keypoint_mapper,
        min_confidence: float = 0.3,
        exclude_edge: bool = False,
        ransac_threshold: float = 30.0,
    ) -> dict[str, Any] | None:
        """Estimate homography using RANSAC.

        Args:
            keypoint_mapper: KeypointMapper instance.
            min_confidence: Minimum confidence threshold.
            exclude_edge: Whether to exclude edge keypoints (lens distortion).
            ransac_threshold: RANSAC reprojection threshold in pixels.

        Returns:
            Calibration result or None.
        """
        # Get all correspondences
        img_pts, world_pts = self.get_all_correspondences(
            keypoint_mapper, min_confidence, exclude_non_ground=True, exclude_edge=exclude_edge
        )

        if len(img_pts) < 4:
            logger.warning(f"Not enough points for calibration: {len(img_pts)}")
            return None

        # Pre-filter 1: remove keypoints from both ends of the pitch
        img_pts, world_pts = self._filter_inconsistent_sides(img_pts, world_pts)

        # Pre-filter 2: remove points that violate spatial consistency
        # (image x and world x should have monotonic relationship)
        img_pts, world_pts = self._filter_spatial_outliers(img_pts, world_pts)

        if len(img_pts) < 4:
            logger.warning(f"Not enough points after filtering: {len(img_pts)}")
            return None

        # Use RANSAC - it handles random noise/outliers
        H, mask = cv2.findHomography(world_pts, img_pts, cv2.RANSAC, ransac_threshold)

        if H is None:
            logger.warning("Homography estimation failed")
            return None

        # Compute error
        inlier_count = int(mask.sum()) if mask is not None else len(img_pts)
        if mask is not None and inlier_count >= 4:
            inlier_idx = mask.ravel() == 1
            error = self._compute_reprojection_error(
                img_pts[inlier_idx], world_pts[inlier_idx], H
            )
        else:
            error = self._compute_reprojection_error(img_pts, world_pts, H)

        # Cache filtered points and inlier mask for reuse in calibrate()
        inlier_mask = mask.ravel() == 1 if mask is not None else np.ones(len(img_pts), dtype=bool)
        return {
            "homography": H,
            "error": error,
            "ransac_thresh": ransac_threshold,
            "num_points": len(img_pts),
            "inliers": inlier_count,
            "inlier_mask": inlier_mask,
            "img_pts": img_pts,
            "world_pts": world_pts,
        }

    def _compute_median_reprojection_error(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        H: np.ndarray,
    ) -> float:
        """Compute median reprojection error.

        Args:
            image_points: Detected 2D points.
            world_points: World 2D coordinates.
            H: Homography (world -> image).

        Returns:
            Median reprojection error in pixels.
        """
        errors = []
        for img_pt, world_pt in zip(image_points, world_points):
            world_h = np.array([world_pt[0], world_pt[1], 1.0])
            img_proj = H @ world_h
            if abs(img_proj[2]) > 1e-6:
                img_proj = img_proj[:2] / img_proj[2]
                error = np.linalg.norm(img_pt - img_proj)
                errors.append(error)
        return float(np.median(errors)) if errors else float("inf")

    def _compute_reprojection_error(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        H: np.ndarray,
    ) -> float:
        """Compute mean reprojection error.

        Args:
            image_points: Detected 2D points.
            world_points: World 2D coordinates.
            H: Homography (world -> image).

        Returns:
            Mean reprojection error in pixels.
        """
        errors = []
        for img_pt, world_pt in zip(image_points, world_points):
            world_h = np.array([world_pt[0], world_pt[1], 1.0])
            img_proj = H @ world_h
            if abs(img_proj[2]) > 1e-6:
                img_proj = img_proj[:2] / img_proj[2]
                error = np.linalg.norm(img_pt - img_proj)
                errors.append(error)
        return np.mean(errors) if errors else float("inf")

    def line_optimizer(
        self,
        H_init: np.ndarray,
        keypoint_mapper,
        line_mapper,
        min_confidence: float = 0.3,
        exclude_edge: bool = False,
        cached_pts: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Refine homography using line constraints.

        Minimizes weighted combination of point and line errors:
            error = (1-alpha) * point_error + alpha * line_error

        Args:
            H_init: Initial homography.
            keypoint_mapper: KeypointMapper instance.
            line_mapper: LineMapper instance.
            min_confidence: Minimum confidence.
            exclude_edge: Whether to exclude edge keypoints (lens distortion).
            cached_pts: Optional (img_pts, world_pts, inlier_mask) from
                heuristic_voting to avoid recomputing correspondences.

        Returns:
            Refined homography.
        """
        if least_squares is None:
            logger.warning("scipy not available, skipping line optimization")
            return H_init

        # Reuse cached filtered points if available, otherwise compute and filter
        if cached_pts is not None:
            img_pts, world_pts, inlier_mask = cached_pts
            # Use inlier points for optimization
            if np.sum(inlier_mask) >= 4:
                img_pts = img_pts[inlier_mask]
                world_pts = world_pts[inlier_mask]
        else:
            img_pts, world_pts = self.get_all_correspondences(
                keypoint_mapper, min_confidence, exclude_non_ground=True, exclude_edge=exclude_edge
            )
            img_pts, world_pts = self._filter_inconsistent_sides(img_pts, world_pts)
            img_pts, world_pts = self._filter_spatial_outliers(img_pts, world_pts)

        # Get line correspondences (filter short lines that provide unreliable constraints)
        img_lines, world_lines = line_mapper.build_line_correspondence(
            self.lines,
            filter_by_confidence=min_confidence,
            exclude_non_ground=True,
            min_length=50.0,  # Minimum 50 pixels to be useful
        )

        if len(img_pts) < 4:
            return H_init

        # Flatten H for optimization (8 DOF, H[2,2] = 1)
        h_init = H_init.flatten()[:8]

        def residuals(h):
            H = np.array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]).reshape(3, 3)

            errors = []

            # Point reprojection errors (weight: 1 - alpha)
            point_weight = np.sqrt(1 - self.alpha)
            for img_pt, world_pt in zip(img_pts, world_pts):
                world_h = np.array([world_pt[0], world_pt[1], 1.0])
                img_proj = H @ world_h
                if abs(img_proj[2]) > 1e-6:
                    img_proj = img_proj[:2] / img_proj[2]
                    err = point_weight * (img_pt - img_proj)
                    errors.extend(err)

            # Line errors (weight: alpha)
            line_weight = np.sqrt(self.alpha)
            for img_line, world_line in zip(img_lines, world_lines):
                # Project world line endpoints to image
                p1_world = np.array([world_line["x1"], world_line["y1"], 1.0])
                p2_world = np.array([world_line["x2"], world_line["y2"], 1.0])

                p1_proj = H @ p1_world
                p2_proj = H @ p2_world

                if abs(p1_proj[2]) < 1e-6 or abs(p2_proj[2]) < 1e-6:
                    continue

                p1_proj = p1_proj[:2] / p1_proj[2]
                p2_proj = p2_proj[:2] / p2_proj[2]

                # Detected line endpoints
                det_p1 = np.array([img_line["x1"], img_line["y1"]])
                det_p2 = np.array([img_line["x2"], img_line["y2"]])

                # Point-to-line distance from detected to projected line
                d1 = self._point_to_line_distance(det_p1, p1_proj, p2_proj)
                d2 = self._point_to_line_distance(det_p2, p1_proj, p2_proj)

                errors.append(line_weight * d1)
                errors.append(line_weight * d2)

            return np.array(errors) if errors else np.array([0.0])

        try:
            result = least_squares(
                residuals,
                h_init,
                method="lm",
                max_nfev=100,
            )

            H_refined = np.array([
                result.x[0], result.x[1], result.x[2],
                result.x[3], result.x[4], result.x[5],
                result.x[6], result.x[7], 1.0
            ]).reshape(3, 3)

            # Verify improvement
            init_error = self._compute_reprojection_error(img_pts, world_pts, H_init)
            refined_error = self._compute_reprojection_error(img_pts, world_pts, H_refined)

            # Only accept if error doesn't increase significantly
            if refined_error <= init_error * 1.05:  # Allow at most 5% increase
                logger.debug(f"Line optimization: {init_error:.1f} -> {refined_error:.1f} px")
                return H_refined
            else:
                logger.debug(f"Line optimization rejected (error increased): {init_error:.1f} -> {refined_error:.1f} px")
                return H_init

        except Exception as e:
            logger.warning(f"Line optimization failed: {e}")
            return H_init

    def hough_line_optimizer(
        self,
        H_init: np.ndarray,
        keypoint_mapper,
        hough_constraints: list[dict[str, Any]],
        min_confidence: float = 0.3,
        exclude_edge: bool = False,
        beta: float = 0.15,
    ) -> np.ndarray:
        """Refine homography using Hough line constraints.

        Minimizes weighted combination of point and Hough line errors:
            error = (1-beta) * point_error + beta * hough_line_error

        Args:
            H_init: Initial homography.
            keypoint_mapper: KeypointMapper instance.
            hough_constraints: List of Hough line constraints from HoughLineMatcher.
            min_confidence: Minimum confidence.
            exclude_edge: Whether to exclude edge keypoints.
            beta: Weight for Hough line constraints (0-1). Default 0.15 to
                  prioritize keypoint accuracy while still benefiting from lines.

        Returns:
            Refined homography.
        """
        if least_squares is None:
            logger.warning("scipy not available, skipping Hough optimization")
            return H_init

        if not hough_constraints:
            return H_init

        # Get point correspondences
        img_pts, world_pts = self.get_all_correspondences(
            keypoint_mapper, min_confidence, exclude_non_ground=True, exclude_edge=exclude_edge
        )

        if len(img_pts) < 4:
            return H_init

        # Flatten H for optimization (8 DOF, H[2,2] = 1)
        h_init = H_init.flatten()[:8]

        def residuals(h):
            H = np.array([h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0]).reshape(3, 3)

            errors = []

            # Point reprojection errors (weight: 1 - beta)
            point_weight = np.sqrt(1 - beta)
            for img_pt, world_pt in zip(img_pts, world_pts):
                world_h = np.array([world_pt[0], world_pt[1], 1.0])
                img_proj = H @ world_h
                if abs(img_proj[2]) > 1e-6:
                    img_proj = img_proj[:2] / img_proj[2]
                    err = point_weight * (img_pt - img_proj)
                    errors.extend(err)

            # Hough line alignment errors (weight: beta)
            hough_weight = np.sqrt(beta)
            for constraint in hough_constraints:
                img_line = constraint["image_line"]
                world_line = constraint["world_line"]
                line_weight = constraint.get("weight", 1.0)

                # Project world line endpoints to image
                p1_world = np.array([world_line["x1"], world_line["y1"], 1.0])
                p2_world = np.array([world_line["x2"], world_line["y2"], 1.0])

                p1_proj = H @ p1_world
                p2_proj = H @ p2_world

                if abs(p1_proj[2]) < 1e-6 or abs(p2_proj[2]) < 1e-6:
                    continue

                p1_proj = p1_proj[:2] / p1_proj[2]
                p2_proj = p2_proj[:2] / p2_proj[2]

                # Detected Hough line endpoints
                det_p1 = np.array([img_line["x1"], img_line["y1"]])
                det_p2 = np.array([img_line["x2"], img_line["y2"]])

                # Point-to-line distance from Hough line to projected template line
                d1 = self._point_to_line_distance(det_p1, p1_proj, p2_proj)
                d2 = self._point_to_line_distance(det_p2, p1_proj, p2_proj)

                errors.append(hough_weight * line_weight * d1)
                errors.append(hough_weight * line_weight * d2)

            return np.array(errors) if errors else np.array([0.0])

        try:
            result = least_squares(
                residuals,
                h_init,
                method="lm",
                max_nfev=100,
            )

            H_refined = np.array([
                result.x[0], result.x[1], result.x[2],
                result.x[3], result.x[4], result.x[5],
                result.x[6], result.x[7], 1.0
            ]).reshape(3, 3)

            # Verify that keypoint error doesn't increase too much
            init_error = self._compute_reprojection_error(img_pts, world_pts, H_init)
            refined_error = self._compute_reprojection_error(img_pts, world_pts, H_refined)

            # Only accept if keypoint error stays within 10% of original
            # This ensures Hough constraints help but don't hurt keypoint accuracy
            if refined_error <= init_error * 1.1:
                logger.debug(f"Hough optimization accepted: {init_error:.1f} -> {refined_error:.1f} px")
                return H_refined
            else:
                logger.debug(f"Hough optimization rejected (keypoint error increased): {init_error:.1f} -> {refined_error:.1f} px")
                return H_init

        except Exception as e:
            logger.warning(f"Hough optimization failed: {e}")
            return H_init

    @staticmethod
    def _point_to_line_distance(
        point: np.ndarray,
        line_p1: np.ndarray,
        line_p2: np.ndarray,
    ) -> float:
        """Compute distance from point to line.

        Args:
            point: Point (x, y).
            line_p1: First point on line.
            line_p2: Second point on line.

        Returns:
            Distance from point to line.
        """
        d = line_p2 - line_p1
        d_len = np.linalg.norm(d)

        if d_len < 1e-6:
            return np.linalg.norm(point - line_p1)

        # Cross product formula
        ap = point - line_p1
        cross = abs(ap[0] * d[1] - ap[1] * d[0])

        return cross / d_len

    def refine_with_line_endpoints(
        self,
        H_init: np.ndarray,
        keypoint_mapper,
        endpoint_img_pts: np.ndarray,
        endpoint_world_pts: np.ndarray,
        min_confidence: float = 0.3,
        exclude_edge: bool = False,
    ) -> np.ndarray:
        """Refine homography by adding line endpoints as additional keypoints.

        Args:
            H_init: Initial homography.
            keypoint_mapper: KeypointMapper instance.
            endpoint_img_pts: Image coordinates of line endpoints (N, 2).
            endpoint_world_pts: World coordinates of line endpoints (N, 2).
            min_confidence: Minimum confidence for keypoints.
            exclude_edge: Whether to exclude edge keypoints.

        Returns:
            Refined homography.
        """
        if len(endpoint_img_pts) == 0:
            return H_init

        # Get original keypoint correspondences
        img_pts, world_pts = self.get_all_correspondences(
            keypoint_mapper, min_confidence, exclude_non_ground=True, exclude_edge=exclude_edge
        )

        if len(img_pts) < 4:
            return H_init

        # Combine keypoints with line endpoints
        combined_img = np.vstack([img_pts, endpoint_img_pts])
        combined_world = np.vstack([world_pts, endpoint_world_pts])

        # Re-estimate homography with RANSAC
        H_refined, mask = cv2.findHomography(combined_world, combined_img, cv2.RANSAC, 5.0)

        if H_refined is None:
            return H_init

        # Verify that keypoint error doesn't increase significantly
        init_error = self._compute_reprojection_error(img_pts, world_pts, H_init)
        refined_error = self._compute_reprojection_error(img_pts, world_pts, H_refined)

        if refined_error <= init_error * 1.1:  # Accept if within 10% of original
            logger.debug(
                f"Line endpoint refinement: {init_error:.1f} -> {refined_error:.1f} px "
                f"(+{len(endpoint_img_pts)} endpoints)"
            )
            return H_refined
        else:
            logger.debug(
                f"Line endpoint refinement rejected: {init_error:.1f} -> {refined_error:.1f} px"
            )
            return H_init

    def calibrate(
        self,
        keypoint_mapper,
        line_mapper,
        min_confidence: float = 0.3,
        use_line_refinement: bool = True,
        exclude_edge: bool = False,
        ransac_threshold: float = 10.0,
        method: str = "iterative_pnp",
    ) -> dict[str, Any] | None:
        """Full calibration pipeline using PnP RANSAC.

        Primary path: solvePnPRansac (3D physical model, distortion-tolerant)
        Fallback: findHomography RANSAC (2D, if PnP fails)

        Args:
            keypoint_mapper: KeypointMapper instance.
            line_mapper: LineMapper instance.
            min_confidence: Minimum detection confidence.
            use_line_refinement: Whether to apply line refinement.
            exclude_edge: Whether to exclude edge keypoints.
            ransac_threshold: PnP RANSAC reprojection threshold (pixels).
            method: Calibration method - "iterative_pnp" or "h_decompose".

        Returns:
            Calibration result with homography and metadata.
        """
        # Primary: PnP-based calibration
        if method == "h_decompose":
            result = self._h_decompose_calibrate(
                keypoint_mapper, min_confidence, ransac_threshold,
            )
        else:
            result = self._iterative_pnp_calibrate(
                keypoint_mapper, min_confidence, ransac_threshold,
            )

        if result is not None:
            return {
                "homography": result["homography"],
                "initial_error": result["error"],
                "final_error": result["error"],
                "num_keypoints": len([k for k in self.keypoints if k.get("confidence", 0) >= min_confidence]),
                "num_lines": len(self.lines),
                "num_intersections": len(self.line_intersections),
                "total_points": result["num_points"],
                "inliers": result["inliers"],
                "camera_params": result["camera_params"],
                "img_pts": result["img_pts"],
                "inlier_mask": result["inlier_mask"],
            }

        # Fallback: 2D homography RANSAC
        logger.info("PnP failed, falling back to findHomography RANSAC")
        fb = self.heuristic_voting(
            keypoint_mapper, min_confidence, exclude_edge=exclude_edge,
            ransac_threshold=ransac_threshold,
        )
        if fb is None:
            return None

        return {
            "homography": fb["homography"],
            "initial_error": fb["error"],
            "final_error": fb["error"],
            "num_keypoints": len([k for k in self.keypoints if k.get("confidence", 0) >= min_confidence]),
            "num_lines": len(self.lines),
            "num_intersections": len(self.line_intersections),
            "total_points": fb["num_points"],
            "inliers": fb.get("inliers", fb["num_points"]),
            "camera_params": None,
        }
