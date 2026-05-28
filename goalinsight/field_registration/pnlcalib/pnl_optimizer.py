"""Joint Points + Lines (PnL) optimization for camera calibration.

This module implements the PnL optimization approach from PnLCalib,
which refines camera parameters using both point and line constraints.

Reference:
    Gutierrez-Perez & Agudo, "PnLCalib: Single-View Camera-Field Calibration
    using Points and Lines", arXiv 2024.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from ...utils.projection import project_points_2d

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

logger = logging.getLogger(__name__)


class PnLOptimizer:
    """Joint Points + Lines optimizer for camera calibration.

    This optimizer refines camera pose using both:
    1. Point reprojection errors (2D-3D correspondences)
    2. Line reprojection errors (point-to-line distance in image space)

    The optimization minimizes:
        E = sum_i ||p_i - proj(P_i)||^2 + lambda * sum_j d(l_j, proj(L_j))^2

    where:
        - p_i: detected 2D keypoints
        - P_i: corresponding 3D world points
        - l_j: detected 2D line (from extremities)
        - L_j: 3D world line
        - proj(): camera projection function
        - d(): point-to-line distance
        - lambda: relative weight of line constraints
    """

    def __init__(
        self,
        line_weight: float = 1.0,
        max_iterations: int = 100,
        ftol: float = 1e-6,
        xtol: float = 1e-6,
    ):
        """Initialize PnL optimizer.

        Args:
            line_weight: Relative weight of line errors vs point errors.
            max_iterations: Maximum optimization iterations.
            ftol: Function tolerance for convergence.
            xtol: Parameter tolerance for convergence.
        """
        if least_squares is None:
            raise ImportError("scipy is required for PnL optimization")

        self.line_weight = line_weight
        self.max_iterations = max_iterations
        self.ftol = ftol
        self.xtol = xtol

    def optimize(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_lines: list[dict[str, Any]] | None = None,
        world_lines: list[dict[str, Any]] | None = None,
        camera_matrix: np.ndarray | None = None,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
        image_size: tuple[int, int] = (1920, 1080),
    ) -> dict[str, Any] | None:
        """Perform joint PnL optimization.

        Args:
            image_points: 2D image points (N, 2).
            world_points: 3D world points (N, 3).
            image_lines: Detected lines with x1, y1, x2, y2.
            world_lines: World lines with endpoints in 3D.
            camera_matrix: Camera intrinsic matrix.
            initial_rvec: Initial rotation vector (from PnP).
            initial_tvec: Initial translation vector (from PnP).
            image_size: Image size (width, height).

        Returns:
            Dictionary with optimized camera parameters, or None if failed.
        """
        image_points = np.array(image_points, dtype=np.float64)
        world_points = np.array(world_points, dtype=np.float64)

        # Ensure 3D points
        if world_points.shape[1] == 2:
            world_points = np.hstack([world_points, np.zeros((len(world_points), 1))])

        # Get initial estimate from PnP if not provided
        if initial_rvec is None or initial_tvec is None:
            if camera_matrix is None:
                w, h = image_size
                camera_matrix = np.array([
                    [1500.0, 0, w / 2],
                    [0, 1500.0, h / 2],
                    [0, 0, 1],
                ], dtype=np.float64)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                world_points,
                image_points,
                camera_matrix,
                None,
                iterationsCount=1000,
                reprojectionError=8.0,
            )

            if not success:
                logger.warning("PnP initialization failed")
                return None

            initial_rvec = rvec.flatten()
            initial_tvec = tvec.flatten()
        else:
            initial_rvec = np.array(initial_rvec).flatten()
            initial_tvec = np.array(initial_tvec).flatten()

        if camera_matrix is None:
            w, h = image_size
            camera_matrix = np.array([
                [1500.0, 0, w / 2],
                [0, 1500.0, h / 2],
                [0, 0, 1],
            ], dtype=np.float64)

        # Check if we have line constraints
        use_lines = (
            image_lines is not None
            and world_lines is not None
            and len(image_lines) > 0
            and len(world_lines) > 0
        )

        if not use_lines:
            # Just refine using points
            return self._optimize_points_only(
                image_points,
                world_points,
                camera_matrix,
                initial_rvec,
                initial_tvec,
            )

        # Joint optimization with points and lines
        return self._optimize_pnl(
            image_points,
            world_points,
            image_lines,
            world_lines,
            camera_matrix,
            initial_rvec,
            initial_tvec,
        )

    def _optimize_points_only(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        camera_matrix: np.ndarray,
        initial_rvec: np.ndarray,
        initial_tvec: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize using only point constraints.

        Args:
            image_points: 2D image points.
            world_points: 3D world points.
            camera_matrix: Camera intrinsic matrix.
            initial_rvec: Initial rotation vector.
            initial_tvec: Initial translation vector.

        Returns:
            Optimized camera parameters.
        """
        # Initial parameters: [rvec (3), tvec (3)]
        x0 = np.concatenate([initial_rvec, initial_tvec])

        def residuals(x):
            projected = project_points_2d(
                world_points, x[:3], x[3:6], camera_matrix,
            )
            return (image_points - projected).flatten()

        result = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=self.max_iterations,
            ftol=self.ftol,
            xtol=self.xtol,
        )

        rvec = result.x[:3]
        tvec = result.x[3:6]

        # Compute final reprojection error
        projected = project_points_2d(
            world_points, rvec, tvec, camera_matrix,
        )
        reproj_error = np.mean(np.linalg.norm(image_points - projected, axis=1))

        return {
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "reprojection_error": float(reproj_error),
            "success": result.success,
            "optimization_cost": float(result.cost),
            "num_points": len(image_points),
            "num_lines": 0,
        }

    def _optimize_pnl(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_lines: list[dict[str, Any]],
        world_lines: list[dict[str, Any]],
        camera_matrix: np.ndarray,
        initial_rvec: np.ndarray,
        initial_tvec: np.ndarray,
    ) -> dict[str, Any]:
        """Optimize using both point and line constraints.

        Args:
            image_points: 2D image points.
            world_points: 3D world points.
            image_lines: Detected 2D lines.
            world_lines: 3D world lines.
            camera_matrix: Camera intrinsic matrix.
            initial_rvec: Initial rotation vector.
            initial_tvec: Initial translation vector.

        Returns:
            Optimized camera parameters.
        """
        # Parse world lines into 3D endpoints
        world_line_endpoints = []
        matched_image_lines = []

        for img_line in image_lines:
            line_id = img_line.get("id", -1)
            # Find matching world line
            for world_line in world_lines:
                if world_line.get("id") == line_id:
                    # World line has p1 (x1, y1, z1) and p2 (x2, y2, z2)
                    p1 = np.array([
                        world_line.get("x1", 0),
                        world_line.get("y1", 0),
                        world_line.get("z1", 0),
                    ])
                    p2 = np.array([
                        world_line.get("x2", 0),
                        world_line.get("y2", 0),
                        world_line.get("z2", 0),
                    ])
                    world_line_endpoints.append((p1, p2))
                    matched_image_lines.append(img_line)
                    break

        if not world_line_endpoints:
            # No matching lines, fall back to points only
            logger.info("No matching world lines found, using points only")
            return self._optimize_points_only(
                image_points,
                world_points,
                camera_matrix,
                initial_rvec,
                initial_tvec,
            )

        # Initial parameters
        x0 = np.concatenate([initial_rvec, initial_tvec])

        num_points = len(image_points)
        num_lines = len(world_line_endpoints)
        line_weight = self.line_weight

        def residuals(x):
            rvec = x[:3]
            tvec = x[3:6]

            errors = []

            # Point reprojection errors
            if num_points > 0:
                projected = project_points_2d(
                    world_points, rvec, tvec, camera_matrix,
                )
                point_errors = (image_points - projected).flatten()
                errors.extend(point_errors)

            # Line reprojection errors
            for i, (p1_3d, p2_3d) in enumerate(world_line_endpoints):
                img_line = matched_image_lines[i]

                # Project world line endpoints to image
                line_3d = np.array([p1_3d, p2_3d])
                proj_endpoints = project_points_2d(
                    line_3d, rvec, tvec, camera_matrix,
                )
                proj_p1 = proj_endpoints[0]
                proj_p2 = proj_endpoints[1]

                # Detected line endpoints
                det_p1 = np.array([img_line["x1"], img_line["y1"]])
                det_p2 = np.array([img_line["x2"], img_line["y2"]])

                # Compute point-to-line distances
                # Distance from detected endpoints to projected line
                d1 = self._point_to_line_distance(det_p1, proj_p1, proj_p2)
                d2 = self._point_to_line_distance(det_p2, proj_p1, proj_p2)

                # Also compute reverse (projected to detected line)
                d3 = self._point_to_line_distance(proj_p1, det_p1, det_p2)
                d4 = self._point_to_line_distance(proj_p2, det_p1, det_p2)

                # Use symmetric error
                line_error = np.sqrt(line_weight) * np.array([d1, d2, d3, d4])
                errors.extend(line_error)

            return np.array(errors)

        result = least_squares(
            residuals,
            x0,
            method="lm",
            max_nfev=self.max_iterations,
            ftol=self.ftol,
            xtol=self.xtol,
        )

        rvec = result.x[:3]
        tvec = result.x[3:6]

        # Compute final point reprojection error
        projected = project_points_2d(
            world_points, rvec, tvec, camera_matrix,
        )
        point_reproj_error = np.mean(np.linalg.norm(image_points - projected, axis=1))

        # Compute final line reprojection error
        line_errors = []
        for i, (p1_3d, p2_3d) in enumerate(world_line_endpoints):
            img_line = matched_image_lines[i]
            line_3d = np.array([p1_3d, p2_3d])
            proj_endpoints = project_points_2d(
                line_3d, rvec, tvec, camera_matrix,
            )
            proj_p1 = proj_endpoints[0]
            proj_p2 = proj_endpoints[1]

            det_p1 = np.array([img_line["x1"], img_line["y1"]])
            det_p2 = np.array([img_line["x2"], img_line["y2"]])

            d1 = self._point_to_line_distance(det_p1, proj_p1, proj_p2)
            d2 = self._point_to_line_distance(det_p2, proj_p1, proj_p2)
            line_errors.extend([d1, d2])

        line_reproj_error = np.mean(line_errors) if line_errors else 0.0

        return {
            "rvec": rvec,
            "tvec": tvec,
            "camera_matrix": camera_matrix,
            "reprojection_error": float(point_reproj_error),
            "line_reprojection_error": float(line_reproj_error),
            "success": result.success,
            "optimization_cost": float(result.cost),
            "num_points": num_points,
            "num_lines": num_lines,
        }

    @staticmethod
    def _point_to_line_distance(
        point: np.ndarray,
        line_p1: np.ndarray,
        line_p2: np.ndarray,
    ) -> float:
        """Compute distance from a point to a line defined by two points.

        Args:
            point: Point (x, y).
            line_p1: First point on line (x, y).
            line_p2: Second point on line (x, y).

        Returns:
            Distance from point to line.
        """
        # Line direction
        d = line_p2 - line_p1
        d_len = np.linalg.norm(d)

        if d_len < 1e-6:
            # Degenerate line
            return np.linalg.norm(point - line_p1)

        # Compute perpendicular distance
        # Using cross product formula: |AP x d| / |d|
        ap = point - line_p1
        cross = abs(ap[0] * d[1] - ap[1] * d[0])

        return cross / d_len

    def refine_with_fallback(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_lines: list[dict[str, Any]] | None = None,
        world_lines: list[dict[str, Any]] | None = None,
        camera_matrix: np.ndarray | None = None,
        image_size: tuple[int, int] = (1920, 1080),
    ) -> dict[str, Any] | None:
        """Attempt PnL optimization with fallback to points-only.

        If PnL optimization fails or produces poor results, falls back to
        optimization using only point constraints.

        Args:
            image_points: 2D image points.
            world_points: 3D world points.
            image_lines: Detected 2D lines (optional).
            world_lines: 3D world lines (optional).
            camera_matrix: Camera intrinsic matrix (optional).
            image_size: Image size (width, height).

        Returns:
            Optimized camera parameters, or None if all methods fail.
        """
        # Try PnL optimization first
        try:
            result = self.optimize(
                image_points=image_points,
                world_points=world_points,
                image_lines=image_lines,
                world_lines=world_lines,
                camera_matrix=camera_matrix,
                image_size=image_size,
            )

            if result is not None and result.get("success"):
                # Check if reprojection error is reasonable
                if result.get("reprojection_error", float("inf")) < 20.0:
                    logger.info(
                        f"PnL optimization succeeded: "
                        f"point_error={result['reprojection_error']:.2f}px, "
                        f"line_error={result.get('line_reprojection_error', 0):.2f}px"
                    )
                    return result
                else:
                    logger.warning(
                        f"PnL optimization produced high error: "
                        f"{result['reprojection_error']:.2f}px, falling back"
                    )

        except Exception as e:
            logger.warning(f"PnL optimization failed: {e}")

        # Fallback to points-only
        logger.info("Falling back to points-only optimization")
        try:
            world_points = np.array(world_points, dtype=np.float64)
            if world_points.shape[1] == 2:
                world_points = np.hstack([world_points, np.zeros((len(world_points), 1))])

            if camera_matrix is None:
                w, h = image_size
                camera_matrix = np.array([
                    [1500.0, 0, w / 2],
                    [0, 1500.0, h / 2],
                    [0, 0, 1],
                ], dtype=np.float64)

            success, rvec, tvec, _ = cv2.solvePnPRansac(
                world_points,
                np.array(image_points, dtype=np.float64),
                camera_matrix,
                None,
            )

            if success:
                return self._optimize_points_only(
                    np.array(image_points),
                    world_points,
                    camera_matrix,
                    rvec.flatten(),
                    tvec.flatten(),
                )

        except Exception as e:
            logger.error(f"Fallback optimization also failed: {e}")

        return None
