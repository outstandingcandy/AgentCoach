"""PnL solver for camera calibration and field registration."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

logger = logging.getLogger(__name__)


class PnLSolver:
    """Solve Perspective-n-Lines/Points for camera calibration."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize PnL solver.

        Args:
            config: Solver configuration.
                - ransac_iterations: Number of RANSAC iterations
                - reprojection_threshold: Threshold for RANSAC inliers
                - pnl_refine: Enable joint point+line refinement (default False)
                - line_weight: Relative weight of line constraints
                - hfov_degrees: Horizontal field of view for focal estimation
                - auto_focal: If True, auto-estimate focal length
        """
        self.config = config or {}
        self.ransac_iterations = self.config.get("ransac_iterations", 1000)
        self.reprojection_threshold = self.config.get("reprojection_threshold", 8.0)

        # PnL optimization settings
        self.pnl_refine = self.config.get("pnl_refine", False)
        self.line_weight = self.config.get("line_weight", 1.0)

        # Default camera intrinsics (can be overridden)
        # 1500 is a reasonable default for ~60 degree HFOV on 1920px width
        self.default_fx = self.config.get("default_fx", 1500.0)
        self.default_fy = self.config.get("default_fy", 1500.0)

        # Auto focal estimation settings
        self._hfov_degrees = self.config.get("hfov_degrees", None)
        self._auto_focal = self.config.get("auto_focal", False)

        # PnL optimizer (lazy initialization)
        self._pnl_optimizer = None

    def focal_from_hfov(self, hfov_degrees: float, width: int) -> float:
        """Calculate focal length from horizontal field of view.

        Args:
            hfov_degrees: Horizontal field of view in degrees.
            width: Image width in pixels.

        Returns:
            Focal length in pixels.
        """
        return width / (2.0 * np.tan(np.radians(hfov_degrees / 2.0)))

    def estimate_focal_from_homography(
        self,
        H: np.ndarray,
        image_size: tuple[int, int],
    ) -> float | None:
        """Estimate focal length from homography using orthogonality constraints.

        Uses the fact that rotation matrix columns should be orthogonal
        to estimate the focal length from a planar homography.

        Args:
            H: 3x3 homography matrix (world to image).
            image_size: Image dimensions (width, height).

        Returns:
            Estimated focal length, or None if estimation fails.
        """
        w, h = image_size

        # Normalize homography
        H = H / H[2, 2]

        # Extract columns
        h1 = H[:, 0]
        h2 = H[:, 1]

        # For a planar homography H = K[r1 r2 t], we have:
        # r1^T r2 = 0 (orthogonality)
        # |r1| = |r2| (same norm)

        # Using simplified estimation: assume principal point at center
        cx, cy = w / 2.0, h / 2.0

        # Estimate using orthogonality constraint
        # h1 = [f*r1x + cx*r1z, f*r1y + cy*r1z, r1z]
        # This is approximate - full estimation requires Zhang's method

        # Simple heuristic: compute from the ratio of homography elements
        # For typical broadcast cameras, HFOV is 50-90 degrees
        # This gives focal lengths of 1100-2000px for 1920 width

        try:
            # Use the trace-based estimation
            A = H[:2, :2]
            # For orthonormal columns: A^T A should be f^2 * I
            ATA = A.T @ A

            # Estimate f from diagonal
            f_sq = (ATA[0, 0] + ATA[1, 1]) / 2.0
            if f_sq > 0:
                f_estimated = np.sqrt(f_sq)
                # Sanity check: focal length should be reasonable
                if 500 < f_estimated < 5000:
                    logger.debug(f"Estimated focal length: {f_estimated:.1f}px")
                    return float(f_estimated)
        except Exception as e:
            logger.debug(f"Focal estimation failed: {e}")

        return None

    def get_camera_matrix(
        self,
        image_size: tuple[int, int],
        H: np.ndarray | None = None,
    ) -> np.ndarray:
        """Get camera intrinsic matrix, optionally auto-estimating focal length.

        Args:
            image_size: Image dimensions (width, height).
            H: Optional homography for focal length estimation.

        Returns:
            3x3 camera intrinsic matrix.
        """
        w, h = image_size
        fx, fy = self.default_fx, self.default_fy

        # Try to estimate focal length from HFOV if provided
        if self._hfov_degrees is not None:
            fx = self.focal_from_hfov(self._hfov_degrees, w)
            fy = fx  # Assume square pixels
            logger.debug(f"Focal from HFOV ({self._hfov_degrees}°): {fx:.1f}px")

        # Try to estimate from homography if auto_focal enabled
        elif self._auto_focal and H is not None:
            estimated_f = self.estimate_focal_from_homography(H, image_size)
            if estimated_f is not None:
                fx, fy = estimated_f, estimated_f

        return np.array([
            [fx, 0, w / 2],
            [0, fy, h / 2],
            [0, 0, 1],
        ], dtype=np.float32)

    def set_hfov(self, hfov_degrees: float) -> None:
        """Set horizontal field of view for focal length estimation.

        Args:
            hfov_degrees: Horizontal FOV in degrees (typically 50-90 for broadcast).
        """
        self._hfov_degrees = hfov_degrees
        logger.info(f"Set HFOV to {hfov_degrees}°")

    def solve_homography(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
    ) -> np.ndarray | None:
        """Solve for homography matrix using point correspondences.

        Args:
            image_points: 2D image points, shape (N, 2).
            world_points: 2D world points on pitch, shape (N, 2).

        Returns:
            3x3 homography matrix or None if failed.
        """
        if len(image_points) < 4 or len(world_points) < 4:
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Use RANSAC for robust estimation
        H, mask = cv2.findHomography(
            image_points,
            world_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.reprojection_threshold,
        )

        return H

    def solve_pnp(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        camera_matrix: np.ndarray | None = None,
        dist_coeffs: np.ndarray | None = None,
        image_size: tuple[int, int] | None = None,
        homography: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Solve PnP problem for camera pose estimation.

        Args:
            image_points: 2D image points, shape (N, 2).
            world_points: 3D world points (z=0 for pitch), shape (N, 3).
            camera_matrix: Camera intrinsic matrix.
            dist_coeffs: Distortion coefficients.
            image_size: Image size (width, height) for default intrinsics.
            homography: Optional homography for auto focal estimation.

        Returns:
            Tuple of (rotation_vector, translation_vector, camera_matrix)
            or None if failed.
        """
        if len(image_points) < 4:
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Ensure world points are 3D (z=0 for pitch)
        if world_points.shape[1] == 2:
            world_points = np.hstack([world_points, np.zeros((len(world_points), 1))])

        # Default camera matrix if not provided
        if camera_matrix is None:
            if image_size is None:
                image_size = (1920, 1080)
            camera_matrix = self.get_camera_matrix(image_size, homography)

        if dist_coeffs is None:
            dist_coeffs = np.zeros(4, dtype=np.float32)

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            world_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            iterationsCount=self.ransac_iterations,
            reprojectionError=self.reprojection_threshold,
        )

        if not success:
            return None

        return rvec, tvec, camera_matrix

    def solve_pnl(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        image_lines: list[dict[str, Any]] | None = None,
        world_lines: list[dict[str, Any]] | None = None,
        camera_matrix: np.ndarray | None = None,
        image_size: tuple[int, int] | None = None,
        homography: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
        """Solve PnL problem using joint point and line optimization.

        This method extends solve_pnp by incorporating line constraints
        for improved camera calibration accuracy.

        Args:
            image_points: 2D image points, shape (N, 2).
            world_points: 3D world points (z=0 for pitch), shape (N, 3).
            image_lines: Detected 2D lines with x1, y1, x2, y2.
            world_lines: 3D world lines with endpoints.
            camera_matrix: Camera intrinsic matrix.
            image_size: Image size (width, height) for default intrinsics.
            homography: Optional homography for auto focal estimation.

        Returns:
            Tuple of (rotation_vector, translation_vector, camera_matrix, stats)
            or None if failed. Stats contains optimization metrics.
        """
        if len(image_points) < 4:
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Ensure world points are 3D
        if world_points.shape[1] == 2:
            world_points = np.hstack([world_points, np.zeros((len(world_points), 1))])

        # Default camera matrix
        if camera_matrix is None:
            if image_size is None:
                image_size = (1920, 1080)
            camera_matrix = self.get_camera_matrix(image_size, homography)

        # Initialize PnL optimizer if needed
        if self._pnl_optimizer is None:
            from .pnlcalib import PnLOptimizer
            self._pnl_optimizer = PnLOptimizer(line_weight=self.line_weight)

        # Run PnL optimization
        result = self._pnl_optimizer.refine_with_fallback(
            image_points=image_points,
            world_points=world_points,
            image_lines=image_lines,
            world_lines=world_lines,
            camera_matrix=camera_matrix,
            image_size=image_size,
        )

        if result is None:
            logger.warning("PnL optimization failed")
            return None

        rvec = result["rvec"].reshape(3, 1)
        tvec = result["tvec"].reshape(3, 1)

        stats = {
            "reprojection_error": result.get("reprojection_error", 0.0),
            "line_reprojection_error": result.get("line_reprojection_error", 0.0),
            "optimization_cost": result.get("optimization_cost", 0.0),
            "num_points": result.get("num_points", 0),
            "num_lines": result.get("num_lines", 0),
            "success": result.get("success", False),
        }

        return rvec, tvec, camera_matrix, stats

    def decompose_homography(
        self,
        H: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> list[dict[str, np.ndarray]]:
        """Decompose homography into rotation and translation.

        Args:
            H: 3x3 homography matrix.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            List of possible decomposition solutions.
        """
        # Decompose homography
        num_solutions, rotations, translations, normals = cv2.decomposeHomographyMat(
            H, camera_matrix
        )

        solutions = []
        for i in range(num_solutions):
            solutions.append({
                "rotation": rotations[i],
                "translation": translations[i],
                "normal": normals[i],
            })

        return solutions

    def compute_camera_params(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> dict[str, Any]:
        """Compute camera parameters from pose.

        Args:
            rvec: Rotation vector.
            tvec: Translation vector.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            Dictionary of camera parameters.
        """
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)

        # Camera position in world coordinates
        camera_position = -R.T @ tvec.flatten()

        # Extract intrinsic parameters
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Compute field of view
        h_fov = 2 * np.arctan(cx / fx) * 180 / np.pi
        v_fov = 2 * np.arctan(cy / fy) * 180 / np.pi

        return {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "rotation_matrix": R.tolist(),
            "rotation_vector": rvec.flatten().tolist(),
            "translation": tvec.flatten().tolist(),
            "camera_position": camera_position.tolist(),
            "horizontal_fov": float(h_fov),
            "vertical_fov": float(v_fov),
        }

    def project_points(
        self,
        world_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project 3D world points to 2D image points.

        Args:
            world_points: 3D world points, shape (N, 3).
            rvec: Rotation vector.
            tvec: Translation vector.
            camera_matrix: Camera intrinsic matrix.
            dist_coeffs: Distortion coefficients.

        Returns:
            2D image points, shape (N, 2).
        """
        from ..utils.projection import project_points_2d
        return project_points_2d(world_points, rvec, tvec, camera_matrix, dist_coeffs)

    def unproject_to_pitch(
        self,
        image_points: np.ndarray,
        H: np.ndarray,
    ) -> np.ndarray:
        """Unproject 2D image points to pitch coordinates using homography.

        Args:
            image_points: 2D image points, shape (N, 2).
            H: Homography matrix (image to pitch).

        Returns:
            2D pitch coordinates, shape (N, 2).
        """
        image_points = np.array(image_points, dtype=np.float32)

        # Add homogeneous coordinate
        ones = np.ones((len(image_points), 1))
        homogeneous = np.hstack([image_points, ones])

        # Apply homography
        pitch_homogeneous = (H @ homogeneous.T).T

        # Normalize
        pitch_points = pitch_homogeneous[:, :2] / pitch_homogeneous[:, 2:3]

        return pitch_points

    def compute_reprojection_error(
        self,
        image_points: np.ndarray,
        world_points: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float:
        """Compute mean reprojection error.

        Args:
            image_points: Observed 2D image points.
            world_points: 3D world points.
            rvec: Rotation vector.
            tvec: Translation vector.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            Mean reprojection error in pixels.
        """
        projected = self.project_points(
            world_points, rvec, tvec, camera_matrix
        )

        errors = np.linalg.norm(image_points - projected, axis=1)
        return float(np.mean(errors))


class FieldRegistrar:
    """Complete field registration pipeline."""

    def __init__(
        self,
        pitch_template: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ):
        """Initialize field registrar.

        Args:
            pitch_template: Pitch template with keypoints and lines.
            config: Registration configuration.
        """
        self.pitch_template = pitch_template
        self.config = config or {}
        self.pnl_solver = PnLSolver(config)

        # Build template lookup
        self._build_template_lookup()

    def _build_template_lookup(self) -> None:
        """Build lookup dictionary for template keypoints and lines."""
        self.template_keypoints = {}
        self.template_lines = {}

        if self.pitch_template is None:
            return

        keypoints = self.pitch_template.get("keypoints", [])
        for kp in keypoints:
            kp_id = kp[0]
            self.template_keypoints[kp_id] = {
                "id": kp_id,
                "name": kp[1],
                "x": kp[2],
                "y": kp[3],
            }

        # Build line lookup from template
        lines = self.pitch_template.get("lines", [])
        for line in lines:
            line_id = line[0]
            line_name = line[1]
            keypoint_ids = line[2]

            # Get world coordinates for line endpoints
            # Lines are defined by a list of keypoint IDs
            if len(keypoint_ids) >= 2:
                start_kp = self.template_keypoints.get(keypoint_ids[0])
                end_kp = self.template_keypoints.get(keypoint_ids[-1])

                if start_kp and end_kp:
                    self.template_lines[line_id] = {
                        "id": line_id,
                        "name": line_name,
                        "x1": start_kp["x"],
                        "y1": start_kp["y"],
                        "z1": 0.0,
                        "x2": end_kp["x"],
                        "y2": end_kp["y"],
                        "z2": 0.0,
                    }

    def register(
        self,
        detected_keypoints: list[dict[str, Any]],
        image_size: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Register field from detected keypoints.

        Args:
            detected_keypoints: List of detected keypoints with id, x, y.
            image_size: Image size (width, height).

        Returns:
            Registration result with homography and camera params,
            or None if registration failed.
        """
        if not self.template_keypoints:
            return None

        # Match detected keypoints to template
        image_points = []
        world_points = []

        for det_kp in detected_keypoints:
            kp_id = det_kp.get("id")
            if kp_id in self.template_keypoints:
                template_kp = self.template_keypoints[kp_id]
                image_points.append([det_kp["x"], det_kp["y"]])
                world_points.append([template_kp["x"], template_kp["y"]])

        if len(image_points) < 4:
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Compute homography
        H = self.pnl_solver.solve_homography(image_points, world_points)
        if H is None:
            return None

        # Solve PnP for camera parameters
        world_points_3d = np.hstack([world_points, np.zeros((len(world_points), 1))])
        pnp_result = self.pnl_solver.solve_pnp(
            image_points,
            world_points_3d,
            image_size=image_size,
        )

        camera_params = {}
        if pnp_result is not None:
            rvec, tvec, camera_matrix = pnp_result
            camera_params = self.pnl_solver.compute_camera_params(
                rvec, tvec, camera_matrix
            )

            # Compute reprojection error
            error = self.pnl_solver.compute_reprojection_error(
                image_points, world_points_3d, rvec, tvec, camera_matrix
            )
            camera_params["reprojection_error"] = error

        return {
            "homography": H.tolist(),
            "camera_params": camera_params,
            "num_keypoints": len(image_points),
        }

    def register_with_lines(
        self,
        detected_keypoints: list[dict[str, Any]],
        detected_lines: list[dict[str, Any]],
        image_size: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Register field using both keypoints and lines (PnL optimization).

        This method uses joint point+line optimization for improved accuracy.

        Args:
            detected_keypoints: List of detected keypoints with id, x, y.
            detected_lines: List of detected lines with id, x1, y1, x2, y2.
            image_size: Image size (width, height).

        Returns:
            Registration result with homography, camera params, and PnL stats,
            or None if registration failed.
        """
        if not self.template_keypoints:
            return None

        # Match detected keypoints to template
        image_points = []
        world_points = []

        for det_kp in detected_keypoints:
            kp_id = det_kp.get("id")
            if kp_id in self.template_keypoints:
                template_kp = self.template_keypoints[kp_id]
                image_points.append([det_kp["x"], det_kp["y"]])
                world_points.append([template_kp["x"], template_kp["y"]])

        if len(image_points) < 4:
            return None

        image_points = np.array(image_points, dtype=np.float32)
        world_points = np.array(world_points, dtype=np.float32)

        # Match detected lines to template
        matched_image_lines = []
        matched_world_lines = []

        for det_line in detected_lines:
            line_id = det_line.get("id")
            if line_id in self.template_lines:
                template_line = self.template_lines[line_id]
                matched_image_lines.append(det_line)
                matched_world_lines.append(template_line)

        # Compute homography
        H = self.pnl_solver.solve_homography(image_points, world_points)
        if H is None:
            return None

        # Solve PnL for camera parameters
        world_points_3d = np.hstack([world_points, np.zeros((len(world_points), 1))])

        if matched_world_lines and self.pnl_solver.pnl_refine:
            # Use joint PnL optimization
            pnl_result = self.pnl_solver.solve_pnl(
                image_points,
                world_points_3d,
                image_lines=matched_image_lines,
                world_lines=matched_world_lines,
                image_size=image_size,
            )

            if pnl_result is not None:
                rvec, tvec, camera_matrix, pnl_stats = pnl_result
                camera_params = self.pnl_solver.compute_camera_params(
                    rvec, tvec, camera_matrix
                )
                camera_params["reprojection_error"] = pnl_stats.get("reprojection_error", 0.0)
                camera_params["line_reprojection_error"] = pnl_stats.get("line_reprojection_error", 0.0)

                return {
                    "homography": H.tolist(),
                    "camera_params": camera_params,
                    "num_keypoints": len(image_points),
                    "num_lines": len(matched_image_lines),
                    "pnl_stats": pnl_stats,
                }

        # Fallback to standard PnP
        pnp_result = self.pnl_solver.solve_pnp(
            image_points,
            world_points_3d,
            image_size=image_size,
        )

        camera_params = {}
        if pnp_result is not None:
            rvec, tvec, camera_matrix = pnp_result
            camera_params = self.pnl_solver.compute_camera_params(
                rvec, tvec, camera_matrix
            )
            error = self.pnl_solver.compute_reprojection_error(
                image_points, world_points_3d, rvec, tvec, camera_matrix
            )
            camera_params["reprojection_error"] = error

        return {
            "homography": H.tolist(),
            "camera_params": camera_params,
            "num_keypoints": len(image_points),
            "num_lines": len(matched_image_lines),
        }

    def transform_to_pitch(
        self,
        image_points: np.ndarray,
        homography: np.ndarray,
    ) -> np.ndarray:
        """Transform image points to pitch coordinates.

        Args:
            image_points: 2D image points.
            homography: Homography matrix.

        Returns:
            2D pitch coordinates.
        """
        return self.pnl_solver.unproject_to_pitch(image_points, homography)

    def is_on_pitch(
        self,
        pitch_point: tuple[float, float],
    ) -> bool:
        """Check if a point is within the pitch boundaries.

        Args:
            pitch_point: Point in pitch coordinates (x, y).

        Returns:
            True if point is on pitch.
        """
        if self.pitch_template is None:
            return True

        pitch_length = self.pitch_template.get("pitch", {}).get("length", 105)
        pitch_width = self.pitch_template.get("pitch", {}).get("width", 68)

        x, y = pitch_point
        half_length = pitch_length / 2
        half_width = pitch_width / 2

        return -half_length <= x <= half_length and -half_width <= y <= half_width
