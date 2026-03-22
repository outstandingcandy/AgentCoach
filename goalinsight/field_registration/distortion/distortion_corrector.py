"""Lens distortion correction using the division model.

Implements grid search to find the optimal distortion coefficient
that makes detected straight lines as straight as possible.

Division Model:
    r_u = r_d / (1 + k * r_d²)

Where:
    - r_d = distorted point distance from image center
    - r_u = undistorted point distance from image center
    - k < 0 = barrel distortion (typical for wide-angle lenses)
    - k > 0 = pincushion distortion
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .line_sampler import LineSampler

logger = logging.getLogger(__name__)


class DistortionCorrector:
    """Estimate and correct lens distortion using the division model.

    Uses grid search over detected straight lines to find the distortion
    coefficient k that minimizes line curvature (straightness error).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize distortion corrector.

        Args:
            config: Configuration dictionary with optional keys:
                - k_range: Tuple (min_k, max_k) for grid search range.
                    Default (-1e-5, 1e-5).
                - k_steps: Number of steps in grid search. Default 200.
                - min_lines: Minimum lines required for estimation. Default 2.
                - num_sample_points: Points per line for straightness. Default 50.
        """
        config = config or {}
        self.k_range = config.get("k_range", (-1e-5, 1e-5))
        self.k_steps = config.get("k_steps", 200)
        self.min_lines = config.get("min_lines", 2)
        self.num_sample_points = config.get("num_sample_points", 50)

        self._line_sampler = LineSampler(
            min_line_length=config.get("min_line_length", 200.0),
            preferred_orientations=config.get("preferred_orientations", ["horizontal"]),
        )

        self._cached_k: float | None = None
        self._estimation_details: dict | None = None

    @property
    def cached_k(self) -> float | None:
        """Get the cached distortion coefficient from last estimation."""
        return self._cached_k

    @property
    def estimation_details(self) -> dict | None:
        """Get details from the last distortion estimation."""
        return self._estimation_details

    def estimate_distortion(
        self,
        lines: list[dict],
        image_size: tuple[int, int],
        cache_result: bool = True,
    ) -> float:
        """Estimate distortion coefficient from detected lines.

        Uses grid search to find k that minimizes total straightness error
        across all calibration lines.

        Args:
            lines: List of detected line dictionaries with:
                - x1, y1, x2, y2: Endpoints
                - length: Line length in pixels
                - orientation: "horizontal", "vertical", or "diagonal"
            image_size: Image dimensions as (width, height).
            cache_result: If True, cache the result for later use.

        Returns:
            Estimated distortion coefficient k.
            Returns 0.0 if insufficient lines for estimation.
        """
        w, h = image_size
        center = (w / 2.0, h / 2.0)

        # Select calibration lines
        calib_lines = self._line_sampler.select_calibration_lines(lines)

        if len(calib_lines) < self.min_lines:
            logger.debug(
                f"Insufficient lines for distortion estimation: "
                f"{len(calib_lines)} < {self.min_lines}"
            )
            return 0.0

        # Sample points from each line
        all_points = self._line_sampler.sample_multiple_lines(
            calib_lines, self.num_sample_points
        )

        # Grid search for optimal k
        k_values = np.linspace(self.k_range[0], self.k_range[1], self.k_steps)
        best_k = 0.0
        best_error = float("inf")
        errors_by_k = []

        for k in k_values:
            total_error = 0.0
            for points in all_points:
                undistorted = self.undistort_points(points, k, center)
                error = self.compute_line_straightness_error(undistorted)
                total_error += error

            errors_by_k.append((k, total_error))

            if total_error < best_error:
                best_error = total_error
                best_k = k

        # Store estimation details for debugging/visualization
        self._estimation_details = {
            "best_k": best_k,
            "best_error": best_error,
            "num_lines": len(calib_lines),
            "center": center,
            "errors_by_k": errors_by_k,
            "calib_lines": calib_lines,
        }

        if cache_result:
            self._cached_k = best_k

        logger.info(
            f"Estimated distortion k={best_k:.2e} "
            f"(error={best_error:.2f}px from {len(calib_lines)} lines)"
        )

        return best_k

    def undistort_points(
        self,
        points: np.ndarray,
        k: float,
        center: tuple[float, float],
    ) -> np.ndarray:
        """Apply division model undistortion to points.

        Formula: r_u = r_d / (1 + k * r_d²)

        Args:
            points: Array of shape (N, 2) with (x, y) coordinates.
            k: Distortion coefficient.
            center: Image center (cx, cy).

        Returns:
            Undistorted points of shape (N, 2).
        """
        if k == 0:
            return points.copy()

        points = np.asarray(points, dtype=np.float64)
        cx, cy = center

        # Shift to center
        x = points[:, 0] - cx
        y = points[:, 1] - cy

        # Compute radial distance
        r_d = np.sqrt(x**2 + y**2)

        # Division model: r_u = r_d / (1 + k * r_d²)
        r_d_sq = r_d**2
        scale = 1.0 / (1.0 + k * r_d_sq)

        # Avoid division by zero for points at center
        scale = np.where(r_d > 1e-6, scale, 1.0)

        # Apply undistortion
        x_u = x * scale
        y_u = y * scale

        # Shift back from center
        undistorted = np.column_stack([x_u + cx, y_u + cy])

        return undistorted

    def distort_points(
        self,
        points: np.ndarray,
        k: float,
        center: tuple[float, float],
    ) -> np.ndarray:
        """Apply division model distortion to points (inverse of undistort).

        For the division model, the inverse requires solving:
        r_d = r_u * (1 + k * r_d²)

        We solve this iteratively using Newton's method.

        Args:
            points: Array of shape (N, 2) with undistorted (x, y) coordinates.
            k: Distortion coefficient.
            center: Image center (cx, cy).

        Returns:
            Distorted points of shape (N, 2).
        """
        if k == 0:
            return points.copy()

        points = np.asarray(points, dtype=np.float64)
        cx, cy = center

        # Shift to center
        x_u = points[:, 0] - cx
        y_u = points[:, 1] - cy
        r_u = np.sqrt(x_u**2 + y_u**2)

        # Initial guess: r_d ≈ r_u
        r_d = r_u.copy()

        # Newton's method to solve: r_d / (1 + k * r_d²) = r_u
        # => r_d - r_u * (1 + k * r_d²) = 0
        # => f(r_d) = r_d - r_u - k * r_u * r_d² = 0
        # => f'(r_d) = 1 - 2 * k * r_u * r_d
        for _ in range(10):
            f = r_d - r_u - k * r_u * r_d**2
            f_prime = 1 - 2 * k * r_u * r_d
            # Avoid division by zero
            f_prime = np.where(np.abs(f_prime) > 1e-10, f_prime, 1e-10)
            r_d = r_d - f / f_prime

        # Compute scale factor
        scale = np.where(r_u > 1e-6, r_d / r_u, 1.0)

        # Apply distortion
        x_d = x_u * scale
        y_d = y_u * scale

        # Shift back from center
        distorted = np.column_stack([x_d + cx, y_d + cy])

        return distorted

    @staticmethod
    def compute_line_straightness_error(points: np.ndarray) -> float:
        """Compute straightness error using PCA.

        Fits a line to the points using PCA and computes the average
        orthogonal distance from points to the fitted line.

        Args:
            points: Array of shape (N, 2) with (x, y) coordinates.

        Returns:
            Mean orthogonal distance to the best-fit line (in pixels).
            Returns 0.0 if insufficient points.
        """
        if len(points) < 3:
            return 0.0

        points = np.asarray(points, dtype=np.float64)

        # Center the points
        mean = np.mean(points, axis=0)
        centered = points - mean

        # PCA to find principal direction
        cov = np.cov(centered.T)

        # Handle degenerate case (all points nearly identical)
        if np.any(np.isnan(cov)) or np.all(np.abs(cov) < 1e-10):
            return 0.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Principal direction (largest eigenvalue)
        principal_idx = np.argmax(eigenvalues)
        principal_dir = eigenvectors[:, principal_idx]

        # Orthogonal direction (smallest eigenvalue = variance perpendicular to line)
        orthogonal_idx = 1 - principal_idx
        orthogonal_dir = eigenvectors[:, orthogonal_idx]

        # Project points onto orthogonal direction to get distances
        distances = np.abs(np.dot(centered, orthogonal_dir))

        # Mean distance is the straightness error
        return float(np.mean(distances))

    def undistort_keypoints(
        self,
        keypoints: list[dict],
        image_size: tuple[int, int],
        k: float | None = None,
    ) -> list[dict]:
        """Undistort keypoint coordinates.

        Args:
            keypoints: List of keypoint dictionaries with "x" and "y" keys.
            image_size: Image dimensions as (width, height).
            k: Distortion coefficient. If None, uses cached value.

        Returns:
            List of keypoints with undistorted coordinates.
        """
        if k is None:
            k = self._cached_k
        if k is None or k == 0:
            return keypoints

        w, h = image_size
        center = (w / 2.0, h / 2.0)

        # Extract points
        points = np.array([[kp["x"], kp["y"]] for kp in keypoints])

        # Undistort
        undistorted = self.undistort_points(points, k, center)

        # Update keypoints
        result = []
        for i, kp in enumerate(keypoints):
            kp_copy = kp.copy()
            kp_copy["x"] = float(undistorted[i, 0])
            kp_copy["y"] = float(undistorted[i, 1])
            result.append(kp_copy)

        return result

    def undistort_lines(
        self,
        lines: list[dict],
        image_size: tuple[int, int],
        k: float | None = None,
    ) -> list[dict]:
        """Undistort line endpoint coordinates.

        Args:
            lines: List of line dictionaries with x1, y1, x2, y2 keys.
            image_size: Image dimensions as (width, height).
            k: Distortion coefficient. If None, uses cached value.

        Returns:
            List of lines with undistorted endpoint coordinates.
        """
        if k is None:
            k = self._cached_k
        if k is None or k == 0:
            return lines

        w, h = image_size
        center = (w / 2.0, h / 2.0)

        result = []
        for line in lines:
            # Extract endpoints
            points = np.array([
                [line["x1"], line["y1"]],
                [line["x2"], line["y2"]],
            ])

            # Undistort
            undistorted = self.undistort_points(points, k, center)

            # Update line
            line_copy = line.copy()
            line_copy["x1"] = float(undistorted[0, 0])
            line_copy["y1"] = float(undistorted[0, 1])
            line_copy["x2"] = float(undistorted[1, 0])
            line_copy["y2"] = float(undistorted[1, 1])

            # Recalculate length
            dx = line_copy["x2"] - line_copy["x1"]
            dy = line_copy["y2"] - line_copy["y1"]
            line_copy["length"] = float(np.sqrt(dx**2 + dy**2))

            result.append(line_copy)

        return result

    def reset_cache(self) -> None:
        """Clear the cached distortion coefficient."""
        self._cached_k = None
        self._estimation_details = None
