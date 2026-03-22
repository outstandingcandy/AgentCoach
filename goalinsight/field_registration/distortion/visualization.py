"""Visualization tools for lens distortion correction."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .distortion_corrector import DistortionCorrector
from .line_sampler import LineSampler

logger = logging.getLogger(__name__)


class DistortionVisualizer:
    """Visualize lens distortion correction effects."""

    def __init__(self, corrector: DistortionCorrector | None = None):
        """Initialize visualizer.

        Args:
            corrector: DistortionCorrector instance to use.
                If None, creates a new one.
        """
        self.corrector = corrector or DistortionCorrector()
        self._line_sampler = LineSampler()

    def visualize_correction(
        self,
        frame: np.ndarray,
        lines: list[dict],
        k: float,
        output_path: str | Path,
        max_lines: int = 5,
    ) -> None:
        """Generate before/after comparison of distortion correction.

        Creates a side-by-side image showing:
        - Left: Original lines (curved due to distortion)
        - Right: Undistorted lines (should be straighter)

        Args:
            frame: Input frame (BGR format).
            lines: List of detected line dictionaries.
            k: Distortion coefficient.
            output_path: Path to save the visualization image.
            max_lines: Maximum number of lines to visualize.
        """
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)

        # Create side-by-side canvas
        canvas = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Copy frames
        vis_orig = frame.copy()
        vis_undist = frame.copy()

        # Select calibration lines
        calib_lines = self._line_sampler.select_calibration_lines(lines)[:max_lines]

        # Draw lines on both sides
        colors = [
            (0, 255, 255),   # Cyan
            (255, 0, 255),   # Magenta
            (255, 255, 0),   # Yellow
            (0, 255, 0),     # Green
            (255, 128, 0),   # Orange
        ]

        for i, line in enumerate(calib_lines):
            color = colors[i % len(colors)]

            # Sample points along line
            pts = self._line_sampler.sample_line_points(line, 50)

            # Draw original (distorted) line on left
            for j in range(len(pts) - 1):
                pt1 = tuple(pts[j].astype(int))
                pt2 = tuple(pts[j + 1].astype(int))
                cv2.line(vis_orig, pt1, pt2, color, 2)

            # Draw undistorted line on right
            pts_u = self.corrector.undistort_points(pts, k, center)
            for j in range(len(pts_u) - 1):
                pt1 = tuple(pts_u[j].astype(int))
                pt2 = tuple(pts_u[j + 1].astype(int))
                cv2.line(vis_undist, pt1, pt2, color, 2)

            # Draw line endpoints
            cv2.circle(vis_orig, tuple(pts[0].astype(int)), 5, color, -1)
            cv2.circle(vis_orig, tuple(pts[-1].astype(int)), 5, color, -1)
            cv2.circle(vis_undist, tuple(pts_u[0].astype(int)), 5, color, -1)
            cv2.circle(vis_undist, tuple(pts_u[-1].astype(int)), 5, color, -1)

        # Assemble canvas
        canvas[:, :w] = vis_orig
        canvas[:, w:] = vis_undist

        # Add labels
        cv2.putText(
            canvas, "Before (Distorted)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )
        cv2.putText(
            canvas, f"After (k={k:.2e})", (w + 20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        # Draw center crosshair
        cross_color = (128, 128, 128)
        cx, cy = int(center[0]), int(center[1])
        cv2.line(canvas, (cx - 20, cy), (cx + 20, cy), cross_color, 1)
        cv2.line(canvas, (cx, cy - 20), (cx, cy + 20), cross_color, 1)
        cv2.line(canvas, (w + cx - 20, cy), (w + cx + 20, cy), cross_color, 1)
        cv2.line(canvas, (w + cx, cy - 20), (w + cx, cy + 20), cross_color, 1)

        # Add divider line
        cv2.line(canvas, (w, 0), (w, h), (255, 255, 255), 2)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

        logger.info(f"Saved distortion visualization to {output_path}")

    def visualize_grid_search(
        self,
        output_path: str | Path,
    ) -> None:
        """Plot the grid search results showing error vs k value.

        Requires that estimate_distortion() has been called first.

        Args:
            output_path: Path to save the plot image.
        """
        details = self.corrector.estimation_details
        if details is None:
            logger.warning("No estimation details available. Run estimate_distortion() first.")
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            errors_by_k = details["errors_by_k"]
            k_values = [x[0] for x in errors_by_k]
            errors = [x[1] for x in errors_by_k]
            best_k = details["best_k"]
            best_error = details["best_error"]

            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(k_values, errors, "b-", linewidth=1.5, label="Total Error")
            ax.axvline(best_k, color="r", linestyle="--", label=f"Best k={best_k:.2e}")
            ax.scatter([best_k], [best_error], c="red", s=100, zorder=5)

            ax.set_xlabel("Distortion Coefficient k", fontsize=12)
            ax.set_ylabel("Straightness Error (pixels)", fontsize=12)
            ax.set_title("Grid Search for Distortion Coefficient", fontsize=14)
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add annotation for best k
            ax.annotate(
                f"k={best_k:.2e}\nerror={best_error:.2f}px",
                xy=(best_k, best_error),
                xytext=(10, 30),
                textcoords="offset points",
                fontsize=10,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
            )

            plt.tight_layout()
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150)
            plt.close()

            logger.info(f"Saved grid search plot to {output_path}")

        except ImportError:
            logger.warning("matplotlib not available, skipping grid search visualization")

    def visualize_undistortion_map(
        self,
        image_size: tuple[int, int],
        k: float,
        output_path: str | Path,
        grid_spacing: int = 50,
    ) -> None:
        """Visualize the undistortion mapping as a grid.

        Shows how a regular grid gets distorted/undistorted.

        Args:
            image_size: Image dimensions as (width, height).
            k: Distortion coefficient.
            output_path: Path to save the visualization.
            grid_spacing: Spacing between grid lines in pixels.
        """
        w, h = image_size
        center = (w / 2.0, h / 2.0)

        # Create blank canvas
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw original grid in gray
        for x in range(0, w, grid_spacing):
            cv2.line(canvas, (x, 0), (x, h), (50, 50, 50), 1)
        for y in range(0, h, grid_spacing):
            cv2.line(canvas, (0, y), (w, y), (50, 50, 50), 1)

        # Draw undistorted grid
        # Generate grid points
        x_coords = np.arange(0, w, grid_spacing // 2)
        y_coords = np.arange(0, h, grid_spacing // 2)

        # Draw vertical lines
        for x in range(0, w, grid_spacing):
            points = np.array([[x, y] for y in y_coords])
            undist = self.corrector.undistort_points(points, k, center)
            for i in range(len(undist) - 1):
                pt1 = tuple(undist[i].astype(int))
                pt2 = tuple(undist[i + 1].astype(int))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

        # Draw horizontal lines
        for y in range(0, h, grid_spacing):
            points = np.array([[x, y] for x in x_coords])
            undist = self.corrector.undistort_points(points, k, center)
            for i in range(len(undist) - 1):
                pt1 = tuple(undist[i].astype(int))
                pt2 = tuple(undist[i + 1].astype(int))
                cv2.line(canvas, pt1, pt2, (0, 255, 0), 1)

        # Draw center
        cx, cy = int(center[0]), int(center[1])
        cv2.circle(canvas, (cx, cy), 5, (0, 0, 255), -1)

        # Add legend
        cv2.putText(
            canvas, f"k = {k:.2e}", (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.putText(
            canvas, "Gray: Original grid", (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1
        )
        cv2.putText(
            canvas, "Green: Undistorted grid", (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
        )

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), canvas)

        logger.info(f"Saved undistortion map to {output_path}")
