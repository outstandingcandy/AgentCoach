"""Line segment point sampling utilities for distortion estimation."""

from __future__ import annotations

import numpy as np


class LineSampler:
    """Sample points along detected line segments.

    Used to extract points along straight lines for distortion estimation.
    Distorted lines appear curved, so we sample points and measure
    how straight they are after undistortion.
    """

    def __init__(
        self,
        min_line_length: float = 200.0,
        preferred_orientations: list[str] | None = None,
    ):
        """Initialize line sampler.

        Args:
            min_line_length: Minimum line length to consider for sampling.
            preferred_orientations: Preferred line orientations for sampling.
                Defaults to ["horizontal"] since sidelines show distortion best.
        """
        self.min_line_length = min_line_length
        self.preferred_orientations = preferred_orientations or ["horizontal"]

    def select_calibration_lines(
        self,
        lines: list[dict],
        max_lines: int = 3,
    ) -> list[dict]:
        """Select best lines for distortion calibration.

        Prefers long horizontal lines (touchlines/sidelines) as they
        best reveal barrel/pincushion distortion.

        Args:
            lines: List of detected line dictionaries with keys:
                - x1, y1, x2, y2: Endpoints
                - length: Line length in pixels
                - orientation: "horizontal", "vertical", or "diagonal"
            max_lines: Maximum number of lines to select.

        Returns:
            List of selected lines, sorted by suitability.
        """
        # Filter by minimum length
        valid_lines = [
            l for l in lines
            if l.get("length", 0) >= self.min_line_length
        ]

        if not valid_lines:
            return []

        # Score lines: prefer long horizontal lines
        def score_line(line: dict) -> float:
            length_score = min(1.0, line.get("length", 0) / 500.0)
            orientation = line.get("orientation", "")

            if orientation in self.preferred_orientations:
                orientation_bonus = 0.5
            elif orientation == "diagonal":
                orientation_bonus = 0.2
            else:
                orientation_bonus = 0.0

            return length_score + orientation_bonus

        # Sort by score
        scored = sorted(valid_lines, key=score_line, reverse=True)

        return scored[:max_lines]

    def sample_line_points(
        self,
        line: dict,
        num_points: int = 50,
    ) -> np.ndarray:
        """Sample evenly spaced points along a line segment.

        Args:
            line: Line dictionary with x1, y1, x2, y2 endpoints.
            num_points: Number of points to sample.

        Returns:
            Array of shape (num_points, 2) with (x, y) coordinates.
        """
        x1 = line["x1"]
        y1 = line["y1"]
        x2 = line["x2"]
        y2 = line["y2"]

        t = np.linspace(0, 1, num_points)
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return np.column_stack([x, y])

    def sample_multiple_lines(
        self,
        lines: list[dict],
        num_points_per_line: int = 50,
    ) -> list[np.ndarray]:
        """Sample points from multiple lines.

        Args:
            lines: List of line dictionaries.
            num_points_per_line: Points to sample per line.

        Returns:
            List of point arrays, one per line.
        """
        return [
            self.sample_line_points(line, num_points_per_line)
            for line in lines
        ]
