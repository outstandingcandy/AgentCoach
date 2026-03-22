"""Hough line matching for soccer field calibration.

This module provides functionality to match Hough-detected lines (without semantic labels)
to the soccer pitch template lines using geometric constraints and an initial homography.

The matching is based on:
1. Line orientation (horizontal/vertical/diagonal)
2. Proximity after projection using initial homography
3. Angle similarity between detected and projected template lines
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .line_mapping import (
    HALF_L,
    HALF_W,
    PENALTY_HALF_W,
    PENALTY_AREA_DEPTH,
    GOAL_HALF_W,
    GOAL_AREA_DEPTH,
    LineMapper,
)


class HoughLineMatcher:
    """Match Hough-detected lines to soccer pitch template lines.

    Since Hough lines lack semantic labels, this class uses geometric matching
    to find the most likely corresponding template line for each detected line.
    """

    # Template lines organized by orientation
    # Format: {"orientation": [(name, p1, p2), ...]}
    TEMPLATE_LINES = {
        "horizontal": [
            # Touchlines (side lines)
            ("side_line_top", (-HALF_L, HALF_W), (HALF_L, HALF_W)),
            ("side_line_bottom", (-HALF_L, -HALF_W), (HALF_L, -HALF_W)),
            # Penalty area horizontal lines (left side)
            ("big_rect_left_top", (-HALF_L, PENALTY_HALF_W), (-HALF_L + PENALTY_AREA_DEPTH, PENALTY_HALF_W)),
            ("big_rect_left_bottom", (-HALF_L + PENALTY_AREA_DEPTH, -PENALTY_HALF_W), (-HALF_L, -PENALTY_HALF_W)),
            # Penalty area horizontal lines (right side)
            ("big_rect_right_top", (HALF_L, PENALTY_HALF_W), (HALF_L - PENALTY_AREA_DEPTH, PENALTY_HALF_W)),
            ("big_rect_right_bottom", (HALF_L - PENALTY_AREA_DEPTH, -PENALTY_HALF_W), (HALF_L, -PENALTY_HALF_W)),
            # Goal area horizontal lines (left side)
            ("small_rect_left_top", (-HALF_L, GOAL_HALF_W), (-HALF_L + GOAL_AREA_DEPTH, GOAL_HALF_W)),
            ("small_rect_left_bottom", (-HALF_L + GOAL_AREA_DEPTH, -GOAL_HALF_W), (-HALF_L, -GOAL_HALF_W)),
            # Goal area horizontal lines (right side)
            ("small_rect_right_top", (HALF_L, GOAL_HALF_W), (HALF_L - GOAL_AREA_DEPTH, GOAL_HALF_W)),
            ("small_rect_right_bottom", (HALF_L - GOAL_AREA_DEPTH, -GOAL_HALF_W), (HALF_L, -GOAL_HALF_W)),
        ],
        "vertical": [
            # Center line
            ("middle_line", (0, -HALF_W), (0, HALF_W)),
            # Goal lines
            ("side_line_left", (-HALF_L, -HALF_W), (-HALF_L, HALF_W)),
            ("side_line_right", (HALF_L, -HALF_W), (HALF_L, HALF_W)),
            # Penalty area vertical lines
            ("big_rect_left_side", (-HALF_L + PENALTY_AREA_DEPTH, PENALTY_HALF_W),
             (-HALF_L + PENALTY_AREA_DEPTH, -PENALTY_HALF_W)),
            ("big_rect_right_side", (HALF_L - PENALTY_AREA_DEPTH, PENALTY_HALF_W),
             (HALF_L - PENALTY_AREA_DEPTH, -PENALTY_HALF_W)),
            # Goal area vertical lines
            ("small_rect_left_side", (-HALF_L + GOAL_AREA_DEPTH, GOAL_HALF_W),
             (-HALF_L + GOAL_AREA_DEPTH, -GOAL_HALF_W)),
            ("small_rect_right_side", (HALF_L - GOAL_AREA_DEPTH, GOAL_HALF_W),
             (HALF_L - GOAL_AREA_DEPTH, -GOAL_HALF_W)),
        ],
        "diagonal": [],  # Center circle tangents are curved, not handled here
    }

    def __init__(
        self,
        angle_tolerance: float = 20.0,
        distance_tolerance: float = 50.0,
    ):
        """Initialize the Hough line matcher.

        Args:
            angle_tolerance: Maximum angle difference (degrees) for matching.
            distance_tolerance: Maximum distance (pixels) for matching.
        """
        self.angle_tolerance = angle_tolerance
        self.distance_tolerance = distance_tolerance

    def match_lines(
        self,
        hough_lines: list[dict[str, Any]],
        H: np.ndarray,
        image_size: tuple[int, int] = (1920, 1080),
        unique_template: bool = True,
    ) -> list[dict[str, Any]]:
        """Match Hough lines to template lines.

        Args:
            hough_lines: List of Hough-detected lines with orientation.
            H: World-to-image homography (3x3).
            image_size: Image dimensions (width, height).
            unique_template: If True, only keep best match per template line.

        Returns:
            List of matched lines with template world coordinates.
            Each entry has: hough_line, template_name, template_world_coords, distance.
        """
        matches = []
        w, h = image_size

        for hough_line in hough_lines:
            orientation = hough_line.get("orientation", "diagonal")
            if orientation == "diagonal":
                continue  # Skip diagonal lines for now

            # Get candidate template lines for this orientation
            candidates = self.TEMPLATE_LINES.get(orientation, [])
            if not candidates:
                continue

            # Find best matching template line
            best_match = None
            best_distance = float("inf")

            for name, world_p1, world_p2 in candidates:
                # Project template line to image
                img_p1 = self._project_point(world_p1, H)
                img_p2 = self._project_point(world_p2, H)

                if img_p1 is None or img_p2 is None:
                    continue

                # Check if projected line is in view
                if not self._line_in_bounds(img_p1, img_p2, w, h, margin=200):
                    continue

                # Compute distance between Hough line and projected template line
                distance = self._line_to_line_distance(
                    (hough_line["x1"], hough_line["y1"], hough_line["x2"], hough_line["y2"]),
                    (img_p1[0], img_p1[1], img_p2[0], img_p2[1]),
                )

                # Check angle compatibility
                hough_angle = hough_line["angle"]
                template_angle = np.arctan2(img_p2[1] - img_p1[1], img_p2[0] - img_p1[0]) * 180 / np.pi
                angle_diff = abs(hough_angle - template_angle)
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff > self.angle_tolerance:
                    continue

                if distance < best_distance and distance < self.distance_tolerance:
                    best_distance = distance
                    best_match = {
                        "hough_line": hough_line,
                        "template_name": name,
                        "template_world_p1": world_p1,
                        "template_world_p2": world_p2,
                        "projected_p1": img_p1,
                        "projected_p2": img_p2,
                        "distance": distance,
                        "angle_diff": angle_diff,
                    }

            if best_match is not None:
                matches.append(best_match)

        # If unique_template, keep only best match per template line
        if unique_template and matches:
            best_per_template: dict[str, dict] = {}
            for match in matches:
                name = match["template_name"]
                if name not in best_per_template or match["distance"] < best_per_template[name]["distance"]:
                    best_per_template[name] = match
            matches = list(best_per_template.values())

        return matches

    def _project_point(
        self,
        world_pt: tuple[float, float],
        H: np.ndarray,
    ) -> tuple[float, float] | None:
        """Project a world point to image using homography.

        Args:
            world_pt: World point (x, y).
            H: World-to-image homography.

        Returns:
            Image point (x, y) or None if behind camera.
        """
        pt = np.array([world_pt[0], world_pt[1], 1.0])
        img_pt = H @ pt
        if abs(img_pt[2]) < 1e-6:
            return None
        return (img_pt[0] / img_pt[2], img_pt[1] / img_pt[2])

    def _line_in_bounds(
        self,
        p1: tuple[float, float],
        p2: tuple[float, float],
        w: int,
        h: int,
        margin: int = 0,
    ) -> bool:
        """Check if line is at least partially in image bounds.

        Args:
            p1: First point.
            p2: Second point.
            w: Image width.
            h: Image height.
            margin: Extra margin outside image.

        Returns:
            True if line is at least partially visible.
        """
        # Check if at least one endpoint is in extended bounds
        in_bounds_p1 = -margin < p1[0] < w + margin and -margin < p1[1] < h + margin
        in_bounds_p2 = -margin < p2[0] < w + margin and -margin < p2[1] < h + margin
        return in_bounds_p1 or in_bounds_p2

    def _line_to_line_distance(
        self,
        line1: tuple[float, float, float, float],
        line2: tuple[float, float, float, float],
    ) -> float:
        """Compute distance between two lines.

        Uses the average of point-to-line distances for both endpoints.

        Args:
            line1: First line (x1, y1, x2, y2).
            line2: Second line (x1, y1, x2, y2).

        Returns:
            Average distance between lines.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # Point-to-line distances from line1 endpoints to line2
        d1 = self._point_to_line_distance((x1, y1), (x3, y3), (x4, y4))
        d2 = self._point_to_line_distance((x2, y2), (x3, y3), (x4, y4))

        # Point-to-line distances from line2 endpoints to line1
        d3 = self._point_to_line_distance((x3, y3), (x1, y1), (x2, y2))
        d4 = self._point_to_line_distance((x4, y4), (x1, y1), (x2, y2))

        return (d1 + d2 + d3 + d4) / 4

    @staticmethod
    def _point_to_line_distance(
        point: tuple[float, float],
        line_p1: tuple[float, float],
        line_p2: tuple[float, float],
    ) -> float:
        """Compute distance from point to line.

        Args:
            point: Point (x, y).
            line_p1: First point on line.
            line_p2: Second point on line.

        Returns:
            Perpendicular distance from point to line.
        """
        px, py = point
        x1, y1 = line_p1
        x2, y2 = line_p2

        dx = x2 - x1
        dy = y2 - y1
        d_len = np.sqrt(dx * dx + dy * dy)

        if d_len < 1e-6:
            return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)

        # Cross product formula for point-to-line distance
        cross = abs((px - x1) * dy - (py - y1) * dx)
        return cross / d_len

    def compute_alignment_error(
        self,
        matches: list[dict[str, Any]],
    ) -> float:
        """Compute total alignment error for matched lines.

        Args:
            matches: List of matched lines from match_lines().

        Returns:
            Sum of squared distances for all matches.
        """
        if not matches:
            return 0.0

        total_error = 0.0
        for match in matches:
            total_error += match["distance"] ** 2

        return total_error

    def get_endpoint_keypoints(
        self,
        matches: list[dict[str, Any]],
        H: np.ndarray,
        max_distance: float = 15.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract Hough line endpoints as pseudo-keypoints.

        Projects Hough line endpoints to world coordinates using the template
        line as a guide. Only uses high-quality matches.

        Args:
            matches: List of matched lines from match_lines().
            H: Current homography (world -> image).
            max_distance: Maximum match distance to use.

        Returns:
            Tuple of (image_points, world_points) arrays.
        """
        img_pts = []
        world_pts = []

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        for match in matches:
            if match["distance"] > max_distance:
                continue

            hough_line = match["hough_line"]
            world_p1 = np.array(match["template_world_p1"])
            world_p2 = np.array(match["template_world_p2"])

            # Line direction vector
            line_vec = world_p2 - world_p1
            line_len = np.linalg.norm(line_vec)
            if line_len < 0.1:
                continue
            line_vec = line_vec / line_len

            # Process both Hough endpoints
            for h_x, h_y in [(hough_line["x1"], hough_line["y1"]),
                             (hough_line["x2"], hough_line["y2"])]:
                # Project Hough endpoint to world
                h_pt_h = np.array([h_x, h_y, 1.0])
                world_pt = H_inv @ h_pt_h
                if abs(world_pt[2]) < 1e-6:
                    continue
                world_pt = world_pt[:2] / world_pt[2]

                # Clamp to template line segment
                pt_vec = world_pt - world_p1
                proj_len = np.dot(pt_vec, line_vec)
                proj_len = np.clip(proj_len, 0, line_len)

                clamped_world = world_p1 + proj_len * line_vec

                img_pts.append([h_x, h_y])
                world_pts.append(clamped_world)

        if not img_pts:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)

        return np.array(img_pts), np.array(world_pts)

    def get_line_constraints(
        self,
        matches: list[dict[str, Any]],
        max_distance: float = 30.0,
        image_margin: float = 0.15,
        image_size: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        """Convert matched lines to optimization constraints.

        Args:
            matches: List of matched lines.
            max_distance: Maximum distance for a match to be used as constraint.
            image_margin: Margin from image edges to exclude (fraction of width/height).
                          Lines near edges often have worse calibration due to lens distortion.
            image_size: Image size (width, height) for margin filtering.

        Returns:
            List of constraint dictionaries with image and world line coords.
        """
        constraints = []
        for match in matches:
            # Only use high-quality matches
            if match["distance"] > max_distance:
                continue

            hough_line = match["hough_line"]

            # Filter out lines near image edges (often have lens distortion issues)
            if image_size is not None:
                w, h = image_size
                x_margin = w * image_margin
                y_margin = h * image_margin

                # Check if line midpoint is in the central region
                mid_x = (hough_line["x1"] + hough_line["x2"]) / 2
                mid_y = (hough_line["y1"] + hough_line["y2"]) / 2

                if mid_x < x_margin or mid_x > w - x_margin:
                    continue
                if mid_y < y_margin or mid_y > h - y_margin:
                    continue

            # Weight based on distance: closer matches get higher weight
            # At distance=0 -> weight=1.0, at distance=30 -> weight=0.25
            weight = 1.0 / (1.0 + match["distance"] / 10.0)

            constraints.append({
                "image_line": {
                    "x1": hough_line["x1"],
                    "y1": hough_line["y1"],
                    "x2": hough_line["x2"],
                    "y2": hough_line["y2"],
                },
                "world_line": {
                    "x1": match["template_world_p1"][0],
                    "y1": match["template_world_p1"][1],
                    "x2": match["template_world_p2"][0],
                    "y2": match["template_world_p2"][1],
                },
                "weight": weight,
            })
        return constraints
