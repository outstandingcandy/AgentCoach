"""Line mapping for PnLCalib soccer field lines.

This module provides the mapping between PnLCalib's 23 line classes
and their 3D world coordinates on a standard soccer pitch.

Reference:
    PnLCalib uses 23 line classes for soccer field detection.
    Each line is defined by two 3D endpoints in world coordinates.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Standard FIFA pitch dimensions (in meters, centered at origin)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_L = PITCH_LENGTH / 2  # 52.5
HALF_W = PITCH_WIDTH / 2   # 34.0

# Penalty area dimensions
PENALTY_AREA_WIDTH = 40.32  # 16.5m from each side of goal
PENALTY_AREA_DEPTH = 16.5
PENALTY_HALF_W = PENALTY_AREA_WIDTH / 2  # 20.16

# Goal area dimensions
GOAL_AREA_WIDTH = 18.32  # 5.5m from each side of goal
GOAL_AREA_DEPTH = 5.5
GOAL_HALF_W = GOAL_AREA_WIDTH / 2  # 9.16

# Goal dimensions
GOAL_WIDTH = 7.32
GOAL_HEIGHT = 2.44
GOAL_HALF_W_ACTUAL = GOAL_WIDTH / 2  # 3.66


class LineMapper:
    """Maps PnLCalib line class IDs to 3D world coordinates.

    PnLCalib defines 23 line classes. Each line has two 3D endpoints
    in world coordinates (x, y, z) where z=0 for ground lines.

    Line classes:
        0-5: Penalty area lines (Big rect.)
        6-11: Goal lines (crossbar and posts)
        12: Center line (Middle line)
        13-16: Touchlines and goal lines (Side lines)
        17-22: Goal area lines (Small rect.)
    """

    # Line definitions: {id: {"name": str, "p1": (x, y, z), "p2": (x, y, z)}}
    # Coordinates are in meters, origin at center of pitch
    # x-axis: along length (positive = right goal)
    # y-axis: along width (positive = top)
    # z-axis: vertical (positive = up)

    LINE_DEFINITIONS = {
        # Penalty area (Big rect.) - Left side (x = -HALF_L)
        0: {  # Big rect. left top
            "name": "Big rect. left top",
            "p1": (-HALF_L, PENALTY_HALF_W, 0),
            "p2": (-HALF_L + PENALTY_AREA_DEPTH, PENALTY_HALF_W, 0),
        },
        1: {  # Big rect. left side
            "name": "Big rect. left side",
            "p1": (-HALF_L + PENALTY_AREA_DEPTH, PENALTY_HALF_W, 0),
            "p2": (-HALF_L + PENALTY_AREA_DEPTH, -PENALTY_HALF_W, 0),
        },
        2: {  # Big rect. left bottom
            "name": "Big rect. left bottom",
            "p1": (-HALF_L + PENALTY_AREA_DEPTH, -PENALTY_HALF_W, 0),
            "p2": (-HALF_L, -PENALTY_HALF_W, 0),
        },

        # Penalty area (Big rect.) - Right side (x = +HALF_L)
        3: {  # Big rect. right top
            "name": "Big rect. right top",
            "p1": (HALF_L, PENALTY_HALF_W, 0),
            "p2": (HALF_L - PENALTY_AREA_DEPTH, PENALTY_HALF_W, 0),
        },
        4: {  # Big rect. right side
            "name": "Big rect. right side",
            "p1": (HALF_L - PENALTY_AREA_DEPTH, PENALTY_HALF_W, 0),
            "p2": (HALF_L - PENALTY_AREA_DEPTH, -PENALTY_HALF_W, 0),
        },
        5: {  # Big rect. right bottom
            "name": "Big rect. right bottom",
            "p1": (HALF_L - PENALTY_AREA_DEPTH, -PENALTY_HALF_W, 0),
            "p2": (HALF_L, -PENALTY_HALF_W, 0),
        },

        # Goal posts and crossbar - Left goal (x = -HALF_L)
        6: {  # Goal left crossbar (z = GOAL_HEIGHT)
            "name": "Goal left crossbar",
            "p1": (-HALF_L, -GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
            "p2": (-HALF_L, GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },
        7: {  # Goal left post left (near y = -GOAL_HALF_W_ACTUAL)
            "name": "Goal left post left",
            "p1": (-HALF_L, -GOAL_HALF_W_ACTUAL, 0),
            "p2": (-HALF_L, -GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },
        8: {  # Goal left post right (near y = +GOAL_HALF_W_ACTUAL)
            "name": "Goal left post right",
            "p1": (-HALF_L, GOAL_HALF_W_ACTUAL, 0),
            "p2": (-HALF_L, GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },

        # Goal posts and crossbar - Right goal (x = +HALF_L)
        9: {  # Goal right crossbar (z = GOAL_HEIGHT)
            "name": "Goal right crossbar",
            "p1": (HALF_L, -GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
            "p2": (HALF_L, GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },
        10: {  # Goal right post left
            "name": "Goal right post left",
            "p1": (HALF_L, -GOAL_HALF_W_ACTUAL, 0),
            "p2": (HALF_L, -GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },
        11: {  # Goal right post right
            "name": "Goal right post right",
            "p1": (HALF_L, GOAL_HALF_W_ACTUAL, 0),
            "p2": (HALF_L, GOAL_HALF_W_ACTUAL, GOAL_HEIGHT),
        },

        # Center line
        12: {  # Middle line (center line at x=0)
            "name": "Middle line",
            "p1": (0, -HALF_W, 0),
            "p2": (0, HALF_W, 0),
        },

        # Touchlines and goal lines (Side lines)
        13: {  # Side line top (y = +HALF_W)
            "name": "Side line top",
            "p1": (-HALF_L, HALF_W, 0),
            "p2": (HALF_L, HALF_W, 0),
        },
        14: {  # Side line left (goal line, x = -HALF_L)
            "name": "Side line left",
            "p1": (-HALF_L, -HALF_W, 0),
            "p2": (-HALF_L, HALF_W, 0),
        },
        15: {  # Side line right (goal line, x = +HALF_L)
            "name": "Side line right",
            "p1": (HALF_L, -HALF_W, 0),
            "p2": (HALF_L, HALF_W, 0),
        },
        16: {  # Side line bottom (y = -HALF_W)
            "name": "Side line bottom",
            "p1": (-HALF_L, -HALF_W, 0),
            "p2": (HALF_L, -HALF_W, 0),
        },

        # Goal area (Small rect.) - Left side
        17: {  # Small rect. left top
            "name": "Small rect. left top",
            "p1": (-HALF_L, GOAL_HALF_W, 0),
            "p2": (-HALF_L + GOAL_AREA_DEPTH, GOAL_HALF_W, 0),
        },
        18: {  # Small rect. left side
            "name": "Small rect. left side",
            "p1": (-HALF_L + GOAL_AREA_DEPTH, GOAL_HALF_W, 0),
            "p2": (-HALF_L + GOAL_AREA_DEPTH, -GOAL_HALF_W, 0),
        },
        19: {  # Small rect. left bottom
            "name": "Small rect. left bottom",
            "p1": (-HALF_L + GOAL_AREA_DEPTH, -GOAL_HALF_W, 0),
            "p2": (-HALF_L, -GOAL_HALF_W, 0),
        },

        # Goal area (Small rect.) - Right side
        20: {  # Small rect. right top
            "name": "Small rect. right top",
            "p1": (HALF_L, GOAL_HALF_W, 0),
            "p2": (HALF_L - GOAL_AREA_DEPTH, GOAL_HALF_W, 0),
        },
        21: {  # Small rect. right side
            "name": "Small rect. right side",
            "p1": (HALF_L - GOAL_AREA_DEPTH, GOAL_HALF_W, 0),
            "p2": (HALF_L - GOAL_AREA_DEPTH, -GOAL_HALF_W, 0),
        },
        22: {  # Small rect. right bottom
            "name": "Small rect. right bottom",
            "p1": (HALF_L - GOAL_AREA_DEPTH, -GOAL_HALF_W, 0),
            "p2": (HALF_L, -GOAL_HALF_W, 0),
        },
    }

    # Lines that are NOT on the ground plane (z != 0)
    # These include crossbars and goal posts
    NON_GROUND_LINES = {6, 7, 8, 9, 10, 11}  # Goal crossbars and posts

    # Ground-only lines (z = 0 for both endpoints)
    GROUND_LINES = set(range(23)) - NON_GROUND_LINES

    def __init__(self):
        """Initialize line mapper."""
        pass

    def get_line_world_coords(self, line_id: int) -> dict[str, Any] | None:
        """Get world coordinates for a line by its ID.

        Args:
            line_id: PnLCalib line class ID (0-22).

        Returns:
            Dictionary with line name and 3D endpoints, or None if invalid ID.
        """
        if line_id not in self.LINE_DEFINITIONS:
            return None

        line_def = self.LINE_DEFINITIONS[line_id]
        return {
            "id": line_id,
            "name": line_def["name"],
            "x1": line_def["p1"][0],
            "y1": line_def["p1"][1],
            "z1": line_def["p1"][2],
            "x2": line_def["p2"][0],
            "y2": line_def["p2"][1],
            "z2": line_def["p2"][2],
        }

    def build_line_correspondence(
        self,
        detected_lines: list[dict[str, Any]],
        filter_by_confidence: float = 0.0,
        exclude_non_ground: bool = False,
        min_length: float = 50.0,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build correspondence between detected lines and world lines.

        Args:
            detected_lines: List of detected lines from HRNet model.
                Each has: id, x1, y1, x2, y2, confidence (optional).
            filter_by_confidence: Minimum confidence threshold.
            exclude_non_ground: If True, exclude goal posts and crossbars.
            min_length: Minimum line length in pixels. Lines shorter than this
                       are filtered out as they provide unreliable constraints.

        Returns:
            Tuple of (image_lines, world_lines) with matching indices.
        """
        image_lines = []
        world_lines = []

        for line in detected_lines:
            line_id = line.get("id", -1)

            # Check confidence
            confidence = line.get("confidence", 1.0)
            if confidence < filter_by_confidence:
                continue

            # Check if non-ground line should be excluded
            if exclude_non_ground and line_id in self.NON_GROUND_LINES:
                continue

            # Check minimum line length
            dx = line["x2"] - line["x1"]
            dy = line["y2"] - line["y1"]
            length = (dx * dx + dy * dy) ** 0.5
            if length < min_length:
                continue

            # Get world coordinates
            world_line = self.get_line_world_coords(line_id)
            if world_line is None:
                continue

            image_lines.append({
                "id": line_id,
                "x1": line["x1"],
                "y1": line["y1"],
                "x2": line["x2"],
                "y2": line["y2"],
                "confidence": confidence,
            })
            world_lines.append(world_line)

        return image_lines, world_lines

    def get_line_endpoints_as_keypoints(
        self,
        detected_lines: list[dict[str, Any]],
        filter_by_confidence: float = 0.0,
        exclude_non_ground: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Convert line endpoints to keypoint format for homography.

        This extracts the endpoints of detected lines and treats them
        as additional keypoints for homography estimation.

        Args:
            detected_lines: List of detected lines.
            filter_by_confidence: Minimum confidence threshold.
            exclude_non_ground: If True, exclude non-ground lines.

        Returns:
            Tuple of (image_points, world_points, point_names).
            - image_points: (N, 2) array of image coordinates
            - world_points: (N, 3) array of world coordinates (x, y, z)
            - point_names: List of descriptive names
        """
        image_points = []
        world_points = []
        point_names = []

        for line in detected_lines:
            line_id = line.get("id", -1)

            # Check confidence
            confidence = line.get("confidence", 1.0)
            if confidence < filter_by_confidence:
                continue

            # Check if non-ground
            if exclude_non_ground and line_id in self.NON_GROUND_LINES:
                continue

            # Get world coordinates
            world_line = self.get_line_world_coords(line_id)
            if world_line is None:
                continue

            # Add endpoint 1
            image_points.append([line["x1"], line["y1"]])
            world_points.append([world_line["x1"], world_line["y1"], world_line["z1"]])
            point_names.append(f"{world_line['name']}_p1")

            # Add endpoint 2
            image_points.append([line["x2"], line["y2"]])
            world_points.append([world_line["x2"], world_line["y2"], world_line["z2"]])
            point_names.append(f"{world_line['name']}_p2")

        if not image_points:
            return np.array([]).reshape(0, 2), np.array([]).reshape(0, 3), []

        return (
            np.array(image_points, dtype=np.float64),
            np.array(world_points, dtype=np.float64),
            point_names,
        )

    @classmethod
    def get_line_name(cls, line_id: int) -> str:
        """Get human-readable name for a line class.

        Args:
            line_id: Line class ID.

        Returns:
            Line class name or "Unknown".
        """
        if line_id in cls.LINE_DEFINITIONS:
            return cls.LINE_DEFINITIONS[line_id]["name"]
        return f"Unknown line {line_id}"

    @classmethod
    def is_ground_line(cls, line_id: int) -> bool:
        """Check if a line is on the ground plane.

        Args:
            line_id: Line class ID.

        Returns:
            True if the line is on the ground (z=0).
        """
        return line_id in cls.GROUND_LINES
