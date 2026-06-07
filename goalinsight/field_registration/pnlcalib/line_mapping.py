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

# FIFA defaults — used only as a fallback when no pitch_dims is passed.
# Live consumers (kids / 7-a-side / etc.) override per-instance.
_FIFA_DEFAULTS: dict[str, float] = {
    "pitch_length": 105.0,
    "pitch_width": 68.0,
    "penalty_area_length": 16.5,
    "penalty_area_width": 40.32,
    "goal_area_length": 5.5,
    "goal_area_width": 18.32,
    "goal_length": 7.32,
    "goal_height": 2.44,
}


def _build_line_definitions(pitch_dims: dict | None) -> dict[int, dict[str, Any]]:
    """Materialize the 23-line table for a given pitch geometry."""
    d = dict(_FIFA_DEFAULTS)
    if pitch_dims:
        d.update({k: v for k, v in pitch_dims.items() if k in _FIFA_DEFAULTS})
    hl = d["pitch_length"] / 2.0
    hw = d["pitch_width"] / 2.0
    pa_d = d["penalty_area_length"]
    pa_hw = d["penalty_area_width"] / 2.0
    ga_d = d["goal_area_length"]
    ga_hw = d["goal_area_width"] / 2.0
    g_hw = d["goal_length"] / 2.0
    g_h = d["goal_height"]

    return {
        # Penalty area (Big rect.) - Left
        0: {"name": "Big rect. left top",
            "p1": (-hl, pa_hw, 0), "p2": (-hl + pa_d, pa_hw, 0)},
        1: {"name": "Big rect. left side",
            "p1": (-hl + pa_d, pa_hw, 0), "p2": (-hl + pa_d, -pa_hw, 0)},
        2: {"name": "Big rect. left bottom",
            "p1": (-hl + pa_d, -pa_hw, 0), "p2": (-hl, -pa_hw, 0)},
        # Penalty area (Big rect.) - Right
        3: {"name": "Big rect. right top",
            "p1": (hl, pa_hw, 0), "p2": (hl - pa_d, pa_hw, 0)},
        4: {"name": "Big rect. right side",
            "p1": (hl - pa_d, pa_hw, 0), "p2": (hl - pa_d, -pa_hw, 0)},
        5: {"name": "Big rect. right bottom",
            "p1": (hl - pa_d, -pa_hw, 0), "p2": (hl, -pa_hw, 0)},
        # Goal frame - Left
        6: {"name": "Goal left crossbar",
            "p1": (-hl, -g_hw, g_h), "p2": (-hl, g_hw, g_h)},
        7: {"name": "Goal left post left",
            "p1": (-hl, -g_hw, 0), "p2": (-hl, -g_hw, g_h)},
        8: {"name": "Goal left post right",
            "p1": (-hl, g_hw, 0), "p2": (-hl, g_hw, g_h)},
        # Goal frame - Right
        9: {"name": "Goal right crossbar",
            "p1": (hl, -g_hw, g_h), "p2": (hl, g_hw, g_h)},
        10: {"name": "Goal right post left",
             "p1": (hl, -g_hw, 0), "p2": (hl, -g_hw, g_h)},
        11: {"name": "Goal right post right",
             "p1": (hl, g_hw, 0), "p2": (hl, g_hw, g_h)},
        # Center line
        12: {"name": "Middle line",
             "p1": (0, -hw, 0), "p2": (0, hw, 0)},
        # Touchlines / goal lines
        13: {"name": "Side line top",
             "p1": (-hl, hw, 0), "p2": (hl, hw, 0)},
        14: {"name": "Side line left",
             "p1": (-hl, -hw, 0), "p2": (-hl, hw, 0)},
        15: {"name": "Side line right",
             "p1": (hl, -hw, 0), "p2": (hl, hw, 0)},
        16: {"name": "Side line bottom",
             "p1": (-hl, -hw, 0), "p2": (hl, -hw, 0)},
        # Goal area (Small rect.) - Left
        17: {"name": "Small rect. left top",
             "p1": (-hl, ga_hw, 0), "p2": (-hl + ga_d, ga_hw, 0)},
        18: {"name": "Small rect. left side",
             "p1": (-hl + ga_d, ga_hw, 0), "p2": (-hl + ga_d, -ga_hw, 0)},
        19: {"name": "Small rect. left bottom",
             "p1": (-hl + ga_d, -ga_hw, 0), "p2": (-hl, -ga_hw, 0)},
        # Goal area (Small rect.) - Right
        20: {"name": "Small rect. right top",
             "p1": (hl, ga_hw, 0), "p2": (hl - ga_d, ga_hw, 0)},
        21: {"name": "Small rect. right side",
             "p1": (hl - ga_d, ga_hw, 0), "p2": (hl - ga_d, -ga_hw, 0)},
        22: {"name": "Small rect. right bottom",
             "p1": (hl - ga_d, -ga_hw, 0), "p2": (hl, -ga_hw, 0)},
    }


# Default (FIFA) table — used by the `@classmethod` accessors that don't have
# an instance. Names and ground/non-ground membership are pitch-independent.
_DEFAULT_LINE_DEFINITIONS = _build_line_definitions(None)


# --- Legacy module-level constants (FIFA-only; kept for back-compat with --
# `hough_line_matcher.py` and any downstream code that still imports them).
# New code should call `_build_line_definitions(pitch_dims)` or instantiate
# ``LineMapper(pitch_dims=...)`` instead.
PITCH_LENGTH = _FIFA_DEFAULTS["pitch_length"]
PITCH_WIDTH = _FIFA_DEFAULTS["pitch_width"]
HALF_L = PITCH_LENGTH / 2
HALF_W = PITCH_WIDTH / 2
PENALTY_AREA_WIDTH = _FIFA_DEFAULTS["penalty_area_width"]
PENALTY_AREA_DEPTH = _FIFA_DEFAULTS["penalty_area_length"]
PENALTY_HALF_W = PENALTY_AREA_WIDTH / 2
GOAL_AREA_WIDTH = _FIFA_DEFAULTS["goal_area_width"]
GOAL_AREA_DEPTH = _FIFA_DEFAULTS["goal_area_length"]
GOAL_HALF_W = GOAL_AREA_WIDTH / 2
GOAL_WIDTH = _FIFA_DEFAULTS["goal_length"]
GOAL_HEIGHT = _FIFA_DEFAULTS["goal_height"]
GOAL_HALF_W_ACTUAL = GOAL_WIDTH / 2


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

    # The class-level table is the FIFA fallback; instances override it via
    # ``__init__`` when a non-default pitch is configured. Both names and
    # ground/non-ground membership are pitch-independent — only the (x, y, z)
    # endpoints depend on the active pitch geometry.
    LINE_DEFINITIONS = _DEFAULT_LINE_DEFINITIONS

    # Lines that are NOT on the ground plane (z != 0): goal crossbars + posts.
    NON_GROUND_LINES = {6, 7, 8, 9, 10, 11}
    GROUND_LINES = set(range(23)) - NON_GROUND_LINES

    def __init__(self, pitch_dims: dict | None = None):
        """Initialize line mapper.

        Args:
            pitch_dims: Pitch geometry overrides (FIFA when omitted).
                Recognized keys mirror :data:`_FIFA_DEFAULTS`. Drives
                ``self.LINE_DEFINITIONS`` so non-FIFA pitches get correct
                world endpoints — the class-level attribute is only used
                when no instance is available (e.g. classmethod accessors).
        """
        # Shadow the class attribute on this instance only.
        self.LINE_DEFINITIONS = _build_line_definitions(pitch_dims)

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
