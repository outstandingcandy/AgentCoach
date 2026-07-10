"""Keypoint mapping between PnLCalib (58) and SoccerNet-GSR (115) formats.

This module provides mapping utilities to convert between different keypoint
annotation formats used by various soccer field detection models.

PnLCalib uses 58 keypoints, while SoccerNet-GSR uses 115 keypoints.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class KeypointMapper:
    """Map keypoints between different annotation formats.

    Supports conversion between:
    - PnLCalib: 58 keypoints
    - SoccerNet-GSR: 115 keypoints

    The mapping is based on semantic correspondence of keypoint locations
    on the soccer field template.
    """

    # PnLCalib 57 keypoint world coordinates (0-indexed, centered at pitch center)
    # From PnLCalib/utils/utils_calib.py - keypoint_world_coords_2D
    # Original coords are in pitch coordinate (0,0 at top-left, 105x68, y-down)
    # Centered: x - 52.5, and y flipped: 34 - y (to convert y-down to y-up)
    PNLCALIB_WORLD_COORDS_2D = [
        [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
        [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
        [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
        [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
        [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
        [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
        [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
        [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
        [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
        [88.5, 34.], [94., 34.]
    ]  # 57 keypoints (indices 0-56)

    # Center the coordinates (pitch center at origin)
    PNLCALIB_WORLD_COORDS_2D = [[x - 52.5, 34 - y] for x, y in PNLCALIB_WORLD_COORDS_2D]

    # PnLCalib keypoint names (0-indexed)
    # After centering with 34-y, positive y = top (away from camera), negative y = bottom (near camera)
    PNLCALIB_KEYPOINTS = [
        (0, "top_left_corner"),              # [0, 0] -> [-52.5, 34]
        (1, "top_center"),                   # [52.5, 0] -> [0, 34]
        (2, "top_right_corner"),             # [105, 0] -> [52.5, 34]
        (3, "left_penalty_top_corner"),      # [0, 13.84] -> [-52.5, 20.16]
        (4, "left_penalty_top_front"),       # [16.5, 13.84] -> [-36, 20.16]
        (5, "right_penalty_top_front"),      # [88.5, 13.84] -> [36, 20.16]
        (6, "right_penalty_top_corner"),     # [105, 13.84] -> [52.5, 20.16]
        (7, "left_goal_area_top_corner"),    # [0, 24.84] -> [-52.5, 9.16]
        (8, "left_goal_area_top_front"),     # [5.5, 24.84] -> [-47, 9.16]
        (9, "right_goal_area_top_front"),    # [99.5, 24.84] -> [47, 9.16]
        (10, "right_goal_area_top_corner"),  # [105, 24.84] -> [52.5, 9.16]
        (11, "left_goal_top_post_ground"),   # [0, 30.34] -> [-52.5, 3.66]
        (12, "left_goal_top_post_top"),      # [0, 30.34] -> [-52.5, 3.66] (on crossbar)
        (13, "right_goal_top_post_ground"),  # [105, 30.34] -> [52.5, 3.66]
        (14, "right_goal_top_post_top"),     # [105, 30.34] -> [52.5, 3.66]
        (15, "left_goal_bottom_post_ground"),  # [0, 37.66] -> [-52.5, -3.66]
        (16, "left_goal_bottom_post_top"),     # [0, 37.66] -> [-52.5, -3.66]
        (17, "right_goal_bottom_post_ground"), # [105, 37.66] -> [52.5, -3.66]
        (18, "right_goal_bottom_post_top"),    # [105, 37.66] -> [52.5, -3.66]
        (19, "left_goal_area_bottom_corner"),  # [0, 43.16] -> [-52.5, -9.16]
        (20, "left_goal_area_bottom_front"),   # [5.5, 43.16] -> [-47, -9.16]
        (21, "right_goal_area_bottom_front"),  # [99.5, 43.16] -> [47, -9.16]
        (22, "right_goal_area_bottom_corner"), # [105, 43.16] -> [52.5, -9.16]
        (23, "left_penalty_bottom_corner"),    # [0, 54.16] -> [-52.5, -20.16]
        (24, "left_penalty_bottom_front"),     # [16.5, 54.16] -> [-36, -20.16]
        (25, "right_penalty_bottom_front"),    # [88.5, 54.16] -> [36, -20.16]
        (26, "right_penalty_bottom_corner"),   # [105, 54.16] -> [52.5, -20.16]
        (27, "bottom_left_corner"),            # [0, 68] -> [-52.5, -34]
        (28, "bottom_center"),                 # [52.5, 68] -> [0, -34]
        (29, "bottom_right_corner"),           # [105, 68] -> [52.5, -34]
        (30, "left_penalty_arc_top"),          # [16.5, 26.68] -> [-36, 7.32]
        (31, "center_circle_top"),             # [52.5, 24.85] -> [0, 9.15]
        (32, "right_penalty_arc_top"),         # [88.5, 26.68] -> [36, 7.32]
        (33, "left_penalty_arc_bottom"),       # [16.5, 41.31] -> [-36, -7.31]
        (34, "center_circle_bottom"),          # [52.5, 43.15] -> [0, -9.15]
        (35, "right_penalty_arc_bottom"),      # [88.5, 41.31] -> [36, -7.31]
        (36, "left_penalty_arc_inner_top"),    # [19.99, 32.29] -> [-32.51, 1.71]
        (37, "center_circle_left_top"),        # [43.68, 31.53] -> [-8.82, 2.47]
        (38, "center_circle_right_top"),       # [61.31, 31.53] -> [8.81, 2.47]
        (39, "right_penalty_arc_inner_top"),   # [85, 32.29] -> [32.5, 1.71]
        (40, "left_penalty_arc_inner_bottom"), # [19.99, 35.7] -> [-32.51, -1.7]
        (41, "center_circle_left_bottom"),     # [43.68, 36.46] -> [-8.82, -2.46]
        (42, "center_circle_right_bottom"),    # [61.31, 36.46] -> [8.81, -2.46]
        (43, "right_penalty_arc_inner_bottom"),# [85, 35.7] -> [32.5, -1.7]
        (44, "left_penalty_spot"),             # [11, 34] -> [-41.5, 0]
        (45, "left_penalty_arc_center"),       # [16.5, 34] -> [-36, 0]
        (46, "left_penalty_arc_right"),        # [20.15, 34] -> [-32.35, 0]
        (47, "center_circle_top_left"),        # [46.03, 27.53] -> [-6.47, 6.47]
        (48, "center_circle_top_right"),       # [58.97, 27.53] -> [6.47, 6.47]
        (49, "center_circle_left"),            # [43.35, 34] -> [-9.15, 0]
        (50, "center_spot"),                   # [52.5, 34] -> [0, 0]
        (51, "center_circle_right"),           # [61.5, 34] -> [9, 0]
        (52, "center_circle_bottom_left"),     # [46.03, 40.47] -> [-6.47, -6.47]
        (53, "center_circle_bottom_right"),    # [58.97, 40.47] -> [6.47, -6.47]
        (54, "right_penalty_arc_left"),        # [84.85, 34] -> [32.35, 0]
        (55, "right_penalty_arc_center"),      # [88.5, 34] -> [36, 0]
        (56, "right_penalty_spot"),            # [94, 34] -> [41.5, 0]
    ]

    # Mapping from PnLCalib keypoint ID to SoccerNet-GSR keypoint ID
    # Based on semantic correspondence of keypoint names and locations
    PNLCALIB_TO_SOCCERNET: dict[int, int] = {
        # Pitch corners
        0: 11,   # left_top_corner -> left_touchline_top
        1: 13,   # right_top_corner -> right_touchline_top
        2: 12,   # left_bottom_corner -> left_touchline_bottom
        3: 14,   # right_bottom_corner -> right_touchline_bottom
        # Center line
        4: 9,    # center_top -> center_line_top
        5: 10,   # center_bottom -> center_line_bottom
        # Center circle
        6: 0,    # center_spot
        7: 1,    # center_circle_top
        8: 2,    # center_circle_bottom
        9: 3,    # center_circle_left
        10: 4,   # center_circle_right
        11: 5,   # center_circle_top_left
        12: 6,   # center_circle_top_right
        13: 7,   # center_circle_bottom_left
        14: 8,   # center_circle_bottom_right
        # Left penalty area
        15: 15,  # left_penalty_area_top_corner
        16: 16,  # left_penalty_area_bottom_corner
        17: 17,  # left_penalty_area_top_front
        18: 18,  # left_penalty_area_bottom_front
        19: 19,  # left_penalty_spot
        20: 20,  # left_penalty_arc_top
        21: 21,  # left_penalty_arc_bottom
        22: 22,  # left_penalty_arc_center
        # Left goal area
        23: 23,  # left_goal_area_top_corner
        24: 24,  # left_goal_area_bottom_corner
        25: 25,  # left_goal_area_top_front
        26: 26,  # left_goal_area_bottom_front
        # Left goal posts
        27: 27,  # left_goal_top_post
        28: 28,  # left_goal_bottom_post
        # Right penalty area
        29: 29,  # right_penalty_area_top_corner
        30: 30,  # right_penalty_area_bottom_corner
        31: 31,  # right_penalty_area_top_front
        32: 32,  # right_penalty_area_bottom_front
        33: 33,  # right_penalty_spot
        34: 34,  # right_penalty_arc_top
        35: 35,  # right_penalty_arc_bottom
        36: 36,  # right_penalty_arc_center
        # Right goal area
        37: 37,  # right_goal_area_top_corner
        38: 38,  # right_goal_area_bottom_corner
        39: 39,  # right_goal_area_top_front
        40: 40,  # right_goal_area_bottom_front
        # Right goal posts
        41: 41,  # right_goal_top_post
        42: 42,  # right_goal_bottom_post
        # Corner arcs
        43: 44,  # left_top_corner_arc_inner
        44: 45,  # left_top_corner_arc_outer
        45: 48,  # left_bottom_corner_arc_inner
        46: 49,  # left_bottom_corner_arc_outer
        47: 52,  # right_top_corner_arc_inner
        48: 53,  # right_top_corner_arc_outer
        49: 56,  # right_bottom_corner_arc_inner
        50: 57,  # right_bottom_corner_arc_outer
    }

    # Reverse mapping
    SOCCERNET_TO_PNLCALIB: dict[int, int] = {
        v: k for k, v in PNLCALIB_TO_SOCCERNET.items()
    }

    def __init__(self):
        """Initialize keypoint mapper."""
        # Build world coordinate lookup from PnLCalib definitions (0-indexed)
        self._pnlcalib_world_coords = {}
        for i, coords in enumerate(self.PNLCALIB_WORLD_COORDS_2D):
            self._pnlcalib_world_coords[i] = (coords[0], coords[1])

    def pnlcalib_to_soccernet(
        self,
        keypoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert PnLCalib keypoints to SoccerNet-GSR format.

        Args:
            keypoints: List of keypoint dicts with 'id', 'x', 'y', 'confidence'.
                      IDs should be in PnLCalib format (0-57).

        Returns:
            List of keypoints with IDs converted to SoccerNet-GSR format (0-114).
        """
        converted = []
        for kp in keypoints:
            pnlcalib_id = kp["id"]
            if pnlcalib_id in self.PNLCALIB_TO_SOCCERNET:
                soccernet_id = self.PNLCALIB_TO_SOCCERNET[pnlcalib_id]
                converted.append({
                    "id": soccernet_id,
                    "x": kp["x"],
                    "y": kp["y"],
                    "confidence": kp.get("confidence", 1.0),
                    "pnlcalib_id": pnlcalib_id,  # Keep original ID for reference
                })
        return converted

    def soccernet_to_pnlcalib(
        self,
        keypoints: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert SoccerNet-GSR keypoints to PnLCalib format.

        Args:
            keypoints: List of keypoint dicts with IDs in SoccerNet-GSR format.

        Returns:
            List of keypoints with IDs converted to PnLCalib format.
        """
        converted = []
        for kp in keypoints:
            soccernet_id = kp["id"]
            if soccernet_id in self.SOCCERNET_TO_PNLCALIB:
                pnlcalib_id = self.SOCCERNET_TO_PNLCALIB[soccernet_id]
                converted.append({
                    "id": pnlcalib_id,
                    "x": kp["x"],
                    "y": kp["y"],
                    "confidence": kp.get("confidence", 1.0),
                    "soccernet_id": soccernet_id,
                })
        return converted

    def get_world_coordinates(
        self,
        keypoint_id: int,
        format: str = "pnlcalib",
    ) -> tuple[float, float] | None:
        """Get world coordinates for a keypoint.

        Args:
            keypoint_id: Keypoint ID.
            format: Either "pnlcalib" or "soccernet".

        Returns:
            (x, y) world coordinates in meters, or None if not found.
        """
        if format == "pnlcalib":
            return self._pnlcalib_world_coords.get(keypoint_id)
        elif format == "soccernet":
            # Convert to PnLCalib ID first
            if keypoint_id in self.SOCCERNET_TO_PNLCALIB:
                pnlcalib_id = self.SOCCERNET_TO_PNLCALIB[keypoint_id]
                return self._pnlcalib_world_coords.get(pnlcalib_id)
        return None

    # Non-ground keypoint IDs (goal post tops on crossbar, z != 0)
    # These should be excluded from homography calculation as they are not on the ground plane
    NON_GROUND_KEYPOINTS = {
        12,  # left_goal_bottom_post_top (on crossbar)
        14,  # right_goal_bottom_post_top (on crossbar)
        16,  # left_goal_top_post_top (on crossbar)
        18,  # right_goal_top_post_top (on crossbar)
    }

    # Height of goal crossbar (z-positive = up; matches solvePnP's own
    # world-frame freedom, and goalinsight.annotation.pitch.geometry).
    GOAL_CROSSBAR_HEIGHT = 2.44

    # Edge keypoint IDs (near goal lines, affected by lens distortion)
    # These should be excluded when camera has wide-angle lens distortion
    EDGE_KEYPOINTS = {
        # Left goal line area (x = -52.5)
        0,   # bottom_left_corner
        3,   # left_penalty_bottom_corner
        7,   # left_goal_area_bottom_corner
        11,  # left_goal_bottom_post_ground
        15,  # left_goal_top_post_ground
        19,  # left_goal_area_top_corner
        23,  # left_penalty_top_corner
        27,  # top_left_corner
        # Right goal line area (x = 52.5)
        2,   # bottom_right_corner
        6,   # right_penalty_bottom_corner
        10,  # right_goal_area_bottom_corner
        13,  # right_goal_bottom_post_ground
        17,  # right_goal_top_post_ground
        22,  # right_goal_area_top_corner
        26,  # right_penalty_top_corner
        29,  # top_right_corner
    }

    def build_correspondence_matrix(
        self,
        pnlcalib_keypoints: list[dict[str, Any]],
        filter_by_confidence: float = 0.0,
        exclude_non_ground: bool = True,
        exclude_edge: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Build 2D-3D correspondence arrays for PnP solving.

        Args:
            pnlcalib_keypoints: Detected keypoints in PnLCalib format.
            filter_by_confidence: Minimum confidence to include.
            exclude_non_ground: If True, exclude goal post top keypoints that
                               are not on the ground plane (z != 0). Default True.
            exclude_edge: If True, exclude keypoints near goal lines that are
                         affected by lens distortion. Default False.

        Returns:
            Tuple of:
                - image_points: 2D image coordinates (N, 2)
                - world_points: 3D world coordinates (N, 3) with z=0
                - keypoint_ids: List of keypoint IDs used
        """
        image_points = []
        world_points = []
        keypoint_ids = []

        for kp in pnlcalib_keypoints:
            if kp.get("confidence", 1.0) < filter_by_confidence:
                continue

            kp_id = kp["id"]

            # Skip non-ground keypoints (goal post tops) for homography
            if exclude_non_ground and kp_id in self.NON_GROUND_KEYPOINTS:
                continue

            # Skip edge keypoints (near goal lines, affected by lens distortion)
            if exclude_edge and kp_id in self.EDGE_KEYPOINTS:
                continue

            world_xy = self.get_world_coordinates(kp_id, format="pnlcalib")

            if world_xy is not None:
                image_points.append([kp["x"], kp["y"]])
                world_points.append([world_xy[0], world_xy[1], 0.0])
                keypoint_ids.append(kp_id)

        return (
            np.array(image_points, dtype=np.float32) if image_points else np.zeros((0, 2)),
            np.array(world_points, dtype=np.float32) if world_points else np.zeros((0, 3)),
            keypoint_ids,
        )

    def build_3d_correspondence_matrix(
        self,
        pnlcalib_keypoints: list[dict[str, Any]],
        filter_by_confidence: float = 0.3,
        exclude_non_ground: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Build 2D-3D correspondences including non-ground points with z=+2.44.

        Unlike build_correspondence_matrix, this includes crossbar keypoints
        with their actual 3D height for use with PnP solving.

        Args:
            pnlcalib_keypoints: Detected keypoints in PnLCalib format.
            filter_by_confidence: Minimum confidence to include.
            exclude_non_ground: If True, exclude crossbar points (same as
                build_correspondence_matrix). Default False to include them.

        Returns:
            Tuple of:
                - image_points: (N, 2)
                - world_points_3d: (N, 3) with z=+2.44 for crossbar, z=0 otherwise
                - keypoint_ids: List of keypoint IDs used
        """
        image_points = []
        world_points = []
        keypoint_ids = []

        for kp in pnlcalib_keypoints:
            if kp.get("confidence", 1.0) < filter_by_confidence:
                continue

            kp_id = kp["id"]

            if exclude_non_ground and kp_id in self.NON_GROUND_KEYPOINTS:
                continue

            world_xy = self.get_world_coordinates(kp_id, format="pnlcalib")
            if world_xy is None:
                continue

            z = self.GOAL_CROSSBAR_HEIGHT if kp_id in self.NON_GROUND_KEYPOINTS else 0.0
            image_points.append([kp["x"], kp["y"]])
            world_points.append([world_xy[0], world_xy[1], z])
            keypoint_ids.append(kp_id)

        return (
            np.array(image_points, dtype=np.float32) if image_points else np.zeros((0, 2)),
            np.array(world_points, dtype=np.float32) if world_points else np.zeros((0, 3)),
            keypoint_ids,
        )

    @classmethod
    def get_num_keypoints(cls, format: str = "pnlcalib") -> int:
        """Get the number of keypoints in a format.

        Args:
            format: Either "pnlcalib" or "soccernet".

        Returns:
            Number of keypoints.
        """
        if format == "pnlcalib":
            return 58
        elif format == "soccernet":
            return 115
        else:
            raise ValueError(f"Unknown format: {format}")

    @classmethod
    def get_keypoint_name(cls, keypoint_id: int, format: str = "pnlcalib") -> str:
        """Get the name of a keypoint by ID.

        Args:
            keypoint_id: Keypoint ID.
            format: Either "pnlcalib" or "soccernet".

        Returns:
            Keypoint name or "unknown".
        """
        if format == "pnlcalib":
            for kp in cls.PNLCALIB_KEYPOINTS:
                if kp[0] == keypoint_id:
                    return kp[1]
        return f"keypoint_{keypoint_id}"
