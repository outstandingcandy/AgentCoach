"""Physical camera calibrator with 7-DOF optimization.

The input images are assumed to be already undistorted, so distortion coefficients
are fixed at zero and the principal point is fixed at the image center.
Only 7 parameters are optimized: rvec(3), tvec(3), and focal length f.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares

from ..utils.projection import project_points_2d
from .pnlcalib.curve_utils import (
    compute_cumulated_lengths,
    interpolate_on_polyline,
    sample_points_on_image_line,
)
from .pnlcalib.line_mapping import LineMapper

logger = logging.getLogger(__name__)

# Ground-only line IDs (exclude crossbars and goal posts)
GROUND_LINE_IDS = set(range(23)) - {6, 7, 8, 9, 10, 11}


def build_field_template(
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> tuple[list[list[float]], dict, dict]:
    """Build field template coordinates for given pitch dimensions.

    Penalty area (16.5m × 40.32m), goal area (5.5m × 18.32m), goal (7.32m × 2.44m),
    penalty spot (11m), and center circle (9.15m radius) are FIFA standard and do not
    change with pitch size.

    Args:
        pitch_length: Pitch length in meters (default 105 = FIFA).
        pitch_width: Pitch width in meters (default 68 = FIFA).

    Returns:
        Tuple of (world_coords_2d, line_definitions, line_intersections).
        - world_coords_2d: list of [x, y] centered at pitch center (57 keypoints)
        - line_definitions: dict matching LineMapper.LINE_DEFINITIONS format
        - line_intersections: dict matching LINE_INTERSECTIONS format
    """
    HL = pitch_length / 2
    HW = pitch_width / 2

    # Fixed marking dimensions (FIFA standard)
    PA_DEPTH = 16.5
    PA_HW = 20.16   # penalty area half-width
    GA_DEPTH = 5.5
    GA_HW = 9.16    # goal area half-width
    G_HW = 3.66     # goal half-width
    G_H = 2.44      # goal height
    PS_DIST = 11.0   # penalty spot from goal line
    CR = 9.15        # center circle radius

    # --- Raw coordinates in pitch-local system (0,0 at top-left, y-down) ---
    # Then centered: x - HL, HW - y
    pa_top_y = HW - PA_HW
    pa_bot_y = HW + PA_HW
    ga_top_y = HW - GA_HW
    ga_bot_y = HW + GA_HW
    g_top_y = HW - G_HW
    g_bot_y = HW + G_HW

    # Penalty arc points (radius CR from penalty spot, at penalty area front line x)
    # Left penalty spot at (PS_DIST, HW), arc top: x=PA_DEPTH
    arc_dy = (CR**2 - (PA_DEPTH - PS_DIST)**2) ** 0.5
    # Center circle points (radius CR from center)
    # Top/bottom: at y = HW ± CR
    # Left/right: at x = HL ± CR
    # Diagonal points at 45°: offset = CR * cos(45°) ≈ CR / sqrt(2)
    cc_diag = CR / (2**0.5)

    raw = [
        # 0-2: pitch top edge
        [0., 0.], [HL, 0.], [pitch_length, 0.],
        # 3-6: penalty area top
        [0., pa_top_y], [PA_DEPTH, pa_top_y],
        [pitch_length - PA_DEPTH, pa_top_y], [pitch_length, pa_top_y],
        # 7-10: goal area top
        [0., ga_top_y], [GA_DEPTH, ga_top_y],
        [pitch_length - GA_DEPTH, ga_top_y], [pitch_length, ga_top_y],
        # 11-14: goal top posts (ground + crossbar top, same x/y)
        [0., g_top_y], [0., g_top_y],
        [pitch_length, g_top_y], [pitch_length, g_top_y],
        # 15-18: goal bottom posts
        [0., g_bot_y], [0., g_bot_y],
        [pitch_length, g_bot_y], [pitch_length, g_bot_y],
        # 19-22: goal area bottom
        [0., ga_bot_y], [GA_DEPTH, ga_bot_y],
        [pitch_length - GA_DEPTH, ga_bot_y], [pitch_length, ga_bot_y],
        # 23-26: penalty area bottom
        [0., pa_bot_y], [PA_DEPTH, pa_bot_y],
        [pitch_length - PA_DEPTH, pa_bot_y], [pitch_length, pa_bot_y],
        # 27-29: pitch bottom edge
        [0., pitch_width], [HL, pitch_width], [pitch_length, pitch_width],
        # 30-35: penalty arc + center circle top/bottom (symmetric)
        [PA_DEPTH, HW - arc_dy],             # 30: left penalty arc top
        [HL, HW - CR],                       # 31: center circle top
        [pitch_length - PA_DEPTH, HW - arc_dy],  # 32: right penalty arc top
        [PA_DEPTH, HW + arc_dy],             # 33: left penalty arc bottom
        [HL, HW + CR],                       # 34: center circle bottom
        [pitch_length - PA_DEPTH, HW + arc_dy],  # 35: right penalty arc bottom
        # 36-43: penalty arc inner + center circle diagonal points
        [PS_DIST + CR * (PA_DEPTH - PS_DIST) / CR, HW - CR * arc_dy / CR],
        # ^ simplification: just use actual geometry
    ]
    # Actually, the circle/arc diagonal points are complex. Let me compute them properly.
    # Points 36-43 are at specific angles on their respective circles.
    # Use the original offsets from circle centers (these don't depend on pitch size).

    # Offsets of arc/circle points from their centers (from original 105×68 coords):
    # Left penalty arc center = (11, 34) in old coords
    # Original kp36 = (19.99, 32.29) → offset from (11, 34) = (8.99, -1.71)
    # Original kp40 = (19.99, 35.7)  → offset from (11, 34) = (8.99, 1.7)
    # Center circle center = (52.5, 34)
    # Original kp37 = (43.68, 31.53) → offset from (52.5, 34) = (-8.82, -2.47)
    # Original kp38 = (61.31, 31.53) → offset = (8.81, -2.47)
    # Original kp41 = (43.68, 36.46) → offset = (-8.82, 2.46)
    # Original kp42 = (61.31, 36.46) → offset = (8.81, 2.46)
    # Right penalty arc center = (94, 34) in old coords
    # Original kp39 = (85, 32.29)    → offset from (94, 34) = (-9, -1.71)
    # Original kp43 = (85, 35.7)     → offset from (94, 34) = (-9, 1.7)

    # Circle point offsets (fixed, radius-dependent only)
    left_ps = [PS_DIST, HW]
    right_ps = [pitch_length - PS_DIST, HW]
    center = [HL, HW]

    # Offsets from original data (invariant to pitch size)
    lpa_offsets = {
        36: (8.99, -1.71), 40: (8.99, 1.7),
        45: (5.5, 0.0), 46: (9.15, 0.0), 44: (0.0, 0.0),
    }
    cc_offsets = {
        37: (-8.82, -2.47), 38: (8.81, -2.47),
        41: (-8.82, 2.46), 42: (8.81, 2.46),
        47: (-6.47, -6.47), 48: (6.47, -6.47),
        49: (-9.15, 0.0), 50: (0.0, 0.0), 51: (9.0, 0.0),
        52: (-6.47, 6.47), 53: (6.47, 6.47),
        31: (0.0, -9.15), 34: (0.0, 9.15),
    }
    rpa_offsets = {
        39: (-9.0, -1.71), 43: (-9.0, 1.7),
        54: (-9.15, 0.0), 55: (-5.5, 0.0), 56: (0.0, 0.0),
    }

    # Build full raw coords list (57 keypoints)
    raw_coords = [
        # 0-29: field structure points
        [0., 0.], [HL, 0.], [pitch_length, 0.],
        [0., pa_top_y], [PA_DEPTH, pa_top_y],
        [pitch_length - PA_DEPTH, pa_top_y], [pitch_length, pa_top_y],
        [0., ga_top_y], [GA_DEPTH, ga_top_y],
        [pitch_length - GA_DEPTH, ga_top_y], [pitch_length, ga_top_y],
        [0., g_top_y], [0., g_top_y],
        [pitch_length, g_top_y], [pitch_length, g_top_y],
        [0., g_bot_y], [0., g_bot_y],
        [pitch_length, g_bot_y], [pitch_length, g_bot_y],
        [0., ga_bot_y], [GA_DEPTH, ga_bot_y],
        [pitch_length - GA_DEPTH, ga_bot_y], [pitch_length, ga_bot_y],
        [0., pa_bot_y], [PA_DEPTH, pa_bot_y],
        [pitch_length - PA_DEPTH, pa_bot_y], [pitch_length, pa_bot_y],
        [0., pitch_width], [HL, pitch_width], [pitch_length, pitch_width],
        # 30: left penalty arc top
        [PA_DEPTH, HW - arc_dy],
        # 31: center circle top
        [center[0] + cc_offsets[31][0], center[1] + cc_offsets[31][1]],
        # 32: right penalty arc top
        [pitch_length - PA_DEPTH, HW - arc_dy],
        # 33: left penalty arc bottom
        [PA_DEPTH, HW + arc_dy],
        # 34: center circle bottom
        [center[0] + cc_offsets[34][0], center[1] + cc_offsets[34][1]],
        # 35: right penalty arc bottom
        [pitch_length - PA_DEPTH, HW + arc_dy],
        # 36-43: arc/circle diagonal points
        [left_ps[0] + lpa_offsets[36][0], left_ps[1] + lpa_offsets[36][1]],
        [center[0] + cc_offsets[37][0], center[1] + cc_offsets[37][1]],
        [center[0] + cc_offsets[38][0], center[1] + cc_offsets[38][1]],
        [right_ps[0] + rpa_offsets[39][0], right_ps[1] + rpa_offsets[39][1]],
        [left_ps[0] + lpa_offsets[40][0], left_ps[1] + lpa_offsets[40][1]],
        [center[0] + cc_offsets[41][0], center[1] + cc_offsets[41][1]],
        [center[0] + cc_offsets[42][0], center[1] + cc_offsets[42][1]],
        [right_ps[0] + rpa_offsets[43][0], right_ps[1] + rpa_offsets[43][1]],
        # 44-46: left penalty area points
        [left_ps[0] + lpa_offsets[44][0], left_ps[1] + lpa_offsets[44][1]],
        [left_ps[0] + lpa_offsets[45][0], left_ps[1] + lpa_offsets[45][1]],
        [left_ps[0] + lpa_offsets[46][0], left_ps[1] + lpa_offsets[46][1]],
        # 47-53: center circle points
        [center[0] + cc_offsets[47][0], center[1] + cc_offsets[47][1]],
        [center[0] + cc_offsets[48][0], center[1] + cc_offsets[48][1]],
        [center[0] + cc_offsets[49][0], center[1] + cc_offsets[49][1]],
        [center[0] + cc_offsets[50][0], center[1] + cc_offsets[50][1]],
        [center[0] + cc_offsets[51][0], center[1] + cc_offsets[51][1]],
        [center[0] + cc_offsets[52][0], center[1] + cc_offsets[52][1]],
        [center[0] + cc_offsets[53][0], center[1] + cc_offsets[53][1]],
        # 54-56: right penalty area points
        [right_ps[0] + rpa_offsets[54][0], right_ps[1] + rpa_offsets[54][1]],
        [right_ps[0] + rpa_offsets[55][0], right_ps[1] + rpa_offsets[55][1]],
        [right_ps[0] + rpa_offsets[56][0], right_ps[1] + rpa_offsets[56][1]],
    ]

    # Center: x - HL, HW - y (y-up convention)
    world_coords_2d = [[x - HL, HW - y] for x, y in raw_coords]

    # --- Line definitions (same structure as LineMapper.LINE_DEFINITIONS) ---
    line_defs = {
        0:  {"p1": (-HL, PA_HW, 0), "p2": (-HL + PA_DEPTH, PA_HW, 0)},
        1:  {"p1": (-HL + PA_DEPTH, PA_HW, 0), "p2": (-HL + PA_DEPTH, -PA_HW, 0)},
        2:  {"p1": (-HL + PA_DEPTH, -PA_HW, 0), "p2": (-HL, -PA_HW, 0)},
        3:  {"p1": (HL, PA_HW, 0), "p2": (HL - PA_DEPTH, PA_HW, 0)},
        4:  {"p1": (HL - PA_DEPTH, PA_HW, 0), "p2": (HL - PA_DEPTH, -PA_HW, 0)},
        5:  {"p1": (HL - PA_DEPTH, -PA_HW, 0), "p2": (HL, -PA_HW, 0)},
        6:  {"p1": (-HL, -G_HW, G_H), "p2": (-HL, G_HW, G_H)},
        7:  {"p1": (-HL, -G_HW, 0), "p2": (-HL, -G_HW, G_H)},
        8:  {"p1": (-HL, G_HW, 0), "p2": (-HL, G_HW, G_H)},
        9:  {"p1": (HL, -G_HW, G_H), "p2": (HL, G_HW, G_H)},
        10: {"p1": (HL, -G_HW, 0), "p2": (HL, -G_HW, G_H)},
        11: {"p1": (HL, G_HW, 0), "p2": (HL, G_HW, G_H)},
        12: {"p1": (0, -HW, 0), "p2": (0, HW, 0)},
        13: {"p1": (-HL, HW, 0), "p2": (HL, HW, 0)},
        14: {"p1": (-HL, -HW, 0), "p2": (-HL, HW, 0)},
        15: {"p1": (HL, -HW, 0), "p2": (HL, HW, 0)},
        16: {"p1": (-HL, -HW, 0), "p2": (HL, -HW, 0)},
        17: {"p1": (-HL, GA_HW, 0), "p2": (-HL + GA_DEPTH, GA_HW, 0)},
        18: {"p1": (-HL + GA_DEPTH, GA_HW, 0), "p2": (-HL + GA_DEPTH, -GA_HW, 0)},
        19: {"p1": (-HL + GA_DEPTH, -GA_HW, 0), "p2": (-HL, -GA_HW, 0)},
        20: {"p1": (HL, GA_HW, 0), "p2": (HL - GA_DEPTH, GA_HW, 0)},
        21: {"p1": (HL - GA_DEPTH, GA_HW, 0), "p2": (HL - GA_DEPTH, -GA_HW, 0)},
        22: {"p1": (HL - GA_DEPTH, -GA_HW, 0), "p2": (HL, -GA_HW, 0)},
    }

    # --- Line intersections ---
    pa_front_x = HL - PA_DEPTH
    ga_front_x = HL - GA_DEPTH
    line_intersections = {
        (0, 1):   (-pa_front_x,  PA_HW, 0.0,  4),
        (1, 2):   (-pa_front_x, -PA_HW, 0.0, 24),
        (0, 14):  (-HL,          PA_HW, 0.0,  3),
        (2, 14):  (-HL,         -PA_HW, 0.0, 23),
        (3, 4):   ( pa_front_x,  PA_HW, 0.0,  5),
        (4, 5):   ( pa_front_x, -PA_HW, 0.0, 25),
        (3, 15):  ( HL,          PA_HW, 0.0,  6),
        (5, 15):  ( HL,         -PA_HW, 0.0, 26),
        (17, 18): (-ga_front_x,  GA_HW, 0.0,  8),
        (18, 19): (-ga_front_x, -GA_HW, 0.0, 20),
        (17, 14): (-HL,          GA_HW, 0.0,  7),
        (19, 14): (-HL,         -GA_HW, 0.0, 19),
        (20, 21): ( ga_front_x,  GA_HW, 0.0,  9),
        (21, 22): ( ga_front_x, -GA_HW, 0.0, 21),
        (20, 15): ( HL,          GA_HW, 0.0, 10),
        (22, 15): ( HL,         -GA_HW, 0.0, 22),
        (13, 14): (-HL,          HW,    0.0,  0),
        (13, 15): ( HL,          HW,    0.0,  2),
        (16, 14): (-HL,         -HW,    0.0, 27),
        (16, 15): ( HL,         -HW,    0.0, 29),
        (12, 13): ( 0.0,         HW,    0.0,  1),
        (12, 16): ( 0.0,        -HW,    0.0, 28),
    }

    return world_coords_2d, line_defs, line_intersections

# Keypoints that lie on each ground line (ordered by arc-length parameter t).
# Copied from BroadTrackCalibrator to avoid import dependency.
LINE_KEYPOINTS: dict[int, list[int]] = {
    0:  [3, 4],                         # Big rect. left top
    1:  [4, 30, 45, 33, 24],            # Big rect. left side
    2:  [24, 23],                        # Big rect. left bottom
    3:  [6, 5],                          # Big rect. right top
    4:  [5, 32, 55, 35, 25],            # Big rect. right side
    5:  [25, 26],                        # Big rect. right bottom
    12: [28, 34, 50, 31, 1],            # Middle line
    13: [0, 1, 2],                       # Side line top
    14: [27, 23, 19, 15, 11, 7, 3, 0],  # Side line left (goal line)
    15: [29, 26, 22, 17, 13, 10, 6, 2], # Side line right (goal line)
    16: [27, 28, 29],                    # Side line bottom
    17: [7, 8],                          # Small rect. left top
    18: [8, 20],                         # Small rect. left side
    19: [20, 19],                        # Small rect. left bottom
    20: [10, 9],                         # Small rect. right top
    21: [9, 21],                         # Small rect. right side
    22: [21, 22],                        # Small rect. right bottom
}

# Ground-plane line-line intersection pairs.
# Each entry: (line_id_a, line_id_b) → (world_x, world_y, 0.0, keypoint_id).
# keypoint_id is the PnLCalib keypoint that sits at this intersection;
# if that keypoint is already detected, the intersection is skipped (no duplicate).
LINE_INTERSECTIONS: dict[tuple[int, int], tuple[float, float, float, int]] = {
    # Penalty area left corners
    (0, 1):   (-36.00,  20.16, 0.0,  4),  # Big rect. left top × left side
    (1, 2):   (-36.00, -20.16, 0.0, 24),  # Big rect. left side × left bottom
    (0, 14):  (-52.50,  20.16, 0.0,  3),  # Big rect. left top × goal line left
    (2, 14):  (-52.50, -20.16, 0.0, 23),  # Big rect. left bottom × goal line left
    # Penalty area right corners
    (3, 4):   ( 36.00,  20.16, 0.0,  5),  # Big rect. right top × right side
    (4, 5):   ( 36.00, -20.16, 0.0, 25),  # Big rect. right side × right bottom
    (3, 15):  ( 52.50,  20.16, 0.0,  6),  # Big rect. right top × goal line right
    (5, 15):  ( 52.50, -20.16, 0.0, 26),  # Big rect. right bottom × goal line right
    # Goal area left corners
    (17, 18): (-47.00,   9.16, 0.0,  8),  # Small rect. left top × left side
    (18, 19): (-47.00,  -9.16, 0.0, 20),  # Small rect. left side × left bottom
    (17, 14): (-52.50,   9.16, 0.0,  7),  # Small rect. left top × goal line left
    (19, 14): (-52.50,  -9.16, 0.0, 19),  # Small rect. left bottom × goal line left
    # Goal area right corners
    (20, 21): ( 47.00,   9.16, 0.0,  9),  # Small rect. right top × right side
    (21, 22): ( 47.00,  -9.16, 0.0, 21),  # Small rect. right side × right bottom
    (20, 15): ( 52.50,   9.16, 0.0, 10),  # Small rect. right top × goal line right
    (22, 15): ( 52.50,  -9.16, 0.0, 22),  # Small rect. right bottom × goal line right
    # Pitch corners
    (13, 14): (-52.50,  34.00, 0.0,  0),  # Touchline top × goal line left
    (13, 15): ( 52.50,  34.00, 0.0,  2),  # Touchline top × goal line right
    (16, 14): (-52.50, -34.00, 0.0, 27),  # Touchline bottom × goal line left
    (16, 15): ( 52.50, -34.00, 0.0, 29),  # Touchline bottom × goal line right
    # Center line endpoints
    (12, 13): (  0.00,  34.00, 0.0,  1),  # Center line × touchline top
    (12, 16): (  0.00, -34.00, 0.0, 28),  # Center line × touchline bottom
}


class PhysicalCalibrator:
    """Camera calibrator with 7-DOF optimization: rvec(3), tvec(3), f.

    Distortion is fixed at zero and principal point at image center.
    Only focal length is optimized among intrinsics.
    """

    def __init__(
        self,
        K: np.ndarray,
        image_size: tuple[int, int],
        ransac_reproj_error: float = 15.0,
        line_weight: float = 1.0,
        line_sample_points: int = 20,
        focal_bounds: tuple[float, float] = (1200.0, 2200.0),
        world_residual_weight: float = 0.0,
        world_error_threshold: float = 5.0,
        camera_position: tuple[float, float, float] | None = None,
        position_weight: float = 50.0,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
    ):
        """Initialize with image size and focal length guess.

        Args:
            K: 3x3 intrinsic matrix (only f=K[0,0] is used as initial guess;
               cx/cy are overridden to image center).
            image_size: (width, height) of video frames.
            ransac_reproj_error: PnP RANSAC reprojection threshold in pixels.
            line_weight: Weight alpha for line residuals vs point residuals.
            line_sample_points: Number of points sampled along each line.
            focal_bounds: (min_f, max_f) absolute bounds for focal length.
            world_residual_weight: Weight for world-space back-projection residuals.
                0 = disabled.
            world_error_threshold: World-space error threshold (meters) for iterative
                outlier rejection. Set to float('inf') to disable.
            camera_position: Known camera position (x, y, z) in meters. If set, adds
                residuals penalizing deviation from this position. None = no constraint.
            position_weight: Weight for camera position residuals (pixels-equivalent).
            pitch_length: Pitch length in meters (default 105 = FIFA standard).
            pitch_width: Pitch width in meters (default 68 = FIFA standard).
        """
        self.width, self.height = image_size
        # Fix principal point at image geometric center, zero distortion
        self.K = np.array([
            [K[0, 0], 0, self.width / 2.0],
            [0, K[0, 0], self.height / 2.0],
            [0, 0, 1],
        ], dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)
        self.ransac_reproj_error = ransac_reproj_error
        self.line_weight = line_weight
        self.line_sample_points = line_sample_points
        self.focal_bounds = focal_bounds
        self.world_residual_weight = world_residual_weight
        self.world_error_threshold = world_error_threshold
        self.camera_position = camera_position
        self.position_weight = position_weight
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width

        # Build field template (world coords, line defs, intersections)
        self._field_world_coords, self._field_line_defs, self._field_intersections = \
            build_field_template(pitch_length, pitch_width)

        self._keypoints: list[dict] = []
        self._lines: list[dict] = []
        self._last_debug: dict | None = None
        self._last_ransac_debug: dict | None = None

    def update(self, keypoints: list[dict], lines: list[dict]):
        """Update with detected keypoints and lines for current frame."""
        self._keypoints = keypoints
        self._lines = lines

    def calibrate(
        self,
        keypoint_mapper,
        line_mapper=None,
        min_confidence: float = 0.3,
        initial_rvec: np.ndarray | None = None,
        initial_tvec: np.ndarray | None = None,
    ) -> dict[str, Any] | None:
        """Run 6-DOF calibration on current frame data.

        Args:
            keypoint_mapper: KeypointMapper for 3D correspondences.
            line_mapper: LineMapper for line world coords (or None to derive from keypoints).
            min_confidence: Minimum keypoint confidence threshold.
            initial_rvec: Warm-start rotation vector (from previous frame). If None, uses PnP RANSAC.
            initial_tvec: Warm-start translation vector. If None, uses PnP RANSAC.

        Returns:
            Result dict with camera params, or None if calibration failed.
        """
        # Reset debug info for this frame
        self._last_debug = None

        # Step 1: Prepare keypoint correspondences
        img_pts, world_pts, kp_ids, confidences = self._prepare_correspondences(
            keypoint_mapper, min_confidence
        )
        if len(world_pts) < 6:
            logger.debug("Too few keypoints (%d < 6), skipping", len(world_pts))
            self._last_debug = {
                "failure_reason": f"too_few_keypoints ({len(world_pts)} < 6)",
                "img_pts": img_pts,
                "world_pts": world_pts,
                "kp_ids": kp_ids,
                "ransac_info": None,
            }
            return None

        # Step 2: Prepare line constraints
        line_constraints = self._derive_lines_from_keypoints(
            kp_ids, img_pts, keypoint_mapper, line_mapper
        )

        # Step 3: Get initial pose estimate (using detected keypoints only)
        ransac_info = None
        if initial_rvec is not None and initial_tvec is not None:
            # Warm-start from previous frame — skip PnP RANSAC
            rvec_init = initial_rvec.copy().ravel()
            tvec_init = initial_tvec.copy().ravel()
            logger.debug("Using warm-start from previous frame")
        else:
            # Cold start — PnP RANSAC initialization
            pnp_result = self._pnp_ransac_init(world_pts, img_pts)
            if pnp_result is None:
                self._last_debug = {
                    "failure_reason": "pnp_ransac_failed",
                    "img_pts": img_pts,
                    "world_pts": world_pts,
                    "kp_ids": kp_ids,
                    "ransac_info": self._last_ransac_debug,
                }
                return None
            rvec_init, tvec_init, ransac_info = pnp_result

        # Step 3b: Add line-line intersection points for refinement
        isect_img, isect_world, isect_ids = self._compute_line_intersections(
            line_constraints, kp_ids
        )
        n_intersections = len(isect_ids)
        if n_intersections > 0:
            img_pts = np.vstack([img_pts, isect_img])
            world_pts = np.vstack([world_pts, isect_world])
            kp_ids = list(kp_ids) + isect_ids
            logger.debug("Added %d line intersection points (total=%d)",
                         n_intersections, len(kp_ids))

        # Step 4: 7-DOF joint point+line optimization (rvec, tvec, f)
        # with iterative world-error-based outlier rejection
        MAX_WORLD_OUTLIER_ITERS = 3

        cur_img_pts = img_pts
        cur_world_pts = world_pts
        cur_kp_ids = kp_ids
        cur_lc = line_constraints
        cur_rvec = rvec_init
        cur_tvec = tvec_init

        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_fixed = self.dist_coeffs.copy()

        if np.isfinite(self.world_error_threshold):
            for world_iter in range(MAX_WORLD_OUTLIER_ITERS):
                rvec_opt, tvec_opt, f_opt = self._refine_7dof(
                    cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
                )

                # Build optimized intrinsics for this iteration
                K_iter = np.array([
                    [f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]
                ], dtype=np.float64)

                # Compute per-point world error
                _, per_pt_werr = self._compute_world_error(
                    rvec_opt, tvec_opt, cur_img_pts, cur_world_pts,
                    K=K_iter, dist=dist_fixed,
                )

                # Distance-adaptive threshold: use relative error (world_error / distance).
                # A point is an outlier if its relative error exceeds threshold/30.
                # E.g., threshold=5m → max 16.7% relative error; at 30m that's 5m, at 90m that's 15m.
                R_iter, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
                cam_center = -R_iter.T @ tvec_opt.ravel()
                dists = np.linalg.norm(cur_world_pts - cam_center[np.newaxis, :], axis=1)
                dists = np.maximum(dists, 1.0)  # avoid division by zero
                relative_err = per_pt_werr / dists
                relative_thresh = self.world_error_threshold / 30.0  # ~16.7% for threshold=5m

                # Find world-error outliers
                keep_mask = relative_err < relative_thresh
                n_removed = int(np.sum(~keep_mask))
                if n_removed == 0 or world_iter == MAX_WORLD_OUTLIER_ITERS - 1:
                    break

                # Need at least 6 points to continue
                if int(np.sum(keep_mask)) < 6:
                    break

                logger.debug("World-error iter %d: removed %d points (threshold=%.1fm)",
                             world_iter, n_removed, self.world_error_threshold)

                cur_img_pts = cur_img_pts[keep_mask]
                cur_world_pts = cur_world_pts[keep_mask]
                cur_kp_ids = [cur_kp_ids[i] for i in range(len(cur_kp_ids)) if keep_mask[i]]
                cur_rvec = rvec_opt
                cur_tvec = tvec_opt

                # Re-derive line constraints from remaining keypoints
                cur_lc = self._derive_lines_from_keypoints(
                    cur_kp_ids, cur_img_pts, keypoint_mapper, line_mapper
                )
        else:
            # No outlier rejection — use all keypoints, single optimization pass
            rvec_opt, tvec_opt, f_opt = self._refine_7dof(
                cur_rvec, cur_tvec, cur_img_pts, cur_world_pts, cur_lc,
            )

        # Use the filtered points for final result
        img_pts = cur_img_pts
        world_pts = cur_world_pts
        kp_ids = cur_kp_ids
        line_constraints = cur_lc

        # Build optimized intrinsics (cx/cy fixed at center, dist=0)
        K_opt = np.array([
            [f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]
        ], dtype=np.float64)
        dist_opt = dist_fixed

        # Step 5: Sanity check — reject catastrophically bad results
        projected_check = project_points_2d(
            world_pts, rvec_opt, tvec_opt, K_opt, dist_opt,
        )
        median_err_check = float(np.median(np.linalg.norm(
            projected_check - img_pts, axis=1
        )))
        if median_err_check > 200:
            logger.warning("Optimization result rejected (median_err=%.0fpx > 200px)", median_err_check)
            self._last_debug = {
                "failure_reason": f"optimization_diverged (median_err={median_err_check:.0f}px)",
                "img_pts": img_pts,
                "world_pts": world_pts,
                "kp_ids": kp_ids,
                "ransac_info": ransac_info,
            }
            return None

        # Step 6: Compute stats and format result
        mean_error, inlier_mask, inlier_count = self._compute_reprojection_stats(
            rvec_opt, tvec_opt, img_pts, world_pts, K=K_opt, dist=dist_opt
        )
        world_error, per_point_world_errors = self._compute_world_error(
            rvec_opt, tvec_opt, img_pts, world_pts, K=K_opt, dist=dist_opt
        )
        world_error_all, per_kp_world_errors = self._compute_world_error_all(
            rvec_opt, tvec_opt, K=K_opt, dist=dist_opt
        )

        # Build result
        R, _ = cv2.Rodrigues(rvec_opt.reshape(3, 1))
        tvec_col = tvec_opt.reshape(3, 1)

        # Derive ground-plane homography for backward compatibility
        H = K_opt @ np.column_stack([R[:, 0], R[:, 1], tvec_col.ravel()])
        if abs(H[2, 2]) > 1e-10:
            H = H / H[2, 2]

        return {
            "homography": H,
            "final_error": float(mean_error),
            "world_error": float(world_error),
            "world_error_all": float(world_error_all),
            "per_point_world_errors": per_point_world_errors,
            "per_kp_world_errors": per_kp_world_errors,
            "num_keypoints": len(self._keypoints),
            "num_lines": len(self._lines),
            "num_intersections": n_intersections,
            "total_points": len(img_pts),
            "inliers": int(inlier_count),
            "intrinsics_init": {
                "f": float(self.K[0, 0]),
                "cx": float(self.K[0, 2]),
                "cy": float(self.K[1, 2]),
            },
            "camera_params": {
                "K": K_opt.copy(),
                "rvec": rvec_opt.copy(),
                "tvec": tvec_opt.copy(),
                "R": R,
                "dist_coeffs": dist_opt.copy(),
                "focal_length": float(f_opt),
            },
            "camera_pose": {
                "K": K_opt.tolist(),
                "dist_coeffs": dist_opt.tolist(),
                "rvec": rvec_opt.tolist(),
                "tvec": tvec_opt.tolist(),
                "reprojection_error": float(mean_error),
                "world_error": float(world_error),
                "world_error_all": float(world_error_all),
                "inliers_count": int(inlier_count),
            },
            "img_pts": img_pts,
            "world_pts": world_pts,
            "kp_ids": kp_ids,
            "inlier_mask": inlier_mask,
            "line_constraints_count": len(line_constraints),
            "line_constraints": line_constraints,
            "ransac_info": ransac_info,
        }

    def _prepare_correspondences(self, keypoint_mapper, min_confidence):
        """Extract 3D-2D keypoint correspondences including crossbar points."""
        img_pts, world_pts, kp_ids = keypoint_mapper.build_3d_correspondence_matrix(
            self._keypoints,
            filter_by_confidence=min_confidence,
            exclude_non_ground=False,  # Include crossbar points (z=-2.44)
        )

        # Override world coordinates with field template (supports non-standard pitch sizes)
        if len(world_pts) > 0:
            world_pts = world_pts.copy()
            for idx, kid in enumerate(kp_ids):
                if kid < len(self._field_world_coords):
                    wx, wy = self._field_world_coords[kid]
                    z = world_pts[idx, 2]  # preserve z (0 or -2.44 for crossbar)
                    world_pts[idx] = [wx, wy, z]

        # Build confidence array matching the correspondence order
        conf_map = {kp["id"]: kp.get("confidence", 1.0) for kp in self._keypoints}
        confidences = np.array([conf_map.get(kid, 1.0) for kid in kp_ids], dtype=np.float64)

        return (
            img_pts.astype(np.float64) if len(img_pts) > 0 else np.zeros((0, 2), dtype=np.float64),
            world_pts.astype(np.float64) if len(world_pts) > 0 else np.zeros((0, 3), dtype=np.float64),
            kp_ids,
            confidences,
        )

    def _derive_lines_from_keypoints(self, kp_ids, kp_img, keypoint_mapper, line_mapper):
        """Derive line constraints from detected keypoints on field lines.

        For each ground line, if >=2 of its keypoints are detected, connect the
        two most extreme ones in image space and sample points along that segment.
        """
        detected = {}
        for i, kid in enumerate(kp_ids):
            detected[kid] = kp_img[i]

        constraints = []
        non_ground = keypoint_mapper.NON_GROUND_KEYPOINTS

        for line_id, kp_list in LINE_KEYPOINTS.items():
            found = [(kid, detected[kid]) for kid in kp_list
                     if kid in detected and kid not in non_ground]
            if len(found) < 2:
                continue

            # Get world line endpoints
            ld = self._field_line_defs[line_id]
            p1_w = np.array(ld["p1"], dtype=np.float64)
            p2_w = np.array(ld["p2"], dtype=np.float64)
            polyline_3d = np.array([p1_w, p2_w])
            cum_lengths = compute_cumulated_lengths(polyline_3d)

            # Sample N evenly spaced 3D points along the world line
            total_len = cum_lengths[-1]
            if total_len < 1e-6:
                continue
            t_values = np.linspace(0, total_len, self.line_sample_points)
            world_samples = np.array([
                interpolate_on_polyline(polyline_3d, cum_lengths, t)
                for t in t_values
            ])

            # Image line: connect first and last detected keypoint
            first_img = found[0][1]
            last_img = found[-1][1]
            img_samples = sample_points_on_image_line(
                float(first_img[0]), float(first_img[1]),
                float(last_img[0]), float(last_img[1]),
                n_points=self.line_sample_points,
            )

            constraints.append({
                "world_samples": world_samples,
                "img_samples": img_samples,
                "line_id": line_id,
            })

        return constraints

    def _compute_line_intersections(self, line_constraints, existing_kp_ids):
        """Compute intersection points of derived image lines.

        For each pair of lines known to intersect on the field, computes the
        2D image intersection and returns it as an additional correspondence.
        Skips intersections whose keypoint is already detected (avoids duplicates).

        Args:
            line_constraints: list of line constraint dicts from _derive_lines_from_keypoints.
            existing_kp_ids: list of already-detected keypoint IDs.

        Returns:
            (img_pts, world_pts, kp_ids) — arrays of intersection correspondences.
        """
        # Build lookup: line_id → (p1_img, p2_img) from line constraints
        line_endpoints = {}
        for lc in line_constraints:
            lid = lc["line_id"]
            img = lc["img_samples"]
            line_endpoints[lid] = (img[0], img[-1])

        existing_set = set(existing_kp_ids)
        margin = 50  # pixels outside image to still accept

        int_img = []
        int_world = []
        int_ids = []

        for (lid_a, lid_b), (wx, wy, wz, kp_id) in self._field_intersections.items():
            if lid_a not in line_endpoints or lid_b not in line_endpoints:
                continue

            # Skip if the corresponding keypoint is already detected
            if kp_id in existing_set:
                continue

            # Image line a: p1a → p2a
            p1a, p2a = line_endpoints[lid_a]
            # Image line b: p1b → p2b
            p1b, p2b = line_endpoints[lid_b]

            # Parametric line-line intersection
            x1, y1 = float(p1a[0]), float(p1a[1])
            x2, y2 = float(p2a[0]), float(p2a[1])
            x3, y3 = float(p1b[0]), float(p1b[1])
            x4, y4 = float(p2b[0]), float(p2b[1])

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue  # parallel lines

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)

            # Skip if outside image bounds (with margin)
            if ix < -margin or ix > self.width + margin:
                continue
            if iy < -margin or iy > self.height + margin:
                continue

            int_img.append([ix, iy])
            int_world.append([wx, wy, wz])
            int_ids.append(f"isect_{lid_a}_{lid_b}")

        if not int_img:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 3), dtype=np.float64),
                [],
            )

        return (
            np.array(int_img, dtype=np.float64),
            np.array(int_world, dtype=np.float64),
            int_ids,
        )

    def _pnp_ransac_init(self, world_pts, img_pts):
        """Initialize camera pose via multi-focal multi-solver PnP RANSAC.

        Tries multiple focal lengths × multiple solvers and picks the result
        with the most inliers. This avoids bad initialization when the actual
        focal length differs significantly from the profile value.

        Returns:
            (rvec, tvec, ransac_info) or None on failure.
            ransac_info contains inlier indices, per-point errors, and init pose.
        """
        # Sample focal lengths spanning the allowed range
        f_min, f_max = self.focal_bounds
        f_profile = self.K[0, 0]
        focal_candidates = sorted(set([
            f_min,
            (f_min + f_profile) / 2,
            f_profile,
            (f_profile + f_max) / 2,
            f_max,
        ]))
        solvers = [cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_EPNP]
        cx, cy = self.K[0, 2], self.K[1, 2]
        dist = self.dist_coeffs

        best_result = None
        best_inlier_count = -1
        best_f = f_profile

        for f_try in focal_candidates:
            K_try = np.array([[f_try, 0, cx], [0, f_try, cy], [0, 0, 1]], dtype=np.float64)
            for solver in solvers:
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=world_pts.reshape(-1, 1, 3),
                    imagePoints=img_pts.reshape(-1, 1, 2),
                    cameraMatrix=K_try,
                    distCoeffs=dist,
                    reprojectionError=self.ransac_reproj_error,
                    iterationsCount=2000,
                    flags=solver,
                )
                if success and inliers is not None and len(inliers) >= 4:
                    if len(inliers) > best_inlier_count:
                        best_inlier_count = len(inliers)
                        best_result = (rvec, tvec, inliers)
                        best_f = f_try

        if best_result is None:
            logger.debug("PnP RANSAC failed (all trials)")
            self._last_ransac_debug = {
                "success": False,
                "failure_reason": "all_trials_failed",
                "reprojection_threshold": float(self.ransac_reproj_error),
                "total_points": int(len(img_pts)),
                "inlier_count": 0,
                "inlier_indices": [],
            }
            return None

        rvec, tvec, inliers = best_result

        # Update K with the best focal length for reprojection error computation
        K_best = np.array([[best_f, 0, cx], [0, best_f, cy], [0, 0, 1]], dtype=np.float64)
        # Also update self.K so _refine_7dof starts from this f
        self.K[0, 0] = best_f
        self.K[1, 1] = best_f

        # Compute per-point reprojection errors for RANSAC result
        projected = project_points_2d(world_pts, rvec, tvec, K_best, dist)
        per_point_errors = np.linalg.norm(projected - img_pts, axis=1)
        logger.debug("PnP RANSAC best: f=%.0f, %d/%d inliers",
                      best_f, best_inlier_count, len(img_pts))

        inlier_indices = inliers.ravel().tolist()
        inlier_mask = np.zeros(len(img_pts), dtype=bool)
        inlier_mask[inlier_indices] = True

        ransac_info = {
            "success": True,
            "reprojection_threshold": float(self.ransac_reproj_error),
            "total_points": int(len(img_pts)),
            "inlier_count": int(len(inlier_indices)),
            "outlier_count": int(len(img_pts) - len(inlier_indices)),
            "inlier_indices": inlier_indices,
            "inlier_mask": inlier_mask,
            "per_point_errors": per_point_errors,
            "mean_inlier_error": float(np.mean(per_point_errors[inlier_mask])),
            "mean_outlier_error": float(np.mean(per_point_errors[~inlier_mask])) if np.any(~inlier_mask) else 0.0,
            "mean_all_error": float(np.mean(per_point_errors)),
            "rvec_init": rvec.ravel().copy(),
            "tvec_init": tvec.ravel().copy(),
        }

        logger.debug("PnP RANSAC: %d/%d inliers, mean_err=%.2fpx",
                      len(inlier_indices), len(img_pts), ransac_info["mean_all_error"])
        return rvec.ravel(), tvec.ravel(), ransac_info

    def _refine_7dof(self, rvec_init, tvec_init, img_pts, world_pts, line_constraints):
        """7-DOF bounded optimization with point+line residuals.

        State vector: [rvec(3), tvec(3), f] — 7 parameters.
        cx/cy fixed at image center, distortion fixed at zero.
        """
        f_init = self.K[0, 0]
        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_zero = np.zeros(5, dtype=np.float64)

        x0 = np.concatenate([rvec_init.ravel(), tvec_init.ravel(), [f_init]])

        # Pre-compute line normals from detected endpoints (constant during optimization)
        line_normals = []
        line_origins = []
        valid_lc = []
        for lc in line_constraints:
            img_line = lc["img_samples"]
            p1 = img_line[0]
            p2 = img_line[-1]
            line_dir = p2 - p1
            line_len = np.linalg.norm(line_dir)
            if line_len < 1e-6:
                continue
            normal = np.array([-line_dir[1], line_dir[0]]) / line_len
            line_normals.append(normal)
            line_origins.append(p1)
            valid_lc.append(lc)

        n_samples = self.line_sample_points
        per_sample_weight = self.line_weight / np.sqrt(n_samples) if n_samples > 0 else 0.0

        def cost_fn(x):
            rvec = x[:3]
            tvec = x[3:6]
            f = x[6]
            K_cur = np.array([[f, 0, cx_fixed], [0, f, cy_fixed], [0, 0, 1]], dtype=np.float64)

            # Point residuals: projected - detected
            projected = project_points_2d(
                world_pts, rvec, tvec, K_cur, dist_zero,
            )
            point_residuals = (projected - img_pts).ravel()

            all_residuals = [point_residuals]

            # Line residuals (zero out degenerate projections outside image bounds)
            if valid_lc:
                img_bound = max(self.width, self.height) * 3
                for i, lc in enumerate(valid_lc):
                    proj_line = project_points_2d(
                        lc["world_samples"], rvec, tvec, K_cur, dist_zero,
                    )
                    diffs = proj_line - line_origins[i]
                    distances = diffs @ line_normals[i]
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

            # World-space back-projection residuals (vectorized)
            if self.world_residual_weight > 0:
                R_cur, _ = cv2.Rodrigues(rvec)
                cam_center = -R_cur.T @ tvec.ravel()
                pts_undist = cv2.undistortPoints(
                    img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_zero
                ).reshape(-1, 2)
                ones = np.ones((len(pts_undist), 1))
                rays_cam = np.hstack([pts_undist, ones])
                rays_world = (R_cur.T @ rays_cam.T).T
                t_params = -cam_center[2] / rays_world[:, 2]
                wp = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world
                diff_xy = (wp[:, :2] - world_pts[:, :2]) * self.world_residual_weight
                invalid = (np.abs(rays_world[:, 2]) < 1e-10) | (t_params < 0)
                diff_xy[invalid] = 0.0
                all_residuals.append(diff_xy.ravel())

            # Camera position constraint (x, y, z)
            # Replicate into many small residuals so cauchy loss doesn't
            # suppress them — each copy stays near the quadratic regime.
            if self.camera_position is not None:
                R_cur, _ = cv2.Rodrigues(rvec)
                cam_center = -R_cur.T @ tvec.ravel()
                pos_target = np.array(self.camera_position)
                n_copies = 50
                w_per_copy = self.position_weight / np.sqrt(n_copies)
                pos_single = (cam_center - pos_target) * w_per_copy
                all_residuals.append(np.tile(pos_single, n_copies))

            return np.concatenate(all_residuals)

        bounds = (
            [-np.inf] * 6 + [self.focal_bounds[0]],
            [+np.inf] * 6 + [self.focal_bounds[1]],
        )
        result = least_squares(
            cost_fn, x0, method="trf", bounds=bounds,
            loss="cauchy", f_scale=15.0, max_nfev=300,
        )

        return result.x[:3], result.x[3:6], float(result.x[6])

    def _compute_reprojection_stats(self, rvec, tvec, img_pts, world_pts, K=None, dist=None):
        """Compute reprojection error statistics."""
        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        projected = project_points_2d(world_pts, rvec, tvec, K_use, dist_use)
        errors = np.linalg.norm(projected - img_pts, axis=1)

        mean_error = float(np.mean(errors)) if len(errors) > 0 else float("inf")
        # All points that reach this stage participated in optimization —
        # outlier rejection (if any) already happened in the world-error loop
        inlier_mask = np.ones(len(errors), dtype=bool)
        inlier_count = len(errors)

        return mean_error, inlier_mask, inlier_count

    def _compute_world_error(self, rvec, tvec, img_pts, world_pts, K=None, dist=None):
        """Compute world-space back-projection error for detected keypoints (meters).

        Back-projects detected image pixels to ground plane (z=0) and measures
        distance from true world positions.
        """
        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        errors = self._backproject_errors(rvec, tvec, img_pts, world_pts, K_use, dist_use)
        valid = np.isfinite(errors)
        mean_error = float(np.mean(errors[valid])) if valid.any() else float("inf")
        return mean_error, errors

    def _compute_world_error_all(self, rvec, tvec, K=None, dist=None):
        """Compute world-space round-trip error for ALL 57 ground keypoints (meters).

        For each template keypoint: project world→image, then back-project image→world.
        The round-trip deviation reveals projection model accuracy across the full pitch,
        including points outside the camera FOV.
        """
        from .pnlcalib import KeypointMapper

        K_use = K if K is not None else self.K
        dist_use = dist if dist is not None else self.dist_coeffs
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))

        all_world = self._field_world_coords
        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS

        # Collect ground keypoints
        kp_ids = []
        world_3d = []
        for kid, (wx, wy) in enumerate(all_world):
            if kid in non_ground:
                continue
            kp_ids.append(kid)
            world_3d.append([wx, wy, 0.0])
        world_3d = np.array(world_3d, dtype=np.float64)

        # Project world→image
        img_pts = project_points_2d(world_3d, rvec, tvec, K_use, dist_use)

        # Back-project image→world and compute error
        errors = self._backproject_errors(rvec, tvec, img_pts, world_3d, K_use, dist_use)

        # Build per-keypoint dict
        per_kp = {}
        valid_errors = []
        for i, kid in enumerate(kp_ids):
            per_kp[kid] = float(errors[i])
            if np.isfinite(errors[i]):
                valid_errors.append(errors[i])

        mean_error = float(np.mean(valid_errors)) if valid_errors else float("inf")
        return mean_error, per_kp

    def _backproject_errors(self, rvec, tvec, img_pts, world_pts, K, dist):
        """Back-project image points to ground plane and compute distance from world_pts."""
        R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
        cam_center = -R.T @ tvec.ravel()

        pts_undist = cv2.undistortPoints(
            img_pts.reshape(-1, 1, 2).astype(np.float64), K, dist
        ).reshape(-1, 2)

        errors = np.full(len(pts_undist), float("inf"))
        for j in range(len(pts_undist)):
            ray_world = R.T @ np.array([pts_undist[j, 0], pts_undist[j, 1], 1.0])
            if abs(ray_world[2]) < 1e-10:
                continue
            t_param = -cam_center[2] / ray_world[2]
            if t_param < 0:
                continue
            wp = cam_center + t_param * ray_world
            errors[j] = np.sqrt((wp[0] - world_pts[j, 0])**2 + (wp[1] - world_pts[j, 1])**2)
        return errors

    def joint_optimize_intrinsics(self, frame_data_list):
        """Cross-frame joint optimization: shared focal length, per-frame extrinsics.

        Solves for a single f across all frames while each frame keeps its own
        (rvec, tvec). cx/cy fixed at image center, distortion fixed at zero.

        Args:
            frame_data_list: list of dicts, each with:
                - "rvec": (3,) initial rotation vector
                - "tvec": (3,) initial translation vector
                - "img_pts": (N, 2) detected image points
                - "world_pts": (N, 3) corresponding world points
                - "line_constraints": list of line constraint dicts

        Returns:
            dict with optimized focal length and per-frame extrinsics, or None.
        """
        n_frames = len(frame_data_list)
        if n_frames < 2:
            logger.warning("Joint optimization needs ≥2 frames, got %d", n_frames)
            return None

        N_INTRINSICS = 1  # f only
        f_init = self.K[0, 0]
        cx_fixed = self.K[0, 2]
        cy_fixed = self.K[1, 2]
        dist_zero = np.zeros(5, dtype=np.float64)

        # State vector: [f] + [per-frame rvec(3), tvec(3)] × N = 1 + 6N parameters
        x0_parts = [np.array([f_init])]
        for fd in frame_data_list:
            x0_parts.append(fd["rvec"].ravel())
            x0_parts.append(fd["tvec"].ravel())
        x0 = np.concatenate(x0_parts)

        # Pre-compute line normals per frame (constant during optimization)
        frame_line_info = []
        n_samples = self.line_sample_points
        per_sample_weight = self.line_weight / np.sqrt(n_samples) if n_samples > 0 else 0.0

        for fd in frame_data_list:
            normals, origins, valid_lcs = [], [], []
            for lc in fd.get("line_constraints", []):
                img_line = lc["img_samples"]
                p1, p2 = img_line[0], img_line[-1]
                line_dir = p2 - p1
                line_len = np.linalg.norm(line_dir)
                if line_len < 1e-6:
                    continue
                normal = np.array([-line_dir[1], line_dir[0]]) / line_len
                normals.append(normal)
                origins.append(p1)
                valid_lcs.append(lc)
            frame_line_info.append((normals, origins, valid_lcs))

        img_bound = max(self.width, self.height) * 3

        def cost_fn(x):
            f = x[0]
            K_cur = np.array([[f, 0, cx_fixed], [0, f, cy_fixed], [0, 0, 1]], dtype=np.float64)

            all_residuals = []

            for i, fd in enumerate(frame_data_list):
                offset = N_INTRINSICS + i * 6
                rvec = x[offset:offset + 3].reshape(3, 1)
                tvec = x[offset + 3:offset + 6].reshape(3, 1)
                img_pts = fd["img_pts"]
                world_pts = fd["world_pts"]

                # Point residuals
                projected = project_points_2d(
                    world_pts, rvec, tvec, K_cur, dist_zero,
                )
                all_residuals.append((projected - img_pts).ravel())

                # Line residuals
                normals, origins, valid_lcs = frame_line_info[i]
                for j, lc in enumerate(valid_lcs):
                    proj_line = project_points_2d(
                        lc["world_samples"], rvec, tvec, K_cur, dist_zero,
                    )
                    distances = (proj_line - origins[j]) @ normals[j]
                    valid = (np.abs(proj_line[:, 0]) < img_bound) & (np.abs(proj_line[:, 1]) < img_bound)
                    distances[~valid] = 0.0
                    all_residuals.append(distances * per_sample_weight)

                # World-space back-projection residuals (vectorized)
                if self.world_residual_weight > 0:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    cam_center = -R_cur.T @ tvec.ravel()
                    pts_undist = cv2.undistortPoints(
                        img_pts.reshape(-1, 1, 2).astype(np.float64), K_cur, dist_zero
                    ).reshape(-1, 2)
                    ones = np.ones((len(pts_undist), 1))
                    rays_cam = np.hstack([pts_undist, ones])
                    rays_world = (R_cur.T @ rays_cam.T).T
                    t_params = -cam_center[2] / rays_world[:, 2]
                    wp = cam_center[np.newaxis, :] + t_params[:, np.newaxis] * rays_world
                    diff_xy = (wp[:, :2] - world_pts[:, :2]) * self.world_residual_weight
                    invalid = (np.abs(rays_world[:, 2]) < 1e-10) | (t_params < 0)
                    diff_xy[invalid] = 0.0
                    all_residuals.append(diff_xy.ravel())

                # Camera position constraint (x, y, z) — replicated for cauchy robustness
                if self.camera_position is not None:
                    R_cur, _ = cv2.Rodrigues(rvec)
                    cam_center = -R_cur.T @ tvec.ravel()
                    pos_target = np.array(self.camera_position)
                    n_copies = 200
                    w_per_copy = self.position_weight / np.sqrt(n_copies)
                    pos_single = (cam_center - pos_target) * w_per_copy
                    all_residuals.append(np.tile(pos_single, n_copies))

            return np.concatenate(all_residuals)

        # Bounds: shared f bounded, per-frame extrinsics unbounded
        lb = [self.focal_bounds[0]]
        ub = [self.focal_bounds[1]]
        for _ in range(n_frames):
            lb.extend([-np.inf] * 6)
            ub.extend([+np.inf] * 6)

        logger.info("Joint optimization: %d frames, %d params", n_frames, len(x0))
        result = least_squares(
            cost_fn, x0, method="trf", bounds=(lb, ub),
            loss="soft_l1", f_scale=15.0, max_nfev=2000,
        )

        # Extract results
        f_opt = float(result.x[0])
        K_opt = np.array([[f_opt, 0, cx_fixed], [0, f_opt, cy_fixed], [0, 0, 1]], dtype=np.float64)

        per_frame = []
        for i in range(n_frames):
            offset = N_INTRINSICS + i * 6
            rvec_i = result.x[offset:offset + 3]
            tvec_i = result.x[offset + 3:offset + 6]
            per_frame.append({"rvec": rvec_i, "tvec": tvec_i})

        logger.info("Joint result: f=%.1f, cost=%.2f", f_opt, result.cost)

        return {
            "K": K_opt,
            "dist_coeffs": dist_zero.copy(),
            "f": f_opt,
            "cx": float(cx_fixed),
            "cy": float(cy_fixed),
            "per_frame": per_frame,
            "cost": float(result.cost),
            "n_frames": n_frames,
        }
