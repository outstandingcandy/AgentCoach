"""Map HRNet keypoint indices to PnLCalib keypoint indices via world coordinates.

The user's annotation uses hrnet_index, which differs from PnLCalib's keypoint
indices. Since both refer to the same soccer field, we can match them by world
coordinates.
"""

from __future__ import annotations

import numpy as np


# PnLCalib 57 keypoint world coordinates (1-indexed in code, stored as 0-indexed here)
# From PnLCalib/utils/utils_calib.py - keypoint_world_coords_2D
# Original coords use y-down convention (0,0 at top-left)
# Centered: x - 52.5, y flipped: 34 - y (to convert y-down to y-up)
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


class HRNetToPnLCalibMapper:
    """Map HRNet keypoint indices to PnLCalib indices via world coordinate matching.

    The user's annotations have world coordinates that match PnLCalib's coordinate
    system (centered at pitch center). We build a mapping by finding the closest
    PnLCalib keypoint for each HRNet keypoint based on world coordinates.
    """

    def __init__(self, tolerance: float = 0.5):
        """Initialize mapper.

        Args:
            tolerance: Maximum distance (meters) for coordinate matching.
        """
        self.tolerance = tolerance
        self.pnlcalib_coords = np.array(PNLCALIB_WORLD_COORDS_2D)
        self._hrnet_to_pnlcalib_cache: dict[int, int | None] = {}

    def find_pnlcalib_index(self, world_x: float, world_y: float) -> int | None:
        """Find PnLCalib keypoint index for a world coordinate.

        Args:
            world_x: X coordinate in meters (centered at pitch center).
            world_y: Y coordinate in meters (centered at pitch center).

        Returns:
            PnLCalib keypoint index (1-indexed, matching their convention),
            or None if no match found within tolerance.
        """
        point = np.array([world_x, world_y])
        distances = np.linalg.norm(self.pnlcalib_coords - point, axis=1)
        min_idx = np.argmin(distances)
        min_dist = distances[min_idx]

        if min_dist <= self.tolerance:
            # PnLCalib uses 1-indexed keypoints
            return int(min_idx + 1)
        return None

    def map_annotation_point(
        self,
        pixel: list[float],
        world: list[float],
        hrnet_index: int | None = None,
    ) -> dict | None:
        """Map a single annotation point to PnLCalib format.

        Args:
            pixel: [x, y] pixel coordinates.
            world: [x, y] world coordinates (centered at pitch center).
            hrnet_index: Original HRNet index (for caching).

        Returns:
            Dict with 'pnlcalib_id', 'x', 'y', 'in_frame' if matched,
            None otherwise.
        """
        pnlcalib_id = self.find_pnlcalib_index(world[0], world[1])

        if pnlcalib_id is None:
            return None

        # Cache the mapping for this hrnet_index
        if hrnet_index is not None:
            self._hrnet_to_pnlcalib_cache[hrnet_index] = pnlcalib_id

        return {
            'pnlcalib_id': pnlcalib_id,
            'x': pixel[0],
            'y': pixel[1],
            'in_frame': True,  # Annotated points are always in frame
            'world': world,
        }

    def convert_annotation(
        self,
        annotation: dict,
        image_width: int = 1920,
        image_height: int = 1080,
    ) -> dict[int, dict]:
        """Convert a full annotation file to PnLCalib keypoint format.

        Args:
            annotation: The loaded annotation JSON (with 'all_points' key).
            image_width: Image width for in_frame check.
            image_height: Image height for in_frame check.

        Returns:
            Dict mapping PnLCalib keypoint ID (1-indexed) to keypoint info:
            {kp_id: {'x': px, 'y': py, 'in_frame': bool}}
        """
        keypoints = {}

        points = annotation.get('all_points', [])
        for point in points:
            pixel = point['pixel']
            world = point['world']
            hrnet_idx = point.get('hrnet_index')

            # Prefer the explicit pnlcalib_id written by the annotator (added
            # 2026-05). Older annotation files don't have it; fall back to
            # the world-coord matcher.
            saved_pnl_id = point.get('pnlcalib_id')
            if saved_pnl_id is not None and saved_pnl_id >= 0:
                # Annotator stores 0-indexed channel; PnLCalib downstream
                # uses 1-indexed (channel n -> id n+1).
                pnlcalib_id = int(saved_pnl_id) + 1
                x, y = float(pixel[0]), float(pixel[1])
                if hrnet_idx is not None:
                    self._hrnet_to_pnlcalib_cache[hrnet_idx] = pnlcalib_id
            else:
                mapped = self.map_annotation_point(pixel, world, hrnet_idx)
                if mapped is None:
                    continue
                pnlcalib_id = mapped['pnlcalib_id']
                x, y = mapped['x'], mapped['y']

            in_frame = 0 <= x < image_width and 0 <= y < image_height

            keypoints[pnlcalib_id] = {
                'x': x,
                'y': y,
                'in_frame': in_frame,
            }

        return keypoints

    def get_mapping_report(self) -> str:
        """Generate a report of discovered mappings."""
        lines = ["HRNet to PnLCalib Index Mapping:"]
        lines.append("-" * 40)

        for hrnet_idx, pnlcalib_idx in sorted(self._hrnet_to_pnlcalib_cache.items()):
            if pnlcalib_idx is not None:
                world = PNLCALIB_WORLD_COORDS_2D[pnlcalib_idx - 1]
                lines.append(
                    f"  HRNet {hrnet_idx:3d} -> PnLCalib {pnlcalib_idx:3d} "
                    f"(world: [{world[0]:6.2f}, {world[1]:6.2f}])"
                )

        return "\n".join(lines)
