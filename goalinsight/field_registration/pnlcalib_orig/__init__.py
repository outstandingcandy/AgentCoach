"""Faithful port of upstream `mguti97/PnLCalib` calibration pipeline.

Mirrors `utils/utils_calib.py` and friends from
https://github.com/mguti97/PnLCalib closely enough that:

- Calibration uses ``cv2.calibrateCamera`` (Zhang) instead of solvePnPRansac.
- z!=0 crossbar points are folded into the calibration via the upstream
  3-plane reparameterization (``get_per_plane_correspondences`` →
  ``change_plane_coords``).
- The 18-combo ``heuristic_voting`` (3 modes × 6 RANSAC thresholds) and
  ``line_optimizer`` (point+line residuals, alpha=0.7) are present.

Differences from upstream — all *additions*, never logic edits:

1. ``FramebyFrameCalib`` accepts ``pitch_dims`` so non-FIFA pitches
   (e.g. kids_soccer 66.28×43.15×2.15) can be calibrated. Defaults give
   exactly the upstream literal values (105×68×2.44).
2. Internal keypoint IDs are 1-indexed (matching upstream); the project
   uses 0-indexed names. ``PnLCalibIdMap`` bridges at the boundary.

The existing project ``pnlcalib/`` package (a heavy refactor of upstream)
is *not* touched by this module; it remains the production default until
``field_registration.pnlcalib_orig.enabled: true`` is set.
"""

from .pitch_template import FIFA_DEFAULTS, build_keypoint_table
from .utils_calib import FramebyFrameCalib, pan_tilt_roll_to_orientation
from .id_mapping import (
    LINE_NAME_TO_UPSTREAM_LINE_ID,
    NON_GROUND_UPSTREAM_IDS,
    PnLCalibIdMap,
)

__all__ = [
    "FIFA_DEFAULTS",
    "FramebyFrameCalib",
    "LINE_NAME_TO_UPSTREAM_LINE_ID",
    "NON_GROUND_UPSTREAM_IDS",
    "PnLCalibIdMap",
    "build_keypoint_table",
    "pan_tilt_roll_to_orientation",
]
