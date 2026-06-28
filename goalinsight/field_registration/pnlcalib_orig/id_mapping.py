"""ID bridges between project (0-indexed) and upstream PnLCalib (1-indexed).

Two systems disagree on two axes:

1. **Off-by-one**: project IDs are 0-indexed; upstream is 1-indexed. The
   ``+1`` rule fixes most cases.
2. **Crossbar pair ordering on the LEFT goal**: upstream lists each post's
   crossbar end (``z = -goal_height``) *before* its ground foot, while
   the project HRNet head emits ground foot *before* crossbar end. This
   swaps the (foot, crossbar) pair within the upstream sequence for IDs
   11/12 and 15/16. The right goal happens to align with naive ``+1``
   because project IDs 13/14/17/18 land on upstream 14/15/18/19 in
   matching order.

Verified at module import: every name in
``annotation.pitch.keypoints.PITCH_POINT_TO_PNLCALIB_ID`` resolves to a
world coord that agrees (1e-3 m) with ``build_keypoint_table()``'s
upstream-indexed table — if that fails, the swap below is wrong for the
active pitch and we ``raise`` rather than silently miscalibrate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .pitch_template import build_keypoint_table

if TYPE_CHECKING:
    from goalinsight.annotation.pitch.geometry import SoccerPitch


# Upstream 1-indexed crossbar / non-ground IDs.
NON_GROUND_UPSTREAM_IDS: set[int] = {12, 15, 16, 19}


# Project 0-indexed → upstream 1-indexed for the LEFT-goal foot/crossbar
# pairs (only place naive ``+1`` is wrong; right goal aligns naturally).
_PROJ_TO_UPSTREAM_KP_OVERRIDE: dict[int, int] = {
    11: 13,   # L_GOAL_BR_POST  (top-side foot)        proj 11 → upstream 13
    12: 12,   # L_GOAL_TR_POST  (top-side crossbar)    proj 12 → upstream 12
    15: 17,   # L_GOAL_BL_POST  (bot-side foot)        proj 15 → upstream 17
    16: 16,   # L_GOAL_TL_POST  (bot-side crossbar)    proj 16 → upstream 16
}


# Project line-class 0-indexed → upstream 1-indexed line ID. Project's
# class names don't 1-1 match upstream's (e.g. "Big rect. left side" vs
# upstream's "Big rect. left main"), so this is hand-curated.
LINE_NAME_TO_UPSTREAM_LINE_ID: dict[str, int] = {
    "Big rect. left bottom": 1,
    "Big rect. left main": 2,
    "Big rect. left side": 2,             # project alias for upstream "main"
    "Big rect. left top": 3,
    "Big rect. right bottom": 4,
    "Big rect. right main": 5,
    "Big rect. right side": 5,
    "Big rect. right top": 6,
    "Goal left crossbar": 7,
    "Goal left post left ": 8,            # upstream has trailing space
    "Goal left post left": 8,             # project no-space alias
    "Goal left post right": 9,
    "Goal right crossbar": 10,
    "Goal right post left": 11,
    "Goal right post right": 12,
    "Middle line": 13,
    "Side line bottom": 14,
    "Side line left": 15,
    "Side line right": 16,
    "Side line top": 17,
    "Small rect. left bottom": 18,
    "Small rect. left main": 19,
    "Small rect. left side": 19,
    "Small rect. left top": 20,
    "Small rect. right bottom": 21,
    "Small rect. right main": 22,
    "Small rect. right side": 22,
    "Small rect. right top": 23,
}


def project_kp_to_upstream(proj_id: int) -> int:
    """0-indexed project id → 1-indexed upstream id."""
    if proj_id in _PROJ_TO_UPSTREAM_KP_OVERRIDE:
        return _PROJ_TO_UPSTREAM_KP_OVERRIDE[proj_id]
    return proj_id + 1


class PnLCalibIdMap:
    """Bridge project keypoint/line IDs to upstream's 1-indexed scheme.

    Constructing an instance asserts the override table is consistent
    with the parametric pitch geometry — if SoccerPitch ever flips
    crossbar T/B ordering or PITCH_POINT_TO_PNLCALIB_ID is edited, the
    constructor raises.
    """

    def __init__(self, pitch: "SoccerPitch | None" = None, tol: float = 1e-3):
        # Lazy import: see module-top note on circular dependency.
        from goalinsight.annotation.pitch.geometry import SoccerPitch
        from goalinsight.annotation.pitch.keypoints import _build_pitch_points

        if pitch is None:
            pitch = SoccerPitch()
        self.pitch = pitch
        self.tol = tol

        # Build the upstream-indexed pitch table for this pitch's dims.
        pitch_dims = {
            k: getattr(pitch, k.upper())
            for k in [
                "pitch_length", "pitch_width", "penalty_area_width",
                "penalty_area_length", "goal_area_width", "goal_area_length",
                "goal_line_to_penalty_mark", "center_circle_radius",
                "goal_height", "goal_length",
            ]
        }
        table = build_keypoint_table(pitch_dims)
        self._kp_table = table["keypoint_world_coords_2D"]
        self._kp_aux_table = table["keypoint_aux_world_coords_2D"]
        self._goal_height = table["goal_height"]

        # Verify every named keypoint round-trips through the override.
        # ``PITCH_POINTS`` is built from the *active* pitch; rebuild it
        # locally so we don't depend on whatever was last set.
        pitch_points = _build_pitch_points(pitch)
        self._verify_consistency(pitch_points)

    def _verify_consistency(self, pitch_points: dict[str, np.ndarray]) -> None:
        """Cross-check name → world via two paths agrees within tol.

        D-shape penalty areas (futsal) extend ``L/R_PENALTY_AREA_TL/BL/
        TR/BR_CORNER`` to mean the arc-tangent-on-goal-line points
        (±(g_w + pa_d)) instead of the rect corners at ±pa_w — the
        upstream table doesn't model arc tangents, so these 4 names
        intentionally disagree. Skip them on D-PA pitches; everything
        else still has to round-trip.
        """
        from goalinsight.annotation.pitch.keypoints import (
            PITCH_POINT_TO_PNLCALIB_ID,
        )
        pa_shape = getattr(self.pitch, "PENALTY_AREA_SHAPE", "rect")
        d_pa_overrides = {
            "L_PENALTY_AREA_TL_CORNER", "L_PENALTY_AREA_BL_CORNER",
            "R_PENALTY_AREA_TR_CORNER", "R_PENALTY_AREA_BR_CORNER",
        } if pa_shape == "d" else set()
        bad: list[str] = []
        for name, proj_id in PITCH_POINT_TO_PNLCALIB_ID.items():
            if name in d_pa_overrides:
                continue
            up_id = project_kp_to_upstream(proj_id)
            if up_id < 1 or up_id > len(self._kp_table):
                bad.append(f"{name}: upstream id {up_id} out of range")
                continue

            xy_table = self._kp_table[up_id - 1]
            xy_named = pitch_points[name]
            # Both upstream table and project geometry use the same
            # convention here (centered, but upstream is y-DOWN, project
            # is y-UP). Flip y when comparing.
            if not (np.isfinite(xy_table[0]) and np.isfinite(xy_table[1])):
                continue  # NaN aux points are fine
            xt, yt = float(xy_table[0]), -float(xy_table[1])  # to y-up
            xn, yn = float(xy_named[0]), float(xy_named[1])
            if abs(xt - xn) > self.tol or abs(yt - yn) > self.tol:
                bad.append(
                    f"{name}: proj_id={proj_id} up_id={up_id} "
                    f"table=({xt:.3f},{yt:.3f}) named=({xn:.3f},{yn:.3f})"
                )
        if bad:
            raise RuntimeError(
                "pnlcalib_orig.id_mapping consistency check failed:\n"
                + "\n".join(bad)
            )

    # ------------------------------------------------------------------
    # Public conversion helpers.
    # ------------------------------------------------------------------
    def name_to_upstream_id(self, name: str) -> int | None:
        """Pitch-point name → upstream 1-indexed id, or None if unknown."""
        from goalinsight.annotation.pitch.keypoints import (
            PITCH_POINT_TO_PNLCALIB_ID,
        )
        proj_id = PITCH_POINT_TO_PNLCALIB_ID.get(name)
        if proj_id is None:
            return None
        return project_kp_to_upstream(proj_id)

    def world_xyz_to_upstream_id(
        self, world_xyz: tuple[float, float, float], tol: float = 0.05,
    ) -> int | None:
        """Reverse-lookup an upstream id by world (x, y, z) match.

        Searches both the main 57-keypoint table and the 16-aux table.
        Y-axis is flipped (project y-up → upstream y-down) before match.
        Returns the *upstream 1-indexed* id, or None on no match.
        """
        x_proj, y_proj, z_proj = world_xyz
        # Convert to upstream y-down convention.
        y_up_table = -y_proj  # upstream y-down: top = small y
        # Note: upstream tables are centered + y-down. After "y-up table"
        # conversion via `-y_proj`, both tables share the same y-down
        # frame so a direct distance check works.
        # ``z_proj`` is in y-up project frame too; upstream uses
        # z = -goal_height for crossbars, same convention.
        target = np.array([x_proj, y_up_table], dtype=float)

        best_id, best_dist = None, float("inf")

        # Main table (1..57); only allow non-ground IDs when z != 0.
        is_ground = abs(z_proj) < 1e-6
        for i, xy in enumerate(self._kp_table):
            up_id = i + 1
            if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                continue
            non_ground = up_id in NON_GROUND_UPSTREAM_IDS
            if is_ground and non_ground:
                continue
            if not is_ground and not non_ground:
                continue
            d = float(np.hypot(xy[0] - target[0], xy[1] - target[1]))
            if d < best_dist:
                best_dist = d
                best_id = up_id

        # Aux table (58..73); ground only.
        if is_ground:
            for i, xy in enumerate(self._kp_aux_table):
                up_id = 58 + i
                if not (np.isfinite(xy[0]) and np.isfinite(xy[1])):
                    continue
                d = float(np.hypot(xy[0] - target[0], xy[1] - target[1]))
                if d < best_dist:
                    best_dist = d
                    best_id = up_id

        if best_dist > tol:
            return None
        return best_id

    def detector_output_to_upstream_dict(
        self, keypoints: list[dict[str, Any]],
    ) -> dict[int, dict[str, float]]:
        """Convert ``KeypointDetector.detect(convert_to_soccernet=False)``
        output (list of {id, x, y, confidence}, project 0-indexed) to the
        upstream ``{1-indexed-id: {x, y, p}}`` dict that
        ``FramebyFrameCalib.update`` expects.
        """
        out: dict[int, dict[str, float]] = {}
        for kp in keypoints:
            proj_id = kp.get("id")
            if proj_id is None:
                continue
            up_id = project_kp_to_upstream(int(proj_id))
            out[up_id] = {
                "x": float(kp["x"]),
                "y": float(kp["y"]),
                "p": float(kp.get("confidence", 1.0)),
            }
        return out

    def detector_lines_to_upstream_dict(
        self, lines: list[dict[str, Any]],
    ) -> dict[int, dict[str, float]]:
        """Convert ``LineDetector.detect`` output to the upstream
        ``{1-indexed-line-id: {x_1, y_1, x_2, y_2}}`` dict.
        """
        out: dict[int, dict[str, float]] = {}
        for ln in lines:
            name = ln.get("class_name")
            if name is None:
                continue
            up_id = LINE_NAME_TO_UPSTREAM_LINE_ID.get(name)
            if up_id is None:
                continue
            out[up_id] = {
                "x_1": float(ln["x1"]),
                "y_1": float(ln["y1"]),
                "x_2": float(ln["x2"]),
                "y_2": float(ln["y2"]),
            }
        return out
