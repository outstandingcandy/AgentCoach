"""Pitch-line overlay for annotation previews.

Builds the world-space polylines that describe the active pitch
(outline, center line, center circle, penalty area, goal frames,
center spot + penalty marks) and delegates projection + drawing to
``utils.projection.draw_world_polylines`` so the annotate page and
field-registration / tracking visualisations all paint through the
same code path.

Two projection modes are supported:

- ``cam={"K", "dist", "rvec", "tvec"}`` — full PnP camera model
  (cv2.projectPoints with distortion). Use this whenever the solver
  produced non-trivial dist_coeffs.
- ``cam=None`` + ``H`` — planar (homography) projection only, for the
  early "no PnP yet" case. Ignores distortion; the overlay drifts at
  the edges if dist != 0.
"""

import math

import cv2
import numpy as np

from ..utils.projection import (
    draw_world_landmarks,
    draw_world_polylines,
)
from . import pitch_constants


def _project_world_with_depth(
    world_2d: tuple[float, float],
    H_world_to_image: np.ndarray,
) -> tuple[float, float] | None:
    """Project a ground-plane world point and return None if behind horizon.

    H_world_to_image is `inv(H_image_to_world)`. The homogeneous w-coordinate
    (third row) goes through zero on the horizon line; world points beyond
    the horizon have w < 0 and ``cv2.perspectiveTransform`` would mirror
    them (visually they show up "in the sky"). Skip those.
    """
    homo = H_world_to_image @ np.array(
        [world_2d[0], world_2d[1], 1.0], dtype=np.float64,
    )
    if homo[2] <= 1e-6:
        return None
    return (float(homo[0] / homo[2]), float(homo[1] / homo[2]))


def _build_pitch_polylines() -> tuple[list[np.ndarray], list[tuple[float, float, float]]]:
    """Compose the active pitch as (polylines, landmark dots).

    Returns:
        polylines: list of ``(N, 3)`` float arrays in world coords (z=0
            for ground markings; goal-frame edges carry z = +GOAL_HEIGHT
            for the crossbar endpoints). Each entry is drawn as a single
            connected line.
        landmarks: list of ``(x, y, z)`` world points to render as
            filled dots (centre spot + penalty marks).
    """
    pitch = pitch_constants.get_active_pitch()
    L = pitch.PITCH_LENGTH / 2
    pen = pitch.GOAL_LINE_TO_PENALTY_MARK
    ccr = pitch.CENTER_CIRCLE_RADIUS
    pa_shape = getattr(pitch, "PENALTY_AREA_SHAPE", "rect")

    polylines: list[np.ndarray] = []

    # Straight lines from PITCH_LINES (touchlines, halfway, goal-area
    # rectangles, and — only for rect-PA pitches — the four PA
    # rectangle sides). For D-PA pitches the PA outline is two
    # post-centered arcs + a chord, supplied by build_d_penalty_arcs.
    pa_rect_names = {
        "penalty_left_top", "penalty_left_bottom", "penalty_left_front",
        "penalty_right_top", "penalty_right_bottom", "penalty_right_front",
    }
    for name, ((wx1, wy1), (wx2, wy2)) in pitch_constants.PITCH_LINES.items():
        if pa_shape == "d" and name in pa_rect_names:
            continue
        polylines.append(np.array(
            [[wx1, wy1, 0.0], [wx2, wy2, 0.0]], dtype=np.float64,
        ))

    # Centre circle (always present).
    theta = np.linspace(0.0, 2 * math.pi, 96, dtype=np.float64)
    cc = np.column_stack([
        ccr * np.cos(theta),
        ccr * np.sin(theta),
        np.zeros_like(theta),
    ])
    polylines.append(cc)

    if pa_shape == "d":
        for poly in pitch_constants.build_d_penalty_arcs(pitch):
            arr = np.array(poly, dtype=np.float64)
            polylines.append(np.column_stack([arr, np.zeros(len(arr))]))

        # Goal frames. Pull world coords from PITCH_POINTS so post-top z
        # and post-base z come from the same dict the solver consumed
        # (positive-up, no flip needed).
        from .pitch import keypoints as _pk
        for names in (
            ("L_GOAL_TL_POST", "L_GOAL_TR_POST",
             "L_GOAL_BL_POST", "L_GOAL_BR_POST"),
            ("R_GOAL_TL_POST", "R_GOAL_TR_POST",
             "R_GOAL_BL_POST", "R_GOAL_BR_POST"),
        ):
            pts: dict[str, tuple[float, float, float]] = {}
            for label, kp_name in zip(("tl", "tr", "bl", "br"), names):
                p = _pk.PITCH_POINTS.get(kp_name)
                if p is None or len(p) < 3:
                    break
                pts[label] = (float(p[0]), float(p[1]), float(p[2]))
            if len(pts) != 4:
                continue
            tl, tr, bl, br = pts["tl"], pts["tr"], pts["bl"], pts["br"]
            for a, b in [(tl, bl), (tr, br), (tl, tr), (bl, br)]:
                polylines.append(np.array([a, b], dtype=np.float64))
    else:
        # 11-a-side penalty arc, centered on the penalty mark with the
        # PA-front line clipping the chord. Half-angle by geometry:
        # cos(theta) = (pa_d - pen) / ccr.
        pa_d = pitch.PENALTY_AREA_LENGTH
        dx = pa_d - pen
        ratio = dx / ccr if ccr > 0 else 1.0
        if -1.0 < ratio < 1.0:
            half = math.acos(ratio)
            n = 40
            t_left = np.linspace(-half, half, n)
            t_right = np.linspace(math.pi - half, math.pi + half, n)
            for sign, t in ((-1.0, t_left), (1.0, t_right)):
                cx = sign * (L - pen)
                arc = np.column_stack([
                    cx + ccr * np.cos(t),
                    ccr * np.sin(t),
                    np.zeros_like(t),
                ])
                polylines.append(arc)

    landmarks = [(0.0, 0.0, 0.0), (-L + pen, 0.0, 0.0), (L - pen, 0.0, 0.0)]
    return polylines, landmarks


def _draw_planar(
    out: np.ndarray,
    H: np.ndarray,
    polylines: list[np.ndarray],
    landmarks: list[tuple[float, float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Planar-homography fallback for the early "no PnP yet" case.

    Walks the same polylines + landmarks as the PnP path but goes
    through ``inv(H)`` instead of cv2.projectPoints. Ignores any
    distortion (the planar H is pinhole-only).
    """
    try:
        H_world_to_image = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return
    h, w = out.shape[:2]
    clip = (0, 0, w, h)
    for poly in polylines:
        prev: tuple[int, int] | None = None
        for x, y, _z in poly:  # planar H ignores z
            p = _project_world_with_depth((float(x), float(y)), H_world_to_image)
            if p is None:
                prev = None
                continue
            cur = (int(p[0]), int(p[1]))
            if prev is not None:
                ok, q1, q2 = cv2.clipLine(clip, prev, cur)
                if ok:
                    cv2.line(out, q1, q2, color, thickness)
            prev = cur
    for wx, wy, _wz in landmarks:
        p = _project_world_with_depth((wx, wy), H_world_to_image)
        if p is None:
            continue
        px, py = int(p[0]), int(p[1])
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(out, (px, py), 4, color, -1)


def render_pitch_projection(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    cam: dict | None = None,
) -> np.ndarray:
    """Draw projected pitch lines on a BGR frame.

    Two projection paths:

    - ``cam={"K", "dist", "rvec", "tvec"}``: projects each world point
      through ``cv2.projectPoints`` with the distortion model. Same code
      path the pipeline's field-registration visualisation uses, so the
      yellow overlay on the annotate page and the FR stage-1 jpgs are
      pixel-identical for the same camera.
    - ``cam=None`` (fallback): uses the image↔world planar homography H.
      Pinhole-only, ignores any distortion. Cheaper, used when no full
      PnP state is available yet (right after annotation, before the
      first Compute click).

    Endpoints whose camera-space z ≤ 0 (behind the camera) are skipped
    so the overlay doesn't paint phantom sky.
    """
    out = frame_bgr.copy()
    polylines, landmarks = _build_pitch_polylines()

    if cam is not None:
        K = np.asarray(cam["K"], dtype=np.float64)
        dist = np.asarray(cam["dist"], dtype=np.float64).ravel()
        rvec = np.asarray(cam["rvec"], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(cam["tvec"], dtype=np.float64).reshape(3, 1)
        draw_world_polylines(
            out, polylines, rvec, tvec, K, dist,
            color=color, thickness=thickness,
        )
        draw_world_landmarks(
            out, np.array(landmarks, dtype=np.float64),
            rvec, tvec, K, dist, color=color, radius=4,
        )
    else:
        if H is None:
            return out
        _draw_planar(out, H, polylines, landmarks, color, thickness)

    return out
