"""Minimal pitch-line overlay for annotation previews.

Given an image->world homography, project the pitch lines from
pitch_constants onto a BGR frame. No distortion model — this is just a
sanity-check visualization for the annotator.
"""

import math

import cv2
import numpy as np

from . import pitch_constants


def _project_world_with_depth(
    world_2d: tuple[float, float],
    H_world_to_image: np.ndarray,
) -> tuple[float, float] | None:
    """Project a ground-plane world point and return None if behind horizon.

    H_world_to_image is `inv(H_image_to_world)`. The homogeneous w-coordinate
    (third row) goes through zero on the horizon line; world points beyond the
    horizon have w < 0 and ``cv2.perspectiveTransform`` would mirror them
    (visually they show up "in the sky"). Skip those.
    """
    homo = H_world_to_image @ np.array([world_2d[0], world_2d[1], 1.0],
                                       dtype=np.float64)
    if homo[2] <= 1e-6:
        return None
    return (float(homo[0] / homo[2]), float(homo[1] / homo[2]))


def _make_pnp_projector(cam: dict):
    """Return a ``project_xy(x, y) -> (px, py) | None`` closure that uses
    the full (K, dist, rvec, tvec) from the most recent solve.

    Each world (x, y, 0) is projected via ``cv2.projectPoints`` with the
    distortion model applied. Points whose camera-space z is non-positive
    (behind the camera) return None so the overlay doesn't paint phantom
    sky. We compute the camera-space z by hand because OpenCV's projector
    happily returns garbage for back-facing points.
    """
    K = np.asarray(cam["K"], dtype=np.float64)
    dist = np.asarray(cam["dist"], dtype=np.float64).ravel()
    rvec = np.asarray(cam["rvec"], dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(cam["tvec"], dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)

    def project_xy(x: float, y: float, z: float = 0.0):
        pt = np.array([x, y, z], dtype=np.float64)
        cam_pt = R @ pt + tvec.ravel()
        if cam_pt[2] <= 1e-6:
            return None
        # cv2.projectPoints handles the dist; we just feed the world pt.
        proj, _ = cv2.projectPoints(
            pt.reshape(1, 1, 3), rvec, tvec, K, dist,
        )
        px, py = proj.reshape(2)
        return float(px), float(py)

    return project_xy


def _make_planar_projector(H: np.ndarray):
    """Return a ``project_xy(x, y) -> (px, py) | None`` closure that
    inverts the image↔world planar homography. Pinhole-only fallback
    used when no full PnP state is available."""
    try:
        H_world_to_image = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        H_world_to_image = None

    def project_xy(x: float, y: float, z: float = 0.0):
        if H_world_to_image is None:
            return None
        # Planar H ignores z (it's an image↔world ground-plane mapping),
        # so we discard z here. Callers that need true 3D projection
        # (e.g. goal-frame crossbars) must use the PnP projector path.
        return _project_world_with_depth((x, y), H_world_to_image)

    return project_xy


def _project_polyline(
    img: np.ndarray,
    project_xy,
    samples_xy: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    """Project a y-up world polyline and draw it segment-by-segment.

    ``samples_xy`` is an Nx2 float array of (x, y) world coords in metres.
    Adjacent samples that both project in front of the horizon get joined
    by a clipped line segment; pairs that cross or sit behind the horizon
    are skipped so the overlay doesn't paint phantom sky.
    """
    h, w = img.shape[:2]
    prev = None
    for x, y in samples_xy:
        p = project_xy(float(x), float(y))
        if p is None:
            prev = None
            continue
        if prev is not None:
            ok, q1, q2 = cv2.clipLine(
                (0, 0, w, h),
                (int(prev[0]), int(prev[1])),
                (int(p[0]), int(p[1])),
            )
            if ok:
                cv2.line(img, q1, q2, color, thickness)
        prev = p


def render_pitch_projection(
    frame_bgr: np.ndarray,
    H: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
    cam: dict | None = None,
) -> np.ndarray:
    """Draw projected pitch lines on a BGR frame.

    Two projection paths:

    - ``cam=None`` (default): uses the image↔world planar homography H.
      Pinhole-only, ignores any distortion. Cheaper, used when no full
      PnP state is available.

    - ``cam={"K", "dist", "rvec", "tvec"}``: projects each world point
      through ``cv2.projectPoints`` with the distortion model. Use this
      whenever the solver fit non-trivial dist_coeffs — otherwise the
      planar H ignores them and the overlay drifts at the edges.

    Endpoints whose camera-space z ≤ 0 (behind the camera) are skipped
    so the overlay doesn't paint phantom sky.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if cam is not None:
        project_xy = _make_pnp_projector(cam)
    else:
        if H is None:
            return out
        project_xy = _make_planar_projector(H)

    pitch = pitch_constants.get_active_pitch()
    L = pitch.PITCH_LENGTH / 2
    pen = pitch.GOAL_LINE_TO_PENALTY_MARK
    ccr = pitch.CENTER_CIRCLE_RADIUS
    pa_shape = getattr(pitch, "PENALTY_AREA_SHAPE", "rect")

    # Straight lines (touchlines, halfway, goal-area rectangles, and —
    # only for rect-PA pitches — the four PA rectangle sides). For D-PA
    # pitches the PA outline is two post-centered arcs + a chord; we
    # filter out the rectangle entries from PITCH_LINES and synthesize
    # the arcs below.
    pa_rect_names = {
        "penalty_left_top", "penalty_left_bottom", "penalty_left_front",
        "penalty_right_top", "penalty_right_bottom", "penalty_right_front",
    }
    for name, ((wx1, wy1), (wx2, wy2)) in pitch_constants.PITCH_LINES.items():
        if pa_shape == "d" and name in pa_rect_names:
            continue
        p1 = project_xy(wx1, wy1)
        p2 = project_xy(wx2, wy2)
        if p1 is None or p2 is None:
            continue
        ok, q1, q2 = cv2.clipLine(
            (0, 0, w, h),
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
        )
        if ok:
            cv2.line(out, q1, q2, color, thickness)

    # Center circle — always present, regardless of pitch shape.
    theta = np.linspace(0.0, 2 * math.pi, 96, dtype=np.float64)
    cc_xy = np.column_stack([ccr * np.cos(theta), ccr * np.sin(theta)])
    _project_polyline(out, project_xy, cc_xy, color, thickness)

    if pa_shape == "d":
        # D-shape PA. World-space arcs+chords come from the shared
        # ``build_d_penalty_arcs`` helper so this renderer and the
        # pipeline's ``utils.pitch.get_pitch_template_points`` emit the
        # exact same world points — preventing PA-dimension drift
        # between the annotate page and stage-1 visualization.
        for poly in pitch_constants.build_d_penalty_arcs(pitch):
            arr = np.array(poly, dtype=np.float64)
            _project_polyline(out, project_xy, arr, color, thickness)

        # Goal frames. Both PnP and viz must use the SAME coordinate
        # system — earlier this block hard-coded crossbar z=+GOAL_HEIGHT,
        # which contradicted ``pitch/geometry.py`` (z=-GOAL_HEIGHT for
        # ``*_GOAL_T*_POST`` keypoints) and produced a mismatched
        # rendering whenever the user had standlone post-top
        # annotations. Pull the 4 actual keypoint world coords from
        # ``_pk.PITCH_POINTS`` so post-top z and post-base z come from
        # the same dict the solver consumed at compute time.
        if cam is not None:
            from .pitch import keypoints as _pk
            for side, names in (
                ("left", ("L_GOAL_TL_POST", "L_GOAL_TR_POST",
                          "L_GOAL_BL_POST", "L_GOAL_BR_POST")),
                ("right", ("R_GOAL_TL_POST", "R_GOAL_TR_POST",
                           "R_GOAL_BL_POST", "R_GOAL_BR_POST")),
            ):
                pts = {}
                missing = False
                for label, kp_name in zip(("tl", "tr", "bl", "br"), names):
                    p = _pk.PITCH_POINTS.get(kp_name)
                    if p is None or len(p) < 3:
                        missing = True
                        break
                    # Same z-sign flip as collect_pnp_points: the
                    # keypoint table's z = -GOAL_HEIGHT is the
                    # negative-up legacy convention; the PnP path
                    # solves in positive-up. Flip for projection so
                    # the rendered post-top matches the camera the
                    # solver actually found.
                    pts[label] = (float(p[0]), float(p[1]), -float(p[2]))
                if missing:
                    continue
                tl, tr, bl, br = pts["tl"], pts["tr"], pts["bl"], pts["br"]
                # 4 edges of the goal frame. The goal-line segment
                # (bl→br) visually anchors the frame onto the painted
                # goal line — without it the posts look like they
                # float just below the touchline.
                for (a, b) in [(tl, bl), (tr, br), (tl, tr), (bl, br)]:
                    p1 = project_xy(*a)
                    p2 = project_xy(*b)
                    if p1 is None or p2 is None:
                        continue
                    ok, q1, q2 = cv2.clipLine(
                        (0, 0, w, h),
                        (int(p1[0]), int(p1[1])),
                        (int(p2[0]), int(p2[1])),
                    )
                    if ok:
                        cv2.line(out, q1, q2, color, thickness)
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
                    cx + ccr * np.cos(t), ccr * np.sin(t),
                ])
                _project_polyline(out, project_xy, arc, color, thickness)

    # Landmarks (center spot + penalty marks).
    for wx, wy, r in [(0.0, 0.0, 5), (-L + pen, 0.0, 4), (L - pen, 0.0, 4)]:
        pt = project_xy(wx, wy)
        if pt is not None and 0 <= pt[0] < w and 0 <= pt[1] < h:
            cv2.circle(out, (int(pt[0]), int(pt[1])), r, color, -1)

    return out
