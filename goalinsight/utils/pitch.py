"""Pitch template constants, projection helpers, and top-down pitch drawing.

Extracted from stage1.py to be shared across pipeline stages.

Default pitch dimensions come from
``goalinsight.annotation.pitch_constants.get_active_pitch()`` (set by the
pipeline at config-load time), so non-FIFA pitches are honored automatically.
"""

import cv2
import numpy as np

from ..annotation import pitch_constants
from .projection import project_points_with_visibility


# Keypoint IDs forming pitch line segments (connect consecutive IDs)
PITCH_LINE_KEYPOINTS = [
    [0, 1, 2],                              # top boundary
    [27, 28, 29],                            # bottom boundary
    [0, 3, 7, 11, 15, 19, 23, 27],          # left sideline
    [2, 6, 10, 13, 17, 22, 26, 29],         # right sideline
    [1, 28],                                 # center line
    [3, 4], [23, 24], [24, 4],              # left penalty area
    [6, 5], [26, 25], [25, 5],              # right penalty area
    [7, 8], [19, 20], [20, 8],             # left goal area
    [10, 9], [22, 21], [21, 9],            # right goal area
    [31, 48, 38, 51, 42, 53, 34, 52, 41, 49, 37, 47, 31],  # center circle
    [30, 36, 46, 40, 33],                   # left penalty arc
    [32, 39, 54, 43, 35],                   # right penalty arc
]


def get_pitch_template_points(pitch_length=None, pitch_width=None):
    """Get key points on the pitch template for visualization.

    Single source of truth: ``annotation.pitch_constants.PITCH_LINES``
    (straight segments) + the active-pitch dimensions for arcs and the
    D-shape penalty area. Earlier this function had its own duplicate
    geometry built from ``pitch.PITCH_LENGTH`` etc. — every time a new
    pitch shape was added (e.g. futsal's D), two places had to be
    updated and they drifted. Now this one pulls everything from the
    same dict the annotate page uses, so non-FIFA pitches (kids,
    futsal) stay consistent across all renderers.

    Output remains the legacy ``{name: [(x, y), ...]}`` shape that
    ``_draw_physical_calibration`` consumes; we just synthesize the
    entries from the shared source.

    Args:
        pitch_length: Optional override. When provided MUST match the
            active pitch's length — caller is responsible for setting
            the active pitch before calling. Logged at the top of
            ``physical_runner`` / ``fixed_camera_runner``.
        pitch_width: Same caveat as ``pitch_length``.
    """
    active = pitch_constants.get_active_pitch()
    # Sanity log if caller passed something different — every call site
    # I checked already does ``set_active_pitch(...)`` first.
    pl = pitch_length or active.PITCH_LENGTH
    pw = pitch_width or active.PITCH_WIDTH
    half_l = pl / 2
    half_w = pw / 2

    CR = active.CENTER_CIRCLE_RADIUS
    PA_DEPTH = active.PENALTY_AREA_LENGTH
    PA_HW = active.PENALTY_AREA_WIDTH / 2.0
    PS_DIST = active.GOAL_LINE_TO_PENALTY_MARK
    pa_shape = getattr(active, "PENALTY_AREA_SHAPE", "rect")

    # Re-shape annotation's flat line dict into the legacy
    # multi-line-per-shape entries the renderer expects. Each value is
    # a polyline (list of (x, y)) and consecutive points get connected.
    lines = pitch_constants.PITCH_LINES
    template: dict = {
        'pitch_outline': [
            (-half_l, -half_w), (half_l, -half_w),
            (half_l, half_w), (-half_l, half_w), (-half_l, -half_w),
        ],
        'center_line': [lines['center_line'][0], lines['center_line'][1]],
        'center_circle': [
            (CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, 36)
        ],
        'left_goal_area': [
            lines['goal_area_left_bottom'][0],
            lines['goal_area_left_bottom'][1],
            lines['goal_area_left_front'][0],
            lines['goal_area_left_top'][1],
            lines['goal_area_left_top'][0],
        ],
        'right_goal_area': [
            lines['goal_area_right_bottom'][0],
            lines['goal_area_right_bottom'][1],
            lines['goal_area_right_front'][0],
            lines['goal_area_right_top'][1],
            lines['goal_area_right_top'][0],
        ],
    }

    if pa_shape == "d":
        # Futsal D-shape PA. Sampling lives in
        # ``pitch_constants.build_d_penalty_arcs`` so annotate-side
        # rendering (viz.py) and pipeline-side rendering (this file)
        # consume the *same* world points — drift between the two
        # used to make the projected PA dimensions look different.
        arcs = pitch_constants.build_d_penalty_arcs(active)
        # arcs[0..3] are post-centered quarter-arcs (left-top, left-bot,
        # right-top, right-bot in that order); arcs[4..5] are the two
        # chords at x = ±(L - pa_d). The topdown / projection renderer
        # consumes each polyline as a connected line strip, so we keep
        # the arcs separate (otherwise the line would cross the field
        # diagonally on the way from one arc to the next).
        template['left_penalty_d_arc_top'] = arcs[0]
        template['left_penalty_d_arc_bot'] = arcs[1]
        template['right_penalty_d_arc_top'] = arcs[2]
        template['right_penalty_d_arc_bot'] = arcs[3]
        template['left_penalty_d_chord'] = arcs[4]
        template['right_penalty_d_chord'] = arcs[5]
    else:
        # 11-a-side rectangle PA + penalty-mark-centered arc.
        template['left_penalty'] = [
            lines['penalty_left_bottom'][0],
            lines['penalty_left_bottom'][1],
            lines['penalty_left_front'][0],
            lines['penalty_left_top'][1],
            lines['penalty_left_top'][0],
        ]
        template['right_penalty'] = [
            lines['penalty_right_bottom'][0],
            lines['penalty_right_bottom'][1],
            lines['penalty_right_front'][0],
            lines['penalty_right_top'][1],
            lines['penalty_right_top'][0],
        ]
        # Penalty arc — half-angle = acos((PA_DEPTH - PS_DIST) / CR).
        _arc_dx = PA_DEPTH - PS_DIST
        if CR > 0 and -1.0 < _arc_dx / CR < 1.0:
            _arc_half = float(np.arccos(_arc_dx / CR))
        else:
            _arc_half = float(np.pi / 2.0)
        template['left_penalty_arc'] = [
            (-half_l + PS_DIST + CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(-_arc_half, _arc_half, 20)
        ]
        template['right_penalty_arc'] = [
            (half_l - PS_DIST - CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(-_arc_half, _arc_half, 20)
        ]
    return template


def project_pitch_to_image(H, template_points, camera_params=None):
    """Project pitch template points to image.

    Uses full camera model (cv2.projectPoints) when camera_params is available,
    otherwise falls back to homography projection.
    """
    if camera_params is not None:
        return _project_with_camera_model(template_points, camera_params)

    projected = {}
    for name, points in template_points.items():
        proj_points = []

        # Check Z coordinate equivalent in homography space.
        pts_h = np.array([[pt[0], pt[1], 1.0] for pt in points])
        img_h_all = (H @ pts_h.T).T

        for img_h in img_h_all:
            if img_h[2] <= 0.1:
                proj_points.append(None)
            else:
                img_pt = img_h[:2] / img_h[2]
                proj_points.append((int(img_pt[0]), int(img_pt[1])))
        projected[name] = proj_points
    return projected


def _project_with_camera_model(template_points, camera_params):
    """Project pitch template using full pinhole + distortion model.

    Long straight world segments (e.g. touchlines, which are given as just
    two corner endpoints) must be densified BEFORE projection: with real
    lens distortion a straight world line maps to a *curved* image line, so
    projecting only the endpoints and connecting them with a straight image
    segment cuts the corner — the midpoint can miss the true projected line
    by tens or even hundreds of pixels (measured ~480 px at k1≈0.7). This
    is also why a landmark that genuinely sits on the line (e.g. the
    centre-line/touchline intersection) appeared to float off the drawn
    touchline. Sample every segment at a fixed world-space spacing so the
    projected polyline follows the true distorted curve and passes through
    such landmarks.
    """
    rvec = camera_params["rvec"]
    tvec = camera_params["tvec"]
    K = camera_params["K"]
    dist = camera_params["dist_coeffs"]

    # Max world-space gap (metres) between consecutive samples along a
    # segment. ~0.5 m keeps even a full-length touchline smooth without
    # exploding the point count.
    MAX_SEG_M = 0.5

    projected = {}
    for name, points in template_points.items():
        # Densify: insert intermediate points along each consecutive pair so
        # distortion is captured along the whole segment, not just at nodes.
        dense: list[tuple[float, float]] = []
        for i, pt in enumerate(points):
            if i == 0:
                dense.append((pt[0], pt[1]))
                continue
            prev = points[i - 1]
            seg = np.hypot(pt[0] - prev[0], pt[1] - prev[1])
            n_sub = max(1, int(np.ceil(seg / MAX_SEG_M)))
            for s in range(1, n_sub + 1):
                t = s / n_sub
                dense.append((
                    prev[0] + t * (pt[0] - prev[0]),
                    prev[1] + t * (pt[1] - prev[1]),
                ))
        obj = np.array([[x, y, 0.0] for x, y in dense], dtype=np.float64)
        img_pts, in_front = project_points_with_visibility(obj, rvec, tvec, K, dist)
        proj_points = [
            (int(p[0]), int(p[1])) if vis else None
            for p, vis in zip(img_pts, in_front)
        ]
        projected[name] = proj_points
    return projected


def _draw_topdown_pitch(height, width, result, keypoints, calibrator, keypoint_mapper,
                        pitch_length=None, pitch_width=None):
    """Draw top-down pitch diagram with keypoints at their known world positions.

    The y-axis uses a fixed orientation matching the reference image
    (docs/pitch_keypoints_reference.png): positive y points upward.

    Args:
        height: Output image height.
        width: Output image width.
        result: Calibration result dict (or None).
        keypoints: Detected keypoints list.
        calibrator: FramebyFrameCalib instance.
        keypoint_mapper: KeypointMapper instance for world coordinate lookup.
        pitch_length: Pitch length in meters (default: active pitch length).
        pitch_width: Pitch width in meters (default: active pitch width).

    Returns:
        Top-down pitch image (BGR).
    """
    from ..field_registration.pnlcalib import KeypointMapper

    active = pitch_constants.get_active_pitch()
    pl = pitch_length if pitch_length is not None else active.PITCH_LENGTH
    pw = pitch_width if pitch_width is not None else active.PITCH_WIDTH

    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)  # Forest green background

    # Pitch coordinate -> pixel mapping (with margin)
    margin = 8  # meters of margin around pitch
    scale = min(
        (width - 2) / (pl + 2 * margin),
        (height - 2) / (pw + 2 * margin),
    )
    ox = width / 2   # pixel center
    oy = height / 2

    def w2p(wx, wy):
        """World (meters, center origin) -> pixel. Positive y upward."""
        px = int(ox + wx * scale)
        py = int(oy - wy * scale)
        return (px, py)

    line_color = (255, 255, 255)
    lw = max(1, int(scale * 0.3))

    # Pitch markings come from the single shared renderer in
    # ``annotation.pitch_diagram``. Same code path the annotate page and
    # tracking minimap use — PA shape, goal area, center circle, and
    # penalty arc stay consistent across every top-down rendering.
    from ..annotation.pitch_diagram import draw_pitch_structure
    draw_pitch_structure(
        pitch, w2p, scale=scale, color=line_color, thickness=lw,
        draw_landmarks=True,
        landmark_radius=max(2, int(0.3 * scale)),
        draw_arcs=True,
    )


    if result is None:
        return pitch

    # Build inlier/outlier info
    inlier_img_set = set()
    if "img_pts" in result and "inlier_mask" in result:
        for pt, is_inlier in zip(result["img_pts"], result["inlier_mask"]):
            if is_inlier:
                inlier_img_set.add((round(float(pt[0]), 1), round(float(pt[1]), 1)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    r = max(3, int(0.8 * scale))

    # Draw line intersections at their known world positions
    for inter in calibrator.line_intersections:
        wx, wy = inter["world_x"], inter["world_y"]
        px, py = w2p(wx, wy)
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(pitch, (px, py), r, (255, 0, 255), 2)

    # --- Camera FOV and projected pitch lines on top-down view ---
    cam_params = result.get("camera_params")
    if cam_params is not None:
        img_w, img_h = calibrator.image_size
        K_cam = cam_params["K"]
        dist_cam = cam_params["dist_coeffs"]
        R_cam = cam_params["R"]
        tvec_cam = cam_params["tvec"].flatten()

        def _img_to_world(img_pts_list, max_range=80.0):
            """Back-project image points to ground plane (z=0) via camera model."""
            pts = np.array(img_pts_list, dtype=np.float64).reshape(-1, 1, 2)
            pts_norm = cv2.undistortPoints(pts, K_cam, dist_cam)
            cam_center = -R_cam.T @ tvec_cam
            world = []
            for xn, yn in pts_norm.reshape(-1, 2):
                ray_world = R_cam.T @ np.array([xn, yn, 1.0])
                if abs(ray_world[2]) < 1e-10:
                    world.append(None)
                    continue
                t = -cam_center[2] / ray_world[2]
                if t < 0:
                    world.append(None)
                    continue
                wp = cam_center + t * ray_world
                # Clamp to reasonable pitch range
                if abs(wp[0]) > max_range or abs(wp[1]) > max_range:
                    world.append(None)
                    continue
                world.append((wp[0], wp[1]))
            return world

        # 1. Draw camera FOV boundary (semi-transparent polygon)
        n_edge = 30
        boundary_img = []
        for i in range(n_edge):
            t = i / n_edge
            boundary_img.append((t * img_w, 0))
        for i in range(n_edge):
            t = i / n_edge
            boundary_img.append((img_w, t * img_h))
        for i in range(n_edge):
            t = i / n_edge
            boundary_img.append((img_w * (1 - t), img_h))
        for i in range(n_edge):
            t = i / n_edge
            boundary_img.append((0, img_h * (1 - t)))

        fov_world = _img_to_world(boundary_img)
        fov_px = []
        for wp in fov_world:
            if wp is not None:
                px_pt, py_pt = w2p(wp[0], wp[1])
                # Clamp to reasonable range to avoid huge coordinates
                px_pt = max(-width, min(2 * width, px_pt))
                py_pt = max(-height, min(2 * height, py_pt))
                fov_px.append((px_pt, py_pt))
        if len(fov_px) >= 3:
            overlay = pitch.copy()
            cv2.fillPoly(overlay, [np.array(fov_px, dtype=np.int32)], (60, 180, 60))
            cv2.addWeighted(overlay, 0.3, pitch, 0.7, 0, pitch)
            cv2.polylines(pitch, [np.array(fov_px, dtype=np.int32)], True, (100, 255, 100), 1)

        # Camera position marker — yellow triangle pointing toward pitch
        # centre + crosshair, identical idiom to the annotate page's
        # tactical view and the match-page minimap so all three top-down
        # renderings show the same cam marker.
        cam_world = (-R_cam.T @ tvec_cam).ravel()
        cpx, cpy = w2p(float(cam_world[0]), float(cam_world[1]))
        clipped_cam = not (0 <= cpx < width and 0 <= cpy < height)
        cpx = max(8, min(width - 8, cpx))
        cpy = max(8, min(height - 8, cpy))
        center_px = w2p(0.0, 0.0)
        vx_c, vy_c = center_px[0] - cpx, center_px[1] - cpy
        vlen_c = max((vx_c * vx_c + vy_c * vy_c) ** 0.5, 1e-6)
        ux_c, uy_c = vx_c / vlen_c, vy_c / vlen_c
        perp_x, perp_y = -uy_c, ux_c
        tri = np.array([
            [int(cpx + ux_c * 14), int(cpy + uy_c * 14)],
            [int(cpx - ux_c * 4 + perp_x * 8),
             int(cpy - uy_c * 4 + perp_y * 8)],
            [int(cpx - ux_c * 4 - perp_x * 8),
             int(cpy - uy_c * 4 - perp_y * 8)],
        ], dtype=np.int32)
        cv2.fillPoly(pitch, [tri], (102, 215, 255))   # yellow in BGR
        cv2.polylines(pitch, [tri], True, (0, 0, 0), 1)
        cv2.circle(pitch, (int(cpx), int(cpy)), 3, (0, 0, 0), -1)
        cv2.circle(pitch, (int(cpx), int(cpy)), 2, (102, 215, 255), -1)
        cam_label = (
            f"cam ({float(cam_world[0]):.1f}, {float(cam_world[1]):.1f}, "
            f"{float(cam_world[2]):.1f})"
        )
        if clipped_cam:
            cam_label += " v"
        (tw, th), _ = cv2.getTextSize(
            cam_label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1,
        )
        lx = width - tw - 12
        ly = 18
        cv2.rectangle(pitch, (lx - 4, ly - th - 3), (lx + tw + 4, ly + 4),
                      (0, 0, 0), -1)
        cv2.putText(pitch, cam_label, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (102, 215, 255), 1, cv2.LINE_AA)

        # --- Keypoints on top-down view ---
        # Back-project detected IMAGE pixel to world. Displacement from white
        # template genuinely reveals calibration error.
        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS

        # 3a. Draw detected keypoints (green/red) at back-projected detected pixel positions
        detected_topdown = {}  # kp_id -> (px, py) in top-down pixels
        for kp in keypoints:
            if kp.get("confidence", 0) < 0.3:
                continue
            kp_id = kp["id"]
            if kp_id in non_ground:
                continue
            world_xy = keypoint_mapper.get_world_coordinates(kp_id)
            if world_xy is None:
                continue
            wx, wy = world_xy

            # Back-project the detected image pixel through camera model
            wp = _img_to_world([(kp["x"], kp["y"])], max_range=200.0)
            if wp[0] is not None:
                px, py = w2p(wp[0][0], wp[0][1])
            else:
                px, py = w2p(wx, wy)  # fallback to true position
            detected_topdown[kp_id] = (px, py)
            if not (0 <= px < width and 0 <= py < height):
                continue

            kp_key = (round(float(kp["x"]), 1), round(float(kp["y"]), 1))
            if inlier_img_set and kp_key in inlier_img_set:
                cv2.circle(pitch, (px, py), r, (0, 255, 0), -1)
            else:
                cv2.circle(pitch, (px, py), r, (0, 0, 255), 2)
            cv2.circle(pitch, (px, py), r, (0, 0, 0), 1)
            cv2.putText(pitch, str(kp_id), (px + r + 2, py + 3), font, 0.35, (255, 255, 255), 1)

        # 4. Draw yellow lines connecting back-projected detected keypoints
        for kp_ids in PITCH_LINE_KEYPOINTS:
            for i in range(len(kp_ids) - 1):
                if kp_ids[i] in detected_topdown and kp_ids[i + 1] in detected_topdown:
                    pt1 = detected_topdown[kp_ids[i]]
                    pt2 = detected_topdown[kp_ids[i + 1]]
                    cv2.line(pitch, pt1, pt2, (0, 255, 255), max(1, lw))

    else:
        # No camera params — use homography (if available) or true world positions
        H = result.get("homography") if result is not None else None
        H_inv = None
        if H is not None:
            try:
                H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                H_inv = None

        def _img_to_world_h(img_x, img_y):
            """Back-project image point to world via H_inv."""
            if H_inv is None:
                return None
            pt_h = H_inv @ np.array([img_x, img_y, 1.0])
            if abs(pt_h[2]) < 1e-6:
                return None
            wx, wy = pt_h[0] / pt_h[2], pt_h[1] / pt_h[2]
            if abs(wx) > 200 or abs(wy) > 200:
                return None
            return (wx, wy)

        non_ground = KeypointMapper.NON_GROUND_KEYPOINTS
        detected_topdown = {}
        for kp in keypoints:
            if kp.get("confidence", 0) < 0.3:
                continue
            kp_id = kp["id"]
            if kp_id in non_ground:
                continue
            world_xy = keypoint_mapper.get_world_coordinates(kp_id)
            if world_xy is None:
                continue
            wx, wy = world_xy

            # Back-project via homography if available
            bp = _img_to_world_h(kp["x"], kp["y"])
            if bp is not None:
                px, py = w2p(bp[0], bp[1])
            else:
                px, py = w2p(wx, wy)
            detected_topdown[kp_id] = (px, py)
            if not (0 <= px < width and 0 <= py < height):
                continue

            kp_key = (round(float(kp["x"]), 1), round(float(kp["y"]), 1))
            if inlier_img_set and kp_key in inlier_img_set:
                cv2.circle(pitch, (px, py), r, (0, 255, 0), -1)
            else:
                cv2.circle(pitch, (px, py), r, (0, 0, 255), 2)
            cv2.circle(pitch, (px, py), r, (0, 0, 0), 1)
            cv2.putText(pitch, str(kp_id), (px + r + 2, py + 3), font, 0.35, (255, 255, 255), 1)

        # Draw yellow lines connecting back-projected keypoints
        if H_inv is not None:
            for kp_ids_line in PITCH_LINE_KEYPOINTS:
                for i in range(len(kp_ids_line) - 1):
                    if kp_ids_line[i] in detected_topdown and kp_ids_line[i + 1] in detected_topdown:
                        pt1 = detected_topdown[kp_ids_line[i]]
                        pt2 = detected_topdown[kp_ids_line[i + 1]]
                        cv2.line(pitch, pt1, pt2, (0, 255, 255), max(1, lw))

    return pitch
