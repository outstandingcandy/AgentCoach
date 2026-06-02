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

    Args:
        pitch_length: Pitch length in meters (default: active pitch length).
        pitch_width: Pitch width in meters (default: active pitch width).
    """
    active = pitch_constants.get_active_pitch()
    pl = pitch_length or active.PITCH_LENGTH
    pw = pitch_width or active.PITCH_WIDTH
    half_l = pl / 2
    half_w = pw / 2

    # Marking dimensions track the active pitch — non-FIFA youth pitches
    # have their own penalty/goal-area sizes. Reading FIFA constants here
    # produced an overlay where the outer touchlines match the solver's
    # H but the inner boxes/circle don't, masquerading as "calibration
    # is wrong" when the H was actually correct.
    PA_DEPTH = active.PENALTY_AREA_LENGTH
    PA_HW = active.PENALTY_AREA_WIDTH / 2.0
    GA_DEPTH = active.GOAL_AREA_LENGTH
    GA_HW = active.GOAL_AREA_WIDTH / 2.0
    PS_DIST = active.GOAL_LINE_TO_PENALTY_MARK
    CR = active.CENTER_CIRCLE_RADIUS

    return {
        'pitch_outline': [
            (-half_l, -half_w), (half_l, -half_w),
            (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)
        ],
        'center_line': [(0, -half_w), (0, half_w)],
        'center_circle': [
            (CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(0, 2*np.pi, 36)
        ],
        'left_penalty': [(-half_l, -PA_HW), (-half_l + PA_DEPTH, -PA_HW),
                         (-half_l + PA_DEPTH, PA_HW), (-half_l, PA_HW)],
        'right_penalty': [(half_l, -PA_HW), (half_l - PA_DEPTH, -PA_HW),
                          (half_l - PA_DEPTH, PA_HW), (half_l, PA_HW)],
        'left_goal_area': [(-half_l, -GA_HW), (-half_l + GA_DEPTH, -GA_HW),
                           (-half_l + GA_DEPTH, GA_HW), (-half_l, GA_HW)],
        'right_goal_area': [(half_l, -GA_HW), (half_l - GA_DEPTH, -GA_HW),
                            (half_l - GA_DEPTH, GA_HW), (half_l, GA_HW)],
        'left_penalty_arc': [
            (-half_l + PS_DIST + CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(-0.93, 0.93, 20)
        ],
        'right_penalty_arc': [
            (half_l - PS_DIST - CR * np.cos(a), CR * np.sin(a))
            for a in np.linspace(-0.93, 0.93, 20)
        ],
    }


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
    """Project pitch template using full pinhole + distortion model."""
    rvec = camera_params["rvec"]
    tvec = camera_params["tvec"]
    K = camera_params["K"]
    dist = camera_params["dist_coeffs"]

    projected = {}
    for name, points in template_points.items():
        obj = np.array([[pt[0], pt[1], 0.0] for pt in points], dtype=np.float64)
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

    half_l = pl / 2
    half_w = pw / 2

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

    # Marking dimensions follow the active pitch (kids pitches have
    # smaller penalty/goal areas than FIFA), matching the same fix in
    # get_pitch_template_points above.
    pa_d = active.PENALTY_AREA_LENGTH
    pa_hw = active.PENALTY_AREA_WIDTH / 2.0
    ga_d = active.GOAL_AREA_LENGTH
    ga_hw = active.GOAL_AREA_WIDTH / 2.0
    ps_dist = active.GOAL_LINE_TO_PENALTY_MARK
    cr = active.CENTER_CIRCLE_RADIUS

    # Pitch outline
    cv2.rectangle(pitch, w2p(-half_l, half_w), w2p(half_l, -half_w), line_color, lw)
    # Center line
    cv2.line(pitch, w2p(0, half_w), w2p(0, -half_w), line_color, lw)
    # Center circle
    center_px = w2p(0, 0)
    cv2.circle(pitch, center_px, int(cr * scale), line_color, lw)
    # Center spot
    cv2.circle(pitch, center_px, max(2, int(0.3 * scale)), line_color, -1)
    # Left penalty area
    cv2.rectangle(pitch, w2p(-half_l, pa_hw), w2p(-half_l + pa_d, -pa_hw), line_color, lw)
    # Right penalty area
    cv2.rectangle(pitch, w2p(half_l - pa_d, pa_hw), w2p(half_l, -pa_hw), line_color, lw)
    # Left goal area
    cv2.rectangle(pitch, w2p(-half_l, ga_hw), w2p(-half_l + ga_d, -ga_hw), line_color, lw)
    # Right goal area
    cv2.rectangle(pitch, w2p(half_l - ga_d, ga_hw), w2p(half_l, -ga_hw), line_color, lw)
    # Penalty spots
    cv2.circle(pitch, w2p(-half_l + ps_dist, 0), max(2, int(0.3 * scale)), line_color, -1)
    cv2.circle(pitch, w2p(half_l - ps_dist, 0), max(2, int(0.3 * scale)), line_color, -1)

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
