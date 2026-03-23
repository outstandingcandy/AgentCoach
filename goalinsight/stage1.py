#!/usr/bin/env python3
"""Stage 1: Field Registration - Camera calibration via keypoint and line detection.

This stage:
1. Detects field keypoints and lines using HRNet
2. Computes camera parameters (homography) via PnL optimization
3. Saves calibration results for subsequent stages

Supports configurable backends:
- PnLCalib (default): Uses HRNet keypoint/line detection with PnL optimization
- NBJW: Uses NBJW's FramebyFrameCalib approach
"""

import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .utils.config import get_default_config, get_process_fps_from_config, FrameSampler


# Standard pitch dimensions (in meters, centered at origin)
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0

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


def get_pitch_template_points():
    """Get key points on the pitch template for visualization."""
    half_l = PITCH_LENGTH / 2
    half_w = PITCH_WIDTH / 2

    return {
        'pitch_outline': [
            (-half_l, -half_w), (half_l, -half_w),
            (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)
        ],
        'center_line': [(0, -half_w), (0, half_w)],
        'center_circle': [
            (9.15 * np.cos(a), 9.15 * np.sin(a))
            for a in np.linspace(0, 2*np.pi, 36)
        ],
        'left_penalty': [(-half_l, -20.16), (-36, -20.16), (-36, 20.16), (-half_l, 20.16)],
        'right_penalty': [(half_l, -20.16), (36, -20.16), (36, 20.16), (half_l, 20.16)],
        'left_goal_area': [(-half_l, -9.16), (-47, -9.16), (-47, 9.16), (-half_l, 9.16)],
        'right_goal_area': [(half_l, -9.16), (47, -9.16), (47, 9.16), (half_l, 9.16)],
        'left_penalty_arc': [
            (-half_l + 11 + 9.15 * np.cos(a), 9.15 * np.sin(a))
            for a in np.linspace(-0.93, 0.93, 20)  # ~53 degrees, outside penalty area
        ],
        'right_penalty_arc': [
            (half_l - 11 - 9.15 * np.cos(a), 9.15 * np.sin(a))
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
    
    R, _ = cv2.Rodrigues(rvec)

    projected = {}
    for name, points in template_points.items():
        # Build 3D points (z=0 ground plane)
        obj = np.array([[pt[0], pt[1], 0.0] for pt in points], dtype=np.float64)

        # Per-point camera-space Z check
        cam_pts = (R @ obj.T).T + tvec.flatten()
        img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        proj_points = []
        for j, pt2d in enumerate(img.reshape(-1, 2)):
            if cam_pts[j, 2] <= 0.1:
                proj_points.append(None)  # Behind camera
            else:
                proj_points.append((int(pt2d[0]), int(pt2d[1])))
        projected[name] = proj_points
    return projected


def _draw_topdown_pitch(height, width, result, keypoints, calibrator, keypoint_mapper):
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

    Returns:
        Top-down pitch image (BGR).
    """
    from .field_registration.pnlcalib import KeypointMapper

    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)  # Forest green background

    half_l = PITCH_LENGTH / 2  # 52.5
    half_w = PITCH_WIDTH / 2   # 34.0

    # Pitch coordinate -> pixel mapping (with margin)
    margin = 8  # meters of margin around pitch
    scale = min(
        (width - 2) / (PITCH_LENGTH + 2 * margin),
        (height - 2) / (PITCH_WIDTH + 2 * margin),
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

    # Pitch outline
    cv2.rectangle(pitch, w2p(-half_l, half_w), w2p(half_l, -half_w), line_color, lw)
    # Center line
    cv2.line(pitch, w2p(0, half_w), w2p(0, -half_w), line_color, lw)
    # Center circle (r=9.15m)
    center_px = w2p(0, 0)
    cv2.circle(pitch, center_px, int(9.15 * scale), line_color, lw)
    # Center spot
    cv2.circle(pitch, center_px, max(2, int(0.3 * scale)), line_color, -1)
    # Left penalty area
    cv2.rectangle(pitch, w2p(-half_l, 20.16), w2p(-half_l + 16.5, -20.16), line_color, lw)
    # Right penalty area
    cv2.rectangle(pitch, w2p(half_l - 16.5, 20.16), w2p(half_l, -20.16), line_color, lw)
    # Left goal area
    cv2.rectangle(pitch, w2p(-half_l, 9.16), w2p(-half_l + 5.5, -9.16), line_color, lw)
    # Right goal area
    cv2.rectangle(pitch, w2p(half_l - 5.5, 9.16), w2p(half_l, -9.16), line_color, lw)
    # Penalty spots
    cv2.circle(pitch, w2p(-half_l + 11, 0), max(2, int(0.3 * scale)), line_color, -1)
    cv2.circle(pitch, w2p(half_l - 11, 0), max(2, int(0.3 * scale)), line_color, -1)

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
        detected_topdown = {}  # kp_id → (px, py) in top-down pixels
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
        # No camera params — draw detected keypoints at true world positions
        for kp in keypoints:
            if kp.get("confidence", 0) < 0.3:
                continue
            kp_id = kp["id"]
            if kp_id in KeypointMapper.NON_GROUND_KEYPOINTS:
                continue
            world_xy = keypoint_mapper.get_world_coordinates(kp_id)
            if world_xy is None:
                continue
            wx, wy = world_xy
            px, py = w2p(wx, wy)
            if not (0 <= px < width and 0 <= py < height):
                continue
            kp_key = (round(float(kp["x"]), 1), round(float(kp["y"]), 1))
            if inlier_img_set and kp_key in inlier_img_set:
                cv2.circle(pitch, (px, py), r, (0, 255, 0), -1)
            else:
                cv2.circle(pitch, (px, py), r, (0, 0, 255), 2)
            cv2.circle(pitch, (px, py), r, (0, 0, 0), 1)
            cv2.putText(pitch, str(kp_id), (px + r + 2, py + 3), font, 0.35, (255, 255, 255), 1)

    return pitch


def _draw_label(img, x, y, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4, thickness=1):
    """Draw a text label with black background at (x, y)."""
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    lx, ly = x + 8, y + 4
    cv2.rectangle(img, (lx - 1, ly - th - 1), (lx + tw + 1, ly + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (lx, ly), font, font_scale, (255, 255, 255), thickness)


def draw_vis_keypoints(frame, keypoints):
    """Draw keypoint detection results on the frame."""
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_detected = 0

    for kp in keypoints:
        conf = kp.get("confidence", 0)
        if conf < 0.3:
            continue
        n_detected += 1
        x, y = int(kp["x"]), int(kp["y"])
        # Color by confidence: green (high) -> yellow (low)
        green = int(min(255, conf * 2 * 255))
        blue = 0
        red = int(max(0, (1 - conf) * 2 * 255))
        cv2.circle(vis, (x, y), 6, (blue, green, red), -1)
        cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
        _draw_label(vis, x, y, f"{kp['id']}({conf:.2f})")

    cv2.putText(vis, f"Keypoints: {n_detected} (conf>=0.3)", (10, 25),
                font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Keypoints: {n_detected} (conf>=0.3)", (10, 25),
                font, 0.6, (0, 255, 0), 1)
    return vis


def draw_vis_lines(frame, lines):
    """Draw line detection results on the frame."""
    from .field_registration.pnlcalib import LineMapper

    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_detected = 0

    # Color palette for different line IDs
    colors = [
        (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 165, 255),
        (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    ]

    for line in lines:
        conf = line.get("confidence", 0)
        if conf < 0.15:
            continue
        n_detected += 1
        line_id = line.get("id", -1)
        x1, y1, x2, y2 = int(line["x1"]), int(line["y1"]), int(line["x2"]), int(line["y2"])
        color = colors[line_id % len(colors)]
        lw = 2 if line_id not in LineMapper.NON_GROUND_LINES else 1
        cv2.line(vis, (x1, y1), (x2, y2), color, lw)
        # Label at midpoint
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        _draw_label(vis, mx, my, f"L{line_id}({conf:.2f})")

    cv2.putText(vis, f"Lines: {n_detected} (conf>=0.15)", (10, 25),
                font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, f"Lines: {n_detected} (conf>=0.15)", (10, 25),
                font, 0.6, (0, 255, 255), 1)
    return vis


def draw_vis_intersections(frame, keypoints, lines, calibrator):
    """Draw calibrator update results: keypoints, lines, and computed intersections."""
    from .field_registration.pnlcalib import LineMapper

    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Draw lines (dimmed)
    for line in lines:
        if line.get("confidence", 0) >= 0.15:
            x1, y1, x2, y2 = int(line["x1"]), int(line["y1"]), int(line["x2"]), int(line["y2"])
            is_ground = line.get("id", -1) not in LineMapper.NON_GROUND_LINES
            color = (128, 128, 0) if is_ground else (64, 64, 0)
            cv2.line(vis, (x1, y1), (x2, y2), color, 1)

    # Draw keypoints (dimmed)
    for kp in keypoints:
        if kp.get("confidence", 0) >= 0.3:
            x, y = int(kp["x"]), int(kp["y"])
            cv2.circle(vis, (x, y), 4, (0, 180, 0), -1)

    # Draw intersections (highlighted)
    n_inter = len(calibrator.line_intersections)
    for inter in calibrator.line_intersections:
        ix, iy = int(inter["x"]), int(inter["y"])
        cv2.circle(vis, (ix, iy), 8, (255, 0, 255), 2)
        cv2.circle(vis, (ix, iy), 3, (255, 0, 255), -1)
        src = inter.get("source", "")
        label = f"({inter.get('world_x', 0):.0f},{inter.get('world_y', 0):.0f})"
        if src:
            label = f"{src} {label}"
        cv2.putText(vis, label, (ix + 10, iy - 5), font, 0.3, (255, 0, 255), 1)

    n_kp = len([k for k in keypoints if k.get("confidence", 0) >= 0.3])
    header = f"Intersections: {n_inter} | Keypoints: {n_kp}"
    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 0, 255), 1)
    return vis


def draw_vis_calibration(frame, keypoints, lines, calibrator, result, pitch_template, keypoint_mapper):
    """Draw PnL calibration results: projected pitch overlay + top-down pitch diagram."""
    from .field_registration.pnlcalib import KeypointMapper, LineMapper

    vis = frame.copy()
    h, w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # --- Left panel: camera view with overlay ---

    # Draw projected pitch if calibration succeeded
    if result is not None:
        H = result["homography"]
        cam_params = result.get("camera_params")
        projected = project_pitch_to_image(H, pitch_template, camera_params=cam_params)
        for name, points in projected.items():
            valid_points = [p for p in points if p is not None]
            for i in range(len(valid_points) - 1):
                pt1, pt2 = valid_points[i], valid_points[i + 1]
                if all(-1000 < c < 3000 for c in pt1 + pt2):
                    cv2.line(vis, pt1, pt2, (0, 165, 255), 2)

    # Build inlier set for keypoint coloring
    inlier_img_set = set()
    if result is not None and "img_pts" in result and "inlier_mask" in result:
        for pt, is_inlier in zip(result["img_pts"], result["inlier_mask"]):
            if is_inlier:
                inlier_img_set.add((round(float(pt[0]), 1), round(float(pt[1]), 1)))

    # Draw keypoints with ID labels, distinguishing inlier/outlier
    for kp in keypoints:
        if kp.get("confidence", 0) >= 0.3:
            x, y = int(kp["x"]), int(kp["y"])
            kp_id = kp["id"]

            if kp_id in KeypointMapper.NON_GROUND_KEYPOINTS:
                cv2.circle(vis, (x, y), 6, (0, 0, 255), -1)
            elif result is not None and inlier_img_set:
                kp_key = (round(float(kp["x"]), 1), round(float(kp["y"]), 1))
                if kp_key in inlier_img_set:
                    cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
                else:
                    cv2.circle(vis, (x, y), 6, (0, 0, 255), 2)
            else:
                cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(vis, (x, y), 6, (0, 0, 0), 1)
            _draw_label(vis, x, y, str(kp_id))

    # Draw lines
    for line in lines:
        if line.get("confidence", 0) >= 0.5:
            if line.get("id", -1) not in LineMapper.NON_GROUND_LINES:
                x1, y1, x2, y2 = int(line["x1"]), int(line["y1"]), int(line["x2"]), int(line["y2"])
                cv2.line(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

    # Draw intersections with labels
    for inter in calibrator.line_intersections:
        ix, iy = int(inter["x"]), int(inter["y"])
        cv2.circle(vis, (ix, iy), 7, (255, 0, 255), 2)
        src = inter.get("source", "")
        if src:
            cv2.putText(vis, src, (ix + 9, iy - 3), font, 0.3, (255, 0, 255), 1)

    # Draw failure diagnostics
    if result is None:
        n_kp = len([k for k in keypoints if k.get("confidence", 0) >= 0.3])
        n_lines = len([l for l in lines if l.get("confidence", 0) >= 0.5])
        n_inter = len(calibrator.line_intersections)
        diag_lines = [
            "CALIBRATION FAILED",
            f"Keypoints (conf>=0.3): {n_kp}",
            f"Lines (conf>=0.5): {n_lines}",
            f"Intersections: {n_inter}",
        ]
        for i, text in enumerate(diag_lines):
            cv2.putText(vis, text, (10, 30 + i * 22), font, 0.6, (0, 0, 255), 2)

    # --- Right panel: top-down pitch ---
    pitch = _draw_topdown_pitch(h, w, result, keypoints, calibrator, keypoint_mapper)

    # --- Header text ---
    if result is not None:
        err = result.get("final_error", 0)
        n_in = result.get("inliers", 0)
        n_total = result.get("total_points", 0)
        header = f"Err: {err:.1f}px | Inliers: {n_in}/{n_total}"
        cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, header, (10, 25), font, 0.6, (0, 200, 0), 1)

    # Combine side-by-side
    combined = np.hstack([vis, pitch])
    return combined


def draw_visualization(frame, keypoints, lines, calibrator, result, pitch_template, keypoint_mapper):
    """Draw side-by-side visualization: camera view (left) + top-down pitch (right).

    Kept for backward compatibility. Delegates to draw_vis_calibration.
    """
    return draw_vis_calibration(frame, keypoints, lines, calibrator, result, pitch_template, keypoint_mapper)


def run_stage1(video_path: Path, output_dir: Path, config: dict | None = None):
    """Run Stage 1 field registration.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        config: Optional configuration dict

    Returns:
        Dict with calibration statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Load configuration
    if config is None:
        config = get_default_config()
    process_fps = get_process_fps_from_config(config)

    # Determine backend from config
    fr_config = config.get("field_registration", {})
    backend = fr_config.get("backend", "pnlcalib")
    print(f"Stage 1: Using {backend} backend for field registration")

    # Initialize based on backend
    if backend == "nbjw":
        return _run_stage1_nbjw(video_path, output_dir, vis_dir, config, process_fps)
    elif backend == "broadtrack":
        from .stage1_broadtrack import run_stage1_broadtrack
        return run_stage1_broadtrack(video_path, output_dir, vis_dir, config, process_fps)
    elif backend == "physical":
        from .stage1_physical import run_stage1_physical
        return run_stage1_physical(video_path, output_dir, vis_dir, config, process_fps)
    else:
        return _run_stage1_pnlcalib(video_path, output_dir, vis_dir, config, process_fps)


def _run_stage1_pnlcalib(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using PnLCalib backend."""
    from .field_registration import KeypointDetector, LineDetector
    from .field_registration.pnlcalib import (
        FramebyFrameCalib,
        KeypointMapper,
        LineMapper,
    )

    fr_config = config.get("field_registration", {})
    pnl_config = fr_config.get("pnlcalib", {})

    # Initialize detectors
    print("Stage 1: Initializing keypoint detector (HRNet/PnLCalib)...")
    kp_config = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": pnl_config.get("keypoint_weights", "SV_kp"),
            "model_path": pnl_config.get("keypoint_model_path"),  # Custom fine-tuned model
            "confidence_threshold": pnl_config.get("keypoint_threshold", 0.3434),
        }
    }
    kp_detector = KeypointDetector(kp_config)
    kp_detector.load_model()
    keypoint_mapper = KeypointMapper()

    # Line detection is disabled by default (use_lines=false in config).
    # When disabled, calibration relies purely on keypoints.
    use_lines = pnl_config.get("use_lines", False)
    line_detector = None
    line_mapper = LineMapper()
    if use_lines:
        print("Stage 1: Initializing line detector (HRNet/PnLCalib)...")
        line_config = {
            "backend": "pnlcalib",
            "pnlcalib": {
                "weights": pnl_config.get("line_weights", "SV_lines"),
                "confidence_threshold": pnl_config.get("line_threshold", 0.15),
            }
        }
        line_detector = LineDetector(line_config)
        line_detector.load_model()
    else:
        print("Stage 1: Line detection disabled (keypoint-only mode)")
    ransac_thresh = pnl_config.get("ransac_threshold", 30.0)
    calib_method = pnl_config.get("calibration_method", "iterative_pnp")

    pitch_template = get_pitch_template_points()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampler = FrameSampler(total_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

    # Results storage
    calibration_results = {
        "video_info": {
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "process_fps": process_fps,
        },
        "frames": {}
    }
    homographies = {}

    # Create calibrator once (image_size and alpha don't change per frame)
    calibrator = FramebyFrameCalib(image_size=(width, height), alpha=0.7)

    # Create visualization subdirectories
    vis_kp_dir = vis_dir / "keypoints"
    vis_line_dir = vis_dir / "lines" if use_lines else None
    vis_inter_dir = vis_dir / "intersections" if use_lines else None
    vis_calib_dir = vis_dir / "calibration"
    for d in [vis_kp_dir, vis_line_dir, vis_inter_dir, vis_calib_dir]:
        if d is not None:
            d.mkdir(exist_ok=True)

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)  # ~100 visualizations

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1: Calibrating")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints (and lines if enabled)
        keypoints = kp_detector.detect(frame, convert_to_soccernet=False)
        lines = line_detector.detect(frame) if line_detector is not None else []
        calibrator.update(keypoints, lines)

        # Run calibration
        result = calibrator.calibrate(
            keypoint_mapper, line_mapper,
            min_confidence=0.3, use_line_refinement=use_lines,
            ransac_threshold=ransac_thresh,
            method=calib_method,
        )

        if result is not None:
            homographies[frame_idx] = result["homography"]
            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": result["num_keypoints"],
                "num_lines": result["num_lines"],
                "num_intersections": result["num_intersections"],
                "total_points": result["total_points"],
                "inliers": result["inliers"],
                "reprojection_error": result["final_error"],
            }
            calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}

        # Save visualizations periodically
        if idx % vis_interval == 0:
            fname = f"frame_{frame_idx:05d}.jpg"
            cv2.imwrite(str(vis_kp_dir / fname), draw_vis_keypoints(frame, keypoints))
            if vis_line_dir is not None:
                cv2.imwrite(str(vis_line_dir / fname), draw_vis_lines(frame, lines))
            if vis_inter_dir is not None:
                cv2.imwrite(str(vis_inter_dir / fname), draw_vis_intersections(frame, keypoints, lines, calibrator))
            cv2.imwrite(str(vis_calib_dir / fname), draw_vis_calibration(frame, keypoints, lines, calibrator, result, pitch_template, keypoint_mapper))

    cap.release()

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)

    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": len(sampler),
        "calibrated_frames": calibrated_count,
        "calibration_rate": calibrated_count / len(sampler) if sampler else 0,
    }

    errors = [
        calibration_results["frames"][idx]["reprojection_error"]
        for idx in calibration_results["frames"]
        if calibration_results["frames"][idx].get("calibrated")
    ]
    if errors:
        stats["mean_error"] = float(np.mean(errors))
        stats["median_error"] = float(np.median(errors))

    print(f"\nStage 1 Complete:")
    print(f"  Calibrated: {calibrated_count}/{len(sampler)} ({stats['calibration_rate']*100:.1f}%)")
    if errors:
        print(f"  Median error: {stats['median_error']:.2f} px")

    return stats


def _run_stage1_nbjw(
    video_path: Path,
    output_dir: Path,
    vis_dir: Path,
    config: dict,
    process_fps: float | None,
):
    """Run Stage 1 using NBJW backend."""
    from .field_registration.nbjw import NbjwCalibrator

    fr_config = config.get("field_registration", {})
    nbjw_config = fr_config.get("nbjw", {})
    nbjw_config["device"] = config.get("device", "cuda")

    # Initialize NBJW calibrator
    print("Stage 1: Initializing NBJW calibrator...")
    calibrator = NbjwCalibrator(nbjw_config)
    calibrator.load_models()

    pitch_template = get_pitch_template_points()

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    sampler = FrameSampler(total_frames, fps, process_fps)
    print(f"Video: {total_frames} frames @ {fps:.1f} fps, {width}x{height}")
    print(f"Processing {len(sampler)} frames at {process_fps or fps} fps")

    # Results storage
    calibration_results = {
        "video_info": {
            "path": str(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "process_fps": process_fps,
            "backend": "nbjw",
        },
        "frames": {}
    }
    homographies = {}

    # Process frames
    calibrated_count = 0
    vis_interval = max(1, len(sampler) // 100)  # ~100 visualizations

    for idx, frame_idx in enumerate(tqdm(sampler, desc="Stage 1: Calibrating (NBJW)")):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Detect keypoints
        detection_result = calibrator.detect_keypoints(frame)
        keypoints = detection_result.get("keypoints", {})

        # Compute homography
        homography = calibrator.compute_homography(keypoints, width, height)

        if homography is not None:
            homographies[frame_idx] = homography
            calibration_results["frames"][frame_idx] = {
                "calibrated": True,
                "num_keypoints": len(keypoints),
                "num_lines": len(detection_result.get("lines", {})),
            }
            calibrated_count += 1
        else:
            calibration_results["frames"][frame_idx] = {"calibrated": False}

        # Save visualization periodically
        if idx % vis_interval == 0:
            vis = _draw_visualization_nbjw(
                frame, keypoints, homography, pitch_template, width, height
            )
            cv2.imwrite(str(vis_dir / f"frame_{frame_idx:05d}.jpg"), vis)

    cap.release()

    # Save results
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)

    # Statistics
    stats = {
        "total_frames": total_frames,
        "processed_frames": len(sampler),
        "calibrated_frames": calibrated_count,
        "calibration_rate": calibrated_count / len(sampler) if sampler else 0,
        "backend": "nbjw",
    }

    print(f"\nStage 1 Complete (NBJW):")
    print(f"  Calibrated: {calibrated_count}/{len(sampler)} ({stats['calibration_rate']*100:.1f}%)")

    return stats


def _draw_visualization_nbjw(
    frame: np.ndarray,
    keypoints: dict,
    homography: np.ndarray | None,
    pitch_template: dict,
    width: int,
    height: int,
) -> np.ndarray:
    """Draw visualization for NBJW backend."""
    vis = frame.copy()

    # Draw projected pitch if calibration succeeded
    if homography is not None:
        # Need to invert homography for world -> image projection
        try:
            H_inv = np.linalg.inv(homography)
            H_inv = H_inv / H_inv[-1, -1]
            projected = project_pitch_to_image(H_inv, pitch_template)
            for name, points in projected.items():
                valid_points = [p for p in points if p is not None]
                for i in range(len(valid_points) - 1):
                    pt1, pt2 = valid_points[i], valid_points[i + 1]
                    if all(-1000 < c < 3000 for c in pt1 + pt2):
                        cv2.line(vis, pt1, pt2, (0, 165, 255), 2)
        except np.linalg.LinAlgError:
            pass

    # Draw keypoints (NBJW keypoints are normalized 0-1)
    for idx, kp in keypoints.items():
        if isinstance(idx, int) and idx <= 57:  # Only draw pitch keypoints
            x = int(kp.get("x", 0) * width)
            y = int(kp.get("y", 0) * height)
            cv2.circle(vis, (x, y), 6, (0, 255, 0), -1)
            cv2.putText(vis, str(idx), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    return vis


def main():
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage1_field_registration")
    run_stage1(video_path, output_dir)


if __name__ == "__main__":
    main()
