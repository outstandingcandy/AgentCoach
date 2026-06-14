"""Shared visualization functions for Stage 1 backends."""

import cv2
import numpy as np

from ..utils.pitch import (
    get_pitch_template_points,
    project_pitch_to_image,
    _draw_topdown_pitch,
)


def _draw_label(img, x, y, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.4, thickness=1):
    """Draw a text label with black background at (x, y)."""
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
    lx, ly = x + 8, y + 4
    cv2.rectangle(img, (lx - 1, ly - th - 1), (lx + tw + 1, ly + 2), (0, 0, 0), -1)
    cv2.putText(img, label, (lx, ly), font, font_scale, (255, 255, 255), thickness)


def draw_vis_keypoints(frame, keypoints, conf_threshold=0.3):
    """Draw keypoint detection results on the frame.

    Args:
        frame: BGR frame.
        keypoints: List of dicts from ``KeypointDetector.detect``.
        conf_threshold: Minimum confidence to render. Pass the same value the
            solver was configured with so the overlay reflects what the
            calibrator actually consumed. Default 0.3 matches the typical
            keypoint head threshold.
    """
    vis = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    n_detected = 0

    for kp in keypoints:
        conf = kp.get("confidence", 0)
        if conf < conf_threshold:
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

    header = f"Keypoints: {n_detected} (conf>={conf_threshold:.2f})"
    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (0, 255, 0), 1)
    return vis


def draw_vis_lines(frame, lines, conf_threshold=0.15):
    """Draw line detection results on the frame.

    Args:
        frame: BGR frame.
        lines: List of dicts from ``LineDetector.detect``.
        conf_threshold: Minimum confidence to render. Pass the same value the
            solver was configured with so the overlay reflects what the
            calibrator actually consumed.
    """
    from ..field_registration.pnlcalib import LineMapper

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
        if conf < conf_threshold:
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

    header = f"Lines: {n_detected} (conf>={conf_threshold:.2f})"
    cv2.putText(vis, header, (10, 25), font, 0.6, (255, 255, 255), 2)
    cv2.putText(vis, header, (10, 25), font, 0.6, (0, 255, 255), 1)
    return vis


def draw_vis_intersections(frame, keypoints, lines, calibrator):
    """Draw calibrator update results: keypoints, lines, and computed intersections."""
    from ..field_registration.pnlcalib import LineMapper

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
    from ..field_registration.pnlcalib import KeypointMapper, LineMapper

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
                margin = 200
                pt1_in = -margin < pt1[0] < w + margin and -margin < pt1[1] < h + margin
                pt2_in = -margin < pt2[0] < w + margin and -margin < pt2[1] < h + margin
                if pt1_in and pt2_in:
                    # Bright blue (BGR) — distinguishable from red player
                    # kits / scoreboard graphics that the orange wireframe
                    # used to blend into in JPEG-compressed previews.
                    cv2.line(vis, pt1, pt2, (255, 0, 0), 2)

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
