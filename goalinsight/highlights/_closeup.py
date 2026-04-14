"""Close-up crop utilities for highlight clipping."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CropInfo:
    """Metadata about a single frame's crop region."""

    center: tuple[float, float]   # (cx, cy) in source frame pixels
    crop_box: list[int]           # [x1, y1, x2, y2] actual crop region
    scale: float                  # source_crop_width / output_width


@dataclass
class SmoothState:
    """Carries smoothed center and crop size between frames."""

    cx: float
    cy: float
    crop_w: float
    crop_h: float


def extract_closeup(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, ...],
    output_size: tuple[int, int] = (640, 360),
    padding_factor: float = 2.0,
    prev_state: SmoothState | None = None,
    smooth_alpha: float = 0.3,
) -> tuple[np.ndarray, SmoothState, CropInfo]:
    """Crop and resize a close-up around a bounding box.

    Both center position and crop size are EMA-smoothed to avoid jumps.

    Args:
        frame: Full video frame (H, W, 3) BGR.
        bbox: Bounding box [x1, y1, x2, y2] in pixels.
        output_size: (width, height) of the output crop.
        padding_factor: How much to expand the bbox.
        prev_state: Previous frame's smoothed state (center + crop size).
        smooth_alpha: EMA alpha (lower = smoother).

    Returns:
        (cropped_frame, current_state, crop_info)
    """
    import cv2

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    # Compute target crop size from bbox
    out_w, out_h = output_size
    aspect = out_w / out_h

    bbox_w = (x2 - x1) * padding_factor
    bbox_h = (y2 - y1) * padding_factor

    # Ensure crop matches output aspect ratio
    if bbox_w / max(bbox_h, 1e-6) > aspect:
        crop_w = bbox_w
        crop_h = crop_w / aspect
    else:
        crop_h = bbox_h
        crop_w = crop_h * aspect

    # Minimum crop size to avoid excessive zoom
    min_crop_w = out_w * 0.5
    min_crop_h = out_h * 0.5
    crop_w = max(crop_w, min_crop_w)
    crop_h = max(crop_h, min_crop_h)

    # EMA smooth both center and crop size
    if prev_state is not None:
        cx = smooth_alpha * cx + (1 - smooth_alpha) * prev_state.cx
        cy = smooth_alpha * cy + (1 - smooth_alpha) * prev_state.cy
        crop_w = smooth_alpha * crop_w + (1 - smooth_alpha) * prev_state.crop_w
        crop_h = smooth_alpha * crop_h + (1 - smooth_alpha) * prev_state.crop_h

    state = SmoothState(cx=cx, cy=cy, crop_w=crop_w, crop_h=crop_h)

    # Clamp to frame bounds
    cx1 = int(max(0, cx - crop_w / 2))
    cy1 = int(max(0, cy - crop_h / 2))
    cx2 = int(min(w, cx + crop_w / 2))
    cy2 = int(min(h, cy + crop_h / 2))

    # Re-adjust if clamping shifted the size
    if cx2 - cx1 < crop_w and cx1 == 0:
        cx2 = int(min(w, crop_w))
    if cy2 - cy1 < crop_h and cy1 == 0:
        cy2 = int(min(h, crop_h))
    if cx2 - cx1 < crop_w and cx2 == w:
        cx1 = int(max(0, w - crop_w))
    if cy2 - cy1 < crop_h and cy2 == h:
        cy1 = int(max(0, h - crop_h))

    crop = frame[cy1:cy2, cx1:cx2]
    actual_crop_w = cx2 - cx1
    scale = actual_crop_w / out_w if out_w > 0 else 1.0
    info = CropInfo(center=(cx, cy), crop_box=[cx1, cy1, cx2, cy2], scale=scale)

    if crop.size == 0:
        info = CropInfo(center=(cx, cy), crop_box=[0, 0, w, h], scale=w / out_w)
        return cv2.resize(frame, output_size, interpolation=cv2.INTER_LINEAR), state, info

    resized = cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)
    return resized, state, info


def interpolate_bbox(
    trajectory: list[dict],
    frame: int,
    max_extrapolate: int = 15,
) -> list[float] | None:
    """Linearly interpolate a bbox for *frame* from surrounding observations.

    Each entry in *trajectory* must have 'frame' and 'bbox' keys.
    Returns None if the frame is beyond *max_extrapolate* frames from
    any observation (track lost — caller should fall back to wide view).
    """
    if not trajectory:
        return None

    # Exact match
    for t in trajectory:
        if t["frame"] == frame:
            return t["bbox"]

    # Find surrounding observations
    before = [t for t in trajectory if t["frame"] < frame]
    after = [t for t in trajectory if t["frame"] > frame]

    if before and after:
        b = max(before, key=lambda t: t["frame"])
        a = min(after, key=lambda t: t["frame"])
        gap = a["frame"] - b["frame"]
        if gap > 0:
            alpha = (frame - b["frame"]) / gap
            return [
                b["bbox"][i] + alpha * (a["bbox"][i] - b["bbox"][i])
                for i in range(4)
            ]

    # Extrapolate from nearest observation, but only within limit
    nearest = min(trajectory, key=lambda t: abs(t["frame"] - frame))
    if abs(nearest["frame"] - frame) <= max_extrapolate:
        return nearest["bbox"]
    return None
