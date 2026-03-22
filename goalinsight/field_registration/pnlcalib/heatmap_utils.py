"""Heatmap processing utilities for keypoint and line detection.

This module provides functions for extracting keypoint coordinates from
neural network heatmap outputs using various methods.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def soft_argmax_2d(
    heatmap: torch.Tensor,
    temperature: float = 1.0,
    normalized: bool = True,
) -> torch.Tensor:
    """Compute soft argmax to get sub-pixel coordinates from heatmaps.

    Uses the soft-argmax approach to extract continuous coordinates from
    heatmaps, which is differentiable and provides sub-pixel accuracy.

    Args:
        heatmap: Heatmap tensor of shape (B, C, H, W) or (C, H, W).
        temperature: Softmax temperature (lower = sharper).
        normalized: If True, return coordinates in [0, 1] range.

    Returns:
        Coordinates tensor of shape (B, C, 2) or (C, 2) for (x, y).
    """
    squeeze = False
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)
        squeeze = True

    b, c, h, w = heatmap.shape
    device = heatmap.device

    # Apply softmax over spatial dimensions
    heatmap_flat = heatmap.view(b, c, -1)
    softmax = F.softmax(heatmap_flat / temperature, dim=-1)
    softmax = softmax.view(b, c, h, w)

    # Create coordinate grids
    y_coords = torch.arange(h, dtype=heatmap.dtype, device=device)
    x_coords = torch.arange(w, dtype=heatmap.dtype, device=device)

    if normalized:
        y_coords = y_coords / (h - 1) if h > 1 else y_coords
        x_coords = x_coords / (w - 1) if w > 1 else x_coords

    # Compute expected values
    y_grid = y_coords.view(1, 1, h, 1).expand(b, c, h, w)
    x_grid = x_coords.view(1, 1, 1, w).expand(b, c, h, w)

    y_exp = (softmax * y_grid).sum(dim=(2, 3))
    x_exp = (softmax * x_grid).sum(dim=(2, 3))

    coords = torch.stack([x_exp, y_exp], dim=-1)

    if squeeze:
        coords = coords.squeeze(0)

    return coords


def extract_keypoints_from_heatmap(
    heatmaps: np.ndarray | torch.Tensor,
    confidence_threshold: float = 0.3,
    original_size: tuple[int, int] | None = None,
    method: str = "max",
) -> list[dict[str, Any]]:
    """Extract keypoint coordinates from heatmaps.

    Args:
        heatmaps: Heatmap array of shape (num_keypoints, H, W) or (B, num_keypoints, H, W).
        confidence_threshold: Minimum confidence to consider keypoint detected.
        original_size: Optional (width, height) to scale coordinates to.
        method: Extraction method - "max" for argmax, "soft" for soft-argmax.

    Returns:
        List of keypoint dictionaries with id, x, y, confidence.
    """
    # Convert to numpy if tensor
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    # Handle batch dimension
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]

    num_keypoints, hm_h, hm_w = heatmaps.shape
    keypoints = []

    for kp_id in range(num_keypoints):
        hm = heatmaps[kp_id]
        max_val = np.max(hm)

        if max_val < confidence_threshold:
            continue

        if method == "soft":
            # Soft argmax for sub-pixel accuracy
            hm_tensor = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0)
            coords = soft_argmax_2d(hm_tensor, normalized=True)
            x_norm, y_norm = coords[0, 0].numpy()
            x = x_norm * (hm_w - 1)
            y = y_norm * (hm_h - 1)
        else:
            # Standard argmax
            y, x = np.unravel_index(np.argmax(hm), hm.shape)

        # Scale to original size if provided
        if original_size is not None:
            orig_w, orig_h = original_size
            x = x * orig_w / hm_w
            y = y * orig_h / hm_h

        keypoints.append({
            "id": kp_id,
            "x": float(x),
            "y": float(y),
            "confidence": float(max_val),
        })

    return keypoints


def extract_line_extremities(
    heatmaps: np.ndarray | torch.Tensor,
    num_line_classes: int,
    confidence_threshold: float = 0.5,
    original_size: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Extract line extremity coordinates from heatmaps.

    Args:
        heatmaps: Heatmap array of shape (num_line_classes * 2, H, W).
        num_line_classes: Number of line classes.
        confidence_threshold: Minimum confidence to consider extremity detected.
        original_size: Optional (width, height) to scale coordinates to.

    Returns:
        List of line dictionaries with id, x1, y1, x2, y2, confidence.
    """
    if isinstance(heatmaps, torch.Tensor):
        heatmaps = heatmaps.detach().cpu().numpy()

    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]

    _, hm_h, hm_w = heatmaps.shape
    lines = []

    for line_id in range(num_line_classes):
        # Get heatmaps for both extremities
        ext1_hm = heatmaps[line_id * 2]
        ext2_hm = heatmaps[line_id * 2 + 1]

        max_val1 = np.max(ext1_hm)
        max_val2 = np.max(ext2_hm)

        # Both extremities must be detected
        if max_val1 < confidence_threshold or max_val2 < confidence_threshold:
            continue

        # Get coordinates
        y1, x1 = np.unravel_index(np.argmax(ext1_hm), ext1_hm.shape)
        y2, x2 = np.unravel_index(np.argmax(ext2_hm), ext2_hm.shape)

        # Scale to original size
        if original_size is not None:
            orig_w, orig_h = original_size
            x1 = x1 * orig_w / hm_w
            y1 = y1 * orig_h / hm_h
            x2 = x2 * orig_w / hm_w
            y2 = y2 * orig_h / hm_h

        confidence = min(max_val1, max_val2)
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        lines.append({
            "id": line_id,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "length": float(length),
        })

    return lines


def gaussian_target(
    heatmap_size: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate a 2D Gaussian target heatmap.

    Args:
        heatmap_size: Size of the heatmap (H, W).
        center: Center coordinates (x, y) in heatmap space.
        sigma: Standard deviation of the Gaussian.

    Returns:
        2D Gaussian heatmap of shape (H, W).
    """
    h, w = heatmap_size
    x_c, y_c = center

    y = np.arange(0, h, dtype=np.float32)
    x = np.arange(0, w, dtype=np.float32)
    y, x = np.meshgrid(y, x, indexing='ij')

    heatmap = np.exp(-((x - x_c) ** 2 + (y - y_c) ** 2) / (2 * sigma ** 2))

    return heatmap


def nms_2d(
    heatmap: np.ndarray,
    kernel_size: int = 3,
) -> np.ndarray:
    """Apply 2D non-maximum suppression to heatmap.

    Args:
        heatmap: 2D heatmap array.
        kernel_size: Size of the max pooling kernel.

    Returns:
        Suppressed heatmap with only local maxima.
    """
    hm_tensor = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
    padding = kernel_size // 2
    hm_max = F.max_pool2d(hm_tensor, kernel_size, stride=1, padding=padding)
    keep = (hm_tensor == hm_max).float()
    return (hm_tensor * keep).squeeze().numpy()


def get_multi_peak_coords(
    heatmap: np.ndarray,
    threshold: float = 0.3,
    max_peaks: int = 10,
) -> list[tuple[float, float, float]]:
    """Extract multiple peak coordinates from a heatmap.

    Args:
        heatmap: 2D heatmap array.
        threshold: Minimum value for a peak.
        max_peaks: Maximum number of peaks to return.

    Returns:
        List of (x, y, confidence) tuples.
    """
    # Apply NMS
    hm_nms = nms_2d(heatmap)

    # Find peaks above threshold
    peaks = []
    y_coords, x_coords = np.where(hm_nms >= threshold)

    for y, x in zip(y_coords, x_coords):
        conf = float(hm_nms[y, x])
        peaks.append((float(x), float(y), conf))

    # Sort by confidence and take top-k
    peaks.sort(key=lambda p: p[2], reverse=True)
    return peaks[:max_peaks]


def refine_keypoint_location(
    heatmap: np.ndarray,
    rough_coords: tuple[int, int],
    patch_size: int = 3,
) -> tuple[float, float]:
    """Refine keypoint location using local quadratic fitting.

    Args:
        heatmap: 2D heatmap array.
        rough_coords: Initial (x, y) integer coordinates.
        patch_size: Size of the local patch for fitting.

    Returns:
        Refined (x, y) sub-pixel coordinates.
    """
    x, y = rough_coords
    h, w = heatmap.shape
    half = patch_size // 2

    # Check bounds
    if x < half or x >= w - half or y < half or y >= h - half:
        return float(x), float(y)

    # Extract patch
    patch = heatmap[y - half : y + half + 1, x - half : x + half + 1]

    # Simple subpixel refinement using gradient
    if patch.shape == (3, 3):
        dx = (patch[1, 2] - patch[1, 0]) / 2
        dy = (patch[2, 1] - patch[0, 1]) / 2
        dxx = patch[1, 2] - 2 * patch[1, 1] + patch[1, 0]
        dyy = patch[2, 1] - 2 * patch[1, 1] + patch[0, 1]

        # Avoid division by zero
        if abs(dxx) > 1e-6:
            x_offset = -dx / dxx
            x_offset = np.clip(x_offset, -0.5, 0.5)
        else:
            x_offset = 0.0

        if abs(dyy) > 1e-6:
            y_offset = -dy / dyy
            y_offset = np.clip(y_offset, -0.5, 0.5)
        else:
            y_offset = 0.0

        return float(x + x_offset), float(y + y_offset)

    return float(x), float(y)


def get_keypoints_from_heatmap_maxpool(
    heatmap: torch.Tensor,
    scale: int = 2,
    max_keypoints: int = 1,
    min_keypoint_pixel_distance: int = 1,
    threshold: float = 0.0,
    subpixel_refinement: bool = True,
) -> list[dict[str, Any]]:
    """Extract keypoints using PnLCalib's maxpool-based local maxima detection.

    This function replicates the PnLCalib keypoint extraction method which:
    1. Uses maxpool to find local maxima
    2. Extracts top-k keypoints per channel
    3. Optionally applies sub-pixel refinement using quadratic fitting
    4. Scales coordinates to original image resolution

    Note: The last channel (background) should be excluded before calling this.

    Args:
        heatmap: Heatmap tensor of shape (B, C, H, W) where C excludes background.
        scale: Scale factor to convert heatmap coords to image coords.
        max_keypoints: Max keypoints to extract per class.
        min_keypoint_pixel_distance: Min distance between peaks (for NMS kernel).
        threshold: Min confidence threshold for keypoints.
        subpixel_refinement: If True, apply quadratic fitting for sub-pixel accuracy.
            This typically improves accuracy by 0.5-1.0 pixels. Default True.

    Returns:
        List of keypoint dictionaries with id, x, y, confidence.
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    batch_size, n_channels, height, width = heatmap.shape

    # Convert to numpy for sub-pixel refinement
    heatmap_np = heatmap.detach().cpu().numpy() if subpixel_refinement else None

    # Apply local maxima detection via maxpool
    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = min_keypoint_pixel_distance

    # Pad with 1.0 to suppress border detections (softmax output is [0,1])
    padded_heatmap = F.pad(heatmap, (pad, pad, pad, pad), mode="constant", value=1.0)
    max_pooled_heatmap = F.max_pool2d(padded_heatmap, kernel, stride=1, padding=0)

    # Keep only local maxima
    local_maxima = (max_pooled_heatmap == heatmap).float()
    heatmap_masked = heatmap * local_maxima

    # Extract top-k per channel
    scores, indices = torch.topk(
        heatmap_masked.view(batch_size, n_channels, -1),
        max_keypoints,
        sorted=True
    )

    # Convert flat indices to (y, x) coordinates
    indices_y = torch.div(indices, width, rounding_mode="floor")
    indices_x = indices % width

    # Move to CPU
    indices_y = indices_y.detach().cpu().numpy()
    indices_x = indices_x.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    # Extract keypoints (using first batch only)
    keypoints = []
    for channel_idx in range(n_channels):
        for kp_idx in range(max_keypoints):
            score = scores[0, channel_idx, kp_idx]
            if score < threshold:
                continue

            x_int = int(indices_x[0, channel_idx, kp_idx])
            y_int = int(indices_y[0, channel_idx, kp_idx])

            # Apply sub-pixel refinement if enabled
            if subpixel_refinement and heatmap_np is not None:
                channel_hm = heatmap_np[0, channel_idx]
                x_refined, y_refined = refine_keypoint_location(
                    channel_hm, (x_int, y_int), patch_size=3
                )
                x = x_refined * scale
                y = y_refined * scale
            else:
                x = x_int * scale
                y = y_int * scale

            keypoints.append({
                "id": channel_idx,
                "x": float(x),
                "y": float(y),
                "confidence": float(score),
            })

    return keypoints


def get_lines_from_heatmap_maxpool(
    heatmap: torch.Tensor,
    scale: int = 2,
    min_keypoint_pixel_distance: int = 1,
    threshold: float = 0.0,
) -> list[dict[str, Any]]:
    """Extract line extremities using PnLCalib's maxpool-based detection.

    PnLCalib line model structure:
    - Each channel represents one line class
    - For each channel, the TOP 2 peaks are the two endpoints of that line
    - The last channel is background (should be excluded before calling)

    Note: This function does NOT pad with 1.0 (unlike keypoint extraction).
    The line model uses standard max_pool2d padding for NMS.

    Args:
        heatmap: Heatmap tensor of shape (B, num_line_classes, H, W).
        scale: Scale factor to convert heatmap coords to image coords.
        min_keypoint_pixel_distance: Min distance between peaks.
        threshold: Min confidence threshold for BOTH endpoints.

    Returns:
        List of line dictionaries with id, x1, y1, x2, y2, confidence.
    """
    if heatmap.dim() == 3:
        heatmap = heatmap.unsqueeze(0)

    batch_size, n_channels, _, width = heatmap.shape

    # Apply local maxima detection using max pooling
    # Note: For lines, we use standard padding (NOT padding with 1.0 like keypoints)
    kernel = min_keypoint_pixel_distance * 2 + 1
    pad = int((kernel - 1) / 2)

    max_pooled_heatmap = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    # Keep only local maxima (where value equals max pooled value)
    local_maxima = (max_pooled_heatmap == heatmap)
    # Zero out non-maxima
    heatmap = heatmap * local_maxima

    # Extract top-2 per channel (the two endpoints of each line)
    scores, indices = torch.topk(
        heatmap.view(batch_size, n_channels, -1),
        k=2,  # Always extract 2 points per line
        sorted=True
    )

    # Convert flat indices to (row, col) coordinates
    indices_row = torch.div(indices, width, rounding_mode="floor")
    indices_col = indices % width

    # Move to CPU
    indices_row = indices_row.detach().cpu().numpy()
    indices_col = indices_col.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()

    # Extract lines - each channel is one line class
    lines = []
    for line_idx in range(n_channels):
        # Get both endpoints for this line
        score1 = scores[0, line_idx, 0]
        score2 = scores[0, line_idx, 1]

        # Both endpoints must meet threshold
        if score1 < threshold or score2 < threshold:
            continue

        # PnLCalib returns (u, v) = (col, row) * scale
        # u = x coordinate, v = y coordinate
        x1 = indices_col[0, line_idx, 0] * scale
        y1 = indices_row[0, line_idx, 0] * scale
        x2 = indices_col[0, line_idx, 1] * scale
        y2 = indices_row[0, line_idx, 1] * scale

        confidence = min(float(score1), float(score2))
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        lines.append({
            "id": line_idx,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(confidence),
            "length": float(length),
        })

    return lines
