"""Jersey color histogram extraction for team classification."""

import cv2
import numpy as np


def _extract_jersey_color_hist(crop: np.ndarray, h_bins: int = 30, s_bins: int = 32) -> np.ndarray | None:
    """Extract HSV color histogram from upper body (jersey) region.

    Takes the top 50% of the crop (torso area), converts to HSV, and computes
    a normalized H+S histogram. Ignores V channel for illumination invariance.

    Returns:
        Flattened, L2-normalized histogram vector, or None if crop is too small.
    """
    h, w = crop.shape[:2]
    if h < 10 or w < 5:
        return None

    # Upper 50% = jersey/torso area (skip head ~top 15%)
    y_top = max(0, int(h * 0.15))
    y_bot = int(h * 0.65)
    jersey = crop[y_top:y_bot, :]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)

    # 2D histogram on H and S channels
    hist = cv2.calcHist([hsv], [0, 1], None, [h_bins, s_bins],
                        [0, 180, 0, 256])
    hist = hist.flatten().astype(np.float32)
    norm = np.linalg.norm(hist)
    if norm > 1e-6:
        hist /= norm
    return hist


def _extract_jersey_mean_saturation(crop: np.ndarray) -> float | None:
    """Extract mean saturation of jersey region. Low saturation = achromatic (black/white/grey).

    Returns:
        Mean saturation (0-255), or None if crop is too small.
    """
    h, w = crop.shape[:2]
    if h < 10 or w < 5:
        return None

    y_top = max(0, int(h * 0.15))
    y_bot = int(h * 0.65)
    jersey = crop[y_top:y_bot, :]

    if jersey.size == 0:
        return None

    hsv = cv2.cvtColor(jersey, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())
