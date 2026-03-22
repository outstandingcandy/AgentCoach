"""Bidirectional smoothing for offline homography chain calibration.

Combines forward and backward propagation with confidence-weighted
averaging for smooth, accurate calibration results.
"""

from typing import Any

import numpy as np


class BidirectionalSmoother:
    """Smooth calibration using bidirectional propagation.

    For offline processing, propagates from both anchor endpoints
    and merges results using distance-based weighting.
    """

    def __init__(
        self,
        temporal_smoothing_window: int = 5,
        min_confidence: float = 0.1,
    ):
        """Initialize smoother.

        Args:
            temporal_smoothing_window: Window size for temporal smoothing.
            min_confidence: Minimum confidence to include in averaging.
        """
        self.temporal_smoothing_window = temporal_smoothing_window
        self.min_confidence = min_confidence

    def merge_bidirectional(
        self,
        forward_result: dict[str, Any],
        backward_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge forward and backward propagation results.

        Uses distance-based weighting: frames closer to their anchor
        get higher weight from that direction.

        Args:
            forward_result: Result from forward propagation.
            backward_result: Result from backward propagation.

        Returns:
            Merged result.
        """
        H_fwd = forward_result.get("H")
        H_bwd = backward_result.get("H")

        if H_fwd is None and H_bwd is None:
            return {"H": None, "confidence": 0.0, "source": "none"}

        if H_fwd is None:
            return backward_result

        if H_bwd is None:
            return forward_result

        # Use distance from anchor for weighting (more reliable than confidence)
        dist_fwd = forward_result.get("frames_from_anchor", 1)
        dist_bwd = backward_result.get("frames_from_anchor", 1)

        # Inverse distance weighting: closer to anchor = higher weight
        # Add small epsilon to avoid division by zero
        w_fwd = 1.0 / (dist_fwd + 1)
        w_bwd = 1.0 / (dist_bwd + 1)
        w_total = w_fwd + w_bwd

        w_fwd /= w_total
        w_bwd /= w_total

        # Weighted average of homography matrices
        H_merged = w_fwd * H_fwd + w_bwd * H_bwd

        # Normalize
        H_merged = H_merged / H_merged[2, 2]

        # Combined confidence based on weights
        conf_fwd = forward_result.get("confidence", 0.5)
        conf_bwd = backward_result.get("confidence", 0.5)
        merged_conf = w_fwd * conf_fwd + w_bwd * conf_bwd

        return {
            "H": H_merged,
            "confidence": merged_conf,
            "source": "bidirectional_merge",
            "forward_weight": w_fwd,
            "backward_weight": w_bwd,
            "forward_distance": dist_fwd,
            "backward_distance": dist_bwd,
        }

    def smooth_sequence(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Apply temporal smoothing to a sequence of results.

        Args:
            results: List of calibration results with "H" and "confidence".

        Returns:
            Smoothed results.
        """
        if not results:
            return []

        n = len(results)
        smoothed = []

        half_window = self.temporal_smoothing_window // 2

        for i in range(n):
            # Get window bounds
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Collect valid results in window
            window_H = []
            window_conf = []

            for j in range(start, end):
                H = results[j].get("H")
                conf = results[j].get("confidence", 0.0)

                if H is not None and conf >= self.min_confidence:
                    window_H.append(H)
                    window_conf.append(conf)

            if not window_H:
                smoothed.append(results[i])
                continue

            # Weighted average
            weights = np.array(window_conf)
            weights = weights / weights.sum()

            H_smooth = np.zeros((3, 3))
            for H, w in zip(window_H, weights):
                H_smooth += w * H

            H_smooth = H_smooth / H_smooth[2, 2]

            smoothed_result = results[i].copy()
            smoothed_result["H"] = H_smooth
            smoothed_result["H_unsmoothed"] = results[i].get("H")
            smoothed_result["smoothing_window_size"] = len(window_H)

            smoothed.append(smoothed_result)

        return smoothed

    def smooth_camera_params(
        self,
        params_sequence: list[dict],
    ) -> list[dict]:
        """Apply smoothing to camera parameter sequence.

        Args:
            params_sequence: List of camera parameters dicts.

        Returns:
            Smoothed parameter sequence.
        """
        if not params_sequence:
            return []

        n = len(params_sequence)
        half_window = self.temporal_smoothing_window // 2

        smoothed = []

        for i in range(n):
            start = max(0, i - half_window)
            end = min(n, i + half_window + 1)

            # Collect params in window
            window = params_sequence[start:end]

            # Average each parameter
            smoothed_params = params_sequence[i].copy()

            for key in ["panDegrees", "tiltDegrees", "rollDegrees",
                        "horizontalFieldOfViewDegrees"]:
                values = [p.get(key, 0) for p in window if key in p]
                if values:
                    smoothed_params[key] = np.mean(values)

            smoothed.append(smoothed_params)

        return smoothed

    def apply_gaussian_smoothing(
        self,
        results: list[dict[str, Any]],
        sigma: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Apply Gaussian temporal smoothing.

        Args:
            results: List of calibration results.
            sigma: Gaussian kernel sigma.

        Returns:
            Smoothed results.
        """
        if not results:
            return []

        n = len(results)
        window_size = int(4 * sigma) + 1  # 4 sigma covers 99.99%

        # Create Gaussian kernel
        x = np.arange(-window_size, window_size + 1)
        kernel = np.exp(-x**2 / (2 * sigma**2))

        smoothed = []

        for i in range(n):
            H_sum = np.zeros((3, 3))
            weight_sum = 0.0

            for offset in range(-window_size, window_size + 1):
                j = i + offset
                if j < 0 or j >= n:
                    continue

                H = results[j].get("H")
                conf = results[j].get("confidence", 0.0)

                if H is None or conf < self.min_confidence:
                    continue

                k_weight = kernel[offset + window_size]
                weight = k_weight * conf

                H_sum += weight * H
                weight_sum += weight

            if weight_sum > 0:
                H_smooth = H_sum / weight_sum
                H_smooth = H_smooth / H_smooth[2, 2]

                smoothed_result = results[i].copy()
                smoothed_result["H"] = H_smooth
                smoothed.append(smoothed_result)
            else:
                smoothed.append(results[i])

        return smoothed
