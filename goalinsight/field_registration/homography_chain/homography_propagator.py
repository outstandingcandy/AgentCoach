"""Homography chain propagation from anchor frames.

Propagates calibration from high-confidence anchor frames to
low-confidence frames using accumulated frame-to-frame homographies.

Improved method: Extract rotation from homography decomposition
to avoid accumulated drift in camera parameter extraction.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class PropagationState:
    """State for tracking propagation from an anchor."""

    anchor_frame: int
    anchor_H: np.ndarray  # World -> Image homography at anchor
    accumulated_H: np.ndarray = field(default_factory=lambda: np.eye(3))
    frame_count: int = 0
    cumulative_drift: float = 0.0


class HomographyPropagator:
    """Chain homographies from anchor frames to estimate calibration.

    Given a calibrated anchor frame with homography H_anchor, propagates
    to target frames using:
        H_target = H_anchor @ delta_H_anchor_to_target

    where delta_H is the accumulated frame-to-frame homography.
    """

    def __init__(self, max_drift_threshold: float = 100.0):
        """Initialize propagator.

        Args:
            max_drift_threshold: Maximum cumulative drift before re-anchoring.
        """
        self.max_drift_threshold = max_drift_threshold

        # Forward propagation state (from earlier anchor)
        self.forward_state: PropagationState | None = None

        # Backward propagation state (from later anchor)
        self.backward_state: PropagationState | None = None

        # Cache of computed homographies per frame
        self._cache: dict[int, dict[str, Any]] = {}

    def set_anchor(
        self,
        frame_idx: int,
        H_world_to_image: np.ndarray,
        direction: str = "forward",
    ) -> None:
        """Set anchor point for propagation.

        Args:
            frame_idx: Frame index of anchor.
            H_world_to_image: World -> Image homography at anchor.
            direction: "forward" or "backward" propagation direction.
        """
        state = PropagationState(
            anchor_frame=frame_idx,
            anchor_H=H_world_to_image.copy(),
            accumulated_H=np.eye(3),
            frame_count=0,
            cumulative_drift=0.0,
        )

        if direction == "forward":
            self.forward_state = state
        else:
            self.backward_state = state

        # Cache anchor
        self._cache[frame_idx] = {
            "H": H_world_to_image.copy(),
            "confidence": 1.0,
            "source": "anchor",
            "drift": 0.0,
        }

    def _normalize_homography(self, H: np.ndarray) -> np.ndarray:
        """Normalize homography to preserve scale (det = 1).

        This prevents drift accumulation from small scale factors.

        Args:
            H: Input homography.

        Returns:
            Normalized homography with det ≈ 1.
        """
        det = np.linalg.det(H)
        if abs(det) > 1e-6:
            # Scale to make det = 1 (cube root because 3x3 matrix)
            scale = np.sign(det) * (abs(det) ** (1/3))
            return H / scale
        return H

    def propagate_forward(
        self,
        target_frame: int,
        delta_H: np.ndarray,
        match_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propagate calibration forward to next frame.

        Args:
            target_frame: Target frame index.
            delta_H: Frame-to-frame homography (prev_frame -> target_frame).
            match_metadata: Feature matching metadata.

        Returns:
            Propagated calibration result.
        """
        if self.forward_state is None:
            raise ValueError("Forward anchor not set")

        # Normalize delta_H to prevent scale drift accumulation
        delta_H_normalized = self._normalize_homography(delta_H)

        # Accumulate homography: delta_H transforms image coords
        # delta_H: maps points in prev_frame to target_frame
        # accumulated_H: maps points from anchor_frame to current_frame
        # new_accumulated_H: maps points from anchor_frame to target_frame
        self.forward_state.accumulated_H = delta_H_normalized @ self.forward_state.accumulated_H
        self.forward_state.frame_count += 1

        # Compute drift from delta_H translation
        drift = self._estimate_drift(delta_H)
        self.forward_state.cumulative_drift += drift

        # Compute target homography
        # H_target (world -> image) = accumulated_H @ H_anchor
        H_target = self.forward_state.accumulated_H @ self.forward_state.anchor_H

        # Normalize homography
        H_target = H_target / H_target[2, 2]

        # Estimate confidence based on inlier ratio and drift
        confidence = self._estimate_confidence(
            match_metadata,
            self.forward_state.cumulative_drift,
            self.forward_state.frame_count,
        )

        result = {
            "H": H_target,
            "confidence": confidence,
            "source": "forward_propagation",
            "anchor_frame": self.forward_state.anchor_frame,
            "frames_from_anchor": self.forward_state.frame_count,
            "drift": self.forward_state.cumulative_drift,
            "match_metadata": match_metadata,
        }

        self._cache[target_frame] = result
        return result

    def propagate_backward(
        self,
        target_frame: int,
        delta_H: np.ndarray,
        match_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Propagate calibration backward to previous frame.

        Args:
            target_frame: Target frame index.
            delta_H: Frame-to-frame homography (next_frame -> target_frame).
            match_metadata: Feature matching metadata.

        Returns:
            Propagated calibration result.
        """
        if self.backward_state is None:
            raise ValueError("Backward anchor not set")

        # Normalize delta_H to prevent scale drift accumulation
        delta_H_normalized = self._normalize_homography(delta_H)

        self.backward_state.accumulated_H = delta_H_normalized @ self.backward_state.accumulated_H
        self.backward_state.frame_count += 1

        drift = self._estimate_drift(delta_H)
        self.backward_state.cumulative_drift += drift

        H_target = self.backward_state.accumulated_H @ self.backward_state.anchor_H
        H_target = H_target / H_target[2, 2]

        confidence = self._estimate_confidence(
            match_metadata,
            self.backward_state.cumulative_drift,
            self.backward_state.frame_count,
        )

        result = {
            "H": H_target,
            "confidence": confidence,
            "source": "backward_propagation",
            "anchor_frame": self.backward_state.anchor_frame,
            "frames_from_anchor": self.backward_state.frame_count,
            "drift": self.backward_state.cumulative_drift,
            "match_metadata": match_metadata,
        }

        # Store but don't overwrite if forward already exists
        if target_frame not in self._cache:
            self._cache[target_frame] = result
        else:
            self._cache[target_frame]["backward"] = result

        return result

    def _estimate_drift(self, H: np.ndarray) -> float:
        """Estimate drift magnitude from homography.

        Args:
            H: Frame-to-frame homography.

        Returns:
            Drift magnitude estimate.
        """
        # Translation component gives drift estimate
        if H is None:
            return 10.0  # Penalty for missing homography

        # H[0,2] and H[1,2] are translation components
        tx, ty = H[0, 2], H[1, 2]
        translation_drift = np.sqrt(tx**2 + ty**2)

        # Also consider perspective distortion
        h31, h32 = H[2, 0], H[2, 1]
        perspective_drift = np.sqrt(h31**2 + h32**2) * 1000

        return translation_drift + perspective_drift

    def _estimate_confidence(
        self,
        match_metadata: dict[str, Any] | None,
        cumulative_drift: float,
        frames_from_anchor: int,
    ) -> float:
        """Estimate confidence score for propagated homography.

        Args:
            match_metadata: Feature matching metadata.
            cumulative_drift: Cumulative drift from anchor.
            frames_from_anchor: Number of frames from anchor.

        Returns:
            Confidence score [0, 1].
        """
        # Base confidence from inlier ratio
        if match_metadata:
            inlier_ratio = match_metadata.get("inlier_ratio", 0.5)
            base_confidence = inlier_ratio
        else:
            base_confidence = 0.5

        # Decay based on drift
        drift_factor = np.exp(-cumulative_drift / self.max_drift_threshold)

        # Decay based on distance from anchor
        distance_factor = np.exp(-frames_from_anchor / 100)

        return base_confidence * drift_factor * distance_factor

    def get_cached_result(self, frame_idx: int) -> dict[str, Any] | None:
        """Get cached homography result for frame.

        Args:
            frame_idx: Frame index.

        Returns:
            Cached result or None.
        """
        return self._cache.get(frame_idx)

    def should_re_anchor(self, confidence_threshold: float = 0.3) -> bool:
        """Check if re-anchoring is needed due to drift.

        Args:
            confidence_threshold: Minimum confidence threshold.

        Returns:
            True if re-anchoring is recommended.
        """
        if self.forward_state is None:
            return True

        current_confidence = self._estimate_confidence(
            None,
            self.forward_state.cumulative_drift,
            self.forward_state.frame_count,
        )

        return current_confidence < confidence_threshold

    def clear_cache(self) -> None:
        """Clear cached results."""
        self._cache.clear()

    def reset(self) -> None:
        """Reset all state."""
        self.forward_state = None
        self.backward_state = None
        self._cache.clear()

    @staticmethod
    def extract_pan_from_homography(
        delta_H: np.ndarray,
        K: np.ndarray,
    ) -> float | None:
        """Extract pan rotation angle from frame-to-frame homography.

        Uses cv2.decomposeHomographyMat to extract the rotation matrix,
        then computes the pan angle (rotation around Z axis).

        This is more accurate than accumulating homographies and then
        extracting camera parameters, which suffers from numerical drift.

        Args:
            delta_H: Frame-to-frame homography (3x3).
            K: Camera intrinsic matrix (3x3).

        Returns:
            Pan angle change in degrees, or None if decomposition fails.
            Negative = camera panning left, positive = panning right.
        """
        try:
            num, Rs, ts, normals = cv2.decomposeHomographyMat(delta_H, K)

            if num == 0:
                return None

            # Select solution with positive Z normal (plane facing camera)
            best_idx = 0
            for i in range(num):
                if normals[i][2, 0] > 0:
                    best_idx = i
                    break

            R = Rs[best_idx]

            # Extract pan angle from rotation matrix
            # For broadcast camera: R = Rz(pan) @ Rx(tilt)
            # pan = atan2(R[1,0], R[0,0])
            # Negate because homography convention is inverted
            delta_pan_rad = -np.arctan2(R[1, 0], R[0, 0])

            return float(np.degrees(delta_pan_rad))

        except Exception:
            return None

    @staticmethod
    def build_intrinsic_matrix(
        image_width: int,
        image_height: int,
        hfov_degrees: float,
    ) -> np.ndarray:
        """Build camera intrinsic matrix from FOV.

        Args:
            image_width: Image width in pixels.
            image_height: Image height in pixels.
            hfov_degrees: Horizontal field of view in degrees.

        Returns:
            3x3 camera intrinsic matrix K.
        """
        hfov_rad = np.radians(hfov_degrees)
        focal = image_width / (2 * np.tan(hfov_rad / 2))

        return np.array([
            [focal, 0, image_width / 2],
            [0, focal, image_height / 2],
            [0, 0, 1],
        ], dtype=np.float64)
