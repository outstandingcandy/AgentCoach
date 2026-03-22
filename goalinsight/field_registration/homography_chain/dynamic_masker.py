"""Dynamic masking to exclude players and ball from feature matching.

Uses existing PlayerDetector to detect and mask moving objects,
improving feature matching quality for homography estimation.
"""

from typing import Any

import cv2
import numpy as np


class DynamicMasker:
    """Create binary masks excluding players and ball from frames.

    Masks are used to improve feature matching by excluding
    non-static regions (players, ball) that would cause false matches.
    """

    def __init__(
        self,
        dilation_kernel_size: int = 15,
        player_detector: Any | None = None,
    ):
        """Initialize dynamic masker.

        Args:
            dilation_kernel_size: Kernel size for dilating detection masks.
            player_detector: Optional PlayerDetector instance. If None, will be
                created lazily when needed.
        """
        self.dilation_kernel_size = dilation_kernel_size
        self.dilation_kernel = None
        if dilation_kernel_size > 0:
            self.dilation_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (dilation_kernel_size, dilation_kernel_size),
            )
        self._player_detector = player_detector
        self._detector_loaded = False

    @property
    def player_detector(self):
        """Lazy-load player detector."""
        if self._player_detector is None:
            from goalinsight.tracking.detector import PlayerDetector
            self._player_detector = PlayerDetector({
                "confidence_threshold": 0.3,
                "classes": [0, 32],  # Person and sports ball
            })
        if not self._detector_loaded:
            self._player_detector.load_model()
            self._detector_loaded = True
        return self._player_detector

    def create_mask(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]] | None = None,
        padding: float = 0.15,
    ) -> np.ndarray:
        """Create binary mask excluding detected objects.

        Args:
            frame: Input frame (BGR).
            detections: Pre-computed detections. If None, will run detection.
            padding: Padding ratio around bounding boxes.

        Returns:
            Binary mask where 255 = valid region, 0 = masked region.
        """
        h, w = frame.shape[:2]
        mask = np.ones((h, w), dtype=np.uint8) * 255

        if detections is None:
            detections = self.player_detector.detect(frame)

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox

            # Add padding
            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = int(box_w * padding)
            pad_h = int(box_h * padding)

            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(w, int(x2 + pad_w))
            y2 = min(h, int(y2 + pad_h))

            # Fill with 0 (masked)
            mask[y1:y2, x1:x2] = 0

        # Dilate mask to ensure border pixels are excluded
        if self.dilation_kernel_size > 0 and self.dilation_kernel is not None:
            # Invert, dilate, invert back
            mask_inv = cv2.bitwise_not(mask)
            mask_inv = cv2.dilate(mask_inv, self.dilation_kernel, iterations=1)
            mask = cv2.bitwise_not(mask_inv)

        return mask

    def create_mask_from_boxes(
        self,
        frame_shape: tuple[int, int],
        bboxes: list[list[float]],
        padding: float = 0.15,
    ) -> np.ndarray:
        """Create mask from pre-computed bounding boxes.

        Args:
            frame_shape: Frame dimensions (height, width).
            bboxes: List of bounding boxes [x1, y1, x2, y2].
            padding: Padding ratio around bounding boxes.

        Returns:
            Binary mask.
        """
        h, w = frame_shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            box_w = x2 - x1
            box_h = y2 - y1
            pad_w = int(box_w * padding)
            pad_h = int(box_h * padding)

            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(w, int(x2 + pad_w))
            y2 = min(h, int(y2 + pad_h))

            mask[y1:y2, x1:x2] = 0

        if self.dilation_kernel_size > 0:
            mask_inv = cv2.bitwise_not(mask)
            mask_inv = cv2.dilate(mask_inv, self.dilation_kernel, iterations=1)
            mask = cv2.bitwise_not(mask_inv)

        return mask

    def create_static_mask(
        self,
        frame_shape: tuple[int, int],
        exclude_bottom_ratio: float = 0.05,
        exclude_top_ratio: float = 0.08,
    ) -> np.ndarray:
        """Create static mask excluding common overlay regions.

        Args:
            frame_shape: Frame dimensions (height, width).
            exclude_bottom_ratio: Ratio of bottom to exclude (ads/watermarks).
            exclude_top_ratio: Ratio of top to exclude (scoreboards).

        Returns:
            Static binary mask.
        """
        h, w = frame_shape
        mask = np.ones((h, w), dtype=np.uint8) * 255

        # Exclude top (scoreboard)
        top_exclude = int(h * exclude_top_ratio)
        mask[:top_exclude, :] = 0

        # Exclude bottom (ads)
        bottom_exclude = int(h * exclude_bottom_ratio)
        if bottom_exclude > 0:
            mask[-bottom_exclude:, :] = 0

        return mask

    def combine_masks(self, *masks: np.ndarray) -> np.ndarray:
        """Combine multiple masks with AND operation.

        Args:
            *masks: Variable number of binary masks.

        Returns:
            Combined mask.
        """
        if not masks:
            raise ValueError("At least one mask required")

        result = masks[0].copy()
        for mask in masks[1:]:
            result = cv2.bitwise_and(result, mask)

        return result
