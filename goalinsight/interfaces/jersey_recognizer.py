"""Abstract base class for jersey number recognition backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseJerseyRecognizer(ABC):
    """Abstract base class for jersey number recognition.

    Implementations:
    - Qwen VL: Vision-language model for OCR
    - Claude / Gemini: hosted vision-language model backends
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize jersey recognizer.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def recognize(self, crop: np.ndarray) -> tuple[int | None, float]:
        """Recognize jersey number from a single crop.

        Args:
            crop: Player image crop (BGR format from OpenCV).

        Returns:
            Tuple of (jersey_number, confidence).
            jersey_number is None if no number detected.
            confidence is between 0 and 1.
        """
        pass

    @abstractmethod
    def recognize_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[int | None, float]]:
        """Recognize jersey numbers from multiple crops.

        Args:
            crops: List of player image crops.

        Returns:
            List of (jersey_number, confidence) tuples.
        """
        pass

    def get_upper_body_crop(
        self,
        crop: np.ndarray,
        upper_ratio: float = 0.6,
    ) -> np.ndarray:
        """Extract upper body region from player crop.

        Jersey numbers are typically on the chest/back, so focusing
        on the upper body can improve recognition accuracy.

        Args:
            crop: Full player crop.
            upper_ratio: Ratio of height to keep from top.

        Returns:
            Upper body crop.
        """
        if crop is None or crop.size == 0:
            return crop

        h, w = crop.shape[:2]
        upper_h = int(h * upper_ratio)
        return crop[:upper_h, :]

    @property
    def name(self) -> str:
        """Return backend name for logging."""
        return self.__class__.__name__
