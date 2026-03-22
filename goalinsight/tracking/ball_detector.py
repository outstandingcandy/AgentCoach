"""Ball detection using YOLOv8.

Specialized detector for soccer ball with adjustments for:
- Smaller object size (lower confidence threshold)
- Higher resolution for small object detection
- Size and aspect ratio filtering specific to balls
"""

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class BallDetector:
    """Detect soccer ball in video frames using YOLOv8."""

    # COCO class ID for sports ball
    BALL_CLASS_ID = 32

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ball detector.

        Args:
            config: Detection configuration with ball-specific defaults.
        """
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.config = config or {}
        self.model_name = self.config.get("model", "yolov8x")
        # Lower threshold for small ball detection
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        # Sports ball class (32 in COCO)
        self.classes = self.config.get("classes", [self.BALL_CLASS_ID])
        # Higher resolution for small object detection
        self.imgsz = self.config.get("imgsz", 1920)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Size filtering parameters
        self.min_size = self.config.get("min_size", 5)
        self.max_size = self.config.get("max_size", 100)
        self.min_aspect_ratio = self.config.get("min_aspect_ratio", 0.5)
        self.max_aspect_ratio = self.config.get("max_aspect_ratio", 2.0)

        self.model = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load YOLOv8 model.

        Args:
            model_path: Path to model weights. If None, downloads from ultralytics hub.
        """
        if model_path:
            self.model = YOLO(str(model_path))
        else:
            # Auto-download from ultralytics hub
            self.model = YOLO(f"{self.model_name}.pt")

        self.model.to(self.device)

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect ball in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            List of detection dictionaries with bbox, confidence, center.
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                detections.append({
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "center": [float(center_x), float(center_y)],
                    "confidence": conf,
                    "class": cls,
                    "class_name": self.model.names[cls],
                })

        return detections

    def filter_by_size(
        self,
        detections: list[dict[str, Any]],
        min_size: int | None = None,
        max_size: int | None = None,
        min_aspect_ratio: float | None = None,
        max_aspect_ratio: float | None = None,
    ) -> list[dict[str, Any]]:
        """Filter detections by size and aspect ratio.

        Balls should be roughly circular (aspect ratio ~1.0) and
        within a reasonable size range.

        Args:
            detections: List of detections.
            min_size: Minimum bbox dimension (default from config).
            max_size: Maximum bbox dimension (default from config).
            min_aspect_ratio: Minimum width/height ratio.
            max_aspect_ratio: Maximum width/height ratio.

        Returns:
            Filtered detections.
        """
        min_size = min_size if min_size is not None else self.min_size
        max_size = max_size if max_size is not None else self.max_size
        min_ar = min_aspect_ratio if min_aspect_ratio is not None else self.min_aspect_ratio
        max_ar = max_aspect_ratio if max_aspect_ratio is not None else self.max_aspect_ratio

        filtered = []
        for det in detections:
            bbox = det["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            # Check size constraints
            if width < min_size or height < min_size:
                continue
            if width > max_size or height > max_size:
                continue

            # Check aspect ratio (ball should be roughly circular)
            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < min_ar or aspect_ratio > max_ar:
                continue

            filtered.append(det)

        return filtered

    def filter_by_pitch(
        self,
        detections: list[dict[str, Any]],
        homography: np.ndarray | None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        margin: float = 2.0,
    ) -> list[dict[str, Any]]:
        """Filter detections to only include those on the pitch.

        Args:
            detections: List of detections.
            homography: Image -> world homography matrix.
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.
            margin: Extra margin around pitch boundary.

        Returns:
            Filtered detections that are on the pitch.
        """
        if homography is None:
            return detections

        half_l = pitch_length / 2 + margin
        half_w = pitch_width / 2 + margin

        filtered = []
        for det in detections:
            center = det["center"]

            # Project to world coordinates
            pt_h = np.array([center[0], center[1], 1.0])
            world_h = homography @ pt_h

            if abs(world_h[2]) < 1e-6:
                continue

            world_x = world_h[0] / world_h[2]
            world_y = world_h[1] / world_h[2]

            # Check if on pitch
            if -half_l <= world_x <= half_l and -half_w <= world_y <= half_w:
                det["pitch_position"] = [float(world_x), float(world_y)]
                filtered.append(det)

        return filtered

    def get_best_detection(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Get the best ball detection from a list.

        Selection criteria:
        1. Highest confidence
        2. Most circular (aspect ratio closest to 1.0)

        Args:
            detections: List of detections.

        Returns:
            Best detection or None if no detections.
        """
        if not detections:
            return None

        def score(det: dict[str, Any]) -> float:
            bbox = det["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = width / height if height > 0 else 0

            # Higher confidence is better
            conf_score = det["confidence"]
            # Closer to 1.0 aspect ratio is better
            ar_score = 1.0 - abs(1.0 - aspect_ratio)

            return conf_score * 0.7 + ar_score * 0.3

        return max(detections, key=score)
