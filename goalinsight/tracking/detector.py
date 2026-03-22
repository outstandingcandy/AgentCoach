"""Player detection using YOLOv8."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PlayerDetector:
    """Detect players in soccer video frames using YOLOv8."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize player detector.

        Args:
            config: Detection configuration.
        """
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.config = config or {}
        self.model_name = self.config.get("model", "yolov8x")
        # Lower threshold to 0.4 to match official sn-gamestate baseline
        self.confidence_threshold = self.config.get("confidence_threshold", 0.4)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.classes = self.config.get("classes", [0])  # Person class
        self.imgsz = self.config.get("imgsz", 1280)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

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
        """Detect players in a single frame.

        Args:
            frame: Input frame (BGR format from OpenCV).

        Returns:
            List of detection dictionaries with bbox, confidence, class.
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

                detections.append({
                    "bbox": box.tolist(),  # [x1, y1, x2, y2]
                    "confidence": conf,
                    "class": cls,
                    "class_name": self.model.names[cls],
                })

        return detections

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """Detect players in multiple frames.

        Args:
            frames: List of input frames.

        Returns:
            List of detection lists for each frame.
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
        )

        all_detections = []
        for result in results:
            frame_detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    frame_detections.append({
                        "bbox": box.tolist(),
                        "confidence": conf,
                        "class": cls,
                        "class_name": self.model.names[cls],
                    })

            all_detections.append(frame_detections)

        return all_detections

    def filter_by_size(
        self,
        detections: list[dict[str, Any]],
        min_height: int = 50,
        max_height: int = 500,
        min_aspect_ratio: float = 0.2,
        max_aspect_ratio: float = 1.5,
    ) -> list[dict[str, Any]]:
        """Filter detections by size constraints.

        Args:
            detections: List of detections.
            min_height: Minimum bounding box height.
            max_height: Maximum bounding box height.
            min_aspect_ratio: Minimum width/height ratio.
            max_aspect_ratio: Maximum width/height ratio.

        Returns:
            Filtered detections.
        """
        filtered = []
        for det in detections:
            bbox = det["bbox"]
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]

            if height < min_height or height > max_height:
                continue

            aspect_ratio = width / height if height > 0 else 0
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue

            filtered.append(det)

        return filtered

    def get_detection_crops(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        padding: float = 0.1,
    ) -> list[np.ndarray]:
        """Extract image crops for each detection.

        Args:
            frame: Input frame.
            detections: List of detections.
            padding: Padding ratio around bounding box.

        Returns:
            List of image crops.
        """
        h, w = frame.shape[:2]
        crops = []

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

            crop = frame[y1:y2, x1:x2]
            crops.append(crop)

        return crops

    def filter_by_pitch(
        self,
        detections: list[dict[str, Any]],
        homography: np.ndarray | None,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        margin: float = 5.0,
    ) -> list[dict[str, Any]]:
        """Filter detections to only include those on the pitch.

        Args:
            detections: List of detections.
            homography: Homography matrix. Can be either:
                - world->image (will be inverted)
                - image->world (used directly if it gives reasonable results)
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.
            margin: Extra margin around pitch boundary.

        Returns:
            Filtered detections that are on the pitch.
        """
        if homography is None:
            return detections

        # Try to determine homography direction by testing center point
        # If H is world->image, then H @ [0,0,1] should give image center
        # If H is image->world, then H @ image_center should give world origin
        try:
            H_inv = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return detections

        # Test which direction the homography is in
        # Most homographies from calibration are world->image
        # We need image->world for projection
        test_pt = np.array([0, 0, 1.0])
        result = homography @ test_pt
        result_x = result[0] / result[2] if abs(result[2]) > 1e-6 else 1e6

        # If result is in reasonable image range (0-2000), H is world->image
        # So we use H_inv for image->world projection
        if 0 < result_x < 2000:
            proj_matrix = H_inv  # H is world->image, use inverse
        else:
            proj_matrix = homography  # H is already image->world

        half_l = pitch_length / 2 + margin
        half_w = pitch_width / 2 + margin

        filtered = []
        for det in detections:
            bbox = det["bbox"]
            # Use foot position (bottom center of bbox)
            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]

            # Project to world coordinates
            pt_h = np.array([foot_x, foot_y, 1.0])
            world_h = proj_matrix @ pt_h

            if abs(world_h[2]) < 1e-6:
                continue

            world_x = world_h[0] / world_h[2]
            world_y = world_h[1] / world_h[2]

            # Check if on pitch
            if -half_l <= world_x <= half_l and -half_w <= world_y <= half_w:
                det["pitch_position"] = [float(world_x), float(world_y)]
                filtered.append(det)

        return filtered

    def filter_by_region(
        self,
        detections: list[dict[str, Any]],
        frame_height: int,
        min_y_ratio: float = 0.15,
        max_y_ratio: float = 0.85,
    ) -> list[dict[str, Any]]:
        """Filter detections by vertical position (exclude top/bottom regions).

        This helps filter out:
        - Overlay graphics at top of frame
        - Audience at bottom of frame (for some camera angles)

        Args:
            detections: List of detections.
            frame_height: Height of the frame.
            min_y_ratio: Minimum y position as ratio of frame height.
            max_y_ratio: Maximum y position as ratio of frame height.

        Returns:
            Filtered detections.
        """
        min_y = frame_height * min_y_ratio
        max_y = frame_height * max_y_ratio

        filtered = []
        for det in detections:
            bbox = det["bbox"]
            center_y = (bbox[1] + bbox[3]) / 2

            if min_y <= center_y <= max_y:
                filtered.append(det)

        return filtered
