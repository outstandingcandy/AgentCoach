"""Ball detection using YOLO.

Specialized detector for soccer ball with adjustments for:
- Smaller object size (lower confidence threshold)
- Higher resolution for small object detection
- Size and aspect ratio filtering specific to balls

Supports any ultralytics YOLO model (YOLOv8, YOLO11, etc.) via config.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError:
    AutoDetectionModel = None


class BallDetector:
    """Detect soccer ball in video frames using YOLO."""

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
        self.model_path = self.config.get("model_path", None)
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

        # SAHI sliced inference parameters
        self.use_sahi = self.config.get("use_sahi", False)
        self.sahi_slice_size = self.config.get("sahi_slice_size", 640)
        self.sahi_overlap_ratio = self.config.get("sahi_overlap_ratio", 0.2)
        self.sahi_perform_standard_pred = self.config.get("sahi_perform_standard_pred", True)
        self.sahi_postprocess_type = self.config.get("sahi_postprocess_type", "GREEDYNMM")
        self.sahi_postprocess_match_threshold = self.config.get("sahi_postprocess_match_threshold", 0.5)

        self.model = None
        self._sahi_model = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load YOLO model.

        Args:
            model_path: Path to model weights. If None, uses config model_path
                or downloads from ultralytics hub based on model name.
        """
        path = model_path or self.model_path
        if path:
            self.model = YOLO(str(path))
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

        if self.use_sahi:
            return self._detect_sahi(frame)
        return self._detect_standard(frame)

    def _detect_standard(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Standard YOLO detection on full frame."""
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

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """Detect ball in a batch of frames using standard YOLO inference.

        Args:
            frames: List of input frames (BGR).

        Returns:
            List of detection lists, one per input frame.
        """
        if self.model is None:
            self.load_model()
        if not frames:
            return []

        results = self.model.predict(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self.imgsz,
            verbose=False,
            batch=len(frames),
        )

        all_detections: list[list[dict[str, Any]]] = []
        for result in results:
            detections = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())
                    x1, y1, x2, y2 = box
                    detections.append({
                        "bbox": box.tolist(),
                        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
                        "confidence": conf,
                        "class": cls,
                        "class_name": self.model.names[cls],
                    })
            all_detections.append(detections)

        return all_detections

    def _detect_sahi(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """SAHI sliced inference for improved small ball detection."""
        if AutoDetectionModel is None:
            raise ImportError("sahi package is required for SAHI inference. Install with: pip install sahi")

        # Lazily create the SAHI detection model wrapper
        if self._sahi_model is None:
            model_path = self.model_path or f"{self.model_name}.pt"
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="yolov8",
                model_path=str(model_path),
                confidence_threshold=self.confidence_threshold,
                device=self.device,
                image_size=self.imgsz,
            )

        result = get_sliced_prediction(
            image=frame,
            detection_model=self._sahi_model,
            slice_height=self.sahi_slice_size,
            slice_width=self.sahi_slice_size,
            overlap_height_ratio=self.sahi_overlap_ratio,
            overlap_width_ratio=self.sahi_overlap_ratio,
            perform_standard_pred=self.sahi_perform_standard_pred,
            postprocess_type=self.sahi_postprocess_type,
            postprocess_match_threshold=self.sahi_postprocess_match_threshold,
            verbose=0,
        )

        detections = []
        for pred in result.object_prediction_list:
            cls = pred.category.id
            if cls not in self.classes:
                continue

            bbox = pred.bbox.to_xyxy()
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "center": [float(center_x), float(center_y)],
                "confidence": float(pred.score.value),
                "class": cls,
                "class_name": pred.category.name,
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

    def detect_crop(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        crop_size: int = 300,
        enlarge_to: int = 640,
    ) -> list[dict[str, Any]]:
        """Detect ball in a cropped+enlarged region around expected position.

        Crops a region around `center`, resizes it to `enlarge_to` so that
        a tiny ball becomes large enough for reliable detection, then maps
        detections back to full-frame coordinates.

        Args:
            frame: Full input frame (BGR).
            center: Expected ball (cx, cy) in full-frame pixels.
            crop_size: Side length of the square crop in pixels.
            enlarge_to: Resize the crop to this size before detection.

        Returns:
            List of detection dicts in full-frame coordinates.
        """
        if self.model is None:
            self.load_model()

        h, w = frame.shape[:2]
        cx, cy = center
        half = crop_size // 2

        # Compute crop bounds, clamped to frame
        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []

        crop_h, crop_w = crop.shape[:2]
        scale_x = crop_w / enlarge_to
        scale_y = crop_h / enlarge_to

        enlarged = cv2.resize(crop, (enlarge_to, enlarge_to), interpolation=cv2.INTER_LINEAR)

        # Run standard YOLO on the enlarged crop (ball is now big enough)
        results = self.model.predict(
            enlarged,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=enlarge_to,
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

                # Map back to full-frame coordinates
                bx1 = float(box[0]) * scale_x + x1
                by1 = float(box[1]) * scale_y + y1
                bx2 = float(box[2]) * scale_x + x1
                by2 = float(box[3]) * scale_y + y1

                det_cx = (bx1 + bx2) / 2
                det_cy = (by1 + by2) / 2

                detections.append({
                    "bbox": [bx1, by1, bx2, by2],
                    "center": [det_cx, det_cy],
                    "confidence": conf,
                    "class": cls,
                    "class_name": self.model.names[cls],
                })

        return detections

    def prepare_crop(
        self,
        frame: np.ndarray,
        center: tuple[float, float],
        crop_size: int = 300,
        enlarge_to: int = 640,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """Crop and enlarge a region around expected ball position.

        Returns the enlarged image and metadata needed to map detections
        back to full-frame coordinates.  Returns (None, meta) when the
        crop is empty.
        """
        h, w = frame.shape[:2]
        cx, cy = center
        half = crop_size // 2

        x1 = max(0, int(cx - half))
        y1 = max(0, int(cy - half))
        x2 = min(w, int(cx + half))
        y2 = min(h, int(cy + half))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None, {}

        crop_h, crop_w = crop.shape[:2]
        meta = {
            "x1": x1, "y1": y1,
            "scale_x": crop_w / enlarge_to,
            "scale_y": crop_h / enlarge_to,
        }
        enlarged = cv2.resize(crop, (enlarge_to, enlarge_to), interpolation=cv2.INTER_LINEAR)
        return enlarged, meta

    def detect_crop_batch(
        self,
        images: list[np.ndarray],
        metas: list[dict[str, Any]],
        enlarge_to: int = 640,
    ) -> list[list[dict[str, Any]]]:
        """Run batched YOLO inference on pre-cropped enlarged images.

        Args:
            images: List of enlarged crop images (all same size).
            metas: Corresponding metadata from prepare_crop (x1, y1, scale_x, scale_y).
            enlarge_to: Image size (for YOLO imgsz parameter).

        Returns:
            List of detection lists, one per input image, in full-frame coordinates.
        """
        if self.model is None:
            self.load_model()
        if not images:
            return []

        results = self.model.predict(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=enlarge_to,
            verbose=False,
            batch=len(images),
        )

        all_detections: list[list[dict[str, Any]]] = []
        for result, meta in zip(results, metas):
            detections = []
            boxes = result.boxes
            if boxes is not None and len(boxes):
                x1 = meta["x1"]
                y1 = meta["y1"]
                sx = meta["scale_x"]
                sy = meta["scale_y"]
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    bx1 = float(box[0]) * sx + x1
                    by1 = float(box[1]) * sy + y1
                    bx2 = float(box[2]) * sx + x1
                    by2 = float(box[3]) * sy + y1

                    detections.append({
                        "bbox": [bx1, by1, bx2, by2],
                        "center": [(bx1 + bx2) / 2, (by1 + by2) / 2],
                        "confidence": conf,
                        "class": cls,
                        "class_name": self.model.names[cls],
                    })
            all_detections.append(detections)

        return all_detections

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
