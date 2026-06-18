"""Unified YOLO detector for both players (class 0) and ball (class 32)."""

from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


PERSON_CLASS_ID = 0
BALL_CLASS_ID = 32


class UnifiedDetector:
    """Single YOLO pass detecting both persons and sports balls."""

    def __init__(self, config: dict[str, Any] | None = None):
        if YOLO is None:
            raise ImportError("ultralytics package is required. Install with: pip install ultralytics")

        self.config = config or {}
        self.model_name = self.config.get("model", "yolov8x")
        self.model_path = self.config.get("model_path", None)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.3)
        self.iou_threshold = self.config.get("iou_threshold", 0.45)
        self.classes = self.config.get("classes", [PERSON_CLASS_ID, BALL_CLASS_ID])
        # ``imgsz`` accepts a positive int OR ``0`` / None / "native" —
        # the latter trio means "use the source frame's longer side
        # rounded up to a multiple of 32". Native is preferred for
        # small-ball recall (1920 down-samples a 47×47-px ball to ~23
        # px and YOLO confidence falls below threshold). Cost: ~4×
        # slower at 4K vs 1920.
        cfg_imgsz = self.config.get("imgsz", 0)
        if cfg_imgsz in (None, 0, "native", ""):
            self.imgsz: int | None = None
        else:
            self.imgsz = int(cfg_imgsz)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        path = model_path or self.model_path
        if path:
            self.model = YOLO(str(path))
        else:
            self.model = YOLO(f"{self.model_name}.pt")
        self.model.to(self.device)

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict[str, Any]]]:
        """Run batched inference returning all detections.

        Ball detections (class 32) include a ``center`` field to match
        the :class:`BallDetector` output schema.  Player detections
        (class 0) match the :class:`PlayerDetector` schema.
        """
        if self.model is None:
            self.load_model()

        results = self.model.predict(
            frames,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
            imgsz=self._resolve_imgsz(frames[0]),
            verbose=False,
        )

        all_detections: list[list[dict[str, Any]]] = []
        for result in results:
            frame_detections: list[dict[str, Any]] = []
            boxes = result.boxes
            if boxes is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i].cpu().numpy())
                    cls = int(boxes.cls[i].cpu().numpy())

                    det: dict[str, Any] = {
                        "bbox": box.tolist(),
                        "confidence": conf,
                        "class": cls,
                        "class_name": self.model.names[cls],
                    }

                    if cls == BALL_CLASS_ID:
                        x1, y1, x2, y2 = box
                        det["center"] = [float((x1 + x2) / 2), float((y1 + y2) / 2)]

                    frame_detections.append(det)
            all_detections.append(frame_detections)

        return all_detections

    def _resolve_imgsz(self, frame: np.ndarray) -> int:
        """Return YOLO ``imgsz`` for this frame: configured int, or the
        frame's longer side rounded up to a multiple of 32 (stride)."""
        if self.imgsz is not None:
            return self.imgsz
        h, w = frame.shape[:2]
        long_edge = max(int(h), int(w))
        return ((long_edge + 31) // 32) * 32

    @staticmethod
    def split_by_class(
        detections: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split detections into (players, balls) by class id."""
        players = [d for d in detections if d["class"] == PERSON_CLASS_ID]
        balls = [d for d in detections if d["class"] == BALL_CLASS_ID]
        return players, balls
