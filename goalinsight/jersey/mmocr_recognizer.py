"""Jersey number recognition using MMOCR (DBNet+SAR).

Implements BaseJerseyRecognizer interface.
"""

import logging
from collections import Counter
from typing import Any

import numpy as np

from ..interfaces import BaseJerseyRecognizer

log = logging.getLogger(__name__)


class MMOCRJerseyRecognizer(BaseJerseyRecognizer):
    """Jersey number recognition using MMOCR (DBNet+SAR).

    Implements a two-step OCR pipeline:
    1. DBNet text detection - locates text regions
    2. SAR text recognition - recognizes text
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize jersey number recognizer.

        Args:
            config: Configuration with optional keys:
                - device: Device for inference
                - det_model: DBNet model name
                - rec_model: SAR model name
                - min_confidence: Minimum confidence threshold
                - batch_size: Batch size for inference
        """
        self.config = config or {}
        self.device = self.config.get('device', None)
        self.det_model = self.config.get('det_model', 'dbnet_resnet18_fpnc_1200e_icdar2015')
        self.rec_model = self.config.get('rec_model', 'SAR')
        self.min_confidence = self.config.get('min_confidence', 0.3)
        self.batch_size = self.config.get('batch_size', 1)

        self.text_detector = None
        self.text_recognizer = None
        self._initialized = False

    def _init_models(self) -> None:
        """Lazily initialize MMOCR models."""
        if self._initialized:
            return

        try:
            import torch
            from mmocr.apis import TextDetInferencer, TextRecInferencer

            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            log.info(f"Initializing MMOCR models on {self.device}...")
            log.info(f"  Text Detection: {self.det_model}")
            log.info(f"  Text Recognition: {self.rec_model}")

            self.text_detector = TextDetInferencer(
                self.det_model,
                device=self.device
            )

            self.text_recognizer = TextRecInferencer(
                self.rec_model,
                device=self.device
            )

            self._initialized = True
            log.info("MMOCR models initialized successfully")

        except ImportError as e:
            log.error(f"MMOCR not installed: {e}")
            raise

    def _extract_numbers(self, text: str) -> str | None:
        """Extract numeric characters from text."""
        number = ''.join(c for c in text if c.isdigit())
        return number if number else None

    def _crop_text_region(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Crop text region from image based on detected polygon."""
        try:
            from mmocr.utils import bbox2poly, crop_img, poly2bbox

            quad = bbox2poly(poly2bbox(polygon)).tolist()
            crop = crop_img(image, quad)
            return crop
        except Exception:
            if len(polygon) >= 4:
                x_coords = polygon[::2] if len(polygon.shape) == 1 else polygon[:, 0]
                y_coords = polygon[1::2] if len(polygon.shape) == 1 else polygon[:, 1]
                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    return image[y1:y2, x1:x2]
            return np.zeros((1, 1, 3), dtype=np.uint8)

    def recognize(self, crop: np.ndarray) -> tuple[int | None, float]:
        """Recognize jersey number from a single crop.

        Args:
            crop: Player image crop (BGR format).

        Returns:
            Tuple of (jersey_number, confidence).
        """
        self._init_models()

        if self.text_detector is None or self.text_recognizer is None:
            return None, 0.0

        if crop is None or crop.size == 0:
            return None, 0.0

        try:
            import cv2

            h, w = crop.shape[:2]

            # Upscale small crops
            min_width = self.config.get('min_ocr_width', 128)
            if w < min_width:
                scale = min_width / w
                crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

            # Enhance contrast using CLAHE
            lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

            # Step 1: Text detection with DBNet
            det_result = self.text_detector(
                enhanced,
                return_datasamples=True,
                batch_size=1,
                progress_bar=False
            )

            if not det_result or 'predictions' not in det_result:
                return None, 0.0

            det_predictions = det_result['predictions']
            if not det_predictions:
                return None, 0.0

            det_pred = det_predictions[0]
            if not hasattr(det_pred, 'pred_instances'):
                return None, 0.0

            polygons = det_pred.pred_instances.get('polygons', [])
            if len(polygons) == 0:
                return None, 0.0

            # Step 2: Text recognition with SAR
            rec_inputs = []
            for polygon in polygons:
                text_crop = self._crop_text_region(enhanced, polygon)
                if text_crop.shape[0] > 0 and text_crop.shape[1] > 0:
                    rec_inputs.append(text_crop)

            if not rec_inputs:
                return None, 0.0

            rec_result = self.text_recognizer(
                rec_inputs,
                return_datasamples=True,
                batch_size=self.batch_size,
                progress_bar=False
            )

            if not rec_result or 'predictions' not in rec_result:
                return None, 0.0

            # Find best numeric result
            best_number = None
            best_confidence = 0.0

            for rec_pred in rec_result['predictions']:
                rec_dict = self.text_recognizer.pred2dict(rec_pred)
                text = rec_dict.get('text', '')
                scores = rec_dict.get('scores', 0.0)
                if isinstance(scores, (list, np.ndarray)):
                    confidence = float(np.mean(scores)) if len(scores) > 0 else 0.0
                else:
                    confidence = float(scores)

                if confidence < self.min_confidence:
                    continue

                number_str = self._extract_numbers(text)
                if number_str:
                    number_str = number_str[:2]
                    try:
                        number = int(number_str)
                        if 1 <= number <= 99 and confidence > best_confidence:
                            best_number = number
                            best_confidence = confidence
                    except ValueError:
                        continue

            return best_number, float(best_confidence)

        except Exception as e:
            log.warning(f"Error in jersey number OCR: {e}")
            return None, 0.0

    def recognize_batch(
        self,
        crops: list[np.ndarray],
    ) -> list[tuple[int | None, float]]:
        """Recognize jersey numbers from multiple crops."""
        results = []
        for crop in crops:
            results.append(self.recognize(crop))
        return results


class TrackJerseyNumberVoting:
    """Consolidate jersey numbers across a track using voting."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize voting module."""
        self.config = config or {}
        self.min_votes = self.config.get('min_votes', 2)
        self.min_confidence = self.config.get('min_confidence', 0.4)

    def vote(
        self,
        track_detections: dict[int, list[dict[str, Any]]]
    ) -> dict[int, dict[str, Any]]:
        """Vote on jersey numbers for each track."""
        results = {}

        for track_id, detections in track_detections.items():
            valid_detections = [
                d for d in detections
                if d.get('number') is not None
            ]

            if not valid_detections:
                results[track_id] = {
                    'number': None,
                    'confidence': 0.0,
                    'votes': 0
                }
                continue

            number_votes = Counter()
            number_confidences = {}

            for det in valid_detections:
                num = det['number']
                conf = det.get('confidence', 0.5)
                number_votes[num] += 1
                if num not in number_confidences:
                    number_confidences[num] = []
                number_confidences[num].append(conf)

            most_common = number_votes.most_common(1)[0]
            best_number, vote_count = most_common
            avg_confidence = np.mean(number_confidences[best_number])

            if vote_count >= self.min_votes and avg_confidence >= self.min_confidence:
                results[track_id] = {
                    'number': best_number,
                    'confidence': float(avg_confidence),
                    'votes': vote_count
                }
            else:
                results[track_id] = {
                    'number': None,
                    'confidence': float(avg_confidence) if vote_count > 0 else 0.0,
                    'votes': vote_count
                }

        return results


# Backwards compatibility alias
JerseyNumberRecognizer = MMOCRJerseyRecognizer
