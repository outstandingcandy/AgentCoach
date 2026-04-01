"""Field line detection for soccer pitch registration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class LineDetector:
    """Detect soccer field lines.

    Supports multiple backends:
    - "hough": Classical Hough transform (default)
    - "pnlcalib": PnLCalib HRNet-based line extremity detection
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize line detector.

        Args:
            config: Detection configuration.
                - backend: "hough" (default) or "pnlcalib"
                - hough_threshold, min_line_length, max_line_gap: Hough params
                - pnlcalib: PnLCalib-specific config dict
                    - weights: "SV_lines", "WC14_lines", or "TSWC_lines"
                    - confidence_threshold: Detection threshold
        """
        self.config = config or {}
        self.backend = self.config.get("backend", "hough")
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Hough parameters (for hough backend)
        self.hough_threshold = self.config.get("hough_threshold", 50)
        self.min_line_length = self.config.get("min_line_length", 100)
        self.max_line_gap = self.config.get("max_line_gap", 10)

        # PnLCalib parameters
        if self.backend == "pnlcalib":
            pnlcalib_config = self.config.get("pnlcalib", {})
            self.confidence_threshold = pnlcalib_config.get("confidence_threshold", 0.7867)
            self._pnlcalib_weights = pnlcalib_config.get("weights", "SV_lines")
        else:
            self.confidence_threshold = self.config.get("confidence_threshold", 0.5)

        # Color thresholds for white line detection (in HSV) - used by hough backend
        self.white_lower = np.array([0, 0, 180])
        self.white_upper = np.array([180, 50, 255])

        # Input size for neural network
        self.input_size = (540, 960)  # Height x Width

        self.model = None
        self._using_fallback = True

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load line detection model.

        Args:
            model_path: Path to model weights. If None and backend is "pnlcalib",
                       will auto-download weights.
        """
        if self.backend == "pnlcalib":
            self._load_pnlcalib_model(model_path)
        else:
            # Hough backend doesn't need a model
            self._using_fallback = False

    def _load_pnlcalib_model(self, model_path: str | Path | None = None) -> None:
        """Load PnLCalib HRNet line model.

        Args:
            model_path: Path to model weights. If None, auto-downloads.
        """
        from .pnlcalib import HRNetLineModel, PnLCalibWeightDownloader

        # Get weights path
        if model_path is None:
            downloader = PnLCalibWeightDownloader()
            model_path = downloader.get_line_weights(
                variant=self._pnlcalib_weights.replace("_lines", ""),
                download=True,
            )

        logger.info(f"Loading PnLCalib line model from {model_path}")

        # Create and load model
        self.model = HRNetLineModel()
        self.model.load_pretrained(str(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self._using_fallback = False

        logger.info("PnLCalib line model loaded successfully")

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for line detection.

        Args:
            frame: Input frame (BGR format).

        Returns:
            Preprocessed tensor.
        """
        # Resize to expected input size
        resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        # Note: PnLCalib does NOT use ImageNet normalization, just [0, 1] scaling
        if self.backend != "pnlcalib":
            # Only apply ImageNet normalization for Hough/other backends
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect lines in a frame.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of line dictionaries with endpoints and properties.
        """
        if self.backend == "pnlcalib" and self.model is not None:
            return self._detect_pnlcalib(frame)

        return self._detect_hough(frame)

    def _detect_pnlcalib(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect lines using PnLCalib model.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of line dictionaries.
        """
        from .pnlcalib import get_lines_from_heatmap_maxpool
        from .pnlcalib.hrnet_line import HRNetLineModel

        h, w = frame.shape[:2]

        # Preprocess
        tensor = self.preprocess(frame).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            heatmaps = self.model(tensor)  # (1, num_outputs, H', W')

        # PnLCalib uses softmax, exclude the last channel (background)
        # Heatmaps are at 1/2 resolution
        heatmap_for_extraction = heatmaps[:, :-1, :, :]  # Exclude background

        # Get scaling factors (heatmap -> original frame)
        hm_h, hm_w = heatmaps.shape[2], heatmaps.shape[3]
        scale_x = w / hm_w
        scale_y = h / hm_h

        # Extract line extremities using maxpool-based detection
        # Use scale=1 to get raw heatmap coordinates, then scale manually
        lines = get_lines_from_heatmap_maxpool(
            heatmap_for_extraction,
            scale=1,  # Get raw heatmap coordinates
            threshold=self.confidence_threshold,
        )

        # Apply accurate float scaling to original frame coordinates
        for line in lines:
            line["x1"] = line["x1"] * scale_x
            line["y1"] = line["y1"] * scale_y
            line["x2"] = line["x2"] * scale_x
            line["y2"] = line["y2"] * scale_y
            # Recalculate length with scaled coordinates
            line["length"] = float(np.sqrt(
                (line["x2"] - line["x1"])**2 + (line["y2"] - line["y1"])**2
            ))

        # Add angle and classify
        for line in lines:
            dx = line["x2"] - line["x1"]
            dy = line["y2"] - line["y1"]
            line["angle"] = float(np.arctan2(dy, dx) * 180 / np.pi)
            line["class_name"] = HRNetLineModel.get_line_class_name(line["id"])

        return lines

    def detect_batch(
        self,
        frames: list[np.ndarray],
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """Detect lines in multiple frames using batched GPU inference.

        Args:
            frames: List of input frames (BGR format).
            batch_size: Max frames per GPU forward pass.

        Returns:
            List of line lists, one per input frame.
        """
        if self.backend != "pnlcalib" or self.model is None:
            return [self.detect(f) for f in frames]

        all_results = []
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            chunk_results = self._detect_batch_chunk_pnlcalib(chunk)
            all_results.extend(chunk_results)
        return all_results

    def _detect_batch_chunk_pnlcalib(
        self,
        frames: list[np.ndarray],
    ) -> list[list[dict[str, Any]]]:
        """Run batched PnLCalib line inference on a chunk of frames."""
        from .pnlcalib import get_lines_from_heatmap_maxpool
        from .pnlcalib.hrnet_line import HRNetLineModel

        tensors = [self.preprocess(f) for f in frames]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(batch)  # (B, C, H', W')

        results = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            hm = heatmaps[i : i + 1]

            hm_for_extraction = hm[:, :-1, :, :]
            hm_h, hm_w = hm.shape[2], hm.shape[3]
            scale_x = w / hm_w
            scale_y = h / hm_h

            lines = get_lines_from_heatmap_maxpool(
                hm_for_extraction,
                scale=1,
                threshold=self.confidence_threshold,
            )

            for line in lines:
                line["x1"] = line["x1"] * scale_x
                line["y1"] = line["y1"] * scale_y
                line["x2"] = line["x2"] * scale_x
                line["y2"] = line["y2"] * scale_y
                line["length"] = float(np.sqrt(
                    (line["x2"] - line["x1"])**2 + (line["y2"] - line["y1"])**2
                ))

            for line in lines:
                dx = line["x2"] - line["x1"]
                dy = line["y2"] - line["y1"]
                line["angle"] = float(np.arctan2(dy, dx) * 180 / np.pi)
                line["class_name"] = HRNetLineModel.get_line_class_name(line["id"])

            results.append(lines)

        return results

    def _detect_hough(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect lines using Hough transform.

        Uses direct edge detection on the original image for better results
        on wide-angle cameras where white line masking may fail.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of line dictionaries.
        """
        h, w = frame.shape[:2]

        # Direct edge detection on original image (better for wide-angle cameras)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # Filter out sky region (top 28% of frame)
        edges[:int(h * 0.28), :] = 0

        # Detect lines using Hough transform with tuned parameters
        lines_raw = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,  # Higher threshold to reduce noise
            minLineLength=150,  # Only detect long lines
            maxLineGap=30,
        )

        if lines_raw is None:
            return []

        # Process and filter lines
        lines = []
        for i, line in enumerate(lines_raw):
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            orientation = self._classify_line_orientation(angle)
            lines.append({
                "id": i,
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "length": float(length),
                "angle": float(angle),
                "confidence": min(1.0, length / 300),  # Confidence based on length
                "orientation": orientation,
            })

        # Merge similar lines
        lines = self._merge_similar_lines(lines)

        # Sort by length (descending) and keep top lines
        lines.sort(key=lambda x: x["length"], reverse=True)

        return lines

    def _classify_line_orientation(self, angle: float) -> str:
        """Classify line orientation based on angle.

        Args:
            angle: Line angle in degrees (-180 to 180).

        Returns:
            Orientation category: "horizontal", "vertical", or "diagonal".
        """
        # Normalize angle to [0, 90] range
        abs_angle = abs(angle)
        if abs_angle > 90:
            abs_angle = 180 - abs_angle

        if abs_angle < 20:
            return "horizontal"  # Side lines, penalty area top/bottom
        elif abs_angle > 70:
            return "vertical"    # Middle line, penalty area sides
        else:
            return "diagonal"    # Center circle tangents, other

    def _create_line_mask(self, frame: np.ndarray) -> np.ndarray:
        """Create mask for white field lines.

        Args:
            frame: Input frame (BGR format).

        Returns:
            Binary mask image.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create white mask
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)

        # Optional: Also detect in grayscale for robust detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Combine masks
        mask = cv2.bitwise_or(white_mask, bright_mask)

        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return mask

    def _merge_similar_lines(
        self,
        lines: list[dict[str, Any]],
        angle_threshold: float = 5.0,
        distance_threshold: float = 20.0,
    ) -> list[dict[str, Any]]:
        """Merge lines that are similar.

        Args:
            lines: List of detected lines.
            angle_threshold: Maximum angle difference for merging.
            distance_threshold: Maximum distance between lines for merging.

        Returns:
            Merged lines.
        """
        if len(lines) <= 1:
            return lines

        # Group lines by angle
        merged = []
        used = set()

        for i, line1 in enumerate(lines):
            if i in used:
                continue

            group = [line1]
            used.add(i)

            for j, line2 in enumerate(lines):
                if j in used:
                    continue

                # Check angle similarity
                angle_diff = abs(line1["angle"] - line2["angle"])
                if angle_diff > 90:
                    angle_diff = 180 - angle_diff

                if angle_diff > angle_threshold:
                    continue

                # Check distance
                dist = self._line_distance(line1, line2)
                if dist > distance_threshold:
                    continue

                group.append(line2)
                used.add(j)

            # Merge group into single line
            merged_line = self._merge_line_group(group)
            merged.append(merged_line)

        return merged

    def _line_distance(
        self,
        line1: dict[str, Any],
        line2: dict[str, Any],
    ) -> float:
        """Compute distance between two lines.

        Args:
            line1: First line.
            line2: Second line.

        Returns:
            Distance between line midpoints.
        """
        mid1_x = (line1["x1"] + line1["x2"]) / 2
        mid1_y = (line1["y1"] + line1["y2"]) / 2
        mid2_x = (line2["x1"] + line2["x2"]) / 2
        mid2_y = (line2["y1"] + line2["y2"]) / 2

        return np.sqrt((mid2_x - mid1_x) ** 2 + (mid2_y - mid1_y) ** 2)

    def _merge_line_group(self, group: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge a group of similar lines.

        Args:
            group: List of similar lines.

        Returns:
            Single merged line.
        """
        if len(group) == 1:
            return group[0]

        # Find extreme points
        points = []
        for line in group:
            points.append((line["x1"], line["y1"]))
            points.append((line["x2"], line["y2"]))

        points = np.array(points)

        # Fit line to all points using PCA
        mean = np.mean(points, axis=0)
        centered = points - mean
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        direction = eigenvectors[:, np.argmax(eigenvalues)]

        # Project points onto line
        projections = np.dot(centered, direction)
        min_proj = np.min(projections)
        max_proj = np.max(projections)

        # Compute endpoints
        p1 = mean + min_proj * direction
        p2 = mean + max_proj * direction

        length = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0]) * 180 / np.pi

        orientation = self._classify_line_orientation(angle)
        return {
            "id": group[0]["id"],
            "x1": float(p1[0]),
            "y1": float(p1[1]),
            "x2": float(p2[0]),
            "y2": float(p2[1]),
            "length": float(length),
            "angle": float(angle),
            "orientation": orientation,
            "confidence": max(line.get("confidence", 0) for line in group),
        }

    def classify_lines(
        self,
        lines: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Classify lines into categories based on orientation.

        Args:
            lines: List of detected lines.

        Returns:
            Dictionary with "horizontal", "vertical", and "other" lines.
        """
        horizontal = []
        vertical = []
        other = []

        for line in lines:
            angle = abs(line["angle"])
            if angle > 90:
                angle = 180 - angle

            if angle < 15:
                horizontal.append(line)
            elif angle > 75:
                vertical.append(line)
            else:
                other.append(line)

        return {
            "horizontal": horizontal,
            "vertical": vertical,
            "other": other,
        }

    def find_intersections(
        self,
        lines: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Find intersections between lines.

        Args:
            lines: List of detected lines.

        Returns:
            List of intersection points.
        """
        intersections = []
        intersection_id = 0

        for i, line1 in enumerate(lines):
            for j, line2 in enumerate(lines[i + 1:], start=i + 1):
                # Check if lines are not parallel
                angle_diff = abs(line1["angle"] - line2["angle"])
                if angle_diff < 10 or abs(angle_diff - 180) < 10:
                    continue

                # Compute intersection
                point = self._line_intersection(
                    (line1["x1"], line1["y1"], line1["x2"], line1["y2"]),
                    (line2["x1"], line2["y1"], line2["x2"], line2["y2"]),
                )

                if point is not None:
                    intersections.append({
                        "id": intersection_id,
                        "x": float(point[0]),
                        "y": float(point[1]),
                        "line1_id": line1["id"],
                        "line2_id": line2["id"],
                    })
                    intersection_id += 1

        return intersections

    def _line_intersection(
        self,
        line1: tuple[float, float, float, float],
        line2: tuple[float, float, float, float],
    ) -> tuple[float, float] | None:
        """Compute intersection point of two lines.

        Args:
            line1: First line as (x1, y1, x2, y2).
            line2: Second line as (x1, y1, x2, y2).

        Returns:
            Intersection point (x, y) or None if parallel.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom

        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)


class FieldMask:
    """Generate field region mask to filter detections."""

    def __init__(self):
        """Initialize field mask generator."""
        pass

    def generate_mask(
        self,
        frame: np.ndarray,
        green_threshold: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Generate mask for the field region.

        Args:
            frame: Input frame (BGR format).
            green_threshold: Optional (lower, upper) HSV bounds for green.

        Returns:
            Binary mask where field is white.
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Default green range
        if green_threshold is None:
            green_lower = np.array([35, 40, 40])
            green_upper = np.array([85, 255, 255])
        else:
            green_lower, green_upper = green_threshold

        # Create green mask
        mask = cv2.inRange(hsv, green_lower, green_upper)

        # Morphological operations
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Find largest contour (should be the field)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            mask = np.zeros_like(mask)
            cv2.fillPoly(mask, [largest], 255)

        return mask

    def apply_mask(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Apply mask to frame.

        Args:
            frame: Input frame.
            mask: Binary mask.

        Returns:
            Masked frame.
        """
        return cv2.bitwise_and(frame, frame, mask=mask)
