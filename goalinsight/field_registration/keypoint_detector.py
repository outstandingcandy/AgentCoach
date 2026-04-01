"""Field keypoint detection for soccer pitch registration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except ImportError:
    timm = None

logger = logging.getLogger(__name__)


class KeypointDetector:
    """Detect soccer field keypoints using deep learning.

    Supports multiple backends:
    - "resnet50": Original ResNet50-based detector (115 keypoints, SoccerNet-GSR)
    - "pnlcalib": PnLCalib HRNet-based detector (58 keypoints)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize keypoint detector.

        Note: The fallback Shi-Tomasi corner detection is NOT suitable for
        soccer field registration as it detects ALL corners in the image
        (including audience, ads, etc.). For proper field registration,
        use a model trained on SoccerNet-GSR or similar dataset.

        Args:
            config: Detection configuration.
                - backend: "resnet50" (default) or "pnlcalib"
                - model: Backbone for resnet50 backend
                - pnlcalib: PnLCalib-specific config dict
                    - weights: "SV_kp", "WC14_kp", or "TSWC_kp"
                    - confidence_threshold: Detection threshold
        """
        self.config = config or {}
        self.backend = self.config.get("backend", "resnet50")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Backend-specific configuration
        if self.backend == "pnlcalib":
            pnlcalib_config = self.config.get("pnlcalib", {})
            self.num_keypoints = 58  # PnLCalib uses 58 keypoints
            self.confidence_threshold = pnlcalib_config.get("confidence_threshold", 0.3434)
            self._pnlcalib_weights = pnlcalib_config.get("weights", "SV_kp")
        else:
            self.backbone = self.config.get("model", "resnet50")
            self.num_keypoints = self.config.get("num_keypoints", 115)

        # Input size - same for both backends
        self.input_size = (540, 960)  # Height x Width

        self.model = None
        self._keypoint_mapper = None

        # Track whether we're using fallback (for UI/visualization hints)
        self._using_fallback = True

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load keypoint detection model.

        Args:
            model_path: Path to model weights. If None and backend is "pnlcalib",
                       will auto-download weights. For "resnet50", will use fallback.
        """
        if self.backend == "pnlcalib":
            self._load_pnlcalib_model(model_path)
        else:
            self._load_resnet_model(model_path)

    def _load_pnlcalib_model(self, model_path: str | Path | None = None) -> None:
        """Load PnLCalib HRNet model.

        Args:
            model_path: Path to model weights. If None, checks config then auto-downloads.
        """
        from .pnlcalib import HRNetKeypointModel, PnLCalibWeightDownloader, KeypointMapper

        # Initialize keypoint mapper for 58<->115 conversion
        self._keypoint_mapper = KeypointMapper()

        # Priority: argument > config path > auto-download
        if model_path is None:
            pnlcalib_config = self.config.get("pnlcalib", {})
            model_path = pnlcalib_config.get("model_path")

        if model_path is None:
            # Fallback to auto-download
            downloader = PnLCalibWeightDownloader()
            model_path = downloader.get_keypoint_weights(
                variant=self._pnlcalib_weights.replace("_kp", ""),
                download=True,
            )

        logger.info(f"Loading PnLCalib keypoint model from {model_path}")

        # Create and load model
        self.model = HRNetKeypointModel(num_keypoints=self.num_keypoints)
        self.model.load_pretrained(str(model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        self._using_fallback = False

        logger.info("PnLCalib keypoint model loaded successfully")

    def _load_resnet_model(self, model_path: str | Path | None = None) -> None:
        """Load ResNet-based keypoint model.

        Args:
            model_path: Path to model weights. If None, will use fallback.
        """
        if model_path is None:
            # No model path provided - will use fallback
            self._using_fallback = True
            return

        if timm is None:
            raise ImportError("timm package is required. Install with: pip install timm")

        self.model = KeypointDetectionModel(
            backbone=self.backbone,
            num_keypoints=self.num_keypoints,
        )

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()
        self._using_fallback = False

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for keypoint detection.

        Args:
            frame: Input frame (BGR format).

        Returns:
            Preprocessed tensor.
        """
        # Resize to expected input size (960x540 for PnLCalib)
        resized = cv2.resize(frame, (self.input_size[1], self.input_size[0]))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Convert to tensor and normalize to [0, 1]
        tensor = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0

        # Note: PnLCalib does NOT use ImageNet normalization, just [0, 1] scaling
        if self.backend != "pnlcalib":
            # Only apply ImageNet normalization for ResNet backend
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        return tensor

    def detect(
        self,
        frame: np.ndarray,
        convert_to_soccernet: bool = True,
    ) -> list[dict[str, Any]]:
        """Detect keypoints in a frame.

        Args:
            frame: Input frame (BGR format).
            convert_to_soccernet: If True and using pnlcalib backend, convert
                                 keypoint IDs to SoccerNet-GSR format.

        Returns:
            List of keypoint dictionaries with pixel coordinates and confidence.
        """
        if self.model is None:
            # Use fallback detection if no model loaded
            return self._detect_fallback(frame)

        h, w = frame.shape[:2]

        # Preprocess
        tensor = self.preprocess(frame).unsqueeze(0).to(self.device)

        # Predict
        with torch.no_grad():
            heatmaps = self.model(tensor)  # (1, num_keypoints, H', W')

        # Extract keypoints from heatmaps
        if self.backend == "pnlcalib":
            from .pnlcalib import get_keypoints_from_heatmap_maxpool

            # PnLCalib uses softmax so we exclude the last channel (background)
            # The model outputs 58 channels, first 57 are keypoint heatmaps
            heatmap_for_extraction = heatmaps[:, :-1, :, :]  # Exclude background

            # Get heatmap spatial dimensions for scaling
            hm_h, hm_w = heatmaps.shape[2], heatmaps.shape[3]
            # Compute accurate scale factors (original frame -> heatmap)
            scale_x = w / hm_w
            scale_y = h / hm_h

            # Use scale=1 to get raw heatmap coordinates, then scale manually
            keypoints = get_keypoints_from_heatmap_maxpool(
                heatmap_for_extraction,
                scale=1,  # Get raw heatmap coordinates
                max_keypoints=1,
                threshold=self.confidence_threshold,
            )

            # Apply accurate float scaling to original frame coordinates
            for kp in keypoints:
                kp["x"] = kp["x"] * scale_x
                kp["y"] = kp["y"] * scale_y

            # Convert to SoccerNet-GSR format if requested
            if convert_to_soccernet and self._keypoint_mapper is not None:
                keypoints = self._keypoint_mapper.pnlcalib_to_soccernet(keypoints)
        else:
            keypoints = self._extract_keypoints_from_heatmaps(
                heatmaps[0].cpu().numpy(),
                original_size=(w, h),
            )

        return keypoints

    def _extract_keypoints_from_heatmaps(
        self,
        heatmaps: np.ndarray,
        original_size: tuple[int, int],
    ) -> list[dict[str, Any]]:
        """Extract keypoint coordinates from heatmaps.

        Args:
            heatmaps: Heatmap array of shape (num_keypoints, H, W).
            original_size: Original frame size (width, height).

        Returns:
            List of keypoint dictionaries.
        """
        keypoints = []
        orig_w, orig_h = original_size
        hm_h, hm_w = heatmaps.shape[1:]

        for i in range(len(heatmaps)):
            hm = heatmaps[i]

            # Find maximum
            max_val = np.max(hm)
            if max_val < self.confidence_threshold:
                continue

            # Get coordinates of maximum
            y, x = np.unravel_index(np.argmax(hm), hm.shape)

            # Scale to original size
            orig_x = x * orig_w / hm_w
            orig_y = y * orig_h / hm_h

            keypoints.append({
                "id": i,
                "x": float(orig_x),
                "y": float(orig_y),
                "confidence": float(max_val),
            })

        return keypoints

    def _detect_fallback(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Fallback keypoint detection using classical CV.

        WARNING: This is NOT suitable for soccer field registration!
        Shi-Tomasi corner detection finds ALL corners in the image,
        including audience, advertisements, and other non-field elements.

        For proper field registration, use a model trained on SoccerNet-GSR
        that detects the 115 standard soccer field keypoints.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of keypoint dictionaries with 'is_fallback' flag set.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect corners using Shi-Tomasi (generic corner detection)
        # Note: This detects ANY corners, not just field keypoints!
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
        )

        keypoints = []
        if corners is not None:
            for i, corner in enumerate(corners):
                x, y = corner.ravel()
                keypoints.append({
                    "id": i,
                    "x": float(x),
                    "y": float(y),
                    "confidence": 0.5,
                    "is_fallback": True,  # Flag indicating fallback detection
                })

        return keypoints

    @property
    def using_fallback(self) -> bool:
        """Check if detector is using fallback (generic corner) detection.

        Returns:
            True if using fallback detection (no proper model loaded).
        """
        return self._using_fallback

    @property
    def keypoint_mapper(self):
        """Get keypoint mapper for format conversion.

        Returns:
            KeypointMapper instance if using pnlcalib backend, else None.
        """
        return self._keypoint_mapper

    def detect_batch(
        self,
        frames: list[np.ndarray],
        convert_to_soccernet: bool = True,
        batch_size: int = 8,
    ) -> list[list[dict[str, Any]]]:
        """Detect keypoints in multiple frames using batched GPU inference.

        Args:
            frames: List of input frames (BGR format).
            convert_to_soccernet: If True and using pnlcalib backend, convert
                                 keypoint IDs to SoccerNet-GSR format.
            batch_size: Max frames per GPU forward pass.

        Returns:
            List of keypoint lists, one per input frame.
        """
        if self.model is None:
            return [self._detect_fallback(f) for f in frames]

        all_results = []
        for start in range(0, len(frames), batch_size):
            chunk = frames[start : start + batch_size]
            chunk_results = self._detect_batch_chunk(chunk, convert_to_soccernet)
            all_results.extend(chunk_results)
        return all_results

    def _detect_batch_chunk(
        self,
        frames: list[np.ndarray],
        convert_to_soccernet: bool,
    ) -> list[list[dict[str, Any]]]:
        """Run batched inference on a chunk of frames."""
        # Preprocess and stack
        tensors = [self.preprocess(f) for f in frames]
        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model(batch)  # (B, num_keypoints, H', W')

        results = []
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            hm = heatmaps[i : i + 1]  # Keep batch dim: (1, C, H', W')

            if self.backend == "pnlcalib":
                from .pnlcalib import get_keypoints_from_heatmap_maxpool

                hm_for_extraction = hm[:, :-1, :, :]
                hm_h, hm_w = hm.shape[2], hm.shape[3]
                scale_x = w / hm_w
                scale_y = h / hm_h

                keypoints = get_keypoints_from_heatmap_maxpool(
                    hm_for_extraction,
                    scale=1,
                    max_keypoints=1,
                    threshold=self.confidence_threshold,
                )
                for kp in keypoints:
                    kp["x"] = kp["x"] * scale_x
                    kp["y"] = kp["y"] * scale_y

                if convert_to_soccernet and self._keypoint_mapper is not None:
                    keypoints = self._keypoint_mapper.pnlcalib_to_soccernet(keypoints)
            else:
                keypoints = self._extract_keypoints_from_heatmaps(
                    hm[0].cpu().numpy(), original_size=(w, h),
                )
            results.append(keypoints)

        return results

    def detect_pnlcalib_raw(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Detect keypoints and return in raw PnLCalib format (58 keypoints).

        Only works when using pnlcalib backend.

        Args:
            frame: Input frame (BGR format).

        Returns:
            List of keypoints in PnLCalib format (IDs 0-57).

        Raises:
            RuntimeError: If not using pnlcalib backend.
        """
        if self.backend != "pnlcalib":
            raise RuntimeError("detect_pnlcalib_raw() requires pnlcalib backend")

        return self.detect(frame, convert_to_soccernet=False)

    def match_to_template(
        self,
        detected_keypoints: list[dict[str, Any]],
        template_keypoints: list[list],
    ) -> list[tuple[dict, list]]:
        """Match detected keypoints to template keypoints.

        Args:
            detected_keypoints: List of detected keypoint dicts.
            template_keypoints: List of template keypoints [id, name, x, y].

        Returns:
            List of (detected, template) matched pairs.
        """
        # This is a simplified matching based on keypoint ID
        # In practice, this would use more sophisticated matching
        matches = []

        detected_by_id = {kp["id"]: kp for kp in detected_keypoints}

        for template in template_keypoints:
            kp_id = template[0]
            if kp_id in detected_by_id:
                matches.append((detected_by_id[kp_id], template))

        return matches


class KeypointDetectionModel(nn.Module):
    """Neural network for keypoint heatmap prediction."""

    def __init__(
        self,
        backbone: str = "resnet50",
        num_keypoints: int = 115,
        pretrained: bool = True,
    ):
        """Initialize keypoint detection model.

        Args:
            backbone: Backbone network name.
            num_keypoints: Number of keypoints to detect.
            pretrained: Whether to use pretrained backbone.
        """
        super().__init__()

        if timm is None:
            raise ImportError("timm package is required")

        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4],  # Last feature map
        )

        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 540, 960)
            features = self.backbone(dummy)
            feat_dim = features[0].shape[1]

        # Decoder head
        self.decoder = nn.Sequential(
            nn.Conv2d(feat_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, num_keypoints, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Heatmap tensor of shape (B, num_keypoints, H', W').
        """
        features = self.backbone(x)[0]
        heatmaps = self.decoder(features)
        heatmaps = torch.sigmoid(heatmaps)
        return heatmaps


class KeypointMatcher:
    """Match detected keypoints between frames for tracking."""

    def __init__(self, max_distance: float = 50.0):
        """Initialize matcher.

        Args:
            max_distance: Maximum matching distance in pixels.
        """
        self.max_distance = max_distance

    def match(
        self,
        keypoints1: list[dict[str, Any]],
        keypoints2: list[dict[str, Any]],
    ) -> list[tuple[int, int]]:
        """Match keypoints between two frames.

        Args:
            keypoints1: Keypoints from frame 1.
            keypoints2: Keypoints from frame 2.

        Returns:
            List of (idx1, idx2) matched pairs.
        """
        if not keypoints1 or not keypoints2:
            return []

        # Compute distance matrix
        n1, n2 = len(keypoints1), len(keypoints2)
        distances = np.zeros((n1, n2))

        for i, kp1 in enumerate(keypoints1):
            for j, kp2 in enumerate(keypoints2):
                dx = kp1["x"] - kp2["x"]
                dy = kp1["y"] - kp2["y"]
                distances[i, j] = np.sqrt(dx * dx + dy * dy)

        # Greedy matching
        matches = []
        used1 = set()
        used2 = set()

        while True:
            # Find minimum distance
            min_dist = float("inf")
            min_i, min_j = -1, -1

            for i in range(n1):
                if i in used1:
                    continue
                for j in range(n2):
                    if j in used2:
                        continue
                    if distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        min_i, min_j = i, j

            if min_dist > self.max_distance or min_i < 0:
                break

            matches.append((min_i, min_j))
            used1.add(min_i)
            used2.add(min_j)

        return matches
