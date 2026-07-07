"""ReID feature extraction using OSNet (torchreid).

Implements BaseReIDExtractor interface.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...interfaces import BaseReIDExtractor

try:
    import torchreid
except ImportError:
    torchreid = None

logger = logging.getLogger(__name__)


def _unwrap_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Pull a flat tensor state_dict out of a torchreid-style checkpoint.

    torchreid training saves under "state_dict"; some forks use "model".
    DataParallel adds a "module." prefix that we strip.
    """
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    return {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in checkpoint.items()
    }


class OSNetExtractor(BaseReIDExtractor):
    """Extract ReID features using OSNet from torchreid.

    Outputs 512-dimensional L2-normalized feature vectors.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize OSNet extractor.

        Args:
            config: Configuration with optional keys:
                - model: Model name (default: "osnet_x1_0")
                - feature_dim: Feature dimension (default: 512)
                - batch_size: Batch size for inference (default: 32)
                - device: Device for inference (default: "cuda" if available)
        """
        self.config = config or {}
        self.model_name = self.config.get("model", "osnet_x1_0")
        self._feature_dim = self.config.get("feature_dim", 512)
        self.batch_size = self.config.get("batch_size", 32)
        self.device = self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Image preprocessing parameters
        self.input_size = (256, 128)  # Height x Width for ReID
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.model = None

    def load_model(self, model_path: str | Path | None = None) -> None:
        """Load OSNet model.

        Args:
            model_path: Path to model weights. If None, downloads pretrained.
        """
        if torchreid is None:
            raise ImportError(
                "torchreid package is required. "
                "Install with: pip install torchreid"
            )

        # Build model
        self.model = torchreid.models.build_model(
            name=self.model_name,
            num_classes=1000,  # Placeholder, not used for feature extraction
            pretrained=True if model_path is None else False,
        )

        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = _unwrap_state_dict(checkpoint)
            # Classifier-head shape depends on the training dataset's ID count
            # and is unused for feature extraction. Drop any tensor whose shape
            # disagrees with the freshly-built model.
            model_shapes = {k: v.shape for k, v in self.model.state_dict().items()}
            filtered, shape_mismatched = {}, []
            for k, v in state_dict.items():
                if k in model_shapes and model_shapes[k] != v.shape:
                    shape_mismatched.append(k)
                else:
                    filtered[k] = v
            result = self.model.load_state_dict(filtered, strict=False)
            logger.info(
                "Loaded OSNet weights from %s: matched=%d, missing=%d, "
                "unexpected=%d, shape_mismatched=%d",
                model_path,
                len(filtered) - len(result.unexpected_keys),
                len(result.missing_keys),
                len(result.unexpected_keys),
                len(shape_mismatched),
            )
            non_classifier_missing = [
                k for k in result.missing_keys if not k.startswith("classifier")
            ]
            if non_classifier_missing:
                logger.warning(
                    "OSNet checkpoint missing %d backbone keys: %s",
                    len(non_classifier_missing),
                    non_classifier_missing[:5],
                )

        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for OSNet model.

        Args:
            image: Input image (BGR format from OpenCV).

        Returns:
            Preprocessed tensor.
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1]

        # Convert to PIL and resize
        pil_image = Image.fromarray(image)
        pil_image = pil_image.resize((self.input_size[1], self.input_size[0]))

        # Convert to tensor and normalize
        img_tensor = torch.from_numpy(np.array(pil_image)).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0

        # Normalize
        for c in range(3):
            img_tensor[c] = (img_tensor[c] - self.mean[c]) / self.std[c]

        return img_tensor

    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract ReID features from image crops.

        Args:
            crops: List of person image crops (BGR format).

        Returns:
            Array of feature vectors, shape (N, feature_dim).
        """
        if self.model is None:
            self.load_model()

        if not crops:
            return np.array([]).reshape(0, self._feature_dim)

        # Preprocess all crops
        tensors = [self.preprocess(crop) for crop in crops]
        batch = torch.stack(tensors).to(self.device)

        # Extract features in batches
        all_features = []
        with torch.no_grad():
            for i in range(0, len(batch), self.batch_size):
                batch_slice = batch[i:i + self.batch_size]
                features = self.model(batch_slice)

                # Normalize features
                features = F.normalize(features, p=2, dim=1)
                all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0) if all_features else np.array([]).reshape(0, self._feature_dim)

    @property
    def feature_dim(self) -> int:
        """Return feature dimension (512 for OSNet)."""
        return self._feature_dim


class ReIDGallery:
    """Maintain a gallery of ReID features for identity matching."""

    def __init__(self, feature_dim: int = 512):
        """Initialize ReID gallery.

        Args:
            feature_dim: Dimension of feature vectors.
        """
        self.feature_dim = feature_dim
        self.features: dict[int, list[np.ndarray]] = {}  # track_id -> list of features
        self.metadata: dict[int, dict[str, Any]] = {}  # track_id -> metadata

    def add_feature(
        self,
        track_id: int,
        feature: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a feature to the gallery.

        Args:
            track_id: Track identifier.
            feature: Feature vector.
            metadata: Optional metadata (jersey number, team, etc.).
        """
        if track_id not in self.features:
            self.features[track_id] = []
            self.metadata[track_id] = metadata or {}

        self.features[track_id].append(feature)

        # Update metadata if provided
        if metadata:
            self.metadata[track_id].update(metadata)

    def get_mean_feature(self, track_id: int) -> np.ndarray | None:
        """Get mean feature for a track.

        Args:
            track_id: Track identifier.

        Returns:
            Mean feature vector or None.
        """
        if track_id not in self.features or not self.features[track_id]:
            return None
        return np.mean(self.features[track_id], axis=0)

    def get_all_mean_features(self) -> tuple[list[int], np.ndarray]:
        """Get all mean features in the gallery.

        Returns:
            Tuple of (track_ids, feature_matrix).
        """
        track_ids = []
        features = []

        for track_id in self.features:
            mean_feat = self.get_mean_feature(track_id)
            if mean_feat is not None:
                track_ids.append(track_id)
                features.append(mean_feat)

        if features:
            return track_ids, np.stack(features)
        return [], np.array([])

    def find_similar_track(
        self,
        query_feature: np.ndarray,
        threshold: float = 0.8,
        exclude_ids: list[int] | None = None,
    ) -> tuple[int, float] | None:
        """Find the most similar track in the gallery.

        Args:
            query_feature: Query feature vector.
            threshold: Minimum similarity threshold.
            exclude_ids: Track IDs to exclude from search.

        Returns:
            Tuple of (track_id, similarity) or None if no match.
        """
        exclude_ids = exclude_ids or []
        track_ids, gallery_features = self.get_all_mean_features()

        if len(gallery_features) == 0:
            return None

        # Filter excluded IDs
        valid_mask = [tid not in exclude_ids for tid in track_ids]
        if not any(valid_mask):
            return None

        valid_ids = [tid for tid, valid in zip(track_ids, valid_mask) if valid]
        valid_features = gallery_features[valid_mask]

        # Compute similarity
        query_norm = query_feature / (np.linalg.norm(query_feature) + 1e-8)
        gallery_norm = valid_features / (np.linalg.norm(valid_features, axis=1, keepdims=True) + 1e-8)
        similarities = np.dot(gallery_norm, query_norm)

        best_idx = np.argmax(similarities)
        best_sim = similarities[best_idx]

        if best_sim >= threshold:
            return valid_ids[best_idx], float(best_sim)

        return None

    def clear(self) -> None:
        """Clear the gallery."""
        self.features.clear()
        self.metadata.clear()


# Backwards compatibility alias
ReIDExtractor = OSNetExtractor
