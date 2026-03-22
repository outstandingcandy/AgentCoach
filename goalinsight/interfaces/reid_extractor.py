"""Abstract base class for ReID feature extraction backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseReIDExtractor(ABC):
    """Abstract base class for ReID feature extraction.

    Implementations:
    - OSNet: torchreid OSNet model (512-dim embeddings)
    - PRTReID: BPBreID with HRNet backbone (256-dim embeddings + role prediction)
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize ReID extractor.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def load_model(self, model_path: str | None = None) -> None:
        """Load ReID model.

        Args:
            model_path: Optional path to model weights.
        """
        pass

    @abstractmethod
    def extract(self, crops: list[np.ndarray]) -> np.ndarray:
        """Extract ReID features from image crops.

        Args:
            crops: List of person image crops (BGR format from OpenCV).

        Returns:
            Array of feature vectors, shape (N, feature_dim).
            Features should be L2-normalized.
        """
        pass

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Return feature dimension for this backend.

        Returns:
            Feature vector dimension (e.g., 512 for OSNet, 256 for PRTReID).
        """
        pass

    def extract_single(self, crop: np.ndarray) -> np.ndarray:
        """Extract ReID feature from a single crop.

        Args:
            crop: Person image crop (BGR format).

        Returns:
            Feature vector of shape (feature_dim,).
        """
        features = self.extract([crop])
        return features[0] if len(features) > 0 else np.zeros(self.feature_dim)

    def extract_with_roles(
        self,
        crops: list[np.ndarray],
    ) -> dict[str, Any]:
        """Extract features with optional role predictions.

        This method provides a unified interface for backends that
        also predict roles (like PRTReID). For backends without role
        prediction, returns default 'player' roles.

        Args:
            crops: List of person image crops.

        Returns:
            Dictionary with:
            - 'embeddings': np.ndarray of shape (N, feature_dim)
            - 'roles': list[str] - predicted roles
            - 'role_confidences': np.ndarray of shape (N,)
        """
        embeddings = self.extract(crops)
        n = len(crops)
        return {
            'embeddings': embeddings,
            'roles': ['player'] * n,
            'role_confidences': np.ones(n),
        }

    def compute_similarity(
        self,
        query_features: np.ndarray,
        gallery_features: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between query and gallery features.

        Args:
            query_features: Query feature vectors, shape (Q, D).
            gallery_features: Gallery feature vectors, shape (G, D).

        Returns:
            Similarity matrix of shape (Q, G).
        """
        # Ensure features are normalized
        query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)

        return np.dot(query_norm, gallery_norm.T)

    @property
    def name(self) -> str:
        """Return backend name for logging."""
        return self.__class__.__name__
