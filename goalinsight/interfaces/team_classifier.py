"""Abstract base class for team classification backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseTeamClassifier(ABC):
    """Abstract base class for team classification.

    Implementations:
    - KMeans: Simple KMeans clustering on ReID features
    - Tracklet: Tracklet-based clustering with side labeling
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize team classifier.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def fit(
        self,
        track_features: dict[int, np.ndarray],
        track_positions: dict[int, list[float]] | None = None,
    ) -> dict[int, str]:
        """Fit classifier and assign teams to tracks.

        Args:
            track_features: Dict of track_id -> mean ReID feature vector.
            track_positions: Dict of track_id -> [x, y] pitch position (optional).

        Returns:
            Dict of track_id -> team label.
            Common labels: "team_A", "team_B", "referee", "unknown"
        """
        pass

    @abstractmethod
    def predict(
        self,
        feature: np.ndarray,
        position: list[float] | None = None,
    ) -> str:
        """Predict team for a new track.

        Args:
            feature: ReID feature vector.
            position: Optional pitch position [x, y].

        Returns:
            Team label string.
        """
        pass

    @property
    def is_fitted(self) -> bool:
        """Check if classifier has been fitted."""
        return False

    @property
    def name(self) -> str:
        """Return backend name for logging."""
        return self.__class__.__name__


class BaseTeamSideLabeler(ABC):
    """Abstract base class for team side labeling.

    Assigns 'left' or 'right' labels to teams based on pitch position.
    """

    @abstractmethod
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize side labeler.

        Args:
            config: Backend-specific configuration.
        """
        pass

    @abstractmethod
    def label(
        self,
        track_positions: dict[int, list[float]],
        team_clusters: dict[int, int],
        track_roles: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Assign team sides based on positions.

        Args:
            track_positions: {track_id: [x, y]} - mean position per track.
            team_clusters: {track_id: 0 or 1} - cluster assignment.
            track_roles: {track_id: role} - optional role per track.

        Returns:
            Dictionary {track_id: team_side} where team_side is:
            - 'left': Team on the left side
            - 'right': Team on the right side
            - 'referee': For referee tracks
            - 'unknown': For unassigned tracks
        """
        pass
