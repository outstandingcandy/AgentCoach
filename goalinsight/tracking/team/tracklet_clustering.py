"""Tracklet-based team clustering using KMeans.

Based on SoccerNet sn-gamestate's tracklet_team_clustering_api.py.
"""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from ...interfaces import BaseTeamClassifier


class TrackletTeamClustering(BaseTeamClassifier):
    """Cluster tracks into teams using embeddings.

    Uses KMeans clustering on track-averaged embeddings to separate
    players into two teams. Only 'player' tracks are clustered;
    goalkeepers and referees are handled separately.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize team clustering.

        Args:
            config: Configuration dictionary with optional keys:
                - n_clusters: Number of team clusters (default: 2)
                - random_state: Random seed for reproducibility
                - n_init: Number of KMeans initializations
                - min_samples: Minimum samples needed for clustering
        """
        self.config = config or {}
        self.n_clusters = self.config.get("n_clusters", 2)
        self.random_state = self.config.get("random_state", 42)
        self.n_init = self.config.get("n_init", 10)
        self.min_samples = self.config.get("min_samples", 2)

        self.kmeans = None
        self.cluster_centers_ = None
        self._is_fitted = False

    def fit(
        self,
        track_features: dict[int, np.ndarray],
        track_positions: dict[int, list[float]] | None = None,
        track_roles: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Cluster tracks into teams.

        Args:
            track_features: {track_id: mean_embedding} for each track.
            track_positions: {track_id: [x, y]} - not used in this implementation.
            track_roles: {track_id: role} - only 'player' tracks are clustered.

        Returns:
            Dictionary {track_id: team_label} where team_label is
            "team_A" or "team_B".
        """
        track_roles = track_roles or {}

        # Filter for players only
        player_track_ids = []
        player_embeddings = []

        for track_id, embedding in track_features.items():
            role = track_roles.get(track_id, 'player')

            # Only cluster players
            if role == 'player':
                if embedding is None or len(embedding) == 0:
                    continue
                if np.isnan(embedding).any() or np.isinf(embedding).any():
                    continue

                player_track_ids.append(track_id)
                player_embeddings.append(embedding)

        # Handle edge cases
        if len(player_embeddings) == 0:
            return {}

        if len(player_embeddings) == 1:
            return {player_track_ids[0]: "team_A"}

        if len(player_embeddings) < self.min_samples:
            return {tid: "team_A" for tid in player_track_ids}

        # Stack embeddings into matrix
        embeddings_matrix = np.vstack(player_embeddings)

        # KMeans clustering
        n_clusters = min(self.n_clusters, len(player_embeddings))
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=self.n_init,
        )

        try:
            labels = self.kmeans.fit_predict(embeddings_matrix)
            self.cluster_centers_ = self.kmeans.cluster_centers_
        except Exception as e:
            print(f"KMeans clustering failed: {e}")
            return {tid: "team_A" for tid in player_track_ids}

        # Map labels to team names
        team_clusters = {}
        for track_id, label in zip(player_track_ids, labels):
            team_clusters[track_id] = f"team_{chr(65 + label)}"

        self._is_fitted = True
        return team_clusters

    def predict(
        self,
        feature: np.ndarray,
        position: list[float] | None = None,
    ) -> str:
        """Predict team cluster for a new embedding.

        Args:
            feature: Feature embedding for a single track.
            position: Not used.

        Returns:
            Predicted team label.
        """
        if not self._is_fitted or self.kmeans is None:
            return "unknown"

        label = int(self.kmeans.predict(feature.reshape(1, -1))[0])
        return f"team_{chr(65 + label)}"

    @property
    def is_fitted(self) -> bool:
        """Check if classifier has been fitted."""
        return self._is_fitted

    def get_cluster_balance(self, team_clusters: dict[int, str]) -> dict[str, int]:
        """Get count of tracks per cluster."""
        balance = {"team_A": 0, "team_B": 0}
        for team in team_clusters.values():
            if team in balance:
                balance[team] += 1
        return balance


class IncrementalTeamClustering:
    """Incremental team clustering for online processing."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize incremental clustering."""
        self.config = config or {}
        self.base_clustering = TrackletTeamClustering(config)

        self.track_embeddings: dict[int, list[np.ndarray]] = {}
        self.track_roles: dict[int, str] = {}

    def add_observation(
        self,
        track_id: int,
        embedding: np.ndarray,
        role: str = 'player',
    ) -> None:
        """Add a new embedding observation for a track."""
        if track_id not in self.track_embeddings:
            self.track_embeddings[track_id] = []

        self.track_embeddings[track_id].append(embedding)
        self.track_roles[track_id] = role

    def get_mean_embeddings(self) -> dict[int, np.ndarray]:
        """Get mean embedding for each track."""
        mean_embeddings = {}
        for track_id, embeddings in self.track_embeddings.items():
            if embeddings:
                mean_embeddings[track_id] = np.mean(embeddings, axis=0)
        return mean_embeddings

    def cluster(self) -> dict[int, str]:
        """Perform clustering on all tracks."""
        mean_embeddings = self.get_mean_embeddings()
        return self.base_clustering.fit(mean_embeddings, track_roles=self.track_roles)

    def clear(self) -> None:
        """Clear all track history."""
        self.track_embeddings.clear()
        self.track_roles.clear()
