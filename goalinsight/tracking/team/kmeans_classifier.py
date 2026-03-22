"""Team classification using KMeans clustering on ReID features.

Implements BaseTeamClassifier interface.
"""

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ...interfaces import BaseTeamClassifier


class KMeansTeamClassifier(BaseTeamClassifier):
    """Classify players into teams using KMeans on ReID features.

    Uses 3-cluster KMeans: two teams + referees/others.
    Assigns based on cluster sizes (referees are smallest cluster).
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize team classifier.

        Args:
            config: Configuration dict with keys:
                - n_teams: Number of teams (default: 2)
                - use_position: Whether to use pitch position (default: True)
                - position_weight: Weight for position features (default: 0.1)
                - min_samples_per_team: Min samples before classification (default: 5)
        """
        config = config or {}
        self.n_teams = config.get("n_teams", 2)
        self.use_position = config.get("use_position", True)
        self.position_weight = config.get("position_weight", 0.1)
        self.min_samples = config.get("min_samples_per_team", 5)

        self.kmeans = None
        self.scaler = StandardScaler()
        self.team_centers = None
        self._is_fitted = False

    def initialize_with_reid(self, embeddings: np.ndarray) -> dict[int, str]:
        """Initialize classifier with ReID embeddings array.

        Convenience method that wraps fit() for use with raw embedding arrays.

        Args:
            embeddings: ReID embeddings array of shape (N, D).

        Returns:
            Dict of synthetic track_id -> team label.
        """
        if len(embeddings) < self.min_samples * 2:
            return {}

        # Create synthetic track IDs for the embeddings
        track_features = {i: emb for i, emb in enumerate(embeddings)}
        return self.fit(track_features)

    def fit(
        self,
        track_features: dict[int, np.ndarray],
        track_positions: dict[int, list[float]] | None = None,
    ) -> dict[int, str]:
        """Fit classifier and assign teams to tracks.

        Uses 3-cluster KMeans: two teams + referees/others.
        Then assigns based on cluster sizes (referees are smallest cluster).

        Args:
            track_features: Dict of track_id -> mean ReID feature vector.
            track_positions: Dict of track_id -> [x, y] pitch position (optional).

        Returns:
            Dict of track_id -> team label ("team_A", "team_B", "referee", etc.)
        """
        if len(track_features) < self.min_samples * 2:
            return {}

        track_ids = list(track_features.keys())
        features = np.array([track_features[tid] for tid in track_ids])

        # Normalize features (L2 normalization for cosine similarity)
        features_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)

        # Optionally add position features (with low weight)
        if self.use_position and track_positions:
            positions = []
            valid_ids = []
            for tid in track_ids:
                if tid in track_positions and track_positions[tid] is not None:
                    positions.append(track_positions[tid])
                    valid_ids.append(tid)

            if len(positions) >= self.min_samples * 2:
                positions = np.array(positions)
                positions_norm = self.scaler.fit_transform(positions)

                idx_map = {tid: i for i, tid in enumerate(track_ids)}
                valid_features = features_norm[[idx_map[tid] for tid in valid_ids]]

                combined = np.hstack([
                    valid_features * (1 - self.position_weight),
                    positions_norm * self.position_weight
                ])

                track_ids = valid_ids
                features_norm = combined

        # Use 3 clusters: team_A, team_B, and referees/others
        n_clusters = 3
        if len(track_ids) < n_clusters:
            n_clusters = max(2, len(track_ids))

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features_norm)
        self.team_centers = self.kmeans.cluster_centers_

        # Count cluster sizes
        cluster_counts = {}
        for label in labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1

        # Sort clusters by size (largest first)
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])

        # Map: largest two clusters are teams, smallest is referee/other
        cluster_to_team = {}
        if len(sorted_clusters) >= 2:
            cluster_to_team[sorted_clusters[0][0]] = "team_A"
            cluster_to_team[sorted_clusters[1][0]] = "team_B"
        if len(sorted_clusters) >= 3:
            cluster_to_team[sorted_clusters[2][0]] = "referee"

        # Assign teams
        assignments = {}
        for i, tid in enumerate(track_ids):
            cluster = labels[i]
            assignments[tid] = cluster_to_team.get(cluster, "unknown")

        self._is_fitted = True
        return assignments

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
            Team label.
        """
        if not self._is_fitted or self.kmeans is None:
            return "unknown"

        # Normalize feature
        feature_norm = feature / (np.linalg.norm(feature) + 1e-8)

        # Optionally add position
        if self.use_position and position is not None:
            position_norm = self.scaler.transform([position])
            combined = np.hstack([
                feature_norm.reshape(1, -1) * (1 - self.position_weight),
                position_norm * self.position_weight
            ])
        else:
            combined = feature_norm.reshape(1, -1)

        # Check distance to team centers
        distances = np.linalg.norm(self.team_centers - combined, axis=1)
        min_dist = np.min(distances)

        # If too far from any team center, might be referee
        if min_dist > 1.0:
            return "referee"

        team_idx = np.argmin(distances)
        return f"team_{chr(65 + team_idx)}"

    @property
    def is_fitted(self) -> bool:
        """Check if classifier has been fitted."""
        return self._is_fitted

    def update_track_team(
        self,
        track_id: int,
        feature: np.ndarray,
        position: list[float] | None = None,
        current_assignments: dict[int, str] | None = None,
    ) -> str:
        """Update team assignment for a track.

        Uses existing assignments for context.
        """
        if current_assignments and track_id in current_assignments:
            return current_assignments[track_id]

        return self.predict(feature, position)


class GoalkeeperDetector:
    """Detect goalkeepers based on position on pitch."""

    def __init__(
        self,
        pitch_length: float = 105.0,
        pitch_width: float = 68.0,
        goal_area_depth: float = 16.5,
    ):
        """Initialize goalkeeper detector.

        Args:
            pitch_length: Pitch length in meters.
            pitch_width: Pitch width in meters.
            goal_area_depth: Depth of penalty area.
        """
        self.half_length = pitch_length / 2
        self.half_width = pitch_width / 2
        self.goal_area_depth = goal_area_depth

    def is_goalkeeper(self, position: list[float]) -> bool:
        """Check if position is in goalkeeper area.

        Args:
            position: [x, y] pitch position.

        Returns:
            True if likely goalkeeper position.
        """
        if position is None:
            return False

        x, y = position

        # Check if in left or right goal area
        in_left_goal_area = (
            x < -self.half_length + self.goal_area_depth and
            abs(y) < 20  # Roughly within goal area width
        )
        in_right_goal_area = (
            x > self.half_length - self.goal_area_depth and
            abs(y) < 20
        )

        return in_left_goal_area or in_right_goal_area

    def classify_role(
        self,
        position: list[float],
        team: str | None = None,
    ) -> str:
        """Classify role based on position.

        Args:
            position: [x, y] pitch position.
            team: Team assignment.

        Returns:
            Role: "goalkeeper", "player", or "referee"
        """
        if team == "referee":
            return "referee"

        if self.is_goalkeeper(position):
            return "goalkeeper"

        return "player"


# Backwards compatibility alias
TeamClassifier = KMeansTeamClassifier
