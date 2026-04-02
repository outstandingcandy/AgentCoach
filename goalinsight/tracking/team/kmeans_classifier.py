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
        track_bbox_heights: dict[int, float] | None = None,
        track_frame_counts: dict[int, int] | None = None,
        track_mean_saturations: dict[int, float] | None = None,
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

        # Use 2 clusters for two teams; referee detection is position-based
        n_clusters = min(2, len(track_ids))

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(features_norm)
        self.team_centers = self.kmeans.cluster_centers_

        # Map clusters to teams by size (larger cluster = team_A)
        cluster_counts = {}
        for label in labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: -x[1])

        cluster_to_team = {}
        if len(sorted_clusters) >= 1:
            cluster_to_team[sorted_clusters[0][0]] = "team_A"
        if len(sorted_clusters) >= 2:
            cluster_to_team[sorted_clusters[1][0]] = "team_B"

        # Detect referees by low saturation (achromatic = black/white/grey kit)
        # This is more reliable than color-distance outlier detection because
        # black referee kits have near-zero saturation in HS space, making them
        # indistinguishable from dark team jerseys by histogram distance alone.
        referee_set = set()
        min_referee_bbox_h = 40.0
        min_referee_frames = 15
        sideline_y_threshold = 23.0
        max_color_refs = 2

        if track_mean_saturations:
            # Compute team saturation baseline: median saturation of all tracks
            all_sats = list(track_mean_saturations.values())
            median_sat = np.median(all_sats)
            # Referee threshold: significantly below median (achromatic/dark kit)
            sat_threshold = median_sat * 0.7

            sat_candidates = []
            for tid in track_ids:
                if tid not in track_mean_saturations:
                    continue
                sat = track_mean_saturations[tid]
                if sat >= sat_threshold:
                    continue
                # Apply same filters: min bbox height, min frames, exclude sideline
                if track_bbox_heights and tid in track_bbox_heights:
                    if track_bbox_heights[tid] < min_referee_bbox_h:
                        continue
                if track_frame_counts and tid in track_frame_counts:
                    if track_frame_counts[tid] < min_referee_frames:
                        continue
                if track_positions and tid in track_positions:
                    if abs(track_positions[tid][1]) > sideline_y_threshold:
                        continue
                sat_candidates.append((tid, sat))

            sat_candidates.sort(key=lambda x: x[1])  # lowest saturation first
            referee_set = {tid for tid, _ in sat_candidates[:max_color_refs]}

            # Debug
            print(f"  Saturation: median={median_sat:.1f}, threshold={sat_threshold:.1f}")
            for tid, sat in sat_candidates:
                marker = " ← REFEREE" if tid in referee_set else ""
                print(f"    T{tid}: sat={sat:.1f}{marker}")

        # Fallback: color-distance outlier detection (if saturation didn't find any)
        if not referee_set:
            distances = self.kmeans.transform(features_norm)  # (N, 2) distances
            min_dists = distances.min(axis=1)
            dist_threshold = np.percentile(min_dists, 85)
            median_dist = np.median(min_dists)
            outlier_candidates = []
            for i, tid in enumerate(track_ids):
                if min_dists[i] > dist_threshold and min_dists[i] > median_dist * 1.5:
                    if track_bbox_heights and tid in track_bbox_heights:
                        if track_bbox_heights[tid] < min_referee_bbox_h:
                            continue
                    if track_frame_counts and tid in track_frame_counts:
                        if track_frame_counts[tid] < min_referee_frames:
                            continue
                    if track_positions and tid in track_positions:
                        if abs(track_positions[tid][1]) > sideline_y_threshold:
                            continue
                    outlier_candidates.append((i, tid, min_dists[i]))

            outlier_candidates.sort(key=lambda x: -x[2])
            referee_set = {tid for _, tid, _ in outlier_candidates[:max_color_refs]}
            if referee_set:
                print(f"  Color outlier referees: {referee_set}")

        assignments = {}
        for i, tid in enumerate(track_ids):
            if tid in referee_set:
                assignments[tid] = "referee"
            else:
                assignments[tid] = cluster_to_team.get(labels[i], "unknown")

        if referee_set:
            print(f"  Referees detected: {referee_set}")

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
        """Check if position is in goalkeeper area."""
        if position is None:
            return False

        x, y = position
        in_left = (
            x < -self.half_length + self.goal_area_depth
            and abs(y) < self.half_width * 0.6
        )
        in_right = (
            x > self.half_length - self.goal_area_depth
            and abs(y) < self.half_width * 0.6
        )
        return in_left or in_right

    def is_linesman(self, position: list[float], margin: float = 2.0) -> bool:
        """Check if position is outside pitch boundary (assistant referee)."""
        if position is None:
            return False
        return abs(position[1]) > self.half_width - margin

    def classify_role(
        self,
        position: list[float],
        team: str | None = None,
    ) -> str:
        """Classify role based on position and team."""
        if team == "referee":
            return "referee"
        if self.is_goalkeeper(position):
            return "goalkeeper"
        return "player"

    def refine_roles(
        self,
        team_assignments: dict[int, str],
        mean_positions: dict[int, list[float]],
        all_positions: dict[int, list[list[float]]] | None = None,
    ) -> tuple[dict[int, str], set[int]]:
        """Post-pass role refinement using mean track positions.

        1. Linesman detection: tracks frequently outside sideline -> referee
        2. Goalkeeper detection: per team, track deepest in goal area -> goalkeeper

        Args:
            team_assignments: Dict of track_id -> team label (modified in-place).
            mean_positions: Dict of track_id -> [x, y] mean pitch position.
            all_positions: Dict of track_id -> list of [x, y] positions (all frames).

        Returns:
            (updated team_assignments, set of goalkeeper track_ids)
        """
        # 1. Linesman detection: use frequency of being outside sideline
        linesman_count = 0
        sideline_threshold = self.half_width  # actual sideline, not inside it
        min_outside_ratio = 0.50  # at least 50% of frames outside sideline

        if all_positions:
            # Max plausible coordinate: positions beyond this are calibration errors
            max_plausible = self.half_length + 10.0
            for tid, positions in all_positions.items():
                if len(positions) < 10:
                    continue
                # Filter out calibration artifacts (physically impossible positions)
                valid = [p for p in positions
                         if abs(p[0]) < max_plausible and abs(p[1]) < max_plausible]
                if len(valid) < 10:
                    continue
                outside_count = sum(1 for p in valid if abs(p[1]) > sideline_threshold)
                ratio = outside_count / len(valid)
                if ratio >= min_outside_ratio:
                    team_assignments[tid] = "referee"
                    linesman_count += 1
                    print(f"  Linesman: T{tid} (outside sideline {ratio:.0%} of frames)")
        else:
            # Fallback to mean position
            for tid, pos in mean_positions.items():
                if self.is_linesman(pos):
                    team_assignments[tid] = "referee"
                    linesman_count += 1

        if linesman_count:
            print(f"  Linesmen detected: {linesman_count}")

        # 2. Goalkeeper detection: find the player closest to each goal line
        #    across ALL teams (not within a single team), then assign the
        #    correct team based on which goal they defend.
        #
        #    This fixes the common problem where KMeans misclassifies a
        #    goalkeeper because their jersey color differs from teammates.
        goalkeeper_tracks = set()

        # Collect all non-referee tracks with positions
        field_tids = [
            t for t, tm in team_assignments.items()
            if tm in ("team_A", "team_B") and t in mean_positions
        ]
        if len(field_tids) >= 4:
            # Determine which team defends which side:
            # the team with lower mean |x| on each side is defending there
            team_mean_x: dict[str, float] = {}
            for team in ["team_A", "team_B"]:
                xs = [
                    mean_positions[t][0]
                    for t in field_tids
                    if team_assignments[t] == team
                ]
                if xs:
                    team_mean_x[team] = float(np.mean(xs))

            # For each goal (left and right), find the nearest player
            for goal_sign, goal_label in [(-1, "left"), (1, "right")]:
                goal_x = goal_sign * self.half_length
                # Sort ALL field players by distance to this goal line
                by_dist = sorted(
                    field_tids,
                    key=lambda t: abs(mean_positions[t][0] - goal_x),
                )
                if len(by_dist) < 2:
                    continue
                candidate = by_dist[0]
                second = by_dist[1]
                cand_dist = abs(mean_positions[candidate][0] - goal_x)
                second_dist = abs(mean_positions[second][0] - goal_x)

                # Goalkeeper must be significantly closer to goal line
                # than anyone else, and within the penalty area depth
                if (second_dist - cand_dist > 5.0
                        and cand_dist < self.goal_area_depth):
                    goalkeeper_tracks.add(candidate)

                    # Correct team assignment: the goalkeeper defends this
                    # goal, so they belong to the team whose field players
                    # are on THIS side (center of mass closer to this goal)
                    if len(team_mean_x) == 2:
                        defending_team = min(
                            team_mean_x,
                            key=lambda tm: abs(team_mean_x[tm] - goal_x),
                        )
                        old_team = team_assignments.get(candidate, "unknown")
                        if old_team != defending_team:
                            team_assignments[candidate] = defending_team
                            print(
                                f"  Goalkeeper team corrected: T{candidate} "
                                f"{old_team} → {defending_team} "
                                f"(defends {goal_label} goal)"
                            )
                    print(
                        f"  Goalkeeper: T{candidate} "
                        f"({team_assignments[candidate]}, "
                        f"dist_to_{goal_label}_goal={cand_dist:.1f}m, "
                        f"gap={second_dist - cand_dist:.1f}m)"
                    )

        return team_assignments, goalkeeper_tracks


# Backwards compatibility alias
TeamClassifier = KMeansTeamClassifier
