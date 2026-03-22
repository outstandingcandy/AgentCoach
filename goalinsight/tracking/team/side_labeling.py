"""Team side labeling based on pitch position.

Based on SoccerNet sn-gamestate's tracklet_team_side_labeling_api.py.
"""

from typing import Any

import numpy as np

from ...interfaces import BaseTeamSideLabeler


class TrackletTeamSideLabeling(BaseTeamSideLabeler):
    """Assign team sides (left/right) based on pitch position.

    Teams are labeled as 'left' or 'right' based on their average
    x-coordinate on the pitch.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize team side labeling.

        Args:
            config: Configuration dictionary with optional keys:
                - pitch_center_x: X coordinate of pitch center (default: 0)
                - goalkeeper_distance_threshold: Distance from goal for GK
        """
        self.config = config or {}
        self.pitch_center_x = self.config.get("pitch_center_x", 0.0)
        self.goalkeeper_distance = self.config.get("goalkeeper_distance_threshold", 40.0)

    def label(
        self,
        track_positions: dict[int, list[float]],
        team_clusters: dict[int, int],
        track_roles: dict[int, str] | None = None,
        use_image_coords: bool = False,
        image_center_x: float = 512.0,
    ) -> dict[int, str]:
        """Assign team sides based on positions.

        Args:
            track_positions: {track_id: [x, y]} - mean position per track.
            team_clusters: {track_id: 0 or 1} - cluster assignment from KMeans.
            track_roles: {track_id: role} - optional role per track.
            use_image_coords: If True, interpret positions as image coords.
            image_center_x: Image center x for image coordinate mode.

        Returns:
            Dictionary {track_id: team_side}.
        """
        track_roles = track_roles or {}
        center_x = image_center_x if use_image_coords else self.pitch_center_x

        # Calculate mean x-coordinate per cluster
        cluster_positions: dict[int, list[float]] = {0: [], 1: []}

        for track_id, cluster in team_clusters.items():
            if track_id in track_positions:
                x = track_positions[track_id][0]
                cluster_positions[cluster].append(x)

        # Determine which cluster is 'left' and 'right'
        mean_x_0 = np.mean(cluster_positions[0]) if cluster_positions[0] else 0.0
        mean_x_1 = np.mean(cluster_positions[1]) if cluster_positions[1] else 0.0

        if mean_x_0 > mean_x_1:
            cluster_to_side = {0: 'right', 1: 'left'}
        else:
            cluster_to_side = {0: 'left', 1: 'right'}

        # Assign sides
        team_sides = {}
        all_track_ids = set(team_clusters.keys()) | set(track_positions.keys())

        for track_id in all_track_ids:
            role = track_roles.get(track_id, 'player')

            if role == 'referee':
                team_sides[track_id] = 'referee'

            elif role == 'goalkeeper':
                # Goalkeeper: assign based on their own position
                if track_id in track_positions:
                    x = track_positions[track_id][0]
                    team_sides[track_id] = 'right' if x > center_x else 'left'
                else:
                    team_sides[track_id] = 'unknown'

            elif role == 'ball':
                team_sides[track_id] = 'ball'

            elif track_id in team_clusters:
                team_sides[track_id] = cluster_to_side[team_clusters[track_id]]

            else:
                team_sides[track_id] = 'unknown'

        return team_sides

    def label_from_positions_only(
        self,
        track_positions: dict[int, list[float]],
        track_roles: dict[int, str] | None = None,
    ) -> dict[int, str]:
        """Assign team sides based on position only (no clustering)."""
        track_roles = track_roles or {}
        team_sides = {}

        for track_id, position in track_positions.items():
            role = track_roles.get(track_id, 'player')

            if role == 'referee':
                team_sides[track_id] = 'referee'
            elif role == 'ball':
                team_sides[track_id] = 'ball'
            else:
                x = position[0]
                team_sides[track_id] = 'right' if x > self.pitch_center_x else 'left'

        return team_sides

    def get_side_balance(self, team_sides: dict[int, str]) -> dict[str, int]:
        """Get count of tracks per side."""
        balance = {'left': 0, 'right': 0, 'referee': 0, 'unknown': 0, 'ball': 0}
        for side in team_sides.values():
            if side in balance:
                balance[side] += 1
        return balance
