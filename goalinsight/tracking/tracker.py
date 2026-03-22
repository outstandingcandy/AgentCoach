"""Multi-object tracking using StrongSORT."""

from typing import Any

import numpy as np

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None


class PlayerTracker:
    """Track players across video frames using StrongSORT-like tracker."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize player tracker.

        Args:
            config: Tracking configuration.
        """
        if DeepSort is None:
            raise ImportError(
                "deep_sort_realtime package is required. "
                "Install with: pip install deep-sort-realtime"
            )

        self.config = config or {}
        self.max_age = self.config.get("max_age", 30)
        self.min_hits = self.config.get("min_hits", 3)
        self.iou_threshold = self.config.get("iou_threshold", 0.3)

        self.tracker = None
        self._initialize_tracker()

    def _initialize_tracker(self) -> None:
        """Initialize the DeepSort tracker."""
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            max_iou_distance=1 - self.iou_threshold,
            max_cosine_distance=0.3,
            embedder="mobilenet",
            embedder_gpu=True,
        )

    def reset(self) -> None:
        """Reset tracker state for new video."""
        self._initialize_tracker()

    def update(
        self,
        detections: list[dict[str, Any]],
        frame: np.ndarray,
        embeddings: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Update tracker with new detections.

        Args:
            detections: List of detections with bbox and confidence.
            frame: Current frame for appearance features.
            embeddings: Optional pre-computed ReID embeddings.

        Returns:
            List of tracked objects with track_id.
        """
        if not detections:
            # Update tracker with empty detections
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert detections to DeepSort format: ([x1, y1, w, h], confidence, class)
        det_list = []
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            conf = det.get("confidence", 1.0)
            cls = det.get("class_name", "person")

            det_list.append(([x1, y1, w, h], conf, cls))

        # Update tracker
        if embeddings is not None:
            tracks = self.tracker.update_tracks(
                det_list,
                frame=frame,
                embeds=embeddings,
            )
        else:
            tracks = self.tracker.update_tracks(det_list, frame=frame)

        # Convert tracks to output format
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]

            tracked_objects.append({
                "track_id": track_id,
                "bbox": ltrb.tolist() if isinstance(ltrb, np.ndarray) else list(ltrb),
                "confidence": track.det_conf if track.det_conf else 1.0,
                "time_since_update": track.time_since_update,
                "hits": track.hits,
            })

        return tracked_objects

    def get_active_track_ids(self) -> list[int]:
        """Get list of currently active track IDs.

        Returns:
            List of active track IDs.
        """
        if self.tracker is None:
            return []

        return [
            track.track_id
            for track in self.tracker.tracks
            if track.is_confirmed()
        ]

    def get_track_history(self, track_id: int) -> list[list[float]] | None:
        """Get bounding box history for a specific track.

        Args:
            track_id: Track identifier.

        Returns:
            List of bounding boxes or None if track not found.
        """
        if self.tracker is None:
            return None

        for track in self.tracker.tracks:
            if track.track_id == track_id:
                # DeepSort doesn't store full history by default
                # Return current position only
                return [track.to_ltrb().tolist()]

        return None


class TrackState:
    """Container for track state information."""

    def __init__(self, track_id: int):
        """Initialize track state.

        Args:
            track_id: Unique track identifier.
        """
        self.track_id = track_id
        self.bbox_history: list[list[float]] = []
        self.reid_embeddings: list[np.ndarray] = []
        self.jersey_numbers: list[int | None] = []
        self.team_labels: list[str | None] = []
        self.roles: list[str] = []
        self.frame_ids: list[int] = []

    def add_observation(
        self,
        frame_id: int,
        bbox: list[float],
        reid_embedding: np.ndarray | None = None,
        jersey_number: int | None = None,
        team: str | None = None,
        role: str = "player",
    ) -> None:
        """Add a new observation to the track.

        Args:
            frame_id: Frame index.
            bbox: Bounding box [x1, y1, x2, y2].
            reid_embedding: ReID feature vector.
            jersey_number: Detected jersey number.
            team: Team classification.
            role: Player role.
        """
        self.frame_ids.append(frame_id)
        self.bbox_history.append(bbox)
        self.jersey_numbers.append(jersey_number)
        self.team_labels.append(team)
        self.roles.append(role)

        if reid_embedding is not None:
            self.reid_embeddings.append(reid_embedding)

    def get_mean_embedding(self) -> np.ndarray | None:
        """Get mean ReID embedding for this track.

        Returns:
            Mean embedding vector or None.
        """
        if not self.reid_embeddings:
            return None
        return np.mean(self.reid_embeddings, axis=0)

    def get_majority_jersey(self) -> int | None:
        """Get majority-voted jersey number.

        Returns:
            Most common jersey number or None.
        """
        valid_numbers = [n for n in self.jersey_numbers if n is not None]
        if not valid_numbers:
            return None

        from collections import Counter
        counter = Counter(valid_numbers)
        return counter.most_common(1)[0][0]

    def get_majority_team(self) -> str | None:
        """Get majority-voted team label.

        Returns:
            Most common team label or None.
        """
        valid_teams = [t for t in self.team_labels if t is not None]
        if not valid_teams:
            return None

        from collections import Counter
        counter = Counter(valid_teams)
        return counter.most_common(1)[0][0]


class TrackManager:
    """Manage track states across frames."""

    def __init__(self):
        """Initialize track manager."""
        self.tracks: dict[int, TrackState] = {}

    def update_track(
        self,
        track_id: int,
        frame_id: int,
        bbox: list[float],
        reid_embedding: np.ndarray | None = None,
        jersey_number: int | None = None,
        team: str | None = None,
        role: str = "player",
    ) -> TrackState:
        """Update or create a track with new observation.

        Args:
            track_id: Track identifier.
            frame_id: Frame index.
            bbox: Bounding box.
            reid_embedding: ReID feature vector.
            jersey_number: Detected jersey number.
            team: Team classification.
            role: Player role.

        Returns:
            Updated TrackState.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id)

        self.tracks[track_id].add_observation(
            frame_id=frame_id,
            bbox=bbox,
            reid_embedding=reid_embedding,
            jersey_number=jersey_number,
            team=team,
            role=role,
        )

        return self.tracks[track_id]

    def get_track(self, track_id: int) -> TrackState | None:
        """Get track state by ID.

        Args:
            track_id: Track identifier.

        Returns:
            TrackState or None.
        """
        return self.tracks.get(track_id)

    def get_all_tracks(self) -> list[TrackState]:
        """Get all track states.

        Returns:
            List of all TrackState objects.
        """
        return list(self.tracks.values())

    def finalize_tracks(self) -> dict[int, dict[str, Any]]:
        """Finalize all tracks with majority voting.

        Returns:
            Dictionary of track_id to finalized track info.
        """
        finalized = {}
        for track_id, track in self.tracks.items():
            finalized[track_id] = {
                "track_id": track_id,
                "jersey_number": track.get_majority_jersey(),
                "team": track.get_majority_team(),
                "num_observations": len(track.frame_ids),
                "frame_range": (
                    min(track.frame_ids) if track.frame_ids else None,
                    max(track.frame_ids) if track.frame_ids else None,
                ),
            }
        return finalized
