"""Track dataclass and lifecycle status enum.

A :class:`Track` is one continuous identity through the video. The
tracker keeps a list of tracks and matches incoming detections to
them every frame; this file defines just the data carrier — the
matching / promotion / deletion rules live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class TrackStatus(Enum):
    """Track lifecycle status."""
    TENTATIVE = 1  # Newly created — needs ``n_init`` consecutive hits to confirm
    CONFIRMED = 2  # Active confirmed track
    DELETED = 3    # Marked for removal at end of update()


@dataclass
class Track:
    """Single object track."""
    track_id: int
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: int = 1
    age: int = 1
    time_since_update: int = 0

    # Kalman state (typed loosely to avoid circular import with kalman.py)
    kalman_state: object | None = None

    # Appearance features (ReID embeddings)
    features: list = field(default_factory=list)
    smooth_feature: np.ndarray | None = None

    # Detection info
    bbox: list = field(default_factory=list)  # [x1, y1, x2, y2]
    confidence: float = 0.0
    class_id: int = 0

    # Recent bbox-center history for stationary-detection. Stores at
    # most ``stationary_window`` (cx, cy) tuples so the lifecycle
    # cleanup pass can detect ghost tracks where YOLO keeps
    # re-detecting the same static background object.
    center_history: list = field(default_factory=list)

    # Last successful projection of the bbox foot-point onto the pitch
    # in world coordinates (metres). Refreshed whenever the orchestrator
    # passes a calibration-derived ``pitch_pos`` on the matched
    # detection. Used by the gating step to reject implausible matches
    # in metric space rather than fragile pixel-space aspect/h gates.
    # ``None`` when calibration was unavailable for the matched frame.
    pitch_pos: tuple | None = None

    # Attributes propagated from detections / set by downstream stages.
    team: str | None = None
    jersey_number: int | None = None
    role: str = "player"

    def update_feature(self, feature: np.ndarray, alpha: float = 0.9):
        """Update smooth feature with EMA."""
        self.features.append(feature)
        # Keep only last 100 features
        if len(self.features) > 100:
            self.features = self.features[-100:]

        if self.smooth_feature is None:
            self.smooth_feature = feature.copy()
        else:
            self.smooth_feature = alpha * self.smooth_feature + (1 - alpha) * feature
            # Normalize
            self.smooth_feature /= np.linalg.norm(self.smooth_feature) + 1e-8

    def get_mean_feature(self) -> np.ndarray | None:
        """Get mean of all stored features (used by consolidation)."""
        if not self.features:
            return None
        return np.mean(self.features, axis=0)
