"""StrongSORT tracking package.

Public API: :class:`StrongSORTTracker`. Internals (Track, KalmanFilter)
are exposed for testing and downstream code that needs to inspect
tracker state.
"""

from .kalman import KalmanFilter, KalmanState, chi2inv95
from .track import Track, TrackStatus
from .tracker import StrongSORTTracker

__all__ = [
    "StrongSORTTracker",
    "Track",
    "TrackStatus",
    "KalmanFilter",
    "KalmanState",
    "chi2inv95",
]
