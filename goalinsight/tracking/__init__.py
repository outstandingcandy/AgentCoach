"""Stage 2: Tracking and Identification modules."""

from .detector import PlayerDetector
from .tracker import PlayerTracker, TrackState, TrackManager
from .strongsort_tracker import StrongSORTTracker, Track, TrackStatus

# Ball detection and tracking
from .ball_detector import BallDetector
from .ball_tracker import BallTracker
from .unified_detector import UnifiedDetector

# ReID (used by tracker); team classification is consumed by the
# track_consolidation stage.
from .reid.osnet_extractor import OSNetExtractor, ReIDGallery
from .team.kmeans_classifier import KMeansTeamClassifier, GoalkeeperDetector

__all__ = [
    "PlayerDetector",
    "PlayerTracker",
    "TrackState",
    "TrackManager",
    "StrongSORTTracker",
    "Track",
    "TrackStatus",
    # Ball tracking
    "BallDetector",
    "BallTracker",
    "UnifiedDetector",
    # ReID and team classification
    "OSNetExtractor",
    "ReIDGallery",
    "KMeansTeamClassifier",
    "GoalkeeperDetector",
]
