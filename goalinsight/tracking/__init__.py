"""Stage 2: Tracking and Identification modules."""

from .detector import PlayerDetector
from .tracker import PlayerTracker, TrackState, TrackManager
from .strongsort_tracker import StrongSORTTracker, Track, TrackStatus
from .role_classifier import RoleClassifier

# Ball detection and tracking
from .ball_detector import BallDetector
from .ball_tracker import BallTracker
from .unified_detector import UnifiedDetector

# ReID and team classification
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
    "RoleClassifier",
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
