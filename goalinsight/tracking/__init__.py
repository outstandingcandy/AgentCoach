"""Stage 2: Tracking and Identification modules."""

from .detector import PlayerDetector
from .tracker import PlayerTracker, TrackState, TrackManager
from .strongsort_tracker import StrongSORTTracker, Track, TrackStatus
from .role_classifier import RoleClassifier

# Ball detection and tracking
from .ball_detector import BallDetector
from .ball_tracker import BallTracker, BallTrack, BallStatus, BallKalmanFilter

# Import from new locations (with backwards compatibility aliases)
from .reid.osnet_extractor import OSNetExtractor, ReIDGallery
from .team.kmeans_classifier import KMeansTeamClassifier, GoalkeeperDetector

# Backwards compatibility aliases
ReIDExtractor = OSNetExtractor
TeamClassifier = KMeansTeamClassifier

# Jersey recognition moved to ../jersey/ module
from ..jersey.qwen_recognizer import QwenJerseyRecognizer, JerseyNumberAggregator
JerseyRecognizer = QwenJerseyRecognizer

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
    "BallTrack",
    "BallStatus",
    "BallKalmanFilter",
    # New names
    "OSNetExtractor",
    "KMeansTeamClassifier",
    "QwenJerseyRecognizer",
    "ReIDGallery",
    "JerseyNumberAggregator",
    # Backwards compatibility aliases
    "ReIDExtractor",
    "JerseyRecognizer",
    "RoleClassifier",
    "TeamClassifier",
    "GoalkeeperDetector",
]
