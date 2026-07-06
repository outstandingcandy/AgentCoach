"""Abstract interfaces for GoalInsight pipeline components.

This module defines abstract base classes for all swappable components:
- Calibration (PnLCalib, NBJW)
- ReID extraction (OSNet, PRTReID)
- Jersey recognition (Qwen VL, Claude, Gemini)
- Team classification (KMeans, Tracklet clustering)
- Team side labeling
- Visualization (Minimal, Step-by-step)
"""

from .calibration import BaseCalibrator
from .reid_extractor import BaseReIDExtractor
from .jersey_recognizer import BaseJerseyRecognizer
from .team_classifier import BaseTeamClassifier, BaseTeamSideLabeler
from .visualizer import BaseVisualizer

__all__ = [
    "BaseCalibrator",
    "BaseReIDExtractor",
    "BaseJerseyRecognizer",
    "BaseTeamClassifier",
    "BaseTeamSideLabeler",
    "BaseVisualizer",
]
