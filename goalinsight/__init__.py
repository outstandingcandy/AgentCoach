"""GoalInsight: Soccer Video Analysis Pipeline.

Modules:
- Field Registration: Camera calibration via keypoint/line detection
- Tracking & Identification: Player detection, tracking, and ReID
- Post-processing: Temporal consistency and tracklet merging
- Goal Detection: Ball tracking and goal-line crossing detection

Supports multiple backends for each component:
- Calibration: PnLCalib (default), BroadTrack, Physical, NBJW
- ReID: OSNet (default), PRTReID
- Jersey recognition: Qwen VL (default), MMOCR
- Team classification: KMeans (default), Tracklet clustering
- Visualization: Minimal (default), Step-by-step
"""

from .pipeline import Pipeline, Stage, PipelineContext, STAGE_REGISTRY
from .video_processor import VideoProcessor, VideoSampler
from .highlights import run_highlights
from .preprocessing.runner import get_segments_for_pipeline

# Factory functions for creating components
from .utils.factories import (
    get_calibrator,
    get_reid_extractor,
    get_jersey_recognizer,
    get_team_classifier,
    get_visualizer,
    get_side_labeler,
)

__all__ = [
    # Pipeline
    "Pipeline",
    "Stage",
    "PipelineContext",
    "STAGE_REGISTRY",
    # Video processing
    "VideoProcessor",
    "VideoSampler",
    "get_segments_for_pipeline",
    # Factory functions
    "get_calibrator",
    "get_reid_extractor",
    "get_jersey_recognizer",
    "get_team_classifier",
    "get_visualizer",
    "get_side_labeler",
    "run_highlights",
]
