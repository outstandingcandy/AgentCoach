"""GoalInsight: 3-Stage Soccer Analysis Pipeline.

Stage 1: Field Registration - Camera calibration via keypoint/line detection
Stage 2: Tracking & Identification - Player detection, tracking, and ReID
Stage 3: Post-processing - Temporal consistency and tracklet merging

Supports multiple backends for each component:
- Calibration: PnLCalib (default), NBJW
- ReID: OSNet (default), PRTReID
- Jersey recognition: Qwen VL (default), MMOCR
- Team classification: KMeans (default), Tracklet clustering
- Visualization: Minimal (default), Step-by-step
"""

from .stage0 import run_stage0, get_segments_for_pipeline
from .stage1 import run_stage1
from .stage2 import run_stage2
from .stage3 import run_stage3
from .video_processor import VideoProcessor, VideoSampler

# Factory functions for creating components
from .utils.config import (
    get_calibrator,
    get_reid_extractor,
    get_jersey_recognizer,
    get_team_classifier,
    get_visualizer,
    get_side_labeler,
)

__all__ = [
    # Stage runners
    "run_stage0",
    "get_segments_for_pipeline",
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "VideoProcessor",
    "VideoSampler",
    # Factory functions
    "get_calibrator",
    "get_reid_extractor",
    "get_jersey_recognizer",
    "get_team_classifier",
    "get_visualizer",
    "get_side_labeler",
]
