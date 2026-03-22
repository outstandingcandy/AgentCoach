"""Preprocessing module for shot detection and video segmentation.

This module provides Stage 0 preprocessing for the GoalInsight pipeline:
- Shot boundary detection to identify camera cuts
- Video segmentation to split videos into continuous shots
"""

from dataclasses import dataclass


@dataclass
class ShotBoundary:
    """Represents a detected shot boundary (camera cut).

    Attributes:
        frame_idx: Frame index where the shot change occurs.
        score: Detection confidence score (0-1).
        method: Detection method that identified this boundary.
    """

    frame_idx: int
    score: float
    method: str


@dataclass
class SegmentInfo:
    """Information about a video segment.

    Attributes:
        segment_id: Unique identifier for this segment.
        start_frame: First frame of the segment (inclusive).
        end_frame: Last frame of the segment (exclusive).
        duration_frames: Number of frames in the segment.
        duration_seconds: Duration in seconds.
        output_path: Path to the segmented video file (if exported).
    """

    segment_id: int
    start_frame: int
    end_frame: int
    duration_frames: int
    duration_seconds: float
    output_path: str | None = None


from .shot_detector import ShotDetector
from .video_segmenter import VideoSegmenter

__all__ = [
    "ShotBoundary",
    "SegmentInfo",
    "ShotDetector",
    "VideoSegmenter",
]
