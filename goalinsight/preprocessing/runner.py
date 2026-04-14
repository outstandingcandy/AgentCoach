"""Shot Detection & Video Segmentation runner.

Detects shot boundaries (camera cuts) in the video, segments it into
continuous shots, and exports frame ranges for downstream processing.
"""

import json
import logging
from pathlib import Path

from . import ShotBoundary, ShotDetector, SegmentInfo, VideoSegmenter
from ..utils.config import get_default_config

logger = logging.getLogger(__name__)


def run_shot_detection(
    video_path: Path,
    output_dir: Path,
    config: dict | None = None,
) -> dict:
    """Run shot detection and segmentation.

    Args:
        video_path: Path to input video.
        output_dir: Directory for output files.
        config: Optional configuration dict.

    Returns:
        Dict with preprocessing statistics and segment info.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = get_default_config()

    sd_config = config.get("shot_detection", {})
    vs_config = config.get("video_segmentation", {})

    logger.info(f"Shot Detection: Processing {video_path.name}")

    if not sd_config.get("enabled", True):
        logger.info("Shot Detection: Disabled, treating video as single segment")
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        segments = [
            SegmentInfo(
                segment_id=0,
                start_frame=0,
                end_frame=total_frames,
                duration_frames=total_frames,
                duration_seconds=total_frames / fps,
            )
        ]
        boundaries = []
    else:
        logger.info("Shot Detection: Detecting shot boundaries...")
        detector = ShotDetector(config)
        boundaries = detector.detect(video_path)
        logger.info(f"Shot Detection: Found {len(boundaries)} shot boundaries")

        logger.info("Shot Detection: Segmenting video...")
        segmenter = VideoSegmenter(config)
        segments = segmenter.segment(video_path, boundaries, output_dir)
        logger.info(f"Shot Detection: Created {len(segments)} segments")

    _save_boundaries(boundaries, output_dir)

    stats = {
        "video_path": str(video_path),
        "num_boundaries": len(boundaries),
        "num_segments": len(segments),
        "segments": [
            {
                "segment_id": seg.segment_id,
                "start_frame": seg.start_frame,
                "end_frame": seg.end_frame,
                "duration_frames": seg.duration_frames,
                "duration_seconds": seg.duration_seconds,
                "output_path": seg.output_path,
            }
            for seg in segments
        ],
        "shot_detection_enabled": sd_config.get("enabled", True),
        "video_export_enabled": vs_config.get("enabled", True),
    }

    with open(output_dir / "shot_detection_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Shot Detection Complete:")
    logger.info(f"  Shot boundaries: {len(boundaries)}")
    logger.info(f"  Segments: {len(segments)}")
    for seg in segments:
        logger.info(f"    [{seg.segment_id}] frames {seg.start_frame}-{seg.end_frame} ({seg.duration_seconds:.1f}s)")

    return stats


def _save_boundaries(boundaries: list[ShotBoundary], output_dir: Path) -> None:
    """Save shot boundary information to JSON."""
    boundary_data = {
        "num_boundaries": len(boundaries),
        "boundaries": [
            {
                "frame_idx": b.frame_idx,
                "score": b.score,
                "method": b.method,
            }
            for b in boundaries
        ],
    }
    with open(output_dir / "shot_boundaries.json", "w") as f:
        json.dump(boundary_data, f, indent=2)


def get_segments_for_pipeline(output_dir: Path) -> list[SegmentInfo]:
    """Load segment information for pipeline processing."""
    return VideoSegmenter.load_frame_ranges(output_dir)
