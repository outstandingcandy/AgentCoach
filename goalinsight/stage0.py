#!/usr/bin/env python3
"""Stage 0: Shot Detection & Video Segmentation (Preprocessing).

This stage runs before the main 3-stage pipeline:
1. Detects shot boundaries (camera cuts) in the video
2. Segments the video into continuous shots
3. Exports frame ranges for downstream processing

This ensures each segment is "one continuous shot" which is essential for:
- Field registration (camera parameters remain consistent)
- Tracking (no identity switches from cuts)
"""

import json
from pathlib import Path

from .preprocessing import ShotBoundary, ShotDetector, SegmentInfo, VideoSegmenter
from .utils.config import get_default_config


def run_stage0(
    video_path: Path,
    output_dir: Path,
    config: dict | None = None,
) -> dict:
    """Run Stage 0 shot detection and segmentation.

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

    # Load configuration
    if config is None:
        config = get_default_config()

    sd_config = config.get("shot_detection", {})
    vs_config = config.get("video_segmentation", {})

    print(f"Stage 0: Processing {video_path.name}")

    # Check if shot detection is enabled
    if not sd_config.get("enabled", True):
        print("Stage 0: Shot detection disabled, treating video as single segment")
        # Create single segment for entire video
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
        # Detect shot boundaries
        print("Stage 0: Detecting shot boundaries...")
        detector = ShotDetector(config)
        boundaries = detector.detect(video_path)
        print(f"Stage 0: Found {len(boundaries)} shot boundaries")

        # Segment video
        print("Stage 0: Segmenting video...")
        segmenter = VideoSegmenter(config)
        segments = segmenter.segment(video_path, boundaries, output_dir)
        print(f"Stage 0: Created {len(segments)} segments")

    # Save shot boundary info
    _save_boundaries(boundaries, output_dir)

    # Build stats
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

    # Save summary
    with open(output_dir / "stage0_summary.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\nStage 0 Complete:")
    print(f"  Shot boundaries: {len(boundaries)}")
    print(f"  Segments: {len(segments)}")
    for seg in segments:
        print(f"    [{seg.segment_id}] frames {seg.start_frame}-{seg.end_frame} ({seg.duration_seconds:.1f}s)")

    return stats


def _save_boundaries(boundaries: list[ShotBoundary], output_dir: Path) -> None:
    """Save shot boundary information to JSON.

    Args:
        boundaries: List of detected shot boundaries.
        output_dir: Output directory.
    """
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


def get_segments_for_pipeline(
    stage0_dir: Path,
) -> list[SegmentInfo]:
    """Load segment information for pipeline processing.

    Helper function to load segments from Stage 0 output for use in
    subsequent stages.

    Args:
        stage0_dir: Directory containing Stage 0 outputs.

    Returns:
        List of segment info objects.
    """
    return VideoSegmenter.load_frame_ranges(stage0_dir)


def main():
    """Run Stage 0 on test video."""
    video_path = Path("data/raw_videos/segments/segment_000.mkv")
    output_dir = Path("data/processed/stage0_preprocessing")
    run_stage0(video_path, output_dir)


if __name__ == "__main__":
    main()
