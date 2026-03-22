"""Video segmentation based on shot boundaries.

Splits video into continuous segments at detected shot boundaries.
Can either export segment files or just provide frame ranges for lazy processing.
"""

import json
import subprocess
from pathlib import Path

import cv2

from . import SegmentInfo, ShotBoundary


class VideoSegmenter:
    """Segments video based on shot boundaries.

    Can export segments as separate video files or just provide frame ranges
    for lazy processing by downstream stages.
    """

    def __init__(self, config: dict | None = None):
        """Initialize video segmenter.

        Args:
            config: Configuration dict with video_segmentation settings.
        """
        config = config or {}
        vs_config = config.get("video_segmentation", {})

        self.enabled = vs_config.get("enabled", True)
        self.output_format = vs_config.get("output_format", "mkv")
        self.export_frame_ranges = vs_config.get("export_frame_ranges", True)
        self.accurate_cut = vs_config.get("accurate_cut", True)  # Re-encode for frame-accurate cuts

    def segment(
        self,
        video_path: str | Path,
        boundaries: list[ShotBoundary],
        output_dir: str | Path,
    ) -> list[SegmentInfo]:
        """Segment video at shot boundaries.

        Args:
            video_path: Path to source video.
            boundaries: List of detected shot boundaries.
            output_dir: Directory for output files.

        Returns:
            List of segment information.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Build segment ranges
        segments = self._build_segments(boundaries, total_frames, fps)

        # Export frame ranges JSON
        if self.export_frame_ranges:
            self._export_frame_ranges(segments, output_dir)

        # Export video segments if enabled
        if self.enabled:
            segments = self._export_segments(video_path, segments, output_dir, fps)

        return segments

    def _build_segments(
        self,
        boundaries: list[ShotBoundary],
        total_frames: int,
        fps: float,
    ) -> list[SegmentInfo]:
        """Build segment list from boundaries.

        Args:
            boundaries: Detected shot boundaries.
            total_frames: Total frames in video.
            fps: Video frame rate.

        Returns:
            List of segment info objects.
        """
        segments = []

        # Sort boundaries by frame index
        sorted_boundaries = sorted(boundaries, key=lambda b: b.frame_idx)

        # Build segment ranges
        start_frame = 0
        for i, boundary in enumerate(sorted_boundaries):
            end_frame = boundary.frame_idx
            duration_frames = end_frame - start_frame

            if duration_frames > 0:
                segments.append(
                    SegmentInfo(
                        segment_id=len(segments),
                        start_frame=start_frame,
                        end_frame=end_frame,
                        duration_frames=duration_frames,
                        duration_seconds=duration_frames / fps,
                    )
                )

            start_frame = end_frame

        # Add final segment
        if start_frame < total_frames:
            duration_frames = total_frames - start_frame
            segments.append(
                SegmentInfo(
                    segment_id=len(segments),
                    start_frame=start_frame,
                    end_frame=total_frames,
                    duration_frames=duration_frames,
                    duration_seconds=duration_frames / fps,
                )
            )

        # If no boundaries detected, return single segment for entire video
        if not segments:
            segments.append(
                SegmentInfo(
                    segment_id=0,
                    start_frame=0,
                    end_frame=total_frames,
                    duration_frames=total_frames,
                    duration_seconds=total_frames / fps,
                )
            )

        return segments

    def _export_frame_ranges(
        self, segments: list[SegmentInfo], output_dir: Path
    ) -> None:
        """Export segment frame ranges to JSON.

        Args:
            segments: List of segment info.
            output_dir: Output directory.
        """
        frame_ranges = {
            "num_segments": len(segments),
            "segments": [
                {
                    "segment_id": seg.segment_id,
                    "start_frame": seg.start_frame,
                    "end_frame": seg.end_frame,
                    "duration_frames": seg.duration_frames,
                    "duration_seconds": seg.duration_seconds,
                }
                for seg in segments
            ],
        }

        with open(output_dir / "frame_ranges.json", "w") as f:
            json.dump(frame_ranges, f, indent=2)

    def _export_segments(
        self,
        video_path: Path,
        segments: list[SegmentInfo],
        output_dir: Path,
        fps: float,
    ) -> list[SegmentInfo]:
        """Export video segments as separate files.

        Args:
            video_path: Source video path.
            segments: List of segment info.
            output_dir: Output directory.
            fps: Video frame rate.

        Returns:
            Updated list of segments with output paths.
        """
        from tqdm import tqdm

        updated_segments = []

        for seg in tqdm(segments, desc="Stage 0: Exporting segments"):
            output_path = output_dir / f"segment_{seg.segment_id:03d}.{self.output_format}"

            # Calculate time positions
            start_time = seg.start_frame / fps
            duration = seg.duration_frames / fps

            if self.accurate_cut:
                # Frame-accurate cutting with re-encoding
                # Use -ss after -i for accurate seeking, then re-encode
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(video_path),
                    "-ss", str(start_time),
                    "-t", str(duration),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "18",  # High quality
                    "-c:a", "aac",
                    "-b:a", "128k",
                    str(output_path),
                ]
            else:
                # Fast copy mode (keyframe-aligned, may be slightly inaccurate)
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-ss", str(start_time),
                    "-i", str(video_path),
                    "-t", str(duration),
                    "-c", "copy",
                    "-avoid_negative_ts", "make_zero",
                    str(output_path),
                ]

            try:
                subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                updated_seg = SegmentInfo(
                    segment_id=seg.segment_id,
                    start_frame=seg.start_frame,
                    end_frame=seg.end_frame,
                    duration_frames=seg.duration_frames,
                    duration_seconds=seg.duration_seconds,
                    output_path=str(output_path),
                )
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to export segment {seg.segment_id}: {e.stderr}")
                updated_seg = seg

            updated_segments.append(updated_seg)

        return updated_segments

    @staticmethod
    def load_frame_ranges(output_dir: str | Path) -> list[SegmentInfo]:
        """Load segment info from frame_ranges.json.

        Args:
            output_dir: Directory containing frame_ranges.json.

        Returns:
            List of segment info objects.
        """
        output_dir = Path(output_dir)
        frame_ranges_path = output_dir / "frame_ranges.json"

        if not frame_ranges_path.exists():
            raise FileNotFoundError(f"Frame ranges not found: {frame_ranges_path}")

        with open(frame_ranges_path) as f:
            data = json.load(f)

        segments = []
        for seg_data in data.get("segments", []):
            # Check for exported segment file
            output_path = output_dir / f"segment_{seg_data['segment_id']:03d}.mkv"
            if not output_path.exists():
                output_path = None
            else:
                output_path = str(output_path)

            segments.append(
                SegmentInfo(
                    segment_id=seg_data["segment_id"],
                    start_frame=seg_data["start_frame"],
                    end_frame=seg_data["end_frame"],
                    duration_frames=seg_data["duration_frames"],
                    duration_seconds=seg_data["duration_seconds"],
                    output_path=output_path,
                )
            )

        return segments
