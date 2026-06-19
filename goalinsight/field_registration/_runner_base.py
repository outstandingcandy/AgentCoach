"""Shared utilities for field registration runners.

Extracts common boilerplate: video opening, frame sampling, result storage
initialization, output saving, and statistics computation.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..utils.config import FrameSampler


@dataclass
class VideoInfo:
    """Video metadata extracted from cv2.VideoCapture."""
    cap: cv2.VideoCapture
    total_frames: int
    fps: float
    width: int
    height: int


def open_video(video_path: Path) -> VideoInfo:
    """Open a video file and extract metadata.

    Returns a VideoInfo with the open capture object (caller must release).

    Raises:
        RuntimeError: If the video cannot be opened.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    return VideoInfo(
        cap=cap,
        total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=cap.get(cv2.CAP_PROP_FPS),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )


def probe_video(video_path: Path) -> tuple[int, float, int, int]:
    """Read (frame_count, fps, width, height) without holding a capture open.

    Used by callers that only need metadata (resolver, PipelineContext init)
    and don't want to leak an open ``cv2.VideoCapture``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def make_sampler(
    video: VideoInfo,
    process_fps: float | None,
    backend_label: str = "Stage 1",
) -> FrameSampler:
    """Build a FrameSampler and print video / sampling info."""
    sampler = FrameSampler(video.total_frames, video.fps, process_fps)
    print(f"Video: {video.total_frames} frames @ {video.fps:.1f} fps, "
          f"{video.width}x{video.height}")
    print(f"Processing {len(sampler)} frames at {process_fps or video.fps} fps")
    return sampler


def init_calibration_results(
    video_path: Path,
    video: VideoInfo,
    process_fps: float | None,
    extra_video_info: dict[str, Any] | None = None,
    extra_top_level: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create the base ``calibration_results`` dict.

    Args:
        extra_video_info: Additional keys to merge into ``video_info``
            (e.g. ``pitch_length``, ``pitch_width``).
        extra_top_level: Additional top-level keys
            (e.g. ``{"backend": "homography"}``).
    """
    results: dict[str, Any] = {
        "video_info": {
            "path": str(video_path),
            "total_frames": video.total_frames,
            "fps": video.fps,
            "width": video.width,
            "height": video.height,
            "process_fps": process_fps,
            **(extra_video_info or {}),
        },
        "frames": {},
    }
    if extra_top_level:
        results.update(extra_top_level)
    return results


def save_calibration_outputs(
    output_dir: Path,
    calibration_results: dict[str, Any],
    homographies: dict[int, Any],
    camera_poses: dict[int, Any] | None = None,
    json_default: Any = None,
) -> None:
    """Write the standard output files (metadata JSON + homographies pickle).

    Args:
        camera_poses: If provided, also writes ``camera_poses.pkl`` and
            ``camera_poses.json``.
        json_default: Optional ``default`` callable for ``json.dump``.
    """
    with open(output_dir / "calibration_metadata.json", "w") as f:
        json.dump(calibration_results, f, default=json_default)
    with open(output_dir / "homographies.pkl", "wb") as f:
        pickle.dump(homographies, f)

    if camera_poses is not None:
        with open(output_dir / "camera_poses.pkl", "wb") as f:
            pickle.dump(camera_poses, f)

        # Human-readable JSON version
        poses_json: dict[str, Any] = {}
        for fidx, pose in camera_poses.items():
            entry: dict[str, Any] = {}
            for key in ("K", "dist_coeffs"):
                if key in pose:
                    entry[key] = np.array(pose[key]).tolist()
            for key in ("rvec", "tvec"):
                if key in pose:
                    entry[key] = np.array(pose[key]).flatten().tolist()
            poses_json[str(fidx)] = entry

        with open(output_dir / "camera_poses.json", "w") as f:
            json.dump(poses_json, f, indent=2)


def compute_calibration_stats(
    calibration_results: dict[str, Any],
    video: VideoInfo,
    sampler: FrameSampler,
    calibrated_count: int,
    *,
    error_key: str = "reprojection_error",
    exclude_interpolated: bool = False,
    extra_stats: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute standard calibration statistics.

    Returns a stats dict with at least ``total_frames``, ``processed_frames``,
    ``calibrated_frames``, ``calibration_rate``, and optional ``mean_error``,
    ``median_error``.
    """
    stats: dict[str, Any] = {
        "total_frames": video.total_frames,
        "processed_frames": len(sampler),
        "calibrated_frames": calibrated_count,
        "calibration_rate": calibrated_count / len(sampler) if sampler else 0,
    }
    if extra_stats:
        stats.update(extra_stats)

    frames = calibration_results["frames"]
    errors = [
        frames[idx][error_key]
        for idx in frames
        if frames[idx].get("calibrated")
        and (not exclude_interpolated or not frames[idx].get("interpolated"))
        and frames[idx].get(error_key) is not None
    ]
    if errors:
        stats["mean_error"] = float(np.mean(errors))
        stats["median_error"] = float(np.median(errors))

    # World errors (if present)
    world_errors = [
        frames[idx]["world_error"]
        for idx in frames
        if frames[idx].get("calibrated")
        and (not exclude_interpolated or not frames[idx].get("interpolated"))
        and frames[idx].get("world_error") is not None
    ]
    if world_errors:
        stats["mean_world_error"] = float(np.mean(world_errors))
        stats["median_world_error"] = float(np.median(world_errors))

    return stats


def print_calibration_summary(
    stats: dict[str, Any],
    label: str = "Stage 1",
) -> None:
    """Print a human-readable calibration summary."""
    n = stats["calibrated_frames"]
    total = stats["processed_frames"]
    rate = stats["calibration_rate"] * 100
    print(f"\n{label} Complete:")
    print(f"  Calibrated: {n}/{total} ({rate:.1f}%)")
    if "median_error" in stats:
        print(f"  Median error: {stats['median_error']:.2f} px")
    if "mean_error" in stats:
        print(f"  Mean error: {stats['mean_error']:.2f} px")
    if "median_world_error" in stats:
        print(f"  Median world error: {stats['median_world_error']:.2f} m")
