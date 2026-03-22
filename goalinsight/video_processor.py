"""Video frame extraction and processing utilities."""

from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np
from tqdm import tqdm


class VideoProcessor:
    """Process video files and extract frames."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize video processor.

        Args:
            config: Processing configuration.
        """
        self.config = config or {}
        self.target_fps = self.config.get("frame_rate", 25)
        self.resize_width = self.config.get("resize_width")
        self.resize_height = self.config.get("resize_height")

    def get_video_info(self, video_path: str | Path) -> dict[str, Any]:
        """Get video file information.

        Args:
            video_path: Path to video file.

        Returns:
            Dictionary with video metadata.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        info = {
            "path": str(video_path),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            "fourcc": int(cap.get(cv2.CAP_PROP_FOURCC)),
        }

        cap.release()
        return info

    def extract_frames(
        self,
        video_path: str | Path,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
        show_progress: bool = True,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Extract frames from video.

        Args:
            video_path: Path to video file.
            start_frame: Starting frame index.
            end_frame: Ending frame index (None for all).
            step: Frame step (1 for every frame).
            show_progress: Whether to show progress bar.

        Yields:
            Tuple of (frame_id, timestamp, frame).
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if end_frame is None:
            end_frame = total_frames

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Calculate number of frames to process
        num_frames = (end_frame - start_frame + step - 1) // step

        progress = tqdm(
            total=num_frames,
            desc="Extracting frames",
            disable=not show_progress,
        )

        frame_id = start_frame
        while frame_id < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = frame_id / fps

            # Resize if configured
            frame = self._resize_frame(frame)

            yield frame_id, timestamp, frame

            # Skip frames according to step
            frame_id += step
            if step > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            progress.update(1)

        progress.close()
        cap.release()

    def extract_frames_at_fps(
        self,
        video_path: str | Path,
        target_fps: float | None = None,
        show_progress: bool = True,
    ) -> Iterator[tuple[int, float, np.ndarray]]:
        """Extract frames at a target frame rate.

        Args:
            video_path: Path to video file.
            target_fps: Target frame rate (None to use config).
            show_progress: Whether to show progress bar.

        Yields:
            Tuple of (frame_id, timestamp, frame).
        """
        target_fps = target_fps or self.target_fps
        info = self.get_video_info(video_path)
        source_fps = info["fps"]

        # Calculate frame step
        if target_fps >= source_fps:
            step = 1
        else:
            step = int(round(source_fps / target_fps))

        yield from self.extract_frames(
            video_path,
            step=step,
            show_progress=show_progress,
        )

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame if configured.

        Args:
            frame: Input frame.

        Returns:
            Resized frame or original.
        """
        if self.resize_width is None and self.resize_height is None:
            return frame

        h, w = frame.shape[:2]
        target_w = self.resize_width or w
        target_h = self.resize_height or h

        if (w, h) == (target_w, target_h):
            return frame

        return cv2.resize(frame, (target_w, target_h))

    def extract_frame_at(
        self,
        video_path: str | Path,
        frame_id: int,
    ) -> np.ndarray | None:
        """Extract a single frame by index.

        Args:
            video_path: Path to video file.
            frame_id: Frame index to extract.

        Returns:
            Frame or None if failed.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        return self._resize_frame(frame)

    def save_frame(
        self,
        frame: np.ndarray,
        output_path: str | Path,
    ) -> None:
        """Save a frame to file.

        Args:
            frame: Frame to save.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

    def create_video_writer(
        self,
        output_path: str | Path,
        width: int,
        height: int,
        fps: float = 25.0,
        codec: str = "mp4v",
    ) -> cv2.VideoWriter:
        """Create a video writer.

        Args:
            output_path: Output video path.
            width: Frame width.
            height: Frame height.
            fps: Output frame rate.
            codec: Video codec.

        Returns:
            OpenCV VideoWriter object.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        return cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))


class FrameBatcher:
    """Batch frames for efficient processing."""

    def __init__(self, batch_size: int = 16):
        """Initialize frame batcher.

        Args:
            batch_size: Number of frames per batch.
        """
        self.batch_size = batch_size

    def batch_iterator(
        self,
        frame_iterator: Iterator[tuple[int, float, np.ndarray]],
    ) -> Iterator[list[tuple[int, float, np.ndarray]]]:
        """Create batched iterator from frame iterator.

        Args:
            frame_iterator: Iterator yielding (frame_id, timestamp, frame).

        Yields:
            List of (frame_id, timestamp, frame) tuples.
        """
        batch = []

        for item in frame_iterator:
            batch.append(item)

            if len(batch) >= self.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch


class VideoSampler:
    """Sample frames from video for initialization."""

    def __init__(self, num_samples: int = 10):
        """Initialize video sampler.

        Args:
            num_samples: Number of frames to sample.
        """
        self.num_samples = num_samples

    def sample_uniformly(
        self,
        video_path: str | Path,
    ) -> list[tuple[int, np.ndarray]]:
        """Sample frames uniformly from video.

        Args:
            video_path: Path to video file.

        Returns:
            List of (frame_id, frame) tuples.
        """
        processor = VideoProcessor()
        info = processor.get_video_info(video_path)
        total_frames = info["frame_count"]

        # Calculate sample indices
        indices = np.linspace(0, total_frames - 1, self.num_samples, dtype=int)

        samples = []
        for idx in indices:
            frame = processor.extract_frame_at(video_path, int(idx))
            if frame is not None:
                samples.append((int(idx), frame))

        return samples

    def sample_keyframes(
        self,
        video_path: str | Path,
        threshold: float = 30.0,
    ) -> list[tuple[int, np.ndarray]]:
        """Sample keyframes based on scene changes.

        Args:
            video_path: Path to video file.
            threshold: Scene change threshold.

        Returns:
            List of (frame_id, frame) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        keyframes = []
        prev_frame = None
        frame_id = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if prev_frame is not None:
                # Compute frame difference
                diff = cv2.absdiff(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY),
                )
                mean_diff = np.mean(diff)

                if mean_diff > threshold:
                    keyframes.append((frame_id, frame.copy()))

                    if len(keyframes) >= self.num_samples:
                        break

            prev_frame = frame.copy()
            frame_id += 1

        cap.release()

        # If not enough keyframes, add some uniform samples
        if len(keyframes) < self.num_samples // 2:
            keyframes = self.sample_uniformly(video_path)

        return keyframes[:self.num_samples]
