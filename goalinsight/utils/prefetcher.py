"""Background frame prefetching for pipeline stages.

Provides two prefetcher classes:
- FramePrefetcher: Simple frame reader (used by stage2)
- DetectionPrefetcher: Frame reader + keypoint/line detection (used by stage1 physical)
"""

import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


class FramePrefetcher:
    """Read video frames in a background thread to overlap IO with GPU inference."""

    def __init__(self, video_path: str | Path, frame_indices: list[int], prefetch_size: int = 4):
        self._cap = cv2.VideoCapture(str(video_path))
        self._frame_indices = frame_indices
        self._buffer: dict[int, np.ndarray] = {}
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._prefetch_size = prefetch_size
        self._done = False

        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self):
        for fidx in self._frame_indices:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = self._cap.read()
            if not ret:
                frame = None

            with self._cond:
                # Wait if buffer is full
                while len(self._buffer) >= self._prefetch_size and not self._done:
                    self._cond.wait(timeout=1.0)
                if self._done:
                    break
                self._buffer[fidx] = frame
                self._cond.notify_all()

        with self._cond:
            self._done = True
            self._cond.notify_all()
        self._cap.release()

    def get(self, frame_idx: int, timeout: float = 30.0) -> np.ndarray | None:
        with self._cond:
            while frame_idx not in self._buffer:
                if self._done and frame_idx not in self._buffer:
                    return None
                self._cond.wait(timeout=timeout)
            frame = self._buffer.pop(frame_idx)
            self._cond.notify_all()  # Unblock reader if waiting on full buffer
            return frame

    def shutdown(self):
        with self._cond:
            self._done = True
            self._cond.notify_all()
        self._thread.join(timeout=5)


class DetectionPrefetcher:
    """Prefetch frames and run keypoint/line detection in background threads.

    Overlaps I/O + detection with calibration on the main thread.
    Uses a dedicated video reader thread and a pool for detection.
    """

    def __init__(
        self,
        video_path: str,
        frame_indices: list[int],
        kp_detector,
        line_detector,
        num_workers: int = 2,
        prefetch_size: int = 4,
    ):
        self.frame_indices = frame_indices
        self.kp_detector = kp_detector
        self.line_detector = line_detector
        self.prefetch_size = prefetch_size

        # Results queue (ordered dict-like)
        self._results: dict[int, tuple] = {}
        self._lock = threading.Lock()
        self._ready = threading.Condition(self._lock)
        self._done = False

        # Start background pipeline
        self._executor = ThreadPoolExecutor(max_workers=num_workers)
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(str(video_path),),
            daemon=True,
        )
        self._reader_thread.start()

    def _reader_loop(self, video_path: str):
        """Read frames and submit detection jobs."""
        cap = cv2.VideoCapture(video_path)
        pending = deque()  # Track pending futures to limit prefetch

        for frame_idx in self.frame_indices:
            # Limit prefetch depth
            while len(pending) >= self.prefetch_size:
                # Wait for oldest to complete
                oldest_idx, oldest_future = pending[0]
                oldest_future.result()  # Block until done
                pending.popleft()

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            future = self._executor.submit(
                self._detect_frame, frame_idx, frame,
            )
            pending.append((frame_idx, future))

        # Wait for all remaining
        for idx, future in pending:
            future.result()

        cap.release()
        with self._lock:
            self._done = True
            self._ready.notify_all()

    def _detect_frame(self, frame_idx: int, frame: np.ndarray):
        """Run detection on a single frame."""
        keypoints = self.kp_detector.detect(frame, convert_to_soccernet=False)
        lines = self.line_detector.detect(frame) if self.line_detector is not None else []

        with self._lock:
            self._results[frame_idx] = (frame, keypoints, lines)
            self._ready.notify_all()

    def get(self, frame_idx: int, timeout: float = 30.0) -> tuple[np.ndarray, list, list] | None:
        """Get detection results for a frame, blocking until ready.

        Returns None if the frame could not be read (e.g. video shorter than
        reported by CAP_PROP_FRAME_COUNT).
        """
        with self._lock:
            while frame_idx not in self._results:
                if self._done and frame_idx not in self._results:
                    return None
                self._ready.wait(timeout=timeout)
                if frame_idx not in self._results and self._done:
                    return None
            result = self._results.pop(frame_idx)
            return result

    def shutdown(self):
        self._reader_thread.join(timeout=5)
        self._executor.shutdown(wait=True)
