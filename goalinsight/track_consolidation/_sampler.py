"""Representative-frame sampler for each track_id.

For each track, picks the K best frames by ``bbox_area × sharpness``
(Laplacian variance), seeks the source video, and returns an upper-body
crop with optional 2× upscale when the bbox is small.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SampledFrame:
    track_id: int
    frame_id: int
    bbox: list[float]
    score: float
    crop: np.ndarray  # BGR upper-body crop


def _upper_body(crop: np.ndarray, ratio: float) -> np.ndarray:
    h = crop.shape[0]
    return crop[: int(h * ratio), :]


def _sharpness(crop: np.ndarray) -> float:
    """Variance of Laplacian — low value = blurry."""
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def pick_top_frames(
    track_frames: list[tuple[int, list[float]]],
    k: int,
) -> list[tuple[int, list[float]]]:
    """Pick *k* frames with largest bboxes (pre-seek, no crop yet).

    Sharpness is cheap-ish but requires reading the frame; we use bbox
    area as a fast coarse pre-filter to narrow candidates, then rely on
    a second pass for sharpness within those candidates.
    """
    if not track_frames:
        return []
    # Take 3× candidates by area, then sharpness re-ranks inside seek loop
    pool = sorted(track_frames, key=lambda fr: -_bbox_area(fr[1]))[: k * 3]
    # Spread candidates across the track's time range so crops aren't
    # all from the same moment.
    pool.sort(key=lambda fr: fr[0])
    if len(pool) <= k:
        return pool
    step = len(pool) / k
    return [pool[int(i * step)] for i in range(k)]


def sample_crops_for_tracks(
    video_path: str | Path,
    tracks_by_frame: dict[str, list[dict[str, Any]]],
    track_ids: list[int],
    k: int,
    min_bbox_height: int,
    upper_ratio: float,
) -> dict[int, list[SampledFrame]]:
    """Return ``{track_id: [SampledFrame, ...]}`` for every candidate track.

    Opens the source video once and seeks to each selected frame.  For
    efficiency, collects all (frame_id, track_id, bbox) tuples first,
    sorts by frame_id, and does a single forward pass through the video.
    """
    # Build per-track frame lists
    track_ids_set = set(track_ids)
    by_track: dict[int, list[tuple[int, list[float]]]] = {}
    for fk, tracks in tracks_by_frame.items():
        try:
            fid = int(fk)
        except (TypeError, ValueError):
            continue
        for t in tracks:
            tid = int(t["track_id"])
            if tid not in track_ids_set:
                continue
            bbox = t.get("bbox")
            if bbox is None:
                continue
            by_track.setdefault(tid, []).append((fid, list(bbox)))
    # Rank and select candidates per track
    wanted: list[tuple[int, int, list[float]]] = []  # (frame_id, track_id, bbox)
    for tid in track_ids:
        picks = pick_top_frames(by_track.get(tid, []), k)
        for fid, bbox in picks:
            wanted.append((fid, tid, bbox))
    wanted.sort(key=lambda x: x[0])

    # Group by frame so we only seek each frame once
    by_frame: dict[int, list[tuple[int, list[float]]]] = {}
    for fid, tid, bbox in wanted:
        by_frame.setdefault(fid, []).append((tid, bbox))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    result: dict[int, list[SampledFrame]] = {tid: [] for tid in track_ids}
    wanted_frames = sorted(by_frame.keys())
    if not wanted_frames:
        cap.release()
        return result

    # Sequential scan — faster and more predictable than random seeks.
    # We skip-read frames between wanted ones (grab() without decode).
    target_set = set(wanted_frames)
    max_wanted = wanted_frames[-1]
    total_wanted = len(wanted_frames)
    done = 0
    fid = 0
    try:
        while fid <= max_wanted:
            if fid in target_set:
                ok, frame = cap.read()
                if not ok:
                    break
                for tid, bbox in by_frame[fid]:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    x1 = max(0, x1); y1 = max(0, y1)
                    x2 = min(frame.shape[1], x2); y2 = min(frame.shape[0], y2)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    crop = frame[y1:y2, x1:x2]
                    if crop.shape[0] < min_bbox_height:
                        scale = max(1.0, min_bbox_height / crop.shape[0])
                        crop = cv2.resize(
                            crop,
                            (int(crop.shape[1] * scale),
                             int(crop.shape[0] * scale)),
                            interpolation=cv2.INTER_CUBIC,
                        )
                    upper = _upper_body(crop, upper_ratio)
                    if upper.size == 0:
                        continue
                    score = _bbox_area(bbox) * (_sharpness(upper) + 1.0)
                    result[tid].append(SampledFrame(
                        track_id=tid, frame_id=fid, bbox=bbox,
                        score=score, crop=upper,
                    ))
                done += 1
                if done % 500 == 0:
                    print(f"    sampler: {done}/{total_wanted} frames processed",
                          flush=True)
            else:
                if not cap.grab():
                    break
            fid += 1
    finally:
        cap.release()

    # Per-track: keep top-k by sharpness-weighted score; preserve temporal order
    for tid in list(result.keys()):
        frames = result[tid]
        if len(frames) > k:
            frames = sorted(frames, key=lambda f: -f.score)[:k]
        frames.sort(key=lambda f: f.frame_id)
        result[tid] = frames

    return result
