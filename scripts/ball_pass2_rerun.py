"""Re-run ball detection pass 2 with increased max_interpolation_gap.

Diagnostic script: checks if YOLO can find the ball in gap frames
by cropping around the interpolated position.
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalinsight.tracking.ball_detector import BallDetector
from goalinsight.tracking.ball_tracker import BallTracker
from goalinsight.tracking.ball_pipeline import _interpolate_tracked_position


def main():
    tracking_dir = Path("output/full_pipeline/full/tracking")
    video_path = Path("data/raw_videos/football_sunday_full.mp4")

    with open(tracking_dir / "ball_debug_log.json") as f:
        debug_log = json.load(f)

    with open("configs/default.yaml") as f:
        config = yaml.safe_load(f)

    max_gap_old = 30
    max_gap_new = 300
    crop_size = 300
    enlarge_to = 640
    batch_size = 32

    # Reconstruct detections from debug_log
    all_ball_detections: dict[int, list[dict]] = {}
    for fidx_str, entry in debug_log.items():
        dets = entry.get("detections", [])
        if dets:
            all_ball_detections[int(fidx_str)] = dets

    # Build preliminary tracker
    ball_tracking_config = config.get("ball_tracking", {})
    prelim_tracker = BallTracker(ball_tracking_config)
    prelim_positions: dict[int, tuple[float, float]] = {}

    all_frames = sorted(int(k) for k in debug_log.keys())
    for fidx in all_frames:
        dets = all_ball_detections.get(fidx, [])
        tracks = prelim_tracker.update(dets)
        for t in tracks:
            if not t.get("predicted", False):
                prelim_positions[fidx] = tuple(t["center"])
                break

    print(f"Loaded {len(all_ball_detections)} frames with detections")
    print(f"Preliminary tracker: {len(prelim_positions)} tracked positions")

    # Find NEW pass 2 frames (gap > old, <= new)
    pass2_tasks: list[tuple[int, tuple[float, float]]] = []
    for fidx in all_frames:
        if fidx in prelim_positions:
            continue
        old = _interpolate_tracked_position(fidx, prelim_positions, max_gap_old)
        if old is not None:
            continue
        new = _interpolate_tracked_position(fidx, prelim_positions, max_gap_new)
        if new is not None:
            pass2_tasks.append((fidx, new))

    print(f"\nNew pass 2 tasks: {len(pass2_tasks)} frames")
    if not pass2_tasks:
        print("Nothing to do!")
        return

    # Show gap info
    gaps = []
    prev_f = None
    for fidx, _ in pass2_tasks:
        if prev_f is not None and fidx - prev_f > 10:
            gaps.append((prev_f, fidx))
        prev_f = fidx
    first_f = pass2_tasks[0][0]
    last_f = pass2_tasks[-1][0]
    print(f"  Frame range: {first_f} - {last_f}")

    # Init detector
    ball_config = config.get("ball_detection", {})
    ball_detector = BallDetector(ball_config)

    # Process
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    new_dets = 0
    det_frames = []

    num_batches = (len(pass2_tasks) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(num_batches), desc="Pass 2 rerun"):
        batch = pass2_tasks[batch_idx * batch_size : (batch_idx + 1) * batch_size]

        images = []
        metas = []
        fidxs = []

        for fidx, center in batch:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            if not ret:
                continue
            img, meta = ball_detector.prepare_crop(frame, center, crop_size, enlarge_to)
            if img is not None:
                images.append(img)
                metas.append(meta)
                fidxs.append(fidx)

        if not images:
            continue

        results = ball_detector.detect_crop_batch(images, metas, enlarge_to=enlarge_to)

        for fidx, crop_dets in zip(fidxs, results):
            crop_dets = ball_detector.filter_by_size(crop_dets)
            if crop_dets:
                new_dets += 1
                best = max(crop_dets, key=lambda d: d["confidence"])
                cx, cy = best["center"]
                det_frames.append((fidx, cx, cy, best["confidence"]))

    cap.release()

    print(f"\n=== Results ===")
    print(f"Scanned {len(pass2_tasks)} frames, found ball in {new_dets} frames")

    if det_frames:
        print(f"\nNew detections:")
        for fidx, cx, cy, conf in det_frames:
            t = fidx / fps
            m = int(t // 60)
            s = t % 60
            print(f"  frame={fidx} ({m}:{s:04.1f}) pixel=({cx:.0f},{cy:.0f}) conf={conf:.3f}")
    else:
        print("\nNo new ball detections found in the gap frames.")
        print("The ball is likely not visible (inside goal net / out of frame).")


if __name__ == "__main__":
    main()
