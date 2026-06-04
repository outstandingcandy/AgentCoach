#!/usr/bin/env python3
"""Evaluate a fine-tuned PnLCalib keypoint model on its own training set.

What it answers:

1. **Recall:** for each annotated PnLCalib id, did the model fire on
   that channel? (If a channel was supervised but doesn't activate,
   training failed for that id.)
2. **Pixel error:** when the model fires on an annotated channel, how
   far from the ground-truth pixel is the predicted location?
3. **Phantom activations:** which channels fire on the image even
   though that id was *not* in the ground-truth annotation? These are
   the "channel-id confusion" cases that broke frame 240.

Usage:

    python scripts/eval_finetune_on_train.py \\
        --model data/finetuned_models/run_20260604_080144/models/best_model.pt \\
        --annotations_dir output/annotations/kids_soccer_union \\
        --videos_root data/raw_videos \\
        --conf_threshold 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_video_path(annot: dict, videos_root: Path) -> Path:
    """The all_points json carries video_name; raw frame may already be cached."""
    video_name = annot.get("video_name", "")
    candidates = [
        videos_root / f"{video_name}.mp4",
        videos_root / video_name / f"{video_name}.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"video {video_name} not found under {videos_root}")


def _load_frame(video_path: Path, frame_idx: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help="Fine-tuned PnLCalib KP model (.pt)")
    parser.add_argument("--annotations_dir", required=True,
                        help="Directory with frame_*_all_points.json + frame_*_raw.jpg")
    parser.add_argument("--videos_root", default="data/raw_videos",
                        help="Fallback to read frames if raw.jpg is missing")
    parser.add_argument("--conf_threshold", type=float, default=0.15)
    parser.add_argument("--match_radius_px", type=float, default=30.0,
                        help="A detection counts as recovering an annotation "
                             "only if it lands within this many pixels.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    annot_dir = Path(args.annotations_dir)
    videos_root = Path(args.videos_root)

    # Lazy import — keep `--help` snappy.
    from goalinsight.field_registration.keypoint_detector import KeypointDetector
    from goalinsight.annotation.pitch.keypoints import (
        PITCH_POINT_TO_PNLCALIB_ID, PITCH_POINTS,
    )
    pnl_to_name = {pnl: name for name, pnl in PITCH_POINT_TO_PNLCALIB_ID.items()}

    kp_cfg = {
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": "SV_kp",
            "model_path": args.model,
            "confidence_threshold": args.conf_threshold,
        },
    }
    detector = KeypointDetector(kp_cfg)
    detector.load_model()

    annot_files = sorted(annot_dir.glob("frame_*_all_points.json"))
    if not annot_files:
        raise SystemExit(f"No annotation files under {annot_dir}")
    logger.info("evaluating on %d frames", len(annot_files))

    # Aggregates
    recall_by_id: dict[int, list[bool]] = defaultdict(list)   # pnl_id -> [hit?]
    pixel_err_by_id: dict[int, list[float]] = defaultdict(list)
    phantom_by_id: dict[int, int] = defaultdict(int)
    frame_summaries = []

    for af in annot_files:
        with open(af) as f:
            annot = json.load(f)
        fid = int(annot["frame_idx"])
        gt_points = annot.get("all_points", [])
        gt_by_id: dict[int, tuple[float, float]] = {}
        for p in gt_points:
            pid = p.get("pnlcalib_id", -1)
            if pid < 0:
                continue
            gt_by_id[pid] = tuple(p["pixel"])

        # Load image
        raw_jpg = af.with_name(af.name.replace("_all_points.json", "_raw.jpg"))
        if raw_jpg.exists():
            frame = cv2.imread(str(raw_jpg))
        else:
            video = _resolve_video_path(annot, videos_root)
            frame = _load_frame(video, fid)
            if frame is None:
                logger.warning("can't read frame %d from %s", fid, video)
                continue

        # Detect
        kps = detector.detect(frame, convert_to_soccernet=False)

        # Match each annotation to nearest detection of same id (if any)
        matched_ids = set()
        for pid, (gx, gy) in gt_by_id.items():
            same_id = [k for k in kps if k["id"] == pid]
            if not same_id:
                recall_by_id[pid].append(False)
                continue
            # Pick closest in pixel space
            same_id.sort(key=lambda k: (k["x"] - gx) ** 2 + (k["y"] - gy) ** 2)
            best = same_id[0]
            d = float(np.hypot(best["x"] - gx, best["y"] - gy))
            within = d <= args.match_radius_px
            recall_by_id[pid].append(within)
            if within:
                pixel_err_by_id[pid].append(d)
                matched_ids.add(pid)

        # Phantom activations: ids the model fired on but weren't in ground truth.
        # Note: detector returns at most one detection per channel by default.
        for k in kps:
            pid = k["id"]
            if pid not in gt_by_id:
                phantom_by_id[pid] += 1

        n_gt = len(gt_by_id)
        n_recovered = sum(
            1 for pid in gt_by_id if recall_by_id[pid] and recall_by_id[pid][-1]
        )
        n_phantom = sum(1 for k in kps if k["id"] not in gt_by_id)
        frame_summaries.append({
            "frame_idx": fid,
            "video": annot.get("video_name", ""),
            "gt_count": n_gt,
            "recovered": n_recovered,
            "phantom": n_phantom,
            "total_detected": len(kps),
        })

    # ----- Report -------------------------------------------------------
    print("\n=== Per-frame summary ===")
    print(f"{'frame':>7}  {'video':<24}  {'gt':>3}  {'rec':>3}  {'phantom':>7}  {'detect':>6}")
    for s in frame_summaries:
        print(f"  {s['frame_idx']:>5}  {s['video']:<24}  {s['gt_count']:>3}  "
              f"{s['recovered']:>3}  {s['phantom']:>7}  {s['total_detected']:>6}")

    print("\n=== Per-id recall (annotated channels only) ===")
    print(f"{'pnl_id':>6}  {'name':<35}  {'hits/total':>10}  {'mean_err_px':>12}")
    rows = []
    for pid in sorted(recall_by_id):
        hits = sum(recall_by_id[pid])
        total = len(recall_by_id[pid])
        errs = pixel_err_by_id.get(pid, [])
        name = pnl_to_name.get(pid, "?")
        rows.append((pid, name, hits, total,
                     float(np.mean(errs)) if errs else float("nan")))
    rows.sort(key=lambda r: (r[2] / max(r[3], 1), r[0]))   # worst recall first
    for pid, name, hits, total, mean_err in rows:
        err_str = f"{mean_err:>9.1f}" if mean_err == mean_err else "       —"
        print(f"  {pid:>4}  {name:<35}  {hits:>4}/{total:<4}  {err_str}")

    print("\n=== Phantom activations (model fired on channel without ground truth) ===")
    if not phantom_by_id:
        print("  (none — model only activates supervised channels)")
    else:
        # Filter: only show ids that NEVER had ground truth across the whole set
        gt_ids_anywhere = set(recall_by_id)
        purely_phantom = {pid: n for pid, n in phantom_by_id.items()
                          if pid not in gt_ids_anywhere}
        partial_phantom = {pid: n for pid, n in phantom_by_id.items()
                           if pid in gt_ids_anywhere}

        print(f"  Channels with ZERO ground truth that fire anyway "
              f"(true cross-id confusion):")
        for pid in sorted(purely_phantom, key=lambda p: -purely_phantom[p]):
            n = purely_phantom[pid]
            name = pnl_to_name.get(pid, "?")
            print(f"    pnl_id={pid:>2}  {name:<35}  fires in {n:>2} frame(s)")

        if partial_phantom:
            print(f"\n  Channels that fire on frames where they weren't annotated "
                  f"(but were elsewhere — could be ok):")
            for pid in sorted(partial_phantom, key=lambda p: -partial_phantom[p]):
                n = partial_phantom[pid]
                name = pnl_to_name.get(pid, "?")
                print(f"    pnl_id={pid:>2}  {name:<35}  fires in {n:>2} frame(s)")

    # Aggregate stats
    all_hits = sum(sum(v) for v in recall_by_id.values())
    all_total = sum(len(v) for v in recall_by_id.values())
    all_err = [e for v in pixel_err_by_id.values() for e in v]
    print(f"\nOverall recall: {all_hits}/{all_total} = "
          f"{100 * all_hits / max(all_total, 1):.1f}%")
    if all_err:
        print(f"Overall mean pixel error (matched detections): "
              f"{np.mean(all_err):.1f}px, median {np.median(all_err):.1f}px")

    return 0


if __name__ == "__main__":
    sys.exit(main())
