"""Pick frames to annotate next based on calibration_metadata.json.

Goal: surface the frames where HRNet currently under-recalls (zero or
<4 keypoints detected). Within those, sample uniformly across the time
axis so the annotator covers the full clip rather than a clustered burst.

Output:
  - stdout list of (frame_idx, time_s, num_kp_detected) — copy/paste into
    the web annotator's frame jump.
  - optional ``--export-images <dir>``: write the actual frames to disk
    so they can be inspected before annotating.

Usage:
  python scripts/select_finetune_candidates.py output/<run_dir> \\
      --video data/raw_videos/<file.mp4> --n 30 --bucket zero
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--video", type=Path,
                    help="Path to source video (defaults to video_info.path).")
    ap.add_argument("--n", type=int, default=30,
                    help="Number of candidate frames to surface.")
    ap.add_argument("--bucket", choices=["zero", "low", "any-failed", "low-conf"],
                    default="any-failed",
                    help="zero: HRNet detected 0 keypoints. "
                         "low: HRNet detected 1-3. "
                         "any-failed: zero+low (default). "
                         "low-conf: calibrated but high rep_err.")
    ap.add_argument("--rep-err-threshold", type=float, default=10.0,
                    help="Min rep_err for low-conf bucket.")
    ap.add_argument("--export-images", type=Path,
                    help="If set, dump selected frames as JPGs to this dir.")
    args = ap.parse_args()

    fr = args.run_dir / "field_registration"
    meta = json.load(open(fr / "calibration_metadata.json"))
    vinfo = meta["video_info"]
    video_path = args.video or Path(vinfo["path"])
    fps = float(vinfo["fps"])

    candidates: list[tuple[int, dict]] = []
    for fidx_s, v in meta["frames"].items():
        fidx = int(fidx_s)
        if args.bucket == "zero":
            if not v.get("calibrated") and v.get("num_keypoints_detected") == 0:
                candidates.append((fidx, v))
        elif args.bucket == "low":
            kp = v.get("num_keypoints_detected")
            if not v.get("calibrated") and kp is not None and 1 <= kp < 4:
                candidates.append((fidx, v))
        elif args.bucket == "any-failed":
            kp = v.get("num_keypoints_detected")
            if not v.get("calibrated") and kp is not None and kp < 4:
                candidates.append((fidx, v))
        elif args.bucket == "low-conf":
            if v.get("calibrated") and v.get("reprojection_error", 0) >= args.rep_err_threshold:
                candidates.append((fidx, v))

    if not candidates:
        print(f"No frames matched bucket='{args.bucket}'. Did the run record detection_info?")
        print("(Older runs only have 'calibrated' for failed frames — re-run pnlcalib first.)")
        return

    # Uniform-temporal sampling across the available pool.
    candidates.sort(key=lambda x: x[0])
    if len(candidates) > args.n:
        idxs = np.linspace(0, len(candidates) - 1, args.n, dtype=int)
        candidates = [candidates[i] for i in idxs]

    print(f"# Bucket: {args.bucket}  ({len(candidates)} frames selected from "
          f"{sum(1 for fk,fv in meta['frames'].items() if not fv.get('calibrated'))} failed)")
    print(f"# Video: {video_path.name}  fps={fps}")
    print(f"# {'frame_idx':>10}  {'time(s)':>8}  {'kp_det':>6}  {'lines':>5}")
    for fidx, v in candidates:
        t = fidx / fps
        kp = v.get("num_keypoints_detected", -1)
        ln = v.get("num_lines_detected", -1)
        print(f"  {fidx:10d}  {t:8.2f}  {kp:6d}  {ln:5d}")

    if args.export_images:
        args.export_images.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"\nWARN: could not open video {video_path}")
            return
        for fidx, _ in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(args.export_images / f"frame_{fidx:06d}.jpg"), frame)
        cap.release()
        print(f"\nWrote {len(candidates)} images to {args.export_images}")


if __name__ == "__main__":
    main()
