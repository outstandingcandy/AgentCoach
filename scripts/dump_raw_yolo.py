"""Dump raw YOLO detections per sampled frame to JSON, separately from
the tracker output. Useful for offline inspection of what entered (and
got filtered out of) the tracker.

For each sampled frame writes one JSON capturing four stages:
  1. raw          — every YOLO detection (player + ball, conf>=detector.conf)
  2. post_conf    — players only, conf >= player_confidence_threshold (0.5)
  3. post_size    — after filter_by_size (matches orchestrator.py:666-669)
  4. post_pitch   — after _filter_by_pitch_undistorted (matches orchestrator.py:678-682)

Run:
  python scripts/dump_raw_yolo.py \\
      --video data/raw_videos/kids_soccer_clip_1250_1310.mp4 \\
      --poses output/kids_prtreid_iou_fix/field_registration/camera_poses.pkl \\
      --output output/kids_prtreid_iou_fix/yolo_raw \\
      --config configs/kids_soccer_physical.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import yaml

from goalinsight.tracking.detector import PlayerDetector
from goalinsight.tracking.unified_detector import (
    BALL_CLASS_ID, PERSON_CLASS_ID, UnifiedDetector,
)
from goalinsight.tracking.pitch_projection import _filter_by_pitch_undistorted
from goalinsight.utils.config import merge_configs


def load_config(path: Path) -> dict:
    default_path = REPO / "configs/default.yaml"
    with open(default_path) as f:
        cfg = yaml.safe_load(f) or {}
    if path != default_path:
        with open(path) as f:
            override = yaml.safe_load(f) or {}
        cfg = merge_configs(cfg, override)
    return cfg


def project_to_pitch(bbox, pose):
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)
    R, _ = cv2.Rodrigues(np.array(pose["rvec"], dtype=np.float64))
    tvec = np.array(pose["tvec"], dtype=np.float64).flatten()
    H = K @ np.column_stack([R[:, 0], R[:, 1], tvec])
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    px = (bbox[0] + bbox[2]) / 2
    py = bbox[3]
    pt = np.array([[[px, py]]], dtype=np.float64)
    pt_undist = cv2.undistortPoints(pt, K, dist, P=K).reshape(-1, 2)[0]
    ph = H_inv @ np.array([pt_undist[0], pt_undist[1], 1.0])
    if abs(ph[2]) < 1e-6:
        return None
    return [float(ph[0] / ph[2]), float(ph[1] / ph[2])]


def slim_det(d, pose=None):
    out = {
        "bbox": [round(float(x), 2) for x in d["bbox"]],
        "confidence": round(float(d["confidence"]), 4),
        "class": int(d["class"]),
        "class_name": d.get("class_name"),
    }
    if pose is not None:
        pp = project_to_pitch(d["bbox"], pose)
        if pp is not None:
            out["pitch_position"] = [round(pp[0], 4), round(pp[1], 4)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--poses", required=True, help="camera_poses.pkl from field_registration")
    ap.add_argument("--output", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Optional cap on number of sampled frames")
    args = ap.parse_args()

    config = load_config(Path(args.config))
    sample_stride = int(config.get("sample", {}).get("stride", 3))
    unified_cfg = config.get("unified_detection", {}) or {}
    player_conf_thresh = float(unified_cfg.get(
        "player_confidence_threshold",
        config.get("detection", {}).get("confidence_threshold", 0.5),
    ))
    fr_phys = config.get("field_registration", {}).get("physical", {})
    pitch_half_l = float(fr_phys.get("pitch_length", 105.0)) / 2
    pitch_half_w = float(fr_phys.get("pitch_width", 68.0)) / 2

    out_dir = Path(args.output)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    with open(args.poses, "rb") as f:
        camera_poses = pickle.load(f)

    detector = UnifiedDetector({
        "model": unified_cfg.get("model", "yolov8x"),
        "model_path": unified_cfg.get("model_path"),
        "confidence_threshold": float(unified_cfg.get("confidence_threshold", 0.3)),
        "iou_threshold": float(unified_cfg.get("iou_threshold", 0.45)),
        "imgsz": int(unified_cfg.get("imgsz", 1920)),
        "classes": [PERSON_CLASS_ID, BALL_CLASS_ID],
    })
    detector.load_model()
    pdet = PlayerDetector({})

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {args.video}")
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = list(range(0, total_video_frames, sample_stride))
    if args.max_frames is not None:
        sample_frames = sample_frames[: args.max_frames]

    print(f"video: {args.video}")
    print(f"  total frames: {total_video_frames}  sample stride: {sample_stride}")
    print(f"  sampled: {len(sample_frames)}")
    print(f"  player_confidence_threshold: {player_conf_thresh}")
    print(f"  pitch dims: {pitch_half_l*2:.2f} x {pitch_half_w*2:.2f}")
    print(f"  output: {out_dir}")

    summary_rows = []
    for fidx in tqdm(sample_frames, desc="dump"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        all_dets = detector.detect_batch([frame])[0]
        players_raw, balls_raw = UnifiedDetector.split_by_class(all_dets)

        # Stage 2: post-conf for players
        players_post_conf = [d for d in players_raw if d["confidence"] >= player_conf_thresh]

        # Stage 3: post-size
        players_post_size = pdet.filter_by_size(
            players_post_conf, min_height=25, max_height=350,
            min_aspect_ratio=0.25, max_aspect_ratio=1.0,
        )

        # Stage 4: post-pitch
        pose = camera_poses.get(fidx)
        if pose is not None:
            players_post_pitch = _filter_by_pitch_undistorted(
                players_post_size, pose, margin=5.0,
                pitch_half_length=pitch_half_l,
                pitch_half_width=pitch_half_w,
            )
        else:
            players_post_pitch = players_post_size

        record = {
            "frame_index": fidx,
            "image_size": [frame.shape[1], frame.shape[0]],
            "has_camera_pose": pose is not None,
            "counts": {
                "raw_players": len(players_raw),
                "raw_balls": len(balls_raw),
                "post_conf_players": len(players_post_conf),
                "post_size_players": len(players_post_size),
                "post_pitch_players": len(players_post_pitch),
            },
            "players_raw": [slim_det(d, pose) for d in players_raw],
            "players_post_conf": [slim_det(d, pose) for d in players_post_conf],
            "players_post_size": [slim_det(d, pose) for d in players_post_size],
            "players_post_pitch": [slim_det(d, pose) for d in players_post_pitch],
            "balls_raw": [slim_det(d, pose) for d in balls_raw],
        }
        with open(frames_dir / f"frame_{fidx:06d}.json", "w") as f:
            json.dump(record, f, indent=1)

        summary_rows.append({
            "frame_index": fidx,
            **record["counts"],
            "has_camera_pose": pose is not None,
        })

    cap.release()

    summary = {
        "video": args.video,
        "config": str(args.config),
        "sample_stride": sample_stride,
        "player_confidence_threshold": player_conf_thresh,
        "pitch_half_length": pitch_half_l,
        "pitch_half_width": pitch_half_w,
        "frames": summary_rows,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\ndone. {len(summary_rows)} frames written to {frames_dir}")
    print(f"summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
