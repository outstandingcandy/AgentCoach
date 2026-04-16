#!/usr/bin/env python3
"""Create a picture-in-picture comparison video.

Enhanced highlight plays full-screen; the raw source plays as a small window
in the bottom-right corner with matching segment timing (including slow-motion
replay).  When the raw segments end before the enhanced clip, the raw PiP
freezes on its last frame.

Usage:
    python tools/make_comparison.py \
        --enhanced output/full_pipeline/full/video_enhancement/goal_highlight/goal_13427_left.mp4 \
        --raw data/raw_videos/football_sunday_full.mp4 \
        --output output/comparison_goal_13427.mp4

    # With labels:
    python tools/make_comparison.py \
        --enhanced output/full_pipeline/full/video_enhancement/goal_highlight/goal_13427_left.mp4 \
        --raw data/raw_videos/football_sunday_full.mp4 \
        --output output/comparison_goal_13427.mp4 \
        --label
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


def probe_video(path: str) -> dict:
    """Get video stream info via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,duration,r_frame_rate,nb_frames",
        "-of", "json", path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)["streams"][0]
    num, den = map(int, info["r_frame_rate"].split("/"))
    info["fps"] = num / den
    info["width"] = int(info["width"])
    info["height"] = int(info["height"])
    info["duration"] = float(info["duration"])
    info["nb_frames"] = int(info["nb_frames"])
    return info


def find_highlight_metadata(enhanced_path: str) -> "Path | None":
    """Find the highlight metadata JSON by walking up to the highlights stage."""
    p = Path(enhanced_path)
    parts = list(p.parts)
    for i, part in enumerate(parts):
        if part == "video_enhancement":
            parts[i] = "highlights"
            meta = Path(*parts).with_suffix(".json")
            if meta.exists():
                return meta
    meta = p.with_suffix(".json")
    if meta.exists():
        return meta
    return None


def find_raw_video(enhanced_path: str) -> "Path | None":
    """Try to find the raw video from pipeline_stats.json."""
    p = Path(enhanced_path)
    candidate = p.parent
    for _ in range(6):
        candidate = candidate.parent
        for run_dir in sorted(candidate.glob("run_*")):
            stats = run_dir / "pipeline_stats.json"
            if stats.exists():
                data = json.loads(stats.read_text())
                video = data.get("video_path")
                if video and Path(video).exists():
                    return Path(video)
    return None


def reconstruct_segments(meta: dict) -> list[dict]:
    """Reconstruct the 4-segment plan from highlight metadata.

    The metadata JSON maps source_frame_id → {segment, view_type, ...}.
    Segments in order: buildup → strike → celebration → replay.
    Strike and replay share the same source frames (replay overwrites strike
    in the metadata dict), so we detect 'replay' entries and infer the strike
    range.

    Returns a list of segment dicts:
        [{"name": str, "start": int, "end": int, "speed": float}, ...]
    """
    # Group consecutive frames by segment name
    ordered = []
    prev_seg = None
    for k in sorted(meta.keys(), key=int):
        seg_name = meta[k]["segment"]
        fid = int(k)
        if seg_name != prev_seg:
            ordered.append({"name": seg_name, "start": fid, "end": fid})
            prev_seg = seg_name
        else:
            ordered[-1]["end"] = fid

    segments = []
    for seg in ordered:
        name = seg["name"]
        if name == "buildup":
            segments.append({
                "name": "buildup",
                "start": seg["start"],
                "end": seg["end"],
                "speed": 1.0,
            })
        elif name == "replay":
            # This range was used for both strike (speed=1.0) and replay (speed=0.4)
            # The scorer_analyzer puts strike before celebration, replay after.
            # Insert strike first, replay will be appended after celebration.
            segments.append({
                "name": "strike",
                "start": seg["start"],
                "end": seg["end"],
                "speed": 1.0,
            })
            # Save replay info for later
            _replay = {
                "name": "replay",
                "start": seg["start"],
                "end": seg["end"],
                "speed": 0.4,
            }
        elif name == "celebration":
            segments.append({
                "name": "celebration",
                "start": seg["start"],
                "end": seg["end"],
                "speed": 1.0,
            })

    # Append replay at the end (after celebration)
    if "_replay" in dir() or _replay:  # noqa: F821
        segments.append(_replay)

    return segments


def read_frame(cap: cv2.VideoCapture, frame_id: int) -> "np.ndarray | None":
    """Seek to a specific frame and read it."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ret, frame = cap.read()
    return frame if ret else None


def read_frames_range(cap: cv2.VideoCapture, start: int, end: int) -> list[np.ndarray]:
    """Read frames [start, end] inclusive from the capture."""
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(end - start + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames


def slowmo_interpolate(frames: list[np.ndarray], speed: float) -> list[np.ndarray]:
    """Generate slow-motion frames via linear blending (matches SegmentComposer)."""
    n_src = len(frames)
    n_out = int(n_src / speed)
    result = []
    for i in range(n_out):
        src_pos = i * speed
        src_idx = min(int(src_pos), n_src - 1)
        frac = src_pos - int(src_pos)
        if frac < 0.01 or src_idx + 1 >= n_src:
            result.append(frames[src_idx])
        else:
            blended = cv2.addWeighted(
                frames[src_idx], 1.0 - frac,
                frames[src_idx + 1], frac, 0,
            )
            result.append(blended)
    return result


def main():
    parser = argparse.ArgumentParser(description="Create PiP comparison video")
    parser.add_argument("--enhanced", required=True, help="Path to enhanced highlight video")
    parser.add_argument("--raw", default=None, help="Path to raw source video (auto-detected if omitted)")
    parser.add_argument("--output", required=True, help="Output comparison video path")
    parser.add_argument("--label", action="store_true", help="Add text labels on PiP window")
    parser.add_argument("--pip-scale", type=float, default=0.3,
                        help="PiP window size relative to main video (default 0.3)")
    parser.add_argument("--pip-margin", type=int, default=20,
                        help="PiP margin from edges in pixels (default 20)")
    parser.add_argument("--pip-border", type=int, default=2,
                        help="PiP border thickness in pixels (default 2)")
    args = parser.parse_args()

    enhanced_path = args.enhanced
    if not Path(enhanced_path).exists():
        print(f"Error: enhanced video not found: {enhanced_path}", file=sys.stderr)
        sys.exit(1)

    # Find raw video
    raw_path = args.raw
    if raw_path is None:
        found = find_raw_video(enhanced_path)
        if found is None:
            print("Error: could not auto-detect raw video. Use --raw.", file=sys.stderr)
            sys.exit(1)
        raw_path = str(found)
        print(f"Auto-detected raw video: {raw_path}")

    if not Path(raw_path).exists():
        print(f"Error: raw video not found: {raw_path}", file=sys.stderr)
        sys.exit(1)

    # Find highlight metadata
    meta_path = find_highlight_metadata(enhanced_path)
    if meta_path is None:
        print("Error: could not find highlight metadata JSON", file=sys.stderr)
        sys.exit(1)
    print(f"Highlight metadata: {meta_path}")

    meta = json.loads(meta_path.read_text())

    # Reconstruct segment plan
    segments = reconstruct_segments(meta)
    print("Segment plan:")
    for seg in segments:
        n_src = seg["end"] - seg["start"] + 1
        if seg["speed"] < 1.0:
            n_out = int(n_src / seg["speed"])
        else:
            n_out = n_src
        print(f"  {seg['name']:12s}: frames {seg['start']}-{seg['end']} "
              f"({n_src} src → {n_out} out, speed={seg['speed']})")

    # Probe videos
    enh_info = probe_video(enhanced_path)
    raw_info = probe_video(raw_path)
    print(f"\nEnhanced: {enh_info['width']}x{enh_info['height']} "
          f"@ {enh_info['fps']}fps, {enh_info['duration']:.2f}s, {enh_info['nb_frames']} frames")
    print(f"Raw:      {raw_info['width']}x{raw_info['height']} "
          f"@ {raw_info['fps']}fps, {raw_info['duration']:.2f}s")

    out_fps = enh_info["fps"]
    enh_total_frames = enh_info["nb_frames"]
    enh_w, enh_h = enh_info["width"], enh_info["height"]

    # Build time-synced raw frames, padded with last-frame freeze to match enhanced length
    print(f"\n[1/2] Building time-synced raw frames...")
    raw_cap = cv2.VideoCapture(raw_path)
    if not raw_cap.isOpened():
        print("Error: cannot open raw video", file=sys.stderr)
        sys.exit(1)

    raw_frames: list[np.ndarray] = []
    for seg in segments:
        # Skip replay segment for the raw PiP — freeze on last frame instead
        if seg["name"] == "replay":
            print(f"  {seg['name']:12s}: skipped (freeze on last frame)")
            continue

        src_frames = read_frames_range(raw_cap, seg["start"], seg["end"])
        print(f"  {seg['name']:12s}: read {len(src_frames)} source frames"
              f" → {len(src_frames)} output frames")
        raw_frames.extend(src_frames)

    raw_cap.release()

    # Pad with last frame freeze if raw is shorter than enhanced
    raw_count = len(raw_frames)
    if raw_count < enh_total_frames:
        pad_count = enh_total_frames - raw_count
        print(f"  Padding: {pad_count} freeze frames (last frame held)")
        last_frame = raw_frames[-1]
        for _ in range(pad_count):
            raw_frames.append(last_frame)
    print(f"  Total: {len(raw_frames)} raw frames to match {enh_total_frames} enhanced frames")

    # Composite: enhanced full-screen + raw PiP in bottom-right
    print(f"\n[2/2] Compositing PiP video...")

    pip_scale = args.pip_scale
    pip_margin = args.pip_margin
    border = args.pip_border

    # PiP dimensions (preserve raw aspect ratio)
    pip_w = int(enh_w * pip_scale)
    pip_h = int(pip_w * raw_info["height"] / raw_info["width"])
    # PiP position: bottom-right with margin
    pip_x = enh_w - pip_w - pip_margin
    pip_y = enh_h - pip_h - pip_margin

    print(f"  Main: {enh_w}x{enh_h}")
    print(f"  PiP:  {pip_w}x{pip_h} at ({pip_x}, {pip_y})")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Use ffmpeg to read enhanced frames, overlay raw PiP via python+opencv
    enh_cap = cv2.VideoCapture(enhanced_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_output = str(Path(args.output).with_suffix(".tmp.mp4"))
    writer = cv2.VideoWriter(tmp_output, fourcc, out_fps, (enh_w, enh_h))

    frame_idx = 0
    while True:
        ret, enh_frame = enh_cap.read()
        if not ret:
            break
        if frame_idx >= len(raw_frames):
            break

        # Resize raw frame for PiP
        pip_frame = cv2.resize(raw_frames[frame_idx], (pip_w, pip_h),
                               interpolation=cv2.INTER_LANCZOS4)

        # Draw border around PiP
        if border > 0:
            cv2.rectangle(pip_frame, (0, 0), (pip_w - 1, pip_h - 1),
                          (255, 255, 255), border)

        # Draw "Original" label on PiP
        if args.label:
            label = "Original"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, pip_w / 1920 * 2.0)
            thickness = max(1, int(font_scale * 2))
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            lx, ly = border + 6, th + border + 6
            # Background
            cv2.rectangle(pip_frame, (lx - 4, ly - th - 4),
                          (lx + tw + 4, ly + baseline + 4), (0, 0, 0), cv2.FILLED)
            cv2.putText(pip_frame, label, (lx, ly), font, font_scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)

        # Overlay PiP onto enhanced frame
        enh_frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = pip_frame

        writer.write(enh_frame)
        frame_idx += 1

    writer.release()
    enh_cap.release()
    print(f"  Wrote {frame_idx} frames")

    # Re-encode with libx264 for better compression
    print(f"  Re-encoding with libx264...")
    encode_cmd = [
        "ffmpeg", "-y",
        "-i", tmp_output,
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-movflags", "+faststart", "-an",
        args.output,
    ]
    subprocess.run(encode_cmd, check=True)
    Path(tmp_output).unlink(missing_ok=True)

    # Verify
    out_info = probe_video(args.output)
    print(f"\nOutput: {args.output}")
    print(f"  Resolution: {out_info['width']}x{out_info['height']}")
    print(f"  Duration:   {out_info['duration']:.2f}s ({out_info['nb_frames']} frames)")
    print(f"  FPS:        {out_info['fps']}")
    print("\nDone!")


if __name__ == "__main__":
    main()
