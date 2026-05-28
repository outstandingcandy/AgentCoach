#!/usr/bin/env python3
"""Launch the manual pitch keypoint annotator.

Produces ground-truth annotations for fine-tuning the PnLCalib HRNet keypoint
model. Outputs per frame are written under --output, in the format consumed by
`goalinsight.field_registration.pnlcalib.finetune.point_dataloader`:

    <output>/
      index.json
      <video_name>/
        frame_<idx>_raw.jpg           # Input image for finetune
        frame_<idx>_all_points.json   # Ground-truth points for finetune
        frame_<idx>.json              # Raw annotation data (reference)
        frame_<idx>.jpg               # Annotated visualization
        frame_<idx>.npy               # H0 image->world matrix (reference)

The UI is a FastAPI + plain HTML/JS app (no Gradio). Open
http://localhost:<port>/ in a browser. For remote servers:
    ssh -L 7860:localhost:7860 user@server
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalinsight.annotation import SoccerPitch, run_server


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the pitch annotator web UI",
    )
    parser.add_argument("--videos-root", default="data/raw_videos",
                        help="Directory containing source videos (default: "
                             "data/raw_videos). The first video alphabetically "
                             "is the default active video; switch via the UI.")
    parser.add_argument("--port", "-p", type=int, default=7860,
                        help="Port for the web UI (default: 7860)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host (default: 127.0.0.1; use 0.0.0.0 to expose)")
    parser.add_argument("--output", "-o", default="output/annotations",
                        help="Output directory for annotations")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index (0 = auto-load last annotated)")
    parser.add_argument("--config", "-c", default=None,
                        help="Pipeline YAML config; if it contains a 'pitch:' "
                             "section, those dimensions override FIFA defaults")

    args = parser.parse_args()

    videos_root = Path(args.videos_root)
    if not videos_root.is_dir():
        print(f"Error: --videos-root not a directory: {videos_root}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pitch: SoccerPitch | None = None
    if args.config is not None:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"Error: Config not found: {cfg_path}")
            sys.exit(1)
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
        pitch_cfg = cfg.get("pitch")
        if pitch_cfg:
            pitch = SoccerPitch.from_dict(pitch_cfg)
            print(f"Pitch:   {pitch.PITCH_LENGTH:.1f}m x {pitch.PITCH_WIDTH:.1f}m "
                  f"(goal {pitch.GOAL_LENGTH:.2f}m, from {cfg_path})")

    print(f"Videos:  {videos_root}")
    print(f"Output:  {output_dir}")
    print(f"URL:     http://{args.host}:{args.port}/")
    print()
    print("Workflow:")
    print("  1. Pick a video from the right-side Videos panel.")
    print("  2. Slider/arrows to pick a frame.")
    print("  3. Point mode: click image → pick keypoint → 'Add keypoint'.")
    print("  4. Line mode:  click two endpoints → pick line → 'Add line'.")
    print("  5. ≥6 distinct world points → 'Compute homography' auto-projects rest.")
    print("  6. Confirm derived + auto-projected points; rejected points are dropped.")
    print("  7. 'Save frame' writes frame_<idx>_all_points.json for finetune.")
    print("Shortcuts: ←/→ frame nav, p/l mode, h compute, ⌘/Ctrl+S save, ⌘/Ctrl+Z undo.")
    print("-" * 60)

    run_server(
        videos_root=str(videos_root),
        annotations_dir=str(output_dir),
        host=args.host,
        port=args.port,
        start_frame=args.start_frame,
        pitch=pitch,
    )


if __name__ == "__main__":
    main()
