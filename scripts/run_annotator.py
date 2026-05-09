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

Usage:
    python scripts/run_annotator.py data/raw_videos/clip.mp4 \\
        --port 7860 --output output/annotations
    # For remote servers: ssh -L 7860:localhost:7860 user@server
"""

import argparse
import sys
from pathlib import Path

# Editable install should make this unnecessary, but support direct execution too.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from goalinsight.annotation import AnchorAnnotator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch annotation UI for manual pitch keypoint annotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Outputs frame_<idx>_raw.jpg and frame_<idx>_all_points.json that are "
            "directly consumable by the PnLCalib finetune dataloader."
        ),
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--port", "-p", type=int, default=7860,
                        help="Port for Gradio UI (default: 7860)")
    parser.add_argument("--output", "-o", default="output/annotations",
                        help="Output directory for annotations (default: output/annotations)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="Starting frame index (default: 0 = auto-load last annotated)")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio share link")

    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Video:  {video_path}")
    print(f"Output: {output_dir}")
    print(f"Port:   {args.port}")
    print()
    print("Annotation flow:")
    print("  1. Navigate to desired frame via slider.")
    print("  2. (Point mode) click image -> pick keypoint -> 'Add Annotation'.")
    print("  3. (Line mode)  click two endpoints -> pick line -> 'Add Line'.")
    print("  4. Repeat until ≥ 4 distinct world points (manual + derived).")
    print("  5. 'Compute Homography' -> auto-projects remaining 53 keypoints.")
    print("  6. 'Save & Exit' writes frame_<idx>_all_points.json for finetune.")
    print("-" * 60)

    annotator = AnchorAnnotator(annotations_dir=str(output_dir))
    anchor_frame, H0 = annotator.launch_ui(
        str(video_path),
        port=args.port,
        start_frame=args.start_frame,
        share=args.share,
    )

    if H0 is not None:
        print("\nAnnotation complete!")
        print(f"Anchor frame: {anchor_frame}")
        print(f"Annotations:  {output_dir}/{video_path.stem}/")
    else:
        print("\nNo homography computed.")


if __name__ == "__main__":
    main()
