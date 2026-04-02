#!/usr/bin/env python3
"""Generate highlight clips from GoalInsight pipeline output.

Usage:
    python scripts/run_highlights.py --pipeline-output output/pipeline_007_2
    python scripts/run_highlights.py --pipeline-output output/pipeline_007_2 --video data/raw_videos/foo.mp4
    python scripts/run_highlights.py --pipeline-output output/pipeline_007_2 --pitch-length 91 --pitch-width 55
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from goalinsight.highlights import run_highlights


def main():
    parser = argparse.ArgumentParser(
        description="Generate highlight clips from pipeline output"
    )
    parser.add_argument(
        "--pipeline-output", "-p",
        type=str,
        required=True,
        help="Path to pipeline output directory (with stage1/, stage2/)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output directory for highlight clips (default: <pipeline-output>/highlights)",
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to source video (default: auto-resolve from pipeline metadata)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to highlight config YAML (overrides defaults)",
    )
    parser.add_argument(
        "--pitch-length",
        type=float,
        default=105.0,
        help="Pitch length in meters (default: 105.0)",
    )
    parser.add_argument(
        "--pitch-width",
        type=float,
        default=68.0,
        help="Pitch width in meters (default: 68.0)",
    )
    parser.add_argument(
        "--buildup-seconds",
        type=float,
        default=None,
        help="Max build-up lookback in seconds (default: 10)",
    )
    parser.add_argument(
        "--celebration-seconds",
        type=float,
        default=None,
        help="Celebration segment duration in seconds (default: 4)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = args.output or str(Path(args.pipeline_output) / "highlights")

    # Build config overrides
    config = {}
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f) or {}

    # CLI temporal overrides
    temporal = {}
    if args.buildup_seconds is not None:
        temporal["buildup_max_seconds"] = args.buildup_seconds
    if args.celebration_seconds is not None:
        temporal["celebration_seconds"] = args.celebration_seconds
    if temporal:
        config["temporal"] = {**config.get("temporal", {}), **temporal}

    clips = run_highlights(
        output_dir=output_dir,
        pipeline_output_dir=args.pipeline_output,
        video_path=args.video,
        config=config,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
    )

    if not clips:
        print("\nNo highlights generated (no events detected).")
        return

    print(f"\nGenerated {len(clips)} highlight clip(s):")
    for c in clips:
        print(f"  {c}")


if __name__ == "__main__":
    main()
