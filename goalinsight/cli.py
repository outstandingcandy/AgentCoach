#!/usr/bin/env python3
"""GoalInsight CLI entry point.

Run the analysis pipeline with configurable stages.

Available stages:
- shot_detection: Shot boundary detection & video segmentation
- field_registration: Camera calibration (PnLCalib, BroadTrack, Physical, etc.)
- tracking: Player/ball detection, tracking, ReID, and team classification
- post_processing: Temporal consistency and tracklet merging
- event_detection: Possession, passes, shots, goals, carries, tackles
- highlights: Highlight clip generation
- video_enhancement: Upscaling & frame interpolation (requires video2x)
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from goalinsight.pipeline import Pipeline
from goalinsight.utils.config import get_default_config, load_config, merge_configs


def main():
    parser = argparse.ArgumentParser(
        description="Run GoalInsight pipeline"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for all stages"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--keypoint-model",
        type=str,
        default=None,
        help="Path to fine-tuned keypoint model (overrides config)"
    )
    parser.add_argument(
        "--stages",
        type=str,
        default=None,
        help="Comma-separated stages to run (default: from config or field_registration,tracking,post_processing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages whose output already exists"
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not create timestamped subdirectory"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom run name (used instead of timestamp)"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    base_output_dir = Path(args.output)

    # Create timestamped subdirectory
    if args.no_timestamp:
        output_dir = base_output_dir
    elif args.run_name:
        output_dir = base_output_dir / args.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_output_dir / f"run_{timestamp}"

    print(f"Output directory: {output_dir}")

    # Load configuration
    config = get_default_config()
    if args.config:
        user_config = load_config(args.config)
        config = merge_configs(config, user_config)

    # Override keypoint model path if specified
    if args.keypoint_model:
        if "field_registration" not in config:
            config["field_registration"] = {}
        if "pnlcalib" not in config["field_registration"]:
            config["field_registration"]["pnlcalib"] = {}
        config["field_registration"]["pnlcalib"]["keypoint_model_path"] = args.keypoint_model

    # Build and run pipeline
    if args.stages:
        stage_names = [s.strip() for s in args.stages.split(",")]
        pipeline = Pipeline.from_stage_names(stage_names, config)
    else:
        pipeline = Pipeline(config)

    run_metadata = pipeline.run(
        video_path=video_path,
        output_dir=output_dir,
        skip_existing=args.skip_existing,
    )

    # Print completion summary
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    print("Stages run:")
    for stage_name in run_metadata["stages_run"]:
        stage_dir = output_dir / stage_name
        if stage_dir.exists():
            print(f"  {stage_name}: {stage_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
