#!/usr/bin/env python3
"""Run full GoalInsight pipeline (Stage 1-3) with optional fine-tuned model.

This script chains all three stages:
- Stage 1: Field Registration (keypoint/line detection -> homography)
- Stage 2: Tracking and Identification (YOLOv8 + StrongSORT + ReID)
- Stage 3: Post-processing (majority voting, tracklet merging)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from goalinsight import run_stage1, run_stage2, run_stage3
from goalinsight.utils.config import get_default_config, load_config, merge_configs


def main():
    parser = argparse.ArgumentParser(
        description="Run full GoalInsight pipeline (Stage 1-3)"
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
        default="1,2,3",
        help="Comma-separated stages to run (default: 1,2,3)"
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

    output_dir.mkdir(parents=True, exist_ok=True)
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

    # Parse stages to run
    stages_to_run = [int(s.strip()) for s in args.stages.split(",")]

    # Define output directories
    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"
    stage3_dir = output_dir / "stage3"

    all_stats = {}

    # Stage 1: Field Registration
    if 1 in stages_to_run:
        if args.skip_existing and (stage1_dir / "homographies.pkl").exists():
            print("=" * 60)
            print("STAGE 1: Field Registration [SKIPPED - output exists]")
            print("=" * 60)
        else:
            print("=" * 60)
            print("STAGE 1: Field Registration")
            print("=" * 60)
            model_path = config.get("field_registration", {}).get("pnlcalib", {}).get("keypoint_model_path")
            if model_path:
                print(f"  Using fine-tuned model: {model_path}")
            stats = run_stage1(video_path, stage1_dir, config)
            all_stats["stage1"] = stats
            print()

    # Stage 2: Tracking
    if 2 in stages_to_run:
        if args.skip_existing and (stage2_dir / "tracks.json").exists():
            print("=" * 60)
            print("STAGE 2: Tracking and Identification [SKIPPED - output exists]")
            print("=" * 60)
        else:
            print("=" * 60)
            print("STAGE 2: Tracking and Identification")
            print("=" * 60)
            # Check if Stage 1 output exists
            calibration_dir = stage1_dir if stage1_dir.exists() else None
            if calibration_dir is None:
                print("  Warning: No Stage 1 output found, running without calibration")
            stats = run_stage2(video_path, stage2_dir, calibration_dir, config)
            all_stats["stage2"] = stats
            print()

    # Stage 3: Post-processing
    if 3 in stages_to_run:
        if args.skip_existing and (stage3_dir / "tracks_refined.json").exists():
            print("=" * 60)
            print("STAGE 3: Post-processing [SKIPPED - output exists]")
            print("=" * 60)
        else:
            print("=" * 60)
            print("STAGE 3: Post-processing")
            print("=" * 60)
            # Check if Stage 2 output exists
            if not stage2_dir.exists():
                print("  Error: Stage 2 output required for Stage 3")
                return 1
            stats = run_stage3(stage2_dir, stage3_dir, config)
            all_stats["stage3"] = stats
            print()

    # Save combined statistics and run metadata
    run_metadata = {
        "timestamp": datetime.now().isoformat(),
        "video_path": str(video_path.absolute()),
        "video_name": video_path.name,
        "config_path": args.config,
        "keypoint_model": args.keypoint_model,
        "stages_run": stages_to_run,
        "stats": all_stats,
    }
    with open(output_dir / "pipeline_stats.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print()
    print("Output files:")
    if stage1_dir.exists():
        print(f"  Stage 1: {stage1_dir}/")
        print(f"    - homographies.pkl (camera calibration)")
        print(f"    - calibration_metadata.json")
    if stage2_dir.exists():
        print(f"  Stage 2: {stage2_dir}/")
        print(f"    - tracks.json (raw tracking results)")
        print(f"    - team_assignments.json")
        print(f"    - tracking.mp4 (visualization)")
    if stage3_dir.exists():
        print(f"  Stage 3: {stage3_dir}/")
        print(f"    - tracks_refined.json (post-processed tracks)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
