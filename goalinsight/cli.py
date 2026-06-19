#!/usr/bin/env python3
"""GoalInsight CLI entry point.

Run the analysis pipeline with configurable stages.

Available stages:
- field_registration: Camera calibration (PnLCalib, BroadTrack, Physical, etc.)
- tracking: Player/ball detection, tracking, ReID, and team classification
- event_detection: Possession, passes, shots, goals, carries, tackles
- highlights: Highlight clip generation
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from goalinsight.pipeline import Pipeline
from goalinsight.utils.config import get_default_config, load_config, merge_configs
from goalinsight.utils.config_resolver import resolve_config


def main():
    # Stream INFO from goalinsight.* loggers to stderr so module-level
    # progress (chain calibration, joint optimization, etc.) is visible
    # without each script having to set up its own logging config.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            stream=sys.stderr,
        )
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
        help="Comma-separated stages to run (default: from config or field_registration,tracking)"
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
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip tracking visualization video (saves time and memory)"
    )
    parser.add_argument(
        "--remote-stages",
        type=str,
        default=None,
        help=(
            "Comma-separated stages to execute on SageMaker Processing "
            "Job instead of locally. Currently supported: "
            "field_registration, tracking. Requires the sagemaker block "
            "to be configured in the YAML config."
        ),
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

    # Override visualization setting
    if args.no_viz:
        if "output" not in config:
            config["output"] = {}
        config["output"]["save_visualizations"] = False

    # Wire remote-execution opt-in into the config so stage adapters
    # can read it without us threading a separate flag through Pipeline.
    if args.remote_stages:
        remote = [s.strip() for s in args.remote_stages.split(",") if s.strip()]
        config.setdefault("execution", {})["remote_stages"] = remote

    # Resolve resolution/fps-derived keys (focal bounds, imgsz, gap-fill
    # px thresholds, min_track_frames, …) against the source video so
    # stages see fully-derived values without each yaml hand-encoding
    # them. Legacy explicit keys still win.
    try:
        from goalinsight.field_registration._runner_base import probe_video
        n_frames, fps, width, height = probe_video(video_path)
    except RuntimeError:
        # Tolerate an unreadable / not-yet-existing video so dry-run /
        # config-validation paths still work; resolver no-ops on None.
        n_frames = fps = width = height = None
    config = resolve_config(config, width, height, fps)

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
