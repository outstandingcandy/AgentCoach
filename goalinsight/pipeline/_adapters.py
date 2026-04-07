"""Pipeline stage adapters — bridge between Pipeline framework and business modules."""

import json
from pathlib import Path
from typing import Any

from ._base import Stage, PipelineContext
from ._registry import register_stage


@register_stage
class ShotDetectionStage(Stage):
    name = "shot_detection"
    description = "Shot Detection & Segmentation"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..preprocessing.runner import run_shot_detection

        out = ctx.stage_dir(self.name)
        return run_shot_detection(ctx.video_path, out, ctx.config)

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "shot_boundaries.json").exists()


@register_stage
class FieldRegistrationStage(Stage):
    name = "field_registration"
    description = "Field Registration"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..utils.config import get_default_config, get_process_fps_from_config

        out = ctx.stage_dir(self.name)
        out.mkdir(parents=True, exist_ok=True)
        vis_dir = out / "visualizations"
        vis_dir.mkdir(exist_ok=True)

        config = ctx.config if ctx.config is not None else get_default_config()
        process_fps = get_process_fps_from_config(config)

        fr_config = config.get("field_registration", {})
        backend = fr_config.get("backend", "pnlcalib")
        print(f"Field Registration: Using {backend} backend")

        if backend == "nbjw":
            from ..field_registration.pnlcalib_runner import _run_stage1_nbjw
            return _run_stage1_nbjw(ctx.video_path, out, vis_dir, config, process_fps)
        elif backend == "broadtrack":
            from ..field_registration.broadtrack_runner import run_stage1_broadtrack
            return run_stage1_broadtrack(ctx.video_path, out, vis_dir, config, process_fps)
        elif backend == "physical":
            from ..field_registration.physical_runner import run_stage1_physical
            return run_stage1_physical(ctx.video_path, out, vis_dir, config, process_fps)
        elif backend == "homography":
            from ..field_registration.homography_runner import run_stage1_homography
            return run_stage1_homography(ctx.video_path, out, vis_dir, config, process_fps)
        else:
            from ..field_registration.pnlcalib_runner import _run_stage1_pnlcalib
            return _run_stage1_pnlcalib(ctx.video_path, out, vis_dir, config, process_fps)

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "homographies.pkl").exists()


@register_stage
class TrackingStage(Stage):
    name = "tracking"
    description = "Tracking and Identification"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..tracking.orchestrator import run_tracking

        out = ctx.stage_dir(self.name)
        cal_dir = ctx.stage_dirs.get("field_registration")
        if cal_dir is not None and not cal_dir.exists():
            cal_dir = None
        if cal_dir is None:
            print("  Warning: No field registration output found, running without calibration")
        return run_tracking(ctx.video_path, out, cal_dir, ctx.config)

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "tracks.json").exists()


@register_stage
class PostProcessingStage(Stage):
    name = "post_processing"
    description = "Post-processing Refinement"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..refinement import run_refinement

        tracking_dir = ctx.stage_dirs.get("tracking")
        if tracking_dir is None or not tracking_dir.exists():
            raise RuntimeError("Post-processing requires tracking output")
        out = ctx.stage_dir(self.name)
        return run_refinement(tracking_dir, out, ctx.config)

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "tracks_refined.json").exists()


@register_stage
class GoalDetectionStage(Stage):
    name = "goal_detection"
    description = "Goal Detection"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..goal_detection import detect_goals

        tracking_dir = ctx.stage_dirs.get("tracking")
        if tracking_dir is None:
            tracking_dir = ctx.output_dir / "tracking"

        ball_path = tracking_dir / "ball_tracks.json"
        if not ball_path.exists():
            print("  Warning: ball_tracks.json not found, skipping goal detection")
            return {"goals_detected": 0, "goals": []}

        with open(ball_path) as f:
            ball_tracks = json.load(f)

        cal_dir = ctx.stage_dirs.get("field_registration")
        camera_poses = None
        if cal_dir and (cal_dir / "camera_poses.json").exists():
            with open(cal_dir / "camera_poses.json") as f:
                camera_poses = json.load(f)

        goals = detect_goals(ball_tracks=ball_tracks, camera_poses=camera_poses)

        out = ctx.stage_dir(self.name)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "goals.json", "w") as f:
            json.dump(goals, f, indent=2, default=str)

        return {"goals_detected": len(goals), "goals": goals}
