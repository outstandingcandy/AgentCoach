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
        cal_dir = ctx.stage_dir("field_registration")
        if not cal_dir.exists():
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

        tracking_dir = ctx.stage_dir("tracking")
        if not tracking_dir.exists():
            raise RuntimeError("Post-processing requires tracking output")
        out = ctx.stage_dir(self.name)
        return run_refinement(tracking_dir, out, ctx.config)

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "tracks_refined.json").exists()


@register_stage
class EventDetectionStage(Stage):
    name = "event_detection"
    description = "Event Detection (Possession, Passes, Shots, Goals, Carries, Tackles)"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..events import EventType, detect_events_from_dirs

        tracking_dir = ctx.stage_dir("tracking")
        cal_dir = ctx.stage_dir("field_registration")
        out = ctx.stage_dir(self.name)
        out.mkdir(parents=True, exist_ok=True)

        if not (tracking_dir / "ball_tracks.json").exists():
            print("  Warning: ball_tracks.json not found, skipping event detection")
            return {"total_events": 0}

        events = detect_events_from_dirs(
            tracking_dir=tracking_dir,
            calibration_dir=cal_dir if cal_dir.exists() else None,
            config=ctx.config,
        )

        # Write events.json
        event_dicts = [e.to_dict() for e in events]
        with open(out / "events.json", "w") as f:
            json.dump(event_dicts, f, indent=2, default=str)

        # Also write goals.json for backward compatibility
        goals = [e.to_dict() for e in events if e.event_type == EventType.GOAL]
        with open(out / "goals.json", "w") as f:
            json.dump(goals, f, indent=2, default=str)

        # Render annotated video
        from ..events.visualization import render_event_video
        render_event_video(
            video_path=ctx.video_path,
            events=event_dicts,
            output_path=out / "events.mp4",
            tracking_dir=tracking_dir if tracking_dir.exists() else None,
            banner_duration_sec=ctx.config.get("events", {}).get(
                "banner_duration_sec", 3.0
            ) if ctx.config else 3.0,
        )

        by_type = {}
        for e in events:
            by_type[e.event_type.value] = by_type.get(e.event_type.value, 0) + 1

        return {"total_events": len(events), "by_type": by_type}

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "events.json").exists()


@register_stage
class HighlightsStage(Stage):
    name = "highlights"
    description = "Highlight Clipping"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..highlights import run_highlights

        out = ctx.stage_dir(self.name)
        config = ctx.config.get("highlights", {}) if ctx.config else {}

        clips = run_highlights(
            output_dir=out,
            pipeline_output_dir=ctx.output_dir,
            video_path=ctx.video_path,
            config=config,
        )

        return {"clips": [str(c) for c in clips], "count": len(clips)}

    def should_skip(self, ctx: PipelineContext) -> bool:
        return False  # Always regenerate highlights


@register_stage
class VideoEnhancementStage(Stage):
    name = "video_enhancement"
    description = "Video Enhancement (Upscaling & Frame Interpolation)"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        config = ctx.config.get("video_enhancement", {}) if ctx.config else {}

        if not config.get("enabled", False):
            print("  Video enhancement disabled in config")
            return {"enhanced_clips": [], "count": 0}

        highlights_dir = ctx.stage_dir("highlights")
        if not highlights_dir.exists():
            print("  Warning: No highlights output found, skipping video enhancement")
            return {"enhanced_clips": [], "count": 0}

        from ..video_enhancement import run_video_enhancement

        out = ctx.stage_dir(self.name)
        clips = run_video_enhancement(
            input_dir=highlights_dir,
            output_dir=out,
            config=config,
        )

        return {"enhanced_clips": [str(c) for c in clips], "count": len(clips)}

    def should_skip(self, ctx: PipelineContext) -> bool:
        if not ctx.skip_existing:
            return False
        out = ctx.stage_dir(self.name)
        if not out.exists():
            return False
        return any(out.rglob("*.mp4"))
