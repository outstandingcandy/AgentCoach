"""Pipeline stage adapters — bridge between Pipeline framework and business modules."""

import json
import logging
from pathlib import Path
from typing import Any

from ._base import Stage, PipelineContext
from ._registry import register_stage

logger = logging.getLogger(__name__)


def _default_vis_stride(config: dict[str, Any] | None) -> int:
    """Pipeline-wide default vis stride: top-level ``sample.stride``.

    Per-stage ``<stage>.vis_frame_stride`` takes precedence if set. The
    point of having one source of truth is so vis frames line up with
    the frames each stage actually computed — picking a vis stride that
    isn't a multiple of the compute sample causes overlay holes (frame
    appears in vis but no detection / pose was computed for it).
    """
    if not config:
        return 1
    sample = config.get("sample") or {}
    return int(sample.get("stride", 1))


def _should_run_remote(stage_name: str, config: dict[str, Any] | None) -> bool:
    """Honor the ``execution.remote_stages`` opt-in list in config.

    Returns False (= run locally) unless the stage explicitly appears
    there. Default behavior is unchanged from the local-only era.
    """
    if not config:
        return False
    remote = (config.get("execution") or {}).get("remote_stages") or []
    return stage_name in set(remote)


def _run_remote(stage_name: str, ctx: PipelineContext) -> dict[str, Any]:
    """Submit *stage_name* to SageMaker; download products into ctx.output_dir.

    Returns an empty stats dict — remote runs don't surface the rich
    in-process counters that local runners build up. The product files
    landing on disk are what downstream stages care about anyway.
    """
    from ._remote import SageMakerConfig, run_stage_remote

    sm_config = SageMakerConfig.from_config(ctx.config)
    if sm_config is None:
        raise RuntimeError(
            f"--remote-stages includes '{stage_name}' but the config "
            "lacks a complete sagemaker block (region, role_arn, "
            "image_uri, s3_bucket all required)."
        )
    logger.info("[%s] running remotely on SageMaker", stage_name)
    run_stage_remote(
        stage=stage_name,
        video_path=ctx.video_path,
        output_dir=ctx.output_dir,
        config=ctx.config,
        sm_config=sm_config,
    )
    return {"executed": "remote"}


@register_stage
class FieldRegistrationStage(Stage):
    name = "field_registration"
    description = "Field Registration"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        if _should_run_remote(self.name, ctx.config):
            return _run_remote(self.name, ctx)

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
        if _should_run_remote(self.name, ctx.config):
            return _run_remote(self.name, ctx)

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
        events_cfg = ctx.config.get("events", {}) if ctx.config else {}
        default_stride = _default_vis_stride(ctx.config)
        render_event_video(
            video_path=ctx.video_path,
            events=event_dicts,
            output_path=out / "events.mp4",
            tracking_dir=tracking_dir if tracking_dir.exists() else None,
            banner_duration_sec=events_cfg.get("banner_duration_sec", 3.0),
            vis_frame_stride=int(events_cfg.get("vis_frame_stride", default_stride)),
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

        # Pass video_enhancement config so the composer can upscale
        # source frames before composition (better quality than post-hoc).
        if ctx.config and "video_enhancement" in ctx.config:
            config = {**config, "video_enhancement": ctx.config["video_enhancement"]}

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
class TrackConsolidationStage(Stage):
    name = "track_consolidation"
    description = "Track Consolidation (Claude jersey + ReID merge)"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..track_consolidation import run_track_consolidation

        out = ctx.stage_dir(self.name)
        config = dict(ctx.config.get("track_consolidation", {}) if ctx.config else {})
        # Surface the pipeline-wide processing rate so consolidation can
        # scale its frame-count thresholds (min_track_frames etc.) the
        # same way tracking/orchestrator.py does. None means "no scaling".
        if ctx.config:
            video_cfg = ctx.config.get("video", {}) or {}
            pf = video_cfg.get("tracking_fps") or video_cfg.get("process_fps")
            if pf:
                config.setdefault("_process_fps", pf)
        result = run_track_consolidation(
            output_dir=out,
            pipeline_output_dir=ctx.output_dir,
            video_path=ctx.video_path,
            config=config,
        )

        # Render mp4 + per-frame jpg with consolidated player IDs overlaid.
        # The stage itself only emits JSON (players.json / player_map.json /
        # ...); without this hook track_consolidation/ has no visualization.
        tracking_dir = ctx.stage_dir("tracking")
        if (tracking_dir / "tracks.json").exists() and (out / "players.json").exists():
            try:
                # scripts/ isn't a python package, so import from disk.
                import importlib.util as _ilu
                from pathlib import Path as _P
                _spec = _ilu.spec_from_file_location(
                    "_render_consolidated",
                    _P(__file__).resolve().parents[2] / "scripts" / "render_consolidated_tracking.py",
                )
                _mod = _ilu.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                stride = int(config.get("vis_frame_stride", _default_vis_stride(ctx.config)))
                _mod.render_consolidated_video(
                    video_path=ctx.video_path,
                    tracking_dir=tracking_dir,
                    consolidation_dir=out,
                    output_path=out / "consolidated.mp4",
                    show_panel=True,
                    vis_frame_stride=stride,
                )
            except Exception as exc:
                # Vis failure is non-fatal — the stage's JSON outputs are
                # what downstream stages depend on.
                import logging as _lg
                _lg.getLogger(__name__).warning(
                    "  track_consolidation vis skipped: %s", exc,
                )

        return result

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "player_map.json").exists()


@register_stage
class AnnotatedVideoStage(Stage):
    name = "annotated_video"
    description = "Annotated Full-Video Render (HUD)"

    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        from ..annotated_video import run_annotated_video

        out = ctx.stage_dir(self.name)
        config = ctx.config.get("annotated_video", {}) if ctx.config else {}
        if ctx.config and "video_enhancement" in ctx.config:
            config = {**config, "video_enhancement": ctx.config["video_enhancement"]}

        return run_annotated_video(
            output_dir=out,
            pipeline_output_dir=ctx.output_dir,
            video_path=ctx.video_path,
            config=config,
        )

    def should_skip(self, ctx: PipelineContext) -> bool:
        return ctx.skip_existing and (ctx.stage_dir(self.name) / "annotated.mp4").exists()


