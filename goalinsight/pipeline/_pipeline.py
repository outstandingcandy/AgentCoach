"""Pipeline orchestrator: config-driven stage execution."""

import json
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any

from ._base import PipelineCancelled, PipelineContext, Stage
from ._registry import STAGE_REGISTRY

DEFAULT_STAGES = ["field_registration", "tracking"]


class Pipeline:
    """Config-driven pipeline orchestrator."""

    def __init__(self, config: dict[str, Any]):
        from . import _adapters  # noqa: F401 — trigger registration

        self._config = config
        self._stages: list[Stage] = self._build_stages(config)

    def _build_stages(self, config: dict[str, Any]) -> list[Stage]:
        pipeline_cfg = config.get("pipeline", {})
        stage_names = pipeline_cfg.get("stages", None)

        if stage_names is None:
            stage_names = DEFAULT_STAGES

        stages = []
        for name in stage_names:
            name = str(name).strip()
            if name not in STAGE_REGISTRY:
                raise ValueError(
                    f"Unknown stage '{name}'. Available: {sorted(STAGE_REGISTRY.keys())}"
                )
            stages.append(STAGE_REGISTRY[name]())
        return stages

    @classmethod
    def from_stage_names(cls, names: list[str], config: dict[str, Any]) -> "Pipeline":
        """Create pipeline with an explicit stage list (overrides config)."""
        cfg = dict(config)
        cfg["pipeline"] = {**config.get("pipeline", {}), "stages": names}
        return cls(cfg)

    def run(
        self,
        video_path: Path,
        output_dir: Path,
        skip_existing: bool = False,
        cancel_event: Event | None = None,
    ) -> dict[str, Any]:
        """Run the full pipeline.

        ``cancel_event`` lets a caller (typically the web JobManager)
        request a graceful stop. The pipeline checks the flag at each
        stage boundary; if set, it raises ``PipelineCancelled`` after
        the in-flight stage finishes. The last completed stages are
        already on disk, so re-running with ``skip_existing=True``
        picks up where the cancelled run left off.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Probe video metadata once so stages and the resolver share a
        # consistent view of (W, H, fps, n_frames) without each one
        # re-opening a cv2 capture for the same numbers.
        from ..field_registration._runner_base import probe_video

        try:
            n_frames, fps, width, height = probe_video(video_path)
        except RuntimeError:
            # Tolerate an unreadable / not-yet-existing video here so
            # stages that don't need it (or fail with a clearer error
            # later) can still run; metadata stays None.
            n_frames = fps = width = height = None

        ctx = PipelineContext(
            video_path=video_path,
            output_dir=output_dir,
            config=self._config,
            skip_existing=skip_existing,
            cancel_event=cancel_event,
            video_fps=fps,
            video_width=width,
            video_height=height,
            frame_count=n_frames,
        )

        completed: list[str] = []
        cancelled = False
        for stage in self._stages:
            # Always register the stage dir so later stages can find it
            ctx.stage_dir(stage.name)

            if ctx.is_cancelled():
                print("=" * 60)
                print(f"PIPELINE CANCELLED — skipping remaining stages")
                print("=" * 60)
                cancelled = True
                break

            if stage.should_skip(ctx):
                print("=" * 60)
                print(f"{stage.description} [SKIPPED - output exists]")
                print("=" * 60)
                completed.append(stage.name)
                continue

            print("=" * 60)
            print(stage.description)
            print("=" * 60)

            stats = stage.run(ctx)
            ctx.stage_stats[stage.name] = stats
            completed.append(stage.name)
            print()

        run_metadata = {
            "timestamp": datetime.now().isoformat(),
            "video_path": str(video_path.absolute()),
            "video_name": video_path.name,
            "stages_run": completed,
            "cancelled": cancelled,
            "stats": ctx.stage_stats,
        }
        with open(output_dir / "pipeline_stats.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        if cancelled:
            raise PipelineCancelled(
                f"cancelled after {len(completed)} stage(s): {completed}"
            )
        return run_metadata
