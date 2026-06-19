"""Pipeline base classes: Stage ABC and PipelineContext."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Any


class PipelineCancelled(RuntimeError):
    """Raised when a caller cancels the pipeline mid-run.

    Stages can't always be interrupted mid-loop (cv2/torch loops we
    don't own), so the cancel point is at stage boundaries. Anything
    in-progress when ``cancel_event`` fires runs to completion; the
    next stage doesn't start.
    """


@dataclass
class PipelineContext:
    """Shared state carried between pipeline stages."""

    video_path: Path
    output_dir: Path
    config: dict[str, Any]
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    stage_stats: dict[str, Any] = field(default_factory=dict)
    skip_existing: bool = False
    # Optional cancel hook: a thread-safe Event the JobManager (or any
    # caller) can set to stop the pipeline at the next stage boundary.
    # ``None`` means "no cancel support wired in" — pure-CLI invocations
    # leave it unset and run to completion.
    cancel_event: Event | None = None
    # Source video metadata, cached so resolver/stages don't re-probe.
    # Populated by ``Pipeline.run`` (or any caller building a context
    # directly). Optional because tests/legacy callers may construct a
    # context without a real video; ``video_meta()`` lazily fills these
    # on first access.
    video_fps: float | None = None
    video_width: int | None = None
    video_height: int | None = None
    frame_count: int | None = None

    def stage_dir(self, name: str) -> Path:
        """Get (and register) the output directory for a named stage."""
        if name not in self.stage_dirs:
            self.stage_dirs[name] = self.output_dir / name
        return self.stage_dirs[name]

    def is_cancelled(self) -> bool:
        return self.cancel_event is not None and self.cancel_event.is_set()


class Stage(ABC):
    """Base class for all pipeline stages."""

    name: str = ""
    description: str = ""

    @abstractmethod
    def run(self, ctx: PipelineContext) -> dict[str, Any]:
        """Execute this stage.

        Returns:
            Dict of statistics / metadata from the stage run.
        """
        ...

    def should_skip(self, ctx: PipelineContext) -> bool:
        """Return True if this stage's output already exists and skip is requested."""
        return False
