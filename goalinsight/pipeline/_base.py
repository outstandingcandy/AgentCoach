"""Pipeline base classes: Stage ABC and PipelineContext."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineContext:
    """Shared state carried between pipeline stages."""

    video_path: Path
    output_dir: Path
    config: dict[str, Any]
    stage_dirs: dict[str, Path] = field(default_factory=dict)
    stage_stats: dict[str, Any] = field(default_factory=dict)
    skip_existing: bool = False

    def stage_dir(self, name: str) -> Path:
        """Get (and register) the output directory for a named stage."""
        if name not in self.stage_dirs:
            self.stage_dirs[name] = self.output_dir / name
        return self.stage_dirs[name]


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
