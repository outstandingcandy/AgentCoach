"""Abstract base classes for highlight clipping agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .._context import MatchContext
from .._types import AnalyzedEvent, Event


class BaseEventDetector(ABC):
    """Detects events of a specific type from match trajectory data."""

    event_type: str  # Subclasses declare what they detect ("goal", "fast_break", ...)

    @abstractmethod
    def detect(self, ctx: MatchContext, config: dict[str, Any]) -> list[Event]:
        """Scan match data and return detected events.

        Args:
            ctx: Match context with all pipeline output.
            config: Highlight configuration dict.

        Returns:
            List of detected events.
        """


class BaseSceneAnalyzer(ABC):
    """Enriches an event with player attribution and temporal segments."""

    supported_events: list[str]  # Which event types this analyzer handles

    @abstractmethod
    def analyze(
        self,
        event: Event,
        ctx: MatchContext,
        config: dict[str, Any],
    ) -> AnalyzedEvent:
        """Analyze an event to determine key players and clip segments.

        Args:
            event: The detected event.
            ctx: Match context.
            config: Highlight configuration dict.

        Returns:
            Enriched event with player info and segment plan.
        """


class BaseClipComposer(ABC):
    """Renders an AnalyzedEvent into a video clip."""

    @abstractmethod
    def compose(
        self,
        analyzed: AnalyzedEvent,
        ctx: MatchContext,
        output_dir: Path,
        config: dict[str, Any],
    ) -> Path:
        """Compose a highlight clip from the analyzed event.

        Args:
            analyzed: Analyzed event with segments.
            ctx: Match context (for video path, track data).
            output_dir: Where to write the output clip.
            config: Highlight configuration dict.

        Returns:
            Path to the generated highlight clip.
        """
