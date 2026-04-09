"""Base class for event detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ._types import MatchEvent


class BaseEventDetector(ABC):
    """Base class for all event detectors."""

    name: str = ""
    depends_on: list[str] = []

    @abstractmethod
    def detect(
        self,
        ctx: "EventDetectionContext",  # noqa: F821
        config: dict[str, Any],
    ) -> list[MatchEvent]:
        """Run detection and return discovered events.

        The detector may also mutate *ctx* to store shared state
        (e.g., PossessionDetector populates ctx.possession_spans).
        """
