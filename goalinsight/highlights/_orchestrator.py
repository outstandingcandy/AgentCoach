"""HighlightOrchestrator — chains agents together based on YAML recipes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ._context import MatchContext
from ._types import Event
from .agents.base import BaseClipComposer, BaseEventDetector, BaseSceneAnalyzer

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Factory functions (following goalinsight/utils/config.py pattern)
# ------------------------------------------------------------------


def _get_detector(name: str, config: dict[str, Any]) -> BaseEventDetector:
    if name == "goal":
        from .agents.goal_detector import GoalEventDetector

        return GoalEventDetector()
    raise ValueError(f"Unknown event detector: {name!r}")


def _get_analyzer(name: str, config: dict[str, Any]) -> BaseSceneAnalyzer:
    if name == "scorer":
        from .agents.scorer_analyzer import ScorerAnalyzer

        return ScorerAnalyzer()
    raise ValueError(f"Unknown scene analyzer: {name!r}")


def _get_composer(name: str, config: dict[str, Any]) -> BaseClipComposer:
    if name == "segment":
        from .agents.segment_composer import SegmentComposer

        return SegmentComposer()
    raise ValueError(f"Unknown clip composer: {name!r}")


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------


class HighlightOrchestrator:
    """Run highlight recipes by chaining detector → analyzer → composer."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def run(self, ctx: MatchContext, output_dir: str | Path) -> list[Path]:
        """Execute all configured highlight recipes.

        Args:
            ctx: Match context with pipeline output.
            output_dir: Base directory for highlight clips.

        Returns:
            Paths to generated highlight clips.
        """
        output_dir = Path(output_dir)
        recipes = self.config.get("recipes", [])
        if not recipes:
            logger.warning("No highlight recipes configured")
            return []

        all_clips: list[Path] = []

        for recipe in recipes:
            name = recipe.get("name", "unnamed")
            logger.info("Running highlight recipe: %s", name)

            try:
                clips = self._run_recipe(recipe, ctx, output_dir / name)
                all_clips.extend(clips)
            except Exception:
                logger.exception("Recipe '%s' failed", name)

        logger.info("Generated %d highlight clip(s) total", len(all_clips))
        return all_clips

    def _run_recipe(
        self,
        recipe: dict[str, Any],
        ctx: MatchContext,
        output_dir: Path,
    ) -> list[Path]:
        detector_name = recipe.get("detector", "goal")
        analyzer_name = recipe.get("analyzer", "scorer")
        composer_name = recipe.get("composer", "segment")

        detector = _get_detector(detector_name, self.config)
        analyzer = _get_analyzer(analyzer_name, self.config)
        composer = _get_composer(composer_name, self.config)

        # 1. Detect events
        events: list[Event] = detector.detect(ctx, self.config)
        if not events:
            logger.info("No events detected for recipe '%s'", recipe.get("name"))
            return []

        # 2. Analyze each event and compose clips
        clips: list[Path] = []
        for event in events:
            analyzed = analyzer.analyze(event, ctx, self.config)
            clip_path = composer.compose(analyzed, ctx, output_dir, self.config)
            clips.append(clip_path)
            logger.info(
                "Clip generated: %s (event=%s frame=%d)",
                clip_path.name, event.event_type, event.frame,
            )

        return clips
