"""EventOrchestrator — runs detectors in dependency order."""

from __future__ import annotations

import logging
from typing import Any

from ._ball_physics import compute_ball_states
from ._context import EventDetectionContext
from ._registry import DETECTOR_REGISTRY
from ._types import MatchEvent

logger = logging.getLogger(__name__)


class EventOrchestrator:
    """Run event detectors in topological (dependency) order."""

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def detect_all(
        self, ctx: EventDetectionContext
    ) -> list[MatchEvent]:
        """Run all configured detectors and return accumulated events."""
        # Ensure detectors are registered
        from . import detectors as _  # noqa: F401

        # Pre-compute ball states
        events_cfg = self.config.get("events", {})
        min_conf = events_cfg.get("shot", {}).get("min_confidence", 0.1)
        ctx.ball_states = compute_ball_states(
            ctx.ball_tracks, ctx.fps, min_confidence=min_conf
        )
        ctx.frame_to_ball = {bs.frame: bs for bs in ctx.ball_states}

        logger.info(
            "Ball states computed: %d frames", len(ctx.ball_states)
        )

        # Determine which detectors to run
        enabled = events_cfg.get(
            "detectors",
            ["possession", "pass", "shot", "carry", "defensive"],
        )

        # Topological sort by depends_on
        ordered = self._topo_sort(enabled)

        for name in ordered:
            if name not in DETECTOR_REGISTRY:
                logger.warning("Unknown detector '%s', skipping", name)
                continue

            detector = DETECTOR_REGISTRY[name]()
            logger.info("Running detector: %s", name)
            new_events = detector.detect(ctx, self.config)
            ctx.events.extend(new_events)

        ctx.events.sort(key=lambda e: e.frame)
        return ctx.events

    @staticmethod
    def _topo_sort(names: list[str]) -> list[str]:
        """Topological sort of detector names by depends_on."""
        name_set = set(names)

        # Build dependency graph (only for enabled detectors)
        deps: dict[str, list[str]] = {}
        for name in names:
            cls = DETECTOR_REGISTRY.get(name)
            if cls is None:
                deps[name] = []
                continue
            deps[name] = [
                d for d in cls.depends_on if d in name_set
            ]

        # Kahn's algorithm
        in_degree = {n: len(deps[n]) for n in names}
        queue = [n for n in names if in_degree[n] == 0]
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for n in names:
                if node in deps[n]:
                    in_degree[n] -= 1
                    if in_degree[n] == 0:
                        queue.append(n)

        # Append any remaining (circular deps — shouldn't happen)
        for n in names:
            if n not in result:
                result.append(n)

        return result
