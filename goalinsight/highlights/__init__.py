"""Highlight clipping agent system.

Provides a flexible, agent-based pipeline for generating highlight clips
from GoalInsight pipeline output (trajectory data, ball tracking, etc.).

Architecture: EventDetector → SceneAnalyzer → ClipComposer
Orchestrated by YAML "recipes" for extensibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ._context import MatchContext
from ._orchestrator import HighlightOrchestrator

logger = logging.getLogger(__name__)

# Default highlight configuration (merged under any user-provided config)
_DEFAULT_HIGHLIGHT_CONFIG: dict[str, Any] = {
    "output_fps": 25.0,
    "crossfade_frames": 4,
    "closeup": {
        "output_size": [640, 360],
        "padding_factor": 4.0,
        "smooth_alpha": 0.15,
        "medium_padding_factor": 2.5,
        "ball_padding_factor": 6.0,
    },
    "temporal": {
        "buildup_max_seconds": 10.0,
        "buildup_padding_seconds": 2.0,
        "buildup_view": "wide",
        "strike_pre_seconds": 0.5,
        "strike_post_seconds": 1.5,
        "strike_view": "closeup",
        "celebration_seconds": 10.0,
        "celebration_view": "medium",
        "replay_enabled": True,
        "replay_speed": 0.4,
    },
    "effects": {
        "shooter_spotlight": True,
        "spotlight_color": [0, 215, 255],  # gold BGR
        "spotlight_alpha": 0.4,
        "ball_trail": True,
        "trail_length": 15,
        "trail_color": [0, 255, 255],  # yellow BGR
    },
    "overlays": {
        "enabled": True,
        "show_event_label": True,
        "show_timestamp": True,
    },
    "goal_detection": {
        "min_confidence": 0.15,
    },
    "recipes": [
        {
            "name": "goal_highlight",
            "detector": "goal",
            "analyzer": "scorer",
            "composer": "segment",
        },
    ],
}


def _merge(base: dict, override: dict) -> dict:
    """Deep-merge *override* into *base* (non-destructive)."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge(result[k], v)
        else:
            result[k] = v
    return result


def run_highlights(
    output_dir: str | Path,
    pipeline_output_dir: str | Path,
    video_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> list[Path]:
    """Generate highlight clips from pipeline output.

    Args:
        output_dir: Directory to write highlight clips.
        pipeline_output_dir: Path to pipeline output (stage1/, stage2/, stage3/).
        video_path: Path to the source video file. If None, auto-resolved
            from the pipeline's calibration_metadata.json.
        config: Highlight configuration (merged with defaults).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        List of paths to generated highlight clips.
    """
    merged_config = _merge(_DEFAULT_HIGHLIGHT_CONFIG, config or {})

    ctx = MatchContext.from_output_dir(
        pipeline_output_dir=pipeline_output_dir,
        video_path=video_path,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
    )
    logger.info(
        "MatchContext loaded: %d frames, %.1f fps, %dx%d, pitch=%.0fx%.0fm",
        ctx.frame_count, ctx.fps, ctx.width, ctx.height,
        ctx.pitch_length, ctx.pitch_width,
    )

    orchestrator = HighlightOrchestrator(merged_config)
    return orchestrator.run(ctx, output_dir)
