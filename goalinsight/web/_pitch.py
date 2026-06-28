"""Per-run pitch resolution for the web layer.

Reads ``workspace/configs/<video_stem>.yaml`` (the same file the annotate
page consumes via ``set_solver_config``) and returns a SoccerPitch for
the given run. Used by the match-detail and analytics endpoints so each
HTTP request resolves its pitch from disk — independent of the
process-global ``pitch_constants.get_active_pitch()`` state that can
race across runs when one web process serves many videos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..annotation import per_video_settings
from ..annotation.pitch.geometry import SoccerPitch


def resolve_pitch_for_run(
    ctx: Any,
    video_path: Any,
    workspace: Any,
) -> SoccerPitch:
    """Build a SoccerPitch for the given run, independent of process state.

    Source precedence:

    1. ``workspace/configs/<video_stem>.yaml`` — carries ``pitch_type``
       (or inline ``pitch:`` block) so PA_SHAPE / goal length / non-FIFA
       dimensions travel with the video. Same file the annotate page
       reads via :func:`per_video_settings.load_pitch` /
       :meth:`AnchorAnnotator.set_solver_config`.
    2. Legacy ``workspace/annotations/<stem>/overrides.yaml`` for runs
       authored before the per-video config editor existed.
    3. Run's calibration_metadata (``ctx.pitch_length`` / ``pitch_width``)
       — only the outer dimensions; PA_SHAPE defaults to ``rect``.
    4. FIFA defaults.
    """
    pitch_dims: dict[str, Any] = {}

    if workspace is not None and video_path is not None:
        cfg_path = workspace.config_for(video_path)
        if cfg_path.exists():
            import yaml as _yaml

            from ..utils.config_resolver import expand_pitch_type
            try:
                raw = _yaml.safe_load(cfg_path.read_text()) or {}
                expand_pitch_type(raw)
                pitch_dims = raw.get("pitch") or {}
            except (OSError, _yaml.YAMLError):
                pitch_dims = {}
        if not pitch_dims:
            try:
                pitch_dims = per_video_settings.load_pitch(
                    workspace.annotations_dir, Path(video_path).stem,
                ) or {}
            except Exception:
                pitch_dims = {}

    if pitch_dims:
        try:
            return SoccerPitch.from_dict(pitch_dims)
        except (TypeError, ValueError):
            pass

    # No per-video yaml — fall back to scaled FIFA with the run's
    # calibration_metadata dims (best we can do without a pitch_type).
    L = float(getattr(ctx, "pitch_length", None) or 105.0)
    W = float(getattr(ctx, "pitch_width", None) or 68.0)
    base = SoccerPitch()
    if abs(L - base.PITCH_LENGTH) < 1e-3 and abs(W - base.PITCH_WIDTH) < 1e-3:
        return base
    sx = L / base.PITCH_LENGTH
    sy = W / base.PITCH_WIDTH
    return SoccerPitch(
        pitch_length=L,
        pitch_width=W,
        penalty_area_length=base.PENALTY_AREA_LENGTH * sx,
        penalty_area_width=base.PENALTY_AREA_WIDTH * sy,
        goal_area_length=base.GOAL_AREA_LENGTH * sx,
        goal_area_width=base.GOAL_AREA_WIDTH * sy,
        goal_line_to_penalty_mark=base.GOAL_LINE_TO_PENALTY_MARK * sx,
        center_circle_radius=base.CENTER_CIRCLE_RADIUS * min(sx, sy),
    )
