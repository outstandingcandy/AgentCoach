"""Annotated full-video render — HUD-style overlays on the entire match video.

Public entrypoint: :func:`run_annotated_video`.

This stage consumes the outputs of ``field_registration`` and ``tracking``
(via :class:`~goalinsight.highlights._context.MatchContext`) and writes a
single long MP4 with per-player glow rings, jersey/ID labels, a ball
trajectory trail, a weak outline on the ball-carrier, and a top-down
minimap pinned to the bottom-right of the frame at the video's aspect
ratio. The output is web-optimized (small GOP, faststart) so the
browser viewer can scrub instantly.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from ..highlights._context import MatchContext
from ._renderer import AnnotatedVideoRenderer

logger = logging.getLogger(__name__)


def run_annotated_video(
    output_dir: str | Path,
    pipeline_output_dir: str | Path,
    video_path: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Render the full annotated video.

    Args:
        output_dir: Directory for this stage's artifacts (usually
            ``<pipeline_output>/annotated_video``).
        pipeline_output_dir: Root pipeline output directory (used to locate
            upstream ``tracking/`` and ``field_registration/`` output).
        video_path: Source video path.
        config: The ``annotated_video:`` config section — may also carry a
            ``video_enhancement`` sub-dict injected by the adapter.

    Returns:
        Stats dict: ``{video, frames, carrier_switches, enhanced_video}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = MatchContext.from_output_dir(
        pipeline_output_dir=pipeline_output_dir,
        video_path=video_path,
    )

    # Stamp each render with a timestamp so re-runs accumulate alongside
    # earlier outputs instead of clobbering them. ``annotated.mp4`` is
    # kept as a stable symlink to the most recent render so existing
    # tooling (and the web viewer) keeps working.
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    annotated_path = output_dir / f"annotated_{stamp}.mp4"
    latest_link = output_dir / "annotated.mp4"

    renderer = AnnotatedVideoRenderer(ctx, config)
    stats = renderer.render(annotated_path)

    # Refresh the "latest" pointer atomically (replace any prior file or
    # symlink). Use a relative target so the link stays valid if the
    # output_dir is moved.
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    try:
        latest_link.symlink_to(annotated_path.name)
    except OSError:
        # Filesystem doesn't support symlinks — copy instead.
        shutil.copy2(annotated_path, latest_link)

    # Prune older timestamped renders so disk usage doesn't grow without
    # bound. Default keeps the 3 most recent (configurable via
    # ``annotated_video.keep_last`` in the config).
    _prune_old_renders(
        output_dir, keep=int(config.get("keep_last", 3)),
        protect={annotated_path.name},
    )

    enhanced_path: Path | None = None
    ve_cfg = config.get("video_enhancement")
    if ve_cfg and ve_cfg.get("enabled"):
        enhanced_path = _enhance(annotated_path, output_dir, ve_cfg)

    stats_out = {
        "video": str(annotated_path),
        "frames": stats["frames"],
        "carrier_switches": stats["carrier_switches"],
        "enhanced_video": str(enhanced_path) if enhanced_path else None,
    }
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats_out, f, indent=2)

    return stats_out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _prune_old_renders(
    output_dir: Path, keep: int, protect: set[str],
) -> None:
    """Keep the *keep* newest ``annotated_<stamp>.mp4`` files; delete the rest.

    Files in *protect* are never deleted (used to guard the just-written
    render even if the user sets keep < 1).
    """
    if keep <= 0:
        return
    # Sort by mtime descending — newest first. Filenames embed a stamp so
    # mtime ordering is robust to clock skew and filename quirks alike.
    candidates = sorted(
        output_dir.glob("annotated_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates[keep:]:
        if path.name in protect:
            continue
        try:
            path.unlink()
            logger.info("annotated_video: pruned old render %s", path.name)
        except OSError as exc:
            logger.warning("annotated_video: could not prune %s: %s",
                           path.name, exc)


def _enhance(
    annotated_path: Path,
    output_dir: Path,
    ve_cfg: dict[str, Any],
) -> Path | None:
    """Pipe the annotated mp4 through video2x (fail-soft)."""
    from ..video_enhancement import run_video_enhancement

    enhance_in = output_dir / "_enhance_in"
    enhance_out = output_dir / "_enhance_out"
    enhance_in.mkdir(parents=True, exist_ok=True)
    staged = enhance_in / annotated_path.name
    staged.unlink(missing_ok=True)
    try:
        staged.symlink_to(annotated_path.resolve())
    except OSError:
        shutil.copy2(annotated_path, staged)

    try:
        enhanced = run_video_enhancement(enhance_in, enhance_out, ve_cfg)
    except Exception:
        logger.warning("Video enhancement failed; keeping un-enhanced output.",
                       exc_info=True)
        enhanced = []
    finally:
        shutil.rmtree(enhance_in, ignore_errors=True)

    if not enhanced:
        shutil.rmtree(enhance_out, ignore_errors=True)
        return None

    final = output_dir / "annotated_enhanced.mp4"
    shutil.move(str(enhanced[0]), str(final))
    shutil.rmtree(enhance_out, ignore_errors=True)
    return final
