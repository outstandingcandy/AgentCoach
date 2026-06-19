"""Resolve resolution-/fps-coupled config keys against a video's metadata.

Per-video YAMLs should describe physical facts the system can't observe
(camera position, pitch dims, sensor identity, model paths). Anything
mechanically derivable from the video's (width, height, fps) — focal
bounds in pixel units, gap-fill reprojection thresholds, YOLO ``imgsz``,
fps-scaled frame counts — is computed here so users don't hand-edit it
on every new video.

Single entry point: ``resolve_config(config, width, height, fps)``.

Legacy explicit keys always win, so existing configs keep working.
"""

from __future__ import annotations

import math
from typing import Any


def _hfov_deg_to_focal_px(hfov_deg: float, width: int) -> float:
    """Convert a horizontal field-of-view (degrees) to a focal length in px.

    f = W / (2 * tan(hfov / 2))
    """
    return width / (2.0 * math.tan(math.radians(hfov_deg) / 2.0))


def resolve_config(
    config: dict[str, Any],
    width: int | None,
    height: int | None,
    fps: float | None,
) -> dict[str, Any]:
    """Fill in resolution/fps-derived config values.

    Mutates a *copy* of ``config`` so the caller's dict is left alone.

    When metadata is missing (``None``), every derivation that depends on
    it is skipped — the merged config flows through unchanged. This lets
    callers (tests, weird CLI paths) build a context without a real video.
    """
    out = _deep_copy_dict(config)

    # Default video.process_fps to min(30, source_fps) when unset. The
    # legacy default ("not set" -> "process every frame") was a foot-gun
    # for any source over 30 fps; tracking/track_consolidation thresholds
    # were tuned for ~30 fps and silently break above that.
    video_cfg = out.setdefault("video", {})
    if video_cfg.get("process_fps") is None and fps:
        video_cfg["process_fps"] = float(min(30.0, fps))

    # Effective fps used for fps-scaled derivations below. Mirrors the
    # rule in tracking/orchestrator.py:291.
    process_fps = video_cfg.get("process_fps")
    if process_fps and fps:
        effective_fps = process_fps if process_fps < fps else fps
    else:
        effective_fps = process_fps or fps

    # ---------- field_registration.physical ----------
    fr = out.setdefault("field_registration", {})
    phys = fr.setdefault("physical", {})

    # focal_hfov_deg_bounds: [minDeg, maxDeg]  ->  focal_bounds (px)
    # Camera-physics native; replaces the resolution-coupled ``focal_bounds``.
    # If the user already set ``focal_bounds`` (legacy), keep it untouched.
    if "focal_bounds" not in phys and width:
        hfov = phys.get("focal_hfov_deg_bounds")
        if hfov and len(hfov) == 2:
            f_max = _hfov_deg_to_focal_px(float(hfov[0]), width)  # narrow fov -> long focal
            f_min = _hfov_deg_to_focal_px(float(hfov[1]), width)  # wide fov -> short focal
            phys["focal_bounds"] = [float(f_min), float(f_max)]

    # gap_fill_chain.{anchor_max_reproj_frac, overwrite_above_reproj_frac}
    # -> *_px (legacy keys win when present).
    chain = phys.setdefault("gap_fill_chain", {})
    if width:
        if (
            "anchor_max_reproj_px" not in chain
            and "anchor_max_reproj_frac" in chain
        ):
            chain["anchor_max_reproj_px"] = float(chain["anchor_max_reproj_frac"]) * width
        if (
            "overwrite_above_reproj_px" not in chain
            and "overwrite_above_reproj_frac" in chain
        ):
            chain["overwrite_above_reproj_px"] = float(chain["overwrite_above_reproj_frac"]) * width

    # ---------- unified_detection.imgsz ----------
    # Default to min(1920, max(W, H)) when unset. YOLO letterboxes either
    # way; this just stops 1080p sources from being upscaled to 1920 long
    # edge (which can't recover detail and slows inference).
    unified = out.setdefault("unified_detection", {})
    if unified.get("imgsz") is None and width and height:
        unified["imgsz"] = int(min(1920, max(width, height)))

    # ---------- track_consolidation.min_track_seconds ----------
    # When the user provides ``min_track_seconds``, convert to the legacy
    # ``min_track_frames`` (which the runner already auto-scales by
    # effective_fps/10). The runner also has its own fallback default
    # (``max(1, round(30 * _fps_scale))``) so we only set the converted
    # value when the user explicitly opted in via seconds *and* didn't
    # also set min_track_frames.
    tc = out.setdefault("track_consolidation", {})
    if (
        "min_track_frames" not in tc
        and "min_track_seconds" in tc
        and effective_fps
    ):
        tc["min_track_frames"] = int(round(float(tc["min_track_seconds"]) * float(effective_fps)))

    return out


def _deep_copy_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Recursive copy that only descends into dicts/lists.

    ``copy.deepcopy`` would also clone tensors / numpy arrays / Path
    objects we may have stashed on the config; we just want to avoid
    sharing nested-dict references with the caller.
    """
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            out[k] = [_deep_copy_dict(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out
