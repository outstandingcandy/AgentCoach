"""Build per-player profiles: front/back crops, heatmap PNG, distance run.

Inputs (all from upstream stages):
- ``track_consolidation/players.json`` — list of consolidated players, each
  with ``player_id``, ``source_tracks: [tid, ...]``, ``team``, ``role``,
  ``jersey_number``.
- ``track_consolidation/llm_inputs/per_track.json`` — per-original-track
  metadata; the ``crops`` array gives us each saved crop's filename,
  sharpness ``score`` and OCR ``visible_digits`` verdict (the orientation
  signal we use to pick front vs back).
- ``track_consolidation/llm_inputs/tracks/<name>.jpg`` — the actual crop
  images sampled during consolidation.
- ``track_consolidation/tracks.json`` — per-frame, player-id-keyed track
  list (used both for heatmap and distance computation; identical to
  what MatchContext exposes as ``player_tracks``).

Outputs (under ``output/<run>/player_profile/``):
- ``players_profile.json`` — list of profile dicts, one per player.
- ``crops/<player_id>_front.jpg`` / ``<player_id>_back.jpg`` — picked crops.
- ``heatmaps/<player_id>.png`` — pitch heatmap PNG.
"""

from __future__ import annotations

import io
import json
import logging
import math
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Same speed/gap thresholds match_tools.get_player_stats uses, so the
# numbers reported on the match page agree with the stage output.
_MAX_GAP_S = 1.0
_MAX_REALISTIC_MPS = 12.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_player_profiles(
    pipeline_output_dir: Path,
    out_dir: Path,
    *,
    heatmap_bins: int = 30,
) -> dict[str, Any]:
    """Build per-player profiles, writing all artifacts under *out_dir*.

    Returns a stats dict for the pipeline driver — total players,
    counts of front / back crops actually picked, etc.
    """
    pipeline_output_dir = Path(pipeline_output_dir)
    out_dir = Path(out_dir)
    crops_dir = out_dir / "crops"
    heatmaps_dir = out_dir / "heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    consol_dir = pipeline_output_dir / "track_consolidation"
    players_path = consol_dir / "players.json"
    if not players_path.exists():
        raise RuntimeError(
            f"player_profile stage requires track_consolidation output: "
            f"{players_path} missing"
        )
    players = json.loads(players_path.read_text())

    per_track_path = consol_dir / "llm_inputs" / "per_track.json"
    per_track_index = _load_per_track(per_track_path)

    tracks = _load_tracks(consol_dir / "tracks.json")
    fps = _load_fps(pipeline_output_dir)

    # Pre-compute per-player observations once instead of scanning
    # tracks.json N times (one per player).
    obs_by_pid = _index_observations_by_player(tracks)

    # Pitch dimensions for the heatmap come from calibration metadata so
    # non-FIFA pitches (kids 66.28×43.15) render at the right aspect.
    pitch_length, pitch_width = _load_pitch_dims(pipeline_output_dir)

    crops_root = consol_dir / "llm_inputs" / "tracks"
    profiles: list[dict[str, Any]] = []
    front_count = back_count = heatmap_count = 0

    for entry in players:
        pid = entry.get("player_id")
        if not pid:
            continue
        source_tids = entry.get("source_tracks") or []

        front_src, back_src = _pick_orientation_crops(
            source_tids, per_track_index, crops_root,
        )
        front_url = back_url = None
        if front_src is not None:
            dest = crops_dir / f"{_safe_name(pid)}_front.jpg"
            shutil.copyfile(front_src, dest)
            front_url = f"crops/{dest.name}"
            front_count += 1
        if back_src is not None:
            dest = crops_dir / f"{_safe_name(pid)}_back.jpg"
            shutil.copyfile(back_src, dest)
            back_url = f"crops/{dest.name}"
            back_count += 1

        positions = obs_by_pid.get(pid, [])
        distance_m, max_speed, avg_speed = _trajectory_stats(positions, fps)

        heatmap_url = None
        if positions:
            png = _render_heatmap_png(
                [(p[1], p[2]) for p in positions],
                bins=heatmap_bins,
                pitch_length=pitch_length,
                pitch_width=pitch_width,
                title=f"{pid}",
            )
            if png is not None:
                hpath = heatmaps_dir / f"{_safe_name(pid)}.png"
                hpath.write_bytes(png)
                heatmap_url = f"heatmaps/{hpath.name}"
                heatmap_count += 1

        profiles.append({
            "player_id": pid,
            "team_id": entry.get("team"),
            "role": entry.get("role"),
            "jersey_number": entry.get("jersey_number"),
            "frames_observed": len(positions),
            "distance_m": round(distance_m, 2),
            "max_speed_mps": round(max_speed, 2),
            "avg_speed_mps": round(avg_speed, 2),
            "front_crop": front_url,
            "back_crop": back_url,
            "heatmap": heatmap_url,
        })

    profile_path = out_dir / "players_profile.json"
    profile_path.write_text(json.dumps(profiles, indent=2))

    return {
        "players": len(profiles),
        "front_crops": front_count,
        "back_crops": back_count,
        "heatmaps": heatmap_count,
    }


# ---------------------------------------------------------------------------
# Crop picker — uses the OCR "visible_digits" verdict as a facing hint.
# ---------------------------------------------------------------------------


def _pick_orientation_crops(
    source_tids: list[int],
    per_track_index: dict[int, list[dict[str, Any]]],
    crops_root: Path,
) -> tuple[Path | None, Path | None]:
    """Pick the sharpest front-facing and back-facing crop across all
    source tracks of a consolidated player.

    Front: ``ocr_visible == 'back not visible'`` — VLM saw the chest /
    front of the kit, not the number.
    Back: ``ocr_visible`` is a digit string (or ``partial: <digit>``) —
    the number is at least partially readable, so the back is facing the
    camera.

    Falls back to the highest-score crop overall when no labelled
    candidate exists, so even orientation-ambiguous players still get a
    representative thumbnail.
    """
    fronts: list[tuple[float, Path]] = []
    backs: list[tuple[float, Path]] = []
    any_crop: list[tuple[float, Path]] = []

    for tid in source_tids:
        for crop in per_track_index.get(int(tid), []):
            name = crop.get("name")
            if not name:
                continue
            path = crops_root / name
            if not path.exists():
                continue
            score = float(crop.get("score") or 0.0)
            label = (crop.get("ocr_visible") or "").strip().lower()
            any_crop.append((score, path))
            if label == "back not visible":
                fronts.append((score, path))
            elif _looks_like_back(label):
                backs.append((score, path))

    front = max(fronts, key=lambda x: x[0])[1] if fronts else None
    back = max(backs, key=lambda x: x[0])[1] if backs else None

    # If only one orientation surfaced, fill the other slot with the
    # best remaining crop so users always see two thumbnails when crops
    # are available at all.
    if any_crop:
        any_crop.sort(key=lambda x: -x[0])
        used = {front, back}
        for _, path in any_crop:
            if front is None and path not in used:
                front = path
                used.add(path)
            elif back is None and path not in used:
                back = path
                used.add(path)
            if front is not None and back is not None:
                break
    return front, back


def _looks_like_back(label: str) -> bool:
    """Verdict implies the number is at least partially visible."""
    if not label:
        return False
    if label.startswith("partial: "):
        rest = label[len("partial: "):]
        return rest and rest[0].isdigit()
    return label[0].isdigit()


# ---------------------------------------------------------------------------
# Trajectory + heatmap utilities
# ---------------------------------------------------------------------------


def _trajectory_stats(
    positions: list[tuple[int, float, float]],
    fps: float,
) -> tuple[float, float, float]:
    """Total distance + speed stats. Mirrors match_tools.get_player_stats so
    the per-player numbers shown on the match page line up exactly."""
    if not positions:
        return 0.0, 0.0, 0.0
    positions = sorted(positions, key=lambda p: p[0])
    distance = 0.0
    max_speed = 0.0
    speeds: list[float] = []
    for (f1, x1, y1), (f2, x2, y2) in zip(positions, positions[1:]):
        dt = (f2 - f1) / fps if fps else 0
        if dt <= 0 or dt > _MAX_GAP_S:
            continue
        d = math.hypot(x2 - x1, y2 - y1)
        speed = d / dt
        if speed > _MAX_REALISTIC_MPS:
            continue
        distance += d
        speeds.append(speed)
        max_speed = max(max_speed, speed)
    avg = (sum(speeds) / len(speeds)) if speeds else 0.0
    return distance, max_speed, avg


def _render_heatmap_png(
    positions: list[tuple[float, float]],
    *,
    bins: int,
    pitch_length: float,
    pitch_width: float,
    title: str,
) -> bytes | None:
    """Render a heatmap PNG over a schematic top-down pitch.

    Self-contained matplotlib rendering: outline + halfway line +
    center circle + both penalty / goal areas, then the histogram-based
    heatmap on top. Stays decoupled from the global
    ``pitch_constants.get_active_pitch()`` state so a kids-pitch run
    renders against the kids pitch even when the host process was
    booted with FIFA defaults.
    """
    if not positions:
        return None
    # Lazy matplotlib import — keeps stage import cheap when heatmaps
    # aren't being generated (e.g. tests).
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = pitch_length / 2.0
    W = pitch_width / 2.0
    # Standard-pitch ratios scaled to whatever pitch we're on. Same
    # ratios the SoccerPitch dataclass uses for its defaults; keeps the
    # schematic recognisable on non-FIFA dims.
    pa_l = pitch_length * (16.5 / 105.0)
    pa_w = pitch_width * (40.32 / 68.0) / 2.0
    ga_l = pitch_length * (5.5 / 105.0)
    ga_w = pitch_width * (18.32 / 68.0) / 2.0
    ccr = min(pitch_length, pitch_width) * (9.15 / 68.0)

    fig, ax = plt.subplots(figsize=(6.0, 6.0 * pitch_width / pitch_length),
                           dpi=100)
    ax.set_facecolor("#1a3a1a")
    line = "#cfe3ff"

    # Outline + halfway line + center circle.
    ax.plot([-L, L, L, -L, -L], [W, W, -W, -W, W], color=line, lw=1.5)
    ax.plot([0, 0], [W, -W], color=line, lw=1.5)
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(ccr * np.cos(theta), ccr * np.sin(theta), color=line, lw=1.2)

    # Penalty + goal areas, both ends.
    for sign in (-1, 1):
        x_goal = sign * L
        x_pa = sign * (L - pa_l)
        x_ga = sign * (L - ga_l)
        ax.plot([x_goal, x_pa, x_pa, x_goal],
                [pa_w, pa_w, -pa_w, -pa_w], color=line, lw=1.2)
        ax.plot([x_goal, x_ga, x_ga, x_goal],
                [ga_w, ga_w, -ga_w, -ga_w], color=line, lw=1.2)

    # Histogram-based heatmap, drawn last so it overlays the lines.
    xs = np.array([p[0] for p in positions])
    ys = np.array([p[1] for p in positions])
    H, xedges, yedges = np.histogram2d(
        xs, ys, bins=bins, range=[[-L, L], [-W, W]],
    )
    ax.pcolormesh(xedges, yedges, H.T, alpha=0.65,
                  shading="auto", cmap="hot")

    margin = max(pitch_length, pitch_width) * 0.04
    ax.set_xlim(-L - margin, L + margin)
    ax.set_ylim(-W - margin, W + margin)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=11, color="#222")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_per_track(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    blob = json.loads(path.read_text())
    raw = blob.get("tracks") or {}
    out: dict[int, list[dict[str, Any]]] = {}
    for k, v in raw.items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        out[tid] = v.get("crops") or []
    return out


def _load_tracks(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _load_fps(pipeline_output_dir: Path) -> float:
    meta = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        try:
            return float(json.loads(meta.read_text()).get("video_info", {}).get("fps") or 30.0)
        except Exception:  # noqa: BLE001
            return 30.0
    return 30.0


def _load_pitch_dims(pipeline_output_dir: Path) -> tuple[float, float]:
    meta = pipeline_output_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        try:
            vi = json.loads(meta.read_text()).get("video_info", {}) or {}
            return float(vi.get("pitch_length") or 105.0), float(vi.get("pitch_width") or 68.0)
        except Exception:  # noqa: BLE001
            pass
    return 105.0, 68.0


def _index_observations_by_player(
    tracks: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[int, float, float]]]:
    out: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for k, items in tracks.items():
        try:
            f = int(k)
        except (TypeError, ValueError):
            continue
        for t in items:
            pid = t.get("track_id")
            pp = t.get("pitch_position") or [None, None]
            if pid is None or pp[0] is None:
                continue
            out[str(pid)].append((f, float(pp[0]), float(pp[1])))
    return out


def _safe_name(pid: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(pid))
