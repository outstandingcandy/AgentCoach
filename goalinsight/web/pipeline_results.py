"""Pipeline-results API: per-stage stats + visualization manifests.

Backs the right-hand panel of ``/pipeline?run=<name>``: when the user
clicks a stage badge the page fetches a manifest describing what
artefacts that stage produced (vis directories, key files, stats).

Stages each have idiosyncratic on-disk layouts, so the manifest builder
is a small per-stage table rather than a generic recursive scan. Every
file path is relative to ``<workspace>/runs/<run>/`` so the front end
can prefix ``/runs_static/`` directly.

Routes are attached via ``register_pipeline_results_routes(app, workspace)``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from ._workspace import Workspace
from .library import _safe_filename

logger = logging.getLogger(__name__)


_FRAME_INDEX_RE = re.compile(r"frame_(\d+)")


def _frame_indices(dir_path: Path, ext: str = ".jpg") -> list[int]:
    """List sorted frame indices from ``frame_<NNN>.<ext>`` files."""
    if not dir_path.is_dir():
        return []
    out = []
    for p in dir_path.iterdir():
        if not p.suffix == ext:
            continue
        m = _FRAME_INDEX_RE.search(p.name)
        if m:
            out.append(int(m.group(1)))
    out.sort()
    return out


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def _read_text(path: Path, max_bytes: int = 200_000) -> str | None:
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return f.read(max_bytes)
    except OSError:
        return None


@dataclass
class VisDir:
    """One scrub-able image folder (per stage may have several).

    ``url_prefix`` is the path under ``/runs_static/<run>/`` so the
    front-end builds ``/runs_static/<run>/<url_prefix>/frame_NNNNN.jpg``.
    """
    name: str                     # display label
    url_prefix: str               # path under run dir, no trailing slash
    frames: list[int]             # sorted frame indices (`frame_<idx>.jpg`)
    digits: int = 5               # zero-padding width on disk
    ext: str = ".jpg"


@dataclass
class StageManifest:
    stage: str
    exists: bool                  # stage dir present
    stats: dict[str, Any] = field(default_factory=dict)
    vis_dirs: list[VisDir] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    text_files: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _file_entry(path: Path, run_dir: Path) -> dict[str, Any]:
    rel = path.relative_to(run_dir).as_posix()
    return {
        "name": path.name,
        "url": rel,                       # client prepends /runs_static/<run>/
        "size": path.stat().st_size if path.exists() else 0,
    }


def _digits_of(dir_path: Path, ext: str = ".jpg") -> int:
    """Return zero-padding width inferred from the first matching file."""
    if not dir_path.is_dir():
        return 5
    for p in dir_path.iterdir():
        if p.suffix == ext:
            m = _FRAME_INDEX_RE.search(p.name)
            if m:
                return len(m.group(1))
    return 5


def _build_field_registration(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "field_registration"
    if not stage_dir.is_dir():
        return StageManifest(stage="field_registration", exists=False)

    vis_root = stage_dir / "visualizations"
    vis = []
    for sub in ("calibration", "keypoints", "lines"):
        d = vis_root / sub
        if d.is_dir():
            digits = _digits_of(d)
            vis.append(VisDir(
                name=sub,
                url_prefix=f"field_registration/visualizations/{sub}",
                frames=_frame_indices(d),
                digits=digits,
            ))

    files: list[dict[str, Any]] = []
    for fname in ("camera_poses.json", "calibration_metadata.json",
                  "homographies.pkl", "camera_poses.pkl"):
        p = stage_dir / fname
        if p.exists():
            files.append(_file_entry(p, run_dir))

    return StageManifest(
        stage="field_registration",
        exists=True,
        stats=pipeline_stats.get("field_registration", {}),
        vis_dirs=vis,
        files=files,
    )


def _build_tracking(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "tracking"
    if not stage_dir.is_dir():
        return StageManifest(stage="tracking", exists=False)

    vis: list[VisDir] = []
    # Tracking-render frames (top-down + raw overlays)
    frames_dir = stage_dir / "frames"
    if frames_dir.is_dir():
        vis.append(VisDir(
            name="tracking",
            url_prefix="tracking/frames",
            frames=_frame_indices(frames_dir),
            digits=_digits_of(frames_dir),
        ))
    yolo_dir = stage_dir / "yolo_raw" / "frames"
    if yolo_dir.is_dir():
        vis.append(VisDir(
            name="yolo_raw (drop-stage colour)",
            url_prefix="tracking/yolo_raw/frames",
            frames=_frame_indices(yolo_dir),
            digits=_digits_of(yolo_dir),
        ))
    ball_dir = stage_dir / "ball_detection_diag"
    if ball_dir.is_dir():
        vis.append(VisDir(
            name="ball detection diag",
            url_prefix="tracking/ball_detection_diag",
            frames=_frame_indices(ball_dir),
            digits=_digits_of(ball_dir),
        ))

    files: list[dict[str, Any]] = []
    for fname in ("tracking.mp4", "tracks.json",
                  "ball_tracks.json", "team_assignments.json",
                  "track_features.json", "ball_debug_log.json", "statistics.json"):
        p = stage_dir / fname
        if p.exists():
            files.append(_file_entry(p, run_dir))

    # Per-orig-track lifetimes: list of frames where each tid appears.
    # tracking/tracks.json is the raw tracker output (additive
    # consolidation writes its rewritten copy under track_consolidation/).
    lifetimes: dict[str, dict[str, Any]] = {}
    src_path = stage_dir / "tracks.json"
    src_data = _read_json(src_path)
    if isinstance(src_data, dict):
        # Map orig_tid → sorted list of frames it appears in.
        per_tid_frames: dict[int, list[int]] = {}
        for frame_key, dets in src_data.items():
            try:
                fidx = int(frame_key)
            except (TypeError, ValueError):
                continue
            for det in dets or []:
                # tracking/tracks.json is always raw — track_id is the
                # integer tid from the tracker.
                tid = det.get("track_id")
                if not isinstance(tid, int):
                    continue
                per_tid_frames.setdefault(tid, []).append(fidx)
        for tid, frames in per_tid_frames.items():
            frames.sort()
            lifetimes[str(tid)] = {
                "tid": tid,
                "first_frame": frames[0],
                "last_frame": frames[-1],
                "n_frames": len(frames),
                "frames": frames,
            }

    manifest_stats = dict(pipeline_stats.get("tracking", {}))
    if lifetimes:
        manifest_stats["_track_lifetimes"] = lifetimes
        # Also expose the consolidation player_map.json if present so
        # the front-end can label each orig tid with its eventual
        # consolidated player_id.
        pmap = _read_json(run_dir / "track_consolidation" / "player_map.json")
        if isinstance(pmap, dict):
            manifest_stats["_player_map"] = pmap

    return StageManifest(
        stage="tracking",
        exists=True,
        stats=manifest_stats,
        vis_dirs=vis,
        files=files,
    )


def _build_track_consolidation(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "track_consolidation"
    if not stage_dir.is_dir():
        return StageManifest(stage="track_consolidation", exists=False)

    vis: list[VisDir] = []
    frames_dir = stage_dir / "frames"
    if frames_dir.is_dir():
        vis.append(VisDir(
            name="consolidated frames",
            url_prefix="track_consolidation/frames",
            frames=_frame_indices(frames_dir),
            digits=_digits_of(frames_dir),
        ))

    # Build the per-track thumb grid from llm_inputs/ — these are the
    # *exact* crops the jersey LLM saw. Falls back to the legacy
    # track_thumbs/ dir for runs that pre-date the llm_inputs dump.
    thumbs: list[dict[str, str]] = []
    llm_dir = stage_dir / "llm_inputs"
    llm_tracks_dir = llm_dir / "tracks"
    if llm_tracks_dir.is_dir():
        for p in sorted(llm_tracks_dir.iterdir()):
            if p.suffix == ".jpg":
                thumbs.append({
                    "name": p.name,
                    "url": p.relative_to(run_dir).as_posix(),
                })
    else:
        legacy_dir = stage_dir / "track_thumbs"
        if legacy_dir.is_dir():
            for p in sorted(legacy_dir.iterdir()):
                if p.suffix == ".jpg":
                    thumbs.append({
                        "name": p.name,
                        "url": p.relative_to(run_dir).as_posix(),
                    })

    # The full LLM-input bundle (per-track vote + reasoning + position +
    # movement + scene + team-exemplar URLs) so the front-end can show
    # everything Claude saw alongside the verdict.
    llm_inputs: dict[str, Any] = {}
    per_track = _read_json(llm_dir / "per_track.json")
    if isinstance(per_track, dict):
        exemplar_urls: dict[str, list[str]] = {"team_A": [], "team_B": []}
        ex_dir = llm_dir / "team_exemplars"
        if ex_dir.is_dir():
            for p in sorted(ex_dir.iterdir()):
                if p.suffix != ".jpg":
                    continue
                rel = p.relative_to(run_dir).as_posix()
                if p.name.startswith("team_A_"):
                    exemplar_urls["team_A"].append(rel)
                elif p.name.startswith("team_B_"):
                    exemplar_urls["team_B"].append(rel)
        llm_inputs = {
            "n_tracks": per_track.get("n_tracks"),
            "n_team_A_exemplars": per_track.get("n_team_A_exemplars"),
            "n_team_B_exemplars": per_track.get("n_team_B_exemplars"),
            "scene": per_track.get("scene") or {},
            "tracks": per_track.get("tracks") or {},
            "team_exemplar_urls": exemplar_urls,
        }

    files: list[dict[str, Any]] = []
    for fname in ("consolidated.mp4", "tracks.json", "team_assignments.json",
                  "players.json", "player_map.json",
                  "consolidated.stats.json", "stats.json", "scene.json",
                  "jersey_votes.json"):
        p = stage_dir / fname
        if p.exists():
            files.append(_file_entry(p, run_dir))
    if (llm_dir / "per_track.json").exists():
        files.append(_file_entry(llm_dir / "per_track.json", run_dir))

    text_files = {}
    md = stage_dir / "consolidated_tracks.md"
    if md.exists():
        text_files["consolidated_tracks.md"] = _read_text(md) or ""

    manifest = StageManifest(
        stage="track_consolidation",
        exists=True,
        stats=pipeline_stats.get("track_consolidation", {}),
        vis_dirs=vis,
        files=files,
        text_files=text_files,
    )
    # Stash thumbs in stats blob so the front-end can render a grid.
    if thumbs:
        manifest.stats = {**manifest.stats, "_track_thumbs": thumbs}
    if llm_inputs:
        manifest.stats = {**manifest.stats, "_llm_inputs": llm_inputs}
    # Original-tid → consolidated-player mapping so the front-end can
    # group thumbs first by consolidated player, then by orig track.
    pmap = _read_json(stage_dir / "player_map.json")
    if isinstance(pmap, dict):
        # Keys on disk are stringified ints; keep them strings (filenames
        # use zero-padded ints anyway, conversion happens in JS).
        manifest.stats = {**manifest.stats, "_player_map": pmap}
    return manifest


def _build_event_detection(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "event_detection"
    if not stage_dir.is_dir():
        return StageManifest(stage="event_detection", exists=False)
    vis = []
    frames_dir = stage_dir / "frames"
    if frames_dir.is_dir():
        vis.append(VisDir(
            name="events overlay",
            url_prefix="event_detection/frames",
            frames=_frame_indices(frames_dir),
            digits=_digits_of(frames_dir),
        ))
    files = []
    for fname in ("events.mp4", "events.json", "goals.json"):
        p = stage_dir / fname
        if p.exists():
            files.append(_file_entry(p, run_dir))
    stats = dict(pipeline_stats.get("event_detection", {}))
    # Inline events.json (small) so the page can show an event list.
    events = _read_json(stage_dir / "events.json")
    if isinstance(events, list):
        stats["_events"] = events[:200]  # cap so payload stays small
    return StageManifest(
        stage="event_detection",
        exists=True,
        stats=stats,
        vis_dirs=vis,
        files=files,
    )


def _build_annotated_video(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "annotated_video"
    if not stage_dir.is_dir():
        return StageManifest(stage="annotated_video", exists=False)
    files = []
    for fname in ("annotated.mp4", "annotated_web.mp4"):
        p = stage_dir / fname
        if p.exists():
            files.append(_file_entry(p, run_dir))
    return StageManifest(
        stage="annotated_video",
        exists=True,
        stats=pipeline_stats.get("annotated_video", {}),
        files=files,
    )


def _build_highlights(run_dir: Path, pipeline_stats: dict) -> StageManifest:
    stage_dir = run_dir / "highlights"
    if not stage_dir.is_dir():
        return StageManifest(stage="highlights", exists=False)
    files = []
    for p in sorted(stage_dir.iterdir()):
        if p.suffix in (".mp4", ".json"):
            files.append(_file_entry(p, run_dir))
    return StageManifest(
        stage="highlights",
        exists=True,
        stats=pipeline_stats.get("highlights", {}),
        files=files,
    )


_BUILDERS = {
    "field_registration": _build_field_registration,
    "tracking": _build_tracking,
    "track_consolidation": _build_track_consolidation,
    "event_detection": _build_event_detection,
    "annotated_video": _build_annotated_video,
    "highlights": _build_highlights,
}


def _resolve_run_dir(workspace: Workspace, run_name: str) -> Path:
    safe = _safe_filename(run_name)
    run_dir = workspace.run_dir(safe)
    if not run_dir.is_dir():
        raise HTTPException(404, f"run not found: {run_name}")
    return run_dir


def register_pipeline_results_routes(app: FastAPI, workspace: Workspace) -> None:
    """Attach read-only per-stage manifest endpoints.

    GET /api/runs/{run}/pipeline/overview  — top-level run summary
    GET /api/runs/{run}/pipeline/stage/{stage}  — manifest for one stage
    """

    @app.get("/api/runs/{run_name}/pipeline/overview")
    def pipeline_overview(run_name: str) -> JSONResponse:
        run_dir = _resolve_run_dir(workspace, run_name)
        ps = _read_json(run_dir / "pipeline_stats.json") or {}
        present = {
            stage: (run_dir / stage).is_dir()
            for stage in _BUILDERS
        }
        return JSONResponse({
            "run_name": run_name,
            "video_path": ps.get("video_path"),
            "video_name": ps.get("video_name"),
            "timestamp": ps.get("timestamp"),
            "stages_run": ps.get("stages_run", []),
            "stage_present": present,
            "totals": ps.get("stats", {}),
        })

    @app.get("/api/runs/{run_name}/pipeline/stage/{stage}")
    def stage_manifest(run_name: str, stage: str) -> JSONResponse:
        run_dir = _resolve_run_dir(workspace, run_name)
        builder = _BUILDERS.get(stage)
        if builder is None:
            raise HTTPException(404, f"unknown stage: {stage}")
        ps = _read_json(run_dir / "pipeline_stats.json") or {}
        ps_stats = ps.get("stats", {})
        manifest = builder(run_dir, ps_stats)
        # Convert dataclasses to dicts for JSON.
        out = asdict(manifest)
        # asdict turns VisDir into dict — already correct shape.
        return JSONResponse(out)
