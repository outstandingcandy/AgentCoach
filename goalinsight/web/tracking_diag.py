"""Tracking-diagnostics API: read-only endpoints for the YOLO-raw +
track_audit artefacts produced by the tracking stage.

These artefacts (per-frame JSON + diagnostic JPGs under
``<run>/tracking/yolo_raw/`` plus ``<run>/tracking/track_audit.json``)
are inspected during triage of ID switches and dropouts. The web
endpoints exposed here back the ``/tracking/{run_name}`` page in
``static/tracking.html`` — slider scrub through frames, click-to-jump
to dropout/switch events.

Routes are attached via ``register_tracking_diag_routes(app, workspace)``.
All routes are read-only (no writes or mutations), and any missing
artefact returns a 404 with a message that points the user at the
``tracking.dump_yolo_raw`` config flag / the audit script.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from ._workspace import Workspace
from .library import _safe_filename

logger = logging.getLogger(__name__)


def _frame_index_from_str(s: str) -> int:
    """Parse a frame-index path segment. All-digit, ≤7 chars (max video
    sample range we care about: 9_999_999 frames). Anything else is
    treated as a 400 — same defensive shape as ``_safe_filename``."""
    if not s.isdigit() or len(s) > 7:
        raise HTTPException(400, f"invalid frame index: {s!r}")
    return int(s)


def _resolve_yolo_raw_dir(workspace: Workspace, run_name: str) -> Path:
    """Return ``<run>/tracking/yolo_raw/`` or raise 404 with a hint."""
    run_dir = workspace.run_dir(_safe_filename(run_name))
    if not run_dir.exists():
        raise HTTPException(404, f"run not found: {run_name}")
    yolo_dir = run_dir / "tracking" / "yolo_raw"
    if not yolo_dir.exists():
        raise HTTPException(
            404,
            f"yolo_raw not generated for run {run_name!r}; re-run "
            f"tracking with tracking.dump_yolo_raw=true",
        )
    return yolo_dir


def _load_json(path: Path) -> Any:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise HTTPException(404, f"missing: {path.name}") from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"corrupt JSON: {path.name}: {exc}") from exc


def register_tracking_diag_routes(app: FastAPI, workspace: Workspace) -> None:
    """Attach read-only tracking-diagnostics endpoints to *app*.

    Endpoints:
      GET  /api/runs/{run_name}/tracking/diag/summary
      GET  /api/runs/{run_name}/tracking/diag/audit
      GET  /api/runs/{run_name}/tracking/diag/frames/{frame_index}
      GET  /api/runs/{run_name}/tracking/diag/frames/{frame_index}.jpg
      GET  /api/runs/{run_name}/tracking/diag/tracks/{frame_index}
    """

    @app.get("/api/runs/{run_name}/tracking/diag/summary")
    def diag_summary(run_name: str) -> JSONResponse:
        yolo_dir = _resolve_yolo_raw_dir(workspace, run_name)
        summary = _load_json(yolo_dir / "summary.json")

        # Augment with sorted frame indices (the page's slider snaps to
        # only-sampled frames). Older summaries don't list frames; in
        # that case fall back to filesystem scan of frames/*.json.
        frames = summary.get("frames") or []
        frame_indices = sorted(
            int(f["frame_index"])
            for f in frames if "frame_index" in f
        )
        if not frame_indices:
            frame_indices = sorted(
                int(p.stem.split("_")[-1])
                for p in (yolo_dir / "frames").glob("frame_*.json")
            )

        run_dir = workspace.run_dir(_safe_filename(run_name))
        audit_available = (run_dir / "tracking" / "track_audit.json").exists()

        return JSONResponse({
            **summary,
            "run_name": run_name,
            "frame_indices": frame_indices,
            "audit_available": audit_available,
        })

    @app.get("/api/runs/{run_name}/tracking/diag/audit")
    def diag_audit(run_name: str) -> JSONResponse:
        run_dir = workspace.run_dir(_safe_filename(run_name))
        path = run_dir / "tracking" / "track_audit.json"
        if not path.exists():
            raise HTTPException(
                404,
                f"track_audit.json missing for run {run_name!r}; "
                f"run scripts/audit_track_dropouts.py to generate it",
            )
        return JSONResponse(_load_json(path))

    @app.get("/api/runs/{run_name}/tracking/diag/frames/{frame_index}")
    def diag_frame_json(run_name: str, frame_index: str) -> JSONResponse:
        idx = _frame_index_from_str(frame_index)
        yolo_dir = _resolve_yolo_raw_dir(workspace, run_name)
        path = yolo_dir / "frames" / f"frame_{idx:06d}.json"
        if not path.exists():
            raise HTTPException(404, f"frame {idx} not in yolo_raw")
        return JSONResponse(_load_json(path))

    @app.get("/api/runs/{run_name}/tracking/diag/frames/{frame_index}/jpg")
    def diag_frame_jpg(run_name: str, frame_index: str) -> FileResponse:
        # JPG lives at /frames/{idx}/jpg rather than /frames/{idx}.jpg
        # because Starlette's default path converter binds dots into
        # the parameter (so /frames/372.jpg sends "372.jpg" to the
        # handler, not "372"). A nested segment side-steps that.
        idx = _frame_index_from_str(frame_index)
        yolo_dir = _resolve_yolo_raw_dir(workspace, run_name)
        path = yolo_dir / "frames" / f"frame_{idx:06d}.jpg"
        if not path.exists():
            raise HTTPException(404, f"frame {idx} jpg not generated")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/api/runs/{run_name}/tracking/diag/tracks/{frame_index}")
    def diag_tracks_at_frame(run_name: str, frame_index: str) -> JSONResponse:
        """Return tracks.json[frame_index] (tids + bboxes after the
        tracker), so the front-end can overlay tid labels on the JPG.

        Reads tracks.json fresh on every call — small (<1 MB on 200-frame
        runs) and the slow consumer here is the JPG, not this JSON.
        """
        idx = _frame_index_from_str(frame_index)
        run_dir = workspace.run_dir(_safe_filename(run_name))
        # Tracking diag UI shows raw integer tids; tracking/tracks.json
        # is always the raw tracker output (consolidation writes its
        # own file under track_consolidation/).
        tracks_path = run_dir / "tracking" / "tracks.json"
        if not tracks_path.exists():
            raise HTTPException(404, "tracks.json missing for run")
        data = _load_json(tracks_path)
        # tracks.json keys are stringified frame indices.
        return JSONResponse({
            "frame_index": idx,
            "tracks": data.get(str(idx), []),
        })
