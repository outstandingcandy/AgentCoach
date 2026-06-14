"""Library API: video upload, video listing, run listing, config listing.

Routes are attached via ``register_library_routes(app, workspace)``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from ._runs import _detect_stage_completion, list_runs
from ._workspace import VIDEO_EXTS, Workspace

logger = logging.getLogger(__name__)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    """Reject path traversal; collapse other unsafe chars to underscore.

    The user controls the upload filename and the run name; both feed
    directly into on-disk paths so we strip anything that could escape the
    workspace before using them.
    """
    base = Path(name).name  # drop any leading dirs
    if not base or base.startswith("."):
        raise HTTPException(400, f"invalid name: {name!r}")
    cleaned = _SAFE_NAME_RE.sub("_", base)
    if not cleaned or cleaned in {".", ".."}:
        raise HTTPException(400, f"invalid name: {name!r}")
    return cleaned


def _read_pipeline_stages(config_path: Path) -> list[str]:
    """Read the ``pipeline.stages`` list out of a YAML config.

    Returns an empty list when the file's missing, unparseable, or
    doesn't declare a stages list. Caller falls back to defaults.
    """
    if not config_path.exists():
        return []
    try:
        import yaml  # local import: keeps startup fast
    except ImportError:
        return []
    try:
        data = yaml.safe_load(config_path.read_text()) or {}
    except yaml.YAMLError:
        return []
    pipeline = data.get("pipeline") or {}
    stages = pipeline.get("stages") or []
    return [str(s) for s in stages if s]


def _extract_cover_frame(video_path: Path, dest: Path) -> bool:
    """Grab a single frame ~10% into the video and write it as JPEG.

    10% in (rather than frame 0) avoids the all-black opening / fade-in
    that's common in broadcast clips. Returns False when cv2 can't open
    the file or seek fails — caller surfaces an HTTP error.
    """
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return False
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = max(0, int(n * 0.1)) if n else 0
        if target:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fall back to the first frame if the seek landed past EOF.
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            return False
        # Downscale wide frames so the cover stays small (~50 KB).
        h, w = frame.shape[:2]
        max_w = 480
        if w > max_w:
            scale = max_w / float(w)
            frame = cv2.resize(frame, (max_w, int(h * scale)),
                               interpolation=cv2.INTER_AREA)
        return bool(cv2.imwrite(str(dest), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 80]))
    finally:
        cap.release()


def _probe_video_meta(path: Path) -> dict[str, Any]:
    """Best-effort: open with cv2 to read fps/frame_count/wh.

    Returns an empty dict if cv2 fails; callers tolerate missing fields.
    """
    try:
        import cv2  # local import: keep startup fast for non-video paths
    except Exception:  # noqa: BLE001
        return {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "fps": float(fps),
            "frame_count": n,
            "duration_s": (n / fps) if fps else 0.0,
            "width": w,
            "height": h,
        }
    finally:
        cap.release()


def _annotation_summary(workspace: Workspace) -> dict[str, dict[str, Any]]:
    """Read the annotation index once and return a stem→summary map.

    The index format (``annotations/index.json``) is keyed by video stem
    (``Path(video).stem``) and stores ``{frames: [...], last_modified}``
    per video. We surface frame count + last_modified for the library UI.
    """
    idx_path = workspace.annotations_dir / "index.json"
    if not idx_path.exists():
        return {}
    try:
        idx = json.loads(idx_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    raw = idx.get("annotations") or {}
    out: dict[str, dict[str, Any]] = {}
    for stem, info in raw.items():
        frames = info.get("frames") or []
        out[stem] = {
            "frame_count": len(frames),
            "last_modified": info.get("last_modified"),
        }
    return out


def _runs_by_video(workspace: Workspace) -> dict[str, list[dict[str, Any]]]:
    """For each video stem, return runs that reference it, newest first.

    Match is by ``Path(pipeline_stats.video_path).stem`` falling back to
    ``run.json`` when pipeline_stats is missing (run created but no
    stage has finished yet). Each entry carries the run name, the
    timestamp of the most recent pipeline run, and the stage map the
    library + match index already use.
    """
    if not workspace.runs_dir.is_dir():
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run_dir in workspace.runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        video_stem: str | None = None
        timestamp: str | None = None
        stages_run: list[str] = []
        stats_path = run_dir / "pipeline_stats.json"
        if stats_path.exists():
            try:
                stats = json.loads(stats_path.read_text())
                vp = stats.get("video_path")
                if vp:
                    video_stem = Path(vp).stem
                timestamp = stats.get("timestamp")
                stages_run = list(stats.get("stages_run") or [])
            except (OSError, json.JSONDecodeError):
                pass
        if video_stem is None:
            run_json = run_dir / "run.json"
            if run_json.exists():
                try:
                    rj = json.loads(run_json.read_text())
                    vp = rj.get("video_path")
                    if vp:
                        video_stem = Path(vp).stem
                except (OSError, json.JSONDecodeError):
                    pass
        if video_stem is None:
            continue
        grouped.setdefault(video_stem, []).append({
            "run_name": run_dir.name,
            "timestamp": timestamp,
            "stages_run": stages_run,
            "stages": _detect_stage_completion(run_dir),
        })
    # Sort each group with the freshest run first; runs without a
    # timestamp sink to the bottom alphabetically by name.
    for stem, entries in grouped.items():
        entries.sort(
            key=lambda e: (e["timestamp"] is None,
                           e["timestamp"] or "",
                           e["run_name"]),
            reverse=False,
        )
        # Newest-first: invert the time component without disturbing
        # the alphabetical fallback for null timestamps.
        entries.sort(
            key=lambda e: (e["timestamp"] is not None, e["timestamp"] or ""),
            reverse=True,
        )
    return grouped


def register_library_routes(app: FastAPI, workspace: Workspace) -> None:
    @app.get("/api/library/videos")
    def list_videos() -> JSONResponse:
        annotations = _annotation_summary(workspace)
        runs_by_video = _runs_by_video(workspace)
        out: list[dict[str, Any]] = []
        for ext in VIDEO_EXTS:
            for p in sorted(workspace.videos_dir.glob(f"*{ext}")):
                meta = _probe_video_meta(p)
                runs = runs_by_video.get(p.stem, [])
                out.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "cover_url": f"/api/library/videos/{p.name}/cover.jpg",
                    "annotation": annotations.get(p.stem),
                    "runs": runs,
                    "latest_run": runs[0] if runs else None,
                    **meta,
                })
        return JSONResponse(out)

    @app.get("/api/library/videos/{name}/cover.jpg")
    def video_cover(name: str) -> FileResponse:
        """Return a single representative frame as the video cover.

        Caches the JPEG under ``workspace/videos/.covers/<stem>.jpg``;
        regenerates if the source video is newer than the cache, so
        re-uploading a video with the same name doesn't serve the
        stale cover.
        """
        safe = _safe_filename(name)
        video_path = workspace.videos_dir / safe
        if not video_path.exists():
            raise HTTPException(404, f"video not found: {name}")
        cover_dir = workspace.videos_dir / ".covers"
        cover_dir.mkdir(parents=True, exist_ok=True)
        cover_path = cover_dir / f"{video_path.stem}.jpg"
        try:
            need_rebuild = (
                not cover_path.exists()
                or cover_path.stat().st_mtime < video_path.stat().st_mtime
            )
        except OSError:
            need_rebuild = True
        if need_rebuild:
            ok = _extract_cover_frame(video_path, cover_path)
            if not ok:
                raise HTTPException(500, "failed to extract cover frame")
        return FileResponse(
            cover_path, media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=300"},
        )

    @app.post("/api/library/upload")
    async def upload_video(file: UploadFile) -> JSONResponse:
        if not file.filename:
            raise HTTPException(400, "missing filename")
        suffix = Path(file.filename).suffix.lower()
        if suffix not in VIDEO_EXTS:
            raise HTTPException(
                400,
                f"unsupported video extension {suffix!r}; allowed={VIDEO_EXTS}",
            )
        name = _safe_filename(file.filename)
        dest = workspace.videos_dir / name
        if dest.exists():
            raise HTTPException(409, f"video {name!r} already exists")
        with dest.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
        await file.close()
        return JSONResponse({
            "name": dest.name,
            "stem": dest.stem,
            "path": str(dest),
            "size_bytes": dest.stat().st_size,
            **_probe_video_meta(dest),
        })

    @app.get("/api/library/runs")
    def list_runs_route() -> JSONResponse:
        return JSONResponse(list_runs(workspace))

    @app.post("/api/library/runs")
    async def create_run(request: Request) -> JSONResponse:
        payload = await request.json()
        run_name = _safe_filename(str(payload.get("run_name") or ""))
        video_name = payload.get("video_name")
        config_name = payload.get("config_name") or "default.yaml"

        run_dir = workspace.run_dir(run_name)
        if run_dir.exists():
            raise HTTPException(409, f"run {run_name!r} already exists")

        video_path: Path | None = None
        if video_name:
            video_path = workspace.videos_dir / _safe_filename(str(video_name))
            if not video_path.exists():
                raise HTTPException(404, f"video not found: {video_name}")

        config_path = CONFIGS_DIR / _safe_filename(str(config_name))
        if not config_path.exists():
            raise HTTPException(404, f"config not found: {config_name}")

        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps({
            "run_name": run_name,
            "video_path": str(video_path) if video_path else None,
            "config_path": str(config_path),
        }, indent=2))
        return JSONResponse({
            "run_name": run_name,
            "run_dir": str(run_dir),
            "video_path": str(video_path) if video_path else None,
            "config_path": str(config_path),
        }, status_code=201)

    @app.get("/api/library/runs/{run_name}")
    def get_run(run_name: str) -> JSONResponse:
        safe = _safe_filename(run_name)
        run_dir = workspace.run_dir(safe)
        if not run_dir.is_dir():
            raise HTTPException(404, f"run not found: {run_name}")
        run_json = run_dir / "run.json"
        meta = {}
        if run_json.exists():
            try:
                meta = json.loads(run_json.read_text())
            except (OSError, json.JSONDecodeError):
                meta = {}
        return JSONResponse({
            "run_name": safe,
            "run_dir": str(run_dir),
            **meta,
        })

    @app.delete("/api/library/runs/{run_name}")
    def delete_run(run_name: str) -> JSONResponse:
        safe = _safe_filename(run_name)
        run_dir = workspace.run_dir(safe)
        if not run_dir.is_dir():
            raise HTTPException(404, f"run not found: {run_name}")
        shutil.rmtree(run_dir)
        return JSONResponse({"deleted": safe})

    @app.get("/api/library/configs")
    def list_configs() -> JSONResponse:
        # Resolve each config's pipeline.stages list inline so the
        # library can submit a pipeline job without a second
        # request-per-click. Configs that don't declare stages fall
        # back to default.yaml's list.
        out = []
        if not CONFIGS_DIR.is_dir():
            return JSONResponse(out)
        default_stages = _read_pipeline_stages(CONFIGS_DIR / "default.yaml")
        for p in sorted(CONFIGS_DIR.glob("*.yaml")):
            stages = _read_pipeline_stages(p) or default_stages
            out.append({"name": p.name, "path": str(p), "stages": stages})
        return JSONResponse(out)
