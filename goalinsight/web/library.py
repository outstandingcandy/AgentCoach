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
from fastapi.responses import JSONResponse

from ._runs import list_runs
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


def register_library_routes(app: FastAPI, workspace: Workspace) -> None:
    @app.get("/api/library/videos")
    def list_videos() -> JSONResponse:
        out: list[dict[str, Any]] = []
        for ext in VIDEO_EXTS:
            for p in sorted(workspace.videos_dir.glob(f"*{ext}")):
                meta = _probe_video_meta(p)
                out.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    **meta,
                })
        return JSONResponse(out)

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
        out = []
        if CONFIGS_DIR.is_dir():
            for p in sorted(CONFIGS_DIR.glob("*.yaml")):
                out.append({"name": p.name, "path": str(p)})
        return JSONResponse(out)
