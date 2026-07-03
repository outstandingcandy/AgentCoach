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

import yaml
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from ._runs import _detect_stage_completion, list_runs
from ._workspace import VIDEO_EXTS, Workspace
from ..utils.config import load_config, merge_configs

logger = logging.getLogger(__name__)

# Three user-facing video pipeline templates (fifa / futsal / children)
# live here. The library API exposes them as the "builtin" layer next
# to user-staged workspace/configs/*.yaml. The rest of the repo's
# ``configs/`` (pitches.yaml, camera_profiles.yaml) is infrastructure,
# not user-pickable.
CONFIGS_DIR = (
    Path(__file__).resolve().parents[2] / "configs" / "templates"
)
# Reference libraries the scene-setup wizard reads for its dropdowns.
# Both live at repo-root ``configs/`` — user-editable in the dev
# workflow, baked into the docker image via ``COPY configs/`` at build.
CAMERA_PROFILES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "camera_profiles.yaml"
)
PITCHES_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "pitches.yaml"
)

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


# ``sv_kp`` is the builtin/auto-downloaded pretrained model id; every
# other id is a fine-tuned run directory name under workspace/models/.
_BUILTIN_KP_MODEL_ID = "sv_kp"

# Cache of loaded KeypointDetector instances keyed by model id, so the
# 3-frame preview doesn't reload HRNet weights for each frame or each
# refresh. Small (usually 1-2 entries: SV_kp + the user's fine-tune).
_KP_DETECTOR_CACHE: dict[str, Any] = {}


def _discover_keypoint_models(workspace: Workspace) -> list[dict[str, Any]]:
    """List the builtin SV_kp plus fine-tuned keypoint models on disk.

    Fine-tuned models are produced by the annotate → train loop and land
    under ``workspace/models/keypoint_<name>/`` with the actual weights
    at a nested ``.../best_model.pt`` (see ``jobs.py`` submit_train). We
    surface the first ``best_model.pt`` found per run dir. The builtin
    entry always comes first and carries ``path: None`` (+ ``pitch_type:
    None`` — it works on any pitch) so the config layer omits
    ``keypoint_model_path`` for it and the detector auto-downloads SV_kp.

    Each fine-tuned model may carry a ``model_meta.json`` at its dir root
    with ``{"pitch_type", "label"}`` to associate it with the pitch it
    was trained for; the wizard uses ``pitch_type`` to filter the picker
    down to models that match the video's chosen pitch.
    """
    out: list[dict[str, Any]] = [{
        "id": _BUILTIN_KP_MODEL_ID,
        "label": "Pretrained (SV_kp)",
        "path": None,
        "pitch_type": None,
        "builtin": True,
    }]
    models_dir = workspace.models_dir
    if not models_dir.is_dir():
        return out
    for run_dir in sorted(models_dir.glob("keypoint_*")):
        if not run_dir.is_dir():
            continue
        weights = next(iter(sorted(run_dir.rglob("best_model.pt"))), None)
        if weights is None:
            continue
        # Association metadata (pitch_type + friendly label), if present.
        meta: dict[str, Any] = {}
        meta_path = run_dir / "model_meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text()) or {}
            except (OSError, json.JSONDecodeError):
                meta = {}
        # Default label: the dir name, or a friendly timestamp when the
        # dir is the legacy ``keypoint_<YYYYMMDD_HHMMSS>`` form.
        label = meta.get("label") or run_dir.name
        if not meta.get("label"):
            ts = run_dir.name[len("keypoint_"):]
            if len(ts) == 15 and ts[8] == "_":  # YYYYMMDD_HHMMSS
                label = (
                    f"Fine-tuned {ts[:4]}-{ts[4:6]}-{ts[6:8]} "
                    f"{ts[9:11]}:{ts[11:13]}"
                )
        out.append({
            "id": run_dir.name,
            "label": label,
            "path": str(weights.resolve()),
            "pitch_type": meta.get("pitch_type"),
            "builtin": False,
        })
    return out


def _get_keypoint_detector(model_id: str, model_path: str | None):
    """Return a cached KeypointDetector for *model_id*, loading on miss.

    ``model_path`` is None for the builtin SV_kp (auto-download); any
    other value is a fine-tuned ``best_model.pt`` path.
    """
    det = _KP_DETECTOR_CACHE.get(model_id)
    if det is not None:
        return det
    from ..field_registration.keypoint_detector import KeypointDetector

    det = KeypointDetector({
        "backend": "pnlcalib",
        "pnlcalib": {
            "weights": "SV_kp",
            "confidence_threshold": 0.3,
            "model_path": model_path,
        },
    })
    det.load_model()
    _KP_DETECTOR_CACHE[model_id] = det
    return det


def _resolve_pitch_for_video(workspace: Workspace, video_stem: str):
    """Return a ``SoccerPitch`` for *video_stem* from its saved config.

    Reads the top-level ``pitch_type`` out of the per-video
    ``workspace/configs/<stem>.yaml`` (falls back to the FIFA-sized
    default when absent/unknown). Each pitch type has its own keypoint
    world-coordinate layout AND outline, so the top-down panel must be
    built from the right one.
    """
    from ..annotation import pitch_types
    from ..annotation.pitch.geometry import SoccerPitch

    pitch_type = None
    cfg_path = workspace.configs_dir / f"{video_stem}.yaml"
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
            pitch_type = data.get("pitch_type")
        except (OSError, yaml.YAMLError):
            pitch_type = None
    if pitch_type:
        try:
            return SoccerPitch(**pitch_types.resolve(str(pitch_type)))
        except (KeyError, TypeError):
            pass
    return SoccerPitch()  # FIFA-sized default


def _render_keypoints_topdown(keypoints, pitch, conf_threshold=0.3):
    """Render a top-down view of *pitch* with detected keypoints placed on it.

    A keypoint's landmark position is pitch-dependent: the detector emits
    PnLCalib channel ids (raw, ``convert_to_soccernet=False``); each id
    names a landmark (``PITCH_POINT_TO_PNLCALIB_ID`` inverse), and that
    landmark's WORLD coordinate is computed from the active pitch's
    dimensions (``keypoint_utils.get_hrnet_keypoints_2d`` reads the
    live ``PITCH_POINTS``). So on a futsal pitch id 0 (TL_PITCH_CORNER)
    lands at (-20, 10), on FIFA at (-52.5, 34). The outline is drawn from
    the SAME pitch, so points and lines are consistent. Returns a BGR
    image the caller resizes for hconcat.

    NOTE: mutates process-global active-pitch state via ``set_active_pitch``
    — matches the annotate web module's pattern; fine for this single-user
    tool where the preview is a quick synchronous call.
    """
    import cv2
    import numpy as np  # noqa: F401 — used by pitch_diagram helpers

    from ..annotation import keypoint_utils, pitch_constants
    from ..annotation.pitch.keypoints import PITCH_POINT_TO_PNLCALIB_ID
    from ..annotation.pitch_diagram import draw_pitch_structure, make_pitch_canvas

    # Make this pitch active so keypoint world coords resolve against it.
    pitch_constants.set_active_pitch(pitch)
    kp_world = keypoint_utils.get_hrnet_keypoints_2d()  # name -> (x, y) on this pitch
    id_to_name = {v: k for k, v in PITCH_POINT_TO_PNLCALIB_ID.items()}

    scale = 12  # px per metre — fine for a preview panel
    margin = 20
    img, to_px, _w, _h = make_pitch_canvas(scale, margin, pitch=pitch)
    draw_pitch_structure(
        img, to_px, scale=scale, color=(255, 255, 255), thickness=2,
        draw_landmarks=True, landmark_radius=3, draw_arcs=True, pitch=pitch,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    n_shown = 0
    for kp in keypoints:
        if kp.get("confidence", 0) < conf_threshold:
            continue
        name = id_to_name.get(kp["id"])
        world = kp_world.get(name) if name else None
        if world is None:
            continue
        px, py = to_px(world[0], world[1])
        conf = float(kp.get("confidence", 0))
        green = int(min(255, conf * 2 * 255))
        red = int(max(0, (1 - conf) * 2 * 255))
        cv2.circle(img, (px, py), 5, (0, green, red), -1)
        cv2.circle(img, (px, py), 5, (0, 0, 0), 1)
        cv2.putText(img, str(kp["id"]), (px + 6, py + 3), font, 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        n_shown += 1

    cv2.putText(img, f"Top-down: {n_shown} pts", (8, 20), font, 0.5,
                (255, 255, 255), 2)
    cv2.putText(img, f"Top-down: {n_shown} pts", (8, 20), font, 0.5,
                (0, 255, 0), 1)
    return img


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

    Primary source: ``annotations/index.json`` (written by the
    annotator UI), keyed by video stem with
    ``{frames: [...], last_modified}`` per video.

    Fallback: any ``annotations/<stem>/frame_*.json`` files on disk —
    so externally-staged annotations (e.g. copied into a docker
    workspace from a pre-baked location, or shared between users via
    rsync) still show as "annotated" in the library UI even when no
    index.json was written.
    """
    idx_path = workspace.annotations_dir / "index.json"
    out: dict[str, dict[str, Any]] = {}
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text())
        except (OSError, json.JSONDecodeError):
            idx = {}
        for stem, info in (idx.get("annotations") or {}).items():
            frames = info.get("frames") or []
            out[stem] = {
                "frame_count": len(frames),
                "last_modified": info.get("last_modified"),
            }

    # Filesystem fallback for stems missing from the index — gives the
    # library a useful annotation summary even for hand-staged dirs.
    # ``last_modified`` must be an ISO string (not a float) because the
    # frontend's fmtTimestamp() calls .replace() on it and would crash
    # the entire video card render on a float.
    if workspace.annotations_dir.exists():
        from datetime import datetime, timezone
        for sub in workspace.annotations_dir.iterdir():
            if not sub.is_dir() or sub.name in out:
                continue
            frames = sorted(sub.glob("frame_*.json"))
            if not frames:
                continue
            mtime = max(f.stat().st_mtime for f in frames)
            out[sub.name] = {
                "frame_count": len(frames),
                "last_modified": datetime.fromtimestamp(
                    mtime, tz=timezone.utc,
                ).isoformat(),
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
                video_config_path = workspace.config_for(p)
                out.append({
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "cover_url": f"/api/library/videos/{p.name}/cover.jpg",
                    "annotation": annotations.get(p.stem),
                    "runs": runs,
                    "latest_run": runs[0] if runs else None,
                    "has_video_config": video_config_path.exists(),
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

        # Per-video saved config wins over the dropdown selection when
        # present. Authored from the library Edit-Config modal; lives at
        # workspace/configs/<stem>.yaml. Unless the request explicitly
        # asks for a different base ("use_video_config": False), we
        # prefer the saved file so a user who edited a config doesn't
        # have to remember to re-pick it on every run.
        use_video_cfg = bool(payload.get("use_video_config", True))
        config_path: Path | None = None
        if video_path is not None and use_video_cfg:
            vc = workspace.config_for(video_path)
            if vc.exists():
                config_path = vc
        if config_path is None:
            # Resolution order matches list_configs / get_config_raw:
            # workspace-staged configs win over the builtin ones, so a
            # user who dropped a custom yaml into workspace/configs/
            # can pick it from the dropdown without it landing on 404.
            safe = _safe_filename(str(config_name))
            ws_cfg = workspace.configs_dir / safe
            config_path = ws_cfg if ws_cfg.exists() else (CONFIGS_DIR / safe)
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
        # request-per-click.
        #
        # Two sources, in priority order:
        #   1. ``workspace/configs/*.yaml`` — user-staged configs
        #      (per-video overrides, the 3 templates the docker
        #      entrypoint seeds, anything the user has saved).
        #   2. ``configs/templates/*.yaml`` — the 3 built-in templates
        #      (fifa / futsal / children) shipped with the install.
        out = []
        seen: set[str] = set()
        if workspace.configs_dir.is_dir():
            for p in sorted(workspace.configs_dir.glob("*.yaml")):
                stages = _read_pipeline_stages(p)
                out.append({
                    "name": p.name, "path": str(p), "stages": stages,
                    "source": "workspace",
                })
                seen.add(p.name)
        if CONFIGS_DIR.is_dir():
            for p in sorted(CONFIGS_DIR.glob("*.yaml")):
                if p.name in seen:
                    continue
                stages = _read_pipeline_stages(p)
                out.append({
                    "name": p.name, "path": str(p), "stages": stages,
                    "source": "builtin",
                })
        return JSONResponse(out)

    @app.get("/api/library/configs/{name}/raw")
    def get_config_raw(name: str) -> JSONResponse:
        """Return the raw YAML text of a base config under ./configs.

        Used by the library Edit-Config modal to seed the textarea from
        whichever base config the user picked. ``name`` is matched
        against the file name (not the path) and sanitised against path
        traversal.

        Resolution order matches ``list_configs``: workspace overrides
        win, then the builtin repo configs.
        """
        safe = _safe_filename(name)
        ws_path = workspace.configs_dir / safe
        path = ws_path if ws_path.exists() else (CONFIGS_DIR / safe)
        if not path.exists():
            raise HTTPException(404, f"config not found: {name}")
        try:
            text = path.read_text()
        except OSError as exc:
            raise HTTPException(500, f"failed to read {name}: {exc}") from exc
        return JSONResponse({"name": safe, "path": str(path), "text": text})

    @app.get("/api/library/pitch_profiles")
    def list_pitch_profiles() -> JSONResponse:
        """Return ``[{key, label, description}]`` for the scene-setup wizard.

        Reads ``configs/pitches.yaml`` — the same file
        ``goalinsight.annotation.pitch_types`` resolves against, so the
        keys shown here are guaranteed to be valid ``pitch_type:`` values.
        """
        out = []
        try:
            data = yaml.safe_load(PITCHES_PATH.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(500, f"pitches.yaml unreadable: {exc}") from exc
        for key, entry in (data.get("profiles") or {}).items():
            if not isinstance(entry, dict):
                continue
            out.append({
                "key": key,
                "label": entry.get("label", key),
                "description": entry.get("description", ""),
            })
        return JSONResponse(sorted(out, key=lambda p: p["key"]))

    @app.get("/api/library/camera_profiles")
    def list_camera_profiles() -> JSONResponse:
        """Return ``[{key, label, image_size}]`` for the wizard camera dropdown.

        Excludes the K/dist_coeffs numeric matrices — those aren't user-
        facing. The wizard only needs the key + a human-readable label.
        """
        out = []
        try:
            data = yaml.safe_load(CAMERA_PROFILES_PATH.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise HTTPException(500, f"camera_profiles.yaml unreadable: {exc}") from exc
        for key, entry in (data.get("profiles") or {}).items():
            if not isinstance(entry, dict):
                continue
            out.append({
                "key": key,
                "label": entry.get("label", key),
                "image_size": entry.get("image_size"),
            })
        return JSONResponse(sorted(out, key=lambda p: p["key"]))

    @app.get("/api/library/keypoint_models")
    def list_keypoint_models() -> JSONResponse:
        """Return ``[{id, label, path, builtin}]`` for the wizard picker.

        The builtin SV_kp is always first; fine-tuned models discovered
        under ``workspace/models/keypoint_*/`` follow. Used by the
        scene-setup calibrate step's "use a keypoint model" option.
        """
        return JSONResponse(_discover_keypoint_models(workspace))

    @app.get("/api/library/videos/{name}/keypoint_preview.jpg")
    def keypoint_preview(name: str, frame: int = 0,
                         model: str = _BUILTIN_KP_MODEL_ID) -> Response:
        """Overlay a keypoint model's detections on one frame of *name*.

        The wizard renders three of these (at spread-out frame indices)
        so the user can eyeball whether the chosen model lands keypoints
        correctly on this pitch before launching. ``model`` is a model id
        from ``/api/library/keypoint_models``; ``frame`` is a 0-based
        index (clamped to the clip length).
        """
        try:
            import cv2
            import numpy as np
        except ImportError as exc:  # pragma: no cover
            raise HTTPException(500, "opencv unavailable") from exc

        safe = _safe_filename(name)
        video_path = workspace.videos_dir / safe
        if not video_path.exists():
            raise HTTPException(404, f"video not found: {name}")

        # Resolve the model id → weights path via the same discovery the
        # dropdown uses, so an unknown/renamed id can't smuggle a path in.
        models = {m["id"]: m for m in _discover_keypoint_models(workspace)}
        entry = models.get(model)
        if entry is None:
            raise HTTPException(404, f"unknown keypoint model: {model}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(500, "failed to open video")
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            idx = frame
            if total > 0:
                idx = max(0, min(int(frame), total - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok or img is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = cap.read()
            if not ok or img is None:
                raise HTTPException(500, "failed to read frame")
        finally:
            cap.release()

        try:
            det = _get_keypoint_detector(entry["id"], entry.get("path"))
        except Exception as exc:  # noqa: BLE001
            logger.exception("failed to init keypoint detector")
            raise HTTPException(500, f"detector init failed: {exc}") from exc

        try:
            # Raw PnLCalib channel ids (NOT soccernet) so the id ↔ landmark
            # ↔ pitch-world lookup in the top-down panel lines up, and the
            # frame overlay labels match the panel labels.
            keypoints = det.detect(img, convert_to_soccernet=False)
        except Exception as exc:  # noqa: BLE001
            logger.exception("keypoint detection failed")
            raise HTTPException(500, f"detection failed: {exc}") from exc

        from ..field_registration.shared_vis import draw_vis_keypoints

        vis = draw_vis_keypoints(img, keypoints, conf_threshold=0.3)

        # Stitch a top-down pitch beside the frame so the user can read
        # off which pitch landmark each detected keypoint maps to. The
        # panel is built from the video's configured pitch (futsal / kids
        # / fifa) — each pitch has its own landmark world coords AND
        # outline — scaled to the frame height so hconcat lines up.
        try:
            pitch = _resolve_pitch_for_video(workspace, video_path.stem)
            topdown = _render_keypoints_topdown(keypoints, pitch, conf_threshold=0.3)
            fh = vis.shape[0]
            tw = max(1, int(topdown.shape[1] * fh / topdown.shape[0]))
            topdown = cv2.resize(topdown, (tw, fh), interpolation=cv2.INTER_AREA)
            # A thin separator so the two panels read as distinct views.
            sep = np.full((fh, 4, 3), 30, dtype=np.uint8)
            vis = cv2.hconcat([vis, sep, topdown])
        except Exception:  # noqa: BLE001 — panel is a nicety, never fatal
            logger.exception("top-down panel render failed; serving frame only")

        ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            raise HTTPException(500, "jpeg encode failed")
        n_shown = sum(1 for kp in keypoints if kp.get("confidence", 0) >= 0.3)
        return Response(
            content=bytes(buf),
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-store",
                "X-Keypoints-Detected": str(n_shown),
            },
        )

    @app.get("/api/library/videos/{name}/config")
    def get_video_config(name: str) -> JSONResponse:
        """Return the saved per-video config for *name* (or empty meta).

        Response shape: ``{stem, exists, path, text}``. ``exists=False`` +
        empty ``text`` means the user hasn't saved a custom config for
        this video yet — the UI uses that to know whether to pre-fill
        from a base config when opening the modal.
        """
        safe = _safe_filename(name)
        video_path = workspace.videos_dir / safe
        if not video_path.exists():
            raise HTTPException(404, f"video not found: {name}")
        cfg_path = workspace.config_for(video_path)
        exists = cfg_path.exists()
        text = cfg_path.read_text() if exists else ""
        return JSONResponse({
            "stem": video_path.stem,
            "exists": exists,
            "path": str(cfg_path),
            "text": text,
        })

    @app.put("/api/library/videos/{name}/config")
    async def save_video_config(name: str, request: Request) -> JSONResponse:
        """Persist a per-video pipeline config under workspace/configs/.

        Body: ``{text: "<yaml>"}``. We validate that the body is parseable
        YAML and that the top-level is a mapping (so a typo can't write a
        file that crashes the pipeline runner later). Empty text deletes
        the saved config — the run will fall back to the chosen base
        config.
        """
        safe = _safe_filename(name)
        video_path = workspace.videos_dir / safe
        if not video_path.exists():
            raise HTTPException(404, f"video not found: {name}")

        payload = await request.json()
        text = payload.get("text")
        if text is None:
            raise HTTPException(400, "missing 'text' field")
        text = str(text)

        cfg_path = workspace.config_for(video_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        if text.strip() == "":
            # Empty text = "clear the per-video override".
            if cfg_path.exists():
                cfg_path.unlink()
            return JSONResponse({
                "stem": video_path.stem,
                "exists": False,
                "path": str(cfg_path),
                "text": "",
            })

        try:
            import yaml  # local import keeps cold-start fast
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise HTTPException(400, f"invalid YAML: {exc}") from exc
        if data is not None and not isinstance(data, dict):
            raise HTTPException(
                400,
                "config root must be a mapping (a YAML object), "
                f"got {type(data).__name__}",
            )

        cfg_path.write_text(text)
        return JSONResponse({
            "stem": video_path.stem,
            "exists": True,
            "path": str(cfg_path),
            "text": text,
        })

    @app.post("/api/library/videos/{name}/scene-setup")
    async def scene_setup(name: str, request: Request) -> JSONResponse:
        """Wizard-driven per-video config write.

        Accepts a structured payload (pitch, camera profile, fixed vs
        PTZ, physical camera position for fixed rigs) and materialises
        the answers as a per-video ``workspace/configs/<stem>.yaml``.
        Seeds from the matching pitch template
        (``configs/templates/<pitch_type>.yaml``) so downstream stages
        see a fully-populated config, then overlays the wizard's
        physical / camera / mode overrides via ``merge_configs``.

        Payload variants (the wizard sends several, one per step; each
        merges into the same per-video config):

        * basics — ``pitch_type`` + ``camera_profile`` +
          ``camera_position`` + ``focal_hfov_deg_bounds``.
        * model  — ``keypoint_model_id`` (+ ``keypoint_model_path``).
        * moves  — ``camera_moves`` bool → backend (fixed_camera / physical).
        * vis    — ``visualizations`` toggles.
        * ``{"first_annotation": true, "annotation_frame_path": "..."}``
          — the annotator posts this after the user saves a calibration
          frame so the fixed_camera runner can replay that pose.
        """
        safe = _safe_filename(name)
        video_path = workspace.videos_dir / safe
        if not video_path.exists():
            raise HTTPException(404, f"video not found: {name}")

        payload = await request.json()
        cfg_path = workspace.config_for(video_path)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing per-video config if present, else seed from the
        # template matching the payload's pitch_type (or FIFA fallback).
        if cfg_path.exists():
            base = load_config(cfg_path)
        else:
            template_name = payload.get("pitch_type") or "fifa"
            template_map = {
                "fifa": "fifa.yaml",
                "futsal": "futsal.yaml",
                "kids_soccer": "children.yaml",
            }
            tpl_path = CONFIGS_DIR / template_map.get(template_name, "fifa.yaml")
            base = load_config(tpl_path) if tpl_path.exists() else {}

        overlay: dict[str, Any] = {}
        if pt := payload.get("pitch_type"):
            overlay["pitch_type"] = pt

        # The reordered wizard POSTs this endpoint several times, each
        # turn carrying one slice of the scene. All slices merge into the
        # same per-video config under ``field_registration``:
        #
        #   step 2 (basics)   pitch_type + camera_profile + camera_position
        #                     + focal_hfov_deg_bounds → physical block
        #   step 3 (model)    keypoint_model_id/path → keypoint_detection block
        #   step 4 (moves)    camera_moves bool → backend + lock flags:
        #                       moves=false → backend: fixed_camera (solve
        #                         one pose from the model/annotation, reuse)
        #                       moves=true  → backend: physical (re-estimate
        #                         the camera every frame with the model)
        # Backends are independent — no runtime delegation. Both read the
        # camera profile + position from the shared ``physical`` block.
        fr_phys: dict[str, Any] = {}
        if profile := payload.get("camera_profile"):
            fr_phys["camera_profile"] = profile
        if pos := payload.get("camera_position"):
            fr_phys["camera_position"] = [float(v) for v in pos]
        if hfov := payload.get("focal_hfov_deg_bounds"):
            fr_phys["focal_hfov_deg_bounds"] = [float(v) for v in hfov]

        backend: str | None = None
        if "camera_moves" in payload:
            if payload.get("camera_moves"):
                backend = "physical"
                fr_phys["lock_camera_position"] = False
                fr_phys["position_bounds_m"] = [8.0, 8.0, 3.0]
                fr_phys["joint_optimize"] = True
            else:
                backend = "fixed_camera"
                fr_phys["lock_camera_position"] = True
                fr_phys["position_bounds_m"] = [0.0, 0.0, 0.0]
                fr_phys["joint_optimize"] = False

        # First-annotation hook: the annotator calls back with the saved
        # frame path so the fixed_camera runner can replay that pose.
        if payload.get("first_annotation") and (afp := payload.get("annotation_frame_path")):
            fr_phys["annotation_frame_path"] = str(afp)

        # Keypoint-model path → shared ``keypoint_detection`` block, read by
        # both backends (physical per-frame, fixed_camera one-shot solve).
        # The builtin SV_kp entry posts a null path; we then omit
        # ``keypoint_model_path`` so the detector auto-downloads SV_kp.
        if "keypoint_model_id" in payload:
            kp_path = payload.get("keypoint_model_path")
            if kp_path:
                overlay.setdefault("field_registration", {}).setdefault(
                    "keypoint_detection", {},
                )["keypoint_model_path"] = str(kp_path)

        if fr_phys or backend:
            overlay.setdefault("field_registration", {})
            if backend:
                overlay["field_registration"]["backend"] = backend
            if fr_phys:
                overlay["field_registration"].setdefault("physical", {}).update(
                    fr_phys,
                )

        # Per-stage visualization toggles from the wizard's step 4.
        # Each key is optional — when unset, the merged base config's
        # value stays (which for the shipped templates defaults to on).
        vis = payload.get("visualizations") or {}
        if "field_registration" in vis:
            on = bool(vis["field_registration"])
            # ``vis_interval: 0`` disables per-frame vis JPGs in the
            # physical / fixed_camera runners; 30 is the template default.
            overlay.setdefault(
                "field_registration", {}
            ).setdefault("physical", {})["vis_interval"] = 30 if on else 0
        if "tracking" in vis:
            on = bool(vis["tracking"])
            overlay.setdefault("tracking", {})["vis_frames_enabled"] = on
            overlay["tracking"]["dump_yolo_raw"] = on
        if "ball_diag" in vis:
            overlay.setdefault("tracking", {})["dump_ball_diag"] = bool(
                vis["ball_diag"],
            )

        merged = merge_configs(base, overlay) if overlay else base

        try:
            text = yaml.safe_dump(merged, sort_keys=False, width=100)
        except yaml.YAMLError as exc:
            raise HTTPException(500, f"failed to serialize config: {exc}") from exc
        cfg_path.write_text(text)
        return JSONResponse({
            "stem": video_path.stem,
            "path": str(cfg_path),
            "text": text,
            "backend": merged.get("field_registration", {}).get("backend"),
        })
