"""Background job manager for the unified web app.

Three job kinds:

- ``pipeline``: invokes ``Pipeline.from_stage_names(stages, config).run(...)``
  in a worker thread. Stage outputs land under ``workspace/runs/<run>/``.
- ``train``: ``scripts/train_finetune.py --kind {keypoint|line}`` as a
  subprocess. Output weights land under ``workspace/models/<ts>/``.
- ``render_consolidated``: ``scripts/render_consolidated_tracking.py`` as a
  subprocess. Output overlay lands at
  ``workspace/runs/<run>/track_consolidation/players_overlay.mp4``.

All jobs stream their stdout/stderr to a per-job log file under
``workspace/runs/<run>/logs/<job_kind>-<job_id>.log`` (or
``workspace/models/<ts>/log.txt`` for training). The log is tailable via
``GET /api/jobs/{id}/log`` and SSE-streamable via ``GET /api/jobs/{id}/stream``.

Job state is persisted to ``workspace/jobs.json`` so the history survives
process restarts; in-flight jobs are not resumed (they reappear as
``failed`` if interrupted).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse

from ..pipeline import Pipeline
from ..utils.config import get_default_config, load_config, merge_configs
from ._workspace import Workspace

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class JobRecord:
    id: str
    kind: str  # 'pipeline' | 'train' | 'render_consolidated'
    run_name: str | None
    status: str  # 'pending' | 'running' | 'done' | 'failed'
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    log_path: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    result: dict[str, Any] | None = None

    def to_public(self) -> dict[str, Any]:
        d = asdict(self)
        # Don't leak the full filesystem path for log; client uses the API.
        return d


class JobManager:
    def __init__(self, workspace: Workspace, *, max_workers: int = 2):
        self.workspace = workspace
        self.jobs: dict[str, JobRecord] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.Lock()
        self._restore()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _restore(self) -> None:
        if not self.workspace.jobs_file.exists():
            return
        try:
            data = json.loads(self.workspace.jobs_file.read_text())
        except (OSError, json.JSONDecodeError):
            logger.exception("could not load jobs.json")
            return
        for raw in data.get("jobs", []):
            try:
                rec = JobRecord(**raw)
            except TypeError:
                continue
            # Anything that was 'pending' or 'running' before restart did
            # not survive: mark it failed so the UI does not show a
            # phantom in-flight job.
            if rec.status in ("pending", "running"):
                rec.status = "failed"
                rec.error = (rec.error or "") + " (interrupted by restart)"
                rec.finished_at = rec.finished_at or time.time()
            self.jobs[rec.id] = rec

    def _persist(self) -> None:
        with self._lock:
            payload = {"jobs": [j.to_public() for j in self.jobs.values()]}
        try:
            self.workspace.jobs_file.write_text(json.dumps(payload, indent=2))
        except OSError:
            logger.exception("could not write jobs.json")

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit_pipeline(
        self,
        run_name: str,
        stages: list[str],
        config_path: Path,
        *,
        video_path: Path,
        skip_existing: bool = True,
        keypoint_model: Path | None = None,
        no_viz: bool = False,
    ) -> JobRecord:
        run_dir = self.workspace.run_dir(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)
        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        rec = self._new_record(
            "pipeline", run_name,
            payload={
                "stages": stages,
                "config_path": str(config_path),
                "video_path": str(video_path),
                "skip_existing": skip_existing,
                "keypoint_model": str(keypoint_model) if keypoint_model else None,
                "no_viz": no_viz,
            },
        )
        rec.log_path = str(log_dir / f"pipeline-{rec.id}.log")
        self._dispatch(rec, self._run_pipeline)
        return rec

    def submit_train(
        self,
        *,
        kind: str,                      # "keypoint" | "line"
        annotations_dir: Path,
        pretrained: Path,
        extra_argv: list[str] | None = None,
        remote: bool = False,
    ) -> JobRecord:
        if kind not in {"keypoint", "line"}:
            raise ValueError(f"kind must be 'keypoint' or 'line', got {kind!r}")
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.workspace.models_dir / f"{kind}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = self._new_record(
            "train", None,
            payload={
                "kind": kind,
                "annotations_dir": str(annotations_dir),
                "pretrained": str(pretrained),
                "output_dir": str(out_dir),
                "remote": remote,
                "extra_argv": list(extra_argv or []),
            },
        )
        rec.log_path = str(out_dir / "log.txt")
        self._dispatch(rec, self._run_train)
        return rec

    def submit_render_consolidated(
        self,
        run_name: str,
        *,
        video_path: Path,
        max_frames: int | None = None,
    ) -> JobRecord:
        run_dir = self.workspace.run_dir(run_name)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_dir}")
        log_dir = run_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        out_path = run_dir / "track_consolidation" / "players_overlay.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rec = self._new_record(
            "render_consolidated", run_name,
            payload={
                "video_path": str(video_path),
                "tracking_dir": str(run_dir / "tracking"),
                "consolidation_dir": str(run_dir / "track_consolidation"),
                "output": str(out_path),
                "max_frames": max_frames,
            },
        )
        rec.log_path = str(log_dir / f"render-{rec.id}.log")
        self._dispatch(rec, self._run_render_consolidated)
        return rec

    # ------------------------------------------------------------------
    # Workers
    # ------------------------------------------------------------------

    def _run_pipeline(self, rec: JobRecord) -> dict[str, Any]:
        p = rec.payload
        config = get_default_config()
        if p.get("config_path"):
            config = merge_configs(config, load_config(p["config_path"]))
        if p.get("keypoint_model"):
            config.setdefault("field_registration", {}) \
                  .setdefault("pnlcalib", {})["keypoint_model_path"] = p["keypoint_model"]
        if p.get("no_viz"):
            config.setdefault("output", {})["save_visualizations"] = False

        stages = p.get("stages") or None
        pipeline = (
            Pipeline.from_stage_names(stages, config)
            if stages else Pipeline(config)
        )
        run_dir = self.workspace.run_dir(rec.run_name)  # type: ignore[arg-type]
        # Pipeline.run resolves stage_dirs as <output_dir>/<stage_name>/, so
        # we hand it the run dir directly (no extra timestamping).
        log_path = Path(rec.log_path) if rec.log_path else None
        with _LogCapture(log_path):
            metadata = pipeline.run(
                video_path=Path(p["video_path"]),
                output_dir=run_dir,
                skip_existing=bool(p.get("skip_existing", True)),
            )
        return {"stages_run": metadata.get("stages_run", [])}

    def _run_train(self, rec: JobRecord) -> dict[str, Any]:
        p = rec.payload
        argv = [
            sys.executable, "-u", str(REPO_ROOT / "scripts/train_finetune.py"),
            "--kind", p["kind"],
            "--annotations_dir", p["annotations_dir"],
            "--pretrained", p["pretrained"],
            "--output_dir", p["output_dir"],
        ]
        if p.get("remote"):
            argv.append("--remote")
        argv.extend(p.get("extra_argv") or [])
        rc = self._run_subprocess(argv, log_path=rec.log_path, cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"train_finetune.py exited with code {rc}")
        # Trainer writes best_model.pt under <output_dir>/run_<ts>/models/;
        # surface what we can find for the UI.
        out_dir = Path(p["output_dir"])
        candidates = list(out_dir.rglob("best_model.pt")) \
                   + list(out_dir.rglob("best_model_lines.pt"))
        return {
            "output_dir": str(out_dir),
            "weights": [str(c) for c in candidates],
        }

    def _run_render_consolidated(self, rec: JobRecord) -> dict[str, Any]:
        p = rec.payload
        argv = [
            sys.executable, "-u",
            str(REPO_ROOT / "scripts/render_consolidated_tracking.py"),
            "--video", p["video_path"],
            "--tracking-dir", p["tracking_dir"],
            "--consolidation-dir", p["consolidation_dir"],
            "--output", p["output"],
        ]
        if p.get("max_frames"):
            argv += ["--max-frames", str(int(p["max_frames"]))]
        rc = self._run_subprocess(argv, log_path=rec.log_path, cwd=REPO_ROOT)
        if rc != 0:
            raise RuntimeError(f"render_consolidated_tracking exited with code {rc}")
        return {"output": p["output"]}

    # ------------------------------------------------------------------
    # Plumbing
    # ------------------------------------------------------------------

    def _new_record(
        self, kind: str, run_name: str | None,
        *, payload: dict[str, Any],
    ) -> JobRecord:
        rec = JobRecord(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            run_name=run_name,
            status="pending",
            created_at=time.time(),
            payload=payload,
        )
        with self._lock:
            self.jobs[rec.id] = rec
        self._persist()
        return rec

    def _dispatch(self, rec: JobRecord, fn) -> None:
        def runner():
            rec.status = "running"
            rec.started_at = time.time()
            self._persist()
            try:
                rec.result = fn(rec)
                rec.status = "done"
            except BaseException as exc:  # noqa: BLE001 — log everything
                logger.exception("job %s failed", rec.id)
                rec.status = "failed"
                rec.error = repr(exc)
                # Append the trace to the log so the UI shows the cause.
                if rec.log_path:
                    with suppress(OSError):
                        with open(rec.log_path, "a") as fh:
                            import traceback
                            fh.write("\n--- exception ---\n")
                            traceback.print_exc(file=fh)
            finally:
                rec.finished_at = time.time()
                self._persist()
        self.executor.submit(runner)

    def _run_subprocess(
        self, argv: list[str], *, log_path: str | None, cwd: Path,
    ) -> int:
        log_fh = open(log_path, "a") if log_path else subprocess.DEVNULL
        try:
            if isinstance(log_fh, int):
                stream = log_fh
            else:
                log_fh.write(f"$ {shlex.join(argv)}\n")
                log_fh.flush()
                stream = log_fh
            proc = subprocess.Popen(
                argv,
                cwd=str(cwd),
                stdout=stream,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            return proc.wait()
        finally:
            if not isinstance(log_fh, int):
                log_fh.close()

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, job_id: str) -> JobRecord:
        rec = self.jobs.get(job_id)
        if rec is None:
            raise KeyError(job_id)
        return rec

    def list_jobs(self) -> list[dict[str, Any]]:
        return [j.to_public() for j in
                sorted(self.jobs.values(),
                       key=lambda r: r.created_at, reverse=True)]


class _LogCapture:
    """Tee Python logging + stdout/stderr to a file for in-process jobs."""

    def __init__(self, path: Path | None):
        self.path = path
        self._fh = None
        self._handler: logging.Handler | None = None
        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        if self.path is None:
            return self
        self._fh = open(self.path, "a")
        self._handler = logging.StreamHandler(self._fh)
        self._handler.setFormatter(logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s"))
        logging.getLogger().addHandler(self._handler)
        # Also tee stdout/stderr so `print()` calls inside stages are captured.
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = _Tee(self._old_stdout, self._fh)
        sys.stderr = _Tee(self._old_stderr, self._fh)
        return self

    def __exit__(self, *exc):
        if self._fh is None:
            return
        try:
            sys.stdout = self._old_stdout
            sys.stderr = self._old_stderr
            if self._handler is not None:
                logging.getLogger().removeHandler(self._handler)
        finally:
            with suppress(OSError):
                self._fh.close()


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:  # noqa: BLE001
                pass

    def flush(self):
        for st in self.streams:
            with suppress(Exception):
                st.flush()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def register_jobs_routes(app: FastAPI, manager: JobManager) -> None:
    @app.get("/api/jobs")
    def list_jobs() -> JSONResponse:
        return JSONResponse(manager.list_jobs())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> JSONResponse:
        try:
            rec = manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"job not found: {job_id}")
        return JSONResponse(rec.to_public())

    @app.get("/api/jobs/{job_id}/log")
    def get_log(job_id: str, tail_kb: int = 64) -> PlainTextResponse:
        try:
            rec = manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"job not found: {job_id}")
        if not rec.log_path or not Path(rec.log_path).exists():
            return PlainTextResponse("")
        with open(rec.log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            n = min(size, tail_kb * 1024)
            fh.seek(size - n)
            data = fh.read()
        return PlainTextResponse(data.decode("utf-8", "replace"))

    @app.get("/api/jobs/{job_id}/stream")
    def stream_log(job_id: str) -> StreamingResponse:
        try:
            rec = manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"job not found: {job_id}")

        async def gen():
            log_path = Path(rec.log_path) if rec.log_path else None
            pos = 0
            while True:
                if log_path and log_path.exists():
                    with open(log_path, "rb") as fh:
                        fh.seek(pos)
                        chunk = fh.read()
                        pos = fh.tell()
                    if chunk:
                        text = chunk.decode("utf-8", "replace")
                        yield f"data: {json.dumps({'log': text})}\n\n"
                if rec.status in ("done", "failed"):
                    yield f"data: {json.dumps({'status': rec.status, 'error': rec.error, 'result': rec.result})}\n\n"
                    return
                await asyncio.sleep(0.5)

        return StreamingResponse(
            gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/jobs/pipeline")
    async def submit_pipeline(request: Request) -> JSONResponse:
        p = await request.json()
        run_name = p.get("run_name")
        stages = p.get("stages") or []
        config_path = p.get("config_path")
        video_path = p.get("video_path")
        if not (run_name and stages and config_path and video_path):
            raise HTTPException(
                400,
                "run_name, stages, config_path, video_path are required",
            )
        rec = manager.submit_pipeline(
            run_name=run_name,
            stages=list(stages),
            config_path=Path(config_path),
            video_path=Path(video_path),
            skip_existing=bool(p.get("skip_existing", True)),
            keypoint_model=Path(p["keypoint_model"]) if p.get("keypoint_model") else None,
            no_viz=bool(p.get("no_viz", False)),
        )
        return JSONResponse(rec.to_public(), status_code=202)

    @app.post("/api/jobs/train")
    async def submit_train(request: Request) -> JSONResponse:
        p = await request.json()
        kind = p.get("kind")
        annotations_dir = p.get("annotations_dir")
        pretrained = p.get("pretrained")
        if not (kind and annotations_dir and pretrained):
            raise HTTPException(
                400,
                "kind, annotations_dir, pretrained are required",
            )
        rec = manager.submit_train(
            kind=kind,
            annotations_dir=Path(annotations_dir),
            pretrained=Path(pretrained),
            extra_argv=p.get("extra_argv") or [],
            remote=bool(p.get("remote", False)),
        )
        return JSONResponse(rec.to_public(), status_code=202)

    @app.post("/api/jobs/render_consolidated")
    async def submit_render(request: Request) -> JSONResponse:
        p = await request.json()
        run_name = p.get("run_name")
        video_path = p.get("video_path")
        if not (run_name and video_path):
            raise HTTPException(400, "run_name and video_path are required")
        try:
            rec = manager.submit_render_consolidated(
                run_name=run_name,
                video_path=Path(video_path),
                max_frames=p.get("max_frames"),
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        return JSONResponse(rec.to_public(), status_code=202)
