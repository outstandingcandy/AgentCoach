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

from ..annotation import per_video_settings
from ..pipeline import Pipeline, PipelineCancelled
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
        # Per-job cancel hooks. Pipeline jobs run in-process so we use
        # threading.Event; subprocess jobs (train / render) get a
        # subprocess.Popen handle instead. Both are kept here so the
        # /cancel endpoint has one lookup table to consult.
        self._cancel_events: dict[str, threading.Event] = {}
        self._procs: dict[str, subprocess.Popen] = {}
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
        kind: str,                                       # "keypoint" | "line"
        annotations_dir: Path | list[Path],              # one dir, or many to combine
        pretrained: Path,
        extra_argv: list[str] | None = None,
        remote: bool = False,
    ) -> JobRecord:
        if kind not in {"keypoint", "line"}:
            raise ValueError(f"kind must be 'keypoint' or 'line', got {kind!r}")
        # Allow a list of dirs for "train this prefix-group" — passed
        # through to the trainer as a single comma-separated argument
        # (its argparse accepts that form).
        if isinstance(annotations_dir, (list, tuple)):
            ann_payload = ",".join(str(d) for d in annotations_dir)
        else:
            ann_payload = str(annotations_dir)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.workspace.models_dir / f"{kind}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = self._new_record(
            "train", None,
            payload={
                "kind": kind,
                "annotations_dir": ann_payload,
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
        """Run the pipeline as a subprocess so cancel can SIGTERM it.

        The previous implementation ran ``Pipeline.run`` in a worker
        thread of the web server's process. ThreadPoolExecutor has no
        thread-kill primitive, so cancel could only fire between
        stages — useless when the bottleneck is mid-stage (e.g.
        ``field_registration`` Pass 1 takes 10+ minutes). Spawning a
        subprocess gives us a real OS process to terminate within
        seconds, at the cost of having to materialize the merged
        config into a YAML file first.
        """
        p = rec.payload
        run_dir = self.workspace.run_dir(rec.run_name)  # type: ignore[arg-type]
        run_dir.mkdir(parents=True, exist_ok=True)

        # Apply per-video annotator overrides to the chosen YAML so the
        # subprocess inherits them. Same merge order the old in-process
        # path used; we just write the result to disk instead of
        # passing a Python dict.
        config = get_default_config()
        if p.get("config_path"):
            config = merge_configs(config, load_config(p["config_path"]))
        config = _merge_per_video_overrides(
            config,
            annotations_dir=self.workspace.annotations_dir,
            video_path=Path(p["video_path"]),
        )
        if p.get("keypoint_model"):
            config.setdefault("field_registration", {}) \
                  .setdefault("pnlcalib", {})["keypoint_model_path"] = p["keypoint_model"]
        if p.get("no_viz"):
            config.setdefault("output", {})["save_visualizations"] = False

        # If a stages list is supplied, encode it on disk too so a user
        # inspecting effective.yaml later sees the exact run shape.
        stages = p.get("stages") or None
        if stages:
            config.setdefault("pipeline", {})["stages"] = list(stages)

        try:
            import yaml  # local import: avoids the import on cold starts
        except ImportError as exc:  # pragma: no cover — yaml ships with the project
            raise RuntimeError(f"yaml module unavailable: {exc}") from exc
        effective_path = run_dir / "logs" / f"effective-{rec.id}.yaml"
        effective_path.parent.mkdir(parents=True, exist_ok=True)
        effective_path.write_text(yaml.safe_dump(config, sort_keys=False))

        argv: list[str] = [
            sys.executable, "-u", "-m", "goalinsight.cli",
            "--video", p["video_path"],
            "--output", str(run_dir),
            "--config", str(effective_path),
            "--no-timestamp",
        ]
        if stages:
            argv += ["--stages", ",".join(stages)]
        if p.get("skip_existing", True):
            argv.append("--skip-existing")
        if p.get("no_viz"):
            argv.append("--no-viz")
        if p.get("keypoint_model"):
            argv += ["--keypoint-model", p["keypoint_model"]]

        rc = self._run_subprocess(
            argv, log_path=rec.log_path, cwd=REPO_ROOT, job_id=rec.id,
        )
        # Distinguish "cancelled by user" from "real failure". SIGTERM
        # produces -signal exit codes on Linux (negative when accessed
        # via subprocess.returncode); also accept the standard
        # 128 + SIGTERM (143) form some runtimes report.
        if rc in (-15, 143):
            raise PipelineCancelled(
                f"pipeline cancelled by user (signal SIGTERM, rc={rc})"
            )
        if rc != 0:
            raise RuntimeError(f"pipeline subprocess exited with code {rc}")

        # Mirror the old return shape so callers + jobs.json keep
        # the same schema.
        stats_path = run_dir / "pipeline_stats.json"
        if stats_path.exists():
            try:
                meta = json.loads(stats_path.read_text())
                return {"stages_run": meta.get("stages_run", [])}
            except (OSError, json.JSONDecodeError):
                pass
        return {"stages_run": stages or []}

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
        rc = self._run_subprocess(argv, log_path=rec.log_path, cwd=REPO_ROOT, job_id=rec.id)
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
        rc = self._run_subprocess(argv, log_path=rec.log_path, cwd=REPO_ROOT, job_id=rec.id)
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
            except PipelineCancelled as exc:
                # User-initiated stop — record but don't log a stack
                # trace (it's not a fault). Status distinguishes
                # cancel-by-user from a real failure.
                rec.status = "cancelled"
                rec.error = str(exc)
                if rec.log_path:
                    with suppress(OSError):
                        with open(rec.log_path, "a") as fh:
                            fh.write(f"\n--- cancelled ---\n{exc}\n")
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

    # ------------------------------------------------------------------
    # Cancel
    # ------------------------------------------------------------------

    def cancel(self, job_id: str) -> bool:
        """Request a graceful stop for ``job_id``.

        Pipeline jobs honour this at the next stage boundary (the
        currently-running stage finishes; later stages are skipped).
        Subprocess jobs (train / render) get a SIGTERM.

        Returns False if the job is unknown, not running, or has no
        cancel hook attached. Returns True when the request was
        delivered — actual termination is asynchronous.
        """
        with self._lock:
            rec = self.jobs.get(job_id)
            if rec is None or rec.status != "running":
                return False
            ev = self._cancel_events.get(job_id)
            proc = self._procs.get(job_id)
        if ev is not None:
            ev.set()
            return True
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                logger.exception("terminate failed for job %s", job_id)
                return False
            return True
        return False

    def _run_subprocess(
        self, argv: list[str], *, log_path: str | None, cwd: Path,
        job_id: str | None = None,
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
            # Register so /cancel can SIGTERM in-flight subprocess jobs.
            if job_id is not None:
                with self._lock:
                    self._procs[job_id] = proc
            try:
                return proc.wait()
            finally:
                if job_id is not None:
                    with self._lock:
                        self._procs.pop(job_id, None)
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


def _merge_per_video_overrides(
    config: dict[str, Any],
    *,
    annotations_dir: Path,
    video_path: Path,
) -> dict[str, Any]:
    """Deep-merge ``annotations_dir/<stem>/overrides.yaml`` into *config*.

    The overrides file is a sparse hand-edited yaml — any subset of the
    full config tree is allowed. We trust the user's keys and let
    ``merge_configs`` apply the diff. Original config is left untouched.
    """
    stem = video_path.stem
    overrides = per_video_settings.load(annotations_dir, stem)
    if not overrides:
        return config
    return merge_configs(config, overrides)


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

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> JSONResponse:
        try:
            rec = manager.get(job_id)
        except KeyError:
            raise HTTPException(404, f"job not found: {job_id}")
        if rec.status != "running":
            raise HTTPException(
                409,
                f"job {job_id} is {rec.status!r}, only running jobs can be cancelled",
            )
        ok = manager.cancel(job_id)
        if not ok:
            raise HTTPException(
                500,
                f"cancel hook missing for job {job_id} — running but not interruptible",
            )
        return JSONResponse({"id": job_id, "cancel_requested": True})

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
