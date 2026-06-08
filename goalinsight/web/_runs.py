"""Lazy per-run state for the unified viewer.

Each pipeline run under ``workspace/runs/<run_name>/`` owns a MatchContext +
ChatEngine. Building these is expensive (loads JSON, opens a Bedrock
client, eagerly parses tracks), so we build on first request and keep at
most ``RUN_CACHE_SIZE`` warm in memory. The least-recently-used run is
closed when capacity is exceeded.
"""

from __future__ import annotations

import logging
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from ..highlights._context import MatchContext
from ._workspace import VIDEO_EXTS, Workspace
from .chat import ChatEngine

logger = logging.getLogger(__name__)

RUN_CACHE_SIZE = 4
CHAT_ARTIFACTS_URL = "/chat_artifacts"


@dataclass
class RunHandle:
    run_name: str
    run_dir: Path
    video_path: Path
    ctx: MatchContext
    # Built lazily on first chat call so analytics-only requests (heatmap,
    # stats) don't pay the boto3 / Bedrock startup cost.
    _engine: ChatEngine | None = None
    _engine_factory: Any = None

    @property
    def engine(self) -> ChatEngine:
        if self._engine is None:
            if self._engine_factory is None:
                raise RuntimeError("engine factory missing for run handle")
            self._engine = self._engine_factory()
        return self._engine

    def close(self) -> None:
        if self._engine is None:
            return
        try:
            self._engine.close()
        except Exception:  # noqa: BLE001
            logger.exception("engine close failed for %s", self.run_name)


def resolve_run_video(run_dir: Path) -> Path:
    """Pick the video file the viewer should serve for *run_dir*.

    Mirrors the original ``create_app`` logic: prefer the web-optimized
    re-encode under ``annotated_video/`` if present, then the full annotated
    video, then any video referenced in ``calibration_metadata.json``.
    """
    video_dir = run_dir / "annotated_video"
    candidates = [video_dir / "annotated_web.mp4", video_dir / "annotated.mp4"]
    for c in candidates:
        if c.exists():
            return c
    # Fall back to the source video recorded in calibration metadata so
    # users can still open a viewer for runs that haven't produced
    # annotated_video yet (e.g. only field_registration ran).
    meta = run_dir / "field_registration" / "calibration_metadata.json"
    if meta.exists():
        import json
        try:
            vi = json.loads(meta.read_text()).get("video_info", {})
            src = vi.get("path")
            if src and Path(src).exists():
                return Path(src)
        except Exception:  # noqa: BLE001
            pass
    raise FileNotFoundError(
        f"no playable video found for run {run_dir.name}: "
        f"{video_dir / 'annotated.mp4'} missing"
    )


class RunRegistry:
    """LRU cache of warm RunHandle objects keyed by run_name."""

    def __init__(self, workspace: Workspace, *, capacity: int = RUN_CACHE_SIZE):
        self.workspace = workspace
        self.capacity = capacity
        self._cache: OrderedDict[str, RunHandle] = OrderedDict()
        # Per-run chat artifact dirs land under workspace/chat_artifacts/<run>;
        # the unified app mounts workspace/chat_artifacts at /chat_artifacts so
        # the existing IMG_RE in viewer.html still matches without changes.
        self.artifacts_root = workspace.root / "chat_artifacts"
        self.artifacts_root.mkdir(parents=True, exist_ok=True)

    def artifact_dir_for(self, run_name: str) -> Path:
        return self.artifacts_root / run_name

    def url_prefix_for(self, run_name: str) -> str:
        return f"{CHAT_ARTIFACTS_URL}/{run_name}"

    def get(self, run_name: str) -> RunHandle:
        if run_name in self._cache:
            self._cache.move_to_end(run_name)
            return self._cache[run_name]

        run_dir = self.workspace.run_dir(run_name).resolve()
        if not run_dir.is_dir():
            raise FileNotFoundError(f"run not found: {run_dir}")

        video_path = resolve_run_video(run_dir)
        ctx = MatchContext.from_output_dir(run_dir, video_path=video_path)

        artifact_dir = self.artifact_dir_for(run_name)
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.url_prefix_for(run_name)

        def _build_engine() -> ChatEngine:
            return ChatEngine(
                ctx=ctx,
                artifact_dir=artifact_dir,
                artifact_url_prefix=prefix,
            )

        handle = RunHandle(
            run_name=run_name,
            run_dir=run_dir,
            video_path=video_path,
            ctx=ctx,
            _engine_factory=_build_engine,
        )
        self._cache[run_name] = handle
        self._evict()
        return handle

    def _evict(self) -> None:
        while len(self._cache) > self.capacity:
            old_name, old_handle = self._cache.popitem(last=False)
            logger.info("evicting run %s from registry", old_name)
            old_handle.close()

    def close_all(self) -> None:
        for handle in self._cache.values():
            handle.close()
        self._cache.clear()


def list_runs(workspace: Workspace) -> list[dict]:
    """Lightweight enumeration for the library page (no MatchContext load)."""
    out: list[dict] = []
    if not workspace.runs_dir.is_dir():
        return out
    for run_dir in sorted(workspace.runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        out.append({
            "run_name": run_dir.name,
            "stages": _detect_stage_completion(run_dir),
        })
    return out


def _detect_stage_completion(run_dir: Path) -> dict[str, bool]:
    return {
        "field_registration": (run_dir / "field_registration" / "homographies.pkl").exists(),
        "tracking": (run_dir / "tracking" / "tracks.json").exists(),
        "track_consolidation": (run_dir / "track_consolidation" / "players.json").exists(),
        "event_detection": (run_dir / "event_detection" / "events.json").exists(),
        "highlights": any((run_dir / "highlights").glob("*.mp4"))
                      if (run_dir / "highlights").is_dir() else False,
        "annotated_video": (run_dir / "annotated_video" / "annotated.mp4").exists(),
        "consolidated_overlay": (run_dir / "track_consolidation" / "players_overlay.mp4").exists(),
    }
