"""Launch the GoalInsight web viewer (annotated video + LLM chat).

Usage:
    python scripts/run_web_viewer.py                 # uses default run-dir
    python scripts/run_web_viewer.py --run-dir <dir> # override
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

from goalinsight.web.app import create_app

DEFAULT_RUN_DIR = Path("output/full_pipeline/full_v2")
DEFAULT_VIDEO_PATH = Path("webapp_videos/annotated.mp4")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=DEFAULT_RUN_DIR, type=Path,
                        help=f"Path to a pipeline run directory "
                             f"(default: {DEFAULT_RUN_DIR}).")
    parser.add_argument("--video", default=DEFAULT_VIDEO_PATH, type=Path,
                        help=f"Path to the annotated video file served by "
                             f"the webapp. Defaults to {DEFAULT_VIDEO_PATH}; "
                             f"falls back to <run-dir>/annotated_video/"
                             f"annotated.mp4 if the default file is missing.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    run_dir = args.run_dir.resolve()

    # Resolve which video file to serve. Prefer the explicit --video / its
    # default (a stable path under webapp_videos/); fall back to the one
    # inside the run-dir so older runs keep working.
    video_path: Path | None = args.video.resolve() if args.video else None
    if video_path is not None and not video_path.exists():
        run_video = run_dir / "annotated_video" / "annotated.mp4"
        if run_video.exists():
            print(f"WARN: --video {video_path} not found; falling back to "
                  f"{run_video}", file=sys.stderr)
            video_path = run_video
        else:
            print(f"ERROR: video not found: {video_path}", file=sys.stderr)
            print(f"       fallback also missing: {run_video}", file=sys.stderr)
            return 1

    expected = [
        video_path,
        run_dir / "tracking" / "tracks.json",
        run_dir / "track_consolidation" / "players.json",
    ]
    missing = [p for p in expected if p is None or not p.exists()]
    if missing:
        print("ERROR: required pipeline outputs missing:", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        print("\nRun the pipeline through annotated_video first.", file=sys.stderr)
        return 1

    app = create_app(run_dir, video_path=video_path)
    print(f"Serving {run_dir}")
    print(f"  video: {video_path}")
    print(f"  on:    http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
