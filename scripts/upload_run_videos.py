"""Sync every mp4 produced by one (or all) pipeline runs to S3.

Reads the bucket from ``GOALINSIGHT_VIDEO_S3_BUCKET`` (or ``--bucket``).
Each mp4 is remuxed with ``+faststart`` before upload so the browser
can begin playback before the full file lands.

Usage:
    python scripts/upload_run_videos.py                       # all runs
    python scripts/upload_run_videos.py sunday_20sec          # one run
    python scripts/upload_run_videos.py --workspace ./workspace --bucket my-bucket
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from goalinsight.web._s3_video import upload_run_videos, video_bucket


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", help="Run names (default: all)")
    parser.add_argument(
        "--workspace", default="./workspace",
        help="Workspace directory containing runs/ (default: ./workspace)",
    )
    parser.add_argument(
        "--bucket", default=None,
        help=("S3 bucket. Falls back to GOALINSIGHT_VIDEO_S3_BUCKET. "
              "Required either way."),
    )
    args = parser.parse_args()

    bucket = args.bucket or video_bucket()
    if not bucket:
        print(
            "ERROR: pass --bucket or set GOALINSIGHT_VIDEO_S3_BUCKET",
            file=sys.stderr,
        )
        return 2

    runs_root = Path(args.workspace) / "runs"
    if not runs_root.is_dir():
        print(f"ERROR: {runs_root} is not a directory", file=sys.stderr)
        return 2

    if args.runs:
        run_dirs = [runs_root / name for name in args.runs]
    else:
        run_dirs = [p for p in sorted(runs_root.iterdir()) if p.is_dir()]

    total_uploaded = 0
    total_skipped = 0
    total_failed = 0
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            print(f"  [skip] {run_dir} — not a directory")
            continue
        print(f"== {run_dir.name} ==", flush=True)
        for ev in upload_run_videos(run_dir, run_name=run_dir.name, bucket=bucket):
            tag = ev["status"]
            label = ev["video"].rsplit("/", 1)[-1]
            print(f"  [{tag:8s}] {label} -> s3://{bucket}/{ev['key']}", flush=True)
            total_uploaded += int(tag == "uploaded")
            total_skipped += int(tag == "skipped")
            total_failed += int(tag == "failed")

    print(
        f"\nDone. uploaded={total_uploaded} skipped={total_skipped} "
        f"failed={total_failed}",
    )
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
