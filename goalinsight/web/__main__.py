"""Entry point for ``goalinsight-web`` — launches the unified workspace app."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn

from .app import create_workspace_app


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the unified GoalInsight web product (library + "
                    "annotator + pipeline + viewer) against a workspace dir.",
    )
    parser.add_argument(
        "--workspace",
        default="./workspace",
        type=Path,
        help="Workspace directory; created if missing.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    app = create_workspace_app(args.workspace)
    print(f"Workspace: {args.workspace.resolve()}")
    print(f"Listening on http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
