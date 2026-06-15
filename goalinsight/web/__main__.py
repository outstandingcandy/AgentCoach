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
    parser.add_argument(
        "--pitch-config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config providing a top-level `pitch:` block "
            "(pitch_length, pitch_width, ...). Used to override the "
            "annotator's default FIFA pitch — required when annotations "
            "were saved against a non-FIFA pitch (e.g. configs/"
            "kids_soccer_physical.yaml)."
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    pitch = None
    if args.pitch_config:
        import yaml
        from ..annotation.pitch.geometry import SoccerPitch
        cfg = yaml.safe_load(args.pitch_config.read_text()) or {}
        pitch_kw = cfg.get("pitch") or {}
        if pitch_kw:
            pitch = SoccerPitch(**pitch_kw)
            print(f"Pitch override: {pitch.PITCH_LENGTH}m × {pitch.PITCH_WIDTH}m")

    app = create_workspace_app(args.workspace, pitch=pitch)
    # If a config was supplied, also bind it as the active solver
    # config so the annotator's compute_homography uses the matching
    # backend / camera_profile out of the box (without forcing the user
    # to click the dropdown after every restart).
    if args.pitch_config and getattr(app.state, "annotator", None) is not None:
        try:
            app.state.annotator.set_solver_config(str(args.pitch_config))
            print(f"Active solver config: {args.pitch_config}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: could not bind solver config: {exc}")
    print(f"Workspace: {args.workspace.resolve()}")
    print(f"Listening on http://{args.host}:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
