"""Goal detection from ball tracking and camera calibration data.

DEPRECATED: This module is preserved for backward compatibility.
New code should use ``goalinsight.events`` instead.
"""

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def detect_goals(
    ball_tracks: dict,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float = 30.0,
    min_confidence: float = 0.15,
    camera_poses: dict | None = None,
) -> list[dict]:
    """Detect goal events from ball tracking data.

    Delegates to ``goalinsight.events``.
    """
    warnings.warn(
        "goalinsight.goal_detection.detect_goals is deprecated. "
        "Use goalinsight.events.detect_events_from_data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from goalinsight.events import EventType, detect_events_from_data

    config: dict[str, Any] = {
        "events": {
            "detectors": ["possession", "shot"],
            "shot": {"min_confidence": min_confidence},
        }
    }
    events = detect_events_from_data(
        ball_tracks=ball_tracks,
        player_tracks={},
        team_assignments={},
        fps=fps,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        camera_poses=camera_poses,
        config=config,
    )

    return [
        _event_to_legacy_dict(e, fps)
        for e in events
        if e.event_type == EventType.GOAL
    ]


def detect_goals_from_output(
    output_dir: str | Path,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
    fps: float | None = None,
    min_confidence: float = 0.15,
) -> list[dict]:
    """Detect goals from a pipeline output directory.

    Delegates to ``goalinsight.events``.
    """
    warnings.warn(
        "goalinsight.goal_detection.detect_goals_from_output is deprecated. "
        "Use goalinsight.events.detect_events_from_output() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from goalinsight.events import EventType, detect_events_from_output

    config: dict[str, Any] = {
        "events": {
            "detectors": ["possession", "shot"],
            "shot": {"min_confidence": min_confidence},
        }
    }
    events = detect_events_from_output(
        pipeline_output_dir=output_dir,
        config=config,
        pitch_length=pitch_length,
        pitch_width=pitch_width,
        fps=fps,
    )

    actual_fps = fps or 30.0
    return [
        _event_to_legacy_dict(e, actual_fps)
        for e in events
        if e.event_type == EventType.GOAL
    ]


def _event_to_legacy_dict(event: Any, fps: float) -> dict:
    """Convert a MatchEvent to the legacy goal detection dict format."""
    meta = event.metadata or {}
    return {
        "frame": event.frame,
        "timestamp": event.frame / fps,
        "goal_side": meta.get("goal_side", "right"),
        "ball_position": meta.get("ball_position_3d", meta.get("ball_position", [0, 0, 0])),
        "ball_pixel": meta.get("ball_pixel"),
        "confidence": event.confidence,
        "ball_speed_mps": meta.get("ball_speed_mps", 0.0),
        "crossbar_validation": meta.get("crossbar_validation"),
    }


def main():
    parser = argparse.ArgumentParser(description="Detect goals from pipeline output")
    parser.add_argument("output_dir", help="Pipeline output directory")
    parser.add_argument("--pitch-length", type=float, default=105.0)
    parser.add_argument("--pitch-width", type=float, default=68.0)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.15)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    goals = detect_goals_from_output(
        output_dir=args.output_dir,
        pitch_length=args.pitch_length,
        pitch_width=args.pitch_width,
        fps=args.fps,
        min_confidence=args.min_confidence,
    )

    if not goals:
        print("No goals detected.")
        return

    print(f"\nDetected {len(goals)} goal(s):\n")
    for i, g in enumerate(goals, 1):
        ts = g["timestamp"]
        minutes = int(ts // 60)
        seconds = ts % 60
        print(f"  Goal {i}:")
        print(f"    Time:     {minutes}:{seconds:05.2f} (frame {g['frame']})")
        print(f"    Side:     {g['goal_side']} goal")
        pos = g["ball_position"]
        print(f"    Position: x={pos[0]:.1f}m, y={pos[1]:.1f}m, h={pos[2]:.2f}m")
        print(f"    Speed:    {g['ball_speed_mps']:.1f} m/s")
        print(f"    Confidence: {g['confidence']:.2f}")
        if g.get("crossbar_validation"):
            cv = g["crossbar_validation"]
            print(f"    Crossbar check: {'PASS' if cv['inside_goal_frame'] else 'FAIL'}")
        print()


if __name__ == "__main__":
    main()
