"""Ball physics utilities: velocity, speed, trajectory smoothing."""

from __future__ import annotations

import logging
import math

import numpy as np

from ._types import BallState

logger = logging.getLogger(__name__)


def compute_ball_states(
    ball_tracks: dict[str, dict],
    fps: float,
    min_confidence: float = 0.1,
    smoothing_window: int = 3,
    max_gap_seconds: float = 1.0,
) -> list[BallState]:
    """Parse ball_tracks, smooth positions, compute velocity per frame.

    Returns a list of BallState sorted by frame.
    """
    # Parse into sorted observations
    obs = _parse_observations(ball_tracks, min_confidence)
    if len(obs) < 2:
        return [_obs_to_state(o, None, 0.0) for o in obs]

    # Median-smooth x, y
    _smooth_positions(obs, smoothing_window)

    # Compute velocity between consecutive observations
    states: list[BallState] = []
    for i, o in enumerate(obs):
        if i == 0:
            states.append(_obs_to_state(o, None, 0.0))
            continue

        prev = obs[i - 1]
        frame_gap = o["frame"] - prev["frame"]
        dt = frame_gap / fps
        if dt <= 0 or dt > max_gap_seconds:
            states.append(_obs_to_state(o, None, 0.0))
            continue

        vx = (o["x"] - prev["x"]) / dt
        vy = (o["y"] - prev["y"]) / dt
        speed = math.hypot(vx, vy)

        # Cap physically impossible speeds
        if speed > 60.0:
            states.append(_obs_to_state(o, None, 0.0))
            continue

        states.append(_obs_to_state(o, [vx, vy], speed))

    return states


def _parse_observations(
    ball_tracks: dict[str, dict], min_confidence: float
) -> list[dict]:
    """Parse ball_tracks.json into sorted list of observations."""
    obs = []
    for frame_str, data in ball_tracks.items():
        pp = data.get("pitch_position")
        if pp is None:
            continue
        conf = data.get("confidence", 0)
        if conf < min_confidence:
            continue
        obs.append(
            {
                "frame": int(frame_str),
                "x": pp[0],
                "y": pp[1],
                "height": data.get("height", 0.0),
                "confidence": conf,
                "center": data.get("center"),
                "position_3d": data.get("position_3d"),
                "predicted": data.get("predicted", False),
            }
        )
    obs.sort(key=lambda o: o["frame"])
    return obs


def _smooth_positions(observations: list[dict], window: int = 3) -> None:
    """Apply median filter to x, y positions in-place."""
    n = len(observations)
    if n < window:
        return

    xs = np.array([o["x"] for o in observations])
    ys = np.array([o["y"] for o in observations])

    half_w = window // 2
    xs_smooth = np.copy(xs)
    ys_smooth = np.copy(ys)
    for i in range(half_w, n - half_w):
        xs_smooth[i] = np.median(xs[i - half_w : i + half_w + 1])
        ys_smooth[i] = np.median(ys[i - half_w : i + half_w + 1])

    for i, o in enumerate(observations):
        o["x"] = float(xs_smooth[i])
        o["y"] = float(ys_smooth[i])


def _obs_to_state(
    obs: dict,
    velocity: list[float] | None,
    speed: float,
) -> BallState:
    pos_3d = obs.get("position_3d")
    return BallState(
        frame=obs["frame"],
        position=[obs["x"], obs["y"]],
        position_3d=pos_3d,
        height=obs.get("height", 0.0),
        velocity=velocity,
        speed=speed,
        confidence=obs.get("confidence", 0.0),
        pixel_center=obs.get("center"),
    )
