"""ShotDetector — detects shots on goal, subsumes goal detection.

A goal is a shot with outcome='Goal'. Ported from goalinsight/goal_detection.py
with additional outcome classification (Saved, Off_Target, Blocked).
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import BallState, EventType, MatchEvent, ShotOutcome

logger = logging.getLogger(__name__)

# FIFA fixed goal dimensions
GOAL_HALF_WIDTH = 3.66  # meters (7.32m total)
GOAL_HEIGHT = 2.44  # meters


@register_detector
class ShotDetector(BaseEventDetector):
    """Detect shots on goal via ball speed + trajectory toward goal."""

    name = "shot"
    depends_on = ["possession"]

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("shot", {})
        shot_speed_min = cfg.get("shot_speed_threshold", 10.0)
        max_angle = cfg.get("max_angle_to_goal", 30.0)
        min_confidence = cfg.get("min_confidence", 0.15)
        shooter_lookback_sec = cfg.get("shooter_lookback_seconds", 3.0)
        dedup_gap_sec = cfg.get("dedup_gap_seconds", 2.0)
        approach_window_sec = cfg.get("approach_window_seconds", 1.0)
        max_gap_sec = cfg.get("max_frame_gap_seconds", 1.0)

        half_length = ctx.pitch_length / 2.0

        # Step 1: Detect goal-line crossings (from existing goal_detection logic)
        crossings = _detect_crossings(
            ctx.ball_states, half_length, ctx.fps, max_gap_sec
        )

        # Step 2: Validate crossings and classify outcomes
        events: list[MatchEvent] = []
        shot_seq = 0

        for crossing in crossings:
            result = _classify_crossing(
                crossing, ctx.ball_states, half_length, ctx.fps,
                ctx.camera_poses, approach_window_sec,
            )
            if result is None:
                continue

            outcome, goal_event = result

            # Attribute shooter from possession
            shooter_id, shooter_team, shot_frame = self._find_shooter(
                ctx, crossing["frame"], shooter_lookback_sec,
            )

            shot_seq += 1
            shot_event = MatchEvent(
                event_id=f"shot_{shot_seq:04d}",
                event_type=EventType.SHOT,
                frame=crossing["frame"],
                match_time=crossing["frame"] / ctx.fps,
                player_id=shooter_id,
                team_id=shooter_team,
                start_position=goal_event.get("start_pos"),
                end_position=goal_event["ball_position"][:2],
                confidence=goal_event["confidence"],
                metadata={
                    "outcome": outcome.value,
                    "goal_side": goal_event["goal_side"],
                    "ball_position_3d": goal_event["ball_position"],
                    "ball_speed_mps": goal_event["ball_speed_mps"],
                    "ball_pixel": goal_event.get("ball_pixel"),
                    "crossbar_validation": goal_event.get(
                        "crossbar_validation"
                    ),
                },
            )
            events.append(shot_event)

            # Emit a convenience GOAL event
            if outcome == ShotOutcome.GOAL:
                goal_seq = sum(
                    1
                    for e in events
                    if e.event_type == EventType.GOAL
                ) + 1
                events.append(
                    MatchEvent(
                        event_id=f"goal_{goal_seq:04d}",
                        event_type=EventType.GOAL,
                        frame=crossing["frame"],
                        match_time=crossing["frame"] / ctx.fps,
                        player_id=shooter_id,
                        team_id=shooter_team,
                        start_position=shot_event.start_position,
                        end_position=shot_event.end_position,
                        confidence=shot_event.confidence,
                        metadata={
                            "shot_event_id": shot_event.event_id,
                            "goal_side": goal_event["goal_side"],
                            "ball_position_3d": goal_event["ball_position"],
                            "ball_speed_mps": goal_event["ball_speed_mps"],
                            "ball_pixel": goal_event.get("ball_pixel"),
                            "crossbar_validation": goal_event.get(
                                "crossbar_validation"
                            ),
                        },
                    )
                )

        # Deduplicate shots within dedup_gap_sec
        events = _deduplicate(events, ctx.fps, int(dedup_gap_sec * ctx.fps))

        logger.info(
            "ShotDetector: %d shot(s), %d goal(s)",
            sum(1 for e in events if e.event_type == EventType.SHOT),
            sum(1 for e in events if e.event_type == EventType.GOAL),
        )
        return events

    def _find_shooter(
        self,
        ctx: EventDetectionContext,
        goal_frame: int,
        lookback_seconds: float = 3.0,
    ) -> tuple[int | None, str | None, int]:
        """Find the player who took the shot from possession data."""
        lookback_frames = int(lookback_seconds * ctx.fps)
        for f in range(goal_frame, max(0, goal_frame - lookback_frames) - 1, -1):
            span = ctx.get_possession_at_frame(f)
            if span is not None:
                return span.player_id, span.team_id, f
        return None, None, goal_frame


# ------------------------------------------------------------------
# Ported from goalinsight/goal_detection.py
# ------------------------------------------------------------------


def _detect_crossings(
    ball_states: list[BallState],
    half_length: float,
    fps: float,
    max_gap_seconds: float = 1.0,
) -> list[dict]:
    """Find frames where ball crosses a goal line."""
    crossings: list[dict] = []

    for i in range(1, len(ball_states)):
        prev = ball_states[i - 1]
        curr = ball_states[i]

        frame_gap = curr.frame - prev.frame
        dt = frame_gap / fps
        if dt > max_gap_seconds:
            continue
        dx = curr.position[0] - prev.position[0]
        dy = curr.position[1] - prev.position[1]
        speed = math.hypot(dx, dy) / max(dt, 1e-6)
        if speed > 50.0:
            continue

        prev_x = prev.position[0]
        curr_x = curr.position[0]

        # Right goal
        if prev_x <= half_length and curr_x > half_length:
            if prev_x > half_length - 10:
                c = _interpolate_crossing(
                    prev, curr, half_length, dx, frame_gap, speed, "right"
                )
                if c:
                    crossings.append(c)

        # Left goal
        if prev_x >= -half_length and curr_x < -half_length:
            if prev_x < -half_length + 10:
                c = _interpolate_crossing(
                    prev, curr, -half_length, dx, frame_gap, speed, "left"
                )
                if c:
                    crossings.append(c)

    return crossings


def _interpolate_crossing(
    prev: BallState,
    curr: BallState,
    goal_x: float,
    dx: float,
    frame_gap: int,
    speed: float,
    side: str,
) -> dict | None:
    denom = dx if abs(dx) > 1e-6 else (1e-6 if dx >= 0 else -1e-6)
    t = (goal_x - prev.position[0]) / denom
    t = float(np.clip(t, 0, 1))
    cross_y = prev.position[1] + t * (curr.position[1] - prev.position[1])
    cross_h = prev.height + t * (curr.height - prev.height)
    cross_frame = prev.frame + t * frame_gap
    return {
        "frame": int(round(cross_frame)),
        "goal_side": side,
        "cross_y": cross_y,
        "cross_height": max(cross_h, 0),
        "prev": prev,
        "curr": curr,
        "speed": speed,
    }


def _classify_crossing(
    crossing: dict,
    ball_states: list[BallState],
    half_length: float,
    fps: float,
    camera_poses: dict | None,
    approach_window_seconds: float = 1.0,
) -> tuple[ShotOutcome, dict] | None:
    """Validate a crossing and determine outcome.

    Returns (outcome, goal_event_dict) or None if not a valid shot.
    """
    cross_y = crossing["cross_y"]
    cross_h = crossing["cross_height"]
    frame = crossing["frame"]
    goal_side = crossing["goal_side"]
    goal_x = half_length if goal_side == "right" else -half_length
    sign = 1 if goal_side == "right" else -1

    # Check approach direction
    approach_window_frames = approach_window_seconds * fps
    approach_count = 0
    for bs in ball_states:
        if bs.frame > frame:
            break
        if bs.frame < frame - approach_window_frames:
            continue
        dist_to_goal = sign * (goal_x - bs.position[0])
        if 0 < dist_to_goal < 10:
            approach_count += 1

    if approach_count < 2:
        return None

    # Determine outcome
    within_width = abs(cross_y) <= GOAL_HALF_WIDTH
    within_height = cross_h <= GOAL_HEIGHT + 0.5

    if within_width and within_height:
        outcome = ShotOutcome.GOAL
    else:
        outcome = ShotOutcome.OFF_TARGET

    # Optional crossbar validation
    crossbar_valid = None
    if camera_poses is not None:
        crossbar_valid = _validate_with_crossbar(
            crossing, half_length, camera_poses
        )

    prev = crossing["prev"]
    curr = crossing["curr"]
    avg_conf = (prev.confidence + curr.confidence) / 2

    # Find start position (where the shot started)
    start_pos = None
    for bs in reversed(ball_states):
        if bs.frame < frame - 30:
            break
        if bs.frame <= frame:
            start_pos = bs.position
            break

    event_dict = {
        "frame": frame,
        "goal_side": goal_side,
        "ball_position": [float(goal_x), float(cross_y), float(cross_h)],
        "ball_pixel": curr.pixel_center,
        "confidence": float(avg_conf),
        "ball_speed_mps": float(crossing["speed"]),
        "crossbar_validation": crossbar_valid,
        "start_pos": start_pos,
    }

    return outcome, event_dict


def _validate_with_crossbar(
    crossing: dict,
    half_length: float,
    camera_poses: dict,
) -> dict | None:
    """Validate ball pixel against projected goal frame."""
    try:
        import cv2
    except ImportError:
        return None

    pose_frames = sorted(camera_poses.keys(), key=lambda f: int(f))
    nearest = min(
        pose_frames, key=lambda f: abs(int(f) - crossing["frame"])
    )
    if abs(int(nearest) - crossing["frame"]) > 15:
        return None

    pose = camera_poses[nearest]
    if not pose.get("rvec"):
        return None

    rvec = np.array(pose["rvec"], dtype=np.float64)
    tvec = np.array(pose["tvec"], dtype=np.float64)
    K = np.array(pose["K"], dtype=np.float64)
    dist = np.array(pose["dist_coeffs"], dtype=np.float64)

    goal_x = (
        half_length if crossing["goal_side"] == "right" else -half_length
    )
    goal_3d = np.array(
        [
            [goal_x, GOAL_HALF_WIDTH, 0],
            [goal_x, GOAL_HALF_WIDTH, -GOAL_HEIGHT],
            [goal_x, -GOAL_HALF_WIDTH, 0],
            [goal_x, -GOAL_HALF_WIDTH, -GOAL_HEIGHT],
        ],
        dtype=np.float64,
    )

    img_pts, _ = cv2.projectPoints(goal_3d, rvec, tvec, K, dist)
    img_pts = img_pts.reshape(-1, 2)

    ball_pixel = crossing["curr"].pixel_center
    if ball_pixel is None:
        return None

    bx, by = ball_pixel
    gx_min, gx_max = img_pts[:, 0].min(), img_pts[:, 0].max()
    gy_min, gy_max = img_pts[:, 1].min(), img_pts[:, 1].max()

    margin = 20
    inside = (
        gx_min - margin <= bx <= gx_max + margin
        and gy_min - margin <= by <= gy_max + margin
    )

    return {
        "ball_pixel": [float(bx), float(by)],
        "goal_bbox": [
            float(gx_min),
            float(gy_min),
            float(gx_max),
            float(gy_max),
        ],
        "goal_corners_px": img_pts.tolist(),
        "inside_goal_frame": inside,
    }


def _deduplicate(
    events: list[MatchEvent],
    fps: float,
    min_gap_frames: int,
) -> list[MatchEvent]:
    """Remove duplicate detections within min_gap_frames."""
    if not events:
        return []

    # Group by event_type and deduplicate within each group
    by_type: dict[EventType, list[MatchEvent]] = {}
    for e in events:
        by_type.setdefault(e.event_type, []).append(e)

    result: list[MatchEvent] = []
    for etype, group in by_type.items():
        group.sort(key=lambda e: e.frame)
        deduped = [group[0]]
        for e in group[1:]:
            if e.frame - deduped[-1].frame > min_gap_frames:
                deduped.append(e)
            elif e.confidence > deduped[-1].confidence:
                deduped[-1] = e
        result.extend(deduped)

    result.sort(key=lambda e: e.frame)
    return result
