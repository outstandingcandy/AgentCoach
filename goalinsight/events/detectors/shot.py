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
from ._utils import find_nearest_player

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
        approach_margin = cfg.get("approach_margin", 3.0)
        tracking_lost_sec = cfg.get("tracking_lost_seconds", 1.0)

        # Step 1a: Detect goal-line crossings
        crossings = _detect_crossings(
            ctx.ball_states, half_length, ctx.fps, max_gap_sec,
            min_speed=shot_speed_min,
        )

        # Step 1b: Detect approach shots (ball near goal + tracking lost or reversal)
        approach_shots = _detect_approach_shots(
            ctx.ball_states, half_length, ctx.fps,
            min_speed=shot_speed_min,
            approach_margin=approach_margin,
            tracking_lost_sec=tracking_lost_sec,
        )
        crossings.extend(approach_shots)
        crossings.sort(key=lambda c: c["frame"])

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

            # Attribute shooter from possession or proximity
            shooter_id, shooter_team, shot_frame = self._find_shooter(
                ctx, crossing["frame"], shooter_lookback_sec,
                goal_side=crossing.get("goal_side"),
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
                    "shooter_frame": shot_frame,
                },
            )
            events.append(shot_event)

            # Emit a convenience GOAL event — inherits player from shot
            if outcome == ShotOutcome.GOAL:
                goal_seq = sum(
                    1
                    for e in events
                    if e.event_type == EventType.GOAL
                ) + 1
                goal_time = round(shot_event.frame / ctx.fps, 1)
                events.append(
                    MatchEvent(
                        event_id=f"goal_{goal_seq:04d}",
                        event_type=EventType.GOAL,
                        frame=shot_event.frame,
                        match_time=goal_time,
                        player_id=shot_event.player_id,
                        team_id=shot_event.team_id,
                        start_position=shot_event.start_position,
                        end_position=shot_event.end_position,
                        confidence=shot_event.confidence,
                        metadata={
                            "shot_event_id": shot_event.event_id,
                            "goal_time": goal_time,
                            **shot_event.metadata,
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
        lookback_seconds: float = 5.0,
        goal_side: str | None = None,
    ) -> tuple[int | None, str | None, int]:
        """Find the player who took the shot.

        Uses pixel-space kick detection (shared with ball_trajectory's
        ``_segment_at_kicks``) to locate the kick frame, then searches
        for the nearest non-goalkeeper player to the ball at that moment.

        Pixel acceleration is more reliable than pitch-coordinate speed
        because pixel centers are direct observations — pitch positions
        depend on 3D fitting quality which degrades for airborne balls.
        """
        from goalinsight.tracking.ball_trajectory import detect_kick_frames

        lookback_frames = int(lookback_seconds * ctx.fps)
        start_frame = max(0, goal_frame - lookback_frames)

        half_length = ctx.pitch_length / 2
        if goal_side is not None:
            goal_x = -half_length if goal_side == "left" else half_length
        else:
            bs = ctx.get_ball_at_frame(goal_frame)
            if bs is not None:
                goal_x = half_length if bs.position[0] > 0 else -half_length
            else:
                goal_x = None

        # Step 1: Find kick frames using pixel-space acceleration
        # (same algorithm as ball_trajectory._segment_at_kicks).
        pixel_obs = [
            (bs.frame, tuple(bs.pixel_center))
            for bs in ctx.ball_states
            if bs.pixel_center is not None
            and start_frame <= bs.frame <= goal_frame
        ]
        kick_frames = detect_kick_frames(pixel_obs)

        # Take the last kick before the goal — that's the shot.
        kick_frame = None
        for kf in reversed(kick_frames):
            if kf <= goal_frame:
                kick_frame = kf
                break

        # Step 2: Search for nearest player around the kick frame.
        search_radius = 3.0
        search_start = kick_frame if kick_frame is not None else goal_frame
        search_end = max(0, search_start - lookback_frames)

        # Exclude goalkeeper-area players (within 3m of goal line)
        if goal_x is not None:
            _gx = goal_x
            exclude_fn = lambda _p, pp: abs(pp[0] - _gx) < 3.0
        else:
            exclude_fn = None

        for f in range(search_start, search_end - 1, -1):
            bs = ctx.get_ball_at_frame(f)
            if bs is None:
                continue

            track_id, team, best_dist = find_nearest_player(
                ctx, f, (bs.position[0], bs.position[1]),
                exclude_fn=exclude_fn,
            )

            if track_id is not None and best_dist <= search_radius:
                logger.info(
                    "Shooter identified: track_id=%d, team=%s, "
                    "distance=%.1fm, frame=%d (kick_frame=%s)",
                    track_id, team, best_dist, f, kick_frame,
                )
                return track_id, team, f

        return None, None, goal_frame


# ------------------------------------------------------------------
# Ported from goalinsight/goal_detection.py
# ------------------------------------------------------------------


def _detect_crossings(
    ball_states: list[BallState],
    half_length: float,
    fps: float,
    max_gap_seconds: float = 1.0,
    min_speed: float = 5.0,
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
        if speed > 50.0 or speed < min_speed:
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


def _detect_approach_shots(
    ball_states: list[BallState],
    half_length: float,
    fps: float,
    min_speed: float = 5.0,
    approach_margin: float = 3.0,
    tracking_lost_sec: float = 1.0,
) -> list[dict]:
    """Detect shots where ball approaches goal but tracking is lost before crossing.

    Two modes:
    1. **Fast approach**: ball near goal at >= min_speed, then tracking lost or reversal.
    2. **Sustained approach**: ball consistently moving toward goal over multiple frames
       at >= 2.0 m/s, reaches within approach_margin of goal line, and tracking is lost
       for an extended period (>= 5× tracking_lost_sec). This catches slower placed shots
       where tracking loss confirms the ball entered/hit the goal area.
    """
    if len(ball_states) < 2:
        return []

    crossings: list[dict] = []
    threshold_x = half_length - approach_margin
    tracking_lost_frames = int(tracking_lost_sec * fps)
    # Sustained approach requires much longer tracking loss to avoid false positives
    sustained_lost_frames = int(5.0 * tracking_lost_sec * fps)
    sustained_min_speed = 2.0  # m/s — minimum for sustained approach
    sustained_min_obs = 4  # Minimum consecutive observations approaching goal

    for i in range(1, len(ball_states)):
        prev = ball_states[i - 1]
        curr = ball_states[i]

        frame_gap = curr.frame - prev.frame
        dt = frame_gap / fps
        if dt <= 0 or dt > 2.0:
            continue

        dx = curr.position[0] - prev.position[0]
        dy = curr.position[1] - prev.position[1]
        speed = math.hypot(dx, dy) / max(dt, 1e-6)
        # Reject impossible speeds
        if speed > 50.0:
            continue

        curr_x = curr.position[0]
        curr_y = curr.position[1]

        # Require reasonable confidence
        if curr.confidence < 0.15:
            continue

        # Determine which goal and whether we're in the approach zone
        for goal_sign, goal_side in [(+1, "right"), (-1, "left")]:
            # Ball must be in the approach zone and moving toward this goal
            in_zone = (goal_sign * curr_x > threshold_x
                       and goal_sign * curr_x <= half_length)
            moving_toward = goal_sign * dx > 0
            if not (in_zone and moving_toward):
                continue

            if speed >= min_speed:
                # Fast approach: normal trigger check
                is_shot = _check_approach_trigger(
                    ball_states, i, tracking_lost_frames, fps, min_speed, goal_sign
                )
            elif speed >= sustained_min_speed:
                # Sustained approach: check consecutive approach + long tracking loss
                is_shot = _check_sustained_approach(
                    ball_states, i, sustained_lost_frames, fps,
                    sustained_min_speed, sustained_min_obs, goal_sign
                )
            else:
                is_shot = False

            if is_shot:
                crossings.append({
                    "frame": curr.frame,
                    "goal_side": goal_side,
                    "cross_y": curr_y,
                    "cross_height": curr.height,
                    "prev": prev,
                    "curr": curr,
                    "speed": speed,
                    "approach_shot": True,
                })

    return crossings


def _check_approach_trigger(
    ball_states: list[BallState],
    idx: int,
    tracking_lost_frames: int,
    fps: float,
    min_speed: float,
    sign: int,
) -> bool:
    """Check if an approach shot triggers at index `idx`.

    Looks ahead a few frames: if tracking is lost soon after (even if
    a smoothed frame sits in between), or ball reverses sharply, it's a shot.
    `sign` is +1 for right goal, -1 for left goal.

    Args:
        ball_states: Full list of ball states.
        idx: Current index in ball_states.
        sign: +1 for right goal approach, -1 for left goal approach.
    """
    curr = ball_states[idx]

    # Last observation in the entire sequence
    if idx >= len(ball_states) - 1:
        return True

    # Look ahead up to 3 entries to find where tracking truly ends.
    # Important: check temporal gaps between consecutive ball_states — a large
    # frame gap between adjacent entries IS the tracking loss we're looking for.
    lookahead = min(3, len(ball_states) - 1 - idx)
    last_near_goal = idx
    for j in range(1, lookahead + 1):
        nxt = ball_states[idx + j]
        prev_state = ball_states[idx + j - 1]
        # If there's a large temporal gap, tracking was lost at prev_state
        if nxt.frame - prev_state.frame > tracking_lost_frames:
            break
        # Still near the goal zone (within approach_margin tolerance)
        if sign * nxt.position[0] > sign * curr.position[0] - 3.0:
            last_near_goal = idx + j
        else:
            break

    # Check if tracking is lost after the last near-goal frame
    last_state = ball_states[last_near_goal]
    if last_near_goal >= len(ball_states) - 1:
        return True
    gap_after = ball_states[last_near_goal + 1].frame - last_state.frame
    if gap_after > tracking_lost_frames:
        return True

    # Check sharp reversal: ball reverses direction fast
    nxt = ball_states[idx + 1]
    nxt_dt = (nxt.frame - curr.frame) / fps
    if nxt_dt > 0:
        nxt_dx = nxt.position[0] - curr.position[0]
        nxt_speed_x = nxt_dx / nxt_dt
        # sign=+1: was going +x, reversal means nxt_speed_x < -min_speed
        # sign=-1: was going -x, reversal means nxt_speed_x > +min_speed
        if sign * nxt_speed_x < -min_speed:
            return True

    return False


def _check_sustained_approach(
    ball_states: list[BallState],
    idx: int,
    tracking_lost_frames: int,
    fps: float,
    min_speed: float,
    min_consecutive: int,
    sign: int,
) -> bool:
    """Check if the ball has been consistently approaching the goal and tracking is then lost.

    This catches slower shots (placed shots, deflections rolling in) where the
    instantaneous speed is below the normal shot threshold but the ball is
    clearly heading toward goal over multiple frames.

    Requires:
    - At least `min_consecutive` observations where ball moves toward the goal
    - Average approach speed >= min_speed
    - Tracking lost for >= tracking_lost_frames after the last observation
    """
    # Count consecutive approach observations looking backwards
    approach_count = 0
    total_dx = 0.0
    total_dt = 0.0

    for j in range(idx, max(idx - 10, 0), -1):
        if j == 0:
            break
        curr_bs = ball_states[j]
        prev_bs = ball_states[j - 1]
        dt = (curr_bs.frame - prev_bs.frame) / fps
        if dt <= 0 or dt > 2.0:
            break
        dx = curr_bs.position[0] - prev_bs.position[0]
        # Must be moving toward the goal (sign=+1 → dx>0, sign=-1 → dx<0)
        if sign * dx <= 0:
            break
        approach_count += 1
        total_dx += abs(dx)
        total_dt += dt

    if approach_count < min_consecutive:
        return False

    avg_speed = total_dx / max(total_dt, 1e-6)
    if avg_speed < min_speed:
        return False

    # Require tracking to be lost for an extended period after
    curr = ball_states[idx]
    lookahead = min(3, len(ball_states) - 1 - idx)
    last_near_goal = idx
    for j in range(1, lookahead + 1):
        nxt = ball_states[idx + j]
        prev_state = ball_states[idx + j - 1]
        # Large temporal gap = tracking already lost
        if nxt.frame - prev_state.frame > tracking_lost_frames:
            break
        if sign * nxt.position[0] > sign * curr.position[0] - 3.0:
            last_near_goal = idx + j
        else:
            break

    if last_near_goal >= len(ball_states) - 1:
        return True

    gap_after = ball_states[last_near_goal + 1].frame - ball_states[last_near_goal].frame
    return gap_after >= tracking_lost_frames


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

    # Check approach direction (skip for approach shots — already validated)
    if not crossing.get("approach_shot"):
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
    from ...utils.projection import project_points_2d

    pose_frames = sorted(camera_poses.keys(), key=lambda f: int(f))
    nearest = min(
        pose_frames, key=lambda f: abs(int(f) - crossing["frame"])
    )
    if abs(int(nearest) - crossing["frame"]) > 15:
        return None

    pose = camera_poses[nearest]
    if not pose.get("rvec"):
        return None

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

    img_pts = project_points_2d(
        goal_3d, pose["rvec"], pose["tvec"],
        np.asarray(pose["K"], dtype=np.float64),
        np.asarray(pose["dist_coeffs"], dtype=np.float64),
    )

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
