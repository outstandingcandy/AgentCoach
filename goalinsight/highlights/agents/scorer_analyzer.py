"""ScorerAnalyzer — player attribution and segment planning for goal events."""

from __future__ import annotations

import logging
import math
from typing import Any

from .._closeup import interpolate_bbox
from .._context import MatchContext
from .._temporal import find_buildup_start, find_celebration_end
from .._types import AnalyzedEvent, ClipSegment, Event
from .base import BaseSceneAnalyzer

logger = logging.getLogger(__name__)


class ScorerAnalyzer(BaseSceneAnalyzer):
    """Analyze goal events: identify scorer and plan clip segments."""

    supported_events = ["goal"]

    def analyze(
        self,
        event: Event,
        ctx: MatchContext,
        config: dict[str, Any],
    ) -> AnalyzedEvent:
        goal_frame = event.frame
        goal_side = event.metadata.get("goal_side", "right")
        fps = ctx.fps

        temporal_cfg = config.get("temporal", {})
        buildup_max = temporal_cfg.get("buildup_max_seconds", 10.0)
        buildup_pad = temporal_cfg.get("buildup_padding_seconds", 2.0)
        celebration_dur = temporal_cfg.get("celebration_seconds", 4.0)
        goal_moment_dur = temporal_cfg.get("goal_moment_seconds", 2.0)

        # --- Player attribution ---
        scorer = self._find_scorer(ctx, goal_frame, goal_side)
        key_players: list[dict[str, Any]] = []
        if scorer is not None:
            key_players.append(scorer)
            logger.info(
                "Scorer identified: track_id=%d, team=%s, distance=%.1fm",
                scorer["track_id"],
                scorer["team"],
                scorer["distance"],
            )
        else:
            logger.warning("Could not identify scorer for goal at frame %d", goal_frame)

        # --- Temporal windows ---
        buildup_start = find_buildup_start(
            ctx, goal_frame, goal_side,
            max_seconds=buildup_max, padding_seconds=buildup_pad,
        )
        celebration_end = find_celebration_end(
            goal_frame, fps,
            duration_seconds=celebration_dur,
            total_frames=ctx.frame_count or None,
        )

        half_moment = int(goal_moment_dur * fps / 2)
        shot_dur = temporal_cfg.get("shot_closeup_seconds", 3.0)
        half_shot = int(shot_dur * fps / 2)

        # Scorer's last-touch frame (the moment of the shot)
        shot_frame = scorer["frame"] if scorer else goal_frame

        # --- Segment plan ---
        segments: list[ClipSegment] = []

        # 1. Build-up: medium shot following the ball
        segments.append(
            ClipSegment(
                name="buildup",
                start_frame=buildup_start,
                end_frame=max(buildup_start, shot_frame - half_shot - 1),
                view_type="medium",
                focus_target="ball",
            )
        )

        # 2. Shot close-up: follow the scorer around the shooting moment
        scorer_id = scorer["track_id"] if scorer else None
        segments.append(
            ClipSegment(
                name="shot",
                start_frame=max(0, shot_frame - half_shot),
                end_frame=shot_frame + half_shot,
                view_type="closeup",
                focus_target="player" if scorer_id else "ball",
                focus_track_id=scorer_id,
            )
        )

        # 3. Goal moment: close-up following the ball into the net
        segments.append(
            ClipSegment(
                name="goal_moment",
                start_frame=max(0, goal_frame - half_moment),
                end_frame=goal_frame + half_moment,
                view_type="closeup",
                focus_target="ball",
                overlays=[{"type": "text", "text": "GOAL!", "position": "top_center"}],
            )
        )

        # 4. Celebration: medium shot following the scorer
        segments.append(
            ClipSegment(
                name="celebration",
                start_frame=goal_frame + half_moment + 1,
                end_frame=celebration_end,
                view_type="medium",
                focus_target="player",
                focus_track_id=scorer_id,
            )
        )

        return AnalyzedEvent(
            event=event,
            key_players=key_players,
            segments=segments,
        )

    # ------------------------------------------------------------------
    # Player attribution internals
    # ------------------------------------------------------------------

    def _find_scorer(
        self,
        ctx: MatchContext,
        goal_frame: int,
        goal_side: str,
        search_radius: float = 3.0,
        lookback_frames: int = 90,
    ) -> dict[str, Any] | None:
        """Find the player who scored.

        Strategy:
        1. Determine which team is attacking the goal (by spatial distribution).
        2. At the goal frame, find the nearest attacking player to the ball.
        3. If none within radius, search the preceding lookback_frames.
        """
        # Try nearby frames if goal_frame has no tracks
        attacking_team = None
        for offset in range(0, 30):
            for f in (goal_frame - offset, goal_frame + offset):
                if f < 0:
                    continue
                attacking_team = self._infer_attacking_team(ctx, f, goal_side)
                if attacking_team is not None:
                    break
            if attacking_team is not None:
                break

        logger.info("Attacking team: %s (goal_side=%s)", attacking_team, goal_side)

        # Search from goal_frame backwards
        for f in range(goal_frame, max(0, goal_frame - lookback_frames) - 1, -1):
            ball = ctx.get_ball_at_frame(f)
            if ball is None or ball.get("pitch_position") is None:
                continue
            bx, by = ball["pitch_position"]

            tracks = ctx.get_tracks_at_frame(f)
            best_track = None
            best_dist = float("inf")

            half_length = ctx.pitch_length / 2
            goal_x = -half_length if goal_side == "left" else half_length

            for t in tracks:
                pp = t.get("pitch_position")
                if pp is None:
                    continue
                team = ctx.get_team_for_track(t["track_id"])
                if team == "referee":
                    continue
                # Skip players hugging the goal line — likely the goalkeeper
                # trying to save, not the scorer. Threshold: within 3m of
                # the goal line.
                dist_to_goal_line = abs(pp[0] - goal_x)
                if dist_to_goal_line < 3.0:
                    continue
                # Only consider the attacking team (or unknown if team not determined)
                if attacking_team and team not in (attacking_team, "unknown"):
                    continue
                dist = math.hypot(pp[0] - bx, pp[1] - by)
                if dist < best_dist:
                    best_dist = dist
                    best_track = t

            if best_track is not None and best_dist <= search_radius:
                return {
                    "track_id": best_track["track_id"],
                    "role": "scorer",
                    "team": ctx.get_team_for_track(best_track["track_id"]),
                    "distance": best_dist,
                    "frame": f,
                }

        return None

    def _infer_attacking_team(
        self,
        ctx: MatchContext,
        goal_frame: int,
        goal_side: str,
    ) -> str | None:
        """Infer which team is attacking the given goal side.

        Heuristic: the defending team has its goalkeeper + defenders near
        the goal, so the team with FEWER players in the goal-side half is
        the attacking team (their home half is the opposite side).

        Example: goal on left → defending team clusters in the left half
        → the team with fewer players in the left half is the attacker.
        """
        tracks = ctx.get_tracks_at_frame(goal_frame)
        if not tracks:
            return None

        # Count players in the goal-side half per team
        team_in_goal_half: dict[str, int] = {}
        team_total: dict[str, int] = {}
        for t in tracks:
            pp = t.get("pitch_position")
            if pp is None:
                continue
            team = ctx.get_team_for_track(t["track_id"])
            if team in ("referee", "unknown"):
                continue

            team_total[team] = team_total.get(team, 0) + 1

            in_goal_half = (
                (goal_side == "left" and pp[0] < 0)
                or (goal_side == "right" and pp[0] > 0)
            )
            if in_goal_half:
                team_in_goal_half[team] = team_in_goal_half.get(team, 0) + 1

        if not team_total:
            return None

        # The defending team has a higher fraction of players in the goal-side half.
        # The attacking team has a lower fraction → they came from the other side.
        team_fractions = {
            team: team_in_goal_half.get(team, 0) / team_total[team]
            for team in team_total
        }
        # Attacking team = lowest fraction in goal-side half
        return min(team_fractions, key=team_fractions.get)
