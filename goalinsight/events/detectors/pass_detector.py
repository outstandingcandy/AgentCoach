"""PassDetector — detects passes from possession transitions."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

from .._base import BaseEventDetector
from .._context import EventDetectionContext
from .._registry import register_detector
from .._types import EventType, MatchEvent, PassOutcome
from ._utils import find_nearest_player

logger = logging.getLogger(__name__)


@register_detector
class PassDetector(BaseEventDetector):
    """Detect passes by analyzing possession transitions with ball speed jumps.

    Two paths emit pass events; the second backfills cases the first
    misses:

    1. **Possession-bridge path** (legacy): when two consecutive
       PossessionSpans straddle a brief ball-speed jump, emit a pass
       between their players. Misses any pass where either side's
       touch was too short to satisfy ``min_consecutive_seconds`` —
       e.g. a one-touch volley, or the ball-detection gap during a
       hard kick that drops ball_states for ~1 s.
    2. **Speed-transition path** (new): scan ball_states directly for
       a slow→fast→slow envelope (the kinematic signature of a pass).
       Attribute the passer by pre-kick possession (mirrors the shot
       detector's logic) and the receiver by who first sits within
       1.0 m of the ball after the speed drops back. Skipped if the
       same window is already covered by a possession-bridge pass.
    """

    name = "pass"
    depends_on = ["possession"]

    def detect(
        self, ctx: EventDetectionContext, config: dict[str, Any]
    ) -> list[MatchEvent]:
        cfg = config.get("events", {}).get("pass", {})
        speed_threshold = cfg.get("pass_speed_threshold", 5.0)
        max_transit_sec = cfg.get("max_transit_seconds", 1.0)

        events: list[MatchEvent] = []

        # ----- Path 1: possession-bridge -----
        events.extend(self._detect_from_spans(
            ctx, speed_threshold, max_transit_sec,
        ))

        # ----- Path 2: speed-transition (backfill) -----
        events.extend(self._detect_from_transitions(
            ctx, cfg, existing_events=events,
        ))

        # Final scrub: a pass where the passer == receiver is by
        # definition not a pass. The possession-bridge path can still
        # emit these when the same player owns two adjacent spans
        # separated by a brief ball-speed blip (e.g. a header back to
        # self, or a tracker artefact). Drop them instead of polluting
        # downstream.
        events = [
            e for e in events
            if e.player_id is None
            or str(e.player_id) != str(e.metadata.get("receiver_id"))
        ]

        # Re-sequence event_ids so they remain pass_0001, pass_0002, …
        events.sort(key=lambda e: e.frame)
        for i, e in enumerate(events, 1):
            e.event_id = f"pass_{i:04d}"

        logger.info(
            "PassDetector: %d pass(es) (%d successful, %d failed)",
            len(events),
            sum(
                1
                for e in events
                if e.metadata.get("outcome") == PassOutcome.SUCCESSFUL.value
            ),
            sum(
                1
                for e in events
                if e.metadata.get("outcome") == PassOutcome.FAILED.value
            ),
        )
        return events

    # ------------------------------------------------------------------
    # Path 1: possession-bridge (original)
    # ------------------------------------------------------------------
    def _detect_from_spans(
        self,
        ctx: EventDetectionContext,
        speed_threshold: float,
        max_transit_sec: float,
    ) -> list[MatchEvent]:
        spans = ctx.possession_spans
        if len(spans) < 2:
            return []

        events: list[MatchEvent] = []

        for i in range(len(spans) - 1):
            span_a = spans[i]
            span_b = spans[i + 1]

            gap_sec = (span_b.start_frame - span_a.end_frame) / ctx.fps
            if gap_sec > max_transit_sec or gap_sec < 0:
                continue

            max_speed = self._max_speed_in_range(
                ctx, span_a.end_frame, span_b.start_frame
            )
            if max_speed < speed_threshold:
                continue

            outcome = (
                PassOutcome.SUCCESSFUL
                if span_a.team_id == span_b.team_id
                else PassOutcome.FAILED
            )
            pass_length = math.hypot(
                span_b.start_position[0] - span_a.end_position[0],
                span_b.start_position[1] - span_a.end_position[1],
            )
            events.append(MatchEvent(
                event_id="pass_PENDING",
                event_type=EventType.PASS,
                frame=span_a.end_frame,
                match_time=span_a.end_frame / ctx.fps,
                player_id=span_a.player_id,
                team_id=span_a.team_id,
                start_position=span_a.end_position,
                end_position=span_b.start_position,
                confidence=1.0,
                metadata={
                    "outcome": outcome.value,
                    "receiver_id": span_b.player_id,
                    "receiver_team": span_b.team_id,
                    "pass_length": round(pass_length, 2),
                    "is_successful": outcome == PassOutcome.SUCCESSFUL,
                    "source": "possession_bridge",
                },
            ))
        return events

    # ------------------------------------------------------------------
    # Path 2: speed-transition (backfill for short / sparse touches)
    # ------------------------------------------------------------------
    def _detect_from_transitions(
        self,
        ctx: EventDetectionContext,
        cfg: dict[str, Any],
        existing_events: list[MatchEvent],
    ) -> list[MatchEvent]:
        # Tunables. Mirror the shot detector's pitch-speed thresholds:
        # at 60 fps with median smoothing a stationary ball still reads
        # ~3-5 m/s, so the slow band has to be lenient.
        SLOW_BEFORE = float(cfg.get("transition_slow_before", 7.0))
        FAST_TRIGGER = float(cfg.get("transition_fast_trigger", 10.0))
        SUSTAIN_FLOOR = float(cfg.get("transition_sustain_floor", 8.0))
        # Consider a pass complete once speed drops back below this. Higher
        # than SLOW_BEFORE because a moving rolling ball under control is
        # not "stopped".
        SETTLE_BELOW = float(cfg.get("transition_settle_below", 7.0))
        possession_radius = float(cfg.get("transition_possession_radius_m", 1.0))
        # Receive radius is intentionally looser than the passer's
        # possession_radius: when the receiver first picks up the ball
        # they're often still 1.5–2 m away (foot extending toward it,
        # or a teammate trapping a long ball). 2 m matches the pitch-
        # space distance where bbox-to-ball can plausibly mean "this
        # player just received it".
        receive_radius = float(cfg.get("transition_receive_radius_m", 2.0))
        passer_lookback_sec = float(cfg.get("transition_passer_lookback_s", 1.0))
        # Receiver search runs forward from settle frame for this long.
        receiver_lookahead_sec = float(cfg.get("transition_receiver_lookahead_s", 1.5))
        # Maximum allowed flight duration. Anything longer is more
        # likely a goal-kick / clearance / shot than a pass.
        max_flight_sec = float(cfg.get("transition_max_flight_s", 4.0))

        if not ctx.ball_states:
            return []

        # Index existing pass frames so the transition path doesn't
        # double-emit on top of the possession-bridge path.
        existing_window = int(0.5 * ctx.fps)  # +/- 0.5s
        existing_frames = sorted(e.frame for e in existing_events)

        def _already_covered(f: int) -> bool:
            for ef in existing_frames:
                if abs(ef - f) <= existing_window:
                    return True
            return False

        # Walk ball_states, find slow→fast → ... →slow envelopes.
        states = ctx.ball_states
        events: list[MatchEvent] = []
        i = 1
        n = len(states)
        while i < n:
            prev_bs = states[i - 1]
            curr_bs = states[i]
            if not (prev_bs.speed <= SLOW_BEFORE
                    and curr_bs.speed >= FAST_TRIGGER):
                i += 1
                continue
            kick_frame = prev_bs.frame
            # Walk forward to find the settle frame where speed drops back.
            j = i
            settle_idx: int | None = None
            while j < n:
                bs_j = states[j]
                if bs_j.frame > kick_frame + int(max_flight_sec * ctx.fps):
                    break
                if (j > i and bs_j.speed <= SETTLE_BELOW
                        and bs_j.speed < SUSTAIN_FLOOR):
                    settle_idx = j
                    break
                j += 1
            if settle_idx is None:
                # Speed never settled within window — could be a shot
                # (handled by ShotDetector) or the ball ran out of view.
                # Skip and move past this kick so we don't loop on it.
                i = j + 1
                continue
            settle_frame = states[settle_idx].frame
            settle_pos = states[settle_idx].position
            kick_pos = prev_bs.position

            i = settle_idx + 1  # advance past this envelope

            if _already_covered(kick_frame):
                continue

            passer_id, passer_team = self._passer_at_kick(
                ctx, kick_frame, passer_lookback_sec, possession_radius,
            )
            if passer_id is None:
                continue

            receiver_id, receiver_team, receive_frame = self._receiver_after_settle(
                ctx, settle_frame, receiver_lookahead_sec, receive_radius,
            )
            if receiver_id is None or str(receiver_id) == str(passer_id):
                # Either nobody picked it up cleanly (out of bounds / lost
                # to GK / shot — we don't emit a pass), or the same
                # player retook it (header-back-to-self, deflection).
                continue

            outcome = (
                PassOutcome.SUCCESSFUL
                if passer_team == receiver_team
                else PassOutcome.FAILED
            )
            pass_length = math.hypot(
                settle_pos[0] - kick_pos[0], settle_pos[1] - kick_pos[1],
            )

            events.append(MatchEvent(
                event_id="pass_PENDING",
                event_type=EventType.PASS,
                frame=kick_frame,
                match_time=kick_frame / ctx.fps,
                player_id=passer_id,
                team_id=passer_team,
                start_position=list(kick_pos),
                end_position=list(settle_pos),
                confidence=0.9,
                metadata={
                    "outcome": outcome.value,
                    "receiver_id": receiver_id,
                    "receiver_team": receiver_team,
                    "pass_length": round(pass_length, 2),
                    "is_successful": outcome == PassOutcome.SUCCESSFUL,
                    "source": "speed_transition",
                    "settle_frame": settle_frame,
                    "receive_frame": receive_frame,
                },
            ))
        return events

    # ---- helpers ----
    @staticmethod
    def _passer_at_kick(
        ctx: EventDetectionContext,
        kick_frame: int,
        lookback_sec: float,
        possession_radius: float,
    ) -> tuple[Any, str | None]:
        """Same idea as ShotDetector._find_shooter: weighted-vote over the
        pre-kick window, the player who was holding the ball wins.
        """
        win_pre = int(lookback_sec * ctx.fps)
        win_start = max(0, kick_frame - win_pre)
        scores: dict = defaultdict(float)
        team_of: dict = {}
        for f in range(win_start, kick_frame + 1):
            bs = ctx.get_ball_at_frame(f)
            if bs is None:
                continue
            tid, team, dist = find_nearest_player(
                ctx, f, (bs.position[0], bs.position[1]),
            )
            if tid is None or dist > possession_radius:
                continue
            w = 0.1 + 0.9 * ((f - win_start) / max(1, kick_frame - win_start))
            scores[tid] += w
            team_of[tid] = team
        if not scores:
            return None, None
        tid = max(scores, key=lambda t: scores[t])
        return tid, team_of[tid]

    @staticmethod
    def _receiver_after_settle(
        ctx: EventDetectionContext,
        settle_frame: int,
        lookahead_sec: float,
        receive_radius: float,
    ) -> tuple[Any, str | None, int | None]:
        """First non-GK player to sit within ``receive_radius`` of the
        ball after the envelope settles."""
        win_post = int(lookahead_sec * ctx.fps)
        for f in range(settle_frame, settle_frame + win_post + 1):
            bs = ctx.get_ball_at_frame(f)
            if bs is None:
                continue
            tid, team, dist = find_nearest_player(
                ctx, f, (bs.position[0], bs.position[1]),
            )
            if tid is None or dist > receive_radius:
                continue
            return tid, team, f
        return None, None, None

    @staticmethod
    def _max_speed_in_range(
        ctx: EventDetectionContext, start_frame: int, end_frame: int
    ) -> float:
        """Find the maximum ball speed between two frames."""
        max_speed = 0.0
        for bs in ctx.ball_states:
            if bs.frame < start_frame:
                continue
            if bs.frame > end_frame:
                break
            if bs.speed > max_speed:
                max_speed = bs.speed
        return max_speed
