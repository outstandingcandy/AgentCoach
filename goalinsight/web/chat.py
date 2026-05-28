"""Chat engine: Bedrock (Claude) grounded in MatchContext structured data.

The LLM never sees pixels — only a compact textual digest of match-wide
stats plus a snapshot of the frame the user is currently paused on.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from ..highlights._context import MatchContext

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a live soccer commentator describing a single match. "
    "Answer the user's questions using ONLY the structured match data "
    "provided in this system message. If the data does not contain the "
    "answer, say so plainly in the SAME LANGUAGE as the user's question "
    "(e.g. in Chinese: \"赛场数据里没有这个信息。\"); "
    "do not append an English fallback line. "
    "Always respond entirely in the language the user used — never mix "
    "Chinese and English in one answer. "
    "Do not invent player names, scores, or events.\n\n"
    "STYLE — speak like a TV commentator, not a data analyst:\n"
    "- Use natural football language: \"in the box\", \"on the wing\", "
    "\"breaks forward\", \"in the final third\", \"near the halfway line\", "
    "\"at the top of the area\", \"down the left flank\", etc.\n"
    "- Refer to players as e.g. \"A-9\" or \"number 9 of team A\"; the "
    "letter prefix is the team and the number is the shirt. Use \"the "
    "goalkeeper\" for GK ids.\n"
    "- NEVER quote raw pitch coordinates, meter offsets, frame numbers, "
    "or speeds in m/s. Translate them into pitch zones and qualitative "
    "phrases (e.g. \"a hard, low strike\", \"a long ball forward\", "
    "\"a quick one-two\").\n"
    "- Translate timestamps into match-clock minutes (\"in the 7th\", "
    "\"a moment ago\", \"just before the goal\") rather than MM:SS.FF "
    "frame counts.\n"
    "- Whenever you mention a specific moment from the match, append a "
    "machine-readable jump tag in square brackets right after the "
    "phrase, formatted as [MM:SS] (e.g. \"the goal in the 7th minute "
    "[07:27]\"). The frontend turns these tags into clickable seek "
    "buttons. Use one tag per moment; do NOT use this format for "
    "anything other than match timestamps.\n"
    "- Keep answers tight (2-5 sentences) unless the user asks for "
    "detail. Energetic but not theatrical.\n\n"
    "DATA INTERPRETATION — coordinates are in meters with the pitch "
    "centered at (0,0); use them internally to decide zones, but never "
    "show the numbers. Rough zone map (length=pitch_length, width="
    "pitch_width):\n"
    "- x near -length/2 = team A's defensive end / left goal;\n"
    "  x near +length/2 = the right goal end.\n"
    "- |x| > length/3 with small |y| = penalty-area / final-third.\n"
    "- |y| > width/3 = wing / touchline; |y| ~ 0 = central channel.\n"
    "Pick whichever team is attacking each end based on event context, "
    "and describe action from the attacker's perspective."
)


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _nearest_frame_key(mapping: dict[str, Any], target: int) -> str | None:
    """Return the string key in *mapping* whose int value is closest to *target*."""
    if not mapping:
        return None
    if str(target) in mapping:
        return str(target)
    best_k = None
    best_d = None
    for k in mapping:
        try:
            diff = abs(int(k) - target)
        except ValueError:
            continue
        if best_d is None or diff < best_d:
            best_d = diff
            best_k = k
    return best_k


@dataclass
class ChatEngine:
    ctx: MatchContext
    model_id: str = "us.anthropic.claude-opus-4-7"
    region: str = "us-east-1"
    max_tokens: int = 1024
    max_retries: int = 2
    event_window_s: float = 5.0

    _client: Any = field(default=None, repr=False)
    _digest: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if boto3 is None:
            raise ImportError("boto3 is required; install with `pip install boto3`")
        boto_cfg = BotoConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            read_timeout=60,
            connect_timeout=10,
        )
        self._client = boto3.client(
            "bedrock-runtime", region_name=self.region, config=boto_cfg,
        )

    # ------------------------------------------------------------------
    # Context builders
    # ------------------------------------------------------------------

    def digest(self) -> str:
        if self._digest is None:
            self._digest = self._build_digest()
        return self._digest

    def _build_digest(self) -> str:
        ctx = self.ctx
        duration = ctx.frame_count / ctx.fps if ctx.fps else 0.0

        teams = Counter(ctx.team_assignments.values())
        team_lines = [f"  {team}: {n} tracked players" for team, n in teams.items()]

        events = ctx.events
        event_counts = Counter(e.get("type", "unknown") for e in events)

        # Always include every goal and shot (these are what users ask
        # about most). Add other salient events up to a limit.
        def fmt_event(e: dict) -> str:
            mt = e.get("match_time", e.get("frame", 0) / ctx.fps)
            outcome = e.get("metadata", {}).get("outcome", "")
            suffix = f" ({outcome})" if outcome else ""
            return (
                f"  {_fmt_time(mt)} f={e.get('frame')} "
                f"{e.get('type')}{suffix} by {e.get('player_id', '?')} "
                f"[{e.get('team_id', '?')}]"
            )

        key_rows = [fmt_event(e) for e in events
                    if e.get("type") in {"goal", "shot"}]
        other_types = {"pass", "tackle", "interception"}
        other_rows = [fmt_event(e) for e in events
                      if e.get("type") in other_types][:30]
        timeline_rows = key_rows + other_rows

        parts = [
            "=== MATCH DIGEST ===",
            f"Video: {ctx.video_path.name}",
            f"Duration: {_fmt_time(duration)} "
            f"({ctx.frame_count} frames @ {ctx.fps:.2f} fps)",
            f"Pitch: {ctx.pitch_length:.1f}m x {ctx.pitch_width:.1f}m",
            "",
            "Teams (from team_assignments):",
            *team_lines,
            "",
            "Event counts (from event_detection):",
            *[f"  {k}: {v}" for k, v in event_counts.most_common()],
        ]
        if timeline_rows:
            parts.extend([
                "",
                f"Salient events timeline (first {len(timeline_rows)}):",
                *timeline_rows,
            ])
        return "\n".join(parts)

    def frame_snapshot(self, current_time: float) -> str:
        ctx = self.ctx
        frame = int(round(current_time * ctx.fps))

        # Players
        player_key = _nearest_frame_key(ctx.player_tracks, frame)
        players = ctx.player_tracks.get(player_key, []) if player_key else []
        compact_players = []
        for t in players:
            compact_players.append({
                "id": t.get("track_id"),
                "team": t.get("team"),
                "role": t.get("role"),
                "jersey": t.get("jersey_number"),
                "pitch": [round(v, 2) for v in (t.get("pitch_position") or [None, None])],
            })

        # Ball
        ball_key = _nearest_frame_key(ctx.ball_tracks, frame)
        ball_raw = ctx.ball_tracks.get(ball_key) if ball_key else None
        ball_info: dict[str, Any] | None = None
        if ball_raw:
            ball_info = {
                "pitch": [round(v, 2) for v in (ball_raw.get("pitch_position") or [None, None])],
                "height_m": round(ball_raw.get("height") or 0.0, 2),
                "on_ground": ball_raw.get("on_ground"),
            }

        # Events within ±window
        w_frames = int(self.event_window_s * ctx.fps)
        lo, hi = frame - w_frames, frame + w_frames
        nearby_events = []
        for e in ctx.events:
            ef = e.get("frame", -1)
            if lo <= ef <= hi:
                nearby_events.append({
                    "t": _fmt_time(e.get("match_time", ef / ctx.fps)),
                    "frame": ef,
                    "type": e.get("type"),
                    "player": e.get("player_id"),
                    "team": e.get("team_id"),
                    "meta": e.get("metadata"),
                })

        return (
            f"=== FRAME SNAPSHOT (time={_fmt_time(current_time)} frame={frame}) ===\n"
            f"Players visible ({len(compact_players)}):\n"
            f"{json.dumps(compact_players, separators=(',', ':'))}\n"
            f"Ball: {json.dumps(ball_info, separators=(',', ':'))}\n"
            f"Events within ±{self.event_window_s:.0f}s ({len(nearby_events)}):\n"
            f"{json.dumps(nearby_events, separators=(',', ':'))}"
        )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def respond(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> str:
        system = (
            f"{SYSTEM_PROMPT}\n\n{self.digest()}\n\n{self.frame_snapshot(current_time)}"
        )
        # Claude API expects user/assistant alternation. We pass through
        # whatever the frontend sends, stripping anything that isn't one
        # of those roles.
        bedrock_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role not in ("user", "assistant") or not content:
                continue
            bedrock_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        if not bedrock_messages:
            return "(empty message)"

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": bedrock_messages,
        }
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                payload = json.loads(resp["body"].read())
                for p in payload.get("content", []):
                    if p.get("type") == "text":
                        return p["text"]
                return ""
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = 0.5 * (2 ** attempt)
                logger.warning(
                    "Bedrock invoke failed (attempt %d/%d): %s — retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        logger.error("Bedrock invoke permanently failed: %s", last_err)
        raise RuntimeError(f"Bedrock call failed: {last_err}")

    def stream(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> Iterator[str]:
        """Yield response text chunks from Bedrock as they arrive.

        Why streaming: middle-box idle timeouts (SSH/IDE port forwards,
        load balancers) silently drop connections that go quiet for more
        than a few seconds while we wait on the LLM. Streaming sends
        bytes within ~1s, keeping the TCP connection lively.
        """
        system = (
            f"{SYSTEM_PROMPT}\n\n{self.digest()}\n\n{self.frame_snapshot(current_time)}"
        )
        bedrock_messages: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content", "")
            if role not in ("user", "assistant") or not content:
                continue
            bedrock_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        if not bedrock_messages:
            yield "(empty message)"
            return

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": bedrock_messages,
        }
        resp = self._client.invoke_model_with_response_stream(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
        )
        for event in resp["body"]:
            chunk = event.get("chunk")
            if not chunk:
                continue
            try:
                data = json.loads(chunk["bytes"])
            except (KeyError, json.JSONDecodeError):
                continue
            if data.get("type") == "content_block_delta":
                delta = data.get("delta") or {}
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    if text:
                        yield text
