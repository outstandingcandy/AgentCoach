"""Chat engine: Bedrock (Claude) grounded in MatchContext via tool_use.

Architecture:
- System prompt = style/format rules + a small schema brief (fps, pitch
  dims, team ids, player ids, event type counts). The schema brief is
  small and stable across turns, so it lives behind a prompt-cache
  breakpoint.
- Match data arrives on demand: Claude calls list_events /
  get_player_stats / get_team_stats / get_frame_snapshot from
  match_tools.py, we run them locally, feed tool_results back, and let
  it produce its final answer.

Why tool_use rather than dumping a digest:
- The model can fetch exactly the slice it needs (e.g. "passes by A-7
  in the first 5 minutes") instead of pattern-matching against a
  pre-baked, truncated list.
- Costs scale with the questions actually asked, not with how dense
  the match is.
- Adding new question types means adding a tool, not editing prompt
  templates.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Iterator

from pathlib import Path

from ..highlights._context import MatchContext
from .code_sandbox import CodeSandbox
from .match_tools import TOOL_DISPATCH, TOOL_SCHEMAS

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:
    boto3 = None
    BotoConfig = None

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a live soccer commentator describing a single match. "
    "Answer the user's questions using ONLY data fetched via the "
    "provided tools and the schema brief in this system message — do "
    "not invent player names, scores, or events. If a tool returns no "
    "data for what the user asked about, say so plainly in the SAME "
    "LANGUAGE as the user's question (e.g. in Chinese: \"赛场数据里没有"
    "这个信息。\"); do not append an English fallback line. Always "
    "respond entirely in the language the user used — never mix "
    "Chinese and English in one answer.\n\n"
    "TOOL USE — these are your only source of match data:\n"
    "- list_events: filter events by type/team/player/time window. "
    "Default to using this for any 'what happened / how many X / "
    "when did Y' question. Possession events are very numerous; "
    "include them only when explicitly asked.\n"
    "- get_player_stats: distance, speed, touches, passes, shots, "
    "goals for one player. Use for 'how far did A-9 run', "
    "'how many shots from B-19', etc.\n"
    "- get_team_stats: possession share, pass success, shots, goals, "
    "tackles, interceptions for one team.\n"
    "- get_frame_snapshot: who is on screen and what's near the "
    "ball at one moment. Use it when the user asks about the moment "
    "they are paused on (the user's `current_time` is provided in "
    "each turn).\n"
    "- run_python: execute Python in a managed sandbox for ad-hoc "
    "computation or plots (heatmaps, distributions, custom "
    "aggregations) the other tools can't answer. Match data is "
    "pre-loaded as ./data/*.json. Save plots to ./output/<name>.png "
    "— they come back as artifacts whose URLs are returned to you. "
    "When you reference one in your answer, embed it as standard "
    "markdown image syntax: `![caption](URL)` so the chat UI renders "
    "it inline. Use plots only when the user asks for a visualisation "
    "or a distribution; for simple numbers, prefer the dedicated "
    "stats tools.\n"
    "Always call a tool before claiming a fact about the match. "
    "Chain tools when needed (e.g. list_events to find the goal, "
    "then get_frame_snapshot of its time to describe the build-up).\n\n"
    "STYLE — speak like a TV commentator, not a data analyst:\n"
    "- Use natural football language: \"in the box\", \"on the wing\", "
    "\"breaks forward\", \"in the final third\", \"near the halfway "
    "line\", \"at the top of the area\", \"down the left flank\", etc.\n"
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


# Cap on how many tool-call rounds we'll let Claude run per user turn.
# Real questions need 1-3 rounds; run_python may need a couple of
# debug iterations on top, so we leave headroom. This is a
# runaway-loop backstop, not a target.
MAX_TOOL_ROUNDS = 10


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


@dataclass
class ChatEngine:
    ctx: MatchContext
    model_id: str = "us.anthropic.claude-opus-4-7"
    region: str = "us-east-1"
    # Generous cap because run_python tool calls can carry hundreds
    # of lines of Python in a single ``code`` argument; truncating at
    # ~1k tokens left input_json incomplete and dispatch failed with
    # "missing required argument: code".
    max_tokens: int = 4096
    max_retries: int = 2
    # Where run_python plot artifacts land on disk; the FastAPI app
    # mounts this directory at ``artifact_url_prefix`` so the chat UI
    # can render the images directly via <img src="...">.
    artifact_dir: Path | None = None
    artifact_url_prefix: str = "/chat_artifacts"

    _client: Any = field(default=None, repr=False)
    _schema_brief: str | None = field(default=None, repr=False)
    _sandbox: CodeSandbox | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if boto3 is None:
            raise ImportError("boto3 is required; install with `pip install boto3`")
        boto_cfg = BotoConfig(
            retries={"max_attempts": 1, "mode": "standard"},
            read_timeout=120,
            connect_timeout=10,
        )
        self._client = boto3.client(
            "bedrock-runtime", region_name=self.region, config=boto_cfg,
        )

    def _ensure_sandbox(self) -> CodeSandbox:
        if self._sandbox is None:
            self._sandbox = CodeSandbox(
                region=self.region,
                pipeline_output_dir=self.ctx.pipeline_output_dir,
            )
        return self._sandbox

    def close(self) -> None:
        if self._sandbox is not None:
            self._sandbox.close()
            self._sandbox = None

    # ------------------------------------------------------------------
    # Schema brief
    # ------------------------------------------------------------------

    def schema_brief(self) -> str:
        if self._schema_brief is None:
            self._schema_brief = self._build_schema_brief()
        return self._schema_brief

    def _build_schema_brief(self) -> str:
        ctx = self.ctx
        duration = ctx.frame_count / ctx.fps if ctx.fps else 0.0

        # Group player ids under their team.
        teams: dict[str, list[str]] = {}
        for pid, tid in ctx.team_assignments.items():
            teams.setdefault(tid, []).append(pid)
        team_lines = []
        for tid in sorted(teams):
            members = sorted(teams[tid])
            team_lines.append(f"  {tid} ({len(members)}): {', '.join(members)}")

        event_counts = Counter(e.get("type", "unknown") for e in ctx.events)
        event_lines = [f"  {k}: {v}" for k, v in event_counts.most_common()]

        return "\n".join([
            "=== MATCH SCHEMA ===",
            f"Video: {ctx.video_path.name}",
            f"Duration: {_fmt_time(duration)} "
            f"({ctx.frame_count} frames @ {ctx.fps:.2f} fps)",
            f"Pitch: {ctx.pitch_length:.1f}m x {ctx.pitch_width:.1f}m",
            "",
            "Teams and player ids (use exactly these strings in tool args):",
            *team_lines,
            "",
            "Event type counts available (call list_events to read them):",
            *event_lines,
        ])

    # ------------------------------------------------------------------
    # Bedrock body builders
    # ------------------------------------------------------------------

    def _build_system_blocks(self, current_time: float) -> list[dict[str, Any]]:
        """System content split into cache-friendly blocks.

        SYSTEM_PROMPT + schema_brief are stable for the lifetime of a
        ChatEngine, so we cache the breakpoint after them. The
        per-turn 'current time' line is small and uncached.
        """
        return [
            {
                "type": "text",
                "text": f"{SYSTEM_PROMPT}\n\n{self.schema_brief()}",
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    f"Current playback time: {_fmt_time(current_time)} "
                    f"({current_time:.2f}s). Use this for "
                    "get_frame_snapshot when the user asks about 'now' "
                    "or 'this moment'."
                ),
            },
        ]

    def _initial_messages(self, history: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Convert the frontend message history into Bedrock format."""
        out: list[dict[str, Any]] = []
        for m in history:
            role = m.get("role")
            content = m.get("content", "")
            if role not in ("user", "assistant") or not content:
                continue
            out.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
        return out

    def _dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call, returning the result dict.

        Errors are returned as ``{"error": str(exc)}`` and surfaced to
        Claude as the tool_result, so it can adjust its plan instead
        of crashing the conversation.
        """
        impl = TOOL_DISPATCH.get(name)
        if impl is None:
            return {"error": f"unknown tool: {name}"}
        # run_python needs sandbox + artifact destination plumbed in;
        # we don't surface those in the public schema. The artifact
        # dir / url prefix come from the engine's configuration.
        injected = dict(args)
        if name == "run_python":
            if self.artifact_dir is None:
                return {"error": (
                    "artifact_dir not configured on ChatEngine; "
                    "run_python is unavailable in this run."
                )}
            injected["sandbox"] = self._ensure_sandbox()
            injected["artifact_dir"] = self.artifact_dir
            injected["artifact_url_prefix"] = self.artifact_url_prefix
        try:
            return impl(self.ctx, **injected)
        except Exception as exc:  # noqa: BLE001
            logger.exception("tool %s failed with args=%s", name, args)
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _log_usage(self, usage: dict[str, Any], round_idx: int) -> None:
        if not usage:
            return
        logger.info(
            "Bedrock usage [round %d]: input=%s cache_read=%s "
            "cache_write=%s output=%s",
            round_idx,
            usage.get("input_tokens"),
            usage.get("cache_read_input_tokens"),
            usage.get("cache_creation_input_tokens"),
            usage.get("output_tokens"),
        )

    # ------------------------------------------------------------------
    # Non-streaming respond (kept for /api/chat parity)
    # ------------------------------------------------------------------

    def respond(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> str:
        system_blocks = self._build_system_blocks(current_time)
        bedrock_messages = self._initial_messages(messages)
        if not bedrock_messages:
            return "(empty message)"

        for round_idx in range(MAX_TOOL_ROUNDS):
            payload = self._invoke_with_retry({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "system": system_blocks,
                "tools": TOOL_SCHEMAS,
                "messages": bedrock_messages,
            })
            self._log_usage(payload.get("usage") or {}, round_idx)

            assistant_blocks = payload.get("content") or []
            tool_uses = [b for b in assistant_blocks if b.get("type") == "tool_use"]

            if not tool_uses:
                # Final answer.
                for b in assistant_blocks:
                    if b.get("type") == "text":
                        return b.get("text", "")
                return ""

            bedrock_messages.append({
                "role": "assistant",
                "content": assistant_blocks,
            })
            tool_results = []
            for tu in tool_uses:
                result = self._dispatch_tool(
                    tu.get("name", ""), tu.get("input") or {},
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.get("id"),
                    "content": json.dumps(result, ensure_ascii=False),
                })
            bedrock_messages.append({"role": "user", "content": tool_results})

        logger.warning("Hit MAX_TOOL_ROUNDS=%d, returning empty", MAX_TOOL_ROUNDS)
        return ""

    def _invoke_with_retry(self, body: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._client.invoke_model(
                    modelId=self.model_id,
                    body=json.dumps(body),
                    contentType="application/json",
                )
                return json.loads(resp["body"].read())
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                wait = 0.5 * (2 ** attempt)
                logger.warning(
                    "Bedrock invoke failed (attempt %d/%d): %s — "
                    "retrying in %.1fs",
                    attempt + 1, self.max_retries + 1, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Bedrock call failed: {last_err}")

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, str]],
        current_time: float,
    ) -> Iterator[str]:
        """Yield response text chunks; tool rounds are internal.

        Why streaming: middle-box idle timeouts (SSH/IDE port forwards,
        load balancers) silently drop connections that go quiet for
        more than a few seconds. We yield bytes within ~1s, keeping
        the TCP connection lively. Tool-call rounds happen inside this
        generator without surfacing their text to the client; only the
        final assistant message is streamed out.
        """
        system_blocks = self._build_system_blocks(current_time)
        bedrock_messages = self._initial_messages(messages)
        if not bedrock_messages:
            yield "(empty message)"
            return

        for round_idx in range(MAX_TOOL_ROUNDS):
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.max_tokens,
                "system": system_blocks,
                "tools": TOOL_SCHEMAS,
                "messages": bedrock_messages,
            }
            resp = self._client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
            )

            # Accumulate the stream into structured content blocks. We
            # don't know up front whether this round will end on
            # tool_use or end_turn, so we buffer text deltas and
            # release them only when we know this is the final round.
            current_blocks: dict[int, dict[str, Any]] = {}
            text_deltas: list[tuple[int, str]] = []
            stop_reason: str | None = None
            usage: dict[str, Any] = {}

            for event in resp["body"]:
                chunk = event.get("chunk")
                if not chunk:
                    continue
                try:
                    data = json.loads(chunk["bytes"])
                except (KeyError, json.JSONDecodeError):
                    continue
                etype = data.get("type")

                if etype == "message_start":
                    msg_usage = (data.get("message") or {}).get("usage") or {}
                    usage.update(msg_usage)

                elif etype == "content_block_start":
                    idx = data.get("index", 0)
                    block = data.get("content_block") or {}
                    if block.get("type") == "tool_use":
                        current_blocks[idx] = {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "_input_json": "",
                        }
                    elif block.get("type") == "text":
                        current_blocks[idx] = {"type": "text", "text": ""}

                elif etype == "content_block_delta":
                    idx = data.get("index", 0)
                    delta = data.get("delta") or {}
                    dtype = delta.get("type")
                    if dtype == "text_delta":
                        text = delta.get("text", "")
                        current_blocks.setdefault(idx, {"type": "text", "text": ""})
                        current_blocks[idx]["text"] += text
                        text_deltas.append((idx, text))
                    elif dtype == "input_json_delta":
                        partial = delta.get("partial_json", "")
                        current_blocks.setdefault(
                            idx, {"type": "tool_use", "_input_json": ""},
                        )
                        current_blocks[idx]["_input_json"] += partial

                elif etype == "message_delta":
                    md = data.get("delta") or {}
                    if "stop_reason" in md:
                        stop_reason = md["stop_reason"]
                    usage.update(data.get("usage") or {})

            self._log_usage(usage, round_idx)

            # Materialise blocks in index order.
            ordered = [current_blocks[i] for i in sorted(current_blocks)]
            for b in ordered:
                if b.get("type") == "tool_use":
                    raw = b.pop("_input_json", "") or "{}"
                    try:
                        b["input"] = json.loads(raw)
                    except json.JSONDecodeError:
                        b["input"] = {}

            if stop_reason == "tool_use":
                # Run tools, append assistant + tool_result, loop again.
                # Don't yield this round's text deltas — they're usually
                # empty anyway, but if Claude narrates while planning,
                # keeping them would interleave with the final answer.
                bedrock_messages.append({
                    "role": "assistant",
                    "content": ordered,
                })
                tool_results = []
                for b in ordered:
                    if b.get("type") != "tool_use":
                        continue
                    result = self._dispatch_tool(b.get("name", ""), b.get("input") or {})
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": b.get("id"),
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                bedrock_messages.append({"role": "user", "content": tool_results})
                continue

            # Final round — stream out the buffered text deltas in order.
            # We buffered them rather than yielding live because we
            # didn't yet know whether this round would end on tool_use.
            for _, text in text_deltas:
                if text:
                    yield text
            return

        logger.warning("Stream hit MAX_TOOL_ROUNDS=%d", MAX_TOOL_ROUNDS)
