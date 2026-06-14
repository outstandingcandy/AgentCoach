"""Tool functions Claude can call to query MatchContext on demand.

Replaces the static "match digest" approach (chat.py used to dump a
truncated event list + counters into the system prompt). Now the model
sees only a small schema in the system prompt and uses these tools to
fetch the data it actually needs to answer each question.

Each TOOL_SCHEMAS entry is a Bedrock/Anthropic tool spec; each
TOOL_DISPATCH entry is the matching Python implementation. Keep the
two in sync.

Conventions:
- Time is always seconds since the start of the video (frame / fps).
- Player ids are strings like "A-9" (team prefix + jersey number).
- Team ids are exactly the values in team_assignments.json
  (typically "team_A", "team_B", "referee", "unknown").
- All numeric outputs are rounded to 2 decimals to keep tool_result
  payloads small.

run_python is special: it talks to the AgentCore Code Interpreter
sandbox (see code_sandbox.py) and needs a sandbox + artifact dir
plumbed in by the caller. ChatEngine injects these via the dispatcher
so the schema visible to the LLM stays a single ``code`` arg.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ._context import MatchContext
from .code_sandbox import CodeSandbox, SANDBOX_DATA_DIR, SANDBOX_OUTPUT_DIR


# Maximum number of items returned by list-shaped tools. Keeps tool_result
# token cost bounded — Claude can paginate via time_range_s if it needs more.
DEFAULT_LIMIT = 50
MAX_LIMIT = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _frame_to_time(frame: int, fps: float) -> float:
    return round(frame / fps, 2) if fps else 0.0


def _in_time_range(t: float, time_range_s: list[float] | None) -> bool:
    if not time_range_s:
        return True
    lo, hi = time_range_s[0], time_range_s[1]
    return lo <= t <= hi


def _summarise_event(e: dict, fps: float) -> dict[str, Any]:
    """Compact event dict suitable for tool_result output."""
    out = {
        "event_id": e.get("event_id"),
        "type": e.get("type"),
        "time_s": _frame_to_time(e.get("frame", 0), fps),
        "frame": e.get("frame"),
        "player_id": e.get("player_id"),
        "team_id": e.get("team_id"),
    }
    md = e.get("metadata") or {}
    # Promote the few fields LLM commentary actually needs; full metadata
    # is available via get_event(event_id) if requested separately.
    for k in ("outcome", "receiver_id", "receiver_team", "pass_length",
              "duration_sec", "shooter_frame", "is_successful"):
        if k in md:
            out[k] = md[k]
    return out


def _nearest_frame_key(mapping: dict[str, Any], target: int) -> str | None:
    if not mapping:
        return None
    if str(target) in mapping:
        return str(target)
    best_k, best_d = None, None
    for k in mapping:
        try:
            diff = abs(int(k) - target)
        except ValueError:
            continue
        if best_d is None or diff < best_d:
            best_d, best_k = diff, k
    return best_k


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def list_events(
    ctx: MatchContext,
    *,
    types: list[str] | None = None,
    team_id: str | None = None,
    player_id: str | None = None,
    time_range_s: list[float] | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Return events filtered by type / team / player / time window."""
    limit = min(int(limit), MAX_LIMIT)
    type_set = set(types) if types else None

    matched: list[dict[str, Any]] = []
    truncated = False
    for e in ctx.events:
        if type_set and e.get("type") not in type_set:
            continue
        if team_id and e.get("team_id") != team_id:
            continue
        if player_id and e.get("player_id") != player_id:
            continue
        t = _frame_to_time(e.get("frame", 0), ctx.fps)
        if not _in_time_range(t, time_range_s):
            continue
        if len(matched) >= limit:
            truncated = True
            break
        matched.append(_summarise_event(e, ctx.fps))

    return {
        "events": matched,
        "count": len(matched),
        "truncated": truncated,
    }


def get_player_stats(
    ctx: MatchContext,
    *,
    player_id: str,
    metrics: list[str] | None = None,
    time_range_s: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-player metrics over an optional time window.

    Available metrics (default = all):
    - distance_m: total ground-plane distance covered.
    - max_speed_mps: peak instantaneous speed (m/s).
    - avg_speed_mps: average speed across observed frames.
    - touches: count of possession/pass/shot/carry events involving the player.
    - passes: number of pass events the player initiated (success+fail).
    - successful_passes: passes with metadata.outcome != 'failed'.
    - shots: number of shot events.
    - goals: number of goal events.
    """
    wanted = set(metrics) if metrics else {
        "distance_m", "max_speed_mps", "avg_speed_mps",
        "touches", "passes", "successful_passes", "shots", "goals",
    }

    fps = ctx.fps or 30.0
    lo_f, hi_f = None, None
    if time_range_s:
        lo_f = int(time_range_s[0] * fps)
        hi_f = int(time_range_s[1] * fps)

    # --- Distance / speed from player_tracks ---
    positions: list[tuple[int, float, float]] = []
    for f_str, items in ctx.player_tracks.items():
        try:
            f = int(f_str)
        except ValueError:
            continue
        if lo_f is not None and not (lo_f <= f <= hi_f):
            continue
        for t in items:
            if t.get("track_id") != player_id:
                continue
            pp = t.get("pitch_position") or [None, None]
            if pp[0] is None:
                continue
            positions.append((f, float(pp[0]), float(pp[1])))
    positions.sort()

    # Reject pairs that imply impossible speeds. A track that drops
    # under occlusion and re-associates to a different on-pitch
    # location can teleport within a single sample step, otherwise
    # producing 100s-1000s m/s garbage. Sprinting tops out near 10
    # m/s, so anything above 12 is structural (re-association /
    # calibration jitter), not motion.
    max_gap_s = 1.0
    max_realistic_mps = 12.0
    distance_m = 0.0
    max_speed = 0.0
    speeds: list[float] = []
    for (f1, x1, y1), (f2, x2, y2) in zip(positions, positions[1:]):
        dt = (f2 - f1) / fps
        if dt <= 0 or dt > max_gap_s:
            continue
        d = math.hypot(x2 - x1, y2 - y1)
        speed = d / dt
        if speed > max_realistic_mps:
            continue
        distance_m += d
        speeds.append(speed)
        max_speed = max(max_speed, speed)
    avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0

    # --- Event-derived counters ---
    touches = passes = successful_passes = shots = goals = 0
    for e in ctx.events:
        if e.get("player_id") != player_id:
            continue
        t = _frame_to_time(e.get("frame", 0), fps)
        if not _in_time_range(t, time_range_s):
            continue
        et = e.get("type")
        if et in {"possession", "pass", "shot", "carry"}:
            touches += 1
        if et == "pass":
            passes += 1
            if (e.get("metadata") or {}).get("outcome") != "failed":
                successful_passes += 1
        elif et == "shot":
            shots += 1
        elif et == "goal":
            goals += 1

    all_metrics = {
        "distance_m": round(distance_m, 2),
        "max_speed_mps": round(max_speed, 2),
        "avg_speed_mps": round(avg_speed, 2),
        "touches": touches,
        "passes": passes,
        "successful_passes": successful_passes,
        "shots": shots,
        "goals": goals,
    }

    return {
        "player_id": player_id,
        "team_id": ctx.team_assignments.get(player_id, "unknown"),
        "observed_frames": len(positions),
        "metrics": {k: v for k, v in all_metrics.items() if k in wanted},
    }


def get_team_stats(
    ctx: MatchContext,
    *,
    team_id: str,
    metrics: list[str] | None = None,
    time_range_s: list[float] | None = None,
) -> dict[str, Any]:
    """Aggregate per-team metrics over an optional time window.

    Available metrics:
    - possession_pct: fraction of total possession-event frames held by team.
    - passes / successful_passes / pass_success_rate
    - shots / goals
    - tackles / interceptions
    """
    wanted = set(metrics) if metrics else {
        "possession_pct", "passes", "successful_passes",
        "pass_success_rate", "shots", "goals",
        "tackles", "interceptions",
    }

    fps = ctx.fps or 30.0

    poss_frames_by_team: Counter = Counter()
    passes = successful_passes = shots = goals = tackles = interceptions = 0

    for e in ctx.events:
        t = _frame_to_time(e.get("frame", 0), fps)
        if not _in_time_range(t, time_range_s):
            continue
        et = e.get("type")
        e_team = e.get("team_id")

        if et == "possession":
            sf = int(e.get("start_frame") or e.get("frame") or 0)
            ef = int(e.get("end_frame") or sf)
            poss_frames_by_team[e_team] += max(1, ef - sf + 1)
            continue

        if e_team != team_id:
            continue
        if et == "pass":
            passes += 1
            if (e.get("metadata") or {}).get("outcome") != "failed":
                successful_passes += 1
        elif et == "shot":
            shots += 1
        elif et == "goal":
            goals += 1
        elif et == "tackle":
            tackles += 1
        elif et == "interception":
            interceptions += 1

    total_poss = sum(poss_frames_by_team.values())
    possession_pct = (
        round(100.0 * poss_frames_by_team.get(team_id, 0) / total_poss, 1)
        if total_poss else 0.0
    )
    pass_success_rate = (
        round(100.0 * successful_passes / passes, 1) if passes else 0.0
    )

    all_metrics = {
        "possession_pct": possession_pct,
        "passes": passes,
        "successful_passes": successful_passes,
        "pass_success_rate": pass_success_rate,
        "shots": shots,
        "goals": goals,
        "tackles": tackles,
        "interceptions": interceptions,
    }

    return {
        "team_id": team_id,
        "metrics": {k: v for k, v in all_metrics.items() if k in wanted},
    }


def get_frame_snapshot(
    ctx: MatchContext,
    *,
    time_s: float,
    event_window_s: float = 5.0,
) -> dict[str, Any]:
    """What was happening at a specific moment.

    Returns the players visible, the ball state, and any events within
    ±event_window_s of *time_s*. Use this when the user pauses on a
    moment ("what just happened?", "who is in the box right now?").
    """
    fps = ctx.fps or 30.0
    frame = int(round(time_s * fps))

    pk = _nearest_frame_key(ctx.player_tracks, frame)
    players = []
    for t in (ctx.player_tracks.get(pk, []) if pk else []):
        pp = t.get("pitch_position") or [None, None]
        players.append({
            "player_id": t.get("track_id"),
            "team": t.get("team"),
            "role": t.get("role"),
            "jersey": t.get("jersey_number"),
            "pitch": [round(v, 2) if v is not None else None for v in pp],
        })

    bk = _nearest_frame_key(ctx.ball_tracks, frame)
    b = ctx.ball_tracks.get(bk) if bk else None
    ball = None
    if b:
        pp = b.get("pitch_position") or [None, None]
        ball = {
            "pitch": [round(v, 2) if v is not None else None for v in pp],
            "height_m": round(b.get("height") or 0.0, 2),
            "on_ground": b.get("on_ground"),
        }

    w = max(1, int(event_window_s * fps))
    lo, hi = frame - w, frame + w
    nearby = [
        _summarise_event(e, fps)
        for e in ctx.events
        if lo <= e.get("frame", -1) <= hi
    ]

    return {
        "time_s": round(time_s, 2),
        "frame": frame,
        "players": players,
        "ball": ball,
        "nearby_events": nearby,
    }


# Stdout cap fed back to Claude. Long stdout would balloon tool_result
# token count; if a script genuinely needs to surface more, it should
# write to a file in the output dir and Claude can readFiles via a
# follow-up code block.
MAX_STDOUT_CHARS = 4000


_ARTIFACT_PRESIGN_TTL_S = 3600
_S3_ARTIFACT_RE = "s3://"


def _upload_and_presign(
    local: Path, bucket: str, key: str, mime: str,
) -> str:
    """Upload *local* to s3://bucket/key and return a presigned GET URL.

    Imported lazily so the local FastAPI flow doesn't need boto3 changes.
    """
    import boto3  # noqa: PLC0415
    s3 = boto3.client("s3")
    s3.upload_file(
        str(local), bucket, key,
        ExtraArgs={"ContentType": mime},
    )
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=_ARTIFACT_PRESIGN_TTL_S,
    )


def run_python(
    ctx: MatchContext,
    *,
    code: str,
    sandbox: CodeSandbox,
    artifact_dir: Path,
    artifact_url_prefix: str,
) -> dict[str, Any]:
    """Run *code* in the AgentCore sandbox and surface stdout + image artifacts.

    Match data files are pre-uploaded to ``./data/`` inside the
    sandbox; new image files written to ``./output/`` are pulled back
    out and saved under ``artifact_dir``. The returned ``artifacts``
    list carries URLs the frontend can render directly.

    URL handling:
    - If *artifact_url_prefix* starts with ``s3://``, treat it as
      ``s3://<bucket>/<prefix>`` and upload each artifact there, then
      return a presigned GET URL (1h TTL). Used in the Runtime
      container, where there is no shared static directory.
    - Otherwise, treat it as a URL prefix the FastAPI app already
      mounts on disk (the local-chat path) and return ``<prefix>/<file>``.
    """
    result = sandbox.run(code, artifact_dir)

    stdout = result.get("stdout", "") or ""
    stderr = result.get("stderr", "") or ""
    truncated = False
    if len(stdout) > MAX_STDOUT_CHARS:
        stdout = stdout[:MAX_STDOUT_CHARS] + "\n... [stdout truncated]"
        truncated = True
    if len(stderr) > MAX_STDOUT_CHARS:
        stderr = stderr[:MAX_STDOUT_CHARS] + "\n... [stderr truncated]"

    use_s3 = artifact_url_prefix.startswith(_S3_ARTIFACT_RE)
    if use_s3:
        # s3://bucket/prefix -> ('bucket', 'prefix')
        rest = artifact_url_prefix[len(_S3_ARTIFACT_RE):].strip("/")
        bucket, _, key_prefix = rest.partition("/")
    else:
        bucket = key_prefix = ""

    artifacts_out = []
    for a in result.get("artifacts") or []:
        local = artifact_dir / a["path"]
        if use_s3 and local.exists():
            key = f"{key_prefix.rstrip('/')}/{a['path']}".lstrip("/")
            url = _upload_and_presign(local, bucket, key, a["mime_type"])
        else:
            url = f"{artifact_url_prefix.rstrip('/')}/{a['path']}"
        # Deliberately drop the bare filename — Claude tends to cite
        # the relative `output/<file>.png` path it just saved to even
        # when the schema explicitly tells it to use `url`. By making
        # `url` the only string identifying the artifact, there's
        # nothing else for it to copy.
        artifacts_out.append({
            "url": url,
            "mime_type": a["mime_type"],
            "embed_as": f"![{a['path'].rsplit('.', 1)[0]}]({url})",
        })

    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": result.get("exit_code"),
        "execution_time_s": result.get("execution_time_s"),
        "artifacts": artifacts_out,
        "stdout_truncated": truncated,
    }


# ---------------------------------------------------------------------------
# Bedrock / Anthropic tool specs
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "list_events",
        "description": (
            "List match events filtered by type, team, player, or time "
            "range. Use to answer 'what passes/shots/goals/tackles/etc "
            "happened in window X' style questions. Possession events "
            "describe who held the ball — use sparingly, they're "
            "numerous. Returns events sorted as stored (chronological)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["possession", "pass", "shot", "goal",
                                 "carry", "tackle", "interception"],
                    },
                    "description": "Restrict to these event types. Omit for all.",
                },
                "team_id": {
                    "type": "string",
                    "description": "Restrict to events by this team (e.g. 'team_A').",
                },
                "player_id": {
                    "type": "string",
                    "description": "Restrict to events by this player (e.g. 'A-9').",
                },
                "time_range_s": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "[start_s, end_s] window in seconds.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Max events to return. Default {DEFAULT_LIMIT}, "
                        f"hard cap {MAX_LIMIT}."
                    ),
                },
            },
        },
    },
    {
        "name": "get_player_stats",
        "description": (
            "Aggregate metrics for one player. Distance and speed come "
            "from tracking; touches/passes/shots/goals come from events. "
            "Use for 'how far did A-9 run', 'how many passes did B-10 "
            "complete', etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_id": {
                    "type": "string",
                    "description": "e.g. 'A-9' or 'B-10'.",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "distance_m", "max_speed_mps", "avg_speed_mps",
                            "touches", "passes", "successful_passes",
                            "shots", "goals",
                        ],
                    },
                    "description": "Specific metrics to compute. Omit for all.",
                },
                "time_range_s": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["player_id"],
        },
    },
    {
        "name": "get_team_stats",
        "description": (
            "Aggregate metrics for one team: possession share, total "
            "passes/shots/goals/tackles. Possession is computed across "
            "all teams over the window so percentages sum to 100."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_id": {
                    "type": "string",
                    "description": "e.g. 'team_A'.",
                },
                "metrics": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "possession_pct", "passes", "successful_passes",
                            "pass_success_rate", "shots", "goals",
                            "tackles", "interceptions",
                        ],
                    },
                },
                "time_range_s": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                },
            },
            "required": ["team_id"],
        },
    },
    {
        "name": "get_frame_snapshot",
        "description": (
            "Snapshot of the match at a specific moment: visible players "
            "with positions, ball state, and events within a small "
            "window. Use when the user is paused at a moment and asks "
            "what's happening or who's involved."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_s": {
                    "type": "number",
                    "description": "Time in seconds from the start of the video.",
                },
                "event_window_s": {
                    "type": "number",
                    "description": "± window for nearby events. Default 5.",
                },
            },
            "required": ["time_s"],
        },
    },
    {
        "name": "run_python",
        "description": (
            "Execute a Python snippet in a managed sandbox to compute "
            "ad-hoc statistics or generate plots. Use this when the "
            "other tools cannot answer (e.g. heatmaps, distributions, "
            "custom aggregations). Match data is pre-uploaded:\n"
            f"  {SANDBOX_DATA_DIR}/events.json — list of events\n"
            f"  {SANDBOX_DATA_DIR}/tracks.json — frame -> [player tracks]\n"
            f"  {SANDBOX_DATA_DIR}/ball_tracks.json — frame -> ball state\n"
            f"  {SANDBOX_DATA_DIR}/team_assignments.json — player_id -> team_id\n"
            f"Save plots as PNG to {SANDBOX_OUTPUT_DIR}/<name>.png — they "
            "will be returned to the user automatically. matplotlib, "
            "numpy, pandas are pre-installed; use matplotlib.use('Agg') "
            "before importing pyplot. Print key numbers via stdout. "
            "Files in /var/data are not accessible — use relative paths "
            f"like '{SANDBOX_DATA_DIR}/events.json'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
        },
    },
]


TOOL_DISPATCH: dict[str, Callable[..., dict[str, Any]]] = {
    "list_events": list_events,
    "get_player_stats": get_player_stats,
    "get_team_stats": get_team_stats,
    "get_frame_snapshot": get_frame_snapshot,
    "run_python": run_python,
}
