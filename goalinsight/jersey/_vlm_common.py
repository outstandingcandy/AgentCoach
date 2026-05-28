"""Shared prompt strings and JSON parsers used by every VLM backend.

All three backends (Claude via Bedrock, Gemini via Google AI Studio,
Qwen via local vLLM server) speak the same structured protocol:

- :func:`build_multi_user_text` – per-track jersey + role prompt
- :const:`MULTI_SYSTEM` – system prompt for per-track recognition
- :func:`parse_multi_response` – parser for the per-track JSON reply
- :func:`parse_scene_response` – parser for the scene-understanding
  JSON reply (``describe_scene``)
- :func:`parse_team_seeds_response` – parser for ``recognize_team_seeds``
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

try:
    import cv2
    import numpy as np
except ImportError:  # pragma: no cover — cv2 is a hard project dep
    cv2 = None
    np = None

logger = logging.getLogger(__name__)


def burn_id_banner(
    crop: "np.ndarray",
    tid: int,
    banner_height: int = 28,
    font_scale: float = 0.7,
    font_thickness: int = 2,
) -> "np.ndarray":
    """Return a copy of *crop* with a white banner at the top printed
    with ``#<tid>``.

    This helps VLMs bind the person-id to the image without relying on
    cross-modal attention between a preceding text line and the image.
    Text is black on white so it survives JPEG compression.
    """
    if cv2 is None or np is None or crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    bh = max(banner_height, int(w * 0.18))  # scale a bit with width
    out = np.empty((h + bh, w, 3), dtype=crop.dtype)
    out[:bh, :, :] = 255  # white banner
    out[bh:, :, :] = crop
    label = f"#{tid}"
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness,
    )
    # Auto-shrink font if it would overflow
    if tw > w - 8:
        shrink = (w - 8) / max(tw, 1)
        font_scale *= shrink
        font_thickness = max(1, int(font_thickness * shrink))
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness,
        )
    x = (w - tw) // 2
    y = (bh + th) // 2
    cv2.putText(
        out, label, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness,
        cv2.LINE_AA,
    )
    return out


SINGLE_PROMPT = (
    "What jersey number is visible on this soccer player? "
    "Reply with ONLY a number 1-99, or 'none' if no number is visible."
)

MULTI_SYSTEM = (
    "You are an expert at analysing soccer broadcast footage. You will "
    "receive multiple crops of the same detected person across different "
    "frames, plus the person's physical location on the pitch.\n\n"
    "CRITICAL: the position label you are given is computed from the "
    "camera calibration and the person's median ground-plane coordinates. "
    "It is a GROUND TRUTH measurement of where they physically are, not "
    "an estimate. You must treat it as a fact and must NOT override it "
    "based on what the crops 'look like'. Crops are 2D projections and "
    "can be misleading (e.g. someone on the pitch near the touchline "
    "can look like they are on the sideline because the camera angle "
    "compresses depth). Trust the label.\n\n"
    "Your job: decide who this person is (player / coach / referee / "
    "linesman / other) and, if a player, read their jersey number — "
    "while strictly obeying the position label's constraints on which "
    "roles are possible."
)

_MULTI_USER_TEMPLATE = (
    "\n\nAbove are {N} crops of the SAME person across different frames.\n\n"
    "=== Physical location ===\n"
    "This person's median pitch position across these frames: {POS}\n\n"
    "Each label means a PHYSICAL location:\n"
    "  - outside  = OUTSIDE the pitch boundary (technical area / bench / "
    "beyond the touchline or goal line). Substitutes, coaches, most "
    "spectators, and linesmen running behind the touchline sit here.\n"
    "  - sideline = INSIDE the pitch, very close to a touchline.\n"
    "  - near_left_goal / near_right_goal = INSIDE the pitch, in or "
    "near a penalty area.\n"
    "  - midfield = INSIDE the pitch, central area.\n\n"
    "=== Movement pattern across the match ===\n"
    "{MOVEMENT}\n\n"
    "Movement-pattern semantics:\n"
    "  - tight_near_goal     : the person stays in a small box right next "
    "to one goal line. EXCLUSIVE TO GOALKEEPERS — no outfield player, "
    "no referee, no linesman moves like this.\n"
    "  - covers_whole_pitch  : the person roams across most of the pitch's "
    "length AND most of its width. Strongly suggests the MAIN REFEREE, "
    "BUT can also occur for short or noisy tracks of outfield players "
    "whose few samples happen to span far. Use this signal only to "
    "PROMOTE a kit-mismatched on-pitch person to referee — never to "
    "OVERRIDE a clear team-kit match.\n"
    "  - touchline_runner    : the person hugs ONE touchline. Suggests a "
    "LINESMAN, but a winger / full-back can also produce this pattern. "
    "Same caveat: use it to promote a kit-mismatched on-pitch person, "
    "never to override a clear team-kit match.\n"
    "  - outfield            : moderate-range motion within the field of "
    "play. Normal outfield player.\n"
    "  - off_pitch_lateral   : motion confined off the pitch. Coach / "
    "substitute / staff / spectator.\n\n"
    "=== STEP-BY-STEP DECISION PROCEDURE (follow exactly) ===\n\n"
    "STEP 1 — Describe what you actually see:\n"
    "  Write a short factual description of what colour/pattern the "
    "target person is wearing. Look at the torso area. Be honest — "
    "do NOT pretend a kit matches when it doesn't.\n\n"
    "STEP 2 — Pitch membership:\n"
    "  on_pitch = (position in {{'midfield', 'near_left_goal', "
    "'near_right_goal', 'sideline'}})\n"
    "  off_pitch = (position == 'outside')\n"
    "  Position is ground-truth from camera calibration; trust it.\n\n"
    "STEP 3 — Role decision. Apply these rules IN ORDER; first match "
    "wins:\n"
    "  3a) movement_pattern == 'tight_near_goal'  → role='goalkeeper', "
    "team='unknown'. A GK kit is usually different from both outfield "
    "kits (green / orange / bright yellow). The movement pattern is "
    "decisive here — do NOT force a GK into team_A or team_B by colour.\n"
    "  3b) on_pitch AND target colour clearly matches team_A_kit → "
    "role='player', team='team_A'. (Even if movement pattern is "
    "'covers_whole_pitch' — short tracks can produce that pattern "
    "spuriously, and a clear team-kit match outweighs movement.)\n"
    "  3c) on_pitch AND target colour clearly matches team_B_kit → "
    "role='player', team='team_B'. (Same caveat as 3b.)\n"
    "  3d) on_pitch AND target colour matches NEITHER team kit AND "
    "movement_pattern == 'covers_whole_pitch' → role='referee', "
    "team='referee'.\n"
    "  3e) on_pitch AND target colour matches NEITHER team kit AND "
    "movement_pattern == 'touchline_runner' → role='linesman', "
    "team='referee'.\n"
    "  3f) on_pitch AND target colour matches NEITHER team kit "
    "(fallback) → role='referee', team='referee'. "
    "BUT: if the crops are too blurry / dark / occluded to read "
    "the kit at all, do NOT default to referee — return "
    "role='unknown', team='unknown' so the track is treated as "
    "an unidentifiable player rather than polluting the referee "
    "cluster.\n"
    "  3g) off_pitch → role='coach' if dressed in suit/jacket, else "
    "'other'. Use 'linesman' only if movement was 'touchline_runner'.\n\n"
    "STEP 4 — Jersey number:\n"
    "  If role != 'player', jersey_number = null.\n"
    "  Otherwise read the 1-99 number on the chest/back of the "
    "jersey. Ignore numbers on shorts/socks. Return null if not "
    "readable. NEVER invent a number.\n\n"
    "STEP 5 — Team for goalkeeper:\n"
    "  If role='goalkeeper', set team='unknown' (the post-processor "
    "decides from goal side). Do NOT force team_A/team_B by colour.\n\n"
    "STEP 6 — Confidence:\n"
    "  confidence ∈ [0, 1] reflects YOUR CERTAINTY. When the movement "
    "pattern matches a role rule (3a/3b/3c) you can use high "
    "confidence (≥0.8). When colour and movement disagree, lower it.\n\n"
    "=== Output format ===\n"
    "Respond with ONLY a single JSON object on one line, no prose, no "
    "markdown:\n"
    '{{"target_kit_description": "<short colour/pattern of what '
    'you actually see>", '
    '"pitch_membership": "on_pitch"|"off_pitch", '
    '"role": "player"|"goalkeeper"|"coach"|"referee"|"linesman"|"other"|"unknown", '
    '"team": "team_A"|"team_B"|"referee"|"none"|"unknown", '
    '"jersey_number": <int 1-99 or null>, '
    '"confidence": <float 0.0-1.0>, '
    '"reasoning": "<one short sentence>"}}\n\n'
    "team rules:\n"
    "  - role=player              → team is team_A or team_B (whichever "
    "kit matches).\n"
    "  - role=goalkeeper          → team='unknown'.\n"
    "  - role=referee or linesman → team='referee'.\n"
    "  - role=coach / other       → team='none'."
)


SCENE_TASK_TEXT = (
    "\n=== Task ===\n"
    "Each person crop has the person's numeric id printed on a white "
    "banner at the TOP of the image (e.g. '#4', '#362'). Read that "
    "banner to identify the person — do NOT rely on the surrounding "
    "text labels to bind ids to crops.\n\n"
    "You are studying a single soccer match. Identify the TWO "
    "OUTFIELD PLAYER KITS — i.e. what the two opposing teams wear. "
    "DO NOT try to identify the referee, goalkeepers, linesmen, or "
    "coaches in this task; they are handled separately.\n\n"
    "Use BOTH the wide frame (which shows the pitch context) AND "
    "the individual person crops to answer:\n\n"
    "1. team_A_kit: jersey colour, shorts colour, any distinguishing "
    "features. Pick whichever team's outfield kit you call team_A.\n"
    "2. team_B_kit: the OTHER team's outfield kit.\n"
    "3. team_A_ref_ids: 3 person ids (read from the white banners) "
    "that clearly wear team_A's outfield kit.\n"
    "4. team_B_ref_ids: 3 person ids that clearly wear team_B's kit.\n\n"
    "IMPORTANT — exclusion rules:\n"
    "  - DO NOT pick a goalkeeper for team_A_ref_ids or team_B_ref_ids. "
    "Goalkeepers wear a different kit from their teammates and would "
    "mislead the colour reference. If a person is standing alone right "
    "in front of a goal, they are probably a goalkeeper — skip them.\n"
    "  - DO NOT pick a referee or linesman.\n"
    "  - Only pick people who are clearly in OUTFIELD play (multiple "
    "teammates around them in the same kit).\n\n"
    "The labels team_A / team_B are arbitrary — pick whichever kit "
    "is team_A, just be consistent across the four fields.\n\n"
    "Respond with ONLY a single JSON object, no prose, no markdown:\n"
    '{"team_A_kit": "<1-sentence description>",\n'
    ' "team_B_kit": "<1-sentence description>",\n'
    ' "team_A_ref_ids": [<int>, <int>, <int>],\n'
    ' "team_B_ref_ids": [<int>, <int>, <int>]}'
)


TEAM_SEEDS_INTRO = (
    "You will see numbered crops of different people from a "
    "single soccer match. Each person is shown in 1 or more "
    "frames — use ALL their images together before deciding "
    "their team. Your job: group them by match kit. "
    "There are exactly TWO opposing teams on the pitch plus "
    "a small number of officials (referee, linesmen) and "
    "possibly staff/coaches/spectators."
)


TEAM_SEEDS_TASK_TEXT = (
    "\nGroup every person above into one of these labels:\n"
    "  - team_A: one of the two match teams (use whichever "
    "colour you like for team_A).\n"
    "  - team_B: the OTHER match team.\n"
    "  - official: main referee or assistant referee (distinct "
    "officiating kit, usually different from both team kits; "
    "often dark or a bright contrasting colour).\n"
    "  - other: anyone not on the pitch (coach, substitute, "
    "spectator, staff).\n\n"
    "Both team kits have distinct colours — choose team_A / "
    "team_B assignments so that all people of the same match "
    "kit get the same label. The two team labels are arbitrary "
    "(team_A can be the red team or the yellow team, your choice).\n\n"
    "Respond with ONLY a single JSON array, no prose, no markdown:\n"
    '[{"id": <int>, "team": "team_A"|"team_B"|"official"|"other", '
    '"kit_description": "<short phrase like red jersey blue shorts>"}, ...]'
)


VALID_ROLES = {
    "player", "goalkeeper", "coach", "referee", "linesman",
    "other", "unknown",
}


def build_multi_user_text(
    n_crops: int,
    position_label: str,
    movement_description: str = "",
) -> str:
    """Render the per-track user prompt with placeholders filled in."""
    movement = movement_description.strip() or (
        "(no movement statistics available — fall back to colour and "
        "position cues only)"
    )
    return _MULTI_USER_TEMPLATE.format(
        N=n_crops, POS=position_label, MOVEMENT=movement,
    )


def build_scene_context_block(scene_description: dict[str, str] | None) -> str | None:
    """Format the Step-1 scene description as a text block to prepend
    to per-track messages.  Returns ``None`` when the dict is empty."""
    if not scene_description:
        return None
    parts = []
    if scene_description.get("team_A_kit"):
        parts.append(f"team_A kit: {scene_description['team_A_kit']}")
    if scene_description.get("team_B_kit"):
        parts.append(f"team_B kit: {scene_description['team_B_kit']}")
    if not parts:
        return None
    return (
        "=== Match context (pre-computed by "
        "scene-understanding pass) ===\n"
        + "\n".join(parts)
    )


def parse_single_response(response: str) -> int | None:
    """Parse the single-image 'what number?' reply."""
    response = response.strip().lower()
    no_number_phrases = (
        "no number", "not visible", "cannot see", "unclear",
        "no jersey", "can't see", "unable", "none", "n/a",
    )
    for phrase in no_number_phrases:
        if phrase in response:
            return None
    m = re.findall(r"\b(\d{1,2})\b", response)
    for s in m:
        n = int(s)
        if 1 <= n <= 99:
            return n
    return None


_VALID_TEAMS = {"team_A", "team_B", "referee", "none", "unknown"}


def parse_multi_response(
    raw: str,
) -> tuple[int | None, float, str, str, str]:
    """Parse ``recognize_multi``'s JSON reply into ``(num, conf,
    reasoning, role, team)``.

    The ``target_kit_description`` field (when emitted) is folded into
    ``reasoning`` so it survives downstream auditing.
    """
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return (None, 0.0, f"no json: {text[:80]}", "unknown", "unknown")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        return (None, 0.0, f"bad json: {exc}", "unknown", "unknown")
    num_raw = obj.get("jersey_number")
    num: int | None
    if num_raw in (None, "none", "null"):
        num = None
    else:
        try:
            n = int(num_raw)
            num = n if 1 <= n <= 99 else None
        except (TypeError, ValueError):
            num = None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    reasoning = str(obj.get("reasoning", "") or "")[:200]
    target_desc = str(obj.get("target_kit_description", "") or "")[:120]
    if target_desc:
        # Prepend the model's own colour description to the reasoning
        # so jersey_votes.json captures both signals.
        reasoning = f"[saw: {target_desc}] {reasoning}"[:300]
    role = str(obj.get("role", "unknown") or "unknown").strip().lower()
    if role not in VALID_ROLES:
        role = "unknown"
    # Guard against the "looks unreadable so I'll default to referee"
    # failure mode — when the kit description itself says the crops
    # are unreadable, the model has no positive evidence for any role.
    # Treat such tracks as unknown so they go through orphan absorption
    # instead of polluting the referee/linesman pools.
    blur_phrases = (
        "too blurry", "cannot discern", "cannot determine", "unreadable",
        "obscured", "not visible", "no visible", "unable to identify",
        "too dark to", "indistinguishable",
    )
    desc_lower = target_desc.lower() if target_desc else ""
    if role in ("referee", "linesman") and any(
        p in desc_lower for p in blur_phrases
    ):
        role = "unknown"
        team = "unknown"
        conf = min(conf, 0.2)
        return (None, conf, reasoning, role, team)
    team = str(obj.get("team", "unknown") or "unknown").strip()
    if team.lower() in {"team_a", "teama", "a"}:
        team = "team_A"
    elif team.lower() in {"team_b", "teamb", "b"}:
        team = "team_B"
    elif team.lower() in {"referee", "official"}:
        team = "referee"
    elif team.lower() in {"none", ""}:
        team = "none"
    else:
        team = "unknown"
    if num is None and role == "player":
        conf = min(conf, 0.0)
    return (num, conf, reasoning, role, team)


def parse_scene_response(raw: str) -> dict[str, Any]:
    """Parse ``describe_scene``'s JSON reply."""
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        logger.warning("describe_scene: no JSON: %s", text[:300])
        return {}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("describe_scene: bad JSON: %s", exc)
        return {}

    def _int_list(v) -> list[int]:
        return [
            int(x) for x in (v or [])
            if isinstance(x, (int, str)) and str(x).lstrip("-").isdigit()
        ]

    return {
        "team_A_kit": str(obj.get("team_A_kit", "")),
        "team_B_kit": str(obj.get("team_B_kit", "")),
        "team_A_ref_ids": _int_list(obj.get("team_A_ref_ids")),
        "team_B_ref_ids": _int_list(obj.get("team_B_ref_ids")),
    }


def parse_team_seeds_response(raw: str) -> dict[int, str]:
    """Parse ``recognize_team_seeds``'s JSON array reply."""
    text = raw.strip()
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        logger.warning("team_seeds: no JSON array: %s", text[:200])
        return {}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        logger.warning("team_seeds: bad JSON: %s", exc)
        return {}
    out: dict[int, str] = {}
    valid = {"team_A", "team_B", "official", "other"}
    for obj in arr:
        try:
            tid = int(obj["id"])
            team = str(obj["team"])
        except (TypeError, KeyError, ValueError):
            continue
        if team in valid:
            out[tid] = team
    return out
