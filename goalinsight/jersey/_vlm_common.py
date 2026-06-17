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
import math
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

# Per-crop OCR — strict reading rules + structured output. Used when
# ``recognize_multi`` fans out one LLM request per crop in parallel.
# The prompt forbids guessing the tens place from a single visible
# digit (e.g. seeing only "0" must NOT be reported as "10" / "20").
SINGLE_OCR_PROMPT = (
    "Read the jersey number on this soccer player's back/chest.\n\n"
    "RULES:\n"
    "  - The crop is supposed to contain ONE target player. If the crop\n"
    "    shows MULTIPLE players overlapping (two or more visible jersey\n"
    "    backs / faces close together so it's ambiguous which one is\n"
    "    the target), return reading=null with\n"
    "    visible_digits='multiple players in crop' — DO NOT pick any\n"
    "    number even if one is clearly readable, because we don't know\n"
    "    which player you'd be reading.\n"
    "  - If you can FULLY see the number's digits, sharp and unobstructed,\n"
    "    return that number (1-99).\n"
    "  - If you can only see ONE digit clearly (e.g. a '0' but the tens\n"
    "    place is occluded by an arm / by another player / by a sock),\n"
    "    return reading=null. DO NOT guess that it's 10 / 20 / 30 / etc.\n"
    "    Instead set visible_digits='partial: <digit>' so the caller knows.\n"
    "  - Ignore numbers on shorts, socks, or in the background.\n"
    "  - If the back is not visible at all (player facing camera /\n"
    "    occluded), return reading=null with visible_digits='back not\n"
    "    visible'.\n\n"
    "Output a single JSON object on one line, no prose:\n"
    '{"reading": <int 1-99 or null>, '
    '"visible_digits": "<short factual>", '
    '"crop_confidence": <float 0.0-1.0>}\n\n'
    "crop_confidence: 0 if reading=null; 0.85+ when digits are sharp,\n"
    "contiguous, and unobstructed; lower for partial / blurry."
)


# Same rules as SINGLE_OCR_PROMPT but applied to a montage of N crops.
# The N crops are all of the SAME player at different moments — the
# model votes across them and returns ONE final number, not per-cell
# verdicts. Cuts both API count and output tokens compared to per-
# crop calls. Wording is grid-specific; the image-list path below uses
# its own variant where each crop arrives as its own image block.
BATCHED_OCR_PROMPT = (
    "You will see ONE composite image containing {N} crops of the SAME "
    "soccer player at different moments. They are arranged in a grid "
    "(rows×cols) for compactness; treat them as {N} independent looks "
    "at the same person's back/chest.\n\n"
    "Determine that player's jersey number. CRITICAL: a single visible "
    "digit is NEVER a number — most jerseys have two digits, and a "
    "single visible digit usually means the OTHER digit is occluded by "
    "an arm, shoulder, sleeve, side-of-body, or simply not facing the "
    "camera. For example, if you see a '6' but no '1' immediately to "
    "its left/right within the SAME crop, the number could be 6, 16, "
    "26, 36, 46, 56, 60–69, 76, 86, or 96 — DO NOT commit. Only commit "
    "to a single-digit number (1–9) when you can see in ≥1 crop the "
    "ENTIRE jersey-back area unobstructed AND that area shows ONE "
    "centred digit with empty fabric on both sides where a tens digit "
    "would otherwise sit.\n\n"
    "RULES:\n"
    "  - The composite image is a GRID of independent crops separated by\n"
    "    cell borders. Each cell is meant to contain ONE target player.\n"
    "    The grid as a whole obviously shows many cells with different\n"
    "    moments — that's expected. The problem case is when a SINGLE\n"
    "    CELL contains the target plus ANOTHER player overlapping the\n"
    "    target's body (e.g. another player's back / shoulder / number\n"
    "    visibly intruding into the same cell as the target).\n"
    "  - ABSOLUTE RULE for multi-player cells: if the number you would\n"
    "    read appears ONLY in cells that also contain another player\n"
    "    overlapping the target, you must NOT report that number — even\n"
    "    if the digits are sharp and easy to read. The number could\n"
    "    belong to either player and we cannot tell which is the target.\n"
    "    Set reading=null with\n"
    "    visible_digits='multiple players in crop'.\n"
    "  - To commit to a number, you need ≥1 cell where (a) exactly one\n"
    "    player is visible (no second player overlapping their body),\n"
    "    and (b) that player's number is sharp and unobstructed:\n"
    "      • full two-digit number visible → return 10-99\n"
    "      • full back area with a single centred digit (no second-\n"
    "        digit space being hidden) → return that single digit (1-9)\n"
    "        and visible_digits='single digit, full back visible'\n"
    "  - Subtle test: if you see a number in cell X but in cell X you\n"
    "    can also see two backs, two shoulders, or two heads close\n"
    "    enough to be in the same cell — DO NOT use that cell.\n"
    "  - Only ever see a single digit but the rest of the back is\n"
    "    occluded / cropped / turned away → reading=null,\n"
    "    visible_digits='partial: <digit>'. DO NOT guess the missing\n"
    "    digit, even if a player is rumoured to wear that low number.\n"
    "  - Ignore numbers on shorts, socks, or background.\n"
    "  - Back never visible across all single-player cells →\n"
    "    reading=null, visible_digits='back not visible'.\n\n"
    "Output a single JSON object on one line, no prose:\n"
    '{{"reading": <int 1-99 or null>, '
    '"visible_digits": "<short factual>", '
    '"confidence": <float 0.0-1.0>}}\n\n'
    "confidence: 0 if reading=null; 0.9+ when at least one crop shows\n"
    "the full number sharp and unobstructed; 0.4–0.6 when committing\n"
    "to a single digit because the full back is visible; lower when\n"
    "crops disagree."
)


# Image-list variant: N crops arrive as N separate image blocks in the
# user message (no montage tile). The "multi-player overlap" risk is
# significantly lower because each image is just one bbox crop with the
# target centred — adjacent players appear only as the small bbox-edge
# slivers they are. Same JSON output shape as the montage prompt so
# downstream parsing is identical.
BATCHED_OCR_PROMPT_IMAGE_LIST = (
    "You will see {N} separate images, each a crop of the SAME soccer "
    "player at a different moment. Each image was extracted from the "
    "video using a single tracker bounding box and is supposed to "
    "contain ONE target player at its centre. Treat them as {N} "
    "independent looks at the same person's back/chest.\n\n"
    "Determine that player's jersey number. CRITICAL: a single visible "
    "digit is NEVER a number — most jerseys have two digits, and a "
    "single visible digit usually means the OTHER digit is occluded by "
    "an arm, shoulder, sleeve, side-of-body, or simply not facing the "
    "camera. For example, if you see a '6' but no '1' immediately to "
    "its left/right within the SAME image, the number could be 6, 16, "
    "26, 36, 46, 56, 60–69, 76, 86, or 96 — DO NOT commit. Only commit "
    "to a single-digit number (1–9) when ≥1 image shows the ENTIRE "
    "jersey-back area unobstructed AND that area shows ONE centred "
    "digit with empty fabric on both sides where a tens digit would "
    "otherwise sit.\n\n"
    "RULES:\n"
    "  - Each image is supposed to contain ONE target player. The bbox\n"
    "    edges may catch slivers of teammates standing nearby — that's\n"
    "    fine; ignore those slivers and read only the player at the\n"
    "    image centre. The problem case is when an image clearly shows\n"
    "    TWO players' jersey backs both fully visible inside the same\n"
    "    image (overlapping torsos, two numbers visible). For those\n"
    "    images you cannot tell which is the target — ignore them.\n"
    "    If MOST images individually show two-or-more fully visible\n"
    "    jersey backs, return reading=null with\n"
    "    visible_digits='multiple players in crop'.\n"
    "  - To commit to a number, you need ≥1 image where (a) the\n"
    "    centred player's body is the dominant figure, and (b) that\n"
    "    player's number is sharp and unobstructed:\n"
    "      • full two-digit number visible → return 10-99\n"
    "      • full back area with a single centred digit (no second-\n"
    "        digit space being hidden) → return that single digit (1-9)\n"
    "        and visible_digits='single digit, full back visible'\n"
    "  - Only ever see a single digit but the rest of the back is\n"
    "    occluded / cropped / turned away → reading=null,\n"
    "    visible_digits='partial: <digit>'. DO NOT guess the missing\n"
    "    digit, even if a player is rumoured to wear that low number.\n"
    "  - Ignore numbers on shorts, socks, or background.\n"
    "  - Back never visible across all images → reading=null,\n"
    "    visible_digits='back not visible'.\n\n"
    "Output a single JSON object on one line, no prose:\n"
    '{{"reading": <int 1-99 or null>, '
    '"visible_digits": "<short factual>", '
    '"confidence": <float 0.0-1.0>}}\n\n'
    "confidence: 0 if reading=null; 0.9+ when at least one image shows\n"
    "the full number sharp and unobstructed; 0.4–0.6 when committing\n"
    "to a single digit because the full back is visible; lower when\n"
    "images disagree."
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
    "  3a) movement_pattern == 'tight_near_goal'  AND  the target's "
    "kit colour clearly DOES NOT match team_A_kit AND DOES NOT match "
    "team_B_kit  → role='goalkeeper', team='unknown'. A GK kit is "
    "deliberately distinct (green / orange / bright yellow / black) "
    "from both outfield kits — that is a Laws-of-the-Game requirement. "
    "If the target wears the same kit as one of the outfield teams, "
    "they are NOT the goalkeeper, no matter how their movement was "
    "labelled — fall through to 3b/3c.\n"
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
    "  3f) on_pitch AND target colour matches NEITHER team kit AND "
    "the kit looks like a referee uniform (typically all black, dark "
    "navy, dark grey, or with a high-vis green/yellow trim — distinct "
    "from coach jackets / parka tracksuits / casual wear) → "
    "role='referee', team='referee'. Use this ONLY when the kit is "
    "visually convincing as a referee uniform; track length is not "
    "required (the same referee may have been split across many short "
    "tracks).\n"
    "  3g) on_pitch AND target colour matches NEITHER team kit AND "
    "the kit does NOT look like a referee uniform (casual jacket, "
    "tracksuit, hoodie, parka, etc.) → role='other'. These are "
    "coaches, substitutes, ball boys, parents, spectators standing "
    "on or near the playing area. Do NOT default to referee for "
    "non-kit people in casual clothing.\n"
    "  3h) off_pitch → role='coach' if dressed in suit/jacket, else "
    "'other'. Use 'linesman' only if movement was 'touchline_runner'.\n\n"
    "REFEREE GUIDANCE — be visually honest. The main referee wears a "
    "purpose-made referee uniform: typically black or dark navy "
    "shorts + matching jersey, often with high-vis (yellow/green/"
    "fluorescent) accents on shoulders or chest. Coaches and parents "
    "standing on the sideline wear casual clothing — parkas, "
    "tracksuit pants, jeans, hoodies, polo shirts — which looks "
    "different from a referee kit even if both are dark-coloured. "
    "If you see casual / non-uniform dark clothing, return "
    "'other' not 'referee'.\n\n"
    "STEP 4 — Jersey number (CRITICAL: per-crop voting):\n"
    "  If role != 'player', skip this step (set jersey verdict null).\n"
    "  Otherwise, examine EACH of the {N} crops INDIVIDUALLY and emit\n"
    "  a per-crop verdict. RULES YOU MUST FOLLOW:\n"
    "    - 'reading' = the digits you can FULLY see, as a string '0'..'99'\n"
    "      Read what is actually on the back. If you can only see ONE\n"
    "      digit clearly (e.g. a '0' but its tens place is occluded by\n"
    "      an arm / by another player) you MUST set reading=null and\n"
    "      visible_digits to the partial you saw — DO NOT guess that\n"
    "      it is 10 / 20 / 30 / etc. NEVER invent a number.\n"
    "    - 'visible_digits' = a short factual description of what you\n"
    "      actually see on the back, e.g. '20', 'partial: 0', 'blurry',\n"
    "      'back not visible', 'occluded by teammate'.\n"
    "    - 'crop_confidence' ∈ [0,1] = confidence in this single crop's\n"
    "      reading. 0 if you set reading=null. Use 0.9+ only when the\n"
    "      digits are sharp, contiguous, and unobstructed.\n"
    "  This is a JURY VOTE: a single hard-to-read crop must NOT\n"
    "  collapse the whole track to a guessed number. Python code will\n"
    "  aggregate the per-crop readings to pick the winning number.\n\n"
    "STEP 5 — Team for goalkeeper:\n"
    "  If role='goalkeeper', set team='unknown' (the post-processor "
    "decides from goal side). Do NOT force team_A/team_B by colour.\n\n"
    "STEP 6 — Confidence:\n"
    "  confidence ∈ [0, 1] reflects YOUR overall track-level certainty\n"
    "  for the role/team decision (NOT the jersey number — that is\n"
    "  derived from per-crop votes). Use ≥0.8 when movement/kit are\n"
    "  decisive, lower it when colour and movement disagree.\n\n"
    "=== Output format ===\n"
    "Respond with ONLY a single JSON object on one line, no prose, no "
    "markdown:\n"
    '{{"target_kit_description": "<short colour/pattern of what '
    'you actually see>", '
    '"pitch_membership": "on_pitch"|"off_pitch", '
    '"role": "player"|"goalkeeper"|"coach"|"referee"|"linesman"|"other"|"unknown", '
    '"team": "team_A"|"team_B"|"referee"|"none"|"unknown", '
    '"crop_verdicts": [{{"image": 1, "visible_digits": "<short>", '
    '"reading": <int 1-99 or null>, "crop_confidence": <float 0.0-1.0>}}, '
    '... {N} entries in image-order ...], '
    '"confidence": <float 0.0-1.0>, '
    '"reasoning": "<one short sentence>"}}\n\n'
    "team rules:\n"
    "  - role=player              → team is team_A or team_B (whichever "
    "kit matches).\n"
    "  - role=goalkeeper          → team='unknown'.\n"
    "  - role=referee or linesman → team='referee'.\n"
    "  - role=coach / other       → team='none'.\n\n"
    "When role != 'player', emit crop_verdicts as an empty list [] — "
    "the field is only meaningful for players."
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


def parse_batched_ocr_response(
    raw: str,
) -> tuple[int | None, float, str]:
    """Parse BATCHED_OCR_PROMPT's reply into a single track-level verdict.

    The batched prompt now asks the model to vote across the montage
    and return ONE number per call (not per-crop verdicts), so the
    output mirrors :func:`parse_single_ocr_response`:
    ``(reading, confidence, visible_digits)``. Returns
    ``(None, 0.0, "")`` on an unparseable reply.
    """
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, 0.0, ""
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        obj = _try_parse_truncated(m.group(0)) or {}
    if not isinstance(obj, dict):
        return None, 0.0, ""
    reading_raw = obj.get("reading")
    try:
        reading = int(reading_raw) if reading_raw not in (
            None, "", "null", "none") else None
    except (TypeError, ValueError):
        reading = None
    if reading is not None and not (1 <= reading <= 99):
        reading = None
    try:
        conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    if reading is None:
        conf = 0.0
    visible = str(obj.get("visible_digits", "") or "")[:80]
    # Defensive: the prompt forbids committing a single visible digit
    # as a number unless the back is fully visible. The LLM still
    # occasionally violates this (e.g. orig 50 on the kids clip:
    # "single digit '6' visible on lower back" → reading=6, but
    # the actual jersey was #16 with the '1' occluded by an arm).
    # When ``visible_digits`` self-describes as a partial / single-digit
    # observation, demote ``reading`` to null so the track-level vote
    # doesn't latch onto the wrong number.
    if reading is not None:
        v_lo = visible.lower()
        partial_markers = (
            "single digit",
            "partial",
            "one digit",
            "1 digit",
            "obscured",
            "occluded",
            "only the",
            "only see",
            "side", # "visible on side" / "from the side"
        )
        looks_partial = any(m in v_lo for m in partial_markers)
        # ``looks_partial`` should not fire when the description
        # explicitly confirms the full back was visible — let those
        # commits through.
        confirms_full = any(
            phrase in v_lo for phrase in (
                "full number", "two digit", "centred", "centered",
                "back fully", "full back",
            )
        )
        if looks_partial and not confirms_full:
            reading = None
            conf = 0.0
    return reading, conf, visible


def parse_single_ocr_response(
    raw: str,
) -> tuple[int | None, float, str]:
    """Parse SINGLE_OCR_PROMPT's JSON reply into
    ``(reading, crop_confidence, visible_digits)``.

    Robust to truncation and to the model emitting a plain integer or
    free text instead of JSON — when the JSON parse fails we fall back
    to ``parse_single_response`` (number-extraction) with cc=0.5.
    """
    text = raw.strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = _try_parse_truncated(m.group(0))
            if obj is None:
                obj = {}
    else:
        obj = {}
    if obj:
        reading_raw = obj.get("reading")
        try:
            reading = int(reading_raw) if reading_raw not in (
                None, "", "null", "none") else None
        except (TypeError, ValueError):
            reading = None
        if reading is not None and not (1 <= reading <= 99):
            reading = None
        try:
            cc = float(obj.get("crop_confidence", 0.0))
        except (TypeError, ValueError):
            cc = 0.0
        cc = max(0.0, min(1.0, cc))
        if reading is None:
            cc = 0.0
        visible = str(obj.get("visible_digits", "") or "")[:80]
        return reading, cc, visible
    # Fallback: free-text reply.
    n = parse_single_response(text)
    return n, (0.5 if n is not None else 0.0), text[:80]


_VALID_TEAMS = {"team_A", "team_B", "referee", "none", "unknown"}


def _try_parse_truncated(json_text: str) -> dict[str, Any] | None:
    """Best-effort recovery when ``max_tokens`` cut the JSON mid-reply.

    Walks the prefix character by character keeping {...} and "..." nesting
    state; when a parse fails, drops the trailing fragment back to the
    last complete key:value pair, closes the object, and retries.
    Returns the dict on success, ``None`` if nothing parseable remains.
    """
    last_safe_end = -1
    in_str = False
    escape = False
    depth = 0
    for i, ch in enumerate(json_text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 1:
            # We just finished a top-level "key":value pair — safe rewind point.
            last_safe_end = i
    if last_safe_end < 0:
        return None
    candidate = json_text[:last_safe_end] + "}"
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def aggregate_crop_verdicts(
    verdicts: list[dict[str, Any]],
) -> tuple[int | None, float, list[tuple[int, float]]]:
    """Aggregate per-crop ``crop_verdicts`` into a track-level verdict.

    Voting: each crop with a clean ``reading`` casts a vote weighted by
    its ``crop_confidence``. Crops where ``reading`` is null contribute
    nothing (their ``visible_digits`` description is informative for
    debugging but does NOT spawn a guess like '0' → 10/20/30).

    Confidence formula combines two factors so that a single weak vote
    (e.g. 1 crop reads "21" at cc=0.5 while 14 are blank) does NOT
    masquerade as 100% certain:

        evidence_strength = winner_score / N_crops_total   (turnout)
        purity            = winner_score / total_voted     (consensus)
        conf              = sqrt(evidence_strength × purity)

    The geometric mean punishes either factor going to zero. Reference
    points on a 15-crop track:

      | scenario                          | winner | total | N  | conf |
      |-----------------------------------|-------:|------:|---:|-----:|
      | 1 vote, cc=0.5, 14 blanks         |   0.50 |  0.50 | 15 | 0.18 |
      | 6 unanimous votes at cc=0.9       |   5.40 |  5.40 | 15 | 0.60 |
      | 4 votes #18 + 2 votes #13 at 0.9  |   3.60 |  5.40 | 15 | 0.40 |
      | 15 unanimous strong               |  13.50 | 13.50 | 15 | 0.95 |

    Returns ``(winner_number, conf, breakdown)``. ``breakdown`` is a
    list of ``(number, weighted_score)`` sorted desc — runner-up
    readings are surfaced so downstream code can detect tracker ID
    switches (a real same-person track would have a clear winner;
    a mixed-identity track has competing numbers).
    """
    score: dict[int, float] = {}
    total_voted = 0.0
    n_crops_total = 0
    for v in verdicts or []:
        if not isinstance(v, dict):
            continue
        n_crops_total += 1
        reading = v.get("reading")
        if reading in (None, "", "null", "none"):
            continue
        try:
            n = int(reading)
        except (TypeError, ValueError):
            continue
        if not (1 <= n <= 99):
            continue
        try:
            cc = float(v.get("crop_confidence", 0.0))
        except (TypeError, ValueError):
            cc = 0.0
        cc = max(0.0, min(1.0, cc))
        if cc <= 0.0:
            continue
        score[n] = score.get(n, 0.0) + cc
        total_voted += cc
    if not score or n_crops_total == 0 or total_voted <= 0:
        return None, 0.0, []
    breakdown = sorted(score.items(), key=lambda kv: -kv[1])
    winner_n, winner_score = breakdown[0]
    evidence = winner_score / float(n_crops_total)
    purity = winner_score / total_voted
    conf = math.sqrt(max(0.0, evidence * purity))
    conf = max(0.0, min(1.0, conf))
    return winner_n, conf, breakdown


def parse_multi_response(
    raw: str,
) -> tuple[int | None, float, str, str, str, list[tuple[int, float]]]:
    """Parse ``recognize_multi``'s JSON reply into ``(num, conf,
    reasoning, role, team, jersey_breakdown)``.

    The ``target_kit_description`` field (when emitted) is folded into
    ``reasoning`` so it survives downstream auditing.

    ``num`` and the jersey portion of ``conf`` are derived from the
    per-crop ``crop_verdicts`` array via :func:`aggregate_crop_verdicts`.
    Falls back to the legacy top-level ``jersey_number`` field when
    the model didn't emit per-crop verdicts (back-compat with caches
    produced before the prompt changed).

    ``jersey_breakdown`` is a list of ``(number, weighted_score)`` so
    the consolidator can see runner-up candidates and inspect cases
    where ReID and the winning number disagree.
    """
    text = raw.strip()
    # First try to find a complete {…} block. If max_tokens cut the
    # reply mid-string, fall back to the prefix from the first '{' and
    # let _try_parse_truncated stitch a syntactically-valid JSON out of
    # whatever fields fully completed before truncation.
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        json_text = m.group(0)
    else:
        i = text.find("{")
        if i < 0:
            return (None, 0.0, f"no json: {text[:80]}", "unknown", "unknown", [])
        json_text = text[i:]
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        obj = _try_parse_truncated(json_text)
        if obj is None:
            return (None, 0.0, f"bad json: {text[:80]}", "unknown", "unknown", [])

    # New path: aggregate per-crop verdicts.
    crop_verdicts = obj.get("crop_verdicts")
    if isinstance(crop_verdicts, list) and crop_verdicts:
        num, jersey_score, breakdown = aggregate_crop_verdicts(crop_verdicts)
    else:
        # Legacy fallback: read the old top-level fields.
        num_raw = obj.get("jersey_number")
        if num_raw in (None, "none", "null"):
            num = None
        else:
            try:
                n = int(num_raw)
                num = n if 1 <= n <= 99 else None
            except (TypeError, ValueError):
                num = None
        breakdown = [(num, 1.0)] if num is not None else []
        jersey_score = 1.0 if num is not None else 0.0

    try:
        track_conf = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        track_conf = 0.0
    track_conf = max(0.0, min(1.0, track_conf))
    # Effective confidence reported up: when a number was decided,
    # combine the LLM's track-level confidence (role/team certainty)
    # with the jersey vote ratio (how dominant the winner is). Take
    # the min so a low-confidence role decision can't masquerade
    # behind a 100% unanimous but tiny set of jersey votes.
    if num is None:
        conf = 0.0
    else:
        conf = min(track_conf, jersey_score)
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
        return (None, conf, reasoning, role, team, [])
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
    return (num, conf, reasoning, role, team, breakdown)


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
