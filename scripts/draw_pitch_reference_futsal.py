#!/usr/bin/env python3
"""Generate a top-down futsal pitch reference diagram (FIFA Futsal Laws).

Same visual style as ``draw_pitch_reference.py`` but for the 5-a-side
(futsal) pitch. The futsal pitch shape differs from 11-a-side:

- Pitch is 40 m x 20 m (FIFA international: 38-42 x 20-25).
- The "penalty area" is a D-shape: two quarter-circles of radius 6 m
  centered on the goal posts, joined by a straight line segment parallel
  to the goal line (length = goal width = 3 m).
- Goal is 3 m wide x 2 m high.
- First penalty mark: 6 m from goal line.
- Second penalty mark: 10 m from goal line (10-m penalty for 6th team foul).
- Center circle radius: 3 m.
- Substitution zone: 5 m on each side of the halfway line on one touchline.
- Corner arc radius: 0.25 m (drawn but not labeled with a keypoint).

Origin at pitch center, y-up, x pointing toward the right goal.
"""

import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


# ---------- futsal dimensions (FIFA standard, meters) ----------
PITCH_L = 40.0
PITCH_W = 20.0
GOAL_W = 3.0
PA_R = 6.0                  # penalty area arc radius
FIRST_PEN = 6.0             # first penalty mark distance from goal line
SECOND_PEN = 10.0           # second penalty mark
CENTER_R = 3.0              # center circle radius
SUB_ZONE_HALF = 2.5         # half-length of each substitution zone (5 m total)
SUB_ZONE_GAP = 5.0          # distance between sub zones across the halfway line
CORNER_R = 0.25

HL = PITCH_L / 2.0           # 20
HW = PITCH_W / 2.0           # 10
G_HW = GOAL_W / 2.0          # 1.5


# ---------- keypoints (curated set; meaningful landmarks) ----------
# Each entry: (id, x, y, category_key)
# Categories use the same palette as the 11-a-side diagram.
KEYPOINTS = [
    # 0-3: pitch corners
    (0, -HL,  HW, "corner"),
    (1,  HL,  HW, "corner"),
    (2, -HL, -HW, "corner"),
    (3,  HL, -HW, "corner"),
    # 4-5: halfway line on touchlines
    (4, 0.0,  HW, "corner"),
    (5, 0.0, -HW, "corner"),

    # 6-9: goal posts (left then right, top then bottom of frame)
    (6, -HL,  G_HW, "goal"),
    (7, -HL, -G_HW, "goal"),
    (8,  HL,  G_HW, "goal"),
    (9,  HL, -G_HW, "goal"),

    # 10-15: penalty-area D anchors
    # Left D: front (apex 6 m from goal line at y=+/-1.5)
    (10, -HL + PA_R,  G_HW, "pa"),     # left D top apex
    (11, -HL + PA_R, -G_HW, "pa"),     # left D bot apex
    (12, -HL + PA_R, 0.0,   "pa"),     # left D forward-most point on axis
    # Right D
    (13,  HL - PA_R,  G_HW, "pa"),
    (14,  HL - PA_R, -G_HW, "pa"),
    (15,  HL - PA_R, 0.0,   "pa"),

    # 16-19: first & second penalty marks
    (16, -HL + FIRST_PEN,  0.0, "spot"),
    (17,  HL - FIRST_PEN,  0.0, "spot"),
    (18, -HL + SECOND_PEN, 0.0, "spot"),
    (19,  HL - SECOND_PEN, 0.0, "spot"),

    # 20: center spot
    (20, 0.0, 0.0, "spot"),
    # 21-22: center circle top/bottom on halfway line
    (21, 0.0,  CENTER_R, "spot"),
    (22, 0.0, -CENTER_R, "spot"),

    # 23-26: substitution zone end markers (one touchline only — Law 3)
    # By convention the sub zones are on the team-bench touchline (y = +HW).
    (23, -SUB_ZONE_GAP / 2 - SUB_ZONE_HALF * 2,  HW, "sub"),  # team A outer
    (24, -SUB_ZONE_GAP / 2,                       HW, "sub"),  # team A inner
    (25,  SUB_ZONE_GAP / 2,                       HW, "sub"),  # team B inner
    (26,  SUB_ZONE_GAP / 2 + SUB_ZONE_HALF * 2,   HW, "sub"),  # team B outer
]


KP_CATEGORIES = {
    "Pitch corners & halfway (0-5)":     {"color": "#00ff00", "marker": "s"},
    "Goal posts (6-9)":                  {"color": "#ff2222", "marker": "o"},
    "Penalty-area D (10-15)":            {"color": "#ffdd00", "marker": "o"},
    "Penalty marks & center (16-22)":    {"color": "#ff44ff", "marker": "o"},
    "Substitution zone (23-26)":         {"color": "#00dddd", "marker": "^"},
}
_cat_key_to_label = {
    "corner": "Pitch corners & halfway (0-5)",
    "goal":   "Goal posts (6-9)",
    "pa":     "Penalty-area D (10-15)",
    "spot":   "Penalty marks & center (16-22)",
    "sub":    "Substitution zone (23-26)",
}


# ---------- line definitions ----------
# Each entry: (line_id, p1, p2, kind). kind ∈ {"outline", "halfway", "goal", "pa_chord", "sub"}
LINES = [
    # Outline (touchlines + goal lines)
    (0,  (-HL,  HW), ( HL,  HW), "outline"),    # top touchline
    (1,  (-HL, -HW), ( HL, -HW), "outline"),    # bottom touchline
    (2,  (-HL, -HW), (-HL,  HW), "outline"),    # left goal line
    (3,  ( HL, -HW), ( HL,  HW), "outline"),    # right goal line
    # Halfway
    (4,  ( 0.0, -HW), ( 0.0,  HW), "halfway"),
    # Goals (visible as line on the goal line between the posts)
    (5,  (-HL, -G_HW), (-HL,  G_HW), "goal"),
    (6,  ( HL, -G_HW), ( HL,  G_HW), "goal"),
    # Penalty-area chord (straight segment connecting the two quarter arcs)
    (7,  (-HL + PA_R, -G_HW), (-HL + PA_R,  G_HW), "pa_chord"),
    (8,  ( HL - PA_R, -G_HW), ( HL - PA_R,  G_HW), "pa_chord"),
    # Substitution zone markers (perpendicular tick marks on touchline)
    # We draw them as short lines crossing the touchline at the 4 anchors.
]


LINE_COLORS = {
    "outline": "white",
    "halfway": "white",
    "goal": "#aaaaaa",
    "pa_chord": "#cccc66",
    "sub": "#00dddd",
}


# ---------- draw ----------
fig, ax = plt.subplots(figsize=(20, 12), facecolor="#1a472a")
ax.set_facecolor("#1a472a")
ax.set_aspect("equal")

# --- straight lines ---
for lid, p1, p2, kind in LINES:
    (x1, y1), (x2, y2) = p1, p2
    lw = 2.0
    ax.plot([x1, x2], [y1, y2], color=LINE_COLORS[kind], linewidth=lw, zorder=1)
    # Label at midpoint with normal offset
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    nx, ny = (-dy / length, dx / length) if length > 0 else (0, 1)
    off = 0.9
    ax.text(
        mx + nx * off, my + ny * off,
        f"L{lid}",
        color=LINE_COLORS[kind],
        fontsize=8,
        fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a472a", edgecolor="none", alpha=0.8),
        zorder=5,
    )

# --- penalty area D (quarter arcs centered on each post) ---
# Left D: two quarter-circles centered on (-HL, +G_HW) and (-HL, -G_HW).
# Right D: centered on (+HL, +/-G_HW). Arcs sweep into the field.
ARC_RES = 200
for sign in [-1, 1]:  # left=-1, right=1
    goal_x = sign * HL
    for post_y in (+G_HW, -G_HW):
        # Quarter-circle from the goal line out to the chord apex.
        # Angle from post into the field: 0..90 deg, but rotated per side/post.
        # Side selection: for left goal (sign=-1) arcs sweep to +x.
        # For top post (post_y=+G_HW): from (-HL, +G_HW + 6) down/around to (-HL+6, +G_HW).
        # For bot post: from (-HL, -G_HW - 6) up/around to (-HL+6, -G_HW).
        if sign < 0:
            # angles relative to post center: 0deg=+x, 90deg=+y (top), -90deg=-y (bot)
            if post_y > 0:
                start, end = 0.0, 90.0
            else:
                start, end = -90.0, 0.0
        else:
            # right goal: arcs sweep to -x
            if post_y > 0:
                start, end = 90.0, 180.0
            else:
                start, end = 180.0, 270.0
        thetas = np.linspace(math.radians(start), math.radians(end), ARC_RES)
        ax.plot(
            goal_x + PA_R * np.cos(thetas),
            post_y + PA_R * np.sin(thetas),
            color="#cccc66", linewidth=2.0, zorder=1,
        )

# --- center circle ---
ax.add_patch(plt.Circle((0, 0), CENTER_R, fill=False, edgecolor="white", linewidth=2, zorder=1))

# --- corner arcs (0.25 m, decorative) ---
for cx, cy, start_deg in [
    (-HL,  HW, -90.0),
    ( HL,  HW, 180.0),
    (-HL, -HW,   0.0),
    ( HL, -HW,  90.0),
]:
    thetas = np.linspace(math.radians(start_deg), math.radians(start_deg + 90), 40)
    ax.plot(
        cx + CORNER_R * np.cos(thetas),
        cy + CORNER_R * np.sin(thetas),
        color="white", linewidth=1.5, zorder=1,
    )

# --- substitution zone tick marks (perpendicular to top touchline) ---
TICK = 0.6
for kid, x, y, cat in KEYPOINTS:
    if cat != "sub":
        continue
    ax.plot([x, x], [y - TICK / 2, y + TICK / 2], color="#00dddd", linewidth=2, zorder=2)

# --- keypoints ---
for kid, x, y, cat in KEYPOINTS:
    label = _cat_key_to_label[cat]
    info = KP_CATEGORIES[label]
    ax.scatter(x, y, c=info["color"], s=60, marker=info["marker"],
               edgecolors="white", linewidths=0.6, zorder=10)

    # Label offset
    ox, oy = 0, 0.7
    if cat == "corner":
        ox, oy = (0.0, 0.9 if y > 0 else -1.0)
    elif cat == "goal":
        # post labels go outside the field
        ox = -1.1 if x < 0 else 1.1
        oy = 0.0
    elif cat == "pa":
        ox, oy = (0, -0.9 if y < 0 else 0.9)
        if kid in (12, 15):
            ox, oy = (0.9 if kid == 12 else -0.9, 0.7)
    elif cat == "spot":
        oy = -0.9 if abs(y) < 1e-6 else (0.9 if y > 0 else -0.9)
    elif cat == "sub":
        oy = 0.9
    ax.text(
        x + ox, y + oy,
        str(kid),
        color=info["color"],
        fontsize=8,
        fontweight="bold",
        ha="center", va="center",
        zorder=11,
    )

# --- annotations for dimensions ---
def dim(x0, y0, x1, y1, text, color="white", offset=0.6):
    """Draw a double-headed dimension arrow with a centered label."""
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.0), zorder=4)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    nx, ny = (-dy / length, dx / length) if length > 0 else (0, 1)
    ax.text(
        mx + nx * offset, my + ny * offset, text,
        color=color, fontsize=8, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#1a472a", edgecolor="none", alpha=0.85),
        zorder=6,
    )

# Length 40m at bottom
dim(-HL, -HW - 1.3, HL, -HW - 1.3, "Pitch length: 40 m")
# Width 20m on the right side
dim(HL + 1.3, -HW, HL + 1.3, HW, "Width: 20 m", offset=0.0)
# Goal width 3m (right goal, just outside)
dim(HL + 0.3, -G_HW, HL + 0.3, G_HW, "Goal 3 m", offset=0.5)
# 6m and 10m on left side (vertical guides drawn small)
dim(-HL, HW + 1.0, -HL + FIRST_PEN, HW + 1.0, "6 m  (1st pen)")
dim(-HL, HW + 2.4, -HL + SECOND_PEN, HW + 2.4, "10 m  (2nd pen)")
# Center circle radius
ax.annotate("", xy=(CENTER_R, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="white", lw=0.8), zorder=4)
ax.text(CENTER_R / 2, -0.5, "R = 3 m", color="white", fontsize=8, ha="center", va="top", zorder=6)

# Substitution zone label
ax.text(0, HW + 0.6, "Sub zones (5 m each, 5 m gap across halfway)",
        color="#00dddd", fontsize=8.5, ha="center", va="bottom",
        fontweight="bold", zorder=6)

# --- axes / title ---
ax.set_xlabel("x (meters)", color="white", fontsize=11)
ax.set_ylabel("y (meters)", color="white", fontsize=11)
ax.tick_params(colors="white", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("white")

ax.set_xlim(-24, 24)
ax.set_ylim(-14, 14)

ax.set_title(
    f"GoalInsight Futsal Pitch Reference  ({len(KEYPOINTS)} keypoints, {len(LINES)} straight lines + 4 arcs)",
    color="white", fontsize=15, fontweight="bold", pad=10,
)

ax.text(-HL, HW - 0.6, "LEFT GOAL", color="white", fontsize=11, fontweight="bold", ha="left", va="top")
ax.text( HL, HW - 0.6, "RIGHT GOAL", color="white", fontsize=11, fontweight="bold", ha="right", va="top")

# --- legend ---
legend_items = [
    mpatches.Patch(facecolor=info["color"], edgecolor="none", label=name)
    for name, info in KP_CATEGORIES.items()
]
legend_items.append(mpatches.Patch(facecolor="white", edgecolor="none",
                                   label="Lines: L0-L8 + 4 quarter arcs (D)"))
ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=8.5,
    facecolor="#1a472a",
    edgecolor="white",
    labelcolor="white",
    framealpha=0.9,
)

output_path = Path(__file__).parent.parent / "docs" / "pitch_reference_futsal.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {output_path}")
