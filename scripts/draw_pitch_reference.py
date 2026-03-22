#!/usr/bin/env python3
"""Generate a top-down pitch reference diagram with keypoint and line IDs."""

import math
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from goalinsight.field_registration.pnlcalib.keypoint_mapping import KeypointMapper
from goalinsight.field_registration.pnlcalib.line_mapping import LineMapper

# ---------- data ----------
KP_COORDS = KeypointMapper.PNLCALIB_WORLD_COORDS_2D  # centered (0,0) = pitch center
KP_NAMES = {idx: name for idx, name in KeypointMapper.PNLCALIB_KEYPOINTS}
NON_GROUND_KP = KeypointMapper.NON_GROUND_KEYPOINTS  # {12,14,16,18}
LINE_DEFS = LineMapper.LINE_DEFINITIONS
NON_GROUND_LINES = LineMapper.NON_GROUND_LINES  # {6,7,8,9,10,11}

# ---------- keypoint categories & colors ----------
KP_CATEGORIES = {
    "Pitch corners (0-2, 27-29)": {
        "ids": {0, 1, 2, 27, 28, 29},
        "color": "#00ff00",  # green
        "marker": "s",
    },
    "Penalty area (3-6, 23-26, 30, 33, 35-36, 40, 43-46, 54-56)": {
        "ids": {3, 4, 5, 6, 23, 24, 25, 26, 30, 33, 35, 36, 40, 43, 44, 45, 46, 54, 55, 56},
        "color": "#ffdd00",  # yellow
        "marker": "o",
    },
    "Goal area (7-10, 19-22)": {
        "ids": {7, 8, 9, 10, 19, 20, 21, 22},
        "color": "#00dddd",  # cyan
        "marker": "o",
    },
    "Center circle/spot (31, 34, 37-38, 41-42, 47-53)": {
        "ids": {31, 34, 37, 38, 41, 42, 47, 48, 49, 50, 51, 52, 53},
        "color": "#ff44ff",  # magenta
        "marker": "o",
    },
    "Goal posts - non-ground (11-18)": {
        "ids": {11, 12, 13, 14, 15, 16, 17, 18},
        "color": "#ff2222",  # red
        "marker": "o",
    },
    "Penalty arc inner (32, 39)": {
        "ids": {32, 39},
        "color": "#ffdd00",
        "marker": "o",
    },
}

# Build id -> category lookup
_kp_color = {}
_kp_marker = {}
for cat_info in KP_CATEGORIES.values():
    for kid in cat_info["ids"]:
        _kp_color[kid] = cat_info["color"]
        _kp_marker[kid] = cat_info["marker"]

# ---------- line categories & colors ----------
LINE_COLORS = {}
for lid in range(23):
    if lid in NON_GROUND_LINES:
        LINE_COLORS[lid] = "#aaaaaa"  # gray for goal posts / crossbar
    elif lid == 12:
        LINE_COLORS[lid] = "white"
    elif lid in {13, 14, 15, 16}:
        LINE_COLORS[lid] = "white"
    elif lid in {0, 1, 2, 3, 4, 5}:
        LINE_COLORS[lid] = "#cccc66"  # penalty area
    else:
        LINE_COLORS[lid] = "#66cccc"  # goal area

# ---------- draw ----------
CENTER_CIRCLE_R = 9.15

fig, ax = plt.subplots(figsize=(20, 12), facecolor="#1a472a")
ax.set_facecolor("#1a472a")
ax.set_aspect("equal")

# Draw lines
for lid, ldef in LINE_DEFS.items():
    x1, y1, z1 = ldef["p1"]
    x2, y2, z2 = ldef["p2"]
    is_3d = lid in NON_GROUND_LINES
    lw = 1.5 if is_3d else 2.0
    ls = "--" if is_3d else "-"
    ax.plot([x1, x2], [y1, y2], color=LINE_COLORS[lid], linewidth=lw, linestyle=ls, zorder=1)
    # Line ID label at midpoint
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    # Offset label slightly so it doesn't overlap the line
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length > 0:
        nx, ny = -dy / length, dx / length  # normal direction
    else:
        nx, ny = 0, 1
    off = 1.8
    ax.text(
        mx + nx * off, my + ny * off,
        f"L{lid}",
        color=LINE_COLORS[lid],
        fontsize=7,
        fontweight="bold",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="#1a472a", edgecolor="none", alpha=0.8),
        zorder=5,
    )

# Draw center circle
circle = plt.Circle((0, 0), CENTER_CIRCLE_R, fill=False, edgecolor="white", linewidth=2, zorder=1)
ax.add_patch(circle)

# Draw penalty arcs (only the part outside penalty area)
for sign in [-1, 1]:  # left, right
    penalty_spot_x = sign * 41.5
    penalty_front_x = sign * (52.5 - 16.5)
    angles = np.linspace(-90, 90, 200)
    arc_x = penalty_spot_x + CENTER_CIRCLE_R * np.cos(np.radians(angles)) * (-sign)
    arc_y = CENTER_CIRCLE_R * np.sin(np.radians(angles))
    # Keep only points outside penalty area
    if sign == -1:
        mask = arc_x > penalty_front_x
    else:
        mask = arc_x < penalty_front_x
    ax.plot(arc_x[mask], arc_y[mask], color="white", linewidth=2, zorder=1)

# Draw keypoints
for kid, (x, y) in enumerate(KP_COORDS):
    color = _kp_color.get(kid, "white")
    marker = _kp_marker.get(kid, "o")
    size = 50
    edge = "white" if kid in NON_GROUND_KP else "none"
    ax.scatter(x, y, c=color, s=size, marker=marker, edgecolors=edge, linewidths=0.8, zorder=10)

    # Label offset to avoid overlap
    ox, oy = 0, 1.5
    # Adjust offsets for crowded areas
    if kid in {11, 12}:
        ox, oy = -2.5, 0
    elif kid in {15, 16}:
        ox, oy = -2.5, 0
    elif kid in {13, 14}:
        ox, oy = 2.5, 0
    elif kid in {17, 18}:
        ox, oy = 2.5, 0
    elif kid in {7, 8}:
        ox, oy = 0, 1.5
    elif kid in {19, 20}:
        ox, oy = 0, -1.8
    elif kid in {9, 10}:
        ox, oy = 0, 1.5
    elif kid in {21, 22}:
        ox, oy = 0, -1.8
    elif kid in {44, 45, 46}:
        oy = -1.8
    elif kid in {54, 55, 56}:
        oy = -1.8
    elif kid in {49, 50, 51}:
        oy = -1.8
    elif kid in {36, 40}:
        ox = -2.0
    elif kid in {39, 43}:
        ox = 2.0

    ax.text(
        x + ox, y + oy,
        str(kid),
        color=color,
        fontsize=7.5,
        fontweight="bold",
        ha="center", va="center",
        zorder=11,
    )

# Axis labels
ax.set_xlabel("x (meters)", color="white", fontsize=11)
ax.set_ylabel("y (meters)", color="white", fontsize=11)
ax.tick_params(colors="white", labelsize=9)
for spine in ax.spines.values():
    spine.set_color("white")

# Margin
ax.set_xlim(-58, 58)
ax.set_ylim(-40, 40)

# Title
ax.set_title(
    f"GoalInsight Pitch Reference  ({len(KP_COORDS)} keypoints, {len(LINE_DEFS)} lines)",
    color="white", fontsize=16, fontweight="bold", pad=12,
)

# Side labels
ax.text(-52.5, 36, "LEFT GOAL", color="white", fontsize=12, fontweight="bold", ha="center")
ax.text(52.5, 36, "RIGHT GOAL", color="white", fontsize=12, fontweight="bold", ha="center")

# Legend
legend_items = []
for cat_name, cat_info in KP_CATEGORIES.items():
    if cat_info == KP_CATEGORIES.get("Penalty arc inner (32, 39)"):
        continue  # merged with penalty area visually
    legend_items.append(
        mpatches.Patch(facecolor=cat_info["color"], edgecolor="none", label=cat_name)
    )
legend_items.append(mpatches.Patch(facecolor="white", edgecolor="none", label="Lines: L0-L22 (dashed = non-ground)"))

leg = ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=8.5,
    facecolor="#1a472a",
    edgecolor="white",
    labelcolor="white",
    framealpha=0.9,
)

output_path = Path(__file__).parent.parent / "docs" / "pitch_reference.png"
output_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {output_path}")
