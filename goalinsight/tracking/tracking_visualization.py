"""Visualization helpers: draw_tracks, draw_ball_track, draw_topdown_pitch."""

import cv2
import numpy as np



# Team colors for visualization
TEAM_COLORS = {
    "team_A": (0, 0, 255),     # Red
    "team_B": (255, 0, 0),     # Blue
    "referee": (0, 255, 255),  # Yellow
    "unknown": (128, 128, 128),  # Gray
    "ball": (0, 165, 255),     # Orange
}


def get_color_for_track(track_id: int, team: str | None = None) -> tuple[int, int, int]:
    """Get color for a track based on team assignment."""
    if team and team in TEAM_COLORS:
        return TEAM_COLORS[team]
    # Fallback to ID-based color
    import random
    random.seed(track_id * 7)
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


def draw_topdown_pitch(
    height: int,
    tracks: list[dict],
    team_assignments: dict[int, str],
    ball_track: dict | None = None,
    ball_trajectory_world: list[list[float]] | None = None,
    pitch_length: float = 105.0,
    pitch_width: float = 68.0,
) -> np.ndarray:
    """Draw top-down pitch diagram with player and ball positions.

    Args:
        height: Output image height (matches camera frame height).
        tracks: List of track dicts with pitch_position, track_id, role.
        team_assignments: Dict of track_id -> team label.
        ball_track: Ball track dict with pitch_position (optional).
        pitch_length: Pitch length in meters.
        pitch_width: Pitch width in meters.

    Returns:
        Top-down pitch image (BGR).
    """
    width = int(height * 0.75)  # 3:4 aspect for pitch
    pitch = np.zeros((height, width, 3), dtype=np.uint8)
    pitch[:] = (34, 139, 34)

    half_l = pitch_length / 2
    half_w = pitch_width / 2
    margin = 6
    scale = min(
        (width - 2) / (pitch_length + 2 * margin),
        (height - 2) / (pitch_width + 2 * margin),
    )
    ox = width / 2
    oy = height / 2

    def w2p(wx, wy):
        px = int(ox + wx * scale)
        py = int(oy - wy * scale)
        return (px, py)

    lc = (255, 255, 255)
    lw = max(1, int(scale * 0.25))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Pitch markings
    cv2.rectangle(pitch, w2p(-half_l, half_w), w2p(half_l, -half_w), lc, lw)
    cv2.line(pitch, w2p(0, half_w), w2p(0, -half_w), lc, lw)
    cv2.circle(pitch, w2p(0, 0), int(9.15 * scale), lc, lw)
    cv2.circle(pitch, w2p(0, 0), max(2, int(0.3 * scale)), lc, -1)
    # Penalty areas
    pa_w = min(20.16, half_w)
    pa_d = min(16.5, half_l * 0.35)
    cv2.rectangle(pitch, w2p(-half_l, pa_w), w2p(-half_l + pa_d, -pa_w), lc, lw)
    cv2.rectangle(pitch, w2p(half_l - pa_d, pa_w), w2p(half_l, -pa_w), lc, lw)
    # Goal areas
    ga_w = min(9.16, half_w * 0.5)
    ga_d = min(5.5, half_l * 0.12)
    cv2.rectangle(pitch, w2p(-half_l, ga_w), w2p(-half_l + ga_d, -ga_w), lc, lw)
    cv2.rectangle(pitch, w2p(half_l - ga_d, ga_w), w2p(half_l, -ga_w), lc, lw)

    # Role-specific markers
    ROLE_SHAPES = {
        "goalkeeper": "diamond",
        "referee": "triangle",
        "player": "circle",
    }

    r = max(5, int(1.2 * scale))
    for track in tracks:
        pos = track.get("pitch_position")
        if pos is None:
            continue

        tid = track["track_id"]
        team = team_assignments.get(tid, track.get("team", "unknown"))
        role = track.get("role", "player")
        color = get_color_for_track(tid, team)

        px, py = w2p(pos[0], pos[1])
        if not (-50 < px < width + 50 and -50 < py < height + 50):
            continue

        shape = ROLE_SHAPES.get(role, "circle")
        if shape == "diamond":
            pts = np.array([
                [px, py - r], [px + r, py], [px, py + r], [px - r, py]
            ], dtype=np.int32)
            cv2.fillPoly(pitch, [pts], color)
            cv2.polylines(pitch, [pts], True, (255, 255, 255), 1)
        elif shape == "triangle":
            pts = np.array([
                [px, py - r], [px + r, py + r], [px - r, py + r]
            ], dtype=np.int32)
            cv2.fillPoly(pitch, [pts], color)
            cv2.polylines(pitch, [pts], True, (255, 255, 255), 1)
        else:
            cv2.circle(pitch, (px, py), r, color, -1)
            cv2.circle(pitch, (px, py), r, (255, 255, 255), 1)

        # Label
        label = f"{tid}"
        if role == "goalkeeper":
            label = f"GK{tid}"
        cv2.putText(pitch, label, (px + r + 2, py + 3), font, 0.35, (255, 255, 255), 1)

    # Ball trajectory trail on pitch
    if ball_trajectory_world and len(ball_trajectory_world) > 1:
        n = len(ball_trajectory_world)
        for i in range(1, n):
            alpha = i / n
            color = (0, int(100 + 65 * alpha), int(180 + 75 * alpha))
            thickness = max(1, int(2 * alpha))
            p1 = w2p(ball_trajectory_world[i - 1][0], ball_trajectory_world[i - 1][1])
            p2 = w2p(ball_trajectory_world[i][0], ball_trajectory_world[i][1])
            cv2.line(pitch, p1, p2, color, thickness)

    # Ball current position
    if ball_track and ball_track.get("pitch_position"):
        bx, by = ball_track["pitch_position"]
        bpx, bpy = w2p(bx, by)
        ball_r = max(6, int(1.0 * scale))
        cv2.circle(pitch, (bpx, bpy), ball_r, TEAM_COLORS["ball"], -1)
        cv2.circle(pitch, (bpx, bpy), ball_r, (255, 255, 255), 2)

    # Legend
    y_leg = 20
    for label, col in [("team_A", TEAM_COLORS["team_A"]),
                        ("team_B", TEAM_COLORS["team_B"]),
                        ("referee", TEAM_COLORS["referee"]),
                        ("GK", TEAM_COLORS["team_A"])]:
        cv2.rectangle(pitch, (5, y_leg - 10), (15, y_leg), col, -1)
        cv2.putText(pitch, label, (18, y_leg), font, 0.35, (255, 255, 255), 1)
        y_leg += 18

    return pitch


def draw_tracks(
    frame: np.ndarray,
    tracks: list[dict],
    track_history: dict,
    team_assignments: dict[int, str],
) -> np.ndarray:
    """Draw tracked bounding boxes and trajectories."""
    vis = frame.copy()

    for track in tracks:
        track_id = track["track_id"]
        bbox = track["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        team = team_assignments.get(track_id, track.get("team", "unknown"))
        color = get_color_for_track(track_id, team)

        # Draw bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label
        role = track.get("role", "player")
        jersey = track.get("jersey_number")
        label = f"ID:{track_id}"
        if team and team != "unknown":
            label = f"{team[5:]}-{track_id}"  # e.g., "A-5"
        if jersey:
            label += f" #{jersey}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Track history (kept for potential future use, not drawn on camera view)
        center = ((x1 + x2) // 2, y2)
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(center)
        if len(track_history[track_id]) > 15:
            track_history[track_id] = track_history[track_id][-15:]

    # Draw legend
    y_offset = 30
    for team_name, color in TEAM_COLORS.items():
        if team_name in team_assignments.values():
            cv2.rectangle(vis, (10, y_offset - 15), (25, y_offset), color, -1)
            cv2.putText(vis, team_name, (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20

    cv2.putText(vis, f"Tracks: {len(tracks)}", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return vis


def _tid_color(tid: int) -> tuple[int, int, int]:
    """Generate a distinct BGR color for a track ID."""
    PALETTE = [
        (0, 165, 255),   # orange
        (255, 100, 0),   # blue
        (0, 220, 120),   # green
        (180, 0, 255),   # magenta
        (255, 220, 0),   # cyan
        (0, 100, 255),   # red-orange
    ]
    return PALETTE[tid % len(PALETTE)]


def draw_ball_track(
    frame: np.ndarray,
    ball_track: dict | None,
    ball_trajectory: list[tuple[float, float]] | None = None,
    ball_detections: list[dict] | None = None,
    ball_trajectories_by_tid: dict[int, list[tuple[float, float]]] | None = None,
) -> np.ndarray:
    """Draw ball position, trajectory, and raw detections on frame.

    Args:
        frame: Input frame.
        ball_track: Tracked ball dict (center, confidence, etc.) or None.
        ball_trajectory: List of recent pixel positions for trail (legacy single trail).
        ball_detections: Raw detection dicts from ball_debug_log
            (each has center, confidence, source).
        ball_trajectories_by_tid: Dict of track_id -> list of (x, y) pixel positions.
            When provided, draws each track's trajectory in a distinct color.
    """
    vis = frame.copy()

    # Draw raw detections first (underneath tracked ball)
    if ball_detections:
        _DET_COLORS = {
            "pass1": (0, 200, 0),      # green
            "pass2": (200, 200, 0),     # cyan-ish
        }
        for det in ball_detections:
            dc = det.get("center", [0, 0])
            dx, dy = int(dc[0]), int(dc[1])
            source = det.get("source", "pass1")
            color = _DET_COLORS.get(source, (0, 200, 0))
            conf = det.get("confidence", 0.0)

            # Small circle for detection
            cv2.circle(vis, (dx, dy), 6, color, 2)
            det_label = f"det {conf:.2f}"
            if ball_track is None:
                det_label += " [no track]"
            cv2.putText(vis, det_label, (dx + 10, dy - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    if ball_track is None:
        return vis

    center = ball_track.get("center", [0, 0])
    cx, cy = int(center[0]), int(center[1])

    # Draw per-track trajectory trails in distinct colors
    if ball_trajectories_by_tid:
        for tid, trail in ball_trajectories_by_tid.items():
            if len(trail) < 2:
                continue
            base_color = _tid_color(tid)
            n = len(trail)
            for i in range(1, n):
                alpha = i / n
                color = tuple(int(c * (0.4 + 0.6 * alpha)) for c in base_color)
                thickness = max(1, int(2 * alpha))
                pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
                pt2 = (int(trail[i][0]), int(trail[i][1]))
                cv2.line(vis, pt1, pt2, color, thickness)
    elif ball_trajectory and len(ball_trajectory) > 1:
        # Fallback: single trajectory trail
        n = len(ball_trajectory)
        for i in range(1, n):
            alpha = i / n
            color = (0, int(100 + 65 * alpha), int(180 + 75 * alpha))
            thickness = max(1, int(2 * alpha))
            pt1 = (int(ball_trajectory[i - 1][0]), int(ball_trajectory[i - 1][1]))
            pt2 = (int(ball_trajectory[i][0]), int(ball_trajectory[i][1]))
            cv2.line(vis, pt1, pt2, color, thickness)

    # Draw ball with outline
    radius = 10
    cv2.circle(vis, (cx, cy), radius, TEAM_COLORS["ball"], -1)
    cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2)

    # Draw velocity arrow
    if "velocity" in ball_track:
        vx, vy = ball_track["velocity"]
        speed = (vx**2 + vy**2) ** 0.5
        if speed > 2.0:
            max_len = 60.0
            arrow_scale = min(4.0, max_len / speed)
            end_x = int(cx + vx * arrow_scale)
            end_y = int(cy + vy * arrow_scale)
            cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.25)

    # Draw label with height info
    conf = ball_track.get("confidence", 0.0)
    status = ball_track.get("status", "unknown")
    label = f"Ball {conf:.2f}"
    if status == "confirmed":
        label += " [OK]"
    height_m = ball_track.get("height", 0.0)
    if height_m and height_m > 0.3:
        label += f" h={height_m:.1f}m"
    cv2.putText(vis, label, (cx + 15, cy + 5),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEAM_COLORS["ball"], 1)

    return vis
