"""Event detection video renderer.

Renders a full video with event annotations overlaid. Each event is displayed
as a banner at the frame it occurs and persists for a configurable duration.
Event IDs match events.json for cross-reference.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm


# Event type colors (BGR)
EVENT_COLORS = {
    "possession": (200, 180, 0),     # Teal
    "pass": (0, 200, 0),             # Green
    "shot": (0, 80, 255),            # Orange-red
    "goal": (0, 0, 255),             # Red
    "carry": (200, 150, 0),          # Cyan-ish
    "tackle": (0, 200, 200),         # Yellow
    "interception": (200, 0, 200),   # Magenta
}

TEAM_COLORS = {
    "team_A": (0, 0, 255),     # Red
    "team_B": (255, 0, 0),     # Blue
}

# Event type display names
EVENT_LABELS = {
    "possession": "POSS",
    "pass": "PASS",
    "shot": "SHOT",
    "goal": "GOAL",
    "carry": "CARRY",
    "tackle": "TACKLE",
    "interception": "INT",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX


def render_event_video(
    video_path: Path,
    events: list[dict],
    output_path: Path,
    tracking_dir: Path | None = None,
    fps: float | None = None,
    banner_duration_sec: float = 3.0,
    vis_frame_stride: int = 10,
) -> None:
    """Render full video with event annotations.

    Args:
        video_path: Source video file.
        events: List of event dicts (as serialized by MatchEvent.to_dict()).
        output_path: Output .mp4 path.
        tracking_dir: If provided, load tracks.json and ball_tracks.json
            to also draw player/ball overlays.
        fps: Override output FPS. If None, uses source video FPS.
        banner_duration_sec: How long each event banner stays on screen.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_fps = fps or video_fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    # Per-frame jpg sink at fixed stride for inspection. mp4 stays full-rate.
    frames_dir = output_path.parent / "frames"
    if vis_frame_stride > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Build frame-indexed event lookup: frame -> list of active events
    banner_frames = int(banner_duration_sec * video_fps)
    event_index = _build_event_index(events, total_frames, banner_frames)

    # Load tracking data for overlay (optional)
    player_tracks = {}
    ball_tracks = {}
    team_assignments = {}
    player_frame_keys = []
    ball_frame_keys = []
    if tracking_dir:
        player_tracks = _load_json(tracking_dir / "tracks.json") or {}
        ball_tracks = _load_json(tracking_dir / "ball_tracks.json") or {}
        team_assignments = _load_json(tracking_dir / "team_assignments.json") or {}
        # Pre-sort frame keys for nearest-frame lookup (tracking may be subsampled)
        player_frame_keys = sorted(int(k) for k in player_tracks.keys())
        ball_frame_keys = sorted(int(k) for k in ball_tracks.keys())

    # Build a sorted list of frames that have events (for the event timeline bar)
    event_frames = sorted(set(e["frame"] for e in events))

    print(f"Rendering event video: {total_frames} frames, {len(events)} events")

    for frame_idx in tqdm(range(total_frames), desc="  Event video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw tracking overlays (players + ball)
        # Tracking data may be subsampled (e.g., every 3rd frame at process_fps=10).
        # Use nearest available frame to avoid flickering.
        if player_tracks or ball_tracks:
            player_key = _find_nearest_key(frame_idx, player_frame_keys)
            ball_key = _find_nearest_key(frame_idx, ball_frame_keys)
            frame = _draw_tracking_overlay(
                frame, player_key, ball_key,
                player_tracks, ball_tracks, team_assignments,
            )

        # Draw active event banners
        active_events = event_index.get(frame_idx, [])
        if active_events:
            frame = _draw_event_banners(frame, active_events, frame_idx, video_fps)

        # Draw event timeline bar at bottom
        frame = _draw_timeline(
            frame, frame_idx, total_frames, events, banner_frames
        )

        out.write(frame)
        if vis_frame_stride > 0 and frame_idx % vis_frame_stride == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), frame)

    cap.release()
    out.release()
    print(f"  Event video saved: {output_path}")
    if vis_frame_stride > 0:
        print(f"  Event per-frame jpg @ every {vis_frame_stride}th: {frames_dir}/")


def _build_event_index(
    events: list[dict],
    total_frames: int,
    banner_frames: int,
) -> dict[int, list[dict]]:
    """Build frame -> active events mapping.

    For events with start_frame/end_frame (possession, carry), the event
    is active for its entire duration. For point events (shot, pass, etc.),
    the banner is shown for banner_frames after the event frame.
    """
    index: dict[int, list[dict]] = {}

    for event in events:
        etype = event.get("type", "")
        event_frame = event["frame"]

        # Determine display range
        if etype in ("possession", "carry") and event.get("start_frame") and event.get("end_frame"):
            start = event["start_frame"]
            end = min(event["end_frame"] + banner_frames, total_frames)
        else:
            start = event_frame
            end = min(event_frame + banner_frames, total_frames)

        for f in range(start, end):
            index.setdefault(f, []).append(event)

    return index


def _draw_event_banners(
    frame: np.ndarray,
    active_events: list[dict],
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Draw event banners on the right side of the frame."""
    h, w = frame.shape[:2]

    # Deduplicate by event_id and prioritize non-possession events
    seen = set()
    unique_events = []
    # Sort: goal/shot/pass first, then carry/interception/tackle, possession last
    priority = {"goal": 0, "shot": 1, "pass": 2, "interception": 3,
                "tackle": 4, "carry": 5, "possession": 6}
    sorted_events = sorted(active_events, key=lambda e: priority.get(e.get("type", ""), 9))

    for e in sorted_events:
        eid = e["event_id"]
        if eid not in seen:
            seen.add(eid)
            unique_events.append(e)

    # Limit to top 5 banners
    display_events = unique_events[:5]

    banner_h = 36
    banner_w = 340
    margin = 8
    x_start = w - banner_w - margin
    y_start = margin

    for i, event in enumerate(display_events):
        etype = event.get("type", "unknown")
        color = EVENT_COLORS.get(etype, (128, 128, 128))
        label = EVENT_LABELS.get(etype, etype.upper())
        eid = event.get("event_id", "")

        y = y_start + i * (banner_h + 6)
        if y + banner_h > h - 50:
            break

        # Semi-transparent banner background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start, y), (x_start + banner_w, y + banner_h),
                       color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (x_start, y), (x_start + banner_w, y + banner_h),
                       color, 2)

        # Event label (type badge)
        cv2.putText(frame, label, (x_start + 6, y + 24),
                     FONT, 0.6, (255, 255, 255), 2)

        # Event details
        detail = _format_event_detail(event)
        cv2.putText(frame, detail, (x_start + 75, y + 24),
                     FONT, 0.45, (255, 255, 255), 1)

        # Event ID (small, top-right corner of banner)
        cv2.putText(frame, eid, (x_start + banner_w - 80, y + 12),
                     FONT, 0.3, (200, 200, 200), 1)

    return frame


def _format_event_detail(event: dict) -> str:
    """Format event detail text for the banner."""
    etype = event.get("type", "")
    team = event.get("team_id", "")
    player = event.get("player_id")
    meta = event.get("metadata", {})

    team_short = "A" if team == "team_A" else ("B" if team == "team_B" else "")

    if etype == "possession":
        dur = meta.get("duration_sec", 0)
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid} {dur:.1f}s"

    elif etype == "pass":
        outcome = meta.get("outcome", "")
        length = meta.get("pass_length", 0)
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid} {length:.0f}m {outcome}"

    elif etype == "shot":
        outcome = meta.get("outcome", "")
        speed = meta.get("ball_speed_mps", 0)
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid} {speed:.0f}m/s {outcome}"

    elif etype == "goal":
        speed = meta.get("ball_speed_mps", 0)
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid} GOAL! {speed:.0f}m/s"

    elif etype == "carry":
        dist = meta.get("distance_m", 0)
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid} {dist:.0f}m"

    elif etype == "interception":
        pid = f"#{player}" if player else ""
        opp = meta.get("opponent_id")
        opp_str = f"vs #{opp}" if opp else ""
        return f"{team_short} {pid} {opp_str}"

    elif etype == "tackle":
        pid = f"#{player}" if player else ""
        return f"{team_short} {pid}"

    return ""


def _find_nearest_key(
    frame_idx: int,
    sorted_keys: list[int],
    max_distance: int = 5,
) -> int | None:
    """Find the nearest frame key to frame_idx via binary search.

    Returns None if no key is within max_distance frames.
    """
    if not sorted_keys:
        return None
    import bisect
    pos = bisect.bisect_left(sorted_keys, frame_idx)
    best = None
    best_dist = max_distance + 1
    for i in (pos - 1, pos):
        if 0 <= i < len(sorted_keys):
            d = abs(sorted_keys[i] - frame_idx)
            if d < best_dist:
                best_dist = d
                best = sorted_keys[i]
    return best


def _draw_tracking_overlay(
    frame: np.ndarray,
    player_key: int | None,
    ball_key: int | None,
    player_tracks: dict,
    ball_tracks: dict,
    team_assignments: dict,
) -> np.ndarray:
    """Draw lightweight player bbox + ball position overlay."""
    h, w = frame.shape[:2]

    # Player tracks
    if player_key is not None:
        tracks = player_tracks.get(str(player_key), [])
        for t in tracks:
            bbox = t.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            tid = t.get("track_id", 0)
            team = team_assignments.get(str(tid), t.get("team", "unknown"))
            color = TEAM_COLORS.get(team, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label
            role = t.get("role", "player")
            label = f"{tid}"
            if role == "goalkeeper":
                label = f"GK-{tid}"
            cv2.putText(frame, label, (x1, y1 - 4), FONT, 0.4, color, 1)

    # Ball track
    if ball_key is not None:
        ball = ball_tracks.get(str(ball_key))
        if ball:
            center = ball.get("center")
            if center:
                cx, cy = int(center[0]), int(center[1])
                cv2.circle(frame, (cx, cy), 8, (0, 165, 255), -1)
                cv2.circle(frame, (cx, cy), 8, (255, 255, 255), 2)

    return frame


def _draw_timeline(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    events: list[dict],
    banner_frames: int,
) -> np.ndarray:
    """Draw a timeline bar at the bottom showing event positions."""
    h, w = frame.shape[:2]
    bar_h = 20
    bar_y = h - bar_h
    margin_x = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, bar_y), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    bar_w = w - 2 * margin_x

    # Draw event markers
    for event in events:
        etype = event.get("type", "")
        if etype == "possession":
            continue  # Skip possession to avoid clutter
        ef = event["frame"]
        x = margin_x + int(ef / max(total_frames, 1) * bar_w)
        color = EVENT_COLORS.get(etype, (128, 128, 128))
        cv2.line(frame, (x, bar_y + 2), (x, h - 2), color, 2)

    # Current position indicator
    cx = margin_x + int(frame_idx / max(total_frames, 1) * bar_w)
    cv2.line(frame, (cx, bar_y), (cx, h), (255, 255, 255), 2)

    # Timestamp
    secs = frame_idx / 30.0  # approximate
    cv2.putText(frame, f"{int(secs//60)}:{int(secs%60):02d}",
                 (cx + 4, h - 5), FONT, 0.35, (255, 255, 255), 1)

    return frame


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
