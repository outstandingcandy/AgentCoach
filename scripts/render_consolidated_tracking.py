#!/usr/bin/env python3
"""Render the tracking video with consolidated player_ids overlaid.

The tracking stage produces ``tracking.mp4`` with raw integer track ids
(1, 2, ...). After ``track_consolidation``, those tracks have been merged
into stable player_ids ("B-21", "A-18", ...) with team / jersey / role
attached. This script reads the post-consolidation ``tracks.json`` and
re-renders the video with player_id labels, team-coloured boxes, and
a side panel summarising every consolidated player.

Usage:

    python scripts/render_consolidated_tracking.py \\
        --video data/raw_videos/kids_soccer_clip_1250_1310.mp4 \\
        --tracking-dir output/tracking \\
        --consolidation-dir output/track_consolidation \\
        --output output/track_consolidation/players_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


# Stable BGR palette for team colours (red and yellow match the kids match
# kit; downstream code can override via consolidation_dir/scene.json).
TEAM_COLOR = {
    "team_A": (60, 60, 220),    # red-ish
    "team_B": (60, 200, 220),   # yellow-ish
}
ROLE_COLOR = {
    "goalkeeper": (255, 200, 0),   # cyan-ish
    "referee":    (60, 200, 60),   # green
    "linesman":   (180, 180, 180), # grey
    "other":      (255, 96, 32),   # blue-ish — unmapped/short/non-player
}
DEFAULT_COLOR = (180, 180, 180)
UNMAPPED_COLOR = (255, 96, 32)     # same blue as role=other


def _player_color(
    player: dict | None, track: dict | None = None,
) -> tuple[int, int, int]:
    """Pick a box colour. Unmapped tracks (role==other or no cluster) go
    blue so they read as 'detection only, no consolidated identity'."""
    if track is not None and str(track.get("role", "")) == "other":
        return UNMAPPED_COLOR
    if not player:
        return UNMAPPED_COLOR
    role = player.get("role", "player")
    if role in ROLE_COLOR:
        return ROLE_COLOR[role]
    team = player.get("team")
    return TEAM_COLOR.get(team, DEFAULT_COLOR)


def _player_label(track_id: str, player: dict | None) -> str:
    """Build a compact label: 'B-21 #21' or 'REF-01' or fallback to track_id."""
    if not player:
        return str(track_id)
    pid = player.get("player_id", track_id)
    jersey = player.get("jersey_number")
    if jersey is not None:
        return f"{pid} #{jersey}"
    return str(pid)


def _draw_side_panel(
    frame: np.ndarray, players: list[dict], visible_ids: set[str], width: int = 360,
) -> np.ndarray:
    """Append a right-side panel showing the player roster."""
    h = frame.shape[0]
    panel = np.zeros((h, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    cv2.putText(panel, "Consolidated players", (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    # Team A first, team B next, then officials.
    grouped: dict[str, list[dict]] = {"team_A": [], "team_B": [], "official": []}
    for p in players:
        if p.get("role") in ("referee", "linesman", "goalkeeper"):
            grouped["official" if p["role"] != "goalkeeper" else (p.get("team") or "official")] = (
                grouped.setdefault(p.get("team") or "official", [])
            )
        team = p.get("team")
        bucket = team if team in ("team_A", "team_B") else "official"
        grouped.setdefault(bucket, []).append(p)

    y = 64
    for bucket in ("team_A", "team_B", "official"):
        items = grouped.get(bucket, [])
        if not items:
            continue
        header = bucket.replace("_", " ").upper()
        cv2.putText(panel, header, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
        y += 22
        for p in sorted(items, key=lambda r: (r.get("player_id") or "")):
            color = _player_color(p)
            label = _player_label(p.get("player_id", "?"), p)
            visible = (p.get("player_id") in visible_ids)
            text_col = (255, 255, 255) if visible else (110, 110, 110)
            # color swatch
            cv2.rectangle(panel, (16, y - 14), (34, y + 2), color, -1)
            cv2.rectangle(panel, (16, y - 14), (34, y + 2), (0, 0, 0), 1)
            # label
            cv2.putText(panel, label, (42, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_col, 1)
            y += 22
            if y > h - 18:
                break
        y += 8

    return np.hstack([frame, panel])


def render_consolidated_video(
    video_path: Path,
    tracking_dir: Path,
    consolidation_dir: Path,
    output_path: Path,
    show_panel: bool = True,
    max_frames: int | None = None,
    vis_frame_stride: int = 0,
) -> dict:
    """Render the tracking video with consolidated player_ids overlaid.

    Reusable from both this CLI and the pipeline's TrackConsolidationStage.
    Returns a stats dict (also written next to ``output_path``).

    ``vis_frame_stride`` > 0 saves ``output_path.parent / 'frames' /
    frame_NNN.jpg`` every Nth video frame for offline auditing.
    """
    with open(tracking_dir / "tracks.json") as f:
        all_tracks: dict = json.load(f)
    with open(consolidation_dir / "players.json") as f:
        players: list[dict] = json.load(f)
    player_by_id = {p["player_id"]: p for p in players}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"can't open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = width + (360 if show_panel else 0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, height))

    frames_dir = output_path.parent / "frames"
    if vis_frame_stride > 0:
        frames_dir.mkdir(parents=True, exist_ok=True)

    appearance: Counter[str] = Counter()
    for tracks_in_frame in all_tracks.values():
        for t in tracks_in_frame:
            appearance[str(t["track_id"])] += 1
    logger.info("players: %d total, %d appear in video",
                len(players), sum(1 for p in players if p["player_id"] in appearance))

    frame_idx = 0
    n_processed = 0
    pbar = tqdm(total=max_frames or total_frames, desc="Consolidated render")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames and n_processed >= max_frames:
            break

        tracks_now = all_tracks.get(str(frame_idx), [])
        visible_ids: set[str] = set()
        for t in tracks_now:
            tid = str(t["track_id"])
            visible_ids.add(tid)
            player = player_by_id.get(tid)
            color = _player_color(player, t)
            label = _player_label(tid, player)
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.putText(frame, f"frame {frame_idx}  |  visible: {len(visible_ids)}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if show_panel:
            frame = _draw_side_panel(frame, players, visible_ids)

        writer.write(frame)
        if vis_frame_stride > 0 and frame_idx % vis_frame_stride == 0:
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), frame)

        frame_idx += 1
        n_processed += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    writer.release()
    logger.info("wrote %s (%d frames, %dx%d @ %.1f fps)",
                output_path, n_processed, out_w, height, fps)

    stats = {
        "total_players": len(players),
        "visible_in_video": sum(1 for p in players if p["player_id"] in appearance),
        "by_team": dict(Counter(p.get("team", "unknown") for p in players)),
        "by_role": dict(Counter(p.get("role", "unknown") for p in players)),
        "appearance_counts": dict(appearance.most_common()),
    }
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("stats → %s", stats_path)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--tracking-dir", required=True,
                        help="Has tracks.json (post-consolidation)")
    parser.add_argument("--consolidation-dir", required=True,
                        help="Has players.json + player_map.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (debug)")
    parser.add_argument("--show-panel", action="store_true", default=True,
                        help="Append a right-side roster panel (default on)")
    parser.add_argument("--no-panel", dest="show_panel", action="store_false")
    parser.add_argument("--vis-frame-stride", type=int, default=0,
                        help="Save frames/frame_NNN.jpg every Nth video frame "
                             "(0 = mp4 only, default).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    render_consolidated_video(
        video_path=Path(args.video),
        tracking_dir=Path(args.tracking_dir),
        consolidation_dir=Path(args.consolidation_dir),
        output_path=Path(args.output),
        show_panel=args.show_panel,
        max_frames=args.max_frames,
        vis_frame_stride=args.vis_frame_stride,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
