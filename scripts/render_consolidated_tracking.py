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


def _player_label(
    track_id: str,
    player: dict | None,
    orig_tid: int | None = None,
) -> str:
    """Build a compact bbox label.

    Format: ``"<player_id> #<jersey> [orig <tid>]"`` — the orig tid
    suffix lets you trace back to the raw tracker fragment in
    tracking/tracks.json (consolidation merges multiple raw tids into
    one player_id; useful when auditing whether a fragmentation got
    fixed). Falls back to just the track_id when there's no cluster
    info.
    """
    orig_suffix = f" [orig {orig_tid}]" if orig_tid is not None else ""
    if not player:
        return f"{track_id}{orig_suffix}"
    pid = player.get("player_id", track_id)
    jersey = player.get("jersey_number")
    if jersey is not None:
        return f"{pid} #{jersey}{orig_suffix}"
    return f"{pid}{orig_suffix}"


def _draw_side_panel(
    frame: np.ndarray, players: list[dict], visible_ids: set[str],
    width: int | None = None,
) -> np.ndarray:
    """Append a right-side panel showing the player roster.

    Panel width and font scale are tied to frame height so the panel
    stays readable when the frame is downscaled to 1080p in the
    web viewer (4K → ~1.4× scale, 1080p → 0.7× — same ~1080-pixel
    panel either way).
    """
    h = frame.shape[0]
    s = max(0.7, h / 1500.0)              # 1080p → 0.72, 4K → 1.44
    if width is None:
        width = max(360, int(round(360 * s)))
    panel = np.zeros((h, width, 3), dtype=np.uint8)
    panel[:] = (24, 24, 28)

    title_scale = 0.85 * s
    header_scale = 0.65 * s
    item_scale = 0.6 * s
    line_h = max(22, int(round(22 * s)))
    text_thick = max(1, int(round(s)))

    cv2.putText(
        panel, "Consolidated players",
        (int(12 * s), int(32 * s)),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255),
        text_thick + 1, cv2.LINE_AA,
    )

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

    y = int(round(64 * s))
    for bucket in ("team_A", "team_B", "official"):
        items = grouped.get(bucket, [])
        if not items:
            continue
        header = bucket.replace("_", " ").upper()
        cv2.putText(
            panel, header, (int(12 * s), y),
            cv2.FONT_HERSHEY_SIMPLEX, header_scale, (200, 200, 200),
            text_thick + 1, cv2.LINE_AA,
        )
        y += line_h
        for p in sorted(items, key=lambda r: (r.get("player_id") or "")):
            color = _player_color(p)
            label = _player_label(p.get("player_id", "?"), p)
            visible = (p.get("player_id") in visible_ids)
            text_col = (255, 255, 255) if visible else (110, 110, 110)
            # color swatch
            sw_x1 = int(round(16 * s)); sw_x2 = int(round(34 * s))
            sw_top = y - int(round(14 * s)); sw_bot = y + int(round(2 * s))
            cv2.rectangle(panel, (sw_x1, sw_top), (sw_x2, sw_bot), color, -1)
            cv2.rectangle(panel, (sw_x1, sw_top), (sw_x2, sw_bot), (0, 0, 0), 1)
            # label
            cv2.putText(
                panel, label, (int(round(42 * s)), y),
                cv2.FONT_HERSHEY_SIMPLEX, item_scale, text_col,
                text_thick, cv2.LINE_AA,
            )
            y += line_h
            if y > h - line_h:
                break
        y += int(round(8 * s))

    return np.hstack([frame, panel])


def render_consolidated_video(
    video_path: Path,
    tracking_dir: Path,
    consolidation_dir: Path,
    output_path: Path,
    show_panel: bool = True,
    max_frames: int | None = None,
    vis_frame_stride: int = 0,
    write_mp4: bool = False,
) -> dict:
    """Render the tracking video with consolidated player_ids overlaid.

    Reusable from both this CLI and the pipeline's TrackConsolidationStage.
    Returns a stats dict (also written next to ``output_path``).

    ``vis_frame_stride`` > 0 saves ``output_path.parent / 'frames' /
    frame_NNN.jpg`` every Nth video frame for offline auditing.

    ``write_mp4`` controls whether the encoded ``output_path`` mp4 is
    actually written. The web UI's /pipeline page only reads the per-
    frame jpgs, so the mp4 is dead weight for that path — encoding 4K
    @ 60fps is the multi-minute tail of the stage. Default False.
    """
    # The consolidation stage writes its tracks.json (player_id strings,
    # jersey numbers populated) into its own dir; the raw tracker file
    # under tracking/ is never mutated.
    with open(consolidation_dir / "tracks.json") as f:
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
    if write_mp4:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, height))
    else:
        writer = None

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

        # Font scale relative to image height. The /pipeline page
        # serves jpgs through ``?w=1080``, and the consolidated frame
        # has a 360-px side panel making the long edge 4200 (vs 3840
        # plain), so a 4K source delivered at w=1080 lands at
        # 1080x535 — frame height halved twice. Scaling on the
        # post-panel composite frame.shape[0] underweights the font.
        # Fix: use the longer-of-two-dims so the panel-stretched
        # frames get a proportional bump.
        long_edge = max(frame.shape[:2])
        font_scale = max(0.6, long_edge / 2700.0)         # 4K = 1.55
        text_thick = max(1, int(round(long_edge / 1700.0)))   # 4K = 2
        rect_thick = max(2, int(round(long_edge / 1100.0)))   # 4K = 3

        tracks_now = all_tracks.get(str(frame_idx), [])
        visible_ids: set[str] = set()
        for t in tracks_now:
            visible_ids.add(str(t["track_id"]))

        want_jpg = (
            vis_frame_stride > 0
            and frame_idx % vis_frame_stride == 0
            and bool(tracks_now)
        )

        # Per-frame JSON sidecar: ships everything the client needs to
        # draw bbox + label overlays from JS — bbox, player_id, jersey,
        # team, role, plus the orig_track_id for the per-track drill-in
        # widget. Written alongside the JPG so the frame slider can
        # fetch them as a pair.
        if want_jpg:
            sidecar_tracks = []
            for t in tracks_now:
                tid = str(t["track_id"])
                player = player_by_id.get(tid) or {}
                bbox = t.get("bbox") or [0, 0, 0, 0]
                sidecar_tracks.append({
                    "player_id": tid,
                    "orig_track_id": t.get("orig_track_id"),
                    "bbox": [float(v) for v in bbox],
                    "jersey_number": (
                        t.get("jersey_number")
                        or player.get("jersey_number")
                    ),
                    "team": t.get("team") or player.get("team"),
                    "role": t.get("role") or player.get("role"),
                    "color_bgr": list(_player_color(player, t)),
                })
            with open(frames_dir / f"frame_{frame_idx:05d}.json", "w") as jf:
                json.dump({
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": round(frame_idx / fps, 3),
                    "n_visible": len(visible_ids),
                    # Source frame dimensions BEFORE the side panel is
                    # appended. Bboxes live in (video_w × video_h)
                    # space; the JS overlay uses these to scale
                    # correctly even when the JPG ships at a different
                    # resolution (e.g. ``?w=1080`` downscaling).
                    "video_w": int(width),
                    "video_h": int(height),
                    # Full JPG size (incl. roster panel) so the JS knows
                    # the original aspect ratio that backs the rendered
                    # ``naturalWidth``.
                    "jpg_w": int(out_w),
                    "jpg_h": int(height),
                    "tracks": sidecar_tracks,
                }, jf)

        if writer is not None:
            # Mp4 path keeps the legacy baked-in overlays for ad-hoc
            # video playback; only the per-frame JPG path went clean.
            mp4_frame = frame.copy()
            for t in tracks_now:
                tid = str(t["track_id"])
                player = player_by_id.get(tid)
                color = _player_color(player, t)
                orig_tid = t.get("orig_track_id")
                label = _player_label(tid, player, orig_tid=orig_tid)
                x1, y1, x2, y2 = (int(v) for v in t["bbox"])
                cv2.rectangle(mp4_frame, (x1, y1), (x2, y2), color, rect_thick)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick,
                )
                cv2.rectangle(
                    mp4_frame, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1,
                )
                cv2.putText(
                    mp4_frame, label, (x1 + 4, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    (255, 255, 255), text_thick, cv2.LINE_AA,
                )
            cv2.putText(
                mp4_frame, f"frame {frame_idx}  |  visible: {len(visible_ids)}",
                (10, int(40 * font_scale + 10)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0),
                text_thick + 1, cv2.LINE_AA,
            )
            if show_panel:
                mp4_frame = _draw_side_panel(mp4_frame, players, visible_ids)
            writer.write(mp4_frame)

        if want_jpg:
            # Clean frame for the web overlay path: just the source
            # video frame, no bbox / label / text / roster panel. The
            # JS overlay draws everything on top. Panel was removed
            # because it (a) changed JPG aspect ratio away from the
            # source video and (b) baked a permanent label/legend
            # block that the user can't control client-side.
            cv2.imwrite(str(frames_dir / f"frame_{frame_idx:05d}.jpg"), frame)

        frame_idx += 1
        n_processed += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
        logger.info("wrote %s (%d frames, %dx%d @ %.1f fps)",
                    output_path, n_processed, out_w, height, fps)
    else:
        logger.info("rendered %d annotated frames (mp4 skipped)",
                    n_processed)

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
