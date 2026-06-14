"""Generate a markdown report of consolidated player clusters.

Reads ``track_consolidation/players.json`` (one PlayerCluster per item)
and ``tracking/tracks.json``, samples K representative frames per
*source* track via the same sampler the consolidation stage uses, dumps
them under ``track_consolidation/track_thumbs/`` and writes
``consolidated_tracks.md`` linking it all up.

Usage:
    python scripts/build_consolidation_md.py \\
        --run-dir output/kids_clip_1250_1310 \\
        --video data/raw_videos/kids_soccer_clip_1250_1310.mp4 \\
        [--frames-per-track 5]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from goalinsight.track_consolidation._sampler import sample_crops_for_tracks


def _load_json(p: Path):
    with open(p) as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--frames-per-track", type=int, default=5)
    ap.add_argument("--min-bbox-height", type=int, default=60)
    ap.add_argument("--upper-ratio", type=float, default=0.65)
    args = ap.parse_args()

    cons_dir = args.run_dir / "track_consolidation"
    tracking_dir = args.run_dir / "tracking"
    players_path = cons_dir / "players.json"
    tracks_path = tracking_dir / "tracks.json"
    pre_path = tracking_dir / "tracks.pre_consolidation.json"
    if not players_path.exists():
        raise SystemExit(f"missing {players_path}")
    if not tracks_path.exists():
        raise SystemExit(f"missing {tracks_path}")

    players = _load_json(players_path)
    raw_tracks = _load_json(tracks_path)

    # The tracker's integer track_ids referenced by players.json (1..N) only
    # survive in two places after consolidation runs:
    #   1) tracks.json's per-detection ``orig_track_id`` field, OR
    #   2) tracks.pre_consolidation.json (pre-rewrite snapshot, rare —
    #      backup is only taken once and gets locked to a stale state if
    #      consolidation has been re-run).
    # Rebuild a sampler-shaped {frame_id: [{track_id, bbox}, ...]} keyed by
    # those original tids so sample_crops_for_tracks can find them.
    tracks_by_frame: dict = {}
    has_orig = any(
        any("orig_track_id" in t for t in tracks)
        for tracks in raw_tracks.values()
    )
    if has_orig:
        print("Using tracks.json (orig_track_id field)")
        for fk, tracks in raw_tracks.items():
            rebuilt = []
            for t in tracks:
                ot = t.get("orig_track_id")
                if ot is None:
                    continue
                rebuilt.append({"track_id": int(ot), "bbox": t.get("bbox")})
            tracks_by_frame[fk] = rebuilt
    elif pre_path.exists():
        print(f"Using {pre_path}")
        tracks_by_frame = _load_json(pre_path)
    else:
        # Pre-consolidation tracks.json — track_id is already int
        print(f"Using {tracks_path}")
        tracks_by_frame = raw_tracks

    all_source_tids = sorted({
        int(t) for p in players for t in p.get("source_tracks", [])
    })

    # Per-track frame count + first/last frame for the report
    frame_count: dict[int, int] = {}
    first_frame: dict[int, int] = {}
    last_frame: dict[int, int] = {}
    for fk, tracks in tracks_by_frame.items():
        try:
            fid = int(fk)
        except (TypeError, ValueError):
            continue
        for t in tracks:
            tid = int(t["track_id"])
            if tid not in set(all_source_tids):
                continue
            frame_count[tid] = frame_count.get(tid, 0) + 1
            if tid not in first_frame or fid < first_frame[tid]:
                first_frame[tid] = fid
            if tid not in last_frame or fid > last_frame[tid]:
                last_frame[tid] = fid

    print(f"Sampling {args.frames_per_track} crops × {len(all_source_tids)} "
          f"source tracks ...", flush=True)
    sampled = sample_crops_for_tracks(
        video_path=args.video,
        tracks_by_frame=tracks_by_frame,
        track_ids=all_source_tids,
        k=args.frames_per_track,
        min_bbox_height=args.min_bbox_height,
        upper_ratio=args.upper_ratio,
    )

    thumbs_dir = cons_dir / "track_thumbs"
    thumbs_dir.mkdir(exist_ok=True)
    saved: dict[int, list[tuple[int, str]]] = {}
    for tid, frames in sampled.items():
        for sf in frames:
            fname = f"track_{tid:04d}_f{sf.frame_id:05d}.jpg"
            cv2.imwrite(str(thumbs_dir / fname), sf.crop)
            saved.setdefault(tid, []).append((sf.frame_id, fname))

    # --- Markdown ----------------------------------------------------
    lines: list[str] = []
    lines.append(f"# Consolidated tracks — `{args.run_dir.name}`\n")
    lines.append(
        f"- Source video: `{args.video}`\n"
        f"- Players (clusters): **{len(players)}**\n"
        f"- Source tracks merged: **{len(all_source_tids)}**\n"
        f"- Thumbnails: `track_consolidation/track_thumbs/`\n"
    )

    # Group by team for readability
    by_team: dict[str, list[dict]] = {}
    for p in players:
        by_team.setdefault(p.get("team", "unknown"), []).append(p)

    for team in sorted(by_team):
        lines.append(f"\n## Team: `{team}`\n")
        team_players = sorted(
            by_team[team],
            key=lambda p: (p.get("role", ""), p.get("player_id", "")),
        )
        for p in team_players:
            pid = p.get("player_id", "?")
            role = p.get("role", "player")
            jersey = p.get("jersey_number")
            jconf = p.get("jersey_confidence", 0.0)
            src = sorted(int(t) for t in p.get("source_tracks", []))

            jersey_str = f"#{jersey} (conf {jconf:.2f})" if jersey is not None \
                else "no jersey"
            lines.append(
                f"\n### {pid} — {role}, {jersey_str}\n"
                f"\nSource tracks ({len(src)}): {src}\n"
            )

            for tid in src:
                fc = frame_count.get(tid, 0)
                ff = first_frame.get(tid)
                lf = last_frame.get(tid)
                span = f"frames {ff}–{lf}" if ff is not None else "no frames"
                lines.append(
                    f"\n#### track {tid} — {fc} obs, {span}\n"
                )
                thumbs = saved.get(tid, [])
                if not thumbs:
                    lines.append("\n_(no crops sampled)_\n")
                    continue
                # Inline the K thumbnails as a row of small images
                for fid, fname in thumbs:
                    rel = f"track_thumbs/{fname}"
                    lines.append(
                        f'<img src="{rel}" alt="t{tid} f{fid}" height="120"> '
                    )
                lines.append("\n")

    md_path = cons_dir / "consolidated_tracks.md"
    md_path.write_text("".join(lines))
    print(f"wrote {md_path} ({len(players)} players, "
          f"{sum(len(v) for v in saved.values())} thumbs)")


if __name__ == "__main__":
    main()
