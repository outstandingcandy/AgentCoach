"""Audit ID dropouts and switches across the whole tracking run.

Joins tracks.json (post-tracker output) with yolo_raw frames JSON to find:
  - dropouts: a tid present at frame f, missing at f+stride, then present
    again at f+2*stride within ~30 frames; AND a YOLO post_pitch
    detection in the gap frame within pitch_gate (4 m) of the
    interpolated tid position. I.e. tracker had a usable detection but
    didn't keep the tid.
  - id_switches: a tid whose successive bbox jumps far in pitch space
    (> 5 m within one sample step) — indicates the tid silently flipped
    to a different physical entity.

Outputs a single audit JSON + a terse stdout summary.

Run:
  python scripts/audit_track_dropouts.py output/kids_prtreid_iou_fix
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def load_tracks(tracks_path: Path) -> dict[int, list[dict]]:
    with open(tracks_path) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_yolo_dir(yolo_dir: Path) -> dict[int, dict]:
    out = {}
    for jf in sorted(yolo_dir.glob("frame_*.json")):
        with open(jf) as f:
            d = json.load(f)
        out[int(d["frame_index"])] = d
    return out


def pitch_dist(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def audit(run_dir: Path, gap_max: int = 30, pitch_gate: float = 4.0,
          switch_thresh_m: float = 5.0) -> dict:
    tracks_path = run_dir / "tracking/tracks.json"
    if not tracks_path.exists():
        # Maybe a track_consolidation run; prefer pre-consolidation
        pre = run_dir / "tracking/tracks.pre_consolidation.json"
        if pre.exists():
            tracks_path = pre
    yolo_dir = run_dir / "tracking/yolo_raw/frames"
    if not (tracks_path.exists() and yolo_dir.exists()):
        raise SystemExit(
            f"missing tracks.json or yolo_raw/ under {run_dir}\n"
            f"  tracks: {tracks_path.exists()}\n"
            f"  yolo:   {yolo_dir.exists()}"
        )

    tracks = load_tracks(tracks_path)
    yolo = load_yolo_dir(yolo_dir)

    frames = sorted(tracks)
    if not frames:
        raise SystemExit("empty tracks")
    stride = frames[1] - frames[0] if len(frames) > 1 else 1

    # tid -> list of (frame, bbox, pitch_pos)
    tid_history: dict = defaultdict(list)
    for f in frames:
        for det in tracks[f]:
            tid = det.get("track_id")
            pp = det.get("pitch_position")
            tid_history[tid].append((f, det.get("bbox"), pp))
    for tid in tid_history:
        tid_history[tid].sort(key=lambda r: r[0])

    # ---- Find ID switches: same tid, big pitch jump between adjacent samples
    switches = []
    for tid, hist in tid_history.items():
        for i in range(1, len(hist)):
            f0, _, p0 = hist[i - 1]
            f1, _, p1 = hist[i]
            if p0 is None or p1 is None:
                continue
            if f1 - f0 > stride * 3:  # too far apart, not adjacent
                continue
            d = pitch_dist(p0, p1)
            if d >= switch_thresh_m:
                switches.append({
                    "tid": tid,
                    "frame_prev": f0,
                    "frame_next": f1,
                    "gap_samples": (f1 - f0) // stride,
                    "pitch_prev": p0,
                    "pitch_next": p1,
                    "jump_m": round(d, 2),
                })
    switches.sort(key=lambda r: -r["jump_m"])

    # ---- Find dropouts: tid present at f, missing at f+stride, present again
    # within `gap_max` frames; AND a YOLO post_pitch det in the gap frame
    # within pitch_gate of the interpolated position.
    dropouts = []
    for tid, hist in tid_history.items():
        if len(hist) < 2:
            continue
        for i in range(1, len(hist)):
            f0, _, p0 = hist[i - 1]
            f1, _, p1 = hist[i]
            if p0 is None or p1 is None:
                continue
            gap = f1 - f0
            if gap <= stride or gap > gap_max:
                continue
            # interpolate position into each gap frame, look for nearby YOLO det
            for f_gap in range(f0 + stride, f1, stride):
                t = (f_gap - f0) / (f1 - f0)
                pi = (p0[0] + t * (p1[0] - p0[0]),
                      p0[1] + t * (p1[1] - p0[1]))
                yrec = yolo.get(f_gap)
                if not yrec:
                    continue
                best = None
                for det in yrec.get("players_post_pitch", []):
                    pp = det.get("pitch_position")
                    if pp is None:
                        continue
                    d = pitch_dist(pi, pp)
                    if best is None or d < best[1]:
                        best = (det, d)
                if best is not None and best[1] < pitch_gate:
                    dropouts.append({
                        "tid": tid,
                        "frame_before": f0,
                        "frame_gap": f_gap,
                        "frame_after": f1,
                        "gap_samples": gap // stride,
                        "interp_pitch": [round(pi[0], 2), round(pi[1], 2)],
                        "candidate_bbox": [round(x, 1) for x in best[0]["bbox"]],
                        "candidate_conf": best[0]["confidence"],
                        "candidate_pitch": best[0].get("pitch_position"),
                        "candidate_pitch_d": round(best[1], 2),
                    })
    dropouts.sort(key=lambda r: (r["tid"], r["frame_gap"]))

    # Tid lifetimes for context
    tid_summary = {}
    for tid, hist in tid_history.items():
        edge_count = 0
        PITCH_HALF_L = 33.14
        PITCH_HALF_W = 21.575
        MARGIN = 3.0
        for f, _, p in hist:
            if p is None:
                edge_count += 1
                continue
            if abs(p[0]) > PITCH_HALF_L - MARGIN or abs(p[1]) > PITCH_HALF_W - MARGIN:
                edge_count += 1
        tid_summary[str(tid)] = {
            "first": hist[0][0],
            "last": hist[-1][0],
            "n": len(hist),
            "edge_frac": round(edge_count / len(hist), 3),
        }

    return {
        "tracks_path": str(tracks_path),
        "yolo_dir": str(yolo_dir),
        "stride": stride,
        "n_frames": len(frames),
        "n_tids": len(tid_history),
        "n_switches": len(switches),
        "n_dropouts": len(dropouts),
        "switches": switches,
        "dropouts": dropouts,
        "tid_summary": tid_summary,
    }


def print_summary(res: dict):
    print(f"tracks: {res['tracks_path']}")
    print(f"yolo:   {res['yolo_dir']}")
    print(f"frames: {res['n_frames']} (stride {res['stride']})")
    print(f"tids:   {res['n_tids']}")
    print()
    sw = res["switches"]
    do = res["dropouts"]
    print(f"=== {len(sw)} ID switch(es) (>= 5 m pitch jump in adjacent samples) ===")
    for r in sw[:30]:
        print(f"  tid={r['tid']:>4}  frame {r['frame_prev']:>4}->{r['frame_next']:<4}  "
              f"jump={r['jump_m']:.2f} m  pitch {r['pitch_prev']} -> {r['pitch_next']}")
    if len(sw) > 30:
        print(f"  ... +{len(sw)-30} more")

    print()
    print(f"=== {len(do)} dropout(s) (tid coast with usable YOLO det in gap, < 4 m) ===")
    for r in do[:50]:
        edge = res['tid_summary'].get(str(r['tid']), {}).get('edge_frac', '?')
        print(f"  tid={r['tid']:>4}  gap@frame {r['frame_gap']:>4}  "
              f"({r['frame_before']}->{r['frame_after']}, {r['gap_samples']} samples)  "
              f"pitch_d={r['candidate_pitch_d']:.2f} m  conf={r['candidate_conf']:.2f}  "
              f"bbox={r['candidate_bbox']}  edge_frac={edge}")
    if len(do) > 50:
        print(f"  ... +{len(do)-50} more")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output run dir, e.g. output/kids_prtreid_iou_fix")
    ap.add_argument("--out", default=None,
                    help="write full audit JSON here (default: <run_dir>/tracking/track_audit.json)")
    ap.add_argument("--gap-max", type=int, default=30)
    ap.add_argument("--pitch-gate", type=float, default=4.0)
    ap.add_argument("--switch-thresh", type=float, default=5.0)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    res = audit(run_dir, gap_max=args.gap_max,
                pitch_gate=args.pitch_gate,
                switch_thresh_m=args.switch_thresh)

    out_path = Path(args.out) if args.out else (run_dir / "tracking/track_audit.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print_summary(res)
    print()
    print(f"full audit -> {out_path}")


if __name__ == "__main__":
    main()
