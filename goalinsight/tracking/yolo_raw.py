"""Raw YOLO detection dump + diagnostic visualisation.

Captures the per-frame state of detections at every filter stage between
YOLO and the tracker, so dropouts (conf/size/pitch) can be inspected
offline. Companion to ``ball_pipeline._render_ball_detection_diag`` —
same threading shape (reader → annotate → writer pool) and same output
structure (per-frame JPGs in ``<output>/yolo_raw/frames``).

Stages tracked per detection:
  - ``raw``: every YOLO output for class 0 (player), all confidences
  - ``post_conf``: passed ``player_confidence_threshold``
  - ``post_size``: passed ``filter_by_size`` (height + aspect ratio)
  - ``post_pitch``: passed ``_filter_by_pitch_undistorted`` (in-bounds)

A detection that drops at stage X gets coloured by X in the visual:
  red = dropped by conf, orange = dropped by size, yellow = dropped by
  pitch, green = passed all four. Ball detections are drawn cyan.

Public API:
  - :func:`dump_yolo_raw_json`: write per-frame JSON records + summary.
  - :func:`render_yolo_raw_diag`: write per-frame annotated JPGs.

Records are produced by the orchestrator at two hook points (fused
detection pass + filter loop), keyed by frame index, schema described
in :class:`YoloRawRecord`.
"""

from __future__ import annotations

import json
import queue as _queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# Drawing colours (BGR).
_COLOR_PASS = (0, 220, 0)         # green — passed all four stages
_COLOR_DROP_PITCH = (0, 220, 220) # yellow — dropped by pitch filter
_COLOR_DROP_SIZE = (0, 140, 255)  # orange — dropped by size filter
_COLOR_DROP_CONF = (60, 60, 220)  # red — dropped by conf filter
_COLOR_BALL = (255, 200, 0)       # cyan — ball detection

_LEGEND = [
    ("passed all", _COLOR_PASS),
    ("dropped: pitch", _COLOR_DROP_PITCH),
    ("dropped: size", _COLOR_DROP_SIZE),
    ("dropped: conf", _COLOR_DROP_CONF),
    ("ball", _COLOR_BALL),
]


@dataclass
class YoloRawRecord:
    """Per-frame raw YOLO snapshot.

    All four stages refer to PLAYER detections (class 0). Ball detections
    are stored separately and not subject to the player confidence /
    pitch filters.

    ``post_*`` lists shadow ``raw`` — they're flat lists of detection
    dicts (same schema as :class:`UnifiedDetector` output: ``bbox``,
    ``confidence``, ``class``, ``class_name``). Drop-stage is computed
    by set-difference at write time.
    """
    frame_index: int
    image_size: tuple[int, int] = (0, 0)  # (W, H)
    has_camera_pose: bool = False
    players_raw: list[dict] = field(default_factory=list)
    players_post_conf: list[dict] = field(default_factory=list)
    players_post_size: list[dict] = field(default_factory=list)
    players_post_pitch: list[dict] = field(default_factory=list)
    balls_raw: list[dict] = field(default_factory=list)


def _slim(d: dict) -> dict:
    """Float32 → rounded floats; drop bulky fields. JSON-friendly."""
    out = {
        "bbox": [round(float(x), 2) for x in d["bbox"]],
        "confidence": round(float(d["confidence"]), 4),
    }
    if "class" in d:
        out["class"] = int(d["class"])
    if "class_name" in d:
        out["class_name"] = d["class_name"]
    pp = d.get("pitch_position")
    if pp is not None:
        out["pitch_position"] = [round(float(pp[0]), 4), round(float(pp[1]), 4)]
    return out


def _bbox_id(det: dict) -> tuple[float, float, float, float, float]:
    """Identity tuple for set-membership across filter stages.

    Detections are dicts (mutable, no __hash__), and downstream filters
    return references to the SAME dict object — but we'd rather not rely
    on object identity (a future caller could rebuild dicts mid-pipeline
    and silently break this). Round to 0.01 px so the bbox+conf identity
    is stable across float32 roundtrips.
    """
    bb = det["bbox"]
    return (
        round(float(bb[0]), 2), round(float(bb[1]), 2),
        round(float(bb[2]), 2), round(float(bb[3]), 2),
        round(float(det["confidence"]), 4),
    )


def dump_yolo_raw_json(
    records: dict[int, YoloRawRecord],
    output_dir: Path | str,
    extra_summary: dict | None = None,
) -> None:
    """Write per-frame JSON + summary.json under ``<output_dir>/yolo_raw/``.

    One file per frame: ``frames/frame_<NNNNNN>.json`` with all four
    player stages, ball detections, image size, pose flag.
    Plus ``summary.json`` with counts per frame and any caller-supplied
    metadata (config snapshot, threshold values).
    """
    out_dir = Path(output_dir) / "yolo_raw"
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for fidx in sorted(records):
        rec = records[fidx]
        payload = {
            "frame_index": rec.frame_index,
            "image_size": list(rec.image_size),
            "has_camera_pose": rec.has_camera_pose,
            "counts": {
                "raw_players": len(rec.players_raw),
                "raw_balls": len(rec.balls_raw),
                "post_conf_players": len(rec.players_post_conf),
                "post_size_players": len(rec.players_post_size),
                "post_pitch_players": len(rec.players_post_pitch),
            },
            "players_raw": [_slim(d) for d in rec.players_raw],
            "players_post_conf": [_slim(d) for d in rec.players_post_conf],
            "players_post_size": [_slim(d) for d in rec.players_post_size],
            "players_post_pitch": [_slim(d) for d in rec.players_post_pitch],
            "balls_raw": [_slim(d) for d in rec.balls_raw],
        }
        with open(frames_dir / f"frame_{fidx:06d}.json", "w") as f:
            json.dump(payload, f, indent=1)
        summary_rows.append({
            "frame_index": fidx,
            **payload["counts"],
            "has_camera_pose": rec.has_camera_pose,
        })

    summary = {
        "frames": summary_rows,
        "totals": {
            "n_frames": len(summary_rows),
            "raw_players": sum(r["raw_players"] for r in summary_rows),
            "raw_balls": sum(r["raw_balls"] for r in summary_rows),
            "post_conf_players": sum(r["post_conf_players"] for r in summary_rows),
            "post_size_players": sum(r["post_size_players"] for r in summary_rows),
            "post_pitch_players": sum(r["post_pitch_players"] for r in summary_rows),
        },
    }
    if extra_summary:
        summary.update(extra_summary)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)


def render_yolo_raw_diag(
    video_path: Path | str,
    records: dict[int, YoloRawRecord],
    output_dir: Path | str,
) -> None:
    """Render per-frame annotated JPGs showing every YOLO detection,
    coloured by which filter stage dropped it.

    Output: ``<output_dir>/yolo_raw/frames/frame_<NNNNNN>.jpg``.
    Uses a reader → annotate → writer-pool pipeline (mirrors
    :func:`_render_ball_detection_diag`) so disk I/O doesn't block.
    """
    if not records:
        return

    diag_dir = Path(output_dir) / "yolo_raw" / "frames"
    diag_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = sorted(records)

    print("\nGenerating raw YOLO diagnostic visualization...")
    totals = {"raw": 0, "passed": 0, "drop_conf": 0, "drop_size": 0, "drop_pitch": 0, "balls": 0}
    for rec in records.values():
        passed = {_bbox_id(d) for d in rec.players_post_pitch}
        post_size = {_bbox_id(d) for d in rec.players_post_size}
        post_conf = {_bbox_id(d) for d in rec.players_post_conf}
        for d in rec.players_raw:
            bid = _bbox_id(d)
            totals["raw"] += 1
            if bid in passed:
                totals["passed"] += 1
            elif bid in post_size:
                totals["drop_pitch"] += 1
            elif bid in post_conf:
                totals["drop_size"] += 1
            else:
                totals["drop_conf"] += 1
        totals["balls"] += len(rec.balls_raw)
    print(f"  totals: {'  '.join(f'{k}={v}' for k, v in totals.items())}")

    # Stage 1: reader thread
    read_q: _queue.Queue = _queue.Queue(maxsize=8)

    def _reader():
        cap = cv2.VideoCapture(str(video_path))
        prev = -1
        for fidx in frame_indices:
            gap = fidx - prev - 1
            if prev >= 0 and 0 <= gap <= 8:
                for _ in range(gap):
                    cap.grab()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ok, frame = cap.read()
            prev = fidx
            if ok:
                read_q.put((fidx, frame))
        read_q.put(None)
        cap.release()

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()

    # Stage 3: writer pool
    writer_pool = ThreadPoolExecutor(max_workers=4)
    write_futures: list = []

    # Stage 2: annotate
    for _ in tqdm(frame_indices, desc="  YOLO diag"):
        item = read_q.get()
        if item is None:
            break
        fidx, frame = item
        rec = records[fidx]

        passed_ids = {_bbox_id(d) for d in rec.players_post_pitch}
        post_size_ids = {_bbox_id(d) for d in rec.players_post_size}
        post_conf_ids = {_bbox_id(d) for d in rec.players_post_conf}

        # Players: classify each raw det into a drop stage. All bboxes
        # share the same line thickness (3 px on 4K; on a 1080p browser
        # render that's ~1 px which is too faint to see) so dropped
        # detections aren't lost in the JPG when the image is downscaled.
        font_scale = max(0.6, frame.shape[1] / 3000.0)
        thickness = max(2, int(round(frame.shape[1] / 1000.0)))
        for d in rec.players_raw:
            bid = _bbox_id(d)
            if bid in passed_ids:
                color = _COLOR_PASS
                label = "OK"
            elif bid in post_size_ids:
                color = _COLOR_DROP_PITCH
                label = "drop:pitch"
            elif bid in post_conf_ids:
                color = _COLOR_DROP_SIZE
                label = "drop:size"
            else:
                color = _COLOR_DROP_CONF
                label = "drop:conf"
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            text = f"{label} {d['confidence']:.2f}"
            ty = max(y1 - 6, 12)
            cv2.putText(frame, text, (x1, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        max(1, thickness - 1), cv2.LINE_AA)

        # Balls
        for d in rec.balls_raw:
            x1, y1, x2, y2 = map(int, d["bbox"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), _COLOR_BALL, 2)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(frame, (cx, cy), 8, _COLOR_BALL, 2)
            text = f"ball {d['confidence']:.2f}"
            cv2.putText(frame, text, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _COLOR_BALL, 1, cv2.LINE_AA)

        # Header bar
        n_raw = len(rec.players_raw)
        n_pass = len(rec.players_post_pitch)
        info = (f"Frame {fidx}  |  raw={n_raw}  conf={len(rec.players_post_conf)}  "
                f"size={len(rec.players_post_size)}  pitch={n_pass}  "
                f"balls={len(rec.balls_raw)}  pose={'Y' if rec.has_camera_pose else 'N'}")
        cv2.rectangle(frame, (0, 0), (len(info) * 9 + 10, 28), (0, 0, 0), -1)
        cv2.putText(frame, info, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Legend
        h = frame.shape[0]
        legend_h = len(_LEGEND) * 22 + 8
        cv2.rectangle(frame, (0, h - legend_h), (200, h), (0, 0, 0), -1)
        for i, (label, color) in enumerate(_LEGEND):
            ly = h - legend_h + 18 + i * 22
            cv2.circle(frame, (12, ly - 4), 6, color, -1)
            cv2.putText(frame, label, (26, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        out_path = str(diag_dir / f"frame_{fidx:06d}.jpg")
        write_futures.append(writer_pool.submit(cv2.imwrite, out_path, frame))

    reader.join()
    for fut in write_futures:
        fut.result()
    writer_pool.shutdown(wait=False)
