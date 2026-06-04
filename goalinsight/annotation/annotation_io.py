"""Load/save per-frame annotation files.

Writes three artifacts per saved frame:
- frame_<idx>.json: manual points, lines, derived intersections, reprojection
  error. Reference for human review.
- frame_<idx>_all_points.json: merged manual + derived + auto-projected points
  in the format consumed by goalinsight.field_registration.pnlcalib.finetune
  (HRNetToPnLCalibMapper reads `all_points[*].{pixel, world}`).
- frame_<idx>_raw.jpg: the original frame (input to finetune dataloader).
- frame_<idx>.jpg: visualization with overlays.
- frame_<idx>.npy: H0 image->world matrix (optional; not used by finetune).
"""

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .pitch import keypoints as _pk
from .pitch.keypoints import (
    PITCH_POINTS_TO_INTERSECTON,
    pitch_name_to_pnlcalib_id,
)


def load_frame_annotation(frame_dir: Path, frame_idx: int) -> dict | None:
    json_path = frame_dir / f"frame_{frame_idx}.json"
    if not json_path.exists():
        return None

    try:
        with open(json_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError):
        return None

    result = {
        "clicked_points": [],
        "world_points": [],
        "keypoint_names": [],
        "annotated_lines": [],
        "derived_points": [],
        # Saved derived points carry their accepted state so reload can
        # match against newly-recomputed intersections by world coord
        # (lines may have been re-resolved to a different active pitch).
        # List of (world_xy, accepted) pairs — index-aligned with
        # ``derived_points`` in this result dict.
        "derived_saved_accepted": [],
        "reprojection_error": float(data.get("reprojection_error", 0.0)),
        "H0": None,
    }

    for pt in data.get("points", []):
        pixel = tuple(pt["pixel"])
        kp_name = pt.get("keypoint_name") or pt.get("name", "")

        if "world" in pt:
            world = tuple(pt["world"])
        elif kp_name in _pk.PITCH_POINTS:
            wp = _pk.PITCH_POINTS[kp_name]
            world = (float(wp[0]), float(wp[1]))
        else:
            print(f"Skipping unknown keypoint '{kp_name}' in {json_path}")
            continue

        result["clicked_points"].append(pixel)
        result["world_points"].append(world)
        result["keypoint_names"].append(kp_name)

    for line in data.get("lines", []):
        result["annotated_lines"].append({
            "pixels": [tuple(line["pixels"][0]), tuple(line["pixels"][1])],
            "world": (tuple(line["world"][0]), tuple(line["world"][1])),
            "name": line["name"],
        })

    for dp in data.get("derived_points", []):
        world = tuple(dp["world"])
        result["derived_points"].append((
            tuple(dp["pixel"]),
            world,
            dp["name"],
        ))
        # Legacy default: pre-fix saves only persisted accepted points,
        # so a missing field means "this point was accepted at save time".
        accepted = bool(dp.get("accepted", True))
        result["derived_saved_accepted"].append((world, accepted))

    h0_path = frame_dir / f"frame_{frame_idx}.npy"
    if h0_path.exists():
        try:
            result["H0"] = np.load(h0_path)
        except Exception:
            pass

    return result


def save_frame_annotation(
    frame_dir: Path,
    frame_idx: int,
    video_path: str,
    video_name: str,
    clicked_points: list[tuple[float, float]],
    world_points: list[tuple[float, float]],
    keypoint_names: list[str],
    annotated_lines: list[dict],
    derived_points: list[tuple],
    reprojection_error: float,
    H0: np.ndarray | None,
    current_frame: np.ndarray | None,
    vis_frame: np.ndarray | None = None,
    auto_projected_points: list[tuple] | None = None,
    derived_accepted: list[bool] | None = None,
) -> bool:
    """Save annotation artifacts for a frame.

    `current_frame` is BGR (OpenCV). `vis_frame` is RGB.
    """
    try:
        frame_dir.mkdir(parents=True, exist_ok=True)

        annotations_data = {
            "frame_idx": frame_idx,
            "video_path": video_path,
            "video_name": video_name,
            "num_manual_points": len(keypoint_names),
            "num_lines": len(annotated_lines),
            "num_derived_points": len(derived_points),
            "reprojection_error": float(reprojection_error),
            "points": [],
            "lines": [],
            "derived_points": [],
            "saved_at": datetime.now().isoformat(),
        }

        for i, (px, py) in enumerate(clicked_points):
            if i < len(keypoint_names):
                wx, wy = world_points[i]
                kp_name = keypoint_names[i]
                hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
                annotations_data["points"].append({
                    "pixel": [float(px), float(py)],
                    "world": [float(wx), float(wy)],
                    "keypoint_name": kp_name,
                    "hrnet_index": hrnet_idx,
                })

        for line in annotated_lines:
            annotations_data["lines"].append({
                "pixels": [
                    [float(x) for x in line["pixels"][0]],
                    [float(x) for x in line["pixels"][1]],
                ],
                "world": [
                    [float(x) for x in line["world"][0]],
                    [float(x) for x in line["world"][1]],
                ],
                "name": line["name"],
            })

        for i, (pixel, world, name) in enumerate(derived_points):
            entry = {
                "pixel": [float(x) for x in pixel],
                "world": [float(x) for x in world],
                "name": name,
            }
            # Persist accepted state per point. Defaults to True for callers
            # that filtered to accepted-only before passing the list (legacy
            # behavior — the data they handed us was implicitly accepted).
            if derived_accepted is not None and i < len(derived_accepted):
                entry["accepted"] = bool(derived_accepted[i])
            else:
                entry["accepted"] = True
            annotations_data["derived_points"].append(entry)

        with open(frame_dir / f"frame_{frame_idx}.json", "w") as f:
            json.dump(annotations_data, f, indent=2)

        if H0 is not None:
            np.save(frame_dir / f"frame_{frame_idx}.npy", H0)

        if current_frame is not None:
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx}_raw.jpg"), current_frame)

        if vis_frame is not None:
            vis_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_dir / f"frame_{frame_idx}.jpg"), vis_bgr)

        if auto_projected_points is not None:
            all_points_data = {
                "frame_idx": frame_idx,
                "video_name": video_name,
                "reprojection_error": float(reprojection_error),
                "saved_at": datetime.now().isoformat(),
                "manual_points": [],
                "derived_points": [],
                "auto_projected_points": [],
                "all_points": [],
            }

            for i, (px, py) in enumerate(clicked_points):
                if i < len(keypoint_names):
                    wx, wy = world_points[i]
                    kp_name = keypoint_names[i]
                    hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
                    pnl_id = pitch_name_to_pnlcalib_id(kp_name)
                    pt_data = {
                        "pixel": [float(px), float(py)],
                        "world": [float(wx), float(wy)],
                        "keypoint_name": kp_name,
                        "hrnet_index": hrnet_idx,
                        "pnlcalib_id": pnl_id,
                        "source": "manual",
                    }
                    all_points_data["manual_points"].append(pt_data)
                    all_points_data["all_points"].append(pt_data)

            for i, (pixel, world, name) in enumerate(derived_points):
                # Only accepted derived points feed finetune. When the
                # caller didn't supply a flag list, treat every entry as
                # accepted (legacy behavior — they'd already filtered).
                if (derived_accepted is not None
                        and i < len(derived_accepted)
                        and not derived_accepted[i]):
                    continue
                pnl_id = pitch_name_to_pnlcalib_id(name)
                pt_data = {
                    "pixel": [float(x) for x in pixel],
                    "world": [float(x) for x in world],
                    "keypoint_name": name,
                    "hrnet_index": -1,
                    "pnlcalib_id": pnl_id,
                    "source": "derived",
                }
                all_points_data["derived_points"].append(pt_data)
                all_points_data["all_points"].append(pt_data)

            for pixel, world, name, hrnet_idx, is_ground in auto_projected_points:
                pnl_id = pitch_name_to_pnlcalib_id(name)
                pt_data = {
                    "pixel": [float(pixel[0]), float(pixel[1])],
                    "world": [float(world[0]), float(world[1])],
                    "keypoint_name": name,
                    "hrnet_index": hrnet_idx,
                    "pnlcalib_id": pnl_id,
                    "is_ground_plane": is_ground,
                    "source": "auto_projected",
                }
                all_points_data["auto_projected_points"].append(pt_data)
                all_points_data["all_points"].append(pt_data)

            all_points_data["total_points"] = len(all_points_data["all_points"])

            with open(frame_dir / f"frame_{frame_idx}_all_points.json", "w") as f:
                json.dump(all_points_data, f, indent=2)

        return True
    except Exception as e:
        print(f"Save error: {e}")
        return False
