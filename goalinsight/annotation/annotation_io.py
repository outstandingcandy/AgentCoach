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

from .keypoint_utils import PITCH_KEYPOINTS, convert_keypoint_name
from .pitch.keypoints import PITCH_POINTS_TO_INTERSECTON


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
        "reprojection_error": float(data.get("reprojection_error", 0.0)),
        "H0": None,
    }

    for pt in data.get("points", []):
        pixel = tuple(pt["pixel"])
        raw_name = pt.get("keypoint_name") or pt.get("name", "")
        kp_name = convert_keypoint_name(raw_name)

        if "world" in pt:
            world = tuple(pt["world"])
        elif kp_name in PITCH_KEYPOINTS:
            world = PITCH_KEYPOINTS[kp_name]
        else:
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
        result["derived_points"].append((
            tuple(dp["pixel"]),
            tuple(dp["world"]),
            dp["name"],
        ))

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

        for pixel, world, name in derived_points:
            annotations_data["derived_points"].append({
                "pixel": [float(x) for x in pixel],
                "world": [float(x) for x in world],
                "name": name,
            })

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
                    pt_data = {
                        "pixel": [float(px), float(py)],
                        "world": [float(wx), float(wy)],
                        "keypoint_name": kp_name,
                        "hrnet_index": hrnet_idx,
                        "source": "manual",
                    }
                    all_points_data["manual_points"].append(pt_data)
                    all_points_data["all_points"].append(pt_data)

            for pixel, world, name in derived_points:
                pt_data = {
                    "pixel": [float(x) for x in pixel],
                    "world": [float(x) for x in world],
                    "keypoint_name": name,
                    "hrnet_index": -1,
                    "source": "derived",
                }
                all_points_data["derived_points"].append(pt_data)
                all_points_data["all_points"].append(pt_data)

            for pixel, world, name, hrnet_idx, is_ground in auto_projected_points:
                pt_data = {
                    "pixel": [float(pixel[0]), float(pixel[1])],
                    "world": [float(world[0]), float(world[1])],
                    "keypoint_name": name,
                    "hrnet_index": hrnet_idx,
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


def load_from_json(output_dir: str) -> tuple[int, np.ndarray | None, dict]:
    """Load a single legacy annotations.json (pre-index format)."""
    output_dir = Path(output_dir)
    result = {
        "clicked_points": [],
        "world_points": [],
        "keypoint_names": [],
        "annotated_lines": [],
        "derived_points": [],
        "reprojection_error": 0.0,
    }

    try:
        with open(output_dir / "annotations.json") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Annotation file not found: {output_dir / 'annotations.json'}")
        return 0, None, result
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in annotations file: {e}")
        return 0, None, result

    anchor_frame = data.get("anchor_frame_idx", 0)
    result["reprojection_error"] = float(data.get("reprojection_error", 0.0))

    for pt in data.get("points", []):
        pixel = tuple(pt["pixel"])
        raw_name = pt.get("keypoint_name") or pt.get("name", "")
        kp_name = convert_keypoint_name(raw_name)

        if "world" in pt:
            world = tuple(pt["world"])
        elif kp_name in PITCH_KEYPOINTS:
            world = PITCH_KEYPOINTS[kp_name]
        else:
            print(f"Warning: Unknown keypoint '{raw_name}', skipping")
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
        result["derived_points"].append((
            tuple(dp["pixel"]),
            tuple(dp["world"]),
            dp["name"],
        ))

    H0 = None
    h0_path = output_dir / "H0.npy"
    if h0_path.exists():
        try:
            H0 = np.load(h0_path)
        except Exception as e:
            print(f"Failed to load H0.npy: {e}")

    return anchor_frame, H0, result
