"""Intermediate result exporter for pipeline debugging."""

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .export_config import IntermediateExportConfig


class IntermediateExporter:
    """Export intermediate results from the pipeline."""

    def __init__(self, output_dir: Path | str, config: IntermediateExportConfig):
        """Initialize the exporter.

        Args:
            output_dir: Base output directory.
            config: Export configuration.
        """
        self.config = config
        self.base_dir = Path(output_dir) / config.output_subdir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.detection_dir = self.base_dir / "detections"
        self.reid_dir = self.base_dir / "reid"
        self.calibration_dir = self.base_dir / "calibration"
        self.tracking_dir = self.base_dir / "tracking"
        self.clustering_dir = self.base_dir / "clustering"
        self.visualization_dir = self.base_dir / "visualization"

        if config.detection.enabled:
            self.detection_dir.mkdir(exist_ok=True)
        if config.reid.enabled:
            self.reid_dir.mkdir(exist_ok=True)
            if config.reid.save_crops:
                (self.reid_dir / "crops").mkdir(exist_ok=True)
        if config.calibration.enabled:
            self.calibration_dir.mkdir(exist_ok=True)
        if config.tracking.enabled:
            self.tracking_dir.mkdir(exist_ok=True)
        if config.clustering.enabled:
            self.clustering_dir.mkdir(exist_ok=True)
        if config.visualization.enabled:
            self.visualization_dir.mkdir(exist_ok=True)

        self._all_detections: list[dict] = []
        self._all_tracks: list[dict] = []
        self._trajectories: dict[int, list[dict]] = {}

    def _should_export_frame(self, frame_idx: int, interval: int) -> bool:
        """Check if this frame should be exported."""
        if interval <= 0:
            return False
        return frame_idx % interval == 0

    def export_frame_detections(
        self,
        frame_idx: int,
        detections: list[dict[str, Any]],
        filtered_detections: list[dict[str, Any]] | None = None,
        calibration_quality: str = "unknown",
        num_keypoints: int = 0,
        filter_info: dict[str, Any] | None = None,
    ) -> None:
        """Export detection results for a frame."""
        cfg = self.config.detection
        if not cfg.enabled:
            return

        det_data = {
            "frame_idx": frame_idx,
            "num_detections": len(detections),
            "num_after_filter": len(filtered_detections) if filtered_detections is not None else len(detections),
            "filter_info": {
                "calibration_quality": calibration_quality,
                "num_keypoints": num_keypoints,
                **(filter_info or {}),
            },
            "detections": [],
        }

        for i, det in enumerate(detections):
            det_entry = {
                "index": i,
                "bbox": [float(x) for x in det["bbox"]],
                "confidence": float(det.get("confidence", 1.0)),
            }

            if filtered_detections is not None and cfg.include_filtered:
                was_filtered = not any(
                    self._iou(det["bbox"], fd["bbox"]) > 0.9
                    for fd in filtered_detections
                )
                det_entry["filtered_out"] = was_filtered
                if was_filtered:
                    det_entry["filter_reason"] = "off_pitch"

            if "pitch_position" in det:
                det_entry["pitch_position"] = det["pitch_position"]

            det_data["detections"].append(det_entry)

        self._all_detections.append(det_data)

        if cfg.save_json and self._should_export_frame(frame_idx, cfg.frame_interval):
            json_path = self.detection_dir / f"frame_{frame_idx:06d}.json"
            with open(json_path, "w") as f:
                json.dump(det_data, f, indent=2)

    def export_reid_results(
        self,
        frame_idx: int,
        crops: list[np.ndarray],
        embeddings: np.ndarray,
        roles: list[str],
        role_confidences: np.ndarray,
    ) -> None:
        """Export ReID results for a frame."""
        cfg = self.config.reid
        if not cfg.enabled:
            return

        if not self._should_export_frame(frame_idx, cfg.frame_interval):
            return

        if cfg.save_embeddings and embeddings is not None and len(embeddings) > 0:
            npz_path = self.reid_dir / f"embeddings_frame_{frame_idx:06d}.npz"
            np.savez_compressed(
                npz_path,
                embeddings=embeddings,
                roles=np.array(roles),
                role_confidences=role_confidences,
            )

        if cfg.save_crops and crops:
            frame_crop_dir = self.reid_dir / "crops" / f"frame_{frame_idx:06d}"
            frame_crop_dir.mkdir(exist_ok=True)

            for i, (crop, role) in enumerate(zip(crops, roles)):
                if crop is not None and crop.size > 0:
                    crop_path = frame_crop_dir / f"crop_{i:03d}_{role}.jpg"
                    cv2.imwrite(str(crop_path), crop)

        if cfg.save_roles:
            roles_data = {
                "frame_idx": frame_idx,
                "num_detections": len(roles),
                "roles": [
                    {"index": i, "role": role, "confidence": float(conf)}
                    for i, (role, conf) in enumerate(zip(roles, role_confidences))
                ],
            }
            json_path = self.reid_dir / f"roles_frame_{frame_idx:06d}.json"
            with open(json_path, "w") as f:
                json.dump(roles_data, f, indent=2)

    def export_calibration_result(
        self,
        frame_idx: int,
        keypoints: dict,
        lines: dict,
        homography: np.ndarray | None,
        image_width: int,
        image_height: int,
    ) -> None:
        """Export calibration results for a frame."""
        cfg = self.config.calibration
        if not cfg.enabled:
            return

        if not self._should_export_frame(frame_idx, cfg.frame_interval):
            return

        calib_data = {
            "frame_idx": frame_idx,
            "image_size": [image_width, image_height],
            "num_keypoints": len(keypoints),
            "num_lines": len(lines),
            "has_homography": homography is not None,
            "keypoints": {},
            "lines": {},
        }

        for kp_id, kp in keypoints.items():
            calib_data["keypoints"][str(kp_id)] = {
                "x_normalized": kp["x"],
                "y_normalized": kp["y"],
                "x_pixel": int(kp["x"] * image_width),
                "y_pixel": int(kp["y"] * image_height),
                "confidence": kp.get("p", 1.0),
            }

        for line_name, points in lines.items():
            calib_data["lines"][line_name] = [
                {
                    "x_normalized": pt["x"],
                    "y_normalized": pt["y"],
                    "x_pixel": int(pt["x"] * image_width),
                    "y_pixel": int(pt["y"] * image_height),
                }
                for pt in points
            ]

        if cfg.save_json:
            json_path = self.calibration_dir / f"frame_{frame_idx:06d}.json"
            with open(json_path, "w") as f:
                json.dump(calib_data, f, indent=2)

        if cfg.save_homography and homography is not None:
            npy_path = self.calibration_dir / f"homography_{frame_idx:06d}.npy"
            np.save(npy_path, homography)

    def export_frame_tracks(
        self,
        frame_idx: int,
        tracks: list[dict[str, Any]],
    ) -> None:
        """Export tracking results for a frame."""
        cfg = self.config.tracking
        if not cfg.enabled:
            return

        track_data = {
            "frame_idx": frame_idx,
            "num_tracks": len(tracks),
            "tracks": [],
        }

        for track in tracks:
            track_entry = {
                "track_id": track["track_id"],
                "bbox": [float(x) for x in track["bbox"]],
            }

            if "role" in track:
                track_entry["role"] = track["role"]
            if "role_confidence" in track:
                track_entry["role_confidence"] = float(track["role_confidence"])
            if "team" in track:
                track_entry["team"] = track["team"]
            if "jersey_number" in track:
                track_entry["jersey_number"] = track["jersey_number"]

            track_data["tracks"].append(track_entry)

            if cfg.save_trajectories:
                tid = track["track_id"]
                if tid not in self._trajectories:
                    self._trajectories[tid] = []
                self._trajectories[tid].append({
                    "frame_idx": frame_idx,
                    "bbox": [float(x) for x in track["bbox"]],
                    "center": [
                        (track["bbox"][0] + track["bbox"][2]) / 2,
                        (track["bbox"][1] + track["bbox"][3]) / 2,
                    ],
                })

        self._all_tracks.append(track_data)

        if cfg.save_json and self._should_export_frame(frame_idx, cfg.frame_interval):
            json_path = self.tracking_dir / f"frame_{frame_idx:06d}.json"
            with open(json_path, "w") as f:
                json.dump(track_data, f, indent=2)

    def export_clustering_results(
        self,
        team_clusters: dict[int, int],
        team_sides: dict[int, str],
        majority_roles: dict[int, str],
        mean_embeddings: dict[int, np.ndarray] | None = None,
    ) -> None:
        """Export clustering results."""
        cfg = self.config.clustering
        if not cfg.enabled:
            return

        cluster_counts: dict[str, int] = {}
        team_counts: dict[str, int] = {"left": 0, "right": 0, "referee": 0, "unknown": 0}

        for tid, cluster in team_clusters.items():
            key = f"cluster_{cluster}"
            cluster_counts[key] = cluster_counts.get(key, 0) + 1

        for tid, side in team_sides.items():
            team_counts[side] = team_counts.get(side, 0) + 1

        summary = {
            "num_tracks": len(team_sides),
            "cluster_counts": cluster_counts,
            "team_counts": team_counts,
            "tracks": {},
        }

        for tid in team_sides:
            summary["tracks"][str(tid)] = {
                "cluster": team_clusters.get(tid, -1),
                "team_side": team_sides.get(tid, "unknown"),
                "role": majority_roles.get(tid, "player"),
            }

        if cfg.save_summary:
            json_path = self.clustering_dir / "clustering_summary.json"
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)

        if cfg.save_embeddings and mean_embeddings:
            track_ids = list(mean_embeddings.keys())
            embeddings_array = np.array([mean_embeddings[tid] for tid in track_ids])
            clusters_array = np.array([team_clusters.get(tid, -1) for tid in track_ids])
            sides_array = np.array([team_sides.get(tid, "unknown") for tid in track_ids])

            npz_path = self.clustering_dir / "cluster_embeddings.npz"
            np.savez_compressed(
                npz_path,
                track_ids=np.array(track_ids),
                embeddings=embeddings_array,
                clusters=clusters_array,
                team_sides=sides_array,
            )

    def save_visualization_frame(
        self,
        frame_idx: int,
        vis_type: str,
        frame: np.ndarray,
    ) -> None:
        """Save a visualization frame."""
        cfg = self.config.visualization
        if not cfg.enabled:
            return

        if not self._should_export_frame(frame_idx, cfg.frame_interval):
            return

        if vis_type == "detection" and not cfg.save_detection_vis:
            return
        if vis_type == "calibration" and not cfg.save_calibration_vis:
            return

        vis_path = self.visualization_dir / f"{vis_type}_frame_{frame_idx:06d}.jpg"
        cv2.imwrite(str(vis_path), frame)

    def finalize(self) -> None:
        """Finalize export by saving accumulated data."""
        if self.config.detection.enabled and self.config.detection.save_summary:
            summary_path = self.detection_dir / "all_frames.json"
            with open(summary_path, "w") as f:
                json.dump(self._all_detections, f)

        if self.config.tracking.enabled and self.config.tracking.save_trajectories:
            traj_path = self.tracking_dir / "trajectories.json"
            trajectories_json = {str(k): v for k, v in self._trajectories.items()}
            with open(traj_path, "w") as f:
                json.dump(trajectories_json, f, indent=2)

        if self.config.reid.enabled and self.config.reid.save_embeddings:
            summary_path = self.reid_dir / "export_summary.json"
            summary = {
                "num_frames_exported": len(self._all_detections),
                "output_directory": str(self.reid_dir),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

        print(f"  Intermediate results saved to {self.base_dir}")

    def _iou(self, box1: list[float], box2: list[float]) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
