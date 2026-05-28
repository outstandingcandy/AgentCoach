"""Unified annotation index across videos.

Storage structure:
    annotations_dir/
    +-- index.json                 # Index metadata
    +-- clip_000/
    |   +-- frame_700.json         # Frame annotation data
    |   +-- frame_700.npy          # H0 matrix
    |   +-- frame_700_raw.jpg      # Raw frame (used by finetune dataloader)
    |   +-- frame_700_all_points.json  # All points (for finetune)
    |   +-- frame_700.jpg          # Visualization with overlays
    +-- ...
"""

import json
from datetime import datetime
from pathlib import Path


class AnnotationIndex:
    def __init__(self, annotations_dir: str = "output/annotations"):
        self.base_dir = Path(annotations_dir)
        self.index_path = self.base_dir / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return {"version": "2.0", "annotations": {}}

    def save_index(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def get_annotated_frames(self, video_name: str) -> list[int]:
        frames = self.index.get("annotations", {}).get(video_name, {}).get("frames", [])
        return sorted(frames)

    def add_frame(self, video_name: str, frame_idx: int) -> None:
        if video_name not in self.index["annotations"]:
            self.index["annotations"][video_name] = {"frames": [], "last_modified": ""}

        frames = self.index["annotations"][video_name]["frames"]
        if frame_idx not in frames:
            frames.append(frame_idx)
            frames.sort()

        self.index["annotations"][video_name]["last_modified"] = datetime.now().isoformat()
        self.save_index()

    def remove_frame(self, video_name: str, frame_idx: int) -> None:
        if video_name not in self.index["annotations"]:
            return
        frames = self.index["annotations"][video_name]["frames"]
        if frame_idx in frames:
            frames.remove(frame_idx)
            self.index["annotations"][video_name]["last_modified"] = datetime.now().isoformat()
            self.save_index()

    def get_video_dir(self, video_name: str) -> Path:
        return self.base_dir / video_name

    def get_all_video_names(self) -> list[str]:
        return sorted(self.index.get("annotations", {}).keys())

    def get_annotated_frame_stats(self) -> list[dict]:
        """Walk every video in the index, read each frame_<idx>.json for
        per-frame stats. Returns rows ready to render in the UI."""
        rows: list[dict] = []
        for video_name, entry in self.index.get("annotations", {}).items():
            video_dir = self.get_video_dir(video_name)
            for frame_idx in sorted(entry.get("frames", [])):
                row = {
                    "video_name": video_name,
                    "frame_idx": int(frame_idx),
                    "num_points": None,
                    "rmse": None,
                    "video_path": None,
                }
                json_path = video_dir / f"frame_{frame_idx}.json"
                if json_path.exists():
                    try:
                        with open(json_path) as f:
                            data = json.load(f)
                        manual = int(data.get("num_manual_points", 0))
                        derived = int(data.get("num_derived_points", 0))
                        row["num_points"] = manual + derived
                        row["rmse"] = float(data.get("reprojection_error", 0.0))
                        row["video_path"] = data.get("video_path")
                    except (json.JSONDecodeError, IOError, ValueError):
                        pass
                rows.append(row)
        return rows
