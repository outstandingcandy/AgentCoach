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
import shutil
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


def migrate_legacy_annotations(
    legacy_dir: str,
    annotations_dir: str = "output/annotations",
) -> bool:
    """Migrate legacy annotations.json + H0.npy into unified storage."""
    legacy_path = Path(legacy_dir)
    json_path = legacy_path / "annotations.json"

    if not json_path.exists():
        print(f"No annotations.json found in {legacy_dir}")
        return False

    try:
        with open(json_path) as f:
            data = json.load(f)

        video_path = data.get("video_path", "")
        video_name = Path(video_path).stem if video_path else legacy_path.name
        frame_idx = data.get("anchor_frame_idx", 0)

        index = AnnotationIndex(annotations_dir)
        target_dir = index.get_video_dir(video_name)
        target_dir.mkdir(parents=True, exist_ok=True)

        data["frame_idx"] = frame_idx
        data["video_name"] = video_name
        data["migrated_from"] = str(legacy_path)
        data["migrated_at"] = datetime.now().isoformat()

        with open(target_dir / f"frame_{frame_idx}.json", "w") as f:
            json.dump(data, f, indent=2)

        h0_path = legacy_path / "H0.npy"
        if h0_path.exists():
            shutil.copy(h0_path, target_dir / f"frame_{frame_idx}.npy")

        vis_path = legacy_path / "anchor_frame.jpg"
        if vis_path.exists():
            shutil.copy(vis_path, target_dir / f"frame_{frame_idx}.jpg")

        index.add_frame(video_name, frame_idx)
        print(f"Migrated {video_name} frame {frame_idx} to {target_dir}")
        return True
    except Exception as e:
        print(f"Migration failed: {e}")
        return False
