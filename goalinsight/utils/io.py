"""Input/Output utilities for annotations."""

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


class AnnotationIO:
    """Handle reading and writing annotations."""

    def __init__(self, output_dir: str | Path | None = None):
        """Initialize AnnotationIO.

        Args:
            output_dir: Default output directory for annotations.
        """
        self.output_dir = Path(output_dir) if output_dir else None

    def save_frame_annotation(
        self,
        annotation: dict[str, Any],
        output_path: str | Path,
    ) -> None:
        """Save single frame annotation to JSON.

        Args:
            annotation: Frame annotation dictionary.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(annotation, f, cls=NumpyEncoder, indent=2)

    def save_video_annotations(
        self,
        annotations: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Save all frame annotations for a video.

        Args:
            annotations: List of frame annotation dictionaries.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as single JSON file with all frames
        output = {
            "num_frames": len(annotations),
            "frames": annotations,
        }

        with open(output_path, "w") as f:
            json.dump(output, f, cls=NumpyEncoder, indent=2)

    def load_video_annotations(self, input_path: str | Path) -> list[dict[str, Any]]:
        """Load video annotations from JSON file.

        Args:
            input_path: Input file path.

        Returns:
            List of frame annotation dictionaries.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {input_path}")

        with open(input_path, "r") as f:
            data = json.load(f)

        if "frames" in data:
            return data["frames"]
        return data

    def create_frame_annotation(
        self,
        frame_id: int,
        timestamp: float,
        field_registration: dict[str, Any] | None = None,
        players: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create a standardized frame annotation dictionary.

        Args:
            frame_id: Frame index.
            timestamp: Frame timestamp in seconds.
            field_registration: Field registration data (homography, camera params).
            players: List of player annotations.

        Returns:
            Standardized frame annotation dictionary.
        """
        annotation = {
            "frame_id": frame_id,
            "timestamp": timestamp,
            "field_registration": field_registration or {},
            "players": players or [],
        }
        return annotation

    def create_player_annotation(
        self,
        track_id: int,
        bbox: list[float],
        pitch_position: list[float] | None = None,
        role: str = "player",
        team: str | None = None,
        jersey_number: int | None = None,
        reid_embedding: list[float] | None = None,
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Create a standardized player annotation dictionary.

        Args:
            track_id: Unique track identifier.
            bbox: Bounding box [x1, y1, x2, y2].
            pitch_position: Position on pitch [x, y] in meters.
            role: Player role (player, goalkeeper, referee).
            team: Team identifier (left, right).
            jersey_number: Jersey number.
            reid_embedding: ReID feature embedding.
            confidence: Detection confidence.

        Returns:
            Standardized player annotation dictionary.
        """
        annotation = {
            "track_id": track_id,
            "bbox": bbox,
            "confidence": confidence,
            "role": role,
        }

        if pitch_position is not None:
            annotation["pitch_position"] = pitch_position
        if team is not None:
            annotation["team"] = team
        if jersey_number is not None:
            annotation["jersey_number"] = jersey_number
        if reid_embedding is not None:
            annotation["reid_embedding"] = reid_embedding

        return annotation

    def export_to_soccernet_format(
        self,
        annotations: list[dict[str, Any]],
        output_path: str | Path,
    ) -> None:
        """Export annotations to SoccerNet format.

        Args:
            annotations: List of frame annotations.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to SoccerNet tracking format
        soccernet_data = {
            "UrlLocal": "",
            "predictions": [],
        }

        for frame_ann in annotations:
            frame_id = frame_ann.get("frame_id", 0)

            for player in frame_ann.get("players", []):
                bbox = player.get("bbox", [0, 0, 0, 0])
                pred = {
                    "imageId": frame_id,
                    "category": player.get("role", "player"),
                    "bbox": {
                        "x": bbox[0],
                        "y": bbox[1],
                        "w": bbox[2] - bbox[0],
                        "h": bbox[3] - bbox[1],
                    },
                    "track_id": player.get("track_id", -1),
                    "team": player.get("team", "unknown"),
                    "jersey_number": player.get("jersey_number"),
                }
                soccernet_data["predictions"].append(pred)

        with open(output_path, "w") as f:
            json.dump(soccernet_data, f, indent=2)
