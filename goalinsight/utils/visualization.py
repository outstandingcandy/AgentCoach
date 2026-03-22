"""Visualization utilities for annotations."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class Visualizer:
    """Visualizer for soccer annotations."""

    # Color palette for teams and roles
    COLORS = {
        "left": (0, 255, 0),       # Green
        "right": (0, 0, 255),      # Red
        "referee": (255, 255, 0),  # Cyan
        "goalkeeper": (255, 0, 255),  # Magenta
        "unknown": (128, 128, 128),  # Gray
    }

    # Keypoint colors
    KEYPOINT_COLOR = (0, 0, 255)    # Red
    LINE_COLOR = (0, 0, 255)        # Red

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize visualizer.

        Args:
            config: Visualization configuration.
        """
        self.config = config or {}
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2

    def draw_players(
        self,
        frame: np.ndarray,
        players: list[dict[str, Any]],
        show_track_id: bool = True,
        show_jersey: bool = True,
        show_team: bool = True,
    ) -> np.ndarray:
        """Draw player bounding boxes and annotations on frame.

        Args:
            frame: Input frame (BGR format).
            players: List of player dictionaries with bbox, track_id, etc.
            show_track_id: Whether to show track IDs.
            show_jersey: Whether to show jersey numbers.
            show_team: Whether to color by team.

        Returns:
            Frame with annotations drawn.
        """
        result = frame.copy()

        for player in players:
            bbox = player.get("bbox", [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)

            # Get color based on team
            team = player.get("team", "unknown")
            role = player.get("role", "player")
            if role in ["referee", "goalkeeper"]:
                color = self.COLORS.get(role, self.COLORS["unknown"])
            else:
                color = self.COLORS.get(team, self.COLORS["unknown"])

            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, self.thickness)

            # Build label
            label_parts = []
            if show_track_id and "track_id" in player:
                label_parts.append(f"#{player['track_id']}")
            if show_jersey and player.get("jersey_number") is not None:
                label_parts.append(str(player["jersey_number"]))

            if label_parts:
                label = " ".join(label_parts)
                label_size = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)[0]

                # Draw label background
                cv2.rectangle(
                    result,
                    (x1, y1 - label_size[1] - 10),
                    (x1 + label_size[0] + 10, y1),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    result,
                    label,
                    (x1 + 5, y1 - 5),
                    self.font,
                    self.font_scale,
                    (255, 255, 255),
                    self.thickness,
                )

        return result

    def draw_field_keypoints(
        self,
        frame: np.ndarray,
        keypoints: list[dict[str, Any]],
        radius: int = 5,
    ) -> np.ndarray:
        """Draw detected field keypoints on frame.

        Args:
            frame: Input frame (BGR format).
            keypoints: List of keypoint dictionaries with pixel coordinates.
            radius: Circle radius for keypoints.

        Returns:
            Frame with keypoints drawn.
        """
        result = frame.copy()

        for kp in keypoints:
            x, y = int(kp.get("x", 0)), int(kp.get("y", 0))
            conf = kp.get("confidence", 1.0)

            # Use red color with intensity based on confidence
            intensity = int(255 * conf)
            # KEYPOINT_COLOR is (0, 0, 255) = Red in BGR
            color = (0, 0, intensity)  # Red with varying intensity

            cv2.circle(result, (x, y), radius, color, -1)

            # Draw keypoint name if available
            name = kp.get("name", "")
            if name:
                cv2.putText(
                    result,
                    name[:10],  # Truncate long names
                    (x + radius + 2, y),
                    self.font,
                    0.4,
                    self.KEYPOINT_COLOR,
                    1,
                )

        return result

    def draw_field_lines(
        self,
        frame: np.ndarray,
        lines: list[tuple[tuple[int, int], tuple[int, int]]],
        color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """Draw detected field lines on frame.

        Args:
            frame: Input frame (BGR format).
            lines: List of line segments as ((x1, y1), (x2, y2)).
            color: Line color (BGR). Uses default if None.

        Returns:
            Frame with lines drawn.
        """
        result = frame.copy()
        color = color or self.LINE_COLOR

        for (x1, y1), (x2, y2) in lines:
            cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        return result

    def draw_pitch_positions(
        self,
        pitch_image: np.ndarray,
        players: list[dict[str, Any]],
        scale: float = 10.0,
    ) -> np.ndarray:
        """Draw player positions on a 2D pitch diagram.

        Args:
            pitch_image: 2D pitch template image.
            players: List of player dictionaries with pitch_position.
            scale: Scale factor for pitch coordinates to pixels.

        Returns:
            Pitch image with player positions marked.
        """
        result = pitch_image.copy()
        center_x = result.shape[1] // 2
        center_y = result.shape[0] // 2

        for player in players:
            pos = player.get("pitch_position")
            if pos is None or len(pos) != 2:
                continue

            # Convert pitch coordinates to image coordinates
            x = int(center_x + pos[0] * scale)
            y = int(center_y - pos[1] * scale)  # Y is inverted

            team = player.get("team", "unknown")
            color = self.COLORS.get(team, self.COLORS["unknown"])

            cv2.circle(result, (x, y), 8, color, -1)
            cv2.circle(result, (x, y), 8, (0, 0, 0), 2)

            # Draw jersey number
            jersey = player.get("jersey_number")
            if jersey is not None:
                cv2.putText(
                    result,
                    str(jersey),
                    (x - 8, y + 4),
                    self.font,
                    0.4,
                    (255, 255, 255),
                    1,
                )

        return result

    def create_pitch_template(
        self,
        width: int = 1050,
        height: int = 680,
    ) -> np.ndarray:
        """Create a blank 2D pitch template image.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            Pitch template image (BGR).
        """
        # Green background
        pitch = np.zeros((height, width, 3), dtype=np.uint8)
        pitch[:] = (34, 139, 34)  # Forest green

        # Draw pitch markings (simplified)
        cx, cy = width // 2, height // 2
        scale = min(width / 115, height / 78)

        # Boundary
        margin = 50
        cv2.rectangle(
            pitch,
            (margin, margin),
            (width - margin, height - margin),
            (255, 255, 255),
            2,
        )

        # Center line
        cv2.line(pitch, (cx, margin), (cx, height - margin), (255, 255, 255), 2)

        # Center circle
        cv2.circle(pitch, (cx, cy), int(9.15 * scale), (255, 255, 255), 2)
        cv2.circle(pitch, (cx, cy), 3, (255, 255, 255), -1)

        # Penalty areas
        pa_w = int(16.5 * scale)
        pa_h = int(40.32 * scale / 2)
        cv2.rectangle(pitch, (margin, cy - pa_h), (margin + pa_w, cy + pa_h), (255, 255, 255), 2)
        cv2.rectangle(pitch, (width - margin - pa_w, cy - pa_h), (width - margin, cy + pa_h), (255, 255, 255), 2)

        # Goal areas
        ga_w = int(5.5 * scale)
        ga_h = int(18.32 * scale / 2)
        cv2.rectangle(pitch, (margin, cy - ga_h), (margin + ga_w, cy + ga_h), (255, 255, 255), 2)
        cv2.rectangle(pitch, (width - margin - ga_w, cy - ga_h), (width - margin, cy + ga_h), (255, 255, 255), 2)

        return pitch

    def save_video(
        self,
        frames: list[np.ndarray],
        output_path: str | Path,
        fps: float = 25.0,
    ) -> None:
        """Save annotated frames as video.

        Args:
            frames: List of annotated frames.
            output_path: Output video path.
            fps: Output frame rate.
        """
        if not frames:
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame in frames:
            writer.write(frame)

        writer.release()
