"""Minimal visualization for GoalInsight pipeline.

Implements BaseVisualizer interface with simple bounding box visualization.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...interfaces import BaseVisualizer


# Color scheme (BGR)
COLORS = {
    'team_A': (0, 0, 255),     # Red
    'team_B': (255, 0, 0),     # Blue
    'left': (0, 255, 0),       # Green
    'right': (0, 0, 255),      # Red
    'referee': (0, 255, 255),  # Yellow
    'goalkeeper': (255, 0, 255),  # Magenta
    'unknown': (128, 128, 128),   # Gray
    'player': (255, 255, 255),    # White
    'detection': (0, 255, 255),   # Yellow
    'ball': (0, 165, 255),        # Orange
    'ball_trail': (128, 200, 255),  # Light orange
}


class MinimalVisualizer(BaseVisualizer):
    """Simple visualization with bounding boxes and labels."""

    def __init__(self, output_dir: Path | str | None = None):
        """Initialize visualizer.

        Args:
            output_dir: Directory to save visualization outputs.
        """
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw detection bounding boxes.

        Args:
            frame: Input frame (BGR format).
            detections: List of detection dicts with 'bbox' and 'confidence'.
            frame_idx: Frame index for display.

        Returns:
            Frame with drawn detections.
        """
        vis = frame.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det.get('confidence', 1.0)
            x1, y1, x2, y2 = map(int, bbox)
            color = COLORS['detection']

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(vis, f"Frame {frame_idx} | {len(detections)} detections",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def draw_tracking(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        track_history: dict[int, list[tuple[int, int]]] | None = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw tracking results with trajectories.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts with 'track_id', 'bbox'.
            track_history: Dictionary of track trajectories.
            frame_idx: Frame index for display.

        Returns:
            Frame with tracking visualization.
        """
        vis = frame.copy()
        track_history = track_history or {}

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            role = track.get('role', 'player')
            team = track.get('team', 'unknown')
            x1, y1, x2, y2 = map(int, bbox)

            # Get color based on team or role
            if team in COLORS:
                color = COLORS[team]
            elif role in COLORS:
                color = COLORS[role]
            else:
                color = COLORS['player']

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"ID:{track_id}"
            cv2.putText(vis, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory
            center = ((x1 + x2) // 2, y2)
            if track_id in track_history:
                history = track_history[track_id][-20:]  # Last 20 points
                for i in range(1, len(history)):
                    cv2.line(vis, history[i-1], history[i], color, 2)

        cv2.putText(vis, f"Frame {frame_idx} | {len(tracks)} tracks",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def draw_final_result(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        roles: dict[int, str],
    ) -> np.ndarray:
        """Draw final result with team sides and roles.

        Args:
            frame: Input frame (BGR format).
            tracks: List of track dicts.
            team_sides: {track_id: 'left'/'right'/'referee'}.
            roles: {track_id: role}.

        Returns:
            Frame with final visualization.
        """
        vis = frame.copy()

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            side = team_sides.get(track_id, 'unknown')
            role = roles.get(track_id, 'player')

            # Determine color
            if role == 'referee':
                color = COLORS['referee']
            elif side in COLORS:
                color = COLORS[side]
            else:
                color = COLORS['unknown']

            # Draw box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Draw label
            jersey_num = track.get('jersey_number')
            if jersey_num is not None:
                label = f"#{jersey_num}"
            else:
                label = f"{track_id}"

            if role == 'goalkeeper':
                label += " GK"
            elif role == 'referee':
                label += " REF"

            # Background for label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Legend
        legend_y = 30
        for side, color in [('left', COLORS['left']), ('right', COLORS['right']),
                            ('referee', COLORS['referee'])]:
            cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y), color, -1)
            cv2.putText(vis, side.capitalize(), (35, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25

        return vis

    def draw_ball(
        self,
        frame: np.ndarray,
        ball_track: dict[str, Any] | None,
        trajectory: list[tuple[float, float]] | None = None,
        show_velocity: bool = True,
    ) -> np.ndarray:
        """Draw ball position and trajectory.

        Args:
            frame: Input frame (BGR format).
            ball_track: Ball track dict with 'center', 'bbox', 'velocity'.
            trajectory: List of recent (x, y) positions for trail.
            show_velocity: Whether to show velocity arrow.

        Returns:
            Frame with ball visualization.
        """
        if ball_track is None:
            return frame

        vis = frame.copy()
        center = ball_track.get("center", [0, 0])
        cx, cy = int(center[0]), int(center[1])

        # Draw trajectory trail with fade effect
        if trajectory and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                color = tuple(int(c * alpha) for c in COLORS['ball_trail'])
                pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                thickness = max(1, int(2 * alpha))
                cv2.line(vis, pt1, pt2, color, thickness)

        # Draw ball position (filled circle with outline)
        radius = 8
        cv2.circle(vis, (cx, cy), radius, COLORS['ball'], -1)
        cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2)

        # Draw velocity arrow
        if show_velocity and "velocity" in ball_track:
            vx, vy = ball_track["velocity"]
            speed = (vx**2 + vy**2) ** 0.5
            if speed > 2.0:  # Only show if moving
                scale = 3.0  # Arrow length scale
                end_x = int(cx + vx * scale)
                end_y = int(cy + vy * scale)
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), COLORS['ball'], 2, tipLength=0.3)

        # Draw confidence label
        conf = ball_track.get("confidence", 0.0)
        status = ball_track.get("status", "unknown")
        label = f"Ball {conf:.2f}"
        if status == "confirmed":
            label += " [OK]"
        cv2.putText(vis, label, (cx + 12, cy + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['ball'], 1)

        return vis
