"""Step-by-step visualization for pipeline debugging.

Implements BaseVisualizer interface with detailed visualization.
Based on baseline/step_visualizer.py.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...interfaces import BaseVisualizer

try:
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# Color scheme (BGR)
COLORS = {
    'left': (0, 255, 0),       # Green
    'right': (0, 0, 255),      # Red
    'referee': (255, 255, 0),  # Cyan
    'goalkeeper': (255, 0, 255),
    'unknown': (128, 128, 128),
    'ball': (0, 165, 255),     # Orange
    'player': (255, 255, 255),
    'detection': (0, 255, 255),
    'keypoint': (0, 255, 0),
    'line': (255, 165, 0),
}

PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0


class StepVisualizer(BaseVisualizer):
    """Visualize each pipeline step with detailed output."""

    def __init__(self, output_dir: Path | str | None = None):
        """Initialize step visualizer."""
        self.output_dir = Path(output_dir) if output_dir else None
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.writers: dict[str, cv2.VideoWriter] = {}
        self.pitch_template = None
        self.pitch_scale = 8

    def _init_pitch_template(self) -> np.ndarray:
        """Create a pitch template for radar visualization."""
        if self.pitch_template is not None:
            return self.pitch_template

        margin = 10
        width = int((PITCH_LENGTH + 2 * margin) * self.pitch_scale)
        height = int((PITCH_WIDTH + 2 * margin) * self.pitch_scale)

        pitch = np.zeros((height, width, 3), dtype=np.uint8)
        pitch[:] = (34, 139, 34)

        def to_pixel(x: float, y: float) -> tuple[int, int]:
            px = int((x + PITCH_LENGTH / 2 + margin) * self.pitch_scale)
            py = int((y + PITCH_WIDTH / 2 + margin) * self.pitch_scale)
            return (px, py)

        color = (255, 255, 255)
        thickness = 2

        # Boundary
        corners = [
            (-PITCH_LENGTH/2, -PITCH_WIDTH/2),
            (PITCH_LENGTH/2, -PITCH_WIDTH/2),
            (PITCH_LENGTH/2, PITCH_WIDTH/2),
            (-PITCH_LENGTH/2, PITCH_WIDTH/2),
        ]
        for i in range(4):
            p1 = to_pixel(*corners[i])
            p2 = to_pixel(*corners[(i+1) % 4])
            cv2.line(pitch, p1, p2, color, thickness)

        # Center line
        cv2.line(pitch, to_pixel(0, -PITCH_WIDTH/2), to_pixel(0, PITCH_WIDTH/2), color, thickness)

        # Center circle
        center = to_pixel(0, 0)
        radius = int(9.15 * self.pitch_scale)
        cv2.circle(pitch, center, radius, color, thickness)
        cv2.circle(pitch, center, 4, color, -1)

        # Penalty areas
        pa_w = 40.32 / 2
        pa_d = 16.5
        cv2.rectangle(pitch, to_pixel(-PITCH_LENGTH/2, -pa_w), to_pixel(-PITCH_LENGTH/2 + pa_d, pa_w), color, thickness)
        cv2.rectangle(pitch, to_pixel(PITCH_LENGTH/2 - pa_d, -pa_w), to_pixel(PITCH_LENGTH/2, pa_w), color, thickness)

        # Goal areas
        ga_w = 18.32 / 2
        ga_d = 5.5
        cv2.rectangle(pitch, to_pixel(-PITCH_LENGTH/2, -ga_w), to_pixel(-PITCH_LENGTH/2 + ga_d, ga_w), color, thickness)
        cv2.rectangle(pitch, to_pixel(PITCH_LENGTH/2 - ga_d, -ga_w), to_pixel(PITCH_LENGTH/2, ga_w), color, thickness)

        # Penalty spots
        cv2.circle(pitch, to_pixel(-PITCH_LENGTH/2 + 11, 0), 4, color, -1)
        cv2.circle(pitch, to_pixel(PITCH_LENGTH/2 - 11, 0), 4, color, -1)

        self.pitch_template = pitch
        return pitch

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: list[dict[str, Any]],
        frame_idx: int = 0,
        show_filter_reason: bool = False,
        filtered_detections: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Draw detection bounding boxes."""
        vis = frame.copy()

        filtered_centers = set()
        if filtered_detections is not None:
            for fd in filtered_detections:
                bbox = fd['bbox']
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                filtered_centers.add((cx, cy))

        num_filtered_out = 0

        for det in detections:
            bbox = det['bbox']
            conf = det.get('confidence', 1.0)
            x1, y1, x2, y2 = map(int, bbox)

            is_filtered_out = False
            if show_filter_reason and filtered_detections is not None:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                is_filtered_out = True
                for fcx, fcy in filtered_centers:
                    if abs(cx - fcx) < 10 and abs(cy - fcy) < 10:
                        is_filtered_out = False
                        break
                if is_filtered_out:
                    num_filtered_out += 1

            color = (128, 128, 128) if is_filtered_out else COLORS['detection']
            thickness = 1 if is_filtered_out else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)

            label = f"{conf:.2f}" if not is_filtered_out else f"{conf:.2f} [filtered]"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        info_text = f"Frame {frame_idx} | {len(detections)} detections"
        if show_filter_reason and num_filtered_out > 0:
            info_text += f" ({num_filtered_out} filtered)"
        cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def draw_tracking(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        track_history: dict[int, list[tuple[int, int]]] | None = None,
        frame_idx: int = 0,
    ) -> np.ndarray:
        """Draw tracking results."""
        vis = frame.copy()
        track_history = track_history or {}

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            role = track.get('role', 'player')
            x1, y1, x2, y2 = map(int, bbox)

            color = COLORS.get(role, COLORS['player'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{track_id}"
            if role != 'player':
                label += f" ({role[:2].upper()})"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            center = ((x1 + x2) // 2, y2)
            if track_id in track_history:
                history = track_history[track_id]
                for i in range(1, len(history)):
                    cv2.line(vis, history[i-1], history[i], color, 2)

        cv2.putText(vis, f"Frame {frame_idx} | {len(tracks)} tracks",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def draw_calibration(
        self,
        frame: np.ndarray,
        keypoints: dict,
        lines: dict,
        homography: np.ndarray | None,
        image_width: int,
        image_height: int,
    ) -> np.ndarray:
        """Draw calibration results."""
        vis = frame.copy()

        # Draw keypoints
        for idx, kp in keypoints.items():
            x = int(kp['x'] * image_width)
            y = int(kp['y'] * image_height)
            conf = kp.get('p', 1.0)
            color = (0, int(255 * conf), 0)
            cv2.circle(vis, (x, y), 5, color, -1)
            cv2.putText(vis, str(idx), (x + 7, y + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Draw lines
        for line_name, points in lines.items():
            color = (255, 165, 0)
            for i in range(len(points) - 1):
                p1 = (int(points[i]['x'] * image_width), int(points[i]['y'] * image_height))
                p2 = (int(points[i+1]['x'] * image_width), int(points[i+1]['y'] * image_height))
                cv2.line(vis, p1, p2, color, 2)

        cv2.putText(vis, f"Keypoints: {len(keypoints)} | Lines: {len(lines)} | H: {'OK' if homography is not None else 'None'}",
                   (10, image_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return vis

    def draw_team_clustering(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_clusters: dict[int, int],
    ) -> np.ndarray:
        """Draw team clustering results."""
        vis = frame.copy()

        cluster_colors = {0: (0, 255, 0), 1: (0, 0, 255)}

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            cluster = team_clusters.get(track_id, -1)
            color = cluster_colors.get(cluster, (128, 128, 128))

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label = f"C{cluster}" if cluster >= 0 else "?"
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(vis, "Cluster 0", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_colors[0], 2)
        cv2.putText(vis, "Cluster 1", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cluster_colors[1], 2)

        return vis

    def draw_final_result(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        roles: dict[int, str],
    ) -> np.ndarray:
        """Draw final result with team sides."""
        vis = frame.copy()

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            x1, y1, x2, y2 = map(int, bbox)

            side = team_sides.get(track_id, 'unknown')
            role = roles.get(track_id, 'player')

            if role == 'referee':
                color = COLORS['referee']
            elif role == 'goalkeeper':
                base_color = COLORS.get(side, COLORS['unknown'])
                color = tuple(min(255, c + 50) for c in base_color)
            else:
                color = COLORS.get(side, COLORS['unknown'])

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            jersey_num = track.get('jersey_number')
            label = f"#{jersey_num}" if jersey_num is not None else f"{track_id}"
            if role == 'goalkeeper':
                label += " GK"
            elif role == 'referee':
                label += " REF"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(vis, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Legend
        legend_y = 30
        for side, color in [('left', COLORS['left']), ('right', COLORS['right']),
                            ('referee', COLORS['referee']), ('goalkeeper', COLORS['goalkeeper'])]:
            cv2.rectangle(vis, (10, legend_y - 15), (30, legend_y), color, -1)
            cv2.putText(vis, side.capitalize(), (35, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            legend_y += 25

        return vis

    def draw_radar_view(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        homography: np.ndarray | None,
        position: str = 'bottom-right',
    ) -> np.ndarray:
        """Draw radar/minimap view."""
        if homography is None:
            return frame

        vis = frame.copy()
        pitch = self._init_pitch_template().copy()
        ph, pw = pitch.shape[:2]

        margin = 10
        def to_pixel(x: float, y: float) -> tuple[int, int]:
            px = int((x + PITCH_LENGTH / 2 + margin) * self.pitch_scale)
            py = int((y + PITCH_WIDTH / 2 + margin) * self.pitch_scale)
            return (px, py)

        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            side = team_sides.get(track_id, 'unknown')

            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]

            pt = homography @ np.array([foot_x, foot_y, 1.0])
            if abs(pt[2]) < 1e-6:
                continue
            world_x = pt[0] / pt[2]
            world_y = pt[1] / pt[2]

            if abs(world_x) > PITCH_LENGTH / 2 + 5 or abs(world_y) > PITCH_WIDTH / 2 + 5:
                continue

            px, py = to_pixel(world_x, world_y)
            color = COLORS.get(side, COLORS['unknown'])

            cv2.circle(pitch, (px, py), 8, color, -1)
            cv2.circle(pitch, (px, py), 8, (0, 0, 0), 1)
            cv2.putText(pitch, str(track_id), (px - 5, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

        # Scale and position radar
        scale = 0.4
        small_pitch = cv2.resize(pitch, (int(pw * scale), int(ph * scale)))
        sh, sw = small_pitch.shape[:2]

        fh, fw = vis.shape[:2]
        if position == 'bottom-right':
            x, y = fw - sw - 20, fh - sh - 20
        else:
            x, y = 20, fh - sh - 20

        alpha = 0.85
        roi = vis[y:y+sh, x:x+sw]
        blended = cv2.addWeighted(small_pitch, alpha, roi, 1 - alpha, 0)
        vis[y:y+sh, x:x+sw] = blended
        cv2.rectangle(vis, (x, y), (x + sw, y + sh), (255, 255, 255), 2)

        return vis

    def draw_ball(
        self,
        frame: np.ndarray,
        ball_track: dict[str, Any] | None,
        trajectory: list[tuple[float, float]] | None = None,
        show_velocity: bool = True,
    ) -> np.ndarray:
        """Draw ball position with trajectory and velocity.

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

        # Draw trajectory trail with gradient fade
        if trajectory and len(trajectory) > 1:
            for i in range(1, len(trajectory)):
                alpha = i / len(trajectory)
                # Gradient from light to full orange
                color = (
                    int(128 + 127 * alpha),  # B
                    int(200 - 35 * alpha),    # G
                    int(255),                 # R
                )
                pt1 = (int(trajectory[i-1][0]), int(trajectory[i-1][1]))
                pt2 = (int(trajectory[i][0]), int(trajectory[i][1]))
                thickness = max(1, int(3 * alpha))
                cv2.line(vis, pt1, pt2, color, thickness)

        # Draw ball with glow effect
        radius = 10
        # Outer glow
        cv2.circle(vis, (cx, cy), radius + 4, (100, 150, 200), 2)
        # Main ball
        cv2.circle(vis, (cx, cy), radius, COLORS['ball'], -1)
        # Highlight
        cv2.circle(vis, (cx, cy), radius, (255, 255, 255), 2)

        # Draw velocity arrow
        if show_velocity and "velocity" in ball_track:
            vx, vy = ball_track["velocity"]
            speed = (vx**2 + vy**2) ** 0.5
            if speed > 2.0:
                scale = 4.0
                end_x = int(cx + vx * scale)
                end_y = int(cy + vy * scale)
                cv2.arrowedLine(vis, (cx, cy), (end_x, end_y), (0, 200, 255), 2, tipLength=0.25)

                # Speed label
                cv2.putText(vis, f"{speed:.1f}px/f", (end_x + 5, end_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

        # Draw info box
        conf = ball_track.get("confidence", 0.0)
        status = ball_track.get("status", "unknown")

        info_x = cx + radius + 8
        info_y = cy - radius

        # Status indicator
        status_color = (0, 255, 0) if status == "confirmed" else (0, 255, 255)
        cv2.circle(vis, (info_x, info_y), 4, status_color, -1)

        # Labels
        cv2.putText(vis, f"Ball", (info_x + 8, info_y + 4),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS['ball'], 1)
        cv2.putText(vis, f"{conf:.2f}", (info_x + 8, info_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

        # Pitch position if available
        pitch_pos = ball_track.get("pitch_position")
        if pitch_pos:
            cv2.putText(vis, f"({pitch_pos[0]:.1f}, {pitch_pos[1]:.1f})m",
                       (info_x + 8, info_y + 32),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

        return vis

    def draw_radar_with_ball(
        self,
        frame: np.ndarray,
        tracks: list[dict[str, Any]],
        team_sides: dict[int, str],
        ball_track: dict[str, Any] | None,
        homography: np.ndarray | None,
        position: str = 'bottom-right',
    ) -> np.ndarray:
        """Draw radar view with ball position.

        Args:
            frame: Input frame (BGR format).
            tracks: List of player track dicts.
            team_sides: {track_id: team_side}.
            ball_track: Ball track dict or None.
            homography: Image -> world homography.
            position: Where to place radar.

        Returns:
            Frame with radar overlay including ball.
        """
        if homography is None:
            return frame

        vis = frame.copy()
        pitch = self._init_pitch_template().copy()
        ph, pw = pitch.shape[:2]

        margin = 10
        def to_pixel(x: float, y: float) -> tuple[int, int]:
            px = int((x + PITCH_LENGTH / 2 + margin) * self.pitch_scale)
            py = int((y + PITCH_WIDTH / 2 + margin) * self.pitch_scale)
            return (px, py)

        # Draw players
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            side = team_sides.get(track_id, 'unknown')

            foot_x = (bbox[0] + bbox[2]) / 2
            foot_y = bbox[3]

            pt = homography @ np.array([foot_x, foot_y, 1.0])
            if abs(pt[2]) < 1e-6:
                continue
            world_x = pt[0] / pt[2]
            world_y = pt[1] / pt[2]

            if abs(world_x) > PITCH_LENGTH / 2 + 5 or abs(world_y) > PITCH_WIDTH / 2 + 5:
                continue

            px, py = to_pixel(world_x, world_y)
            color = COLORS.get(side, COLORS['unknown'])

            cv2.circle(pitch, (px, py), 8, color, -1)
            cv2.circle(pitch, (px, py), 8, (0, 0, 0), 1)

        # Draw ball on radar
        if ball_track:
            ball_pitch_pos = ball_track.get("pitch_position")
            if ball_pitch_pos:
                bx, by = to_pixel(ball_pitch_pos[0], ball_pitch_pos[1])
                # Ball with highlight
                cv2.circle(pitch, (bx, by), 6, COLORS['ball'], -1)
                cv2.circle(pitch, (bx, by), 6, (255, 255, 255), 2)
            else:
                # Try to project from image coordinates
                center = ball_track.get("center")
                if center:
                    pt = homography @ np.array([center[0], center[1], 1.0])
                    if abs(pt[2]) > 1e-6:
                        world_x = pt[0] / pt[2]
                        world_y = pt[1] / pt[2]
                        if abs(world_x) <= PITCH_LENGTH / 2 + 2 and abs(world_y) <= PITCH_WIDTH / 2 + 2:
                            bx, by = to_pixel(world_x, world_y)
                            cv2.circle(pitch, (bx, by), 6, COLORS['ball'], -1)
                            cv2.circle(pitch, (bx, by), 6, (255, 255, 255), 2)

        # Scale and position radar
        scale = 0.4
        small_pitch = cv2.resize(pitch, (int(pw * scale), int(ph * scale)))
        sh, sw = small_pitch.shape[:2]

        fh, fw = vis.shape[:2]
        if position == 'bottom-right':
            x, y = fw - sw - 20, fh - sh - 20
        else:
            x, y = 20, fh - sh - 20

        alpha = 0.85
        roi = vis[y:y+sh, x:x+sw]
        blended = cv2.addWeighted(small_pitch, alpha, roi, 1 - alpha, 0)
        vis[y:y+sh, x:x+sw] = blended
        cv2.rectangle(vis, (x, y), (x + sw, y + sh), (255, 255, 255), 2)

        return vis

    def close_writers(self) -> None:
        """Close all video writers."""
        for writer in self.writers.values():
            writer.release()
        self.writers.clear()
