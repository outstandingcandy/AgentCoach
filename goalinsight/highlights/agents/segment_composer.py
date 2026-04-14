"""SegmentComposer — renders highlight clips from analyzed events."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import json

import cv2
import numpy as np

from goalinsight.video_processor import VideoProcessor

from .._closeup import CropInfo, SmoothState, extract_closeup, interpolate_bbox
from .._context import MatchContext
from .._types import AnalyzedEvent, ClipSegment
from .base import BaseClipComposer

logger = logging.getLogger(__name__)


class SegmentComposer(BaseClipComposer):
    """Compose a highlight clip by stitching segments with optional close-ups."""

    def compose(
        self,
        analyzed: AnalyzedEvent,
        ctx: MatchContext,
        output_dir: Path,
        config: dict[str, Any],
    ) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        event = analyzed.event
        clip_name = (
            f"{event.event_type}_{event.frame}"
            f"_{event.metadata.get('goal_side', '')}.mp4"
        )
        output_path = output_dir / clip_name

        closeup_cfg = config.get("closeup", {})
        out_size = tuple(closeup_cfg.get("output_size", [640, 360]))
        padding_factor = closeup_cfg.get("padding_factor", 2.0)
        smooth_alpha = closeup_cfg.get("smooth_alpha", 0.3)
        medium_padding = closeup_cfg.get("medium_padding_factor", 4.0)
        ball_padding = closeup_cfg.get("ball_padding_factor", 6.0)

        overlay_cfg = config.get("overlays", {})
        overlays_enabled = overlay_cfg.get("enabled", True)

        output_fps = config.get("output_fps", ctx.fps)
        crossfade_frames = config.get("crossfade_frames", 4)

        # Output dimensions: full frame for wide, output_size for closeup
        # Use full frame size so all segments have the same resolution
        out_w, out_h = ctx.width, ctx.height

        vp = VideoProcessor()
        writer = vp.create_video_writer(output_path, out_w, out_h, output_fps)

        prev_segment_last_frames: list[np.ndarray] = []
        # Per-frame metadata: source frame_id → crop info
        frame_metadata: dict[str, dict] = {}

        # Effects config
        effects_cfg = config.get("effects", {})
        spotlight_enabled = effects_cfg.get("shooter_spotlight", False)
        spotlight_color = tuple(effects_cfg.get("spotlight_color", [0, 215, 255]))
        spotlight_alpha = effects_cfg.get("spotlight_alpha", 0.4)
        trail_enabled = effects_cfg.get("ball_trail", False)
        trail_length = effects_cfg.get("trail_length", 15)
        trail_color = tuple(effects_cfg.get("trail_color", [0, 255, 255]))

        # Pre-load data needed for effects
        scorer_id = None
        if analyzed.key_players:
            scorer_id = analyzed.key_players[0].get("track_id")

        for seg_idx, segment in enumerate(analyzed.segments):
            logger.info(
                "Rendering segment '%s' [%d → %d] view=%s",
                segment.name, segment.start_frame, segment.end_frame,
                segment.view_type,
            )

            # Pre-load trajectory for closeup/medium segments
            trajectory: list[dict] | None = None
            if segment.view_type in ("closeup", "medium"):
                if segment.focus_target == "ball":
                    trajectory = self._build_ball_trajectory(
                        ctx, segment.start_frame, segment.end_frame,
                    )
                elif segment.focus_track_id is not None:
                    trajectory = ctx.get_player_trajectory(
                        segment.focus_track_id,
                        segment.start_frame,
                        segment.end_frame,
                    )

            # Pre-load scorer trajectory for spotlight effect (all segments)
            scorer_trajectory_seg: list[dict] | None = None
            if spotlight_enabled and scorer_id is not None:
                scorer_trajectory_seg = ctx.get_player_trajectory(
                    scorer_id, segment.start_frame, segment.end_frame,
                )

            prev_state: SmoothState | None = None
            segment_frames: list[np.ndarray] = []

            for frame_id, _ts, frame in vp.extract_frames(
                ctx.video_path,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame + 1,
                show_progress=False,
            ):
                # --- Draw effects on raw frame before cropping ---
                if trail_enabled:
                    self._draw_ball_trail(
                        frame, ctx, frame_id, trail_length, trail_color,
                    )

                if spotlight_enabled and scorer_trajectory_seg:
                    bbox = interpolate_bbox(scorer_trajectory_seg, frame_id)
                    if bbox is not None:
                        self._draw_spotlight(frame, bbox, spotlight_color, spotlight_alpha)

                rendered_frame, prev_state, crop_info = self._render_frame(
                    frame, frame_id, segment, trajectory,
                    out_w, out_h, out_size,
                    padding_factor if segment.view_type == "closeup"
                    else ball_padding if segment.focus_target == "ball"
                    else medium_padding,
                    smooth_alpha, prev_state,
                    overlays_enabled,
                )
                segment_frames.append(rendered_frame)

                # Record frame metadata
                if crop_info is not None:
                    frame_metadata[str(frame_id)] = {
                        "segment": segment.name,
                        "view_type": segment.view_type,
                        "center": [round(crop_info.center[0], 1),
                                   round(crop_info.center[1], 1)],
                        "crop_box": crop_info.crop_box,
                        "scale": round(crop_info.scale, 4),
                    }

            # Apply slow-motion if needed (e.g. replay segment)
            if segment.speed < 1.0 and len(segment_frames) >= 2:
                segment_frames = self._interpolate_slowmo(
                    segment_frames, segment.speed,
                )

            # Semantic transitions between segments
            if seg_idx > 0 and prev_segment_last_frames:
                transition = segment.transition
                if transition == "crossfade" and crossfade_frames > 0:
                    self._write_crossfade(
                        writer, prev_segment_last_frames, segment_frames,
                        crossfade_frames,
                    )
                    # Skip the cross-faded frames from the current segment
                    segment_frames = segment_frames[crossfade_frames:]
                elif transition == "flash":
                    self._write_flash_transition(
                        writer,
                        prev_segment_last_frames[-1],
                        segment_frames[0] if segment_frames else None,
                    )
                # "cut" — no transition, just concatenate

            for sf in segment_frames:
                writer.write(sf)

            # Save last N frames for cross-fade with next segment
            prev_segment_last_frames = (
                segment_frames[-crossfade_frames:]
                if len(segment_frames) >= crossfade_frames
                else segment_frames[:]
            )

        writer.release()

        # Save per-frame crop metadata JSON alongside the video
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(frame_metadata, f, indent=2)
        logger.info("Frame metadata saved: %s (%d frames)", meta_path, len(frame_metadata))

        logger.info("Highlight clip saved: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Ball trajectory
    # ------------------------------------------------------------------

    @staticmethod
    def _build_ball_trajectory(
        ctx: MatchContext,
        start_frame: int,
        end_frame: int,
    ) -> list[dict]:
        """Build a trajectory list from ball_tracks for use with interpolate_bbox.

        Ball tracks store `center` [cx, cy] instead of a full bbox, so we
        synthesize a small bbox around the center to reuse the same closeup
        pipeline.
        """
        trajectory: list[dict] = []
        # Synthetic ball bbox half-size (pixels) — only affects the minimum
        # crop size anchor, actual zoom is controlled by padding_factor.
        BALL_HALF = 40

        for f in range(start_frame, end_frame + 1):
            ball = ctx.get_ball_at_frame(f)
            if ball is None:
                continue
            center = ball.get("center")
            if center is None:
                continue
            cx, cy = center
            trajectory.append({
                "frame": f,
                "bbox": [cx - BALL_HALF, cy - BALL_HALF,
                         cx + BALL_HALF, cy + BALL_HALF],
            })
        return trajectory

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        frame: np.ndarray,
        frame_id: int,
        segment: ClipSegment,
        trajectory: list[dict] | None,
        out_w: int,
        out_h: int,
        closeup_size: tuple[int, int],
        padding_factor: float,
        smooth_alpha: float,
        prev_state: SmoothState | None,
        overlays_enabled: bool,
    ) -> tuple[np.ndarray, SmoothState | None, CropInfo | None]:
        """Render a single frame according to the segment view type."""
        h, w = frame.shape[:2]

        if segment.view_type in ("closeup", "medium") and trajectory is not None:
            bbox = interpolate_bbox(trajectory, frame_id)
            if bbox is not None:
                crop, state, crop_info = extract_closeup(
                    frame, bbox,
                    output_size=closeup_size,
                    padding_factor=padding_factor,
                    prev_state=prev_state,
                    smooth_alpha=smooth_alpha,
                )
                # Resize crop back to full output dimensions for consistent video
                rendered = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                if overlays_enabled and segment.overlays:
                    self._draw_overlays(rendered, segment.overlays)
                return rendered, state, crop_info
            # Track lost — hold last known crop position instead of jumping to wide
            if prev_state is not None:
                crop, state, crop_info = extract_closeup(
                    frame,
                    [prev_state.cx - prev_state.crop_w / 4,
                     prev_state.cy - prev_state.crop_h / 4,
                     prev_state.cx + prev_state.crop_w / 4,
                     prev_state.cy + prev_state.crop_h / 4],
                    output_size=closeup_size,
                    padding_factor=padding_factor,
                    prev_state=prev_state,
                    smooth_alpha=0.0,  # no smoothing, just hold position
                )
                rendered = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
                if overlays_enabled and segment.overlays:
                    self._draw_overlays(rendered, segment.overlays)
                return rendered, state, crop_info

        # Wide view — full frame, scale = 1.0
        rendered = frame
        if w != out_w or h != out_h:
            rendered = cv2.resize(rendered, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

        if overlays_enabled and segment.overlays:
            self._draw_overlays(rendered, segment.overlays)

        wide_info = CropInfo(
            center=(w / 2.0, h / 2.0),
            crop_box=[0, 0, w, h],
            scale=w / out_w if out_w > 0 else 1.0,
        )
        return rendered, prev_state, wide_info

    # ------------------------------------------------------------------
    # Visual effects
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_spotlight(
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, ...] = (0, 215, 255),
        alpha: float = 0.4,
    ) -> None:
        """Draw a soft, multi-layer glowing ellipse at the player's feet.

        Three concentric layers (large→small) with increasing opacity
        create a Gaussian-like falloff for a natural spotlight effect.
        """
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int(y2)  # bottom of bbox = feet
        base_w = int((x2 - x1) * 0.6)
        base_h = int(base_w * 0.3)
        if base_w < 5 or base_h < 3:
            return

        overlay = frame.copy()
        # Outer glow (large, faint)
        cv2.ellipse(overlay, (cx, cy), (int(base_w * 1.5), int(base_h * 1.5)),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 0.3, frame, 1 - alpha * 0.3, 0, dst=frame)
        # Mid glow
        overlay[:] = frame
        cv2.ellipse(overlay, (cx, cy), (base_w, base_h),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 0.5, frame, 1 - alpha * 0.5, 0, dst=frame)
        # Inner core (bright, small)
        overlay[:] = frame
        cv2.ellipse(overlay, (cx, cy), (int(base_w * 0.5), int(base_h * 0.5)),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, dst=frame)

    @staticmethod
    def _draw_ball_trail(
        frame: np.ndarray,
        ctx: "MatchContext",
        frame_id: int,
        trail_length: int = 15,
        color: tuple[int, ...] = (0, 255, 255),
    ) -> None:
        """Draw a comet-style trajectory trail behind the ball (in-place).

        Uses a single overlay + blend instead of per-segment copies
        (15× fewer frame copies at 1080p).  Newer segments are drawn
        brighter and thicker to create a comet/fade effect.  The trail
        always looks back *trail_length* frames from the current position,
        so it runs throughout the entire highlight.
        """
        start = max(0, frame_id - trail_length)
        points: list[tuple[int, int]] = []
        for f in range(start, frame_id + 1):
            ball = ctx.get_ball_at_frame(f)
            if ball is None:
                continue
            center = ball.get("center")
            if center is None:
                continue
            points.append((int(center[0]), int(center[1])))

        if len(points) < 2:
            return

        # Single overlay — brightness encodes the per-segment fade
        overlay = frame.copy()
        n = len(points)
        for i in range(1, n):
            progress = i / n  # 0→1, newer = brighter/thicker
            thickness = max(1, int(progress * 4 + 1))
            c = tuple(int(v * progress) for v in color)
            cv2.line(overlay, points[i - 1], points[i], c, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    @staticmethod
    def _write_flash_transition(
        writer: cv2.VideoWriter,
        last_frame: np.ndarray,
        first_frame: np.ndarray | None,
    ) -> None:
        """Write a brief white-flash transition (3 frames).

        Broadcast-style "goal confirmed" effect: previous frame fades to
        white, then white fades into the next segment.
        """
        white = np.full_like(last_frame, 255)
        # Fade out: 50/50 blend with white
        writer.write(cv2.addWeighted(last_frame, 0.5, white, 0.5, 0))
        # Peak: pure white
        writer.write(white)
        # Fade in: 50/50 blend from white
        if first_frame is not None:
            writer.write(cv2.addWeighted(first_frame, 0.5, white, 0.5, 0))

    # ------------------------------------------------------------------
    # Slow-motion
    # ------------------------------------------------------------------

    @staticmethod
    def _interpolate_slowmo(
        frames: list[np.ndarray],
        speed: float,
    ) -> list[np.ndarray]:
        """Generate slow-motion frames via linear blending.

        For speed=0.4, each source frame produces ~2.5 output frames.
        Consecutive source frames are cross-blended for smooth motion
        rather than frame duplication.
        """
        n_src = len(frames)
        n_out = int(n_src / speed)
        result: list[np.ndarray] = []
        for i in range(n_out):
            src_pos = i * speed
            src_idx = min(int(src_pos), n_src - 1)
            frac = src_pos - int(src_pos)
            if frac < 0.01 or src_idx + 1 >= n_src:
                result.append(frames[src_idx])
            else:
                blended = cv2.addWeighted(
                    frames[src_idx], 1.0 - frac,
                    frames[src_idx + 1], frac,
                    0,
                )
                result.append(blended)
        return result

    # ------------------------------------------------------------------
    # Overlays
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_overlays(frame: np.ndarray, overlays: list[dict[str, Any]]) -> None:
        """Draw text overlays on a frame (in-place)."""
        h, w = frame.shape[:2]
        for ov in overlays:
            if ov.get("type") != "text":
                continue
            text = ov.get("text", "")
            position = ov.get("position", "top_center")

            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(1.0, w / 1920 * 2.0)
            thickness = max(2, int(scale * 2))
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)

            if position == "top_center":
                x = (w - tw) // 2
                y = th + 30
            elif position == "bottom_center":
                x = (w - tw) // 2
                y = h - 30
            elif position == "center":
                x = (w - tw) // 2
                y = (h + th) // 2
            else:
                x, y = 30, th + 30

            # Background rectangle
            cv2.rectangle(
                frame,
                (x - 10, y - th - 10),
                (x + tw + 10, y + baseline + 10),
                (0, 0, 0),
                cv2.FILLED,
            )
            cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness)

    # ------------------------------------------------------------------
    # Cross-fade
    # ------------------------------------------------------------------

    @staticmethod
    def _write_crossfade(
        writer: cv2.VideoWriter,
        prev_frames: list[np.ndarray],
        next_frames: list[np.ndarray],
        num_frames: int,
    ) -> None:
        """Write a cross-fade transition between two segment boundaries."""
        n = min(num_frames, len(prev_frames), len(next_frames))
        if n == 0:
            return
        for i in range(n):
            alpha = (i + 1) / (n + 1)
            blended = cv2.addWeighted(
                prev_frames[-(n - i)], 1 - alpha,
                next_frames[i], alpha,
                0,
            )
            writer.write(blended)
