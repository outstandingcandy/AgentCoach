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

            prev_state: SmoothState | None = None
            segment_frames: list[np.ndarray] = []

            for frame_id, _ts, frame in vp.extract_frames(
                ctx.video_path,
                start_frame=segment.start_frame,
                end_frame=segment.end_frame + 1,
                show_progress=False,
            ):
                rendered_frame, prev_state, crop_info = self._render_frame(
                    frame, frame_id, segment, trajectory,
                    out_w, out_h, out_size,
                    padding_factor if segment.view_type == "closeup" else medium_padding,
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

            # Cross-fade between segments
            if seg_idx > 0 and prev_segment_last_frames and crossfade_frames > 0:
                self._write_crossfade(
                    writer, prev_segment_last_frames, segment_frames,
                    crossfade_frames,
                )
                # Skip the cross-faded frames from the current segment
                segment_frames = segment_frames[crossfade_frames:]

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
            # Fallback to wide if bbox not available
            prev_state = None

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
