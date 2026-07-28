"""SegmentComposer — renders highlight clips from analyzed events."""

from __future__ import annotations

import logging
import re
import tempfile
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

# Restrict clip filenames to a safe charset. Components are derived from
# event metadata, which downstream code passes into a `bash -c` string for
# the video_enhancement docker mode; without sanitisation, a hostile
# events.json could inject shell commands.
_SAFE_NAME_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]")


def _sanitize(part: str) -> str:
    return _SAFE_NAME_COMPONENT.sub("_", str(part))


def _reencode_h264(path: Path) -> None:
    """Re-encode an mp4 in place to browser-playable H.264 + faststart.

    OpenCV writes MPEG-4 Part 2, which HTML5 <video> can't decode. This
    transcodes to yuv420p H.264 via ffmpeg and swaps the file. Silent
    no-op when ffmpeg is unavailable or errors — the original clip is
    left intact so the render is never lost.
    """
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not found; leaving highlight clip as mp4v "
                       "(may not play in browsers): %s", path)
        return
    tmp_out = path.with_suffix(".h264.mp4")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(path),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(tmp_out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_out.replace(path)
        logger.info("Re-encoded highlight clip to H.264: %s", path)
    except (subprocess.CalledProcessError, OSError) as exc:
        logger.warning("H.264 re-encode failed (%s); keeping mp4v clip: %s",
                       exc, path)
        tmp_out.unlink(missing_ok=True)


class _UpscaledFrameReader:
    """Disk-backed reader for upscaled video frames.

    Keeps the upscaled video on disk and reads frames on demand via
    cv2.VideoCapture, so only one frame is in memory at a time.
    Sequential reads are fast (no seek needed); random access seeks
    to the requested position.
    """

    def __init__(
        self,
        video_path: Path,
        tmpdir: str,
        frame_id_to_idx: dict[int, int],
    ) -> None:
        self._video_path = video_path
        self._tmpdir = tmpdir
        self._frame_id_to_idx = frame_id_to_idx
        self._cap = cv2.VideoCapture(str(video_path))
        self._next_idx = 0  # next sequential index the capture will return

    def read(self, frame_id: int) -> np.ndarray | None:
        """Read a single upscaled frame by its original frame_id."""
        idx = self._frame_id_to_idx.get(frame_id)
        if idx is None:
            return None

        # Seek if not at the expected position
        if idx != self._next_idx:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._next_idx = idx

        ret, frame = self._cap.read()
        if not ret:
            return None
        self._next_idx = idx + 1
        return frame

    def close(self) -> None:
        """Release the capture and clean up the temp directory."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)


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
            f"{_sanitize(event.event_type)}_{_sanitize(event.frame)}"
            f"_{_sanitize(event.metadata.get('goal_side', ''))}.mp4"
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

        # output_fps None/0 → follow the source video's fps so normal-speed
        # segments play at real time (a fixed low value slows a high-fps
        # source: 60fps source at 25fps output plays 2.4× slow).
        output_fps = config.get("output_fps") or ctx.fps
        crossfade_frames = config.get("crossfade_frames", 4)

        # Output dimensions: full frame for wide, output_size for closeup
        # Use full frame size so all segments have the same resolution
        out_w, out_h = ctx.width, ctx.height

        vp = VideoProcessor()

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

        # --- Upscale source frames before composition ---
        ve_cfg = config.get("video_enhancement", {})
        upscale_enabled = ve_cfg.get("enabled", False) and ve_cfg.get("upscale", {}).get("enabled", False)

        upscale_reader: _UpscaledFrameReader | None = None
        upscale_scale = 1
        if upscale_enabled:
            upscale_reader, upscale_scale = self._upscale_source_frames(
                analyzed, ctx, vp, ve_cfg,
            )

        # Adjust output dimensions if upscaled
        if upscale_scale > 1:
            out_w *= upscale_scale
            out_h *= upscale_scale
            out_size = (out_size[0] * upscale_scale, out_size[1] * upscale_scale)

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
                if segment.focus_target == "possession":
                    trajectory = self._build_possession_trajectory(
                        ctx, segment.start_frame, segment.end_frame,
                    )
                elif segment.focus_target == "ball":
                    trajectory = self._build_ball_trajectory(
                        ctx, segment.start_frame, segment.end_frame,
                    )
                elif segment.focus_track_id is not None:
                    trajectory = ctx.get_player_trajectory(
                        segment.focus_track_id,
                        segment.start_frame,
                        segment.end_frame,
                    )

            # Scale trajectory bboxes to match upscaled resolution
            if trajectory and upscale_scale > 1:
                trajectory = [
                    {**t, "bbox": [c * upscale_scale for c in t["bbox"]]}
                    for t in trajectory
                ]

            # Pre-load scorer trajectory for spotlight effect (all segments)
            scorer_trajectory_seg: list[dict] | None = None
            if spotlight_enabled and scorer_id is not None:
                scorer_trajectory_seg = ctx.get_player_trajectory(
                    scorer_id, segment.start_frame, segment.end_frame,
                )
                if scorer_trajectory_seg and upscale_scale > 1:
                    scorer_trajectory_seg = [
                        {**t, "bbox": [c * upscale_scale for c in t["bbox"]]}
                        for t in scorer_trajectory_seg
                    ]

            prev_state: SmoothState | None = None
            segment_frames: list[np.ndarray] = []

            if upscale_reader is not None:
                # Use pre-upscaled frames (read from disk, not memory)
                for frame_id in range(segment.start_frame, segment.end_frame + 1):
                    frame = upscale_reader.read(frame_id)
                    if frame is None:
                        continue
                    frame = frame.copy()

                    if trail_enabled:
                        self._draw_ball_trail(
                            frame, ctx, frame_id, trail_length, trail_color,
                            scale=upscale_scale,
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

                    if crop_info is not None:
                        frame_metadata[str(frame_id)] = {
                            "segment": segment.name,
                            "view_type": segment.view_type,
                            "center": [round(crop_info.center[0], 1),
                                       round(crop_info.center[1], 1)],
                            "crop_box": crop_info.crop_box,
                            "scale": round(crop_info.scale, 4),
                        }
            else:
                # Original path: read from source video
                for frame_id, _ts, frame in vp.extract_frames(
                    ctx.video_path,
                    start_frame=segment.start_frame,
                    end_frame=segment.end_frame + 1,
                    show_progress=False,
                ):
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
                    segment_frames, segment.speed, ve_cfg,
                    output_fps, output_dir,
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

        # Clean up upscaled temp directory
        if upscale_reader is not None:
            upscale_reader.close()

        # OpenCV's VideoWriter emits MPEG-4 Part 2 (``mp4v``), which
        # browsers refuse to decode (<video> raises
        # DEMUXER_ERROR_NO_SUPPORTED_STREAMS). Re-encode to H.264 so the
        # clip plays inline on the web match page. Best-effort: if
        # ffmpeg is missing or fails we keep the mp4v file rather than
        # losing the render.
        _reencode_h264(output_path)

        # Save per-frame crop metadata JSON alongside the video
        meta_path = output_path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(frame_metadata, f, indent=2)
        logger.info("Frame metadata saved: %s (%d frames)", meta_path, len(frame_metadata))

        logger.info("Highlight clip saved: %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # Source frame upscaling
    # ------------------------------------------------------------------

    @staticmethod
    def _upscale_source_frames(
        analyzed: AnalyzedEvent,
        ctx: MatchContext,
        vp: VideoProcessor,
        ve_cfg: dict[str, Any],
    ) -> tuple["_UpscaledFrameReader", int]:
        """Extract source frames, upscale via video2x, return a disk-backed reader.

        Returns:
            (reader, scale_factor) where reader provides frame-by-frame access
            to the upscaled video on disk (no bulk memory allocation), and
            scale_factor is the integer upscale multiplier (e.g. 2).
        """
        from goalinsight.video_enhancement import _make_cmd, _run_video2x, _video2x_args, _upscale_args

        upscale_cfg = ve_cfg.get("upscale", {})
        scale = upscale_cfg.get("scale", 2)
        encoder_cfg = ve_cfg.get("encoder", {"codec": "libx264", "extra": {"crf": "17", "preset": "fast"}})

        # Compute the full frame range across all segments (with trail lookback)
        trail_length = 15

        # Collect unique frame IDs needed (segments may overlap, e.g. strike + replay)
        needed: set[int] = set()
        for seg in analyzed.segments:
            for f in range(max(0, seg.start_frame - trail_length), seg.end_frame + 1):
                needed.add(f)

        # Build ordered frame list
        frame_ids_sorted = sorted(needed)

        logger.info(
            "Upscaling %d source frames [%d → %d] at %dx ...",
            len(frame_ids_sorted), frame_ids_sorted[0], frame_ids_sorted[-1], scale,
        )

        h, w = ctx.height, ctx.width

        # Use a temp directory that persists until the reader is closed
        tmpdir = tempfile.mkdtemp(prefix="_upscale_src_")
        tmp = Path(tmpdir)
        src_path = tmp / "src.mp4"
        dst_path = tmp / "upscaled.mp4"

        # Write source frames in order
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vid_writer = cv2.VideoWriter(str(src_path), fourcc, ctx.fps, (w, h))
        for frame_id, _ts, frame in vp.extract_frames(
            ctx.video_path,
            start_frame=frame_ids_sorted[0],
            end_frame=frame_ids_sorted[-1] + 1,
            show_progress=False,
        ):
            if frame_id in needed:
                vid_writer.write(frame)
        vid_writer.release()

        # Run video2x upscale
        processor = upscale_cfg.get("processor", "realesrgan")
        args = _video2x_args(processor, _upscale_args(upscale_cfg), encoder_cfg)
        cmd = _make_cmd(None, src_path, dst_path, args, ve_cfg)
        _run_video2x(cmd, "highlight_source", "upscale")

        # Remove source video to free disk space
        src_path.unlink(missing_ok=True)

        # Build frame_id → sequential index mapping
        frame_id_to_idx = {fid: i for i, fid in enumerate(frame_ids_sorted)}

        reader = _UpscaledFrameReader(dst_path, tmpdir, frame_id_to_idx)
        logger.info("Upscaled %d frames (output res: %dx%d)", len(frame_ids_sorted), w * scale, h * scale)
        return reader, scale

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

    @staticmethod
    def _build_possession_trajectory(
        ctx: MatchContext,
        start_frame: int,
        end_frame: int,
    ) -> list[dict]:
        """Build a trajectory that follows the player in possession.

        For each frame, pick the player whose bbox-bottom (feet) is
        closest to the ball and return that player's bbox. As the ball
        moves from passer to receiver, the focus hands off between players
        — the camera tracks the human action rather than the tiny ball.

        Falls back to the ball's synthetic bbox on frames where no player
        is near enough (e.g. ball in flight), so the crop keeps moving
        with play instead of freezing.
        """
        # Max feet-to-ball distance (px) to consider a player "in
        # possession". Beyond this the ball is in flight between players.
        POSSESSION_MAX_PX = 250
        BALL_HALF = 40

        trajectory: list[dict] = []
        for f in range(start_frame, end_frame + 1):
            ball = ctx.get_ball_at_frame(f)
            if ball is None:
                continue
            center = ball.get("center")
            if center is None:
                continue
            bx, by = center

            best_bbox = None
            best_dist = POSSESSION_MAX_PX
            for t in ctx.get_tracks_at_frame(f):
                bbox = t.get("bbox")
                if not bbox:
                    continue
                x1, y1, x2, y2 = bbox
                feet_x = (x1 + x2) / 2
                feet_y = y2
                dist = ((feet_x - bx) ** 2 + (feet_y - by) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_bbox = bbox

            if best_bbox is not None:
                trajectory.append({"frame": f, "bbox": list(best_bbox)})
            else:
                # Ball in flight — fall back to a ball-centred crop so the
                # framing follows play toward the receiver.
                trajectory.append({
                    "frame": f,
                    "bbox": [bx - BALL_HALF, by - BALL_HALF,
                             bx + BALL_HALF, by + BALL_HALF],
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
        base_w = int((x2 - x1) * 0.8)
        base_h = int(base_w * 0.35)
        if base_w < 5 or base_h < 3:
            return

        overlay = frame.copy()
        # Outer glow (large, faint)
        cv2.ellipse(overlay, (cx, cy), (int(base_w * 1.8), int(base_h * 1.8)),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, dst=frame)
        # Mid glow
        overlay[:] = frame
        cv2.ellipse(overlay, (cx, cy), (base_w, base_h),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, dst=frame)
        # Inner core (bright, small)
        overlay[:] = frame
        cv2.ellipse(overlay, (cx, cy), (int(base_w * 0.5), int(base_h * 0.5)),
                     0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha * 1.0, frame, 1 - alpha * 1.0, 0, dst=frame)
        # Bright rim outline
        cv2.ellipse(frame, (cx, cy), (base_w, base_h),
                     0, 0, 360, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_ball_trail(
        frame: np.ndarray,
        ctx: "MatchContext",
        frame_id: int,
        trail_length: int = 15,
        color: tuple[int, ...] = (0, 255, 255),
        scale: int = 1,
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
            points.append((int(center[0] * scale), int(center[1] * scale)))

        if len(points) < 2:
            return

        # Single overlay — brightness encodes the per-segment fade
        overlay = frame.copy()
        n = len(points)
        for i in range(1, n):
            progress = i / n  # 0→1, newer = brighter/thicker
            thickness = max(2, int(progress * 6 + 2))
            c = tuple(int(v * (0.3 + 0.7 * progress)) for v in color)
            cv2.line(overlay, points[i - 1], points[i], c, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)
        # Bright dot at current ball position
        cv2.circle(frame, points[-1], max(3, thickness // 2), color, -1, cv2.LINE_AA)

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
        ve_cfg: dict[str, Any] | None = None,
        output_fps: float = 25.0,
        work_dir: Path | None = None,
    ) -> list[np.ndarray]:
        """Generate slow-motion frames.

        When video_enhancement config is available, uses RIFE via video2x
        for high-quality optical-flow interpolation.  Falls back to linear
        blending when video2x is not configured.
        """
        if ve_cfg and ve_cfg.get("enabled", False) and work_dir is not None:
            try:
                return SegmentComposer._interpolate_slowmo_rife(
                    frames, speed, ve_cfg, output_fps, work_dir,
                )
            except Exception as exc:
                logger.warning("RIFE slowmo failed, falling back to linear blend: %s", exc)

        # Fallback: linear blending
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

    @staticmethod
    def _interpolate_slowmo_rife(
        frames: list[np.ndarray],
        speed: float,
        ve_cfg: dict[str, Any],
        output_fps: float,
        work_dir: Path,
    ) -> list[np.ndarray]:
        """Slow-motion via RIFE frame interpolation (video2x).

        Writes source frames to a temp video at *output_fps*, runs RIFE to
        multiply the frame count by ``1/speed`` (rounded to power of 2),
        then reads back the interpolated frames.
        """
        from goalinsight.video_enhancement import _make_cmd, _run_video2x, _video2x_args

        # Compute RIFE multiplier: round 1/speed to nearest power of 2
        raw_mult = 1.0 / speed
        rife_mult = 2
        while rife_mult * 2 <= raw_mult + 0.5:
            rife_mult *= 2

        h, w = frames[0].shape[:2]

        with tempfile.TemporaryDirectory(dir=work_dir, prefix="_slowmo_") as tmpdir:
            tmp = Path(tmpdir)
            src_path = tmp / "src.mp4"
            dst_path = tmp / "dst.mp4"

            # Write source frames
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(src_path), fourcc, output_fps, (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

            # Run RIFE interpolation via video2x
            encoder_cfg = ve_cfg.get("encoder", {"codec": "libx264", "extra": {"crf": "17", "preset": "fast"}})
            args = _video2x_args("rife", ["-m", str(rife_mult)], encoder_cfg)
            cmd = _make_cmd(None, src_path, dst_path, args, ve_cfg)
            _run_video2x(cmd, "slowmo_replay", "interpolate")

            # Read back interpolated frames
            cap = cv2.VideoCapture(str(dst_path))
            result: list[np.ndarray] = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                result.append(frame)
            cap.release()

        if not result:
            raise RuntimeError("RIFE produced no output frames")

        logger.info(
            "RIFE slowmo: %d source → %d interpolated (×%d)",
            len(frames), len(result), rife_mult,
        )
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
