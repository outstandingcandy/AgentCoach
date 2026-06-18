"""Per-player spotlight clip renderer.

Produces ``<run>/player_profile/spotlights/<pid>.mp4`` — a follow-cam
clip that crops the original (often 4K) source video around a single
player so the player stays centered and ~``target_player_height_frac``
tall in a 1080p frame.

Optional overlays (all on by default):

- **Spotlight ellipse** at the player's feet.
- **Pitch trail** showing the past ``trail_seconds`` of pitch motion,
  reprojected to source-pixel space via the per-frame homography.
- **Name badge** (jersey number + role + team color block) in the top-left.

Frames where the player is NOT in view are skipped (presence-only) so the
output is a tight stitch of every appearance — gaps under
``presence_gap_seconds`` are bridged by interpolation; longer gaps are cut.

The renderer streams BGR frames into a single ffmpeg process via stdin
(same pattern as ``goalinsight.annotated_video._renderer``) so it never
writes intermediate frames to disk.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..highlights._closeup import (
    SmoothState,
    extract_closeup,
    interpolate_bbox,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_player_spotlight(
    *,
    video_path: Path,
    player_id: str,
    trajectory: list[dict],
    pitch_positions_by_frame: dict[int, list[tuple[float, float]]] | None = None,
    homographies: dict[int, np.ndarray] | None = None,
    fps: float,
    output_path: Path,
    output_size: tuple[int, int] = (1920, 1080),
    target_player_height_frac: float = 2 / 3,
    presence_gap_seconds: float = 1.0,
    enable_ellipse: bool = True,
    enable_trail: bool = True,
    enable_name_badge: bool = True,
    trail_seconds: float = 1.5,
    name_badge_text: str | None = None,
    name_badge_team_color: tuple[int, int, int] | None = None,
    crf: int = 23,
    preset: str = "medium",
) -> dict[str, Any]:
    """Render a follow-cam spotlight mp4 for one player.

    Args:
        video_path: Source video (preferably 4K for best crop quality).
        player_id: Consolidated player id (used for logging only).
        trajectory: List of ``{"frame": int, "bbox": [x1,y1,x2,y2]}``
            sorted by frame; sparse is fine — gaps are interpolated.
        pitch_positions_by_frame: ``frame -> [(pitch_x, pitch_y), ...]``
            historical pitch coordinates for the trail. Required when
            ``enable_trail`` is True. Caller is responsible for trimming
            this to the relevant player; we just reproject.
        homographies: ``frame -> 3x3`` world→pixel matrix (sparse). Required
            for trail; we use nearest-anchor matching.
        fps: Source video fps; used for output rate and trail-length math.
        output_path: Destination ``.mp4``.
        output_size: ``(width, height)``; default 1080p.
        target_player_height_frac: Player bbox height after resize as a
            fraction of ``output_size[1]``. 2/3 → ~720 px on 1080p.
        presence_gap_seconds: Gaps between consecutive detections shorter
            than this are bridged with linear interpolation; longer gaps
            cut the clip (the player is treated as having left the frame).
        crf / preset: x264 knobs.

    Returns:
        ``{"frames_rendered": int, "duration_s": float}``.
    """
    if not trajectory:
        raise ValueError(f"empty trajectory for player {player_id!r}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = output_size
    if out_w > src_w or out_h > src_h:
        # Asking for 1080p output from a smaller source is fine in
        # principle (cv2.resize will upscale), but warn so kids/test
        # fixtures with small sources are obvious.
        logger.warning(
            "[%s] source %dx%d smaller than requested output %dx%d; "
            "spotlight will upscale", player_id, src_w, src_h, out_w, out_h,
        )

    # Padding factor: extract_closeup picks crop_h = max(bbox_h*pad,
    # crop_w/aspect). For typical taller-than-wide player bboxes the
    # bbox_h branch wins, so player_h_after_resize = (y2-y1)/crop_h * out_h
    # = out_h / pad. Setting pad = 1/target_frac yields the requested
    # player-height fraction.
    padding_factor = max(1.05, 1.0 / max(target_player_height_frac, 0.05))

    # Frame ranges — group consecutive detections into "presence segments".
    gap_frames = max(1, int(round(presence_gap_seconds * fps)))
    segments = _build_presence_segments(trajectory, gap_frames=gap_frames)
    total_frames_to_render = sum(s[1] - s[0] + 1 for s in segments)
    if total_frames_to_render == 0:
        cap.release()
        raise ValueError(f"no presence frames for player {player_id!r}")

    # Sort homography keys once; we'll bisect for nearest-anchor lookups.
    h_keys: list[int] = sorted(homographies.keys()) if homographies else []

    # Trail pre-index: per-frame list of pitch_positions (we'll project at
    # render time using the homography for that frame).
    trail_window = max(1, int(round(trail_seconds * fps)))

    # ffmpeg stdin pipeline — same flags as annotated_video/_renderer.py.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24", "-r", f"{fps:.3f}",
        "-i", "-",
        "-c:v", "libx264", "-crf", str(int(crf)), "-preset", str(preset),
        "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
        "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
        "-movflags", "+faststart", "-an",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        cap.release()
        raise RuntimeError("ffmpeg stdin not available")

    smooth_state: SmoothState | None = None
    frames_rendered = 0
    # Per-output-frame mapping back to the source broadcast frame index.
    # The frontend uses this to keep the right-side 2D pitch + selected-
    # player highlight in sync while the spotlight clip plays — output
    # frame N (i.e. ``video.currentTime * fps == N``) was rendered from
    # source frame ``broadcast_frames[N]``.
    broadcast_frames: list[int] = []
    t0 = time.time()

    # Recent pitch positions for trail (frame, pitch_x, pitch_y).
    trail_buffer: deque[tuple[int, float, float]] = deque(maxlen=trail_window)

    try:
        for seg_start, seg_end in segments:
            # Reset smoothing across hard cuts so the camera doesn't drift
            # in from the previous segment's last position.
            smooth_state = None
            trail_buffer.clear()

            cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)

            for fid in range(seg_start, seg_end + 1):
                ok, frame = cap.read()
                if not ok:
                    break

                bbox = interpolate_bbox(
                    trajectory, fid, max_extrapolate=gap_frames,
                )
                if bbox is None:
                    # Inside the segment, this should be rare (gap_frames
                    # already accounts for sparse detections).
                    continue

                # Trail update: stash today's pitch position so future frames
                # can fade it. We append AFTER drawing so the trail shows
                # *past* positions only, not the current foot location.
                if pitch_positions_by_frame is not None:
                    pps = pitch_positions_by_frame.get(fid)
                    if pps:
                        # If multiple positions for one player one frame
                        # (shouldn't happen post-consolidation), use first.
                        px, py = pps[0]
                        # We push *after* draw so don't add yet.
                        pending_trail = (fid, px, py)
                    else:
                        pending_trail = None
                else:
                    pending_trail = None

                # Draw overlays on the SOURCE frame, then crop. This way
                # the ellipse + trail + badge all scale with the closeup
                # rather than living in the output frame at fixed pixel
                # sizes.
                if enable_ellipse:
                    _draw_spotlight_ellipse(frame, bbox)
                if enable_trail and trail_buffer and homographies:
                    _draw_pitch_trail(
                        frame, trail_buffer, h_keys, homographies,
                    )

                cropped, smooth_state, _info = extract_closeup(
                    frame, bbox,
                    output_size=output_size,
                    padding_factor=padding_factor,
                    prev_state=smooth_state,
                )

                if enable_name_badge and name_badge_text:
                    _draw_name_badge(
                        cropped, name_badge_text,
                        team_color=name_badge_team_color,
                    )

                proc.stdin.write(
                    cropped.tobytes() if cropped.flags["C_CONTIGUOUS"]
                    else np.ascontiguousarray(cropped).tobytes()
                )
                broadcast_frames.append(fid)
                frames_rendered += 1

                if pending_trail is not None:
                    trail_buffer.append(pending_trail)

                if frames_rendered % 500 == 0:
                    elapsed = time.time() - t0
                    rate = frames_rendered / elapsed if elapsed > 0 else 0
                    logger.info(
                        "[%s] %d frames @ %.1f fps", player_id, frames_rendered, rate,
                    )

    finally:
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        cap.release()
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {rc} for player {player_id!r}"
            )

    return {
        "frames_rendered": frames_rendered,
        "duration_s": frames_rendered / fps if fps else 0.0,
        "broadcast_frames": broadcast_frames,
    }


# ---------------------------------------------------------------------------
# Presence segmentation
# ---------------------------------------------------------------------------


def _build_presence_segments(
    trajectory: list[dict],
    *,
    gap_frames: int,
) -> list[tuple[int, int]]:
    """Group sorted trajectory observations into [start, end] segments.

    Two consecutive detections separated by more than ``gap_frames`` start
    a new segment. Inside a segment the player is treated as on-screen
    even at frames without a detection (interpolated by extract_closeup
    via interpolate_bbox).
    """
    if not trajectory:
        return []
    sorted_traj = sorted(trajectory, key=lambda t: t["frame"])
    segments: list[tuple[int, int]] = []
    seg_start = sorted_traj[0]["frame"]
    seg_end = seg_start
    for t in sorted_traj[1:]:
        f = t["frame"]
        if f - seg_end > gap_frames:
            segments.append((seg_start, seg_end))
            seg_start = f
        seg_end = f
    segments.append((seg_start, seg_end))
    return segments


# ---------------------------------------------------------------------------
# Overlay drawing — kept inline (not a class) so this module stays
# import-cheap. _draw_spotlight_ellipse mirrors SegmentComposer's three-
# layer falloff (highlights/agents/segment_composer.py:_draw_spotlight)
# but doesn't pull the highlights system in.
# ---------------------------------------------------------------------------


def _draw_spotlight_ellipse(
    frame: np.ndarray,
    bbox: list[float] | tuple[float, ...],
    color: tuple[int, int, int] = (0, 215, 255),
    alpha: float = 0.4,
) -> None:
    x1, y1, x2, y2 = bbox
    cx = int((x1 + x2) / 2)
    cy = int(y2)  # feet
    base_w = int((x2 - x1) * 0.8)
    base_h = int(base_w * 0.35)
    if base_w < 5 or base_h < 3:
        return

    overlay = frame.copy()
    cv2.ellipse(overlay, (cx, cy), (int(base_w * 1.8), int(base_h * 1.8)),
                0, 0, 360, color, -1)
    cv2.addWeighted(overlay, alpha * 0.4, frame, 1 - alpha * 0.4, 0, dst=frame)
    overlay[:] = frame
    cv2.ellipse(overlay, (cx, cy), (base_w, base_h),
                0, 0, 360, color, -1)
    cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.7, 0, dst=frame)
    overlay[:] = frame
    cv2.ellipse(overlay, (cx, cy), (int(base_w * 0.5), int(base_h * 0.5)),
                0, 0, 360, color, -1)
    cv2.addWeighted(overlay, alpha * 1.0, frame, 1 - alpha * 1.0, 0, dst=frame)
    cv2.ellipse(frame, (cx, cy), (base_w, base_h),
                0, 0, 360, color, 2, cv2.LINE_AA)


def _draw_pitch_trail(
    frame: np.ndarray,
    trail: "deque[tuple[int, float, float]]",
    h_keys: list[int],
    homographies: dict[int, np.ndarray],
) -> None:
    """Draw a fading polyline through the player's recent pitch path.

    Each ``(frame_id, pitch_x, pitch_y)`` entry is reprojected to source-
    pixel space using the homography for the closest anchor frame
    (homographies are sparse). Older points fade and thin out.
    """
    if not trail or not h_keys:
        return

    pts: list[tuple[int, int]] = []
    for fid, px, py in trail:
        H = _nearest_homography(homographies, h_keys, fid)
        if H is None:
            continue
        v = np.asarray([px, py, 1.0], dtype=np.float64)
        w = H @ v
        if abs(w[2]) < 1e-9:
            continue
        u = w[0] / w[2]
        v_ = w[1] / w[2]
        pts.append((int(u), int(v_)))

    if len(pts) < 2:
        return

    n = len(pts)
    base_color = (60, 220, 255)  # warm yellow
    for i in range(1, n):
        # Older segments fade and thin.
        age = (n - i) / n  # 0 = newest, 1 = oldest
        thickness = max(2, int(round(8 * (1 - age))))
        # Blend toward black for older segments.
        c = tuple(int(round(ch * (1 - age * 0.7))) for ch in base_color)
        cv2.line(frame, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)


def _nearest_homography(
    homographies: dict[int, np.ndarray],
    sorted_keys: list[int],
    frame: int,
) -> np.ndarray | None:
    if not sorted_keys:
        return None
    # Binary search via bisect. We accept the closer of left / right.
    import bisect
    pos = bisect.bisect_left(sorted_keys, frame)
    candidates: list[int] = []
    if pos < len(sorted_keys):
        candidates.append(sorted_keys[pos])
    if pos > 0:
        candidates.append(sorted_keys[pos - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda k: abs(k - frame))
    return homographies.get(best)


def _draw_name_badge(
    frame: np.ndarray,
    text: str,
    *,
    team_color: tuple[int, int, int] | None = None,
) -> None:
    """Top-left rounded badge: team color block + jersey/role text.

    Drawn on the OUTPUT frame (post-crop, 1080p) so the badge size is
    consistent across players regardless of crop scale.
    """
    h, w = frame.shape[:2]
    # Badge dimensions scale gently with output height so 720p / 1080p
    # both look reasonable.
    pad = max(8, int(h * 0.012))
    badge_h = max(32, int(h * 0.045))
    color_block_w = badge_h  # square
    # Measure text to size the right half.
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = badge_h / 60.0
    thickness = max(1, int(badge_h / 32))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    text_pad = int(badge_h * 0.4)
    badge_w = color_block_w + text_pad * 2 + tw

    x0 = pad
    y0 = pad
    x1 = x0 + badge_w
    y1 = y0 + badge_h

    # Rounded background — a single filled rectangle is fine; keep it
    # simple and dark with mild transparency.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (24, 28, 32), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, dst=frame)

    # Team color block (left half).
    if team_color is not None:
        cv2.rectangle(
            frame, (x0, y0), (x0 + color_block_w, y1),
            tuple(int(c) for c in team_color), -1,
        )

    # Text (right half).
    text_x = x0 + color_block_w + text_pad
    text_y = y0 + (badge_h + th) // 2 - 2
    cv2.putText(
        frame, text, (text_x, text_y),
        font, scale, (240, 245, 255), thickness, cv2.LINE_AA,
    )

    # Thin border for legibility on busy pitch backgrounds.
    cv2.rectangle(frame, (x0, y0), (x1, y1), (200, 220, 240), 1, cv2.LINE_AA)
