"""Per-frame HUD renderer for the annotated_video stage.

Reads frames from the source video sequentially, overlays HUD elements
using primitives from ``goalinsight.highlights.agents.segment_composer``
and ``goalinsight.tracking.tracking_visualization``, and writes them to
an ``mp4v`` intermediate.  The outer module handles the ffmpeg re-encode.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..highlights._context import MatchContext
from ..highlights.agents.segment_composer import SegmentComposer
from ..tracking.tracking_visualization import (
    TEAM_COLORS,
    draw_topdown_pitch,
    get_color_for_track,
)

logger = logging.getLogger(__name__)

_VALID_TEAMS = ("team_A", "team_B")


def _tid_key(t: dict) -> str:
    """String key for a track — works for int track_ids *and* string
    player_ids (as produced by the track_consolidation stage)."""
    return str(t["track_id"])


def _tid_color_seed(key: str) -> int:
    """Stable integer seed used by :func:`get_color_for_track`'s
    fallback branch when the team is unknown."""
    # Prefer trailing integer suffix (e.g. "A-9" → 9); fall back to hash.
    tail = key.rsplit("-", 1)[-1]
    if tail.lstrip("-").isdigit():
        return int(tail)
    return abs(hash(key)) % 100000


class AnnotatedVideoRenderer:
    """Render a HUD-annotated MP4 from a populated :class:`MatchContext`."""

    def __init__(self, ctx: MatchContext, config: dict[str, Any]):
        self.ctx = ctx
        self.cfg = config

        # Track_id → team lookup.  Keyed under both the string form and
        # the numeric form (when the key is an integer string) so the
        # dict can be shared with :func:`draw_topdown_pitch`, which looks
        # up by whatever type the track records use.
        self.team_map: dict = {}
        for k, v in ctx.team_assignments.items():
            self.team_map[str(k)] = v
            try:
                self.team_map[int(k)] = v
            except (TypeError, ValueError):
                pass

        # Tracking is typically produced at a sub-sampled frame rate
        # (e.g. every 3rd source frame).  Precompute sorted integer
        # keys for both players and ball so we can snap each source
        # frame to its nearest prior tracked frame.
        self._track_keys: list[int] = sorted(
            int(k) for k in ctx.player_tracks.keys() if k.isdigit()
        )
        self._ball_keys: list[int] = sorted(
            int(k) for k in ctx.ball_tracks.keys() if k.isdigit()
        )

        # Precompute ball pitch positions indexed by frame for the minimap
        # trail — saves a dict lookup + key coercion per frame.
        self._ball_pitch_by_frame: dict[int, list[float]] = {}
        for k, b in ctx.ball_tracks.items():
            if not b:
                continue
            pp = b.get("pitch_position")
            if pp is None:
                continue
            try:
                self._ball_pitch_by_frame[int(k)] = [float(pp[0]), float(pp[1])]
            except (TypeError, ValueError):
                continue

        # Speed state
        spd_cfg = self.cfg.get("speed", {})
        self._spd_window = int(spd_cfg.get("window_frames", 5))
        self._spd_alpha = float(spd_cfg.get("ema_alpha", 0.4))
        self._spd_min_samples = int(spd_cfg.get("min_samples", 2))
        self._hist: dict[str, deque] = {}
        self._speeds_mps: dict[str, float] = {}
        self._last_seen: dict[str, int] = {}

        # Carrier state
        car_cfg = self.cfg.get("carrier", {})
        self._carrier_max_d = float(car_cfg.get("max_distance_m", 2.0))
        self._carrier_exclude_gk = bool(car_cfg.get("exclude_goalkeeper", True))
        self._carrier_hyst = int(car_cfg.get("hysteresis_frames", 5))
        self._carrier_color = tuple(int(c) for c in car_cfg.get(
            "outline_color", [0, 255, 255]))
        self._carrier_id: str | None = None
        self._pending_carrier: str | None = None
        self._pending_count: int = 0
        self._carrier_switches: int = 0

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    def render(self, output_path: Path) -> dict[str, Any]:
        """Render the annotated video directly to *output_path* (final mp4).

        Streams BGR frames into a single ``ffmpeg`` process via stdin so
        we never write a giant mp4v intermediate to disk and never
        re-decode it.  This is roughly 4-5× faster end-to-end than
        the previous "mp4v intermediate + libx264 re-encode" two-pass
        pipeline.
        """
        import subprocess
        import time as _time
        cap = cv2.VideoCapture(str(self.ctx.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.ctx.video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or self.ctx.fps or 25.0
        out_fps_cfg = self.cfg.get("output_fps")
        out_fps = float(out_fps_cfg) if out_fps_cfg else float(src_fps)

        enc_cfg = self.cfg.get("encoder", {})
        crf = str(int(enc_cfg.get("crf", 23)))
        preset = str(enc_cfg.get("preset", "veryfast"))
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{out_fps:.3f}",
            "-i", "-",
            "-c:v", "libx264", "-crf", crf, "-preset", preset,
            "-profile:v", "high", "-level", "4.0", "-pix_fmt", "yuv420p",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-movflags", "+faststart", "-an",
            str(output_path),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        if proc.stdin is None:
            cap.release()
            raise RuntimeError("ffmpeg stdin not available")

        frame_id = 0
        t0 = _time.time()
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                annotated = self._render_frame(frame, frame_id, w, h)
                # bgr24 raw, contiguous
                proc.stdin.write(
                    annotated.tobytes() if annotated.flags["C_CONTIGUOUS"]
                    else np.ascontiguousarray(annotated).tobytes()
                )
                frame_id += 1
                if frame_id % 500 == 0:
                    self._cleanup_stale(frame_id)
                if frame_id % 1000 == 0:
                    elapsed = _time.time() - t0
                    fps = frame_id / elapsed if elapsed > 0 else 0
                    logger.info(
                        "  annotated_video: %d frames @ %.1f fps", frame_id, fps,
                    )
                    print(f"    {frame_id} frames @ {fps:.1f} fps", flush=True)
        finally:
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            cap.release()
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited with code {rc}")

        return {"frames": frame_id, "carrier_switches": self._carrier_switches}

    # ------------------------------------------------------------------
    # Per-frame pipeline
    # ------------------------------------------------------------------

    def _render_frame(
        self, frame: np.ndarray, frame_id: int, w: int, h: int,
    ) -> np.ndarray:
        t_frame = self._snap_frame(self._track_keys, frame_id)
        b_frame = self._snap_frame(self._ball_keys, frame_id)
        tracks = (
            self.ctx.get_tracks_at_frame(t_frame) if t_frame is not None else []
        )
        ball = (
            self.ctx.get_ball_at_frame(b_frame) if b_frame is not None else None
        )

        # Advance speed history keyed on the sub-sampled tracking frame so
        # the displacement / elapsed-seconds calculation stays sound.
        self._update_speeds(tracks, t_frame if t_frame is not None else frame_id)
        carrier = self._pick_carrier(tracks, ball)

        # 1. Per-player single-colour outline rings (no fill, no shadow).
        carrier_tid = (
            _tid_key(carrier) if carrier is not None else None
        )
        if self.cfg.get("glow", {}).get("enabled", True):
            self._draw_player_rings(frame, tracks, skip_tid=carrier_tid)
            # Ball-carrier on top: filled spotlight glow (different visual)
            if carrier is not None:
                bbox = carrier.get("bbox")
                if bbox is not None:
                    self._draw_carrier_glow(
                        frame, bbox, self._carrier_color,
                        alpha=float(self.cfg.get("glow", {}).get("alpha", 0.35)),
                    )

        # 3. ID labels (speed and inter-player distance overlays removed)
        self._draw_labels(frame, tracks)

        # 4. Ball trail — reuse SegmentComposer primitive.  The helper
        # walks ``ctx.get_ball_at_frame(f)`` over the window, so we pass
        # the nearest tracking-sample frame so sub-sampled trails line up.
        bt_cfg = self.cfg.get("ball_trail", {})
        if bt_cfg.get("enabled", True) and b_frame is not None:
            SegmentComposer._draw_ball_trail(
                frame, self.ctx, b_frame,
                trail_length=int(bt_cfg.get("length", 15)),
                color=tuple(int(c) for c in bt_cfg.get("color", [0, 255, 255])),
                scale=1,
            )

        # 5. Minimap (bottom-right)
        if self.cfg.get("minimap", {}).get("enabled", True):
            trail_ref = b_frame if b_frame is not None else frame_id
            self._composite_minimap(frame, tracks, ball, trail_ref, w, h)

        return frame

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    @staticmethod
    def _snap_frame(keys: list[int], frame_id: int) -> int | None:
        """Snap *frame_id* to the nearest key in *keys* within a 10-frame
        window.  Returns None if no key is close enough."""
        if not keys:
            return None
        # Binary search for the insertion point
        lo, hi = 0, len(keys)
        while lo < hi:
            mid = (lo + hi) // 2
            if keys[mid] <= frame_id:
                lo = mid + 1
            else:
                hi = mid
        candidates: list[int] = []
        if lo > 0:
            candidates.append(keys[lo - 1])
        if lo < len(keys):
            candidates.append(keys[lo])
        if not candidates:
            return None
        best = min(candidates, key=lambda k: abs(k - frame_id))
        return best if abs(best - frame_id) <= 10 else None

    def _update_speeds(self, tracks: list[dict], frame_id: int) -> None:
        fps = self.ctx.fps or 25.0
        for t in tracks:
            pp = t.get("pitch_position")
            if pp is None:
                continue
            tid = _tid_key(t)
            self._last_seen[tid] = frame_id
            hist = self._hist.get(tid)
            if hist is None:
                hist = deque(maxlen=self._spd_window + 1)
                self._hist[tid] = hist
            hist.append((frame_id, float(pp[0]), float(pp[1])))
            if len(hist) < self._spd_min_samples:
                continue
            f0, x0, y0 = hist[0]
            f1, x1, y1 = hist[-1]
            dt = (f1 - f0) / fps
            if dt <= 0:
                continue
            inst = math.hypot(x1 - x0, y1 - y0) / dt
            inst = max(0.0, min(15.0, inst))
            prev = self._speeds_mps.get(tid, inst)
            self._speeds_mps[tid] = self._spd_alpha * inst + \
                (1.0 - self._spd_alpha) * prev

    def _cleanup_stale(self, frame_id: int) -> None:
        ttl = max(25, self._spd_window * 5)
        cutoff = frame_id - ttl
        stale = [tid for tid, f in self._last_seen.items() if f < cutoff]
        for tid in stale:
            self._hist.pop(tid, None)
            self._speeds_mps.pop(tid, None)
            self._last_seen.pop(tid, None)

    # ------------------------------------------------------------------
    # Ball carrier
    # ------------------------------------------------------------------

    def _pick_carrier(
        self, tracks: list[dict], ball: dict | None,
    ) -> dict | None:
        if ball is None:
            inst = None
        else:
            bpp = ball.get("pitch_position")
            if bpp is None:
                inst = None
            else:
                bx, by = float(bpp[0]), float(bpp[1])
                best_tid: str | None = None
                best_d = float("inf")
                for t in tracks:
                    pp = t.get("pitch_position")
                    if pp is None:
                        continue
                    team = t.get("team")
                    if team not in _VALID_TEAMS:
                        continue
                    if self._carrier_exclude_gk and \
                            t.get("role") == "goalkeeper":
                        continue
                    d = math.hypot(float(pp[0]) - bx, float(pp[1]) - by)
                    if d < best_d:
                        best_d = d
                        best_tid = _tid_key(t)
                inst = best_tid if best_d <= self._carrier_max_d else None

        if inst is None:
            if self._carrier_id is not None:
                self._carrier_switches += 1
            self._carrier_id = None
            self._pending_carrier = None
            self._pending_count = 0
        elif inst == self._carrier_id:
            self._pending_carrier = None
            self._pending_count = 0
        else:
            if inst == self._pending_carrier:
                self._pending_count += 1
            else:
                self._pending_carrier = inst
                self._pending_count = 1
            if self._pending_count >= self._carrier_hyst:
                self._carrier_id = inst
                self._carrier_switches += 1
                self._pending_carrier = None
                self._pending_count = 0

        if self._carrier_id is None:
            return None
        for t in tracks:
            if _tid_key(t) == self._carrier_id:
                return t
        return None

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw_player_rings(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        skip_tid: str | None = None,
    ) -> None:
        """Draw a thin team-coloured ellipse under each player's feet.

        No fill, no addWeighted — just one ``cv2.ellipse`` per track on
        the frame directly.  Cheap (no copy of a 1080p frame) and
        matches the requested "single-colour ring" look.
        """
        for t in tracks:
            tid = _tid_key(t)
            if skip_tid is not None and tid == skip_tid:
                continue
            bbox = t.get("bbox")
            if bbox is None:
                continue
            x1, _y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int(y2)
            base_w = int((x2 - x1) * 0.8)
            base_h = int(base_w * 0.35)
            if base_w < 5 or base_h < 3:
                continue
            team = self.team_map.get(tid, t.get("team", "unknown"))
            color = get_color_for_track(_tid_color_seed(tid), team)
            cv2.ellipse(frame, (cx, cy), (base_w, base_h),
                        0, 0, 360, color, 2, cv2.LINE_AA)

    @staticmethod
    def _draw_carrier_glow(
        frame: np.ndarray,
        bbox: list[float],
        color: tuple[int, ...],
        alpha: float = 0.35,
    ) -> None:
        """Soft ground-shadow glow ring under the ball-carrier's feet.

        Two filled ellipses (outer + mid) blended onto the frame, then a
        bright thin rim drawn on top.  No bounding-box outline.
        """
        x1, _y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        base_w = int((x2 - x1) * 0.9)
        base_h = int(base_w * 0.35)
        if base_w < 5 or base_h < 3:
            return
        overlay = frame.copy()
        cv2.ellipse(overlay, (cx, cy),
                    (int(base_w * 1.6), int(base_h * 1.6)),
                    0, 0, 360, color, -1)
        cv2.ellipse(overlay, (cx, cy), (base_w, base_h),
                    0, 0, 360, color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, dst=frame)
        cv2.ellipse(frame, (cx, cy), (base_w, base_h),
                    0, 0, 360, color, 2, cv2.LINE_AA)

    def _pick_distance_edges(
        self, carrier: dict, tracks: list[dict],
    ) -> list[tuple[dict, float]]:
        cfg = self.cfg.get("distance_network", {})
        n = int(cfg.get("n_opponents", 3))
        max_d = float(cfg.get("max_edge_m", 20.0))
        cpp = carrier.get("pitch_position")
        if cpp is None:
            return []
        cx_m, cy_m = float(cpp[0]), float(cpp[1])
        c_team = carrier.get("team")
        carrier_tid = _tid_key(carrier)
        opps: list[tuple[dict, float]] = []
        for t in tracks:
            if _tid_key(t) == carrier_tid:
                continue
            pp = t.get("pitch_position")
            if pp is None:
                continue
            team = t.get("team")
            if team not in _VALID_TEAMS or team == c_team:
                continue
            d = math.hypot(float(pp[0]) - cx_m, float(pp[1]) - cy_m)
            if d <= max_d:
                opps.append((t, d))
        opps.sort(key=lambda x: x[1])
        return opps[:n]

    def _draw_distance_edge(
        self,
        frame: np.ndarray,
        carrier: dict,
        opp: dict,
        dist_m: float,
        w: int,
        h: int,
    ) -> None:
        c_bb = carrier["bbox"]
        o_bb = opp["bbox"]
        p1 = (int((c_bb[0] + c_bb[2]) / 2), int(c_bb[3]))
        p2 = (int((o_bb[0] + o_bb[2]) / 2), int(o_bb[3]))
        color = tuple(int(c) for c in self.cfg.get(
            "distance_network", {}).get("line_color", [200, 200, 255]))
        cv2.line(frame, p1, p2, color, 2, cv2.LINE_AA)
        cv2.circle(frame, p1, 4, color, -1, cv2.LINE_AA)
        cv2.circle(frame, p2, 4, color, -1, cv2.LINE_AA)

        # Label at midpoint, nudged perpendicular toward the far frame edge
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        # Choose the side that sits farther from the frame centre
        test_a = (mx + nx * 20, my + ny * 20)
        test_b = (mx - nx * 20, my - ny * 20)
        cxw, cyh = w / 2.0, h / 2.0
        if (test_a[0] - cxw) ** 2 + (test_a[1] - cyh) ** 2 >= \
                (test_b[0] - cxw) ** 2 + (test_b[1] - cyh) ** 2:
            nudge_x, nudge_y = nx, ny
        else:
            nudge_x, nudge_y = -nx, -ny
        lx = int(mx + nudge_x * 12)
        ly = int(my + nudge_y * 12)

        text = f"Distance: {dist_m:.1f}m"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(self.cfg.get("labels", {}).get("font_scale", 0.5))
        thickness = max(1, int(scale * 2))
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x0 = lx - tw // 2 - 4
        y0 = ly - th // 2 - 4
        x1b = lx + tw // 2 + 4
        y1b = ly + th // 2 + baseline + 2
        x0 = max(0, x0); y0 = max(0, y0)
        x1b = min(w - 1, x1b); y1b = min(h - 1, y1b)
        bg_alpha = float(self.cfg.get("labels", {}).get("bg_alpha", 0.55))
        if x1b > x0 and y1b > y0:
            roi = frame[y0:y1b, x0:x1b]
            dark = np.zeros_like(roi)
            cv2.addWeighted(dark, bg_alpha, roi, 1 - bg_alpha, 0, dst=roi)
        cv2.putText(frame, text, (lx - tw // 2, ly + th // 2),
                    font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, text, (lx - tw // 2, ly + th // 2),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_labels(self, frame: np.ndarray, tracks: list[dict]) -> None:
        """Draw centered jersey/ID text above each bbox.

        No background rectangle — relies on a black stroke around the
        white text for legibility on any background.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = float(self.cfg.get("labels", {}).get("font_scale", 0.5))
        thickness = max(1, int(scale * 2))
        h, w = frame.shape[:2]
        for t in tracks:
            bbox = t.get("bbox")
            if bbox is None:
                continue
            x1, y1, x2, _y2 = bbox
            tid = _tid_key(t)
            jersey = t.get("jersey_number")
            team = self.team_map.get(tid, t.get("team", "unknown"))
            # Prepend a team prefix (A / B) for outfield players and
            # goalkeepers so adjacent same-number labels (e.g. A-9 vs
            # B-9) are visually distinct.  Officials get their cluster
            # id as-is (REF-01 / LIN-01).
            team_prefix = ""
            if team == "team_A":
                team_prefix = "A "
            elif team == "team_B":
                team_prefix = "B "
            if jersey:
                text = f"{team_prefix}#{jersey}"
            elif "-" in tid:
                text = tid
            else:
                text = f"#{tid}"
            (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
            cx_px = int((x1 + x2) / 2)
            tx = cx_px - tw // 2
            ty = int(y1) - 6
            if ty - th < 0:
                ty = int(y1) + th + 4
            tx = max(2, min(tx, w - tw - 2))
            ty = max(th + 2, min(ty, h - 2))
            cv2.putText(frame, text, (tx, ty),
                        font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(frame, text, (tx, ty),
                        font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Minimap
    # ------------------------------------------------------------------

    def _composite_minimap(
        self,
        frame: np.ndarray,
        tracks: list[dict],
        ball: dict | None,
        frame_id: int,
        w: int,
        h: int,
    ) -> None:
        mm_cfg = self.cfg.get("minimap", {})
        mm_h = max(80, int(h * float(mm_cfg.get("height_frac", 0.28))))
        # Match the video's aspect ratio so the minimap is a small
        # "picture-in-picture" version of the frame, not a portrait box.
        mm_w = max(80, int(mm_h * (w / max(1, h))))
        margin = int(mm_cfg.get("margin_px", 20))
        alpha = float(mm_cfg.get("alpha", 0.85))

        # Recent ball pitch trail (last 30 frames)
        trail: list[list[float]] = []
        for f in range(max(0, frame_id - 30), frame_id + 1):
            p = self._ball_pitch_by_frame.get(f)
            if p is not None:
                trail.append(p)

        mm = draw_topdown_pitch(
            height=mm_h,
            width=mm_w,
            tracks=tracks,
            team_assignments=self.team_map,
            ball_track=ball,
            ball_trajectory_world=trail or None,
            pitch_length=self.ctx.pitch_length,
            pitch_width=self.ctx.pitch_width,
        )
        mm_h_actual, mm_w_actual = mm.shape[:2]
        if mm_h_actual >= h or mm_w_actual >= w:
            return

        # Bottom-right placement
        y0 = h - mm_h_actual - margin
        x0 = w - mm_w_actual - margin
        y1 = y0 + mm_h_actual
        x1 = x0 + mm_w_actual
        if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
            return
        roi = frame[y0:y1, x0:x1]
        cv2.addWeighted(mm, alpha, roi, 1.0 - alpha, 0, dst=roi)
        cv2.rectangle(frame, (x0 - 1, y0 - 1), (x1, y1),
                      (255, 255, 255), 2, cv2.LINE_AA)
