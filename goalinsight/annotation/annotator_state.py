"""Stateful annotator core for manual pitch keypoint/line annotation.

Workflow:
    1. Pick a frame via the web UI (see goalinsight.annotation.web).
    2. Mark keypoints (point mode) and/or lines (line mode). Line-line
       intersections auto-populate as derived points.
    3. Compute Homography: least-squares on manual + derived points. All
       remaining HRNet keypoints are auto-projected (ground via H, crossbars
       via PnP) and saved alongside the manual ones.
    4. Save: writes frame_<idx>.json, frame_<idx>_raw.jpg,
       frame_<idx>_all_points.json, frame_<idx>.jpg, frame_<idx>.npy under
       annotations_dir/<video_name>/.

The *_all_points.json format is the one consumed by
goalinsight.field_registration.pnlcalib.finetune.point_dataloader — world
coords are in PnLCalib y-up convention.

Render and geometry concerns live in sibling modules
(``annotator_render``, ``annotator_geometry``); this file owns the state and
the public method surface that the FastAPI ``web.py`` endpoints call.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from . import pitch_constants
from .annotation_io import (
    load_frame_annotation,
    save_frame_annotation,
)
from .annotator_geometry import (
    compute_auto_projections as _compute_auto_projections,
    compute_homography as _compute_homography,
    compute_line_intersections as _compute_line_intersections,
)
from .annotator_render import (
    draw_pitch_projection,
    encode_jpeg,
    render_tactical_view,
    visualize_annotations,
)
from .homography import MIN_POINTS_FOR_PNLCALIB
from .index import AnnotationIndex
from .pitch import keypoints as _pk
from .pitch.geometry import SoccerPitch
from .pitch.keypoints import PITCH_POINTS_TO_INTERSECTON
from .pitch_diagram import create_lines_diagram, create_pitch_diagram


class AnchorAnnotator:
    """Stateful core for manual pitch keypoint/line annotation."""

    def __init__(
        self,
        annotations_dir: str = "output/annotations",
        pitch: SoccerPitch | None = None,
    ):
        if pitch is not None:
            pitch_constants.set_active_pitch(pitch)
        self.annotations_dir = Path(annotations_dir)
        self.index = AnnotationIndex(annotations_dir)
        self.video_name: str = ""
        self.annotated_frames: list[int] = []

        self.clicked_points: list[tuple[float, float]] = []
        self.world_points: list[tuple[float, float]] = []
        self.keypoint_names: list[str] = []

        self.line_clicks: list[tuple[float, float]] = []
        self.annotated_lines: list[dict] = []

        self.derived_points: list[tuple[tuple[float, float], tuple[float, float], str]] = []
        self.derived_accepted: list[bool] = []
        self.auto_projected_points: list[
            tuple[tuple[float, float], tuple[float, float], str, int, bool]
        ] = []
        self.auto_accepted: list[bool] = []

        self.annotation_mode = "point"
        self.current_frame_idx = 0
        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.current_frame: np.ndarray | None = None
        self.H0: np.ndarray | None = None
        self.reprojection_error = 0.0
        self.show_projection = True
        self.selected_manual_idx: int | None = None
        self.selected_line_idx: int | None = None

    # ------------------------------------------------------------------
    # State lifecycle
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self.clicked_points = []
        self.world_points = []
        self.keypoint_names = []
        self.line_clicks = []
        self.annotated_lines = []
        self.derived_points = []
        self.derived_accepted = []
        self.auto_projected_points = []
        self.auto_accepted = []
        self.H0 = None
        self.reprojection_error = 0.0
        self.selected_manual_idx = None
        self.selected_line_idx = None

    def _load_video(self, video_path: str) -> None:
        if self.cap is not None:
            self.cap.release()
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _check_pitch_consistency(self, video_name: str) -> str | None:
        """Return error message if saved annotations conflict with active pitch.

        Reads the most recent saved frame's manual world coords and compares
        against the active pitch's resolved coords for the same keypoint name.
        Returns None when consistent (or when there's nothing to compare).
        """
        frames = self.index.get_annotated_frames(video_name)
        if not frames:
            return None
        frame_dir = self.index.get_video_dir(video_name)
        json_path = frame_dir / f"frame_{frames[-1]}.json"
        if not json_path.exists():
            return None
        try:
            with open(json_path) as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, IOError):
            return None
        for pt in data.get("points", []):
            name = pt.get("keypoint_name") or pt.get("name")
            saved = pt.get("world")
            if not name or saved is None:
                continue
            active = _pk.PITCH_POINTS.get(name)
            if active is None:
                continue
            if (abs(float(active[0]) - float(saved[0])) > 1e-3
                    or abs(float(active[1]) - float(saved[1])) > 1e-3):
                return (
                    f"Pitch mismatch: '{video_name}' was annotated under "
                    f"different dimensions (e.g. {name} saved at "
                    f"({saved[0]:.2f}, {saved[1]:.2f}) vs current "
                    f"({float(active[0]):.2f}, {float(active[1]):.2f})). "
                    f"Restart with --config matching that pitch."
                )
        return None

    def _get_frame(self, frame_idx: int) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("Video not loaded")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {frame_idx}")

        self.current_frame = frame
        self.current_frame_idx = frame_idx
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _get_frame_dir(self) -> Path:
        return self.index.get_video_dir(self.video_name)

    def _load_frame_annotation(self, frame_idx: int) -> bool:
        data = load_frame_annotation(self._get_frame_dir(), frame_idx)
        if data is None:
            return False

        self._reset_state()
        self.clicked_points = data["clicked_points"]
        self.keypoint_names = data["keypoint_names"]
        self.reprojection_error = data["reprojection_error"]
        self.H0 = data["H0"]

        # Re-resolve world coords against the *active* pitch — saved JSON may
        # have been written under a different pitch (e.g. FIFA defaults), and
        # mixing those with current PITCH_POINTS would give a chimera H.
        pitch_changed = False
        self.world_points = []
        for name, saved in zip(self.keypoint_names, data["world_points"]):
            pt = _pk.PITCH_POINTS.get(name)
            resolved = ((0.0, 0.0) if pt is None
                        else (float(pt[0]), float(pt[1])))
            self.world_points.append(resolved)
            if (abs(resolved[0] - float(saved[0])) > 1e-3
                    or abs(resolved[1] - float(saved[1])) > 1e-3):
                pitch_changed = True

        self.annotated_lines = []
        for ln in data["annotated_lines"]:
            world = pitch_constants.PITCH_LINES.get(ln["name"], ln["world"])
            saved_world = ln["world"]
            if (abs(world[0][0] - saved_world[0][0]) > 1e-3
                    or abs(world[0][1] - saved_world[0][1]) > 1e-3
                    or abs(world[1][0] - saved_world[1][0]) > 1e-3
                    or abs(world[1][1] - saved_world[1][1]) > 1e-3):
                pitch_changed = True
            self.annotated_lines.append({
                "pixels": ln["pixels"],
                "name": ln["name"],
                "world": world,
            })
        # Derived points are line intersections — recompute against refreshed
        # annotated_lines so their world coords reflect the active pitch.
        _compute_line_intersections(self)

        if pitch_changed:
            # Stale H0 was computed under a different pitch — recompute
            # against the refreshed world coords so the overlay reappears
            # without a manual "Compute homography" click.
            self.H0 = None
            self.reprojection_error = 0.0
            if (len(self.keypoint_names) + len(self.derived_points)) >= MIN_POINTS_FOR_PNLCALIB:
                self.compute_homography()
        elif self.H0 is not None:
            _compute_auto_projections(self)

        return True

    def _save_frame_annotation(self, frame_idx: int) -> bool:
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        vis = visualize_annotations(self, frame_rgb)
        if self.H0 is not None:
            vis = draw_pitch_projection(vis, self.H0)

        accepted_derived = [
            pt for pt, ok in zip(self.derived_points, self.derived_accepted) if ok
        ]
        accepted_auto = [
            pt for pt, ok in zip(self.auto_projected_points, self.auto_accepted) if ok
        ]

        success = save_frame_annotation(
            frame_dir=self._get_frame_dir(),
            frame_idx=frame_idx,
            video_path=self.video_path,
            video_name=self.video_name,
            clicked_points=self.clicked_points,
            world_points=self.world_points,
            keypoint_names=self.keypoint_names,
            annotated_lines=self.annotated_lines,
            derived_points=accepted_derived,
            reprojection_error=self.reprojection_error,
            H0=self.H0,
            current_frame=self.current_frame,
            vis_frame=vis,
            auto_projected_points=accepted_auto,
        )

        if success:
            self.index.add_frame(self.video_name, frame_idx)
            if frame_idx not in self.annotated_frames:
                self.annotated_frames.append(frame_idx)
                self.annotated_frames.sort()

        return success

    # ------------------------------------------------------------------
    # Public API used by the web frontend
    # ------------------------------------------------------------------

    def open_video(self, video_path: str, start_frame: int = 0) -> int:
        """Load a video, auto-restore last annotation, return active frame idx."""
        video_name = Path(video_path).stem
        mismatch = self._check_pitch_consistency(video_name)
        if mismatch:
            raise RuntimeError(mismatch)

        self._load_video(video_path)
        self.video_name = video_name
        self.annotated_frames = self.index.get_annotated_frames(self.video_name)

        if start_frame == 0 and self.annotated_frames:
            start_frame = self.annotated_frames[-1]

        self._reset_state()
        # Load the frame BEFORE the annotation so _compute_line_intersections
        # (called by _load_frame_annotation when re-resolving lines under the
        # active pitch) can clip derived points against the frame bounds.
        self._get_frame(start_frame)
        self._load_frame_annotation(start_frame)
        if self.H0 is not None:
            _compute_auto_projections(self)
        return start_frame

    def switch_video(self, video_path: str, frame_idx: int | None = None) -> int:
        """Switch the active video, refusing on pitch mismatch.

        Raises RuntimeError on pitch mismatch. Returns the active frame index.
        """
        return self.open_video(video_path, start_frame=frame_idx or 0)

    def goto_frame(self, frame_idx: int) -> bool:
        """Switch to ``frame_idx``, loading existing annotation if present."""
        frame_idx = int(frame_idx)
        if frame_idx < 0 or frame_idx >= self.total_frames:
            return False
        self._reset_state()
        self._get_frame(frame_idx)
        if self._load_frame_annotation(frame_idx):
            if self.H0 is not None:
                _compute_auto_projections(self)
        return True

    def set_mode(self, mode: str) -> None:
        if mode not in ("point", "line"):
            raise ValueError(f"Unknown mode: {mode}")
        self.annotation_mode = mode
        self.line_clicks = []

    def click(self, px: float, py: float) -> str:
        """Record a click in the current annotation mode."""
        if self.annotation_mode == "point":
            self.clicked_points.append((float(px), float(py)))
            return f"Clicked at ({px:.0f}, {py:.0f}). Pick a keypoint and Add."
        self.line_clicks.append((float(px), float(py)))
        if len(self.line_clicks) == 1:
            return f"Line start at ({px:.0f}, {py:.0f}). Click second point."
        if len(self.line_clicks) == 2:
            return f"Line end at ({px:.0f}, {py:.0f}). Pick a line and Add."
        self.line_clicks = [(float(px), float(py))]
        return f"Reset. Line start at ({px:.0f}, {py:.0f})."

    def add_keypoint(self, keypoint_name: str) -> str:
        if not self.clicked_points:
            return "No pixel clicked yet."
        if keypoint_name not in _pk.PITCH_POINTS:
            return f"Invalid keypoint: {keypoint_name}"
        if keypoint_name in self.keypoint_names:
            return f"{keypoint_name} already annotated."
        if len(self.clicked_points) <= len(self.keypoint_names):
            return "No pending click to attach."
        pt = _pk.PITCH_POINTS[keypoint_name]
        world_coords = (float(pt[0]), float(pt[1]))
        self.world_points.append(world_coords)
        self.keypoint_names.append(keypoint_name)
        hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(keypoint_name, -1)
        return f"Added [{hrnet_idx}] {keypoint_name}"

    def add_line(self, line_name: str) -> str:
        if len(self.line_clicks) < 2:
            return f"Need 2 clicks. Current: {len(self.line_clicks)}"
        if line_name not in pitch_constants.PITCH_LINES:
            return f"Invalid line: {line_name}"
        for existing in self.annotated_lines:
            if existing["name"] == line_name:
                return f"{line_name} already annotated."
        world_coords = pitch_constants.PITCH_LINES[line_name]
        self.annotated_lines.append({
            "pixels": [self.line_clicks[0], self.line_clicks[1]],
            "name": line_name,
            "world": world_coords,
        })
        self.line_clicks = []
        _compute_line_intersections(self)
        return f"Added line: {line_name}"

    def undo(self) -> str:
        if self.line_clicks:
            self.line_clicks.pop()
            return f"Pending line click removed ({len(self.line_clicks)} remaining)"
        if len(self.clicked_points) > len(self.world_points):
            self.clicked_points.pop()
            return "Pending point click removed"
        if self.annotated_lines:
            removed = self.annotated_lines.pop()
            _compute_line_intersections(self)
            return f"Line '{removed['name']}' removed"
        if self.clicked_points:
            removed_name = self.keypoint_names[-1] if self.keypoint_names else "unknown"
            self.clicked_points.pop()
            if self.world_points:
                self.world_points.pop()
            if self.keypoint_names:
                self.keypoint_names.pop()
            return f"Point '{removed_name}' removed"
        return "Nothing to undo"

    def reset(self) -> None:
        self._reset_state()

    def compute_homography(self) -> str:
        return _compute_homography(self)

    # ------------------------------------------------------------------
    # Acceptance toggles
    # ------------------------------------------------------------------

    def toggle_derived(self, idx: int) -> str:
        if not (0 <= idx < len(self.derived_accepted)):
            return f"Invalid derived idx: {idx}"
        self.derived_accepted[idx] = not self.derived_accepted[idx]
        state = "accepted" if self.derived_accepted[idx] else "pending"
        return f"Derived D{idx + 1}: {state}"

    def accept_all_derived(self) -> str:
        self.derived_accepted = [True] * len(self.derived_points)
        return f"Accepted {len(self.derived_accepted)} derived points"

    def reject_all_derived(self) -> str:
        n = len(self.derived_points)
        self.derived_accepted = [False] * n
        return f"Rejected {n} derived points"

    def toggle_auto(self, idx: int) -> str:
        if not (0 <= idx < len(self.auto_accepted)):
            return f"Invalid auto idx: {idx}"
        self.auto_accepted[idx] = not self.auto_accepted[idx]
        state = "accepted" if self.auto_accepted[idx] else "pending"
        return f"Auto idx {idx}: {state}"

    def accept_all_auto(self) -> str:
        self.auto_accepted = [True] * len(self.auto_projected_points)
        return f"Accepted {len(self.auto_accepted)} auto-projected points"

    def reject_all_auto(self) -> str:
        n = len(self.auto_projected_points)
        self.auto_accepted = [False] * n
        return f"Rejected {n} auto-projected points"

    # ------------------------------------------------------------------
    # Selection + deletion of manual points / lines
    # ------------------------------------------------------------------

    def select_manual_point(self, idx: int | None) -> str:
        if idx is not None and not (0 <= idx < len(self.clicked_points)):
            return f"Invalid manual idx: {idx}"
        self.selected_manual_idx = idx
        self.selected_line_idx = None
        return "" if idx is None else f"Selected manual point P{idx + 1}"

    def select_line(self, idx: int | None) -> str:
        if idx is not None and not (0 <= idx < len(self.annotated_lines)):
            return f"Invalid line idx: {idx}"
        self.selected_line_idx = idx
        self.selected_manual_idx = None
        return "" if idx is None else f"Selected line L{idx + 1}"

    def delete_manual_point(self, idx: int) -> str:
        if not (0 <= idx < len(self.clicked_points)):
            return f"Invalid manual idx: {idx}"
        self.clicked_points.pop(idx)
        if idx < len(self.world_points):
            self.world_points.pop(idx)
        if idx < len(self.keypoint_names):
            name = self.keypoint_names.pop(idx)
        else:
            name = "(pending)"
        if self.selected_manual_idx == idx:
            self.selected_manual_idx = None
        elif self.selected_manual_idx is not None and self.selected_manual_idx > idx:
            self.selected_manual_idx -= 1
        return f"Deleted manual point P{idx + 1} ({name})"

    def delete_line(self, idx: int) -> str:
        if not (0 <= idx < len(self.annotated_lines)):
            return f"Invalid line idx: {idx}"
        removed = self.annotated_lines.pop(idx)
        _compute_line_intersections(self)
        if self.selected_line_idx == idx:
            self.selected_line_idx = None
        elif self.selected_line_idx is not None and self.selected_line_idx > idx:
            self.selected_line_idx -= 1
        return f"Deleted line L{idx + 1} ({removed['name']})"

    def set_show_projection(self, show: bool) -> str:
        self.show_projection = bool(show)
        return f"Projection {'shown' if self.show_projection else 'hidden'}"

    def save(self) -> str:
        accepted_derived = sum(self.derived_accepted)
        accepted_auto = sum(self.auto_accepted)
        total_pts = len(self.keypoint_names) + accepted_derived + accepted_auto
        if total_pts == 0:
            return "Error: nothing accepted to save."
        if not self._save_frame_annotation(self.current_frame_idx):
            return f"Error: failed to save frame {self.current_frame_idx}"
        save_dir = self._get_frame_dir()
        breakdown = (
            f"{len(self.keypoint_names)} manual + {accepted_derived} derived "
            f"+ {accepted_auto} auto"
        )
        if self.H0 is None:
            return (
                f"Saved frame {self.current_frame_idx} ({breakdown}) to {save_dir}/"
            )
        return (
            f"Saved frame {self.current_frame_idx} ({breakdown}, "
            f"err={self.reprojection_error:.2f}m) to {save_dir}/"
        )

    # ------------------------------------------------------------------
    # JPEG renderers (thin delegators to annotator_render)
    # ------------------------------------------------------------------

    def render_frame_jpeg(self, with_overlay: bool = True) -> bytes:
        if self.current_frame is None:
            raise RuntimeError("No frame loaded")
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        if with_overlay:
            vis = visualize_annotations(self, frame_rgb)
            if self.H0 is not None and self.show_projection:
                vis = draw_pitch_projection(vis, self.H0)
        else:
            vis = frame_rgb
        return encode_jpeg(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))

    def render_pitch_diagram_jpeg(
        self,
        highlight_keypoint: str | None = None,
        highlight_line: str | None = None,
    ) -> bytes:
        img = create_pitch_diagram(
            highlight_keypoint=highlight_keypoint,
            highlight_line=highlight_line,
            annotated_keypoints=self.keypoint_names,
            annotated_lines=[ln["name"] for ln in self.annotated_lines],
            pitch_lines=pitch_constants.PITCH_LINES,
        )
        return encode_jpeg(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def render_tactical_jpeg(self) -> bytes:
        return encode_jpeg(render_tactical_view(self))

    def render_lines_diagram_jpeg(self, highlight_line: str | None = None) -> bytes:
        img = create_lines_diagram(
            highlight_line=highlight_line,
            annotated_lines=[ln["name"] for ln in self.annotated_lines],
            pitch_lines=pitch_constants.PITCH_LINES,
        )
        return encode_jpeg(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def state_dict(self) -> dict:
        """Serializable annotator state for the frontend."""
        manual = []
        for i, (px, py) in enumerate(self.clicked_points):
            if i < len(self.keypoint_names):
                wx, wy = self.world_points[i]
                kp_name = self.keypoint_names[i]
                hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
                manual.append({
                    "pixel": [float(px), float(py)],
                    "world": [float(wx), float(wy)],
                    "name": kp_name,
                    "hrnet_index": hrnet_idx,
                })
            else:
                manual.append({"pixel": [float(px), float(py)], "pending": True})

        lines = []
        for ln in self.annotated_lines:
            lines.append({
                "name": ln["name"],
                "pixels": [list(map(float, ln["pixels"][0])),
                           list(map(float, ln["pixels"][1]))],
            })

        derived = [
            {
                "pixel": [float(pt[0][0]), float(pt[0][1])],
                "world": [float(pt[1][0]), float(pt[1][1])],
                "name": pt[2],
                "accepted": bool(self.derived_accepted[i])
                if i < len(self.derived_accepted) else False,
            }
            for i, pt in enumerate(self.derived_points)
        ]

        auto_projected = [
            {
                "pixel": [float(pt[0][0]), float(pt[0][1])],
                "world": [float(pt[1][0]), float(pt[1][1])],
                "name": pt[2],
                "hrnet_index": int(pt[3]),
                "is_ground": bool(pt[4]),
                "accepted": bool(self.auto_accepted[i])
                if i < len(self.auto_accepted) else False,
            }
            for i, pt in enumerate(self.auto_projected_points)
        ]

        return {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "frame_idx": self.current_frame_idx,
            "total_frames": self.total_frames,
            "mode": self.annotation_mode,
            "annotated_frames": list(self.annotated_frames),
            "manual_points": manual,
            "lines": lines,
            "line_pending": [list(map(float, p)) for p in self.line_clicks],
            "derived_points": derived,
            "auto_projected_points": auto_projected,
            "homography_computed": self.H0 is not None,
            "reprojection_error": float(self.reprojection_error),
            "show_projection": bool(self.show_projection),
            "selected_manual_idx": self.selected_manual_idx,
            "selected_line_idx": self.selected_line_idx,
            "frame_size": (
                [int(self.current_frame.shape[1]), int(self.current_frame.shape[0])]
                if self.current_frame is not None
                else None
            ),
        }
