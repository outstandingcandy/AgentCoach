"""Gradio UI for anchor-frame annotation.

Workflow:
    1. Pick a frame via slider or dropdown of previously-annotated frames.
    2. Mark keypoints (point mode) and/or lines (line mode). Line-line
       intersections auto-populate as derived points.
    3. Compute Homography: least-squares on manual + derived points. All
       remaining HRNet keypoints are auto-projected (ground via H, crossbars
       via PnP) and saved alongside the manual ones.
    4. Save & Exit: writes frame_<idx>.json, frame_<idx>_raw.jpg,
       frame_<idx>_all_points.json, frame_<idx>.jpg, frame_<idx>.npy under
       annotations_dir/<video_name>/.

The *_all_points.json format is the one consumed by
goalinsight.field_registration.pnlcalib.finetune.point_dataloader — world
coords are in PnLCalib y-up convention.
"""

from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .annotation_io import (
    load_frame_annotation,
    save_frame_annotation,
)
from .homography import (
    compute_homography_ls,
    line_intersection,
    project_3d_with_pnp,
    project_world_to_image,
)
from .index import AnnotationIndex
from .keypoint_utils import (
    PITCH_KEYPOINTS,
    abbreviate_line_name,
    get_hrnet_keypoint_choices,
    parse_keypoint_choice,
)
from .pitch.keypoints import (
    INTERSECTON_TO_PITCH_POINTS,
    NOT_ON_PLANE,
    PITCH_POINTS,
    PITCH_POINTS_TO_INTERSECTON,
)
from .pitch_constants import (
    PITCH_LENGTH,
    PITCH_LINES,
    PITCH_WIDTH,
    get_all_line_names,
)
from .pitch_diagram import create_pitch_diagram
from .viz import render_pitch_projection


class AnchorAnnotator:
    """Gradio-based UI for manual pitch keypoint and line annotation."""

    def __init__(self, annotations_dir: str = "output/annotations"):
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
        self.auto_projected_points: list[
            tuple[tuple[float, float], tuple[float, float], str, int, bool]
        ] = []

        self.annotation_mode = "point"
        self.current_frame_idx = 0
        self.video_path: str | None = None
        self.cap: cv2.VideoCapture | None = None
        self.total_frames = 0
        self.current_frame: np.ndarray | None = None
        self.H0: np.ndarray | None = None
        self.reprojection_error = 0.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_pitch_diagram(
        self,
        highlight_keypoint: str | None = None,
        highlight_line: str | None = None,
        annotated_keypoints: list[str] | None = None,
        annotated_lines: list[str] | None = None,
    ) -> np.ndarray:
        return create_pitch_diagram(
            highlight_keypoint=highlight_keypoint,
            highlight_line=highlight_line,
            annotated_keypoints=annotated_keypoints or [],
            annotated_lines=annotated_lines or [],
            pitch_lines=PITCH_LINES,
        )

    def _draw_pitch_projection(self, frame_rgb: np.ndarray, H: np.ndarray) -> np.ndarray:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        result = render_pitch_projection(frame_bgr, H, color=(0, 255, 255), thickness=2)
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

    def _render_tactical_view(self) -> np.ndarray:
        """Render a tactical-style view of all annotations (y-up world)."""
        scale = 8
        margin = 50
        width = int(PITCH_LENGTH * scale + 2 * margin)
        height = int(PITCH_WIDTH * scale + 2 * margin)

        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (34, 139, 34)

        L, W = PITCH_LENGTH / 2, PITCH_WIDTH / 2

        def world_to_px(x: float, y: float) -> tuple[int, int]:
            px = int((x + L) * scale + margin)
            py = int((W - y) * scale + margin)
            return (px, py)

        white = (255, 255, 255)

        pts = [
            world_to_px(-L, W), world_to_px(L, W),
            world_to_px(L, -W), world_to_px(-L, -W),
        ]
        for i in range(4):
            cv2.line(img, pts[i], pts[(i + 1) % 4], white, 2)

        cv2.line(img, world_to_px(0, W), world_to_px(0, -W), white, 2)
        cv2.circle(img, world_to_px(0, 0), int(9.15 * scale), white, 2)
        cv2.circle(img, world_to_px(0, 0), 4, white, -1)

        pa_w, pa_d = 40.32 / 2, 16.5
        cv2.rectangle(img, world_to_px(-L, pa_w), world_to_px(-L + pa_d, -pa_w), white, 2)
        cv2.rectangle(img, world_to_px(L - pa_d, pa_w), world_to_px(L, -pa_w), white, 2)

        ga_w, ga_d = 18.32 / 2, 5.5
        cv2.rectangle(img, world_to_px(-L, ga_w), world_to_px(-L + ga_d, -ga_w), white, 2)
        cv2.rectangle(img, world_to_px(L - ga_d, ga_w), world_to_px(L, -ga_w), white, 2)

        cv2.circle(img, world_to_px(-L + 11, 0), 4, white, -1)
        cv2.circle(img, world_to_px(L - 11, 0), 4, white, -1)

        cv2.ellipse(img, world_to_px(-L + 11, 0), (int(9.15 * scale), int(9.15 * scale)), 0, -60, 60, white, 2)
        cv2.ellipse(img, world_to_px(L - 11, 0), (int(9.15 * scale), int(9.15 * scale)), 0, 120, 240, white, 2)

        for i, (px, py) in enumerate(self.clicked_points):
            if i < len(self.world_points):
                wx, wy = self.world_points[i]
                tx, ty = world_to_px(wx, wy)
                cv2.circle(img, (tx, ty), 10, (0, 0, 255), -1)
                cv2.circle(img, (tx, ty), 10, (255, 255, 255), 2)
                if i < len(self.keypoint_names):
                    kp_name = self.keypoint_names[i]
                    hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
                    label = f"{hrnet_idx}"
                else:
                    label = f"P{i+1}"
                cv2.putText(img, label, (tx + 12, ty + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for i, (_pixel, world, _name) in enumerate(self.derived_points):
            wx, wy = world
            tx, ty = world_to_px(wx, wy)
            cv2.circle(img, (tx, ty), 8, (255, 0, 255), -1)
            cv2.circle(img, (tx, ty), 8, (255, 255, 255), 2)
            cv2.putText(img, f"D{i+1}", (tx + 10, ty + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        for _pixel, world, _name, _hrnet_idx, is_ground in self.auto_projected_points:
            wx, wy = world
            tx, ty = world_to_px(wx, wy)
            color = (0, 128, 255) if is_ground else (180, 0, 255)
            cv2.circle(img, (tx, ty), 5, color, -1)
            cv2.circle(img, (tx, ty), 5, (255, 255, 255), 1)

        total = len(self.keypoint_names) + len(self.derived_points)
        status = f"Points: {total} (manual: {len(self.keypoint_names)}, derived: {len(self.derived_points)})"
        cv2.putText(img, status, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if self.H0 is not None:
            cv2.putText(img, f"H0 computed (error: {self.reprojection_error:.2f}m)",
                        (10, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return img

    def _get_annotated_line_names(self) -> list[str]:
        return [line_data["name"] for line_data in self.annotated_lines]

    def _compute_line_intersections(self) -> None:
        self.derived_points = []
        if len(self.annotated_lines) < 2:
            return

        for i, line1 in enumerate(self.annotated_lines):
            for j, line2 in enumerate(self.annotated_lines):
                if j <= i:
                    continue

                pixel_int = line_intersection(
                    (line1["pixels"][0], line1["pixels"][1]),
                    (line2["pixels"][0], line2["pixels"][1]),
                )
                world_int = line_intersection(line1["world"], line2["world"])

                if pixel_int and world_int and self.current_frame is not None:
                    px, py = pixel_int
                    h, w = self.current_frame.shape[:2]
                    if 0 <= px < w and 0 <= py < h:
                        name = f"{line1['name']} + {line2['name']}"
                        self.derived_points.append((pixel_int, world_int, name))

    def _reset_state(self) -> None:
        self.clicked_points = []
        self.world_points = []
        self.keypoint_names = []
        self.line_clicks = []
        self.annotated_lines = []
        self.derived_points = []
        self.auto_projected_points = []
        self.H0 = None
        self.reprojection_error = 0.0

    def _load_video(self, video_path: str) -> None:
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

    def _visualize_annotations(self, frame_rgb: np.ndarray) -> np.ndarray:
        vis = frame_rgb.copy()

        for i, line_data in enumerate(self.annotated_lines):
            p1, p2 = line_data["pixels"]
            cv2.line(vis, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 255, 255), 2)
            cv2.circle(vis, (int(p1[0]), int(p1[1])), 5, (0, 255, 255), -1)
            cv2.circle(vis, (int(p2[0]), int(p2[1])), 5, (0, 255, 255), -1)
            mid_x = int((p1[0] + p2[0]) / 2)
            mid_y = int((p1[1] + p2[1]) / 2)
            short_name = abbreviate_line_name(line_data["name"])
            cv2.putText(vis, f"L{i+1}:{short_name}", (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        for i, (px, py) in enumerate(self.line_clicks):
            cv2.circle(vis, (int(px), int(py)), 6, (255, 165, 0), -1)
            cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)
            cv2.putText(vis, f"L-{i+1}", (int(px) + 10, int(py)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        for i, (pixel, _world, _name) in enumerate(self.derived_points):
            px, py = pixel
            cv2.circle(vis, (int(px), int(py)), 8, (255, 0, 255), -1)
            cv2.circle(vis, (int(px), int(py)), 10, (255, 255, 255), 2)
            cv2.putText(vis, f"D{i+1}", (int(px) + 12, int(py) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        for pixel, _world, _name, hrnet_idx, is_ground in self.auto_projected_points:
            px, py = pixel
            color = (0, 128, 255) if is_ground else (180, 0, 255)
            cv2.circle(vis, (int(px), int(py)), 6, color, -1)
            cv2.circle(vis, (int(px), int(py)), 8, (255, 255, 255), 2)
            label = f"[{hrnet_idx}]" if is_ground else f"[{hrnet_idx}]T"
            cv2.putText(vis, label, (int(px) + 10, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
            cv2.putText(vis, label, (int(px) + 10, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        for i, (px, py) in enumerate(self.clicked_points):
            if i < len(self.keypoint_names):
                color = (0, 255, 0)
                kp_name = self.keypoint_names[i]
                hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(kp_name, -1)
                label = f"[{hrnet_idx}] {kp_name}"
            else:
                color = (255, 255, 0)
                label = f"P{i+1}: [click Add]"

            cv2.circle(vis, (int(px), int(py)), 8, color, -1)
            cv2.circle(vis, (int(px), int(py)), 10, (255, 255, 255), 2)
            cv2.putText(vis, label, (int(px) + 15, int(py) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(vis, label, (int(px) + 15, int(py) + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        confirmed = len(self.keypoint_names)
        derived = len(self.derived_points)
        total_pts = confirmed + derived
        mode_str = f"[{self.annotation_mode.upper()} mode]"
        info_text = (
            f"Frame {self.current_frame_idx} | "
            f"Points: {confirmed} manual + {derived} derived = {total_pts} | {mode_str}"
        )
        cv2.putText(vis, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return vis

    def _get_annotations_text(self) -> str:
        text_lines: list[str] = []

        if self.clicked_points:
            text_lines.append("=== Manual Points ===")
            for i, (px, py) in enumerate(self.clicked_points):
                if i < len(self.keypoint_names):
                    name = self.keypoint_names[i]
                    wx, wy = self.world_points[i]
                    hrnet_idx = PITCH_POINTS_TO_INTERSECTON.get(name, -1)
                    text_lines.append(
                        f"P{i+1}. ({px:.0f}, {py:.0f}) -> [{hrnet_idx}] {name} ({wx:.1f}, {wy:.1f})"
                    )
                else:
                    text_lines.append(f"P{i+1}. ({px:.0f}, {py:.0f}) -> [pending]")

        if self.annotated_lines:
            text_lines.append("\n=== Lines ===")
            for i, line in enumerate(self.annotated_lines):
                p1, p2 = line["pixels"]
                text_lines.append(
                    f"L{i+1}. {line['name']}: ({p1[0]:.0f},{p1[1]:.0f})->({p2[0]:.0f},{p2[1]:.0f})"
                )

        if self.derived_points:
            text_lines.append("\n=== Derived Points ===")
            for i, (pixel, world, _name) in enumerate(self.derived_points):
                px, py = pixel
                wx, wy = world
                text_lines.append(
                    f"D{i+1}. ({px:.0f}, {py:.0f}) -> ({wx:.1f}, {wy:.1f})"
                )

        if self.auto_projected_points:
            text_lines.append(f"\n=== Auto-Projected ({len(self.auto_projected_points)}) ===")
            for pixel, _world, name, hrnet_idx, is_ground in self.auto_projected_points:
                px, py = pixel
                marker = "" if is_ground else "T"
                text_lines.append(f"[{hrnet_idx}]{marker} ({px:.0f}, {py:.0f}) -> {name}")

        if not text_lines:
            return "No annotations yet"

        total = len(self.keypoint_names) + len(self.derived_points)
        text_lines.append(f"\n--- Total points for H0: {total} (need 4+) ---")
        return "\n".join(text_lines)

    def _get_frame_dir(self) -> Path:
        return self.index.get_video_dir(self.video_name)

    def _compute_auto_projections(self) -> None:
        """Auto-project all un-annotated HRNet keypoints using H0 (+ PnP for crossbars)."""
        self.auto_projected_points = []
        if self.H0 is None or self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]
        annotated_names = set(self.keypoint_names)

        all_pixel_pts = list(self.clicked_points[:len(self.keypoint_names)])
        all_world_pts = list(self.world_points)
        for pixel, world, _ in self.derived_points:
            all_pixel_pts.append(pixel)
            all_world_pts.append(world)

        for idx, name in INTERSECTON_TO_PITCH_POINTS.items():
            if name in annotated_names:
                continue
            pt_3d = PITCH_POINTS[name]

            if idx in NOT_ON_PLANE:
                pixel = project_3d_with_pnp(
                    (float(pt_3d[0]), float(pt_3d[1]), float(pt_3d[2])),
                    all_pixel_pts, all_world_pts, (w, h),
                )
                is_ground = False
            else:
                pixel = project_world_to_image(
                    (float(pt_3d[0]), float(pt_3d[1])), self.H0,
                )
                is_ground = True

            if pixel is None:
                continue
            px, py = pixel
            margin = 50
            if -margin <= px < w + margin and -margin <= py < h + margin:
                self.auto_projected_points.append((
                    (px, py),
                    (float(pt_3d[0]), float(pt_3d[1])),
                    name,
                    idx,
                    is_ground,
                ))

    def _load_frame_annotation(self, frame_idx: int) -> bool:
        data = load_frame_annotation(self._get_frame_dir(), frame_idx)
        if data is None:
            return False

        self._reset_state()
        self.clicked_points = data["clicked_points"]
        self.world_points = data["world_points"]
        self.keypoint_names = data["keypoint_names"]
        self.annotated_lines = data["annotated_lines"]
        self.derived_points = data["derived_points"]
        self.reprojection_error = data["reprojection_error"]
        self.H0 = data["H0"]

        if self.H0 is not None:
            self._compute_auto_projections()

        return True

    def _save_frame_annotation(self, frame_idx: int) -> bool:
        frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        vis = self._visualize_annotations(frame_rgb)
        if self.H0 is not None:
            vis = self._draw_pitch_projection(vis, self.H0)

        success = save_frame_annotation(
            frame_dir=self._get_frame_dir(),
            frame_idx=frame_idx,
            video_path=self.video_path,
            video_name=self.video_name,
            clicked_points=self.clicked_points,
            world_points=self.world_points,
            keypoint_names=self.keypoint_names,
            annotated_lines=self.annotated_lines,
            derived_points=self.derived_points,
            reprojection_error=self.reprojection_error,
            H0=self.H0,
            current_frame=self.current_frame,
            vis_frame=vis,
            auto_projected_points=self.auto_projected_points,
        )

        if success:
            self.index.add_frame(self.video_name, frame_idx)
            if frame_idx not in self.annotated_frames:
                self.annotated_frames.append(frame_idx)
                self.annotated_frames.sort()

        return success

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def launch_ui(
        self,
        video_path: str,
        port: int = 7860,
        start_frame: int = 0,
        share: bool = False,
    ) -> tuple[int, np.ndarray | None]:
        self._load_video(video_path)
        self.video_name = Path(video_path).stem
        self.annotated_frames = self.index.get_annotated_frames(self.video_name)

        if start_frame == 0 and self.annotated_frames:
            last_annotated = self.annotated_frames[-1]
            if self._load_frame_annotation(last_annotated):
                start_frame = last_annotated
                print(f"Auto-loaded annotation for frame {last_annotated}")

        self.current_frame_idx = start_frame
        initial_frame_rgb = self._get_frame(start_frame)

        if self.H0 is not None:
            self._compute_auto_projections()

        if self.clicked_points:
            initial_frame = self._visualize_annotations(initial_frame_rgb)
            if self.H0 is not None:
                initial_frame = self._draw_pitch_projection(initial_frame, self.H0)
        else:
            initial_frame = initial_frame_rgb

        keypoint_choices = get_hrnet_keypoint_choices()
        line_choices = get_all_line_names()
        _, first_kp_name = parse_keypoint_choice(keypoint_choices[0])

        initial_pitch_diagram = self._create_pitch_diagram(
            highlight_keypoint=first_kp_name,
            annotated_keypoints=self.keypoint_names,
            annotated_lines=self._get_annotated_line_names(),
        )
        initial_tactical_view = self._render_tactical_view()

        # -- Event handlers --
        def on_image_click(evt: gr.SelectData):
            if self.current_frame is None:
                return None, "No frame loaded"

            px, py = evt.index[0], evt.index[1]

            if self.annotation_mode == "point":
                self.clicked_points.append((px, py))
                status = f"Clicked at ({px}, {py}). Select keypoint and click 'Add Annotation'."
            else:
                self.line_clicks.append((px, py))
                if len(self.line_clicks) == 1:
                    status = f"Line start at ({px}, {py}). Click second point."
                elif len(self.line_clicks) == 2:
                    status = f"Line end at ({px}, {py}). Select line name and click 'Add Line'."
                else:
                    self.line_clicks = [(px, py)]
                    status = f"Reset. Line start at ({px}, {py})."

            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            vis = self._visualize_annotations(frame_rgb)
            return vis, status

        def on_frame_change(frame_idx: int):
            frame_idx = int(frame_idx)
            if frame_idx == self.current_frame_idx and self.current_frame is not None:
                frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                vis = self._visualize_annotations(frame_rgb)
                if self.H0 is not None:
                    vis = self._draw_pitch_projection(vis, self.H0)
                return (
                    vis,
                    self._get_annotations_text(),
                    self._create_pitch_diagram(
                        annotated_keypoints=self.keypoint_names,
                        annotated_lines=self._get_annotated_line_names(),
                    ),
                    self._render_tactical_view(),
                )

            self._reset_state()
            try:
                frame_rgb = self._get_frame(frame_idx)
                return (
                    frame_rgb,
                    "",
                    self._create_pitch_diagram(annotated_keypoints=[], annotated_lines=[]),
                    self._render_tactical_view(),
                )
            except Exception:
                return np.zeros((480, 640, 3), dtype=np.uint8), "Error loading frame", None, None

        def on_frame_dropdown_change(frame_str: str):
            if not frame_str:
                return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

            frame_idx = int(frame_str)
            try:
                frame_rgb = self._get_frame(frame_idx)
            except Exception:
                return (gr.update(), "Error loading frame", gr.update(), gr.update(), gr.update(), gr.update())

            if self._load_frame_annotation(frame_idx):
                vis = self._visualize_annotations(frame_rgb)
                if self.H0 is not None:
                    vis = self._draw_pitch_projection(vis, self.H0)
                status = f"Loaded annotation for frame {frame_idx}"
            else:
                self._reset_state()
                vis = frame_rgb
                status = f"Frame {frame_idx} (no annotation)"

            return (
                vis,
                status,
                self._get_annotations_text(),
                self._create_pitch_diagram(
                    annotated_keypoints=self.keypoint_names,
                    annotated_lines=self._get_annotated_line_names(),
                ),
                self._render_tactical_view(),
                gr.update(value=frame_idx),
            )

        def on_keypoint_select(keypoint_choice: str):
            _, keypoint_name = parse_keypoint_choice(keypoint_choice)
            return self._create_pitch_diagram(
                highlight_keypoint=keypoint_name,
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )

        def on_line_select(line_name: str):
            return self._create_pitch_diagram(
                highlight_line=line_name,
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )

        def set_mode(mode: str):
            self.annotation_mode = mode
            self.line_clicks = []
            return f"Mode: {mode.upper()} annotation"

        def add_annotation(keypoint_choice: str):
            hrnet_idx, keypoint_name = parse_keypoint_choice(keypoint_choice)
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            current_vis = self._visualize_annotations(frame_rgb)
            tactical_view = self._render_tactical_view()
            pitch_diagram = self._create_pitch_diagram(
                highlight_keypoint=keypoint_name,
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )

            if not self.clicked_points:
                return current_vis, "No point clicked yet.", self._get_annotations_text(), pitch_diagram, tactical_view
            if keypoint_name not in PITCH_KEYPOINTS:
                return current_vis, f"Invalid keypoint: {keypoint_name}", self._get_annotations_text(), pitch_diagram, tactical_view
            if keypoint_name in self.keypoint_names:
                return current_vis, f"Warning: {keypoint_name} already annotated!", self._get_annotations_text(), pitch_diagram, tactical_view

            world_coords = PITCH_KEYPOINTS[keypoint_name]
            self.world_points.append(world_coords)
            self.keypoint_names.append(keypoint_name)

            vis = self._visualize_annotations(frame_rgb)
            pitch_diagram = self._create_pitch_diagram(
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )
            tactical_view = self._render_tactical_view()

            return vis, f"Added: [{hrnet_idx}] {keypoint_name}", self._get_annotations_text(), pitch_diagram, tactical_view

        def add_line_annotation(line_name: str):
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            current_vis = self._visualize_annotations(frame_rgb)
            tactical_view = self._render_tactical_view()
            pitch_diagram = self._create_pitch_diagram(
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )

            if len(self.line_clicks) < 2:
                return current_vis, f"Need 2 clicks. Current: {len(self.line_clicks)}", self._get_annotations_text(), pitch_diagram, tactical_view
            if line_name not in PITCH_LINES:
                return current_vis, f"Invalid line: {line_name}", self._get_annotations_text(), pitch_diagram, tactical_view
            for existing in self.annotated_lines:
                if existing["name"] == line_name:
                    return current_vis, f"Warning: {line_name} already annotated!", self._get_annotations_text(), pitch_diagram, tactical_view

            world_coords = PITCH_LINES[line_name]
            line_data = {
                "pixels": [self.line_clicks[0], self.line_clicks[1]],
                "name": line_name,
                "world": world_coords,
            }
            self.annotated_lines.append(line_data)
            self.line_clicks = []
            self._compute_line_intersections()

            vis = self._visualize_annotations(frame_rgb)
            pitch_diagram = self._create_pitch_diagram(
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )
            tactical_view = self._render_tactical_view()

            return vis, f"Added line: {line_name}", self._get_annotations_text(), pitch_diagram, tactical_view

        def undo_last():
            message = "Nothing to undo"

            if self.line_clicks:
                self.line_clicks.pop()
                message = f"Pending line click removed ({len(self.line_clicks)} remaining)"
            elif len(self.clicked_points) > len(self.world_points):
                self.clicked_points.pop()
                message = "Pending point click removed"
            elif self.annotated_lines:
                removed = self.annotated_lines.pop()
                self._compute_line_intersections()
                message = f"Line '{removed['name']}' removed"
            elif self.clicked_points:
                removed_name = self.keypoint_names[-1] if self.keypoint_names else "unknown"
                self.clicked_points.pop()
                if self.world_points:
                    self.world_points.pop()
                if self.keypoint_names:
                    self.keypoint_names.pop()
                message = f"Point '{removed_name}' removed"

            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            vis = self._visualize_annotations(frame_rgb)
            pitch_diagram = self._create_pitch_diagram(
                annotated_keypoints=self.keypoint_names,
                annotated_lines=self._get_annotated_line_names(),
            )
            return vis, message, self._get_annotations_text(), pitch_diagram, self._render_tactical_view()

        def reset_annotations():
            self._reset_state()
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            return (
                frame_rgb,
                "All annotations reset",
                "",
                self._create_pitch_diagram(annotated_keypoints=[], annotated_lines=[]),
                self._render_tactical_view(),
            )

        def compute_homography():
            frame_rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
            current_vis = self._visualize_annotations(frame_rgb)
            tactical_view = self._render_tactical_view()

            all_pixel_pts: list[tuple[float, float]] = []
            all_world_pts: list[tuple[float, float]] = []

            for i in range(len(self.keypoint_names)):
                all_pixel_pts.append(self.clicked_points[i])
                all_world_pts.append(self.world_points[i])

            for pixel, world, _ in self.derived_points:
                all_pixel_pts.append(pixel)
                all_world_pts.append(world)

            total_points = len(all_pixel_pts)
            if total_points < 4:
                return current_vis, f"Need at least 4 points. Current: {total_points}", self._get_annotations_text(), tactical_view

            if len({(round(x, 3), round(y, 3)) for x, y in all_world_pts}) < 4:
                return current_vis, "Need 4 DIFFERENT world points.", self._get_annotations_text(), tactical_view

            H, mask, mean_error = compute_homography_ls(all_pixel_pts, all_world_pts)
            if H is None:
                return current_vis, "Homography computation failed.", self._get_annotations_text(), tactical_view

            self.reprojection_error = mean_error
            self.H0 = H
            self._compute_auto_projections()

            vis = self._visualize_annotations(frame_rgb)
            vis = self._draw_pitch_projection(vis, H)
            tactical_view = self._render_tactical_view()

            inliers = int(np.sum(mask)) if mask is not None else total_points
            num_projected = len(self.auto_projected_points)
            status = (
                f"SUCCESS! H0 from {total_points} pts. "
                f"Error: {self.reprojection_error:.2f}m, "
                f"Inliers: {inliers}/{total_points}, "
                f"Projected: {num_projected}"
            )
            return vis, status, self._get_annotations_text(), tactical_view

        def save_and_close():
            total_pts = len(self.keypoint_names) + len(self.derived_points)
            if total_pts == 0:
                return "Error: No annotations to save!", gr.update()

            success = self._save_frame_annotation(self.current_frame_idx)
            if not success:
                return f"Error: Failed to save frame {self.current_frame_idx}", gr.update()

            new_choices = [str(f) for f in self.annotated_frames]
            save_dir = self._get_frame_dir()
            if self.H0 is None:
                msg = f"Saved frame {self.current_frame_idx} to {save_dir}/ ({total_pts} points)"
            else:
                msg = (
                    f"Saved frame {self.current_frame_idx} to {save_dir}/ "
                    f"({total_pts} points, error: {self.reprojection_error:.2f}m)"
                )
            return msg, gr.update(choices=new_choices, value=str(self.current_frame_idx))

        # -- Layout --
        with gr.Blocks(title="Soccer Pitch Annotator") as demo:
            gr.Markdown("# Soccer Pitch Keypoint & Line Annotator (HRNet 57 Points)")

            with gr.Row():
                with gr.Column(scale=2):
                    image = gr.Image(value=initial_frame, label="Video Frame", interactive=True)
                with gr.Column(scale=1):
                    pitch_diagram = gr.Image(value=initial_pitch_diagram, label="Reference", interactive=False)
                    tactical_view = gr.Image(value=initial_tactical_view, label="Tactical View", interactive=False)

            with gr.Row():
                status = gr.Textbox(label="Status", value="Click on image to mark points", lines=2)
                annotations = gr.Textbox(
                    label="Annotations",
                    value=self._get_annotations_text() if self.clicked_points else "",
                    lines=8,
                    interactive=False,
                )

            frame_slider = gr.Slider(
                minimum=0, maximum=self.total_frames - 1, step=1, value=start_frame, label="Frame",
            )

            anno_frame_choices = [str(f) for f in self.annotated_frames] if self.annotated_frames else []
            initial_anno_value = str(start_frame) if start_frame in self.annotated_frames else None

            with gr.Row():
                frame_dropdown = gr.Dropdown(
                    choices=anno_frame_choices, value=initial_anno_value,
                    label="Jump to Annotated Frame",
                )

            with gr.Row():
                mode_radio = gr.Radio(choices=["point", "line"], value="point", label="Mode")
                mode_status = gr.Textbox(value="Mode: POINT annotation", label="", interactive=False)

            with gr.Row():
                keypoint_dropdown = gr.Dropdown(
                    choices=keypoint_choices, value=keypoint_choices[0], label="Keypoint",
                )
                add_btn = gr.Button("Add Annotation", variant="primary")
                line_dropdown = gr.Dropdown(choices=line_choices, value=line_choices[0], label="Line")
                add_line_btn = gr.Button("Add Line")

            with gr.Row():
                undo_btn = gr.Button("Undo Last")
                reset_btn = gr.Button("Reset All")
                compute_btn = gr.Button("Compute Homography", variant="primary")

            save_btn = gr.Button("Save & Exit", variant="stop")

            image.select(fn=on_image_click, outputs=[image, status])
            frame_slider.change(
                fn=on_frame_change, inputs=[frame_slider],
                outputs=[image, annotations, pitch_diagram, tactical_view],
            )
            frame_dropdown.change(
                fn=on_frame_dropdown_change, inputs=[frame_dropdown],
                outputs=[image, status, annotations, pitch_diagram, tactical_view, frame_slider],
            )
            keypoint_dropdown.change(
                fn=on_keypoint_select, inputs=[keypoint_dropdown], outputs=[pitch_diagram],
            )
            line_dropdown.change(
                fn=on_line_select, inputs=[line_dropdown], outputs=[pitch_diagram],
            )
            mode_radio.change(fn=set_mode, inputs=[mode_radio], outputs=[mode_status])
            add_btn.click(
                fn=add_annotation, inputs=[keypoint_dropdown],
                outputs=[image, status, annotations, pitch_diagram, tactical_view],
            )
            add_line_btn.click(
                fn=add_line_annotation, inputs=[line_dropdown],
                outputs=[image, status, annotations, pitch_diagram, tactical_view],
            )
            undo_btn.click(
                fn=undo_last,
                outputs=[image, status, annotations, pitch_diagram, tactical_view],
            )
            reset_btn.click(
                fn=reset_annotations,
                outputs=[image, status, annotations, pitch_diagram, tactical_view],
            )
            compute_btn.click(
                fn=compute_homography,
                outputs=[image, status, annotations, tactical_view],
            )
            save_btn.click(fn=save_and_close, outputs=[status, frame_dropdown])

        demo.launch(server_port=port, share=share, prevent_thread_lock=False)

        if self.cap:
            self.cap.release()

        return self.current_frame_idx, self.H0
