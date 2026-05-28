"""Manual pitch keypoint annotation UI (FastAPI + HTML/JS).

Used to produce ground-truth point annotations for fine-tuning the PnLCalib
HRNet keypoint model. Output format is consumed directly by
`goalinsight.field_registration.pnlcalib.finetune.point_dataloader`.

World-coordinate convention is y-up (top = +W/2), matching PnLCalib.
"""

from .annotation_io import (
    load_frame_annotation,
    save_frame_annotation,
)
from .homography import (
    camera_to_image_to_world,
    line_intersection,
    project_camera_point,
    solve_camera,
)
from . import pitch_constants
from .index import AnnotationIndex
from .keypoint_utils import (
    abbreviate_line_name,
    find_nearest_keypoint,
    get_hrnet_keypoint_choices,
    parse_keypoint_choice,
)
from .pitch.geometry import SoccerPitch
from .pitch_diagram import create_pitch_diagram


def __getattr__(name: str):
    # Lazy import so users who only need headless annotation I/O don't have to
    # install opencv/cv2 dependencies that ui.py pulls in.
    if name == "AnchorAnnotator":
        from .ui import AnchorAnnotator
        return AnchorAnnotator
    if name == "create_app":
        from .web import create_app
        return create_app
    if name == "run_server":
        from .web import run_server
        return run_server
    # PITCH_LINES is mutable (set_active_pitch); read at access time.
    if name == "PITCH_LINES":
        return pitch_constants.PITCH_LINES
    raise AttributeError(f"module 'goalinsight.annotation' has no attribute {name!r}")


__all__ = [
    "AnchorAnnotator",
    "AnnotationIndex",
    "SoccerPitch",
    "PITCH_LINES",
    "get_hrnet_keypoint_choices",
    "parse_keypoint_choice",
    "find_nearest_keypoint",
    "abbreviate_line_name",
    "create_pitch_diagram",
    "camera_to_image_to_world",
    "line_intersection",
    "project_camera_point",
    "solve_camera",
    "load_frame_annotation",
    "save_frame_annotation",
]
