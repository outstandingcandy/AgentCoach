"""Manual pitch keypoint annotation UI (Gradio).

Used to produce ground-truth point annotations for fine-tuning the PnLCalib
HRNet keypoint model. Output format is consumed directly by
`goalinsight.field_registration.pnlcalib.finetune.point_dataloader`.

World-coordinate convention is y-up (top = +W/2), matching PnLCalib.
"""

from .annotation_io import (
    load_frame_annotation,
    load_from_json,
    save_frame_annotation,
)
from .homography import (
    compute_homography_ls,
    compute_homography_with_pnp,
    line_intersection,
    project_3d_with_pnp,
    project_world_to_image,
)
from .index import AnnotationIndex, migrate_legacy_annotations
from .keypoint_utils import (
    LEGACY_TO_HRNET,
    PITCH_KEYPOINTS,
    abbreviate_line_name,
    convert_keypoint_name,
    find_nearest_keypoint,
    get_hrnet_keypoint_choices,
    parse_keypoint_choice,
)
from .pitch_constants import PITCH_LENGTH, PITCH_LINES, PITCH_WIDTH
from .pitch_diagram import create_pitch_diagram


def __getattr__(name: str):
    # Lazy import so users who only need headless annotation I/O don't have to
    # install gradio.
    if name == "AnchorAnnotator":
        from .ui import AnchorAnnotator
        return AnchorAnnotator
    raise AttributeError(f"module 'goalinsight.annotation' has no attribute {name!r}")


__all__ = [
    "AnchorAnnotator",
    "AnnotationIndex",
    "migrate_legacy_annotations",
    "LEGACY_TO_HRNET",
    "PITCH_KEYPOINTS",
    "PITCH_LINES",
    "PITCH_LENGTH",
    "PITCH_WIDTH",
    "get_hrnet_keypoint_choices",
    "parse_keypoint_choice",
    "convert_keypoint_name",
    "find_nearest_keypoint",
    "abbreviate_line_name",
    "create_pitch_diagram",
    "compute_homography_ls",
    "compute_homography_with_pnp",
    "line_intersection",
    "project_world_to_image",
    "project_3d_with_pnp",
    "load_frame_annotation",
    "save_frame_annotation",
    "load_from_json",
]
