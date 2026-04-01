"""Stage 1 backend dispatcher.

Routes to the appropriate calibration backend based on config.
"""

from pathlib import Path

from ..utils.config import get_default_config, get_process_fps_from_config


def run_stage1(video_path: Path, output_dir: Path, config: dict | None = None):
    """Run Stage 1 field registration.

    Args:
        video_path: Path to input video
        output_dir: Directory for output files
        config: Optional configuration dict

    Returns:
        Dict with calibration statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Load configuration
    if config is None:
        config = get_default_config()
    process_fps = get_process_fps_from_config(config)

    # Determine backend from config
    fr_config = config.get("field_registration", {})
    backend = fr_config.get("backend", "pnlcalib")
    print(f"Stage 1: Using {backend} backend for field registration")

    # Initialize based on backend
    if backend == "nbjw":
        from ._pnlcalib import _run_stage1_nbjw
        return _run_stage1_nbjw(video_path, output_dir, vis_dir, config, process_fps)
    elif backend == "broadtrack":
        from ._broadtrack import run_stage1_broadtrack
        return run_stage1_broadtrack(video_path, output_dir, vis_dir, config, process_fps)
    elif backend == "physical":
        from ._physical import run_stage1_physical
        return run_stage1_physical(video_path, output_dir, vis_dir, config, process_fps)
    elif backend == "homography":
        from ._homography import run_stage1_homography
        return run_stage1_homography(video_path, output_dir, vis_dir, config, process_fps)
    else:
        from ._pnlcalib import _run_stage1_pnlcalib
        return _run_stage1_pnlcalib(video_path, output_dir, vis_dir, config, process_fps)
