"""Video enhancement via video2x — upscaling and frame interpolation for highlight clips.

Supports two execution modes:

- **binary** (default): calls a local ``video2x`` executable directly.
- **docker**: runs the ``ghcr.io/k4yt3x/video2x`` container with GPU
  passthrough.  Required on systems where the host glibc is too old for the
  pre-built AppImage (e.g. Ubuntu 20.04).
"""

from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "mode": "docker",  # "binary" or "docker"
    "binary_path": None,
    "docker_image": "ghcr.io/k4yt3x/video2x:latest",
    "upscale": {
        "enabled": True,
        "processor": "realesrgan",
        "scale": 2,
        "model": None,
    },
    "interpolate": {
        "enabled": False,
        "processor": "rife",
        "multiplier": 2,
    },
    "encoder": {
        "codec": "libx264",
        "extra": {"crf": "17", "preset": "slow"},
    },
}


def run_video_enhancement(
    input_dir: Path,
    output_dir: Path,
    config: dict[str, Any] | None = None,
) -> list[Path]:
    """Enhance highlight clips using video2x (upscaling and/or frame interpolation).

    Args:
        input_dir: Directory containing highlight MP4 clips (from highlights stage).
        output_dir: Directory to write enhanced clips to.
        config: ``video_enhancement`` config section.

    Returns:
        List of paths to enhanced clips.
    """
    cfg = {**_DEFAULT_CONFIG, **(config or {})}
    mode = cfg.get("mode", "docker")

    if mode == "binary":
        binary = _find_video2x(cfg)
    else:
        binary = None  # Docker mode — no local binary needed

    clips = sorted(input_dir.rglob("*.mp4"))
    if not clips:
        logger.warning("No MP4 clips found in %s", input_dir)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Video enhancement ({mode} mode): {len(clips)} clip(s) to process")

    enhanced: list[Path] = []
    failed: list[str] = []

    for clip in clips:
        # Preserve subdirectory structure (e.g. goal_highlight/)
        relative = clip.relative_to(input_dir)
        clip_output_dir = output_dir / relative.parent
        clip_output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = _enhance_clip(binary, clip, clip_output_dir, cfg)
            if result is not None:
                enhanced.append(result)
                print(f"    Enhanced: {relative}")
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            logger.warning("Failed to enhance %s: %s", clip.name, exc)
            failed.append(str(relative))

    # Write summary
    summary = {
        "enhanced_clips": [str(p) for p in enhanced],
        "failed_clips": failed,
        "config": {
            "mode": mode,
            "upscale": cfg.get("upscale", {}),
            "interpolate": cfg.get("interpolate", {}),
            "encoder": cfg.get("encoder", {}),
        },
    }
    with open(output_dir / "enhancement_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Video enhancement complete: {len(enhanced)} enhanced, {len(failed)} failed")
    return enhanced


# ---------------------------------------------------------------------------
# Binary mode helpers
# ---------------------------------------------------------------------------


def _find_video2x(config: dict[str, Any]) -> Path:
    """Locate the video2x binary."""
    explicit = config.get("binary_path")
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p
        raise RuntimeError(f"video2x binary not found at configured path: {explicit}")

    found = shutil.which("video2x")
    if found:
        return Path(found)

    raise RuntimeError(
        "video2x binary not found in PATH. "
        "Install video2x (https://github.com/k4yt3x/video2x) or set "
        "video_enhancement.binary_path in config, "
        "or use mode: docker."
    )


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


def _video2x_args(
    processor: str,
    proc_args: list[str],
    encoder_cfg: dict[str, Any],
) -> list[str]:
    """Build the video2x arguments (without binary, -i, -o)."""
    args = ["-p", processor]
    args.extend(proc_args)
    codec = encoder_cfg.get("codec", "libx264")
    args.extend(["-c", codec])
    for key, value in encoder_cfg.get("extra", {}).items():
        args.extend(["-e", f"{key}={value}"])
    return args


def _build_binary_cmd(
    binary: Path,
    input_path: Path,
    output_path: Path,
    v2x_args: list[str],
) -> list[str]:
    """Build a direct video2x subprocess command."""
    return [str(binary), "-i", str(input_path), "-o", str(output_path)] + v2x_args


_DOCKER_DRIVER_FIX = (
    'for lib in /usr/lib/lib*nvidia*.so.535.*; do '
    '  base=${lib%.535*}; '
    '  ln -sf "$lib" "$base" 2>/dev/null; '
    '  for s in .0 .1; do '
    '    [ -L "${base}${s}" ] && ln -sf "$lib" "${base}${s}" || true; '
    '  done; '
    'done'
)


def _build_docker_cmd(
    image: str,
    input_path: Path,
    output_path: Path,
    v2x_args: list[str],
) -> list[str]:
    """Build a ``docker run`` command that runs video2x with GPU access.

    Mounts the input file and output directory into ``/data``, fixes the
    NVIDIA driver symlinks (host driver 535 vs container-bundled 565),
    and runs ``video2x`` inside the container.
    """
    input_abs = input_path.resolve()
    output_abs = output_path.resolve()
    out_dir = output_abs.parent

    # All paths and v2x_args are interpolated into a `bash -c` string for the
    # driver-symlink fixup, so shell-quote every value that came from event
    # metadata, config, or filenames. Without this, a clip filename containing
    # shell metacharacters would inject commands into the docker exec.
    quoted_in = shlex.quote(f"/data/input/{input_abs.name}")
    quoted_out = shlex.quote(f"/data/output/{output_abs.name}")
    quoted_args = " ".join(shlex.quote(a) for a in v2x_args)

    cmd = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v", f"{input_abs}:/data/input/{input_abs.name}:ro",
        "-v", f"{out_dir}:/data/output",
        "--entrypoint", "bash",
        image,
        "-c",
        f'{_DOCKER_DRIVER_FIX} && '
        f'/usr/bin/video2x -i {quoted_in} -o {quoted_out} {quoted_args}',
    ]
    return cmd


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _run_video2x(cmd: list[str], clip_name: str, pass_label: str) -> None:
    """Execute a video2x command with logging."""
    logger.info("[%s] %s: %s", pass_label, clip_name, " ".join(cmd))
    t0 = time.monotonic()

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        logger.error(
            "video2x %s failed for %s (exit %d):\n%s",
            pass_label, clip_name, exc.returncode, exc.stderr,
        )
        raise

    elapsed = time.monotonic() - t0
    logger.info("[%s] %s completed in %.1fs", pass_label, clip_name, elapsed)
    print(f"      {pass_label} done in {elapsed:.1f}s")


def _make_cmd(
    binary: Path | None,
    input_path: Path,
    output_path: Path,
    v2x_args: list[str],
    config: dict[str, Any],
) -> list[str]:
    """Dispatch to binary or docker command builder."""
    mode = config.get("mode", "docker")
    if mode == "binary":
        return _build_binary_cmd(binary, input_path, output_path, v2x_args)  # type: ignore[arg-type]
    image = config.get("docker_image", "ghcr.io/k4yt3x/video2x:latest")
    return _build_docker_cmd(image, input_path, output_path, v2x_args)


def _enhance_clip(
    binary: Path | None,
    clip_path: Path,
    output_dir: Path,
    config: dict[str, Any],
) -> Path | None:
    """Enhance a single clip (one or two passes)."""
    upscale_cfg = config.get("upscale", {})
    interp_cfg = config.get("interpolate", {})
    encoder_cfg = config.get("encoder", {})

    do_upscale = upscale_cfg.get("enabled", False)
    do_interpolate = interp_cfg.get("enabled", False)

    if not do_upscale and not do_interpolate:
        return None

    final_path = output_dir / clip_path.name
    intermediate = output_dir / f"_tmp_upscaled_{clip_path.name}"

    try:
        if do_upscale and do_interpolate:
            # Pass 1: upscale → intermediate
            args1 = _video2x_args(
                upscale_cfg.get("processor", "realesrgan"),
                _upscale_args(upscale_cfg), encoder_cfg,
            )
            _run_video2x(
                _make_cmd(binary, clip_path, intermediate, args1, config),
                clip_path.name, "upscale",
            )
            # Pass 2: interpolate intermediate → final
            args2 = _video2x_args(
                interp_cfg.get("processor", "rife"),
                _interpolate_args(interp_cfg), encoder_cfg,
            )
            _run_video2x(
                _make_cmd(binary, intermediate, final_path, args2, config),
                clip_path.name, "interpolate",
            )
        elif do_upscale:
            args = _video2x_args(
                upscale_cfg.get("processor", "realesrgan"),
                _upscale_args(upscale_cfg), encoder_cfg,
            )
            _run_video2x(
                _make_cmd(binary, clip_path, final_path, args, config),
                clip_path.name, "upscale",
            )
        else:
            args = _video2x_args(
                interp_cfg.get("processor", "rife"),
                _interpolate_args(interp_cfg), encoder_cfg,
            )
            _run_video2x(
                _make_cmd(binary, clip_path, final_path, args, config),
                clip_path.name, "interpolate",
            )
    finally:
        intermediate.unlink(missing_ok=True)

    return final_path


# ---------------------------------------------------------------------------
# Processor-specific argument builders
# ---------------------------------------------------------------------------


def _upscale_args(cfg: dict[str, Any]) -> list[str]:
    """Build processor-specific args for upscaling."""
    args = ["-s", str(cfg.get("scale", 2))]
    model = cfg.get("model")
    if model:
        processor = cfg.get("processor", "realesrgan")
        if processor == "realesrgan":
            args.extend(["--realesrgan-model", model])
        elif processor == "realcugan":
            args.extend(["--realcugan-model", model])
        elif processor == "libplacebo":
            args.extend(["--libplacebo-shader", model])
    return args


def _interpolate_args(cfg: dict[str, Any]) -> list[str]:
    """Build processor-specific args for frame interpolation."""
    args = ["-m", str(cfg.get("multiplier", 2))]
    model = cfg.get("model")
    if model:
        args.extend(["--rife-model", model])
    return args
