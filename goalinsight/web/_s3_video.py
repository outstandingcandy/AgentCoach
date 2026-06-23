"""S3-backed video delivery for the web app.

Two responsibilities:

- ``upload_run_videos`` — sync every mp4 produced by a pipeline run
  (consolidated tracking, annotated HUD, highlight clips, per-player
  spotlights) up to a private S3 bucket. mp4s are remuxed with
  ``+faststart`` first so the browser can begin playback before the
  whole file finishes downloading.
- ``presigned_video_url`` — issue a short-lived signed GET URL the
  browser can hit directly. The web app redirects ``/api/runs/<run>/
  video`` to this URL when ``GOALINSIGHT_VIDEO_S3_BUCKET`` is set so
  EC2 stops being the byte-pump bottleneck.

When the env var is unset the helpers no-op and the FastAPI route
keeps serving from local disk — same behaviour as before, fine for
local development.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import boto3
from botocore.client import Config as BotoConfig
from botocore.exceptions import ClientError
from botocore.signers import CloudFrontSigner

logger = logging.getLogger(__name__)

# All mp4 outputs the web app might want to serve. Globs are relative
# to the run dir. Anything matching is uploaded to S3 under the same
# relative path.
VIDEO_GLOBS: tuple[str, ...] = (
    "track_consolidation/consolidated.mp4",
    "annotated_video/annotated.mp4",
    "annotated_video/annotated_web.mp4",
    "highlights/*.mp4",
    "player_profile/spotlights/*.mp4",
)


def video_bucket() -> str | None:
    """Return the configured bucket or None when video offloading is
    disabled."""
    return os.environ.get("GOALINSIGHT_VIDEO_S3_BUCKET", "").strip() or None


# ----------------------------------------------------------------------
# CloudFront signed URL support
# ----------------------------------------------------------------------
#
# When the three CloudFront env vars are set, presigned-URL helpers
# return a CF signed URL pointing at the configured distribution domain
# instead of an S3 sigv4 URL. CloudFront caches at edge POPs (~600
# globally), which is what cuts cross-region latency by an order of
# magnitude. The S3 path is kept as a fallback so the system still
# works in dev environments where you only have the bucket configured.


def cloudfront_domain() -> str | None:
    """Configured CF distribution domain (e.g. ``d123abc.cloudfront.net``).

    When set, ``presigned_*`` helpers route to CloudFront instead of
    signing the S3 URL directly. Optional — leaving this unset falls
    back to S3 sigv4 URLs.
    """
    return os.environ.get("GOALINSIGHT_VIDEO_CLOUDFRONT_DOMAIN", "").strip() or None


def _cf_key_pair_id() -> str | None:
    return os.environ.get("GOALINSIGHT_VIDEO_CF_KEY_PAIR_ID", "").strip() or None


def _cf_private_key_path() -> str | None:
    return os.environ.get("GOALINSIGHT_VIDEO_CF_PRIVATE_KEY_PATH", "").strip() or None


@lru_cache(maxsize=1)
def _cf_signer() -> "CloudFrontSigner | None":
    """Build a CloudFrontSigner from env-configured credentials.

    Cached so we don't re-parse the PEM on every URL signature.
    Returns None when CF isn't configured or any credential is missing.
    """
    domain = cloudfront_domain()
    key_pair_id = _cf_key_pair_id()
    key_path = _cf_private_key_path()
    if not (domain and key_pair_id and key_path):
        return None
    try:
        # Local import keeps ``cryptography`` optional for dev installs
        # that aren't using CloudFront.
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding
    except ImportError:
        logger.warning(
            "cryptography not installed — CloudFront signed URLs disabled. "
            "pip install cryptography to enable.")
        return None
    try:
        with open(key_path, "rb") as f:
            private_key = serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend(),
            )
    except (OSError, ValueError) as exc:
        logger.warning("failed to load CF private key %s: %s", key_path, exc)
        return None

    def _rsa_sign(message: bytes) -> bytes:
        return private_key.sign(message, padding.PKCS1v15(), hashes.SHA1())

    return CloudFrontSigner(key_pair_id, _rsa_sign)


def _cf_signed_url(key: str, expires: int) -> str | None:
    signer = _cf_signer()
    domain = cloudfront_domain()
    if signer is None or not domain:
        return None
    url = f"https://{domain}/{key.lstrip('/')}"
    expires_at = _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(seconds=int(expires))
    try:
        return signer.generate_presigned_url(url, date_less_than=expires_at)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CloudFront sign failed for %s: %s", key, exc)
        return None


def _video_region() -> str:
    return os.environ.get("AWS_REGION", "us-east-1")


def _client() -> "boto3.client":
    # The presigner uses sigv4; an explicit region keeps the signature
    # routable when the EC2 role's default region differs from the
    # bucket's. Address style ``virtual`` matches what browsers expect.
    return boto3.client(
        "s3",
        region_name=_video_region(),
        config=BotoConfig(signature_version="s3v4", s3={"addressing_style": "virtual"}),
    )


def _faststart_remux(src: Path) -> Path:
    """Return a path to a copy of *src* with the moov atom moved to
    the front. ``-c copy`` keeps the original streams (no re-encode),
    just rewrites the container. Falls back to the original file when
    ffmpeg is unavailable or fails.
    """
    if not shutil.which("ffmpeg"):
        return src
    out = Path(tempfile.gettempdir()) / f"faststart_{os.getpid()}_{src.name}"
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y",
        "-i", str(src),
        "-c", "copy", "-movflags", "+faststart",
        str(out),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=300)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        logger.warning("faststart remux failed for %s: %s — uploading raw", src, exc)
        if out.exists():
            out.unlink()
        return src
    return out


def _expand_videos(run_dir: Path) -> list[Path]:
    out: list[Path] = []
    for glob in VIDEO_GLOBS:
        if "*" in glob:
            out.extend(p for p in run_dir.glob(glob) if p.is_file())
        else:
            p = run_dir / glob
            if p.is_file():
                out.append(p)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    dedup: list[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        dedup.append(p)
    return dedup


def _s3_key(run_name: str, run_dir: Path, video: Path) -> str:
    rel = video.relative_to(run_dir).as_posix()
    return f"runs/{run_name}/{rel}"


def source_video_key(video_path: Path) -> str:
    """S3 key for an input source video (lives in workspace/videos/).

    Source files are shared across runs so we key them by basename
    rather than run name. Storing them under their own prefix keeps
    them separate from per-run derived mp4s.
    """
    return f"videos/{video_path.name}"


def upload_source_video(video_path: Path, bucket: str | None = None) -> str | None:
    """Upload a source video (faststart-remuxed) to ``s3://bucket/videos/``.

    Returns the S3 key on success, None when no bucket is configured.
    Skips when the remote object's size already matches local.
    """
    bucket = bucket or video_bucket()
    if not bucket or not video_path.is_file():
        return None
    s3 = _client()
    key = source_video_key(video_path)
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        if head.get("ContentLength") == video_path.stat().st_size:
            return key
    except ClientError:
        pass
    prepared = _faststart_remux(video_path)
    try:
        s3.upload_file(
            str(prepared), bucket, key,
            ExtraArgs={"ContentType": "video/mp4"},
        )
        return key
    except ClientError as exc:
        logger.warning("source video upload failed for %s: %s", video_path, exc)
        return None
    finally:
        if prepared != video_path and prepared.exists():
            try:
                prepared.unlink()
            except OSError:
                pass


def presigned_url_for_key(
    key: str, *, bucket: str | None = None, expires: int = 1800,
) -> str | None:
    """Sign an arbitrary S3 key (used for source-video keys that don't
    live under ``runs/<run>/...``).

    Prefers CloudFront signed URLs when configured (see
    ``cloudfront_domain``); falls back to S3 sigv4.
    """
    bucket = bucket or video_bucket()
    if not bucket:
        return None
    cf = _cf_signed_url(key, expires)
    if cf is not None:
        return cf
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires),
        )
    except ClientError as exc:
        logger.warning("presign failed for %s: %s", key, exc)
        return None


def s3_key_exists(key: str, *, bucket: str | None = None) -> bool:
    bucket = bucket or video_bucket()
    if not bucket:
        return False
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def upload_run_videos(
    run_dir: Path,
    run_name: str | None = None,
    bucket: str | None = None,
    *,
    skip_if_same_size: bool = True,
) -> Iterator[dict]:
    """Sync every mp4 under *run_dir* to S3.

    Yields ``{video, key, status}`` events; status ∈ {"skipped",
    "uploaded", "failed"}. ``status="failed"`` carries an ``error``
    message — the iterator continues so a single bad file can't block
    the rest.
    """
    bucket = bucket or video_bucket()
    if not bucket:
        return
    run_name = run_name or run_dir.name
    s3 = _client()

    for video in _expand_videos(run_dir):
        key = _s3_key(run_name, run_dir, video)
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            if skip_if_same_size and head.get("ContentLength") == video.stat().st_size:
                yield {"video": str(video), "key": key, "status": "skipped"}
                continue
        except ClientError:
            pass

        prepared = _faststart_remux(video)
        try:
            s3.upload_file(
                str(prepared), bucket, key,
                ExtraArgs={"ContentType": "video/mp4"},
            )
            yield {"video": str(video), "key": key, "status": "uploaded"}
        except ClientError as exc:
            logger.warning("upload failed for %s: %s", video, exc)
            yield {
                "video": str(video), "key": key, "status": "failed",
                "error": str(exc),
            }
        finally:
            if prepared != video and prepared.exists():
                try:
                    prepared.unlink()
                except OSError:
                    pass


def presigned_video_url(
    run_name: str,
    rel_path: str,
    *,
    bucket: str | None = None,
    expires: int = 1800,
) -> str | None:
    """Return a short-lived GET URL for ``runs/<run>/<rel_path>``.

    *rel_path* is the path of the mp4 inside the run directory
    (e.g. ``annotated_video/annotated.mp4``). Prefers a CloudFront
    signed URL when CF env vars are set (so cross-region clients hit
    an edge POP); falls back to an S3 sigv4 URL otherwise. Returns
    None when no bucket is configured at all.
    """
    bucket = bucket or video_bucket()
    if not bucket:
        return None
    key = f"runs/{run_name}/{rel_path.lstrip('/')}"
    cf = _cf_signed_url(key, expires)
    if cf is not None:
        return cf
    try:
        return _client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=int(expires),
        )
    except ClientError as exc:
        logger.warning("presign failed for %s/%s: %s", run_name, rel_path, exc)
        return None


def s3_object_exists(run_name: str, rel_path: str, *, bucket: str | None = None) -> bool:
    """Cheap HEAD check — used to decide whether to redirect to S3 or
    fall back to the local FileResponse path."""
    bucket = bucket or video_bucket()
    if not bucket:
        return False
    key = f"runs/{run_name}/{rel_path.lstrip('/')}"
    try:
        _client().head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False
