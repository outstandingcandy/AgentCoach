"""Entrypoint for the SageMaker Training Job that fine-tunes PnLCalib heads.

The on-disk layout is the standard SageMaker Training Job one:

    /opt/ml/input/data/annotations/<frame_*_all_points.json, frame_*_raw.jpg>
    /opt/ml/input/data/weights/SV_kp                  # pretrained KP head
    /opt/ml/input/data/weights/SV_lines               # pretrained line head
    /opt/ml/input/config/hyperparameters.json         # CLI overrides

    /opt/ml/model/run_<timestamp>/...                 # auto-uploaded to S3

Hyperparameters are passed in via the standard `hyperparameters.json` file
(SageMaker turns each TrainingJob hyperparameter into a key here). We
unpack them into argv-style flags and call the same `main()` already used
by local training, so trainer-side code stays unaware of the remote
context.

Usage (set `kind` hyperparameter to ``keypoint`` or ``line``):

    HyperParameters = {
        "kind": "keypoint",
        "epochs": "100",
        "batch_size": "4",
        "hflip_prob": "0",
        ...
    }

Anything not consumed by this entrypoint is forwarded directly to the
underlying trainer's argparse — which is the source of truth for valid
flags (see ``train_finetune.py:main`` and ``train_finetune_lines.py:main``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# SageMaker Training Job env layout — keep both the actual env vars and a
# /opt/ml fallback so the entrypoint also works in plain Docker for testing.
INPUT_DATA = Path(os.environ.get("SM_INPUT_DIR", "/opt/ml/input/data"))
HPARAMS_PATH = Path(os.environ.get(
    "SM_HP_PATH", "/opt/ml/input/config/hyperparameters.json",
))
MODEL_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))


# Hyperparameter keys consumed by THIS entrypoint (not forwarded to the
# trainer). Everything else gets translated to --<key> <value>.
_LOCAL_KEYS = {"kind"}


def _load_hyperparameters() -> dict[str, str]:
    if not HPARAMS_PATH.exists():
        return {}
    with open(HPARAMS_PATH) as f:
        raw = json.load(f)
    # SageMaker stores everything as strings; that matches what argparse
    # expects when re-parsing, so just pass them through.
    return {str(k): str(v) for k, v in raw.items()}


def _hparams_to_argv(hp: dict[str, str]) -> list[str]:
    """Turn {"epochs": "100", "batch_size": "4"} into ["--epochs", "100", ...].

    Boolean store_true flags are passed by setting the value to "true";
    we drop the value and emit only the flag in that case.
    """
    argv: list[str] = []
    for key, val in hp.items():
        if key in _LOCAL_KEYS:
            continue
        flag = f"--{key}"
        if val.lower() in {"true", "1", "yes"} and key in {"freeze_backbone", "no_vis"}:
            argv.append(flag)
        elif val.lower() in {"false", "0", "no"} and key in {"freeze_backbone", "no_vis"}:
            # action=store_true defaults to False — omitting the flag is enough.
            continue
        else:
            argv.extend([flag, val])
    return argv


def _resolve_pretrained(kind: str, hp: dict[str, str]) -> str:
    """Find the pretrained weight file uploaded by the client.

    The training-CLI client uploads the pretrained file as
    ``/opt/ml/input/data/weights/<basename>``. If the user explicitly
    passed ``--pretrained`` in hyperparameters, honour that path
    verbatim — it might point at a sibling location inside the image.
    """
    if "pretrained" in hp:
        return hp["pretrained"]
    weights = INPUT_DATA / "weights"
    # Conventional names: client uploads SV_kp / SV_lines verbatim, or any
    # file ending with .pt / matching the kind.
    candidates: list[Path] = []
    if weights.is_dir():
        for p in weights.iterdir():
            if not p.is_file():
                continue
            name = p.name.lower()
            if kind == "keypoint" and ("kp" in name or "keypoint" in name):
                candidates.append(p)
            elif kind == "line" and ("line" in name):
                candidates.append(p)
    if not candidates:
        raise RuntimeError(
            f"no pretrained {kind} weight found under {weights}; "
            f"upload SV_kp / SV_lines via TrainingInput name='weights'"
        )
    # Prefer the largest file (full HRNet weights vs partial heads).
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    return str(candidates[0])


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    hp = _load_hyperparameters()
    kind = hp.get("kind")
    if kind not in {"keypoint", "line"}:
        raise RuntimeError(
            f"hyperparameter 'kind' must be 'keypoint' or 'line', got {kind!r}"
        )

    annotations = INPUT_DATA / "annotations"
    if not annotations.is_dir():
        raise RuntimeError(f"annotations missing at {annotations}")

    pretrained = _resolve_pretrained(kind, hp)
    logger.info("kind=%s, annotations=%s, pretrained=%s",
                kind, annotations, pretrained)

    # Build trainer argv. The trainers' main() reads sys.argv directly;
    # patch it so we can call them as functions without subprocessing.
    base_argv = [
        "--annotations_dir", str(annotations),
        "--pretrained", pretrained,
        "--output_dir", str(MODEL_DIR),
    ]
    forwarded = _hparams_to_argv(hp)
    sys.argv = ["train_entrypoint.py"] + base_argv + forwarded
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if kind == "keypoint":
        from goalinsight.field_registration.pnlcalib.finetune.train_finetune import main as run
    else:
        from goalinsight.field_registration.pnlcalib.finetune.train_finetune_lines import main as run

    logger.info("invoking %s trainer with argv=%s", kind, sys.argv[1:])
    run()

    # The trainer wrote to <MODEL_DIR>/run_<timestamp>/. SageMaker tarballs
    # everything under MODEL_DIR into model.tar.gz — no further copy needed.
    runs = sorted(MODEL_DIR.glob("run_*"))
    if runs:
        logger.info("training output: %s", runs[-1])
    return 0


if __name__ == "__main__":
    sys.exit(main())
