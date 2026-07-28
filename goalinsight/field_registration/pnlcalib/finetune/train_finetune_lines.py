"""Fine-tune the PnLCalib HRNet line model on manual line annotations.

Mirrors ``train_finetune.py`` (the keypoint trainer) but swaps the model
class, dataset, loss (plain MSE matching upstream PnLCalib's line training
— see ``PlainMSELoss``), and visualisation. Reuses the optimiser schedule,
the train/validate loop, and the file-writing conventions.

Usage:
    python -m goalinsight.field_registration.pnlcalib.finetune.train_finetune_lines \
        --annotations_dir output/annotations/kids_soccer_clip_1250_1310 \
        --pretrained ~/.cache/goal-insight/pnlcalib/SV_lines \
        --output_dir data/finetuned_line_models \
        --epochs 200 --augment_factor 30
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..hrnet_line import HRNetLineModel
from .line_dataloader import (
    AugmentedLineDataset,
    CachedAugmentedLineDataset,
    LINE_NAME_TO_UPSTREAM_IDX,
)
from .train_finetune import (
    freeze_backbone,
    train_one_epoch,
)


class PlainMSELoss(nn.Module):
    """Upstream-faithful plain MSE for the line head.

    Mirrors upstream PnLCalib's ``model/losses.py::MSELoss`` (used in
    ``train_l.py``): ``nn.MSELoss(reduction='none').mean()`` over all 24
    channels (23 line classes + 1 border channel) with no per-class mask
    and no fg/bg reweighting.

    The ``mask`` arg is accepted for compatibility with the keypoint
    trainer's ``train_one_epoch`` loop but is ignored — upstream's
    SoccerNet line path also runs unmasked.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,  # noqa: ARG002 — upstream-faithful: no mask
    ) -> torch.Tensor:
        return ((pred - target) ** 2).mean()


def _decode_lines_from_heatmap(
    heatmaps: np.ndarray,
    threshold: float = 0.3,
) -> list[dict]:
    """Numpy port of ``get_lines_from_heatmap_maxpool`` for vis only.

    Skips background channel (last one). For each line channel, returns
    the top-2 local maxima as endpoints, gated by ``threshold``.
    """
    out: list[dict] = []
    n_classes = heatmaps.shape[0] - 1  # last is background
    for c in range(n_classes):
        hm = heatmaps[c]
        # Simple top-2 (no NMS — vis only). Avoid double-counting same pixel.
        flat = hm.flatten()
        if flat.size < 2:
            continue
        idx_sorted = np.argpartition(-flat, 2)[:2]
        idx_sorted = idx_sorted[np.argsort(-flat[idx_sorted])]
        s1, s2 = flat[idx_sorted[0]], flat[idx_sorted[1]]
        if s1 < threshold or s2 < threshold:
            continue
        h, w = hm.shape
        y1, x1 = idx_sorted[0] // w, idx_sorted[0] % w
        y2, x2 = idx_sorted[1] // w, idx_sorted[1] % w
        out.append({
            "id": c,
            "x1": int(x1), "y1": int(y1),
            "x2": int(x2), "y2": int(y2),
            "confidence": float(min(s1, s2)),
        })
    return out


def _draw_lines_overlay(
    image: np.ndarray,
    lines: list[dict],
    color: tuple[int, int, int],
    scale: int = 2,
) -> np.ndarray:
    """Draw decoded lines on an RGB image. ``scale`` lifts heatmap-res
    coordinates back to image-res (down_ratio=2 by default)."""
    vis = image.copy()
    for ln in lines:
        x1, y1 = int(ln["x1"]) * scale, int(ln["y1"]) * scale
        x2, y2 = int(ln["x2"]) * scale, int(ln["y2"]) * scale
        cv2.line(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis, f"L{ln['id']}({ln['confidence']:.2f})",
            ((x1 + x2) // 2, (y1 + y2) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1,
        )
    return vis


def visualize_lines(
    image: np.ndarray,
    pred_heatmaps: np.ndarray,
    gt_heatmaps: np.ndarray,
    sample_idx: int,
    conf_thresh: float = 0.3,
    down_ratio: int = 2,
) -> np.ndarray:
    """Draw GT (green) + predicted (yellow) line endpoints + segments."""
    img_bgr = (image * 255).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)

    gt = _decode_lines_from_heatmap(gt_heatmaps, threshold=0.5)
    pred = _decode_lines_from_heatmap(pred_heatmaps, threshold=conf_thresh)

    vis = _draw_lines_overlay(img_bgr, gt, (0, 255, 0), scale=down_ratio)
    vis = _draw_lines_overlay(vis, pred, (0, 255, 255), scale=down_ratio)

    cv2.putText(
        vis,
        f"Sample {sample_idx} | Pred: {len(pred)} | GT: {len(gt)}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    return vis


def validate_lines(
    model,
    dataloader,
    loss_fn,
    device,
    vis_dir: Path | None = None,
    epoch: int = 0,
    save_vis: bool = True,
    max_vis_samples: int = 10,
    conf_thresh: float = 0.3,
    down_ratio: int = 2,
) -> float:
    model.eval()
    total_loss = 0.0
    vis_count = 0
    epoch_vis_dir = None
    if save_vis and vis_dir is not None:
        epoch_vis_dir = vis_dir / f"epoch_{epoch:03d}"
        epoch_vis_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, (images, targets, masks) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, targets, masks)
            total_loss += loss.item()

            if save_vis and epoch_vis_dir is not None and vis_count < max_vis_samples:
                for i in range(images.shape[0]):
                    if vis_count >= max_vis_samples:
                        break
                    sample_idx = batch_idx * dataloader.batch_size + i
                    img_np = images[i].cpu().numpy().transpose(1, 2, 0)
                    vis = visualize_lines(
                        img_np,
                        outputs[i].cpu().numpy(),
                        targets[i].cpu().numpy(),
                        sample_idx,
                        conf_thresh=conf_thresh,
                        down_ratio=down_ratio,
                    )
                    cv2.imwrite(
                        str(epoch_vis_dir / f"sample_{sample_idx:03d}.jpg"),
                        vis,
                    )
                    vis_count += 1

    return total_loss / max(1, len(dataloader))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PnLCalib line model")
    parser.add_argument(
        "--annotations_dir", type=str, required=True,
        help=(
            "Annotation directory; pass a comma-separated list "
            "(e.g. ``dirA,dirB``) to combine multiple per-video dirs "
            "into one training set, matching the keypoint trainer."
        ),
    )
    parser.add_argument("--pretrained", type=str, required=True,
                        help="Path to pretrained SV_lines weights file")
    parser.add_argument("--output_dir", type=str, default="data/finetuned_line_models")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers; >0 parallelises augmentation.")
    parser.add_argument("--val_max_samples", type=int, default=200,
                        help="Cap validation set size (0 = all).")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--augment_factor", type=int, default=30)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument("--image_size", type=int, nargs=2, default=[960, 540])
    parser.add_argument("--zoom_range", type=float, nargs=2, default=[1.0, 1.5])
    parser.add_argument("--zoom_prob", type=float, default=0.5)
    parser.add_argument("--min_endpoints", type=int, default=4)
    parser.add_argument(
        "--hflip_prob", type=float, default=0.5,
        help="Horizontal-flip probability (ON by default so the model "
             "generalises to a camera on the other side of the pitch; the "
             "line mirror map keeps flipped samples correctly labelled).",
    )
    parser.add_argument("--down_ratio", type=int, default=2,
                        help="Heatmap downsample relative to input image")
    parser.add_argument("--sigma", type=float, default=2.0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--val_augment_factor", type=int, default=None)
    parser.add_argument("--val_interval", type=int, default=2)
    parser.add_argument("--no_vis", action="store_true")
    parser.add_argument("--max_vis_samples", type=int, default=10)
    parser.add_argument("--vis_conf_thresh", type=float, default=0.3)
    args = parser.parse_args()

    args.save_vis = not args.no_vis

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Run Directory: {run_dir} ===\n")

    model_dir = run_dir / "models"
    model_dir.mkdir(exist_ok=True)
    vis_dir = run_dir / "vis" if args.save_vis else None
    if vis_dir is not None:
        vis_dir.mkdir(exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    print("Loading pretrained line model...")
    model = HRNetLineModel(num_line_classes=23)
    model.load_pretrained(args.pretrained)
    model = model.to(device)

    if args.freeze_backbone:
        print("Freezing backbone (head-only finetune)")
        freeze_backbone(model, freeze=True)

    ann_dirs = [d.strip() for d in args.annotations_dir.split(",") if d.strip()]
    annotations_arg = ann_dirs if len(ann_dirs) > 1 else ann_dirs[0]
    print(f"Loading annotations from {annotations_arg}")
    train_dataset = AugmentedLineDataset(
        annotations_dir=annotations_arg,
        augment_factor=args.augment_factor,
        image_size=tuple(args.image_size),
        num_classes=23,
        down_ratio=args.down_ratio,
        sigma=args.sigma,
        zoom_range=tuple(args.zoom_range),
        zoom_prob=args.zoom_prob,
        min_endpoints=args.min_endpoints,
        hflip_prob=args.hflip_prob,
    )

    val_aug = args.val_augment_factor if args.val_augment_factor else args.augment_factor
    val_dataset = CachedAugmentedLineDataset(
        annotations_dir=annotations_arg,
        augment_factor=val_aug,
        image_size=tuple(args.image_size),
        num_classes=23,
        down_ratio=args.down_ratio,
        sigma=args.sigma,
        zoom_range=tuple(args.zoom_range),
        zoom_prob=args.zoom_prob,
        min_endpoints=args.min_endpoints,
        hflip_prob=args.hflip_prob,
        seed=args.val_seed,
    )
    if args.val_max_samples and len(val_dataset) > args.val_max_samples:
        from torch.utils.data import Subset
        step = len(val_dataset) / args.val_max_samples
        val_dataset = Subset(val_dataset, [int(i * step) for i in range(args.val_max_samples)])

    print(f"Training samples: {len(train_dataset)}  |  Validation: {len(val_dataset)}")

    _dl_kw = {}
    if args.num_workers > 0:
        _dl_kw = {"num_workers": args.num_workers, "persistent_workers": True,
                  "prefetch_factor": 2}
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **_dl_kw,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, **_dl_kw,
    )

    loss_fn = PlainMSELoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10,
    )

    print("\nStarting line-model training...")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch: {args.batch_size}")
    print(f"Class coverage in annotations: {sorted(set(LINE_NAME_TO_UPSTREAM_IDX.values()))}")
    print("-" * 60)

    best_val_loss = float("inf")
    patience = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history["train_loss"].append(train_loss)

        run_val = (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1
        if run_val:
            val_loss = validate_lines(
                model, val_loader, loss_fn, device,
                vis_dir=vis_dir,
                epoch=epoch + 1,
                save_vis=args.save_vis,
                max_vis_samples=args.max_vis_samples,
                conf_thresh=args.vis_conf_thresh,
                down_ratio=args.down_ratio,
            )
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
        else:
            val_loss = history["val_loss"][-1] if history["val_loss"] else float("inf")

        val_str = f"Val Loss: {val_loss:.6f}" if run_val else "Val: -"
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | {val_str} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        writer.add_scalar("Loss/train", train_loss, epoch + 1)
        if run_val:
            writer.add_scalar("Loss/val", val_loss, epoch + 1)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch + 1)

        if run_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
                torch.save(model.state_dict(), model_dir / "best_model.pt")
                print("  -> Saved best model")
            else:
                patience += 1
            if patience >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    writer.close()
    with open(run_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n{'='*60}\nTraining complete!  Best val loss: {best_val_loss:.6f}")
    print(f"Run dir: {run_dir}\nBest model: {model_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
