"""Fine-tune PnLCalib model using point annotations.

This script fine-tunes a pretrained PnLCalib HRNet model using manually
annotated keypoint data. Designed for small datasets (3-100 frames) with
safeguards against overfitting.

Usage:
    python -m src.goalinsight.field_registration.pnlcalib.finetune.train_finetune \
        --annotations_dir /path/to/annotations \
        --pretrained /path/to/PnLCalib/weights/SV_kp \
        --output_dir data/finetuned_models \
        --epochs 100 \
        --lr 1e-5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# PnLCalib keypoint world coordinates (57 keypoints, 0-indexed)
PNLCALIB_WORLD_COORDS = [
    [0., 0.], [52.5, 0.], [105., 0.], [0., 13.84], [16.5, 13.84], [88.5, 13.84], [105., 13.84],
    [0., 24.84], [5.5, 24.84], [99.5, 24.84], [105., 24.84], [0., 30.34], [0., 30.34],
    [105., 30.34], [105., 30.34], [0., 37.66], [0., 37.66], [105., 37.66], [105., 37.66],
    [0., 43.16], [5.5, 43.16], [99.5, 43.16], [105., 43.16], [0., 54.16], [16.5, 54.16],
    [88.5, 54.16], [105., 54.16], [0., 68.], [52.5, 68.], [105., 68.], [16.5, 26.68],
    [52.5, 24.85], [88.5, 26.68], [16.5, 41.31], [52.5, 43.15], [88.5, 41.31], [19.99, 32.29],
    [43.68, 31.53], [61.31, 31.53], [85., 32.29], [19.99, 35.7], [43.68, 36.46], [61.31, 36.46],
    [85., 35.7], [11., 34.], [16.5, 34.], [20.15, 34.], [46.03, 27.53], [58.97, 27.53],
    [43.35, 34.], [52.5, 34.], [61.5, 34.], [46.03, 40.47], [58.97, 40.47], [84.85, 34.],
    [88.5, 34.], [94., 34.]
]
PNLCALIB_KEYPOINTS = {i: (coords[0], coords[1]) for i, coords in enumerate(PNLCALIB_WORLD_COORDS)}

from ..hrnet import HRNetKeypointModel

from .point_dataloader import AugmentedPointDataset, CachedAugmentedDataset, PointAnnotationDataset


def draw_pitch_template(scale: int = 10) -> np.ndarray:
    """Draw soccer pitch template (105m x 68m)."""
    pitch_w = int(105 * scale)
    pitch_h = int(68 * scale)

    img = np.zeros((pitch_h, pitch_w, 3), dtype=np.uint8)
    img[:] = (34, 139, 34)  # Green

    color = (255, 255, 255)
    thickness = 2

    cv2.rectangle(img, (0, 0), (pitch_w-1, pitch_h-1), color, thickness)
    cx = int(52.5 * scale)
    cv2.line(img, (cx, 0), (cx, pitch_h), color, thickness)
    center = (cx, int(34 * scale))
    cv2.circle(img, center, int(9.15 * scale), color, thickness)
    cv2.circle(img, center, 3, color, -1)
    cv2.rectangle(img, (0, int(13.84*scale)), (int(16.5*scale), int(54.16*scale)), color, thickness)
    cv2.rectangle(img, (int(88.5*scale), int(13.84*scale)), (pitch_w, int(54.16*scale)), color, thickness)
    cv2.rectangle(img, (0, int(24.84*scale)), (int(5.5*scale), int(43.16*scale)), color, thickness)
    cv2.rectangle(img, (int(99.5*scale), int(24.84*scale)), (pitch_w, int(43.16*scale)), color, thickness)

    return img


def extract_keypoints_from_heatmaps(
    heatmaps: np.ndarray,
    conf_thresh: float = 0.3,
    down_ratio: int = 2,
) -> list:
    """Extract keypoint coordinates from heatmaps (0-indexed IDs)."""
    keypoints = []
    num_keypoints = heatmaps.shape[0] - 1

    for i in range(num_keypoints):
        hm = heatmaps[i]
        max_val = hm.max()
        if max_val > conf_thresh:
            max_idx = np.unravel_index(hm.argmax(), hm.shape)
            y, x = max_idx
            keypoints.append({
                'id': i,
                'x': float(x * down_ratio),
                'y': float(y * down_ratio),
                'confidence': float(max_val),
            })

    return keypoints


def visualize_validation(
    image: np.ndarray,
    pred_heatmaps: np.ndarray,
    gt_heatmaps: np.ndarray,
    sample_idx: int,
    conf_thresh: float = 0.3,
) -> np.ndarray:
    """Create visualization with predictions and ground truth on pitch."""
    h, w = image.shape[:2]
    scale = 10

    pitch = draw_pitch_template(scale)
    pitch_h, pitch_w = pitch.shape[:2]
    pitch_scale = h / pitch_h
    pitch_resized = cv2.resize(pitch, (int(pitch_w * pitch_scale), h))

    # Extract keypoints from predictions and GT
    pred_kps = extract_keypoints_from_heatmaps(pred_heatmaps, conf_thresh)
    gt_kps = extract_keypoints_from_heatmaps(gt_heatmaps, conf_thresh=0.5)

    # Draw on image
    vis_img = (image * 255).astype(np.uint8)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)

    # GT: green
    for kp in gt_kps:
        x, y = int(kp['x']), int(kp['y'])
        cv2.circle(vis_img, (x, y), 6, (0, 255, 0), 2)

    # Pred: yellow/red
    for kp in pred_kps:
        x, y = int(kp['x']), int(kp['y'])
        color = (0, 255, 255) if kp['confidence'] > 0.5 else (0, 165, 255)
        cv2.circle(vis_img, (x, y), 5, color, -1)

    cv2.putText(vis_img, f"Sample {sample_idx} | Pred: {len(pred_kps)} | GT: {len(gt_kps)}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Draw on pitch
    pitch_vis = pitch_resized.copy()
    # GT: green circles
    for kp in gt_kps:
        kp_id = kp['id']
        if kp_id in PNLCALIB_KEYPOINTS:
            world_x, world_y = PNLCALIB_KEYPOINTS[kp_id]
            px = int(world_x * scale * pitch_scale)
            py = int(world_y * scale * pitch_scale)
            cv2.circle(pitch_vis, (px, py), 8, (0, 255, 0), 2)

    # Pred: filled circles
    for kp in pred_kps:
        kp_id = kp['id']
        if kp_id in PNLCALIB_KEYPOINTS:
            world_x, world_y = PNLCALIB_KEYPOINTS[kp_id]
            px = int(world_x * scale * pitch_scale)
            py = int(world_y * scale * pitch_scale)
            color = (0, 255, 255) if kp['confidence'] > 0.5 else (0, 165, 255)
            cv2.circle(pitch_vis, (px, py), 5, color, -1)

    combined = np.hstack([vis_img, pitch_vis])
    return combined


class MaskedMSELoss(nn.Module):
    """MSE loss that only considers visible keypoints.

    Only computes loss for keypoint channels where mask == 1,
    ignoring invisible keypoints during training.

    Args:
        pos_weight: Extra weight on the Gaussian-peak pixels of a target
            heatmap (pixels where ``target > pos_thresh``). Each keypoint
            channel's target is a tiny Gaussian (~a dozen non-zero pixels)
            surrounded by ~130k zero pixels; under plain per-pixel MSE the
            network minimises loss by predicting ~0 everywhere, so peaks
            never climb toward 1.0 and most channels stay dead even on the
            training frame. Up-weighting the peak pixels forces the network
            to actually reproduce the Gaussian. ``1.0`` (default) reproduces
            the original unweighted behaviour exactly.
        pos_thresh: Target value above which a pixel counts as "peak".
    """

    def __init__(self, pos_weight: float = 1.0, pos_thresh: float = 0.1):
        super().__init__()
        self.pos_weight = float(pos_weight)
        self.pos_thresh = float(pos_thresh)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute masked MSE loss.

        Args:
            pred: Predicted heatmaps (B, C, H, W).
            target: Target heatmaps (B, C, H, W).
            mask: Binary mask (B, C-1) indicating visible keypoints.

        Returns:
            Mean squared error over visible keypoints only.
        """
        batch_size, num_channels, height, width = pred.shape

        # Expand mask to spatial dimensions
        # mask: (B, C-1) -> (B, C-1, 1, 1) -> broadcast to (B, C-1, H, W)
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)
        mask_spatial = mask_expanded.expand(-1, -1, height, width)

        # Per-pixel squared error for the keypoint channels.
        kp_se = (pred[:, :-1] - target[:, :-1]) ** 2

        if self.pos_weight != 1.0:
            # Weight the peak (Gaussian) pixels more heavily so the network
            # is pushed to reproduce the peak instead of collapsing to
            # all-zero. Denominator matches the weighting to keep a proper
            # weighted mean.
            peak = (target[:, :-1] > self.pos_thresh).float()
            weight = 1.0 + (self.pos_weight - 1.0) * peak
            kp_loss = kp_se * mask_spatial * weight
            denom = (mask_spatial * weight).sum()
        else:
            kp_loss = kp_se * mask_spatial
            denom = mask_spatial.sum()

        # Compute loss for background channel (always included)
        bg_loss = (pred[:, -1:] - target[:, -1:]) ** 2

        # Count number of visible keypoint pixels
        if denom == 0:
            # No visible keypoints, only compute background loss
            return bg_loss.mean()

        # Average loss
        kp_mean = kp_loss.sum() / denom
        bg_mean = bg_loss.mean()

        return 0.5 * kp_mean + 0.5 * bg_mean


def load_model_config(config_path: str | None = None) -> dict:
    """Load model configuration.

    Args:
        config_path: Path to config file, or None to use default.

    Returns:
        Config dictionary.
    """
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    # Default HRNet-W48 config for keypoint detection
    return {
        'MODEL': {
            'NAME': 'cls_hrnet',
            'NUM_JOINTS': 58,  # 57 keypoints + 1 background
            'EXTRA': {
                'STAGE2': {
                    'NUM_MODULES': 1,
                    'NUM_BRANCHES': 2,
                    'BLOCK': 'BASIC',
                    'NUM_BLOCKS': [4, 4],
                    'NUM_CHANNELS': [48, 96],
                    'FUSE_METHOD': 'SUM',
                },
                'STAGE3': {
                    'NUM_MODULES': 4,
                    'NUM_BRANCHES': 3,
                    'BLOCK': 'BASIC',
                    'NUM_BLOCKS': [4, 4, 4],
                    'NUM_CHANNELS': [48, 96, 192],
                    'FUSE_METHOD': 'SUM',
                },
                'STAGE4': {
                    'NUM_MODULES': 3,
                    'NUM_BRANCHES': 4,
                    'BLOCK': 'BASIC',
                    'NUM_BLOCKS': [4, 4, 4, 4],
                    'NUM_CHANNELS': [48, 96, 192, 384],
                    'FUSE_METHOD': 'SUM',
                },
                'FINAL_CONV_KERNEL': 1,
            },
        },
    }


def freeze_backbone(model: nn.Module, freeze: bool = True):
    """Freeze or unfreeze backbone layers.

    Args:
        model: HRNet model.
        freeze: If True, freeze backbone; if False, unfreeze.
    """
    # Freeze everything except the head
    for name, param in model.named_parameters():
        if 'head' not in name:
            param.requires_grad = not freeze


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch.

    Returns:
        Average training loss.
    """
    model.train()
    total_loss = 0.0

    for batch_idx, (images, targets, masks) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, targets, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    vis_dir: Path | None = None,
    epoch: int = 0,
    save_vis: bool = True,
    max_vis_samples: int = 10,
    conf_thresh: float = 0.3,
) -> float:
    """Validate model with optional visualization.

    Args:
        model: Model to validate.
        dataloader: Validation dataloader.
        loss_fn: Loss function.
        device: torch device.
        vis_dir: Base directory for visualizations.
        epoch: Current epoch number.
        save_vis: Whether to save visualizations.
        max_vis_samples: Maximum number of samples to visualize.
        conf_thresh: Confidence threshold for keypoint extraction.

    Returns:
        Average validation loss.
    """
    model.eval()
    total_loss = 0.0
    vis_count = 0

    # Create vis directory for this epoch
    epoch_vis_dir = None
    if save_vis and vis_dir:
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

            # Save visualizations
            if save_vis and epoch_vis_dir and vis_count < max_vis_samples:
                for i in range(images.shape[0]):
                    if vis_count >= max_vis_samples:
                        break

                    sample_idx = batch_idx * dataloader.batch_size + i
                    img_np = images[i].cpu().numpy().transpose(1, 2, 0)  # (C,H,W) -> (H,W,C)
                    pred_hm = outputs[i].cpu().numpy()
                    gt_hm = targets[i].cpu().numpy()

                    vis = visualize_validation(
                        img_np, pred_hm, gt_hm, sample_idx, conf_thresh
                    )

                    vis_path = epoch_vis_dir / f"sample_{sample_idx:03d}.jpg"
                    cv2.imwrite(str(vis_path), vis)
                    vis_count += 1

    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune PnLCalib model")
    parser.add_argument(
        "--annotations_dir",
        type=str,
        required=True,
        help=(
            "Directory containing *_all_points.json and *_raw.jpg files. "
            "Pass a comma-separated list (e.g. ``dirA,dirB``) to combine "
            "multiple per-video annotation directories into one training "
            "set — the annotate-page 'Train this group' button uses this "
            "to lump every video sharing a name prefix."
        ),
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained PnLCalib weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/finetuned_models",
        help="Directory to save fine-tuned model",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
        help="Learning rate (use small value for fine-tuning)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (use 1 for very small datasets)",
    )
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze backbone, only train head layers",
    )
    parser.add_argument(
        "--augment_factor",
        type=int,
        default=10,
        help="Data augmentation factor",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[960, 540],
        help="Image size (width height)",
    )
    parser.add_argument(
        "--zoom_range",
        type=float,
        nargs=2,
        default=[1.0, 1.5],
        help="Zoom factor range (min max)",
    )
    parser.add_argument(
        "--zoom_prob",
        type=float,
        default=0.5,
        help="Probability of applying zoom augmentation",
    )
    parser.add_argument(
        "--min_keypoints",
        type=int,
        default=4,
        help="Minimum keypoints required after zoom crop",
    )
    parser.add_argument(
        "--hflip_prob",
        type=float,
        default=0.5,
        help=(
            "Probability of horizontal flip + mirror-id swap. Set to 0 for "
            "fixed-side cameras: a hflipped image is never an inference-time "
            "view, so mirror-id swap synthesises false positives on the "
            "opposite-half keypoint channels."
        ),
    )
    parser.add_argument(
        "--cuda",
        type=str,
        default="0",
        help="CUDA device ID",
    )
    parser.add_argument(
        "--val_seed",
        type=int,
        default=42,
        help="Random seed for validation augmentation (ensures reproducibility)",
    )
    parser.add_argument(
        "--val_augment_factor",
        type=int,
        default=None,
        help="Augmentation factor for validation (default: same as training)",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Run validation every N epochs (default: 1)",
    )
    parser.add_argument(
        "--no_vis",
        action="store_true",
        help="Disable validation visualizations",
    )
    parser.add_argument(
        "--max_vis_samples",
        type=int,
        default=10,
        help="Max samples to visualize per epoch",
    )
    parser.add_argument(
        "--vis_conf_thresh",
        type=float,
        default=0.3,
        help="Confidence threshold for visualization",
    )
    parser.add_argument(
        "--pos_weight",
        type=float,
        default=1.0,
        help="Weight on the Gaussian-peak pixels in the heatmap MSE loss. "
             ">1 forces the network to reproduce peaks instead of collapsing "
             "to all-zero (fixes dead channels on small datasets, e.g. a "
             "single annotated frame). 1.0 = original unweighted behaviour. "
             "Try ~100 for very sparse data.",
    )
    parser.add_argument(
        "--pos_thresh",
        type=float,
        default=0.1,
        help="Target heatmap value above which a pixel is treated as a "
             "peak for --pos_weight (default: 0.1)",
    )

    args = parser.parse_args()

    # Handle vis flag
    args.save_vis = not args.no_vis

    # Setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create timestamped run directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Run Directory: {run_dir} ===\n")

    # Setup subdirectories
    model_dir = run_dir / "models"
    model_dir.mkdir(exist_ok=True)

    vis_dir = run_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(exist_ok=True)
        print(f"Visualizations: {vis_dir}")

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard: tensorboard --logdir {log_dir}")

    # Create model
    print("Loading pretrained model...")
    model = HRNetKeypointModel(num_keypoints=58)
    model.load_pretrained(args.pretrained)
    model = model.to(device)

    # Optionally freeze backbone
    if args.freeze_backbone:
        print("Freezing backbone layers")
        freeze_backbone(model, freeze=True)

    # Resolve --annotations_dir which may be comma-separated. The
    # downstream Dataset classes accept ``str`` or ``list[str]`` and
    # union the file lists.
    ann_dirs = [d.strip() for d in args.annotations_dir.split(",") if d.strip()]
    annotations_arg = ann_dirs if len(ann_dirs) > 1 else ann_dirs[0]

    # Create dataset
    print(f"Loading annotations from {annotations_arg}")
    print(f"Zoom augmentation: range={args.zoom_range}, prob={args.zoom_prob}")
    train_dataset = AugmentedPointDataset(
        annotations_dir=annotations_arg,
        augment_factor=args.augment_factor,
        image_size=tuple(args.image_size),
        zoom_range=tuple(args.zoom_range),
        zoom_prob=args.zoom_prob,
        min_keypoints=args.min_keypoints,
        hflip_prob=args.hflip_prob,
    )

    # For validation, use pre-cached augmented data (fixed across epochs)
    val_augment = args.val_augment_factor if args.val_augment_factor else args.augment_factor
    val_dataset = CachedAugmentedDataset(
        annotations_dir=annotations_arg,
        augment_factor=val_augment,
        image_size=tuple(args.image_size),
        zoom_range=tuple(args.zoom_range),
        zoom_prob=args.zoom_prob,
        min_keypoints=args.min_keypoints,
        seed=args.val_seed,
        hflip_prob=args.hflip_prob,
    )

    print(f"Training samples: {len(train_dataset)} (with augmentation)")
    print(f"Validation samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for small datasets
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Loss and optimizer
    loss_fn = MaskedMSELoss(
        pos_weight=args.pos_weight,
        pos_thresh=args.pos_thresh,
    )
    if args.pos_weight != 1.0:
        print(
            f"Peak-weighted MSE: pos_weight={args.pos_weight}, "
            f"pos_thresh={args.pos_thresh}"
        )
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
    )

    # Training loop
    print("\nStarting training...")
    print(f"Epochs: {args.epochs}, LR: {args.lr}, Batch size: {args.batch_size}")
    print(f"Validation interval: every {args.val_interval} epochs")
    print("-" * 60)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(args.epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        history['train_loss'].append(train_loss)

        # Validate every N epochs
        run_val = (epoch + 1) % args.val_interval == 0 or epoch == args.epochs - 1

        if run_val:
            val_loss = validate(
                model, val_loader, loss_fn, device,
                vis_dir=vis_dir,
                epoch=epoch + 1,
                save_vis=args.save_vis,
                max_vis_samples=args.max_vis_samples,
                conf_thresh=args.vis_conf_thresh,
            )
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
        else:
            val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')

        # Log to console
        val_str = f"Val Loss: {val_loss:.6f}" if run_val else "Val: -"
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"{val_str} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        if run_val:
            writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch + 1)

        # Save best model (only when validation runs)
        if run_val:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                save_path = model_dir / "best_model.pt"
                torch.save(model.state_dict(), save_path)
                print(f"  -> Saved best model")
            else:
                patience_counter += 1

            # Early stopping (only check when validation runs)
            if patience_counter >= args.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

    # Close TensorBoard writer
    writer.close()

    # Save training history
    # Save training history
    history_path = run_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")

    # Save training config
    config_path = run_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Training config saved to {config_path}")

    # Print mapping report
    print("\n" + train_dataset.mapper.get_mapping_report())

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Run directory: {run_dir}")
    print(f"Best model: {model_dir / 'best_model.pt'}")
    print(f"TensorBoard: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    main()
