"""HRNetv2-w48 for soccer field keypoint detection.

This module implements the HRNet architecture used by PnLCalib for detecting
soccer field keypoints. The model outputs 58 keypoint heatmaps.

Reference:
    - Sun et al., "Deep High-Resolution Representation Learning for Visual Recognition"
    - PnLCalib uses HRNetv2-w48 backbone
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for HRNet."""

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """Bottleneck residual block for HRNet."""

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    """Multi-resolution module for HRNet."""

    def __init__(
        self,
        num_branches: int,
        block: type,
        num_blocks: list[int],
        num_channels: list[int],
        multi_scale_output: bool = True,
    ):
        super().__init__()
        self._check_branches(num_branches, num_blocks, num_channels)

        self.num_branches = num_branches
        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers(num_channels)
        self.relu = nn.ReLU(inplace=True)

    def _check_branches(
        self, num_branches: int, num_blocks: list[int], num_channels: list[int]
    ) -> None:
        if num_branches != len(num_blocks):
            raise ValueError(f"num_branches({num_branches}) != len(num_blocks)({len(num_blocks)})")
        if num_branches != len(num_channels):
            raise ValueError(
                f"num_branches({num_branches}) != len(num_channels)({len(num_channels)})"
            )

    def _make_one_branch(
        self,
        branch_index: int,
        block: type,
        num_blocks: int,
        num_channels: int,
        stride: int = 1,
    ) -> nn.Sequential:
        layers = []
        layers.append(
            block(
                num_channels * block.expansion if branch_index > 0 else num_channels,
                num_channels,
                stride,
                downsample=None,
            )
        )
        for _ in range(1, num_blocks):
            layers.append(block(num_channels * block.expansion, num_channels))

        return nn.Sequential(*layers)

    def _make_branches(
        self,
        num_branches: int,
        block: type,
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.ModuleList:
        branches = []
        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks[i], num_channels[i]))
        return nn.ModuleList(branches)

    def _make_fuse_layers(self, num_channels: list[int]) -> nn.ModuleList:
        if self.num_branches == 1:
            return nn.ModuleList()

        fuse_layers = []
        for i in range(self.num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    # Upsample
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels[j] * BasicBlock.expansion,
                                num_channels[i] * BasicBlock.expansion,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(num_channels[i] * BasicBlock.expansion),
                            nn.Upsample(scale_factor=2 ** (j - i), mode="nearest"),
                        )
                    )
                elif j == i:
                    fuse_layer.append(nn.Identity())
                else:
                    # Downsample
                    conv_layers = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_layers.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_channels[j] * BasicBlock.expansion,
                                        num_channels[i] * BasicBlock.expansion,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_channels[i] * BasicBlock.expansion),
                                )
                            )
                        else:
                            conv_layers.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_channels[j] * BasicBlock.expansion,
                                        num_channels[j] * BasicBlock.expansion,
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        bias=False,
                                    ),
                                    nn.BatchNorm2d(num_channels[j] * BasicBlock.expansion),
                                    nn.ReLU(inplace=True),
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv_layers))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        # Process each branch
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        # Fuse
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = None
            for j in range(self.num_branches):
                if j == 0:
                    y = self.fuse_layers[i][j](x[j])
                else:
                    fuse_out = self.fuse_layers[i][j](x[j])
                    # Handle size mismatch due to rounding
                    if y.shape[2:] != fuse_out.shape[2:]:
                        fuse_out = F.interpolate(
                            fuse_out, size=y.shape[2:], mode="nearest"
                        )
                    y = y + fuse_out
            x_fuse.append(self.relu(y))

        return x_fuse


class HRNetKeypointModel(nn.Module):
    """HRNetv2-w48 for soccer field keypoint detection.

    Architecture:
    - Stage 1: 1 branch, 64 channels, BOTTLENECK
    - Stage 2: 2 branches, [48, 96] channels, BASIC
    - Stage 3: 3 branches, [48, 96, 192] channels, BASIC
    - Stage 4: 4 branches, [48, 96, 192, 384] channels, BASIC

    Input: (B, 3, 540, 960)
    Output: (B, num_keypoints, H/4, W/4) heatmaps

    Attributes:
        num_keypoints: Number of keypoints to detect (default 58 for PnLCalib).
    """

    def __init__(self, num_keypoints: int = 58, config: dict[str, Any] | None = None):
        """Initialize HRNet keypoint model.

        Args:
            num_keypoints: Number of keypoints to detect.
            config: Optional configuration dictionary.
        """
        super().__init__()
        self.num_keypoints = num_keypoints
        self.config = config or {}

        # HRNet-W48 configuration
        self.stage1_channels = 64
        self.stage2_channels = [48, 96]
        self.stage3_channels = [48, 96, 192]
        self.stage4_channels = [48, 96, 192, 384]

        # Stem: conv1 + conv2 with stride 4
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stage 1: Single branch with Bottleneck blocks
        self.layer1 = self._make_layer(Bottleneck, 64, 64, 4)

        # Transition layers
        self.transition1 = self._make_transition_layer([256], self.stage2_channels)
        self.transition2 = self._make_transition_layer(
            [c * BasicBlock.expansion for c in self.stage2_channels], self.stage3_channels
        )
        self.transition3 = self._make_transition_layer(
            [c * BasicBlock.expansion for c in self.stage3_channels], self.stage4_channels
        )

        # Stages 2-4: Multi-resolution modules
        self.stage2 = self._make_stage(
            num_modules=1,
            num_branches=2,
            num_blocks=[4, 4],
            num_channels=self.stage2_channels,
        )

        self.stage3 = self._make_stage(
            num_modules=4,
            num_branches=3,
            num_blocks=[4, 4, 4],
            num_channels=self.stage3_channels,
        )

        self.stage4 = self._make_stage(
            num_modules=3,
            num_branches=4,
            num_blocks=[4, 4, 4, 4],
            num_channels=self.stage4_channels,
        )

        # Head: Aggregate multi-resolution features and predict heatmaps
        # Sum of all channels from stage 4: 48 + 96 + 192 + 384 = 720
        # Plus skip connection from stem: 64
        # Total: 720 + 64 = 784
        self.stem_channels = 64
        final_inp_channels = sum(self.stage4_channels) + self.stem_channels
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # Note: PnLCalib uses nested Sequential and Conv2d with bias
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=1),
                nn.BatchNorm2d(final_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_inp_channels, num_keypoints, kernel_size=1),
                nn.Softmax(dim=1),
            )
        )

    def _make_layer(
        self, block: type, in_channels: int, out_channels: int, num_blocks: int
    ) -> nn.Sequential:
        """Create a layer of residual blocks."""
        downsample = None
        if in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * block.expansion, kernel_size=1, bias=False
                ),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = [block(in_channels, out_channels, downsample=downsample)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels * block.expansion, out_channels))

        return nn.Sequential(*layers)

    def _make_transition_layer(
        self, in_channels: list[int], out_channels: list[int]
    ) -> nn.ModuleList:
        """Create transition layers between stages."""
        num_branches_in = len(in_channels)
        num_branches_out = len(out_channels)

        transition_layers = []
        for i in range(num_branches_out):
            if i < num_branches_in:
                if in_channels[i] != out_channels[i] * BasicBlock.expansion:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels[i],
                                out_channels[i] * BasicBlock.expansion,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels[i] * BasicBlock.expansion),
                            nn.ReLU(inplace=True),
                        )
                    )
                else:
                    transition_layers.append(nn.Identity())
            else:
                # New branch with downsampling
                conv_layers = []
                for j in range(i + 1 - num_branches_in):
                    in_ch = in_channels[-1] if j == 0 else out_channels[i] * BasicBlock.expansion
                    conv_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                in_ch,
                                out_channels[i] * BasicBlock.expansion,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_channels[i] * BasicBlock.expansion),
                            nn.ReLU(inplace=True),
                        )
                    )
                transition_layers.append(nn.Sequential(*conv_layers))

        return nn.ModuleList(transition_layers)

    def _make_stage(
        self,
        num_modules: int,
        num_branches: int,
        num_blocks: list[int],
        num_channels: list[int],
    ) -> nn.Sequential:
        """Create a stage with multiple high-resolution modules."""
        modules = []
        for i in range(num_modules):
            modules.append(
                HighResolutionModule(
                    num_branches=num_branches,
                    block=BasicBlock,
                    num_blocks=num_blocks,
                    num_channels=num_channels,
                    multi_scale_output=True,
                )
            )
        return nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W), expected (B, 3, 540, 960).

        Returns:
            Heatmap tensor of shape (B, num_keypoints, H/4, W/4).
        """
        # Stem - save skip connection after conv1 (before bn1)
        x = self.conv1(x)
        x_skip = x.clone()  # Skip connection: 64 channels at 1/2 resolution
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # Stage 1
        x = self.layer1(x)

        # Transition 1 and Stage 2
        x_list = []
        for i in range(2):
            x_list.append(self.transition1[i](x))
        y_list = self.stage2(x_list)

        # Transition 2 and Stage 3
        x_list = []
        for i in range(3):
            if i < 2:
                x_list.append(self.transition2[i](y_list[i]))
            else:
                x_list.append(self.transition2[i](y_list[-1]))
        y_list = self.stage3(x_list)

        # Transition 3 and Stage 4
        x_list = []
        for i in range(4):
            if i < 3:
                x_list.append(self.transition3[i](y_list[i]))
            else:
                x_list.append(self.transition3[i](y_list[-1]))
        y_list = self.stage4(x_list)

        # Upsample and concatenate all branches to highest resolution (1/4 res)
        h, w = y_list[0].shape[2:]
        x0_h = y_list[0]
        x1_h = F.interpolate(y_list[1], size=(h, w), mode="bilinear", align_corners=False)
        x2_h = F.interpolate(y_list[2], size=(h, w), mode="bilinear", align_corners=False)
        x3_h = F.interpolate(y_list[3], size=(h, w), mode="bilinear", align_corners=False)

        x = torch.cat([x0_h, x1_h, x2_h, x3_h], dim=1)  # 720 channels at 1/4 resolution

        # Upsample to 1/2 resolution and concatenate with skip connection
        x = self.upsample(x)  # Now at 1/2 resolution (same as x_skip)
        x = torch.cat([x, x_skip], dim=1)  # 720 + 64 = 784 channels

        # Head
        out = self.head(x)

        return out

    def load_pretrained(self, weights_path: str) -> None:
        """Load pretrained weights.

        Args:
            weights_path: Path to the weights file.
        """
        state_dict = torch.load(weights_path, map_location="cpu")

        # Handle different state dict formats
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        # Remove prefix if present (e.g., "module.")
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        # Load with partial matching (allows head to have different num_keypoints)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}

        # Check for head layer mismatch
        head_keys = [k for k in pretrained_dict if k.startswith("head")]
        for k in head_keys:
            if pretrained_dict[k].shape != model_dict[k].shape:
                del pretrained_dict[k]

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
