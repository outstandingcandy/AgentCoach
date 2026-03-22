"""HRNetv2-w48 for soccer field line detection.

This module implements the HRNet architecture used by PnLCalib for detecting
soccer field lines. The model outputs heatmaps for line classes.

PnLCalib line detection structure:
- Model outputs 24 channels (23 line classes + 1 background)
- Each line class channel contains BOTH endpoints as top-2 peaks
- The top 2 local maxima in each channel are the two endpoints of that line

Reference:
    - PnLCalib uses a separate HRNet model for line detection
    - 23 line classes covering penalty areas, goal areas, touchlines, etc.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import BasicBlock, Bottleneck, HighResolutionModule


class HRNetLineModel(nn.Module):
    """HRNetv2-w48 for soccer field line detection.

    This model predicts heatmaps for line classes. Each channel represents
    a line class, and the top 2 peaks in each channel are the endpoints.

    Architecture matches HRNetKeypointModel but with 24 output channels
    (23 line classes + 1 background).

    Attributes:
        num_line_classes: Number of line classes to detect (23 for PnLCalib).
        num_outputs: Total output channels (24 = 23 lines + 1 background).
    """

    # Line class definitions from PnLCalib (23 line classes)
    # Each channel contains both endpoints as top-2 peaks
    LINE_CLASSES = [
        "Big rect. left top",        # 0: penalty area left top line
        "Big rect. left side",       # 1: penalty area left side line
        "Big rect. left bottom",     # 2: penalty area left bottom line
        "Big rect. right top",       # 3: penalty area right top line
        "Big rect. right side",      # 4: penalty area right side line
        "Big rect. right bottom",    # 5: penalty area right bottom line
        "Goal left crossbar",        # 6: left goal crossbar
        "Goal left post left",       # 7: left goal left post
        "Goal left post right",      # 8: left goal right post
        "Goal right crossbar",       # 9: right goal crossbar
        "Goal right post left",      # 10: right goal left post
        "Goal right post right",     # 11: right goal right post
        "Middle line",               # 12: center line
        "Side line top",             # 13: top touchline
        "Side line left",            # 14: left goal line
        "Side line right",           # 15: right goal line
        "Side line bottom",          # 16: bottom touchline
        "Small rect. left top",      # 17: goal area left top line
        "Small rect. left side",     # 18: goal area left side line
        "Small rect. left bottom",   # 19: goal area left bottom line
        "Small rect. right top",     # 20: goal area right top line
        "Small rect. right side",    # 21: goal area right side line
        "Small rect. right bottom",  # 22: goal area right bottom line
    ]

    def __init__(
        self,
        num_line_classes: int = 23,
        config: dict[str, Any] | None = None,
    ):
        """Initialize HRNet line model.

        Args:
            num_line_classes: Number of line classes to detect (default 23 for PnLCalib).
            config: Optional configuration dictionary.

        Note:
            PnLCalib line model outputs 24 heatmaps = 23 line classes + 1 background.
            Each line class channel contains BOTH endpoints as the top-2 peaks.
        """
        super().__init__()
        self.num_line_classes = num_line_classes
        self.num_outputs = num_line_classes + 1  # 24 for PnLCalib (23 lines + background)
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
        # Note: PnLCalib line model uses SIGMOID (not Softmax like keypoint model!)
        # This allows each channel to independently predict line presence
        self.head = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(final_inp_channels, final_inp_channels, kernel_size=1),
                nn.BatchNorm2d(final_inp_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(final_inp_channels, self.num_outputs, kernel_size=1),
                nn.Sigmoid(),  # Line model uses Sigmoid, NOT Softmax!
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
            Heatmap tensor of shape (B, num_outputs, H/4, W/4).
            Can be reshaped to (B, num_line_classes, num_extremities, H/4, W/4).
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

        # Remove prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v

        # Load with partial matching
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}

        # Check for head layer mismatch
        head_keys = [k for k in pretrained_dict if k.startswith("head")]
        for k in head_keys:
            if pretrained_dict[k].shape != model_dict[k].shape:
                del pretrained_dict[k]

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)

    @classmethod
    def get_line_class_name(cls, line_idx: int) -> str:
        """Get the name of a line class by index.

        Args:
            line_idx: Line class index.

        Returns:
            Line class name.
        """
        if 0 <= line_idx < len(cls.LINE_CLASSES):
            return cls.LINE_CLASSES[line_idx]
        return f"Unknown line class {line_idx}"
