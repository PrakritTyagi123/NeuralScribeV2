"""
NeuralScribe v2 model — Compact residual CNN with SE attention.

Architecture:
  Stem Conv → [ResBlock + SE] × 4 → Global Avg Pool → Dropout → Linear(num_classes)

Each residual block:
  Conv(3×3) → BN → ReLU → Conv(3×3) → BN → SE → DropPath → + skip → ReLU

Features:
  - Squeeze-and-Excitation channel attention
  - Stochastic depth (drop path) with linearly increasing rate
  - 1×1 projection shortcuts when channels change
  - Forward hooks support for explainability (intermediate activations)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple


class DropPath(nn.Module):
    """Stochastic depth — drop entire residual branch with probability `p`."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Conv2d(channels, mid, 1)
        self.fc2 = nn.Conv2d(mid, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale), inplace=True)
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock(nn.Module):
    """Two-conv residual block with SE attention and drop path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        se_reduction: int = 16,
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, se_reduction)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()

        # Projection shortcut
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.drop_path(out)
        out = F.relu(out + identity, inplace=True)
        return out


class NeuralScribeNet(nn.Module):
    """
    Main classification network for NeuralScribe v2.

    Args:
        num_classes: Number of output classes
        stem_channels: Channels after stem conv (default 32)
        block_channels: List of channel widths per residual block
        se_reduction: SE reduction ratio
        dropout: Dropout before classifier head
        drop_path_rate: Max stochastic depth rate (linearly distributed)
    """

    def __init__(
        self,
        num_classes: int = 118,
        stem_channels: int = 32,
        block_channels: Optional[List[int]] = None,
        se_reduction: int = 16,
        dropout: float = 0.4,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        if block_channels is None:
            block_channels = [64, 128, 256, 320]

        self.num_classes = num_classes

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        num_blocks = len(block_channels)
        dp_rates = [drop_path_rate * i / max(num_blocks - 1, 1) for i in range(num_blocks)]

        self.blocks = nn.ModuleList()
        in_ch = stem_channels
        for i, out_ch in enumerate(block_channels):
            stride = 2 if i > 0 else 1  # Downsample after first block
            self.blocks.append(
                ResidualBlock(in_ch, out_ch, stride=stride, se_reduction=se_reduction, drop_path_rate=dp_rates[i])
            )
            in_ch = out_ch

        # Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(block_channels[-1], num_classes)

        # For explainability — stores intermediate activations when enabled
        self._capture_activations = False
        self._activations: Dict[str, torch.Tensor] = {}

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def enable_activation_capture(self, enabled: bool = True):
        self._capture_activations = enabled
        if not enabled:
            self._activations.clear()

    def get_activations(self) -> Dict[str, torch.Tensor]:
        return dict(self._activations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._capture_activations:
            self._activations.clear()

        x = self.stem(x)
        if self._capture_activations:
            self._activations["stem"] = x.detach()

        for i, block in enumerate(self.blocks):
            x = block(x)
            if self._capture_activations:
                self._activations[f"block_{i}"] = x.detach()

        x = self.pool(x)
        x = x.flatten(1)
        if self._capture_activations:
            self._activations["pooled"] = x.detach()

        x = self.dropout(x)
        logits = self.classifier(x)
        if self._capture_activations:
            self._activations["logits"] = logits.detach()

        return logits

    def forward_with_intermediates(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that returns logits AND intermediate activations for explainability."""
        self.enable_activation_capture(True)
        logits = self.forward(x)
        activations = self.get_activations()
        self.enable_activation_capture(False)
        return logits, activations

    def get_probability_evolution(self, x: torch.Tensor) -> List[Dict]:
        """
        Run forward pass and project intermediate features through the classifier head
        to show how class probabilities evolve through the network.
        Uses adaptive average pooling + linear projection when channel dims differ.
        """
        self.eval()
        evolution = []

        with torch.no_grad():
            feat = self.stem(x)

            for i, block in enumerate(self.blocks):
                feat = block(feat)
                pooled = self.pool(feat).flatten(1)
                # Project pooled features to num_classes for visualization
                if pooled.shape[1] == self.classifier.in_features:
                    probs = F.softmax(self.classifier(pooled), dim=-1)
                else:
                    # Use a simple linear projection for mismatched dims
                    # Scale by ratio to keep magnitudes comparable
                    weight = self.classifier.weight  # (num_classes, in_features)
                    # Adaptive: average-pool the weight columns to match pooled dim
                    # or zero-pad the pooled features
                    in_f = self.classifier.in_features
                    cur_f = pooled.shape[1]
                    if cur_f < in_f:
                        # Zero-pad pooled to match classifier input
                        padded = F.pad(pooled, (0, in_f - cur_f))
                        probs = F.softmax(self.classifier(padded), dim=-1)
                    else:
                        # Truncate (unlikely but safe)
                        probs = F.softmax(self.classifier(pooled[:, :in_f]), dim=-1)
                evolution.append({
                    "layer": f"block_{i}",
                    "probabilities": probs.cpu().squeeze(0).numpy(),
                })

            # Final pass through head
            pooled = self.pool(feat).flatten(1)
            logits = self.classifier(self.dropout(pooled) if not self.training else pooled)
            probs = F.softmax(logits, dim=-1)
            evolution.append({
                "layer": "output",
                "probabilities": probs.cpu().squeeze(0).numpy(),
            })

        return evolution

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> Dict:
        return {
            "num_classes": self.num_classes,
            "total_params": self.count_parameters(),
            "blocks": len(self.blocks),
        }