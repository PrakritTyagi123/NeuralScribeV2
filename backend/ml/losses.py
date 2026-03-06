"""
Loss functions for NeuralScribe v2.
Combined loss: label-smoothed cross entropy + optional focal component.
Also includes mixup/cutmix utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        # NLL for true class
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        # Uniform distribution
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll + self.smoothing * smooth_loss
        return loss.mean()


class FocalLoss(nn.Module):
    """Focal loss to focus on hard examples."""

    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal.mean()
        elif self.reduction == "sum":
            return focal.sum()
        return focal


class CombinedLoss(nn.Module):
    """
    Combined loss: label-smoothed CE + weighted focal loss.

    total = (1 - focal_weight) * LS_CE + focal_weight * Focal
    """

    def __init__(
        self,
        label_smoothing: float = 0.08,
        focal_gamma: float = 1.5,
        focal_weight: float = 0.3,
    ):
        super().__init__()
        self.ls_ce = LabelSmoothingCrossEntropy(label_smoothing)
        self.focal = FocalLoss(focal_gamma)
        self.focal_weight = focal_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss_ce = self.ls_ce(logits, targets)
        if self.focal_weight <= 0:
            return loss_ce
        loss_focal = self.focal(logits, targets)
        return (1 - self.focal_weight) * loss_ce + self.focal_weight * loss_focal


# ── Mixup utilities ──

def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply mixup augmentation.
    Returns: mixed_x, y_a, y_b, lam
    """
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha).sample().item()
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    logits: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute loss for mixup: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)."""
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)