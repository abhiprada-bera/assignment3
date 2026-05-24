"""
utils.py — Masking Helpers, Label Smoothing Loss, and Noam Scheduler
DA6401 Assignment 3: "Attention Is All You Need"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # src_mask: [batch, 1, 1, src_len]
    src_mask = (src == pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    # tgt_pad_mask: [batch, 1, 1, tgt_len]
    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    
    tgt_len = tgt.size(1)
    # subsequent_mask: [1, tgt_len, tgt_len]
    subsequent_mask = torch.triu(torch.ones((1, tgt_len, tgt_len), device=tgt.device), diagonal=1).bool()
    
    # tgt_mask: [batch, 1, tgt_len, tgt_len]
    tgt_mask = tgt_pad_mask | subsequent_mask.unsqueeze(1)
    return tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        return F.cross_entropy(logits, target, label_smoothing=self.smoothing, ignore_index=self.pad_idx)


# ══════════════════════════════════════════════════════════════════════
#  NOAM SCHEDULER
# ══════════════════════════════════════════════════════════════════════

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Applies a warm-up phase where LR increases linearly, followed by
    a decay phase where LR decreases proportional to the inverse square
    root of the step number.

    Args:
        optimizer (torch.optim.Optimizer): Wrapped optimizer.
        d_model          (int)  : Model dimensionality (embedding size).
        warmup_steps     (int)  : Number of warm-up steps before decay begins.
        last_epoch       (int)  : The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scaling factor for the current step.

        Returns:
            float: The scalar multiplier applied to the base learning rate.
        """
        step = max(1, self.last_epoch + 1)
        scale = (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        return scale

    def get_lr(self) -> list[float]:
        """
        Compute learning rates for every param group.

        Called internally by PyTorch's scheduler machinery each step.

        Returns:
            list[float]: New learning rate for each param group in the optimizer.
        """
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]
