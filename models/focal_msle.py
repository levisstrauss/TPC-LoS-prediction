"""
Focal Mean Squared Logarithmic Error Loss
Novel extension for TPC-LoS-prediction

Author: Zakaria Coulibaly
Date: November 2024

Motivation: Standard MSLE treats all predictions equally. However,
long-stay patients (>7 days) are harder to predict and more clinically
important. Focal MSLE adaptively up-weights hard-to-predict samples.

Inspired by: Focal Loss for Dense Object Detection (Lin et al., 2017)
Applied to regression task for the first time in LoS prediction.
"""

import torch
import torch.nn as nn


class FocalMSLE(nn.Module):
    """
    Focal Mean Squared Logarithmic Error

    Modifies MSLE to focus more on hard-to-predict samples by
    up-weighting larger errors.

    Args:
        gamma: Focusing parameter (γ). Higher values increase focus
               on hard samples. Default: 2.0

    Formula:
        L_focal = (MSLE)^(1 + γ/2)

    Where:
        MSLE = (log(y_pred + 1) - log(y_true + 1))^2
    """

    def __init__(self, gamma=2.0):
        super(FocalMSLE, self).__init__()
        self.gamma = gamma
        print(f"Focal MSLE initialized with gamma={gamma}")

    def forward(self, y_hat, y, mask, seq_length, sum_losses=False):
        """
        Args:
            y_hat: Predictions (batch_size, time_steps)
            y: True labels (batch_size, time_steps)
            mask: Mask for valid timesteps (batch_size, time_steps)
            seq_length: Length of each sequence (batch_size,)
            sum_losses: Whether to sum or average losses

        Returns:
            loss: Scalar focal MSLE loss
        """
        # Compute standard MSLE
        log_y_hat = torch.log(y_hat + 1)
        log_y = torch.log(y + 1)
        msle = (log_y_hat - log_y) ** 2

        # Focal weighting: samples with higher error get more weight
        # Add small epsilon for numerical stability
        focal_weight = (msle + 1e-8) ** (self.gamma / 2)

        # Apply focal weighting and mask
        loss = msle * focal_weight * mask

        # Sum over time dimension
        loss = torch.sum(loss, dim=1)

        # Normalize by sequence length if needed
        if not sum_losses:
            loss = loss / seq_length.clamp(min=1)

        # Return mean over batch
        return loss.mean()