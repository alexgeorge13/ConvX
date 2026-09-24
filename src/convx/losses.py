# -*- coding: utf-8 -*-
"""Loss functions for ConvX and ConvX+ frameworks"""
import torch
import torch.nn.functional as F

def convx_loss(scores, labels):
    """
    Stabilized multi-class ConvX loss function.
    
    Args:
        scores (Tensor): Shape [Batch, Classes], non-negative heatmap scores.
        labels (Tensor): Shape [Batch, Classes], multi-hot binary targets.
    """
    scores = torch.clamp(scores, min=1e-6, max=15.0)
    
    # Term 1: Background suppression (minimize activation for labels == 0)
    term1 = (1.0 - labels) * scores
    
    # Term 2: Anomaly activation (maximize activation for labels == 1)
    term2 = -labels * torch.log(-torch.expm1(-scores) + 1e-7)
    
    return torch.mean(term1 + term2)

def convx_plus_loss(heatmaps, gt_masks, is_supervised_batch):
    """
    Symmetric Vectorized ConvX+ Dense WSSS Loss.
    Computes spatial losses uniformly across all channels, then decouples
    the final aggregation to prevent absent classes from diluting the gradients.
    """
    B, C, H, W = heatmaps.shape

    # Ensure heatmaps match target mask resolution dynamically
    if (H, W) != gt_masks.shape[1:]:
        heatmaps = F.interpolate(heatmaps, size=gt_masks.shape[1:], mode='bilinear', align_corners=False)

    # Filter for strictly supervised images in the batch
    supervised_indices = torch.where(is_supervised_batch)[0]
    if len(supervised_indices) == 0:
        return torch.tensor(0.0, device=heatmaps.device, requires_grad=True)

    # Slice out only the supervised slices to save memory and compute
    heatmaps = heatmaps[supervised_indices]
    gt_masks = gt_masks[supervised_indices]  # Shape: (B_sup, H, W)

    # 1. Create a global valid pixel mask (ignores PASCAL VOC void boundary 255)
    mask_valid = (gt_masks != 255).float().unsqueeze(1)

    # 2. Convert the MxN ground-truth map into a One-Hot Channel Tensor via Broadcasting
    channel_ids = torch.arange(1, C + 1, device=heatmaps.device).view(1, C, 1, 1)
    Y = (gt_masks.unsqueeze(1) == channel_ids).float()

    # Pre-clamp continuous activations globally to prevent NaN gradients in log/expm1
    a_j = torch.clamp(heatmaps, min=1e-5)

    # ==========================================================
    # --- TERM 1: NORMAL (Background Suppression for all channels) ---
    # ==========================================================
    normal_mask = (1.0 - Y) * mask_valid
    normal_counts = normal_mask.sum(dim=(2, 3)).clamp(min=1e-7)
    normal_loss = (a_j * normal_mask).sum(dim=(2, 3)) / normal_counts  # Shape: (B_sup, C)

    # ==========================================================
    # --- TERM 2: ANOMALY (Dense Object Activation for all channels) ---
    # ==========================================================
    anomaly_mask = Y * mask_valid
    anomaly_counts = anomaly_mask.sum(dim=(2, 3)).clamp(min=1e-7)
    pixel_wise_anomaly_loss = -torch.log(-torch.expm1(-a_j) + 1e-7)
    anomaly_loss = (pixel_wise_anomaly_loss * anomaly_mask).sum(dim=(2, 3)) / anomaly_counts  # Shape: (B_sup, C)

    # Total unweighted spatial loss per channel
    channel_loss = normal_loss + anomaly_loss  # Shape: (B_sup, C)

    # ==========================================================
    # --- DECOUPLED AGGREGATION (Preventing Gradient Dilution) ---
    # ==========================================================
    # Identify which channels are present vs absent per image
    is_present = (Y.sum(dim=(2, 3)) > 0).float()  # Shape: (B_sup, C)
    is_absent = 1.0 - is_present

    # Pool the present classes independently to preserve their gradient strength
    present_loss = (channel_loss * is_present).sum() / is_present.sum().clamp(min=1)

    # Pool the absent classes independently
    absent_loss = (channel_loss * is_absent).sum() / is_absent.sum().clamp(min=1)

    # Combine both the losses
    total_batch_loss = present_loss + absent_loss

    return total_batch_loss