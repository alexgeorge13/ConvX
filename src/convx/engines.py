# -*- coding: utf-8 -*-
"""Training and evaluation runtime loops optimized for ConvX pipelines."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .config import NUM_CLASSES, DEVICE
from .losses import convx_loss, convx_plus_loss

def run_convx_epoch(model, dataloader, optimizer):
    """Executes a single epoch for standard image-level ConvX weakly supervised training."""
    model.train()
    running_loss = 0.0
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        scores, _ = model(imgs)
        loss = convx_loss(scores, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def run_convx_plus_epoch(model, dataloader, optimizer):
    """Executes a single epoch for Semi-Supervised ConvX+ with Dense Masks."""
    model.train()
    running_loss = 0.0
    for imgs, labels, masks, is_supervised in dataloader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        masks = masks.to(DEVICE)
        is_supervised = is_supervised.to(DEVICE)

        optimizer.zero_grad()
        scores, heatmaps = model(imgs)
        
        # Branch internal outputs to decouple loss calculations
        weak_indices = torch.where(~is_supervised)[0]
        img_loss = convx_loss(scores[weak_indices], labels[weak_indices]) if len(weak_indices) > 0 else torch.tensor(0.0, device=DEVICE)
        pix_loss = convx_plus_loss(heatmaps, masks, is_supervised)
        
        loss = img_loss + (5.0 * pix_loss)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def run_classifier_epoch(model, dataloader, optimizer):
    """Executes a single epoch for the Baseline Standard Classifier."""
    model.train()
    running_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

@torch.no_grad()
def evaluate_validation_loss(model, dataloader):
    model.eval()
    running_val_loss = 0.0
    for imgs, masks in dataloader:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        batch_size = masks.size(0)
        labels = torch.zeros(batch_size, NUM_CLASSES, device=DEVICE)
        for c in range(1, NUM_CLASSES + 1):
            labels[:, c - 1] = (masks == c).any(dim=(1, 2)).float()
        scores, _ = model(imgs)
        loss = convx_loss(scores, labels)
        running_val_loss += loss.item()
    return running_val_loss / len(dataloader)

@torch.no_grad()
def calibrate_and_evaluate_subset_miou(model, dataloader, num_anomaly_classes=20, device=DEVICE):
    model.eval()
    num_steps = 101
    candidates = torch.linspace(0.0, 4.0, steps=num_steps, device=device)

    total_inter = torch.zeros((num_anomaly_classes, num_steps), device=device)
    total_union = torch.zeros((num_anomaly_classes, num_steps), device=device)

    # Pass 1: Calibrate thresholds per class independently
    for images, masks in dataloader:
        images, masks = images.to(device), masks.to(device)
        B, H, W = masks.shape
        _, heatmaps = model(images)
        heatmaps_up = F.interpolate(heatmaps, size=(H, W), mode='bilinear', align_corners=False)

        for c in range(num_anomaly_classes):
            gt_mask_c = (masks == (c + 1))
            if gt_mask_c.sum() == 0:
                continue
            hm_c = heatmaps_up[:, c, :, :]
            for step_idx, t in enumerate(candidates):
                pred_mask_t = hm_c > t
                total_inter[c, step_idx] += (pred_mask_t & gt_mask_c).sum()
                total_union[c, step_idx] += (pred_mask_t | gt_mask_c).sum()

    best_thresholds = torch.zeros(num_anomaly_classes, device=device)
    for c in range(num_anomaly_classes):
        if total_union[c].max() > 0:
            ious = torch.zeros(num_steps, device=device)
            active = total_union[c] > 0
            ious[active] = total_inter[c, active] / total_union[c, active]
            best_thresholds[c] = candidates[torch.argmax(ious).item()]
        else:
            best_thresholds[c] = 1.2

    # Pass 2: Joint Evaluation via ARGMAX multi-class projection
    total_intersections = np.zeros(num_anomaly_classes + 1)
    total_unions = np.zeros(num_anomaly_classes + 1)

    for imgs, masks in dataloader:
        imgs = imgs.to(device)
        _, heatmaps = model(imgs)
        heatmaps_up = F.interpolate(heatmaps, size=(masks.shape[1], masks.shape[2]), mode='bilinear', align_corners=False)
        B, C, H, W = heatmaps_up.shape

        pred_masks = torch.zeros((B, H, W), dtype=torch.long, device=device)
        max_vals = torch.zeros((B, H, W), dtype=heatmaps_up.dtype, device=device)

        for c in range(num_anomaly_classes):
            cls_idx = c + 1
            t = best_thresholds[c]
            hm_c = heatmaps_up[:, c, :, :]
            active = (hm_c > t) & (hm_c > max_vals)
            pred_masks[active] = cls_idx
            max_vals[active] = hm_c[active]

        preds = pred_masks.cpu().numpy()
        masks_np = masks.numpy()
        valid_mask = (masks_np != 255)

        for c in range(num_anomaly_classes + 1):
            inter = np.logical_and(np.logical_and(preds == c, masks_np == c), valid_mask).sum()
            union = np.logical_and(np.logical_or(preds == c, masks_np == c), valid_mask).sum()
            total_intersections[c] += inter
            total_unions[c] += union

    ious = [total_intersections[c] / total_unions[c] for c in range(num_anomaly_classes + 1) if total_unions[c] > 0]
    return np.mean(ious) if ious else 0.0, best_thresholds