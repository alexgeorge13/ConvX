# -*- coding: utf-8 -*-
"""Evaluation metrics and per-class threshold calibration supporting padded / aspect-ratio inputs."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import os
from .config import DEVICE


def calibrate_per_class_thresholds(
    model, calib_loader, num_classes=20, topk_ratio=0.10, device=DEVICE
):
    """
    Calibrates individual, per-class optimal thresholds (tau_c) strictly using 
    image-level presence tags by maximizing each class's image-level F1-score 
    on Top-K% spatial heatmap activations.
    """
    model.eval()
    all_img_scores = []
    all_img_gts = []

    # Step 1: Collect Top-K% spatial scores and image-level ground truth tags
    with torch.no_grad():
        for imgs, targets in tqdm(
            calib_loader, desc=f"🎯 Calibrating Per-Class Thresholds (Top-{int(topk_ratio*100)}% Pooling)"
        ):
            imgs = imgs.to(device)
            out = model(imgs)
            scores, heatmaps = out if isinstance(out, tuple) else (None, out)

            # --- TOP-K% SPATIAL POOLING ---
            B, C, H, W = heatmaps.shape
            flat_heatmaps = heatmaps.view(B, C, -1)
            k = max(1, int(topk_ratio * flat_heatmaps.shape[-1]))
            topk_vals, _ = torch.topk(flat_heatmaps, k=k, dim=-1)
            topk_scores = topk_vals.mean(dim=-1).cpu().numpy()  # Shape: (B, C)

            all_img_scores.append(topk_scores)

            # Derive binary image-level presence tags
            targets_np = targets.numpy() if isinstance(targets, torch.Tensor) else targets
            if targets_np.ndim == 3:  # (B, H, W) Segmentation masks
                gts = np.stack([
                    [(mask == c).any() for c in range(1, num_classes + 1)]
                    for mask in targets_np
                ], axis=0).astype(np.float32)
            else:  # (B, C) Multi-hot class labels
                gts = targets_np.astype(np.float32)

            all_img_gts.append(gts)

    all_img_scores = np.concatenate(all_img_scores, axis=0)  # Shape: (N, C)
    all_img_gts = np.concatenate(all_img_gts, axis=0)        # Shape: (N, C)

    optimal_thresholds = np.zeros(num_classes, dtype=np.float32)

    # Step 2: Optimize tau_c independently per class (0..C-1) on Top-K% scores
    for c in range(num_classes):
        scores_c = all_img_scores[:, c]
        gts_c = all_img_gts[:, c]

        # Candidate thresholds spanning min to max Top-K% activation
        candidates = np.linspace(scores_c.min(), scores_c.max(), 100)
        best_f1 = -1.0
        best_thresh = 0.0

        for thresh in candidates:
            preds_c = (scores_c >= thresh).astype(np.float32)

            tp = np.sum((preds_c == 1) & (gts_c == 1))
            fp = np.sum((preds_c == 1) & (gts_c == 0))
            fn = np.sum((preds_c == 0) & (gts_c == 1))

            denom = 2 * tp + fp + fn
            f1 = (2 * tp) / denom if denom > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh

        optimal_thresholds[c] = best_thresh

    return torch.tensor(optimal_thresholds, dtype=torch.float32, device=device)


# Alias for backwards compatibility with train_convx.py
calibrate_roc_thresholds = calibrate_per_class_thresholds


def compute_average_precision(scores, targets):
    """
    Computes Average Precision (AP) for a single class using area under PR curve.
    - scores: 1D np.array of image-level scalar predictions (Top-K% heatmap scores)
    - targets: 1D np.array of binary ground truth presence (0 or 1)
    """
    if np.sum(targets) == 0:
        return np.nan

    sort_idx = np.argsort(-scores)
    scores_sorted = scores[sort_idx]
    targets_sorted = targets[sort_idx]

    tp = np.cumsum(targets_sorted == 1)
    fp = np.cumsum(targets_sorted == 0)

    recalls = tp / np.sum(targets == 1)
    precisions = tp / (tp + fp)

    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_confusion_matrix(pred_mask, gt_mask, num_classes=21):
    """
    Computes 2D confusion matrix strictly ignoring padded areas and VOC boundary pixels (label 255).
    """
    valid_mask = (gt_mask >= 0) & (gt_mask < num_classes)

    if not np.any(valid_mask):
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    hist = np.bincount(
        num_classes * gt_mask[valid_mask].astype(int) + pred_mask[valid_mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    return hist


def evaluate_full_segmentation_metrics(
    model, val_loader, optimal_threshold, num_classes=21, topk_ratio=0.10, device=DEVICE
):
    """
    Evaluates raw heatmaps across full validation set using per-class thresholds
    for spatial segmentation, and computes image-level classification metrics 
    (mAP, Precision, Recall, F1) using calibrated thresholds on Top-K% heatmap scores.
    """
    model.eval()
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    num_fg_classes = num_classes - 1  # 20 object classes

    all_img_scores = []
    all_img_gts = []

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="🔍 Running Full Evaluation (Pixel + Image Level)"):
            imgs = imgs.to(device)
            masks_np = masks.numpy()
            B, H_gt, W_gt = masks_np.shape

            out = model(imgs)
            scores, heatmaps = out if isinstance(out, tuple) else (None, out)

            # --- IMAGE-LEVEL METRICS PREPARATION (TOP-K% POOLING) ---
            B_hm, C_hm, H_hm, W_hm = heatmaps.shape
            flat_heatmaps = heatmaps.view(B_hm, C_hm, -1)
            k = max(1, int(topk_ratio * flat_heatmaps.shape[-1]))
            topk_vals, _ = torch.topk(flat_heatmaps, k=k, dim=-1)
            img_scalar_scores = topk_vals.mean(dim=-1).cpu().numpy()
            
            img_gts = np.stack([
                [(mask == c).any() for c in range(1, num_classes)]
                for mask in masks_np
            ], axis=0).astype(np.float32)

            all_img_scores.append(img_scalar_scores)
            all_img_gts.append(img_gts)

            # --- SPATIAL PIXEL SEGMENTATION ---
            heatmaps_upsampled = F.interpolate(
                heatmaps, size=(H_gt, W_gt), mode="bilinear", align_corners=False
            )

            if isinstance(optimal_threshold, torch.Tensor):
                thresh_tensor = optimal_threshold.view(1, num_fg_classes, 1, 1).to(device)
            else:
                thresh_tensor = torch.tensor(optimal_threshold, device=device).view(1, num_fg_classes, 1, 1)

            class_responses = heatmaps_upsampled - thresh_tensor

            if scores is not None:
                probs = torch.sigmoid(scores).view(B, num_fg_classes, 1, 1)
                gated_mask = (probs > 0.20)
                class_responses = torch.where(gated_mask, class_responses, torch.tensor(-1e9, device=device))

            bg_response = torch.zeros((B, 1, H_gt, W_gt), device=device)
            augmented_responses = torch.cat([bg_response, class_responses], dim=1)

            pred_masks = torch.argmax(augmented_responses, dim=1).cpu().numpy()

            for p_mask, gt_mask in zip(pred_masks, masks_np):
                total_confusion_matrix += compute_confusion_matrix(p_mask, gt_mask, num_classes=num_classes)

    # =======================================================================
    # 🧮 1. SPATIAL SEGMENTATION METRICS COMPUTATION
    # =======================================================================
    tp = np.diag(total_confusion_matrix)
    fp = total_confusion_matrix.sum(axis=0) - tp
    fn = total_confusion_matrix.sum(axis=1) - tp

    col_sum = total_confusion_matrix.sum(axis=0)
    precision_per_class = np.where(col_sum > 0, tp / col_sum, np.nan)
    mean_precision = np.nanmean(precision_per_class)

    row_sum = total_confusion_matrix.sum(axis=1)
    recall_per_class = np.where(row_sum > 0, tp / row_sum, np.nan)
    mean_recall = np.nanmean(recall_per_class)

    f1_denom = 2 * tp + fp + fn
    f1_per_class = np.where(f1_denom > 0, (2 * tp) / f1_denom, np.nan)
    macro_f1 = np.nanmean(f1_per_class)

    iou_denom = tp + fp + fn
    iou_per_class = np.where(iou_denom > 0, tp / iou_denom, np.nan)
    miou = np.nanmean(iou_per_class)

    total_pixels = total_confusion_matrix.sum()
    overall_accuracy = tp.sum() / total_pixels if total_pixels > 0 else 0.0

    # =======================================================================
    # 🧮 2. IMAGE-LEVEL CLASSIFICATION METRICS COMPUTATION
    # =======================================================================
    all_img_scores = np.concatenate(all_img_scores, axis=0)  # Top-K% scores (N_val, 20)
    all_img_gts = np.concatenate(all_img_gts, axis=0)        # Ground truth presence (N_val, 20)

    # Extract calibrated threshold array (tau_c for classes 0..19)
    if isinstance(optimal_threshold, torch.Tensor):
        tau = optimal_threshold.cpu().numpy()
    else:
        tau = np.array(optimal_threshold)

    # A. Ranking Metric: Image AP & mAP (Using Top-K% scores)
    per_class_image_ap = np.full(num_classes, np.nan)
    for c in range(num_fg_classes):
        ap_c = compute_average_precision(all_img_scores[:, c], all_img_gts[:, c])
        per_class_image_ap[c + 1] = ap_c

    image_map = np.nanmean(per_class_image_ap[1:])

    # B. Decision Metrics: Compare Top-K% Scores directly against Calibrated Image Threshold tau_c
    img_preds = (all_img_scores >= tau[None, :]).astype(np.float32)

    img_tp = np.sum((img_preds == 1) & (all_img_gts == 1), axis=0)
    img_fp = np.sum((img_preds == 1) & (all_img_gts == 0), axis=0)
    img_fn = np.sum((img_preds == 0) & (all_img_gts == 1), axis=0)

    per_class_img_precision = np.full(num_classes, np.nan)
    per_class_img_recall = np.full(num_classes, np.nan)
    per_class_img_f1 = np.full(num_classes, np.nan)

    for c in range(num_fg_classes):
        p_denom = img_tp[c] + img_fp[c]
        r_denom = img_tp[c] + img_fn[c]
        f1_denom = 2 * img_tp[c] + img_fp[c] + img_fn[c]

        prec_c = img_tp[c] / p_denom if p_denom > 0 else 0.0
        rec_c = img_tp[c] / r_denom if r_denom > 0 else 0.0
        f1_c = (2 * img_tp[c]) / f1_denom if f1_denom > 0 else 0.0

        per_class_img_precision[c + 1] = prec_c
        per_class_img_recall[c + 1] = rec_c
        per_class_img_f1[c + 1] = f1_c

    macro_img_precision = np.nanmean(per_class_img_precision[1:])
    macro_img_recall = np.nanmean(per_class_img_recall[1:])
    macro_img_f1 = np.nanmean(per_class_img_f1[1:])

    return {
        # Spatial Segmentation Metrics
        "mIoU": miou,
        "macro_f1": macro_f1,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "overall_accuracy": overall_accuracy,
        "per_class_iou": iou_per_class,
        "per_class_f1": f1_per_class,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
        "confusion_matrix": total_confusion_matrix,
        # Image-Level Classification Metrics
        "image_mAP": image_map,
        "per_class_image_ap": per_class_image_ap,
        "macro_image_precision": macro_img_precision,
        "macro_image_recall": macro_img_recall,
        "macro_image_f1": macro_img_f1,
        "per_class_image_precision": per_class_img_precision,
        "per_class_image_recall": per_class_img_recall,
        "per_class_image_f1": per_class_img_f1
    }


def print_metrics_report(metrics_dict, class_names=None):
    """Prints clean, formatted summary table including pixel and image level metrics."""
    print("\n" + "=" * 115)
    print("📊 FULL VALIDATION EVALUATION REPORT (PIXEL SEGMENTATION & IMAGE CLASSIFICATION)")
    print("=" * 115)
    print(f"  • Overall Pixel Accuracy:          {metrics_dict['overall_accuracy'] * 100:.2f}%")
    print(f"  • Mean Pixel Precision:            {metrics_dict['mean_precision'] * 100:.2f}%")
    print(f"  • Mean Pixel Recall:               {metrics_dict['mean_recall'] * 100:.2f}%")
    print(f"  • Spatial Macro F1-Score:          {metrics_dict['macro_f1']:.4f}")
    print(f"  • Spatial Segmentation mIoU:       {metrics_dict['mIoU']:.4f}")
    print("-" * 115)
    print(f"  • Image-Level Classification mAP:  {metrics_dict['image_mAP'] * 100:.2f}%")
    print(f"  • Image-Level Macro Precision:     {metrics_dict['macro_image_precision'] * 100:.2f}%")
    print(f"  • Image-Level Macro Recall:        {metrics_dict['macro_image_recall'] * 100:.2f}%")
    print(f"  • Image-Level Macro F1-Score:      {metrics_dict['macro_image_f1']:.4f}")
    print("-" * 115)

    headers = (f"{'ID':<3} | {'Class Name':<14} | {'Pix Prec':<8} | {'Pix Rec':<8} | "
               f"{'Pix IoU':<8} | {'Img AP':<8} | {'Img Prec':<8} | {'Img Rec':<8} | {'Img F1':<8}")
    print(headers)
    print("-" * 115)

    if class_names is None:
        class_names = [
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]

    for idx in range(len(metrics_dict["per_class_iou"])):
        c_name = class_names[idx] if idx < len(class_names) else f"Class {idx}"

        p_prec = metrics_dict["per_class_precision"][idx]
        p_rec = metrics_dict["per_class_recall"][idx]
        p_iou = metrics_dict["per_class_iou"][idx]

        i_ap = metrics_dict["per_class_image_ap"][idx]
        i_prec = metrics_dict["per_class_image_precision"][idx]
        i_rec = metrics_dict["per_class_image_recall"][idx]
        i_f1 = metrics_dict["per_class_image_f1"][idx]

        p_prec_str = f"{p_prec:.4f}" if not np.isnan(p_prec) else "N/A"
        p_rec_str = f"{p_rec:.4f}" if not np.isnan(p_rec) else "N/A"
        p_iou_str = f"{p_iou:.4f}" if not np.isnan(p_iou) else "N/A"

        i_ap_str = f"{i_ap * 100:.2f}%" if not np.isnan(i_ap) else "N/A"
        i_prec_str = f"{i_prec * 100:.2f}%" if not np.isnan(i_prec) else "N/A"
        i_rec_str = f"{i_rec * 100:.2f}%" if not np.isnan(i_rec) else "N/A"
        i_f1_str = f"{i_f1:.4f}" if not np.isnan(i_f1) else "N/A"

        print(f"{idx:<3} | {c_name:<14} | {p_prec_str:<8} | {p_rec_str:<8} | "
              f"{p_iou_str:<8} | {i_ap_str:<8} | {i_prec_str:<8} | {i_rec_str:<8} | {i_f1_str:<8}")

    print("=" * 115 + "\n")


"""Visualization utilities for ConvX semantic segmentation predictions."""

def get_pascal_voc_palette():
    """Generates the official 21-class PASCAL VOC color palette."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= ((cid >> 0) & 1) << (7 - j)
            g |= ((cid >> 1) & 1) << (7 - j)
            b |= ((cid >> 2) & 1) << (7 - j)
            cid >>= 3
        palette[i] = [r, g, b]
    return palette


def colorize_mask(mask_np, palette):
    """Converts a 2D class label mask (H, W) to an RGB PIL Image using VOC palette."""
    mask_img = Image.fromarray(mask_np.astype(np.uint8), mode="P")
    mask_img.putpalette(palette.flatten())
    return mask_img.convert("RGB")


def visualize_segmentation_results(
    model, val_loader, optimal_threshold, num_samples=5, device="cuda", save_path="convx_predictions.png"
):
    """
    Plots side-by-side comparison figures for validation samples:
    [ Original Image | Ground Truth Mask | Predicted Mask | Prediction Overlay ]
    """
    model.eval()
    palette = get_pascal_voc_palette()
    num_fg_classes = optimal_threshold.shape[0] if isinstance(optimal_threshold, torch.Tensor) else len(optimal_threshold)

    # Format threshold tensor for broadcasting
    if isinstance(optimal_threshold, torch.Tensor):
        thresh_tensor = optimal_threshold.view(1, num_fg_classes, 1, 1).to(device)
    else:
        thresh_tensor = torch.tensor(optimal_threshold, device=device).view(1, num_fg_classes, 1, 1)

    collected_samples = 0
    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    # Class names for legend / title context
    class_names = [
        "bg", "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "table", "dog",
        "horse", "motorbike", "person", "plant", "sheep", "sofa", "train", "tv"
    ]

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs_dev = imgs.to(device)
            B, H_gt, W_gt = masks.shape

            out = model(imgs_dev)
            scores, heatmaps = out if isinstance(out, tuple) else (None, out)

            # Interpolate low-res heatmaps to GT image dimensions
            heatmaps_upsampled = F.interpolate(
                heatmaps, size=(H_gt, W_gt), mode="bilinear", align_corners=False
            )

            # Apply per-class threshold response margins
            class_responses = heatmaps_upsampled - thresh_tensor

            if scores is not None:
                probs = torch.sigmoid(scores).view(B, num_fg_classes, 1, 1)
                class_responses = torch.where(probs > 0.20, class_responses, torch.tensor(-1e9, device=device))

            bg_response = torch.zeros((B, 1, H_gt, W_gt), device=device)
            augmented = torch.cat([bg_response, class_responses], dim=1)
            pred_masks = torch.argmax(augmented, dim=1).cpu().numpy()

            for b in range(B):
                if collected_samples >= num_samples:
                    break

                # 1. Denormalize Image for display (Assumes standard ImageNet normalization)
                img_tensor = imgs[b].cpu().numpy().transpose(1, 2, 0)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                orig_img = np.clip(std * img_tensor + mean, 0, 1)

                # 2. GT Mask and Pred Mask
                gt_mask = masks[b].numpy()
                gt_mask_clean = np.where(gt_mask == 255, 0, gt_mask)  # Hide 255 border pixels in plot
                pred_mask = pred_masks[b]

                gt_rgb = np.array(colorize_mask(gt_mask_clean, palette)) / 255.0
                pred_rgb = np.array(colorize_mask(pred_mask, palette)) / 255.0

                # 3. Blend Prediction Overlay
                overlay = (0.5 * orig_img + 0.5 * pred_rgb)

                # Identify unique classes in prediction for titling
                present_classes = np.unique(pred_mask)
                detected_names = [class_names[c] for c in present_classes if c > 0]
                detected_str = ", ".join(detected_names) if detected_names else "Background Only"

                # Plot Row
                row = collected_samples
                axes[row, 0].imshow(orig_img)
                axes[row, 0].set_title(f"Sample {row + 1}: Input Image")
                axes[row, 0].axis("off")

                axes[row, 1].imshow(gt_rgb)
                axes[row, 1].set_title("Ground Truth Mask")
                axes[row, 1].axis("off")

                axes[row, 2].imshow(pred_rgb)
                axes[row, 2].set_title(f"Predicted Mask\n({detected_str})")
                axes[row, 2].axis("off")

                axes[row, 3].imshow(overlay)
                axes[row, 3].set_title("Overlay (50% Alpha)")
                axes[row, 3].axis("off")

                collected_samples += 1

            if collected_samples >= num_samples:
                break

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n🖼️ Visual comparison overlay saved successfully to '{save_path}'!")
    plt.close()    