# -*- coding: utf-8 -*-
"""
Simplified WSSS Pipeline using Official `pytorch-grad-cam`:
- Aspect-Ratio Preserved Letterbox Input Canvas (512x512)
- Mixed-Precision (torch.amp.autocast) for Low VRAM Footprint
- Background Cutoff Filter + Class-Wise Otsu Thresholding
- Unpad-Aware Heatmap Extraction to True Original Resolution
- Dual Evaluation: Image-Level (mAP, F1) & Pixel-Level (mIoU) Metrics
"""

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.amp import autocast

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from convx.config import NUM_CLASSES, DEVICE, BATCH_SIZE
from convx.dataset_utils import (
    train_transform_pipeline,
    AlbumentationsDatasetWrapper,
    convx_collate_fn as train_collate_fn,
    seg_collate_fn_for_val
)
from convx.models import StandardBaseline


# =============================================================================
# 0. GRAD-CAM MODEL WRAPPER
# =============================================================================
class GradCAMModelWrapper(nn.Module):
    """
    Wraps the baseline model to guarantee that forward calls return only 
    a single 2D logits tensor (B, C), unwrapping any output tuples.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, tuple) else out


# =============================================================================
# 1. ASPECT-RATIO PRESERVING TRANSFORMS (512x512 Resolution)
# =============================================================================
val_transform_letterbox = A.Compose([
    A.LongestMaxSize(max_size=224),
    A.PadIfNeeded(
        min_height=224,
        min_width=224,
        border_mode=0,
        fill=(0, 0, 0),
        position="top_left"
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# =============================================================================
# 2. UNPADDING & RESIZING UTILITIES
# =============================================================================
def get_target_layer(model):
    """
    Identifies target convolutional layer for pytorch-grad-cam.
    """
    backbone = getattr(model, 'backbone', model)
    if hasattr(backbone, 'layer4'):
        return [backbone.layer4[-1]]
    else:
        convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        return [convs[-1]]


def extract_unpadded_cams(heatmaps, original_shapes, input_canvas_size=(224, 224)):
    """
    Crops out letterbox padding and resizes raw GradCAMs back to true image dimensions.
    """
    B, C, H_feat, W_feat = heatmaps.shape
    H_canvas, W_canvas = input_canvas_size
    upsampled_cams = []

    for i in range(B):
        H_orig, W_orig = original_shapes[i]

        scale = H_canvas / max(H_orig, W_orig)
        valid_h = int(round(H_orig * scale))
        valid_w = int(round(W_orig * scale))

        feat_valid_h = max(1, int(round(valid_h * (H_feat / H_canvas))))
        feat_valid_w = max(1, int(round(valid_w * (W_feat / W_canvas))))

        valid_feat = heatmaps[i:i+1, :, :feat_valid_h, :feat_valid_w]

        cam_resized = F.interpolate(
            valid_feat, size=(H_orig, W_orig), mode="bilinear", align_corners=True
        ).squeeze(0)

        upsampled_cams.append(cam_resized)

    return upsampled_cams


# =============================================================================
# 3. OTSU THRESHOLDING + BACKGROUND CUTOFF
# =============================================================================
def generate_otsu_mask_for_gt_classes(cams_single, gt_classes, H_orig, W_orig, bg_threshold=0.40):
    """
    Applies Otsu's thresholding conditioned on classifier confidence and background cutoff.
    """
    pred_mask = np.zeros((H_orig, W_orig), dtype=np.int64)
    max_activations = np.zeros((H_orig, W_orig), dtype=np.float32)

    for cls_id in gt_classes:
        if cls_id <= 0 or cls_id > cams_single.shape[0]:
            continue

        # Get confidence-weighted float CAM for class
        cam_c = cams_single[cls_id - 1].cpu().numpy()
        c_max = cam_c.max()

        # If class confidence/activation is below background cutoff, skip it entirely
        if c_max < bg_threshold:
            continue

        # Min-max scale ONLY valid regions above background noise
        c_min = cam_c.min()
        if c_max - c_min < 1e-5:
            continue

        cam_norm_float = (cam_c - c_min) / (c_max - c_min)
        cam_norm_uint8 = (cam_norm_float * 255.0).astype(np.uint8)

        # Otsu thresholding
        _, binary_map = cv2.threshold(cam_norm_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Pixel must pass BOTH Otsu threshold AND the absolute background cutoff
        valid_fg = (binary_map > 0) & (cam_c >= bg_threshold)

        update_mask = valid_fg & (cam_c > max_activations)
        pred_mask[update_mask] = cls_id
        max_activations[update_mask] = cam_c[update_mask]

    return pred_mask


# =============================================================================
# 4. METRIC COMPUTATION UTILITIES (IMAGE-LEVEL & PIXEL-LEVEL)
# =============================================================================
def compute_image_level_metrics(all_probs, all_targets, threshold=0.5):
    """
    Computes multi-label image-level classification performance (mAP, Precision, Recall, F1).
    """
    preds = (all_probs >= threshold).astype(int)
    
    tp = (preds * all_targets).sum(axis=0)
    fp = (preds * (1 - all_targets)).sum(axis=0)
    fn = ((1 - preds) * all_targets).sum(axis=0)
    
    prec_per_cls = np.where((tp + fp) > 0, tp / (tp + fp), np.nan)
    rec_per_cls = np.where((tp + fn) > 0, tp / (tp + fn), np.nan)
    f1_per_cls = np.where((prec_per_cls + rec_per_cls) > 0, 
                          2 * prec_per_cls * rec_per_cls / (prec_per_cls + rec_per_cls), 
                          np.nan)
    
    # Calculate Average Precision (AP) per foreground class
    ap_per_cls = []
    for c in range(all_targets.shape[1]):
        target_c = all_targets[:, c]
        prob_c = all_probs[:, c]
        if target_c.sum() == 0:
            ap_per_cls.append(np.nan)
            continue

        sort_idx = np.argsort(-prob_c)
        target_sorted = target_c[sort_idx]
        
        tp_cum = np.cumsum(target_sorted)
        fp_cum = np.cumsum(1 - target_sorted)
        
        recalls = tp_cum / target_c.sum()
        precisions = tp_cum / (tp_cum + fp_cum)
        
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))
        
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
            
        i_list = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i_list + 1] - mrec[i_list]) * mpre[i_list + 1])
        ap_per_cls.append(ap)
        
    return {
        "img_mAP": np.nanmean(ap_per_cls),
        "img_mean_precision": np.nanmean(prec_per_cls),
        "img_mean_recall": np.nanmean(rec_per_cls),
        "img_mean_f1": np.nanmean(f1_per_cls),
        "per_class_ap": np.array(ap_per_cls),
        "per_class_img_prec": prec_per_cls,
        "per_class_img_rec": rec_per_cls,
        "per_class_img_f1": f1_per_cls,
    }


def compute_confusion_matrix(pred_mask, gt_mask, num_classes=21):
    valid_mask = (gt_mask != 255) & (gt_mask >= 0) & (gt_mask < num_classes)
    if not np.any(valid_mask):
        return np.zeros((num_classes, num_classes), dtype=np.int64)

    return np.bincount(
        num_classes * gt_mask[valid_mask].astype(int) + pred_mask[valid_mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)


# =============================================================================
# 5. EVALUATION WITH GRAD-CAM & DUAL METRIC EXTRACTION
# =============================================================================
def evaluate_gradcam_otsu(model, val_loader, num_classes=21, device=DEVICE, bg_threshold=0.40):
    model.eval()
    total_confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    all_img_probs = []
    all_img_targets = []

    target_layers = get_target_layer(model)
    wrapped_model = GradCAMModelWrapper(model)
    num_fg_classes = num_classes - 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    use_amp = "cuda" in str(device)

    with GradCAM(model=wrapped_model, target_layers=target_layers) as cam:
        for imgs, masks in tqdm(val_loader, desc="🔍 Evaluating GradCAM + Gating + Otsu"):
            imgs = imgs.to(device)
            masks_np = [m.numpy() for m in masks]
            orig_shapes = [m.shape for m in masks_np]
            B, _, H_canvas, W_canvas = imgs.shape

            # -------------------------------------------------------------
            # 1. Get model class probabilities for confidence gating
            # -------------------------------------------------------------
            with torch.no_grad():
                logits = model(imgs)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.sigmoid(logits)  # Shape: (B, num_fg_classes)
            # -------------------------------------------------------------

            target_classes_batch = [
                [cls_id - 1 for cls_id in set(np.unique(m)) - {0, 255}]
                for m in masks_np
            ]

            # Collect Image-Level targets and probabilities
            for b in range(B):
                target_vec = np.zeros(num_fg_classes, dtype=np.float32)
                for c in target_classes_batch[b]:
                    if 0 <= c < num_fg_classes:
                        target_vec[c] = 1.0
                all_img_probs.append(probs[b].detach().cpu().numpy())
                all_img_targets.append(target_vec)

            active_classes = set().union(*target_classes_batch)
            heatmaps = torch.zeros((B, num_fg_classes, H_canvas, W_canvas), device=device)

            for c in active_classes:
                if c < 0 or c >= num_fg_classes:
                    continue

                targets = [ClassifierOutputTarget(c)] * B

                # Run GradCAM pass under Automatic Mixed Precision
                with autocast(device_type="cuda" if use_amp else "cpu", dtype=torch.float16, enabled=use_amp):
                    grayscale_cams = cam(input_tensor=imgs, targets=targets)

                # Weight raw CAMs by predicted confidence score
                for b in range(B):
                    score = probs[b, c].item()
                    cam_tensor = torch.from_numpy(grayscale_cams[b]).to(device)
                    heatmaps[b, c] = cam_tensor * score

            unpadded_cams = extract_unpadded_cams(heatmaps, orig_shapes, input_canvas_size=(H_canvas, W_canvas))

            for i, cam_single in enumerate(unpadded_cams):
                gt_mask = masks_np[i]
                gt_classes_present = set(np.unique(gt_mask)) - {0, 255}

                pred_mask = generate_otsu_mask_for_gt_classes(
                    cam_single, gt_classes_present,
                    H_orig=orig_shapes[i][0], W_orig=orig_shapes[i][1],
                    bg_threshold=bg_threshold
                )

                total_confusion_matrix += compute_confusion_matrix(pred_mask, gt_mask, num_classes=num_classes)

            del imgs, heatmaps, unpadded_cams
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Image-Level Multi-Label Classification Performance
    all_img_probs = np.array(all_img_probs)
    all_img_targets = np.array(all_img_targets)
    img_metrics = compute_image_level_metrics(all_img_probs, all_img_targets, threshold=0.5)

    # Pixel-Level Segmentation Performance
    tp = np.diag(total_confusion_matrix)
    fp = total_confusion_matrix.sum(axis=0) - tp
    fn = total_confusion_matrix.sum(axis=1) - tp

    col_sum = total_confusion_matrix.sum(axis=0)
    precision_per_class = np.where(col_sum > 0, tp / col_sum, np.nan)
    row_sum = total_confusion_matrix.sum(axis=1)
    recall_per_class = np.where(row_sum > 0, tp / row_sum, np.nan)

    iou_denom = tp + fp + fn
    iou_per_class = np.where(iou_denom > 0, tp / iou_denom, np.nan)

    total_pixels = total_confusion_matrix.sum()
    overall_accuracy = tp.sum() / total_pixels if total_pixels > 0 else 0.0

    return {
        # Image-level
        "img_mAP": img_metrics["img_mAP"],
        "img_mean_precision": img_metrics["img_mean_precision"],
        "img_mean_recall": img_metrics["img_mean_recall"],
        "img_mean_f1": img_metrics["img_mean_f1"],
        "per_class_img_ap": img_metrics["per_class_ap"],
        "per_class_img_f1": img_metrics["per_class_img_f1"],
        # Pixel-level
        "mIoU": np.nanmean(iou_per_class),
        "mean_precision": np.nanmean(precision_per_class),
        "mean_recall": np.nanmean(recall_per_class),
        "overall_accuracy": overall_accuracy,
        "per_class_iou": iou_per_class,
        "per_class_precision": precision_per_class,
        "per_class_recall": recall_per_class,
    }


def print_metrics_report(metrics_dict, class_names=None):
    print("\n" + "=" * 85)
    print("📊 DUAL EVALUATION REPORT (CLASSIFIER + GRAD-CAM SEGMENTATION)")
    print("=" * 85)
    print("🖼️  IMAGE-LEVEL CLASSIFICATION METRICS (Classifier Performance):")
    print(f"   • Mean Average Precision (mAP): {metrics_dict['img_mAP'] * 100:.2f}%")
    print(f"   • Mean Classification F1 Score: {metrics_dict['img_mean_f1'] * 100:.2f}%")
    print(f"   • Mean Precision @ 0.5:         {metrics_dict['img_mean_precision'] * 100:.2f}%")
    print(f"   • Mean Recall @ 0.5:            {metrics_dict['img_mean_recall'] * 100:.2f}%")
    print("-" * 85)
    print("📐 PIXEL-LEVEL SEGMENTATION METRICS (CAM + Otsu Localization):")
    print(f"   • Spatial Segmentation mIoU:    {metrics_dict['mIoU']:.4f}")
    print(f"   • Overall Pixel Accuracy:       {metrics_dict['overall_accuracy'] * 100:.2f}%")
    print(f"   • Mean Pixel Precision:         {metrics_dict['mean_precision'] * 100:.2f}%")
    print(f"   • Mean Pixel Recall:            {metrics_dict['mean_recall'] * 100:.2f}%")
    print("-" * 85)

    headers = f"{'ID':<3} | {'Class Name':<13} | {'Img AP':<8} | {'Img F1':<8} | {'Pix Prec':<9} | {'Pix Rec':<9} | {'Pix IoU':<8}"
    print(headers)
    print("-" * 85)

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

        p_prec_str = f"{p_prec:.4f}" if not np.isnan(p_prec) else "N/A"
        p_rec_str = f"{p_rec:.4f}" if not np.isnan(p_rec) else "N/A"
        p_iou_str = f"{p_iou:.4f}" if not np.isnan(p_iou) else "N/A"

        # Background class (idx 0) has no image-level classification target
        if idx == 0:
            img_ap_str, img_f1_str = "N/A", "N/A"
        else:
            img_ap = metrics_dict["per_class_img_ap"][idx - 1]
            img_f1 = metrics_dict["per_class_img_f1"][idx - 1]
            img_ap_str = f"{img_ap:.4f}" if not np.isnan(img_ap) else "N/A"
            img_f1_str = f"{img_f1:.4f}" if not np.isnan(img_f1) else "N/A"

        print(f"{idx:<3} | {c_name:<13} | {img_ap_str:<8} | {img_f1_str:<8} | {p_prec_str:<9} | {p_rec_str:<9} | {p_iou_str:<8}")

    print("=" * 85 + "\n")


# =============================================================================
# 6. MAIN PIPELINE EXECUTION
# =============================================================================
def train_classifier_pipeline():
    print(f"📡 [CLASSIFIER INITIALIZATION] Target Compute Unit: {DEVICE}")
    os.makedirs("models", exist_ok=True)
    checkpoint_path = os.path.join("models", "classifier_baseline_best.pth")

    VAL_EVAL_BATCH_SIZE = min(BATCH_SIZE, 2)

    base_val_set = datasets.VOCSegmentation(root="./data_voc", year="2012", image_set="val", download=False)
    val_set = AlbumentationsDatasetWrapper(base_val_set, transform=val_transform_letterbox)
    val_loader = DataLoader(
        val_set, batch_size=VAL_EVAL_BATCH_SIZE, shuffle=False, 
        collate_fn=seg_collate_fn_for_val, num_workers=2, pin_memory=True
    )

    model = StandardBaseline(num_classes=NUM_CLASSES).to(DEVICE)

    if os.path.exists(checkpoint_path):
        print(f"\n📦 Found existing trained model at '{checkpoint_path}'.")
        print("⏩ Running evaluation (512x512 FP16, BG Cutoff=0.40, Otsu)...\n")
        
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

        metrics_dict = evaluate_gradcam_otsu(
            model, val_loader, num_classes=NUM_CLASSES + 1, device=DEVICE, bg_threshold=0.40
        )
        print_metrics_report(metrics_dict)
        return

    base_train_set = datasets.SBDataset(root="./data_sbd", image_set="train_noval", mode="segmentation", download=False)
    train_set = AlbumentationsDatasetWrapper(base_train_set, transform=train_transform_pipeline)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=train_collate_fn, num_workers=2, pin_memory=True
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    num_epochs = 50

    print("🚀 Training starting...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch:02d}/{num_epochs}"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits[0] if isinstance(logits, tuple) else logits, labels)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), checkpoint_path)
    print("✅ Training complete! Running pytorch-grad-cam evaluation...")

    metrics_dict = evaluate_gradcam_otsu(
        model, val_loader, num_classes=NUM_CLASSES + 1, device=DEVICE, bg_threshold=0.40
    )
    print_metrics_report(metrics_dict)


if __name__ == '__main__':
    train_classifier_pipeline()