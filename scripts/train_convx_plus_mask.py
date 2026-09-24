# -*- coding: utf-8 -*-
"""Training Script for ConvX+ (Mask) for Semi-Supervised Image & Dense Pixel Framework"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from convx.config import NUM_CLASSES, DEVICE, BATCH_SIZE, PIXEL_SUPERVISION_COUNT
from convx.dataset_utils import (
    train_transform_pipeline,
    val_transform_pipeline,
    get_supervised_indices,
    SemiSupervisedTrainDataset,
    AlbumentationsDatasetWrapper,
    train_collate_fn_plus,
    seg_collate_fn_for_val
)
from convx.models import ConvX
from convx.losses import convx_loss, convx_plus_loss
from convx.engines import evaluate_validation_loss, calibrate_and_evaluate_subset_miou
from convx.metrics import (
    calibrate_roc_thresholds,
    evaluate_full_segmentation_metrics,
    print_metrics_report,
    visualize_segmentation_results
)


def train_convx_plus_pipeline():
    print(f"📡 [CONVX+ (MASK) INITIALIZATION] Target Compute Unit: {DEVICE}")
    checkpoint_path = os.path.join(
        "models",
        f"convx_plus_mask_{PIXEL_SUPERVISION_COUNT}_per_class_best.pth"
    )

    os.makedirs("models", exist_ok=True)

    # 1. Dataset Instantiations
    base_train_set = datasets.SBDataset(root="./data_sbd", image_set="train_noval", mode="segmentation", download=False)
    base_val_set = datasets.VOCSegmentation(root="./data_voc", year="2012", image_set="val", download=False)

    # 🌟 FETCH INDICES HERE (Only runs once in the main process)
    train_supervised_indices = get_supervised_indices(base_train_set, count_per_class=PIXEL_SUPERVISION_COUNT)

    train_set_plus = SemiSupervisedTrainDataset(
        base_train_set, train_supervised_indices, transform=train_transform_pipeline
    )
    val_set = AlbumentationsDatasetWrapper(base_val_set, transform=val_transform_pipeline)

    train_loader = DataLoader(
        train_set_plus, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=train_collate_fn_plus, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=seg_collate_fn_for_val, num_workers=2, pin_memory=True
    )

    # 2. Calibration Loader for fast in-loop validation (200 images)
    calib_subset = Subset(val_set, list(range(min(200, len(val_set)))))
    calib_loader = DataLoader(
        calib_subset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=seg_collate_fn_for_val
    )

    # 3. Model Setup
    model = ConvX(num_classes=NUM_CLASSES).to(DEVICE)

    # =========================================================================
    # 🛑 CHECKPOINT CHECK: SKIP TRAINING IF CHECKPOINT EXISTS
    # =========================================================================
    if os.path.exists(checkpoint_path):
        print(f"\n📦 Found existing ConvX+ (Mask) checkpoint '{checkpoint_path}'. Skipping training!")
    else:
        print("\n⚙️ No existing ConvX+ (Mask) checkpoint found. Starting semi-supervised training loop...")

        # Differential Learning Rates (Protect Pretrained Backbone)
        backbone_params = [p for name, p in model.named_parameters() if "stage" in name]
        head_params = [p for name, p in model.named_parameters() if "stage" not in name]

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5},
            {'params': head_params, 'lr': 1e-4}
        ], weight_decay=1e-2)

        epochs = 50
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        best_miou = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"ConvX+ (Mask) Epoch {epoch:02d}/{epochs}")

            for imgs, labels, masks, is_supervised in pbar:
                imgs = imgs.to(DEVICE)
                labels = labels.to(DEVICE)
                masks = masks.to(DEVICE)
                is_supervised = is_supervised.to(DEVICE)

                optimizer.zero_grad()
                scores, heatmaps = model(imgs)

                img_loss = convx_loss(scores, labels)

                # Pixel-level loss for supervised subset
                pix_loss = convx_plus_loss(heatmaps, masks, is_supervised)

                loss = img_loss + (5.0 * pix_loss)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}",
                    "LR": f"{scheduler.get_last_lr()[0]:.1e}"
                })

            scheduler.step()

            train_loss = running_loss / len(train_loader)
            val_loss = evaluate_validation_loss(model, val_loader)
            val_miou, _ = calibrate_and_evaluate_subset_miou(
                model, calib_loader, num_anomaly_classes=NUM_CLASSES, device=DEVICE
            )

            print(f"📊 [SUMMARY] Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Fast mIoU: {val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save({'model': model.state_dict(), 'best_miou': best_miou}, checkpoint_path)
                print(f"💾 Checkpoint stored with updated semi-supervised macro mIoU: {best_miou:.4f}")

    # =========================================================================
    # 📊 FULL VALIDATION EVALUATION & THRESHOLD PERSISTENCE
    # =========================================================================
    print(f"\n🔍 Loading ConvX+ (Mask) model from '{checkpoint_path}' for threshold calibration and evaluation...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    # Safely extract model state and existing thresholds if present
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimal_thresh = checkpoint.get("thresholds", None)
    else:
        model.load_state_dict(checkpoint)
        optimal_thresh = None

    model.eval()

    # Calibrate and append thresholds to the checkpoint if not already saved
    if optimal_thresh is None:
        print("🎯 Calibrating per-class decision thresholds on calibration subset...")
        optimal_thresh = calibrate_roc_thresholds(
            model, calib_loader, num_classes=NUM_CLASSES, device=DEVICE
        )

        # Save dictionary containing BOTH weights and thresholds
        torch.save({
            "model": model.state_dict(),
            "thresholds": optimal_thresh
        }, checkpoint_path)
        print(f"💾 Checkpoint successfully updated with calibrated thresholds at: '{checkpoint_path}'")
    else:
        print("⚡ Loaded pre-calibrated thresholds directly from checkpoint!")

    mean_thresh = float(optimal_thresh.mean()) if hasattr(optimal_thresh, "mean") else float(optimal_thresh)
    print(f"✅ Active Thresholds across {NUM_CLASSES} classes (Mean Threshold: {mean_thresh:.4f})")

    # Step 2: Full validation metric evaluation
    metrics = evaluate_full_segmentation_metrics(
        model,
        val_loader=val_loader,
        optimal_threshold=optimal_thresh,
        num_classes=NUM_CLASSES + 1,
        device=DEVICE
    )

    # Step 3: Print full metrics report
    print_metrics_report(metrics)

    os.makedirs("plots", exist_ok=True)

    # Step 4: Generate visual overlays for qualitative analysis
    visualize_segmentation_results(
        model=model,
        val_loader=val_loader,
        optimal_threshold=optimal_thresh,
        num_samples=5,
        device=DEVICE,
        save_path=os.path.join("plots",f"convx_plus_mask_{PIXEL_SUPERVISION_COUNT}_per_class_predictions.png")
    )


if __name__ == '__main__':
    train_convx_plus_pipeline()