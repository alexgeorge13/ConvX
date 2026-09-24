# -*- coding: utf-8 -*-
"""Dataset utilities, custom image wrappers, and data loading pipelines for ConvX & ConvX+."""

import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from .config import NETWORK_INPUT_SIZE, NUM_CLASSES, PIXEL_SUPERVISION_COUNT


# =====================================================================
# 1. ALBUMENTATIONS TRANSFORM PIPELINES
# =====================================================================
train_transform_pipeline = A.Compose([
    A.LongestMaxSize(max_size=max(NETWORK_INPUT_SIZE), p=1.0),
    A.Affine(
        scale=(1.0, 1.0),
        translate_percent=(-0.1, 0.1),
        rotate=(-15, 15),
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=255,
        p=0.5
    ),
    A.PadIfNeeded(
        min_height=NETWORK_INPUT_SIZE[0],
        min_width=NETWORK_INPUT_SIZE[1],
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=255
    ),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform_pipeline = A.Compose([
    A.LongestMaxSize(max_size=max(NETWORK_INPUT_SIZE), p=1.0),
    A.PadIfNeeded(
        min_height=NETWORK_INPUT_SIZE[0],
        min_width=NETWORK_INPUT_SIZE[1],
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        fill_mask=255,
        p=1.0
    ),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


# =====================================================================
# 2. DATASET WRAPPERS
# =====================================================================
class AlbumentationsDatasetWrapper(Dataset):
    """Standard dataset wrapper for ConvX training and evaluation."""
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base_dataset[idx]
        img_np = np.array(img_pil)
        mask_np = np.array(mask_pil)

        augmented = self.transform(image=img_np, mask=mask_np)
        return augmented['image'], augmented['mask'].long()


class SemiSupervisedTrainDataset(Dataset):
    """Semi-supervised wrapper for ConvX+ training."""
    def __init__(self, base_dataset, supervised_indices, transform=None):
        self.base_dataset = base_dataset
        self.supervised_set = set(supervised_indices)
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_pil, mask_pil = self.base_dataset[idx]
        img_np = np.array(img_pil)
        mask_np = np.array(mask_pil)

        if self.transform is not None:
            augmented = self.transform(image=img_np, mask=mask_np)
            img_tensor = augmented['image']
            mask_tensor = augmented['mask'].long()
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask_np).long()

        # Generate image-level multi-hot label vector
        unique_classes = torch.unique(mask_tensor)
        label = torch.zeros(NUM_CLASSES)
        for c in unique_classes:
            if 0 < c <= NUM_CLASSES:
                label[c - 1] = 1.0

        # Boolean flag indicating pixel supervision availability
        is_supervised = torch.tensor(idx in self.supervised_set, dtype=torch.bool)

        return img_tensor, label, mask_tensor, is_supervised


# =====================================================================
# 3. BATCH COLLATE FUNCTIONS
# =====================================================================
def convx_collate_fn(batch):
    """For standard ConvX training: Returns images and image-level labels."""
    imgs, labels = [], []
    for img_tensor, mask_tensor in batch:
        imgs.append(img_tensor)
        unique_classes = torch.unique(mask_tensor)
        label = torch.zeros(NUM_CLASSES)
        for c in unique_classes:
            if 0 < c <= NUM_CLASSES:
                label[c - 1] = 1.0
        labels.append(label)
    return torch.stack(imgs), torch.stack(labels)


def train_collate_fn_plus(batch):
    """For ConvX+ training: Returns imgs, labels, masks, and is_supervised flags."""
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    masks = torch.stack([item[2] for item in batch])
    is_supervised = torch.stack([item[3] for item in batch])
    return imgs, labels, masks, is_supervised


# Alias for backward compatibility
convx_plus_collate_fn = train_collate_fn_plus


def seg_collate_fn_for_val(batch):
    """For Validation & Calibration: Returns stacked image and mask tensors."""
    imgs = torch.stack([item[0] for item in batch], dim=0)
    masks = torch.stack([item[1] for item in batch], dim=0)
    return imgs, masks


# =====================================================================
# 4. DATASET BALANCING INDEX SCANNER
# =====================================================================
def get_pixel_supervised_indices(dataset, target_count_per_class):
    """Analyzes data layout topology to isolate unique samples per target class."""
    img_classes_map = {}
    class_to_imgs = {c: [] for c in range(1, NUM_CLASSES + 1)}

    for idx in range(len(dataset)):
        _, target = dataset[idx]
        mask = np.array(target[0]) if isinstance(target, tuple) else np.array(target)
        unique_classes = np.unique(mask)
        valid_classes = [c for c in unique_classes if 1 <= c <= NUM_CLASSES]
        img_classes_map[idx] = valid_classes
        for c in valid_classes:
            class_to_imgs[c].append(idx)

    selected_indices = set()
    class_counts = {c: 0 for c in range(1, NUM_CLASSES + 1)}
    sorted_classes = sorted(range(1, NUM_CLASSES + 1), key=lambda c: len(class_to_imgs[c]))

    for c in sorted_classes:
        candidates = class_to_imgs[c]
        candidates = sorted(candidates, key=lambda x: len(img_classes_map[x]))

        for idx in candidates:
            if class_counts[c] >= target_count_per_class:
                break
            if idx not in selected_indices:
                can_use = True
                for cls in img_classes_map[idx]:
                    if class_counts[cls] >= target_count_per_class:
                        can_use = False
                        break
                if can_use:
                    selected_indices.add(idx)
                    for cls in img_classes_map[idx]:
                        class_counts[cls] += 1

    for c in range(1, NUM_CLASSES + 1):
        if class_counts[c] < target_count_per_class:
            for idx in class_to_imgs[c]:
                if class_counts[c] >= target_count_per_class:
                    break
                if idx not in selected_indices:
                    selected_indices.add(idx)
                    for cls in img_classes_map[idx]:
                        class_counts[cls] += 1

    return sorted(list(selected_indices))


# =====================================================================
# 5. ENCAPSULATED CACHE SYSTEM & EXPORTS
# =====================================================================
def get_supervised_indices(base_train_set, count_per_class=PIXEL_SUPERVISION_COUNT):
    cache_file = f"supervised_indices_{count_per_class}_per_class.json"

    # Load from cache if available and valid
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            indices = json.load(f)

        print(f"📥 Loaded {len(indices)} per-class supervised indices from cache.")
        return indices

    print(
        f"🔍 Sampling {count_per_class} pixel-supervised images PER CLASS across 20 VOC classes..."
    )

    class_counts = {c: 0 for c in range(1, 21)}  # VOC class IDs 1 to 20
    supervised_indices = set()

    for idx in tqdm(range(len(base_train_set)), desc="Sampling Per-Class Indices"):
        _, mask = base_train_set[idx]
        mask_np = np.array(mask)
        unique_classes = set(np.unique(mask_np)) - {0, 255}

        # Check if this image can satisfy any class still needing samples
        needed = False
        for c in unique_classes:
            if c in class_counts and class_counts[c] < count_per_class:
                needed = True
                break

        if needed:
            supervised_indices.add(idx)
            for c in unique_classes:
                if c in class_counts and class_counts[c] < count_per_class:
                    class_counts[c] += 1

        # Stop early if all 20 classes reached the target count
        if all(count >= count_per_class for count in class_counts.values()):
            print("✅ Successfully found enough samples for all 20 classes!")
            break

    indices = sorted(list(supervised_indices))

    # Save corrected cache
    with open(cache_file, "w") as f:
        json.dump(indices, f)

    print(
        f"🎯 Selected {len(indices)} total images to cover {count_per_class} samples/class."
    )
    return indices
