# =====================================================================
# 📊 SPRINGER PROCEEDINGS: VERTICAL MACRO-COLUMN VISUALIZATION
# File: visual_comparison.py
# =====================================================================
import gc
import os
import sys


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# PyTorch Grad-CAM Imports
try:
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    raise ImportError(
        "pytorch_grad_cam library is required. Install via: pip install grad-cam"
    )

# Local Project Imports
from convx.config import DEVICE, NETWORK_INPUT_SIZE, NUM_CLASSES, VOC_CLASSES
from convx.models import ConvX, StandardBaseline

# =====================================================================
# ⚙️ CONFIGURATION & PATHS
# =====================================================================
MODELS_DIR = "models"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

CKPT_PATHS = {
    "baseline": os.path.join(MODELS_DIR, "classifier_baseline_best.pth"),
    "convx": os.path.join(MODELS_DIR, "convx_standard_best.pth"),
    "convx_star": os.path.join(MODELS_DIR, "convx_star_5_per_class_best.pth"),
    "convx_plus": os.path.join(MODELS_DIR, "convx_plus_5_per_class_best.pth"),
}

DATA_ROOT = "./data_voc"
OUTPUT_PDF_PATH = os.path.join(PLOTS_DIR, "prediction_visualisation.pdf")


# =====================================================================
# 📁 DIRECT FILE-BASED DATASET (NO UTILITIES, DIRECT RESIZING)
# =====================================================================
class DirectVOCDataset(Dataset):
    """Loads raw VOC images and masks directly from files and resizes them directly,
    ignoring original aspect ratios and padding.
    """

    def __init__(self, data_root, image_set="val", target_size=(224, 224)):
        if isinstance(target_size, int):
            self.target_h = self.target_w = target_size
        else:
            self.target_h, self.target_w = target_size

        voc_root = os.path.join(data_root, "VOCdevkit", "VOC2012")
        split_file = os.path.join(voc_root, "ImageSets", "Segmentation", f"{image_set}.txt")

        if not os.path.exists(split_file):
            raise FileNotFoundError(f"VOC split file not found at: {split_file}")

        with open(split_file, "r") as f:
            self.file_names = [line.strip() for line in f.readlines() if line.strip()]

        self.img_dir = os.path.join(voc_root, "JPEGImages")
        self.mask_dir = os.path.join(voc_root, "SegmentationClass")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        name = self.file_names[idx]
        img_path = os.path.join(self.img_dir, f"{name}.jpg")
        mask_path = os.path.join(self.mask_dir, f"{name}.png")

        # Direct file load
        img_bgr = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # PIL read for mask preserves palette integer class IDs correctly
        mask_pil = Image.open(mask_path)
        mask_np = np.array(mask_pil)

        # Force direct resizing (stretching, ignoring aspect ratio)
        img_resized = cv2.resize(img_rgb, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask_np, (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        # Create normalized model input tensor (ImageNet mean & std)
        img_float = img_resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_norm = (img_float - mean) / std

        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).long()

        # Returns normalized tensor, gt mask tensor, raw resized RGB img, raw resized gt mask
        return img_tensor, mask_tensor, img_resized, mask_resized


# =====================================================================
# 🛠️ HELPER CLASSES & FUNCTIONS
# =====================================================================
class ModelWrapper(nn.Module):
    """Wraps multi-output architectures so pytorch_grad_cam receives single tensor predictions."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


def generate_voc_colormap(n_classes=256):
    """Generates standard Pascal VOC color lookup table."""
    def bitget(val, idx):
        return (val >> idx) & 1

    cmap = np.zeros((n_classes, 3), dtype=np.uint8)
    for i in range(n_classes):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= bitget(c, 0) << (7 - j)
            g |= bitget(c, 1) << (7 - j)
            b |= bitget(c, 2) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


VOC_CMAP = generate_voc_colormap(256)


def load_model_checkpoint(path, model_class):
    """Safely loads full model objects or state dictionaries along with calibrated thresholds if present."""
    if not os.path.exists(path):
        print(f"⚠️ Checkpoint file '{path}' not found! Skipping.")
        return None, None

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    thresholds = None

    if isinstance(ckpt, dict):
        # Extract calibrated decision thresholds if saved in the checkpoint
        thresholds = ckpt.get("thresholds", None)
        
        if "model" in ckpt and isinstance(ckpt["model"], nn.Module):
            model = ckpt["model"]
        else:
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
            model = model_class()
            model.load_state_dict(state_dict)
    elif isinstance(ckpt, nn.Module):
        model = ckpt
    else:
        model = ckpt

    model.eval().to(DEVICE)
    return model, thresholds


def resolve_threshold(thresh_container, cls_idx, default_val=1.2):
    """Helper to retrieve class threshold regardless of whether it is stored as a Tensor, Dict, or Array."""
    if thresh_container is None:
        return default_val
    if isinstance(thresh_container, dict):
        return thresh_container.get(cls_idx, thresh_container.get(cls_idx - 1, default_val))
    try:
        # 1-indexed class ID (1..20) converted to 0-indexed tensor/array position (0..19)
        return float(thresh_container[cls_idx - 1])
    except (IndexError, TypeError, KeyError):
        return default_val


# VOC Class Name Lookup Mapping (1-indexed for masks)
STANDARD_VOC_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

if len(VOC_CLASSES) == 21:
    CLASS_NAME_MAP = {i: VOC_CLASSES[i].capitalize() for i in range(1, 21)}
elif len(VOC_CLASSES) == 20:
    CLASS_NAME_MAP = {i + 1: VOC_CLASSES[i].capitalize() for i in range(20)}
else:
    CLASS_NAME_MAP = {i + 1: STANDARD_VOC_NAMES[i].capitalize() for i in range(20)}


# =====================================================================
# 🎨 VISUALIZATION PIPELINE
# =====================================================================
@torch.no_grad()
def visualize_springer_vertical_layout(
    baseline_model, convx_model, convx_star_model, convx_plus_model, dataset,
    gcam, gcam_plus, eigen_cam, score_cam,
    convx_thresh=None, convx_star_thresh=None, convx_plus_thresh=None,
    sample_indices=[4, 159, 1166],
    gc_thresh=0.2, gcpp_thresh=0.2, ec_thresh=0.2, sc_thresh=0.2,
    output_pdf_path=OUTPUT_PDF_PATH
):
    """Predicts and plots images directly resized from disk, using calibrated thresholds if loaded."""
    num_samples = len(sample_indices)

    if baseline_model is not None:
        baseline_model.eval()
    if convx_model is not None:
        convx_model.eval()
    if convx_star_model is not None:
        convx_star_model.eval()
    if convx_plus_model is not None:
        convx_plus_model.eval()

    def apply_heatmap_overlay(map_data, bg_img_uint8):
        bg_float = bg_img_uint8.astype(np.float32) / 255.0
        denom = map_data.max() - map_data.min() + 1e-7
        norm = (map_data - map_data.min()) / denom
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(
            (bg_float * 255).astype(np.uint8), 0.45,
            cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), 0.55, 0
        )
        return blended / 255.0

    target_h, target_w = dataset.target_h, dataset.target_w

    # Grid Setup (8 rows)
    total_rows = 8
    total_cols = num_samples * 2 + (num_samples - 1)

    # 📏 Standard Springer LNCS page text block width = ~4.8 inches (122 mm)
    # Reduced height to 5.2 inches for a tighter vertical format
    fig = plt.figure(figsize=(4.8, 5.2))

    outer_gs = fig.add_gridspec(
        1, num_samples,
        wspace=0.12,  # 📈 Proportionally increased macro-column separation
        left=0.09, right=0.99, top=0.94, bottom=0.01
    )
    axes = np.empty((total_rows, total_cols), dtype=object)

    for i in range(num_samples):
        col_offset = i * 3
        inner_gs = outer_gs[i].subgridspec(
            total_rows, 2, 
            wspace=0.04,  # 🎯 Matched image-mask gap to row gap
            hspace=0.04   # 🎯 Row gap set precisely to 0.04
        )

        for r in range(total_rows):
            axes[r, col_offset] = fig.add_subplot(inner_gs[r, 0])
            axes[r, col_offset + 1] = fig.add_subplot(inner_gs[r, 1])

    plt.rcParams.update({
        'font.size': 5.0,
        'axes.titlesize': 5.0,
        'xtick.labelsize': 5.0,
        'ytick.labelsize': 5.0
    })

    print(f"📐 Processing direct resized samples ({target_w}x{target_h}) for indices: {sample_indices}...")

    for idx_in_grid, sample_idx in enumerate(sample_indices):
        col_offset = idx_in_grid * 3

        # Step 1: Load pre-resized image and ground truth directly from custom dataset
        img_tensor, _, img_display, gt_mask_np = dataset[sample_idx]
        img_input = img_tensor.unsqueeze(0).to(DEVICE)

        # Ground truth classes (1 to 20, ignoring background 0 and border 255)
        contained_classes = [c for c in np.unique(gt_mask_np) if 0 < c <= 20]
        class_names = [CLASS_NAME_MAP[c] for c in contained_classes if c in CLASS_NAME_MAP]
        class_str = "\n".join(class_names) if class_names else "Background"

        # Step 2: Predict and evaluate ONLY predictions that exist in the Ground Truth
        convx_std_heatmap = np.zeros((target_h, target_w))
        convx_star_heatmap = np.zeros((target_h, target_w))
        convx_plus_heatmap = np.zeros((target_h, target_w))

        convx_std_mask = np.zeros((target_h, target_w), dtype=np.int64)
        convx_star_mask = np.zeros((target_h, target_w), dtype=np.int64)
        convx_plus_mask = np.zeros((target_h, target_w), dtype=np.int64)

        with torch.no_grad():
            if convx_model is not None:
                out_std = convx_model(img_input)
                h_std = out_std[1] if isinstance(out_std, (tuple, list)) else out_std
                h_std_up = F.interpolate(h_std, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()

                max_val_std = np.zeros((target_h, target_w))
                for cls_idx in contained_classes:
                    c_0indexed = cls_idx - 1
                    t_std = resolve_threshold(convx_thresh, cls_idx, default_val=1.2)
                    act_std = (h_std_up[c_0indexed] > t_std) & (h_std_up[c_0indexed] > max_val_std)
                    convx_std_mask[act_std] = cls_idx
                    max_val_std[act_std] = h_std_up[c_0indexed][act_std]
                    convx_std_heatmap = np.maximum(convx_std_heatmap, h_std_up[c_0indexed])

            if convx_star_model is not None:
                out_star = convx_star_model(img_input)
                h_star = out_star[1] if isinstance(out_star, (tuple, list)) else out_star
                h_star_up = F.interpolate(h_star, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()

                max_val_star = np.zeros((target_h, target_w))
                for cls_idx in contained_classes:
                    c_0indexed = cls_idx - 1
                    t_star = resolve_threshold(convx_star_thresh, cls_idx, default_val=1.1)
                    act_star = (h_star_up[c_0indexed] > t_star) & (h_star_up[c_0indexed] > max_val_star)
                    convx_star_mask[act_star] = cls_idx
                    max_val_star[act_star] = h_star_up[c_0indexed][act_star]
                    convx_star_heatmap = np.maximum(convx_star_heatmap, h_star_up[c_0indexed])

            if convx_plus_model is not None:
                out_plus = convx_plus_model(img_input)
                h_plus = out_plus[1] if isinstance(out_plus, (tuple, list)) else out_plus
                h_plus_up = F.interpolate(h_plus, size=(target_h, target_w), mode='bilinear', align_corners=False).squeeze(0).cpu().numpy()

                max_val_plus = np.zeros((target_h, target_w))
                for cls_idx in contained_classes:
                    c_0indexed = cls_idx - 1
                    t_plus = resolve_threshold(convx_plus_thresh, cls_idx, default_val=1.0)
                    act_plus = (h_plus_up[c_0indexed] > t_plus) & (h_plus_up[c_0indexed] > max_val_plus)
                    convx_plus_mask[act_plus] = cls_idx
                    max_val_plus[act_plus] = h_plus_up[c_0indexed][act_plus]
                    convx_plus_heatmap = np.maximum(convx_plus_heatmap, h_plus_up[c_0indexed])

        cam_gc_heatmap, cam_gcpp_heatmap = np.zeros((target_h, target_w)), np.zeros((target_h, target_w))
        cam_ec_heatmap, cam_sc_heatmap = np.zeros((target_h, target_w)), np.zeros((target_h, target_w))

        gcam_mask, gcam_plus_mask = np.zeros((target_h, target_w), dtype=np.int64), np.zeros((target_h, target_w), dtype=np.int64)
        ecam_mask, scam_mask = np.zeros((target_h, target_w), dtype=np.int64), np.zeros((target_h, target_w), dtype=np.int64)

        max_val_gc, max_val_gcpp = np.zeros((target_h, target_w)), np.zeros((target_h, target_w))
        max_val_ec, max_val_sc = np.zeros((target_h, target_w)), np.zeros((target_h, target_w))

        for cls_idx in contained_classes:
            c_0indexed = cls_idx - 1
            targets = [ClassifierOutputTarget(c_0indexed)]

            if gcam is not None:
                try:
                    with torch.enable_grad():
                        img_grad = img_input.clone().requires_grad_(True)
                        cam_gc = gcam(input_tensor=img_grad, targets=targets)[0]
                    c_gc_up = F.interpolate(torch.as_tensor(cam_gc).float().view(1, 1, *cam_gc.shape[-2:]), size=(target_h, target_w), mode='bilinear').squeeze().cpu().numpy()
                    cam_gc_heatmap = np.maximum(cam_gc_heatmap, c_gc_up)
                    act_gc = (c_gc_up > gc_thresh) & (c_gc_up > max_val_gc)
                    gcam_mask[act_gc] = cls_idx
                    max_val_gc[act_gc] = c_gc_up[act_gc]
                except Exception:
                    pass

            if gcam_plus is not None:
                try:
                    with torch.enable_grad():
                        img_grad = img_input.clone().requires_grad_(True)
                        cam_gcpp = gcam_plus(input_tensor=img_grad, targets=targets)[0]
                    c_gcpp_up = F.interpolate(torch.as_tensor(cam_gcpp).float().view(1, 1, *cam_gcpp.shape[-2:]), size=(target_h, target_w), mode='bilinear').squeeze().cpu().numpy()
                    cam_gcpp_heatmap = np.maximum(cam_gcpp_heatmap, c_gcpp_up)
                    act_gcpp = (c_gcpp_up > gcpp_thresh) & (c_gcpp_up > max_val_gcpp)
                    gcam_plus_mask[act_gcpp] = cls_idx
                    max_val_gcpp[act_gcpp] = c_gcpp_up[act_gcpp]
                except Exception:
                    pass

            if eigen_cam is not None:
                try:
                    with torch.enable_grad():
                        img_grad = img_input.clone().requires_grad_(True)
                        cam_ec = eigen_cam(input_tensor=img_grad, targets=targets)[0]
                    c_ec_up = F.interpolate(torch.as_tensor(cam_ec).float().view(1, 1, *cam_ec.shape[-2:]), size=(target_h, target_w), mode='bilinear').squeeze().cpu().numpy()
                    cam_ec_heatmap = np.maximum(cam_ec_heatmap, c_ec_up)
                    act_ec = (c_ec_up > ec_thresh) & (c_ec_up > max_val_ec)
                    ecam_mask[act_ec] = cls_idx
                    max_val_ec[act_ec] = c_ec_up[act_ec]
                except Exception:
                    pass

            if score_cam is not None:
                try:
                    with torch.enable_grad():
                        img_grad = img_input.clone().requires_grad_(True)
                        cam_sc = score_cam(input_tensor=img_grad, targets=targets)[0]
                    c_sc_up = F.interpolate(torch.as_tensor(cam_sc).float().view(1, 1, *cam_sc.shape[-2:]), size=(target_h, target_w), mode='bilinear').squeeze().cpu().numpy()
                    cam_sc_heatmap = np.maximum(cam_sc_heatmap, c_sc_up)
                    act_sc = (c_sc_up > sc_thresh) & (c_sc_up > max_val_sc)
                    scam_mask[act_sc] = cls_idx
                    max_val_sc[act_sc] = c_sc_up[act_sc]
                except Exception:
                    pass

        # Step 3: Plot pre-resized image and ground-truth filtered mask predictions
        axes[0, col_offset].imshow(img_display, aspect='auto')
        axes[0, col_offset + 1].imshow(VOC_CMAP[gt_mask_np], aspect='auto')

        axes[1, col_offset].imshow(apply_heatmap_overlay(cam_gc_heatmap, img_display), aspect='auto')
        axes[1, col_offset + 1].imshow(VOC_CMAP[gcam_mask], aspect='auto')

        axes[2, col_offset].imshow(apply_heatmap_overlay(cam_gcpp_heatmap, img_display), aspect='auto')
        axes[2, col_offset + 1].imshow(VOC_CMAP[gcam_plus_mask], aspect='auto')

        axes[3, col_offset].imshow(apply_heatmap_overlay(cam_ec_heatmap, img_display), aspect='auto')
        axes[3, col_offset + 1].imshow(VOC_CMAP[ecam_mask], aspect='auto')

        axes[4, col_offset].imshow(apply_heatmap_overlay(cam_sc_heatmap, img_display), aspect='auto')
        axes[4, col_offset + 1].imshow(VOC_CMAP[scam_mask], aspect='auto')

        axes[5, col_offset].imshow(apply_heatmap_overlay(convx_std_heatmap, img_display), aspect='auto')
        axes[5, col_offset + 1].imshow(VOC_CMAP[convx_std_mask], aspect='auto')

        axes[6, col_offset].imshow(apply_heatmap_overlay(convx_star_heatmap, img_display), aspect='auto')
        axes[6, col_offset + 1].imshow(VOC_CMAP[convx_star_mask], aspect='auto')

        axes[7, col_offset].imshow(apply_heatmap_overlay(convx_plus_heatmap, img_display), aspect='auto')
        axes[7, col_offset + 1].imshow(VOC_CMAP[convx_plus_mask], aspect='auto')

        # Subplot Titles
        axes[0, col_offset].text(
            0.5, 1.03, class_str, transform=axes[0, col_offset].transAxes,
            ha='center', va='bottom', fontweight='bold', fontsize=5.0, linespacing=1.0
        )
        axes[0, col_offset + 1].text(
            0.5, 1.03, "Mask", transform=axes[0, col_offset + 1].transAxes,
            ha='center', va='bottom', fontweight='bold', fontsize=5.0
        )

    # Row Headers
    row_titles = [
        "Input / GT", "Grad-CAM", "Grad-CAM++", "Eigen-CAM",
        "Score-CAM", "ConvX", "ConvX+ (S)", "ConvX+ (M)"
    ]
    for row_idx, title in enumerate(row_titles):
        axes[row_idx, 0].text(
            -0.08, 0.5, title, transform=axes[row_idx, 0].transAxes,
            rotation=90, ha='right', va='center', fontweight='bold', fontsize=5.0
        )

    # Clean Spines/Ticks
    for r in range(total_rows):
        for c in range(total_cols):
            ax = axes[r, c]
            if ax is not None:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

    plt.savefig(output_pdf_path, bbox_inches='tight', pad_inches=0.005, dpi=600)
    print(f"📈 Saved visual comparison PDF to: {output_pdf_path}")

    plt.close(fig)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =====================================================================
# 🚀 MAIN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("=" * 85)
    print("🚀 INITIALIZING VISUAL COMPARISON PIPELINE")
    print("=" * 85)

    # 1. Load Model Checkpoints & Thresholds
    print("\n📦 Loading Model Checkpoints from 'models/'...")
    baseline_model, _ = load_model_checkpoint(CKPT_PATHS["baseline"], StandardBaseline)
    convx_model, convx_thresh = load_model_checkpoint(CKPT_PATHS["convx"], ConvX)
    convx_star_model, convx_star_thresh = load_model_checkpoint(CKPT_PATHS["convx_star"], ConvX)
    convx_plus_model, convx_plus_thresh = load_model_checkpoint(CKPT_PATHS["convx_plus"], ConvX)

    # 2. Configure Target Layers & CAM Explainers
    gcam, gcam_plus, eigen_cam, score_cam = None, None, None, None
    if baseline_model is not None:
        wrapped_baseline = ModelWrapper(baseline_model)

        if hasattr(baseline_model, "backbone"):
            target_layers = [wrapped_baseline.model.backbone[-1]]
        elif hasattr(baseline_model, "layer4"):
            target_layers = [wrapped_baseline.model.layer4]
        else:
            target_layers = [list(wrapped_baseline.model.children())[-2]]

        gcam = GradCAM(model=wrapped_baseline, target_layers=target_layers)
        gcam_plus = GradCAMPlusPlus(model=wrapped_baseline, target_layers=target_layers)
        eigen_cam = EigenCAM(model=wrapped_baseline, target_layers=target_layers)

        score_cam = ScoreCAM(model=wrapped_baseline, target_layers=target_layers)
        score_cam.batch_size = 16

    # 3. Instantiate Direct File Dataset (Direct Resize from disk)
    direct_val_dataset = DirectVOCDataset(
        data_root=DATA_ROOT,
        image_set="val",
        target_size=NETWORK_INPUT_SIZE
    )

    # 4. Run Visual Generation
    visualize_springer_vertical_layout(
        baseline_model=baseline_model,
        convx_model=convx_model,
        convx_star_model=convx_star_model,
        convx_plus_model=convx_plus_model,
        dataset=direct_val_dataset,
        gcam=gcam,
        gcam_plus=gcam_plus,
        eigen_cam=eigen_cam,
        score_cam=score_cam,
        convx_thresh=convx_thresh,
        convx_star_thresh=convx_star_thresh,
        convx_plus_thresh=convx_plus_thresh,
        sample_indices=[4, 159, 1166],
        output_pdf_path=OUTPUT_PDF_PATH
    )