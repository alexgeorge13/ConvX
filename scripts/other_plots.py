# =====================================================================
# ADDITIONAL VISUALIZATIONS & METRICS
# =====================================================================
import glob
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


from convx.config import NETWORK_INPUT_SIZE, VOC_CLASSES

# Try importing torchvision SBDataset for SBD ground-truth matching
try:
    from torchvision.datasets import SBDataset
    HAS_SBD = True
except ImportError:
    HAS_SBD = False

# Paths Configuration
PLOTS_DIR = "plots"
DATA_SCRIBBLES_DIR = "./data_scribbles"
DATA_SBD_DIR = "./data_sbd"

os.makedirs(PLOTS_DIR, exist_ok=True)

# Global Springer Formatting Settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})


# =====================================================================
# 🛠️ HELPER FUNCTIONS & COLORMAPS
# =====================================================================
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

STANDARD_VOC_NAMES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

CLASS_NAME_MAP = {i + 1: STANDARD_VOC_NAMES[i].capitalize() for i in range(20)}


def get_available_scribble_indices(data_scribbles_dir=DATA_SCRIBBLES_DIR):
    """Scans the scribble directory and returns sorted SBD dataset indices that have saved .npy files."""
    if not os.path.exists(data_scribbles_dir):
        return []
    files = glob.glob(os.path.join(data_scribbles_dir, "scribble_*.npy"))
    indices = []
    for f in files:
        base = os.path.basename(f)
        try:
            idx = int(base.replace("scribble_", "").replace(".npy", ""))
            indices.append(idx)
        except ValueError:
            pass
    return sorted(indices)


def visualize_scribble_mask(scribble_mask, bg_val=64):
    """Renders scribble mask cleanly against a dark gray background (matching interactive annotator)."""
    h, w = scribble_mask.shape
    rgb_img = np.full((h, w, 3), fill_value=bg_val, dtype=np.uint8)

    for c in range(21):
        mask_c = (scribble_mask == c)
        if np.any(mask_c):
            rgb_img[mask_c] = VOC_CMAP[c]

    return rgb_img


# =====================================================================
# 📈 MODULAR PLOT 1: SCRIBBLE EXAMPLES (SQUARE SUBPLOTS)
# =====================================================================
def plot_scribble_examples(
    sample_indices=None,
    data_scribbles_dir=DATA_SCRIBBLES_DIR,
    data_sbd_dir=DATA_SBD_DIR,
    target_size=NETWORK_INPUT_SIZE,
    output_pdf_path=os.path.join(PLOTS_DIR, "scribble_examples.pdf")
):
    """Plots 4 sets of images in a 2x2 grid with strictly square subplots.
    Each block contains: [Image + Object Names, Ground-Truth, Manual Scribbles].
    """
    avail_indices = get_available_scribble_indices(data_scribbles_dir)

    # Determine indices to display
    if sample_indices is None:
        if len(avail_indices) >= 4:
            sample_indices = avail_indices[:4]
        else:
            sample_indices = avail_indices
    
    if len(sample_indices) < 4:
        print(f"⚠️ Warning: Found {len(sample_indices)} saved scribbles. Selected indices: {sample_indices}")

    # Load SBD dataset if available
    sbd_dataset = None
    if HAS_SBD and os.path.exists(data_sbd_dir):
        try:
            sbd_dataset = SBDataset(root=data_sbd_dir, image_set="train_noval", mode="segmentation", download=False)
            print("📂 Successfully connected to SBDataset.")
        except Exception as e:
            print(f"⚠️ Could not initialize SBDataset: {e}")

    target_h, target_w = target_size

    # 📏 6 subplots wide x 2 subplots high -> Aspect ratio adjusted so subplots are 1:1 square
    fig = plt.figure(figsize=(4.8, 1.95))

    outer_gs = fig.add_gridspec(
        2, 2,
        wspace=0.10,  # Separation between macro columns
        hspace=0.25,  # Separation between macro rows
        left=0.01, right=0.99, top=0.88, bottom=0.02
    )

    for idx_in_grid in range(min(4, len(sample_indices))):
        sbd_idx = sample_indices[idx_in_grid]
        macro_row = idx_in_grid // 2
        macro_col = idx_in_grid % 2

        inner_gs = outer_gs[macro_row, macro_col].subgridspec(1, 3, wspace=0.04)

        # 1. Load .npy Scribbles
        npy_path = os.path.join(data_scribbles_dir, f"scribble_{sbd_idx}.npy")
        if os.path.exists(npy_path):
            scribble_mask = np.load(npy_path)
        else:
            scribble_mask = np.full((target_h, target_w), fill_value=255, dtype=np.uint8)

        # 2. Load Image & Ground-Truth
        if sbd_dataset is not None and sbd_idx < len(sbd_dataset):
            img_pil, gt_pil = sbd_dataset[sbd_idx]
            img_rgb = np.array(img_pil, dtype=np.uint8)
            gt_mask = np.array(gt_pil, dtype=np.uint8)
        else:
            # Fallback placeholder if dataset files are missing
            img_rgb = np.zeros((target_h, target_w, 3), dtype=np.uint8) + 180
            gt_mask = np.where(scribble_mask != 255, scribble_mask, 0).astype(np.uint8)

        # Resize for consistent grid layout
        img_resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        scribble_resized = cv2.resize(scribble_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        # Extract contained object classes for title (balanced across max 2 lines)
        contained_classes = [c for c in np.unique(gt_resized) if 0 < c <= 20]
        class_names = [CLASS_NAME_MAP.get(c, f"Class {c}") for c in contained_classes]

        if not class_names:
            class_str = "Background"
        elif len(class_names) <= 2:
            class_str = ", ".join(class_names)
        else:
            mid = len(class_names) // 2
            top_line = ", ".join(class_names[:mid])
            bot_line = ", ".join(class_names[mid:])
            class_str = f"{top_line}\n{bot_line}"

        # Create subplots inside subgridspec
        ax_img = fig.add_subplot(inner_gs[0, 0])
        ax_gt = fig.add_subplot(inner_gs[0, 1])
        ax_sc = fig.add_subplot(inner_gs[0, 2])

        # 🎯 Force strict 1:1 square aspect ratio
        ax_img.imshow(img_resized, aspect='equal')
        ax_img.set_title(class_str, fontsize=5.0, fontweight='bold', pad=2)

        gt_vis = VOC_CMAP[gt_resized].copy()
        gt_vis[gt_resized == 255] = [128, 128, 128]
        ax_gt.imshow(gt_vis, aspect='equal')
        ax_gt.set_title("Ground-Truth", fontsize=5.0, fontweight='bold', pad=2)

        scribble_vis = visualize_scribble_mask(scribble_resized, bg_val=64)
        ax_sc.imshow(scribble_vis, aspect='equal')
        ax_sc.set_title("Scribbles", fontsize=5.0, fontweight='bold', pad=2)

        # Format subplot axes
        for ax in (ax_img, ax_gt, ax_sc):
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

    plt.savefig(output_pdf_path, dpi=600, bbox_inches='tight', pad_inches=0.01)
    print(f"📈 Saved Square Scribble Visualizations to: {output_pdf_path}")
    plt.close(fig)


# =====================================================================
# 📈 MODULAR PLOT 2: MIOU VS PIXEL-LEVEL SUPERVISION
# =====================================================================
def plot_supervision_impact(ax=None, save_path=None):
    """Plots mIoU performance scaling across pixel-level image budgets for ConvX, ConvX*, and ConvX+."""
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
        standalone = True

    x1 = np.arange(5)
    x_labels1 = ['0', '1', '2', '5', '10']

    # Performance curves
    miou_convx_star = [0.3470, 0.3712, 0.4032, 0.4460, 0.4843]
    miou_convx_plus = [0.3470, 0.3789, 0.4188, 0.4814, 0.5302]

    # ConvX Baseline (evaluated at 0 pixel-level images)
    ax.scatter(
        x1[0], 0.3470,
        color='#D62728', marker='o', s=55,
        label='ConvX', zorder=4
    )

    # ConvX* Curve
    ax.plot(
        x1, miou_convx_star,
        color='#2CA02C', linestyle='-', linewidth=1.8,
        marker='^', markersize=5.5, label='ConvX+ (S)', zorder=3
    )

    # ConvX+ Curve
    ax.plot(
        x1, miou_convx_plus,
        color='#1F77B4', linestyle='-', linewidth=1.8,
        marker='s', markersize=5.5, label='ConvX+ (M)', zorder=3
    )

    ax.set_xticks(x1)
    ax.set_xticklabels(x_labels1)
    ax.set_xlabel('Pixel-level Annotations per Class')
    ax.set_ylabel('mIoU')
    ax.set_ylim(0.30, 0.58)

    ax.grid(True, linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('(a) Impact of Pixel-Level Supervision')

    if standalone:
        output_path = save_path or os.path.join(PLOTS_DIR, "pixel_supervision_impact.pdf")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"📈 Saved Pixel Supervision Plot to: {output_path}")
        plt.close(fig)


# =====================================================================
# 📈 MODULAR PLOT 3: INFERENCE TIME SCALING
# =====================================================================
def plot_runtime_scaling(ax=None, save_path=None):
    """Plots log-scale inference time across model architectures and CAM baselines."""
    standalone = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
        standalone = True

    x2 = np.arange(3)
    labels2 = ['1', '2', '5']

    models = {
        'Grad-CAM': [28.50, 44.32, 85.49],
        'Grad-CAM++': [31.41, 46.19, 88.35],
        'Eigen-CAM': [11.70, 11.70, 11.70],
        'Score-CAM': [1908.01, 3812.93, 9522.66],
        'ConvX': [9.13, 9.09, 9.17],
        'ConvX+ (S)': [9.21, 9.44, 9.38],
        'ConvX+ (M)': [9.45, 9.62, 9.52],
    }

    colors = {
        'Grad-CAM': '#8C564B',
        'Grad-CAM++': '#9467BD',
        'Eigen-CAM': '#7F7F7F',
        'Score-CAM': '#FF7F0E',
        'ConvX': '#D62728',
        'ConvX+ (S)': '#2CA02C',
        'ConvX+ (M)': '#1F77B4'
    }

    markers = {
        'Grad-CAM': '^',
        'Grad-CAM++': 'D',
        'Eigen-CAM': 'P',
        'Score-CAM': 'v',
        'ConvX': 'o',
        'ConvX+ (S)': '*',
        'ConvX+ (M)': 's'
    }

    for name, values in models.items():
        ax.plot(
            x2, values,
            marker=markers[name],
            color=colors[name],
            linewidth=1.8,
            markersize=6,
            label=name
        )

    ax.set_xticks(x2)
    ax.set_xticklabels(labels2)
    ax.set_xlabel('Number of Classes')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_yscale('log')

    ax.grid(True, which='major', linestyle='--', linewidth=0.6, alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('(b) GPU Inference Time Scaling')

    if standalone:
        output_path = save_path or os.path.join(PLOTS_DIR, "runtime_scaling.pdf")
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"📈 Saved Runtime Scaling Plot to: {output_path}")
        plt.close(fig)


# =====================================================================
# 📊 COMBINED SIDE-BY-SIDE FIGURE GENERATOR
# =====================================================================
def plot_combined_supervision_and_runtime(
    output_pdf_path=os.path.join(PLOTS_DIR, "iou_and_timing_plots.pdf")
):
    """Combines Plot 2 and Plot 3 side-by-side into a single Springer-ready figure."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7.2, 3.2),
        constrained_layout=True
    )

    plot_supervision_impact(ax=ax1)
    plot_runtime_scaling(ax=ax2)

    plt.savefig(output_pdf_path, dpi=600, bbox_inches='tight')
    print(f"📈 Saved Combined Side-by-Side Figure to: {output_pdf_path}")
    plt.close(fig)


# =====================================================================
# 🚀 MAIN ENTRYPOINT
# =====================================================================
if __name__ == "__main__":
    print("=" * 85)
    print("🚀 GENERATING ADDITIONAL SPRINGER PROCEEDINGS VISUALIZATIONS")
    print("=" * 85)

    # 🎯 Choose your exact 4 indices here from your ~81 saved scribbles
    selected_indices = [0, 1, 2, 6]  # <--- Replace these numbers with your chosen indices

    # Check if all selected indices actually exist in ./data_scribbles
    available_scribble_indices = get_available_scribble_indices(DATA_SCRIBBLES_DIR)
    for idx in selected_indices:
        if idx not in available_scribble_indices:
            print(f"⚠️ Warning: 'scribble_{idx}.npy' not found in {DATA_SCRIBBLES_DIR}!")

    # 2. Plot Square Scribble Examples Grid from .npy files
    plot_scribble_examples(
        sample_indices=selected_indices,
        data_scribbles_dir=DATA_SCRIBBLES_DIR,
        data_sbd_dir=DATA_SBD_DIR,
        output_pdf_path=os.path.join(PLOTS_DIR, "scribble_examples.pdf")
    )

    # 3. Plot Standalone Pixel Supervision Impact
    plot_supervision_impact()

    # 4. Plot Standalone Runtime Scaling
    plot_runtime_scaling()

    # 5. Plot Combined Side-by-Side Figure
    plot_combined_supervision_and_runtime(
        output_pdf_path=os.path.join(PLOTS_DIR, "iou_and_timing_plots.pdf")
    )

    print("\n✅ All visualization outputs generated successfully!")