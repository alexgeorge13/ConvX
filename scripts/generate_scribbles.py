# -*- coding: utf-8 -*-
"""Interactive Multi-Stroke Scribble Annotator with Split GT View for ConvX+ (Scribble)"""

import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from convx.dataset_utils import get_supervised_indices
from convx.metrics import get_pascal_voc_palette

# =============================================================================
# ⚙️ CONFIGURATION
# =============================================================================
DEFAULT_BRUSH_SIZE = 7  # Initial brush size in pixels
DATA_DIR = "./data_scribbles"
PREVIEW_DIR = "./scribble_previews"
# =============================================================================

# Global state for mouse callback
drawing = False
last_pt = None
brush_size = DEFAULT_BRUSH_SIZE
user_draw_mask = None


def mouse_callback(event, x, y, flags, param):
    """OpenCV mouse event listener with dual-panel coordinate mapping."""
    global drawing, last_pt, user_draw_mask, brush_size

    if user_draw_mask is None:
        return

    h, w = user_draw_mask.shape
    # Map click coordinates to a single panel width regardless of which panel is clicked
    x_panel = np.clip(x % w, 0, w - 1)
    y_panel = np.clip(y, 0, h - 1)

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        last_pt = (x_panel, y_panel)
        cv2.circle(user_draw_mask, (x_panel, y_panel), max(1, brush_size // 2), 1, -1)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        if last_pt is not None:
            cv2.line(user_draw_mask, last_pt, (x_panel, y_panel), 1, brush_size)
        last_pt = (x_panel, y_panel)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        last_pt = None


def visualize_scribble_mask(scribble_mask, palette):
    """Renders scribble mask cleanly against a dark gray background."""
    h, w = scribble_mask.shape
    rgb_img = np.full((h, w, 3), fill_value=64, dtype=np.uint8)
    palette_flat = palette.flatten()

    for c in range(21):
        mask_c = (scribble_mask == c)
        if np.any(mask_c):
            r = palette_flat[c * 3]
            g = palette_flat[c * 3 + 1]
            b = palette_flat[c * 3 + 2]
            rgb_img[mask_c] = [r, g, b]

    return rgb_img


def run_interactive_annotator():
    global user_draw_mask, brush_size

    print("🎨 Launching ConvX+ Interactive Scribble Annotator (Split GT View)...")
    print("-------------------------------------------------------------------")
    print("🎮 CONTROLS:")
    print("   • Left Click + Drag  : Draw scribbles on either panel")
    print("   • [Space] or [Enter] : Save / Overwrite annotations & go to Next")
    print("   • [S]                : Skip current image without modifying")
    print("   • [C] or [R]         : Clear/Reset current drawn scribbles")
    print("   • [+] / [-]          : Increase / Decrease brush size")
    print("   • [Q] or [ESC]       : Quit application")
    print("-------------------------------------------------------------------\n")

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(PREVIEW_DIR, exist_ok=True)

    base_train = datasets.SBDataset(root="./data_sbd", image_set="train_noval", mode="segmentation", download=False)
    supervised_indices = get_supervised_indices(base_train)
    palette = get_pascal_voc_palette()

    # Pre-build BGR lookup table for VOC palette display
    palette_bgr = palette[:, ::-1]

    win_name = "ConvX+ Interactive Scribble Annotator"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)

    for count, idx in enumerate(supervised_indices):
        save_file = os.path.join(DATA_DIR, f"scribble_{idx}.npy")

        img_pil, gt_pil = base_train[idx]
        gt_mask = np.array(gt_pil, dtype=np.uint8)
        img_rgb = np.array(img_pil, dtype=np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        h, w = gt_mask.shape

        # Check if existing annotation exists and load it into user_draw_mask
        has_existing = os.path.exists(save_file)
        if has_existing:
            existing_scribble = np.load(save_file)
            user_draw_mask = (existing_scribble != 255).astype(np.uint8)
            print(f"📂 [{count + 1}/{len(supervised_indices)}] Index {idx}: Loaded existing annotation from '{save_file}'")
        else:
            user_draw_mask = np.zeros((h, w), dtype=np.uint8)
            print(f"✏️ [{count + 1}/{len(supervised_indices)}] Index {idx}: New image")

        # Pre-render Ground Truth panel (Left)
        gt_bgr = np.zeros_like(img_bgr)
        for c in range(21):
            mask_c = (gt_mask == c)
            if np.any(mask_c):
                gt_bgr[mask_c] = palette_bgr[c]
        # Color unannotated boundaries (255) as dark gray
        gt_bgr[gt_mask == 255] = [128, 128, 128]

        while True:
            # Construct scribble mask from drawn brush pixels + GT mask
            scribble_mask = np.full((h, w), fill_value=255, dtype=np.uint8)
            valid_drawn = (user_draw_mask > 0) & (gt_mask != 255)
            scribble_mask[valid_drawn] = gt_mask[valid_drawn]

            # Build Right Panel (Interactive RGB + Scribbles Overlay)
            right_panel = img_bgr.copy()
            left_panel = gt_bgr.copy()

            if np.any(valid_drawn):
                colored_overlay = np.zeros_like(img_bgr)
                for c in range(21):
                    c_mask = (scribble_mask == c)
                    if np.any(c_mask):
                        colored_overlay[c_mask] = palette_bgr[c]

                # Pure NumPy Blending
                blend_right = (0.3 * img_bgr[valid_drawn] + 0.7 * colored_overlay[valid_drawn]).astype(np.uint8)
                right_panel[valid_drawn] = blend_right

                mask_drawn = user_draw_mask > 0
                blend_left = (0.4 * gt_bgr[mask_drawn] + 0.6 * 255).astype(np.uint8)
                left_panel[mask_drawn] = blend_left

            # Add Header Titles to Panels
            cv2.putText(left_panel, "ORIGINAL GROUND TRUTH", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(left_panel, "ORIGINAL GROUND TRUTH", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.putText(right_panel, "DRAWING CANVAS (RGB)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(right_panel, "DRAWING CANVAS (RGB)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Stack Left and Right panels horizontally
            split_display = np.hstack([left_panel, right_panel])

            # Draw HUD Info Bar across the bottom
            status_tag = "[EXISTING ANNOTATION]" if has_existing else "[NEW]"
            hud = f"Idx: {idx} ({count + 1}/{len(supervised_indices)}) {status_tag} | Brush: {brush_size}px | [Space]: Save | [S]: Skip | [C]: Clear"
            cv2.putText(split_display, hud, (10, split_display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(split_display, hud, (10, split_display.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow(win_name, split_display)
            key = cv2.waitKey(20) & 0xFF

            # Handle Key Inputs
            if key in (27, ord('q')):  # ESC or Q
                print("\n🛑 Annotator closed by user.")
                cv2.destroyAllWindows()
                sys.exit(0)

            elif key == ord('s'):  # S = Skip
                print(f"⏩ Skipped index {idx}.")
                break

            elif key in (32, 13):  # Space or Enter = Save / Overwrite
                if not np.any(valid_drawn):
                    print("⚠️ No scribbles drawn yet! Draw at least one stroke before saving.")
                    continue

                # Save annotation array (.npy)
                np.save(save_file, scribble_mask)
                print(f"✅ Saved annotation to '{save_file}'")

                # Save preview image inside scribble_previews folder for ALL saved images
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(img_pil)
                axes[0].set_title(f"Sample {idx}: Input RGB")
                axes[0].axis("off")

                gt_display = np.where(gt_mask == 255, 0, gt_mask).astype(np.uint8)
                gt_pil_colored = Image.fromarray(gt_display, mode="P")
                gt_pil_colored.putpalette(palette.flatten())

                axes[1].imshow(gt_pil_colored)
                axes[1].set_title("Original Ground Truth")
                axes[1].axis("off")

                scribble_vis = visualize_scribble_mask(scribble_mask, palette)
                axes[2].imshow(scribble_vis)
                axes[2].set_title("Manual Scribbles (Gray = Unannotated 255)")
                axes[2].axis("off")

                plt.tight_layout()
                preview_path = os.path.join(PREVIEW_DIR, f"scribble_preview_{idx}.png")
                plt.savefig(preview_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"🖼️ Saved preview image to '{preview_path}'")

                break  # Proceed to next image

            elif key in (ord('c'), ord('r')):  # Clear/Reset
                user_draw_mask.fill(0)
                print("🧹 Canvas cleared.")

            elif key in (ord('+'), ord('=')):  # Increase brush size
                brush_size = min(49, brush_size + 2)

            elif key in (ord('-'), ord('_')):  # Decrease brush size
                brush_size = max(1, brush_size - 2)

    cv2.destroyAllWindows()
    print(f"\n🎉 Annotation session complete! Annotations saved in '{DATA_DIR}/' and previews in '{PREVIEW_DIR}/'.")


if __name__ == "__main__":
    run_interactive_annotator()