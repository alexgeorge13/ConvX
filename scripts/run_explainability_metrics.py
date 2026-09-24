# =====================================================================
# 📊 EXPLAINABILITY METRICS AUDIT (Production-Ready Script)
# =====================================================================
import gc
import os
from contextlib import contextmanager
import sys


import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.integrate import simpson
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import VOCSegmentation
from tqdm import tqdm

# PyTorch Grad-CAM Imports
try:
    from pytorch_grad_cam import EigenCAM, GradCAM, GradCAMPlusPlus, ScoreCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except ImportError:
    raise ImportError(
        "pytorch_grad_cam library is required. Install via: pip install grad-cam"
    )

# Local Project Imports
from convx.config import DEVICE, NETWORK_INPUT_SIZE
from convx.dataset_utils import (
    AlbumentationsDatasetWrapper,
    seg_collate_fn_for_val,
    val_transform_pipeline,
)
from convx.models import ConvX, StandardBaseline

# =====================================================================
# ⚙️ CONFIGURATION & METHOD TOGGLES
# =====================================================================
ENABLED_METHODS = {
    "Grad-CAM": True,  # Baseline + Grad-CAM
    "Grad-CAM++": True,  # Baseline + Grad-CAM++
    "Eigen-CAM": True,  # Baseline + Eigen-CAM
    "Score-CAM": True,  # Baseline + Score-CAM (Heavy GPU RAM usage)
    "ConvX": True,  # ConvX Standard (Native Explainability)
    "ConvX+ (S)": True,  # ConvX+ (S) 5-per-class (Native Explainability)
    "ConvX+ (M)": True,  # ConvX+ (M) 5-per-class (Native Explainability)
}

# Folder Paths
MODELS_DIR = "models"
PLOTS_DIR = "plots"

os.makedirs(PLOTS_DIR, exist_ok=True)

CKPT_PATHS = {
    "baseline": os.path.join(MODELS_DIR, "classifier_baseline_best.pth"),
    "convx": os.path.join(MODELS_DIR, "convx_standard_best.pth"),
    "convx_plus_scribble": os.path.join(MODELS_DIR, "convx_plus_scribble_5_per_class_best.pth"),
    "convx_plus_mask": os.path.join(MODELS_DIR, "convx_plus_mask_5_per_class_best.pth"),
}

DATA_ROOT = "./data_voc"
NUM_SAMPLES = 200  # Number of validation images to evaluate
EVAL_STEPS = 20  # Deletion/Insertion perturbation steps
BATCH_SIZE = 1


# =====================================================================
# 🛠️ HELPER CLASSES & SAFETY CONTROLS
# =====================================================================
@contextmanager
def suppress_stdout_stderr():
    """Completely silences internal print statements and nested tqdm bars."""
    with open(os.devnull, "w") as fnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = fnull
        sys.stderr = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class ModelWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


class XAIMetricsEngine:

    def __init__(self, steps=20):
        self.steps = steps

    def get_neutral_canvas(self, img_tensor):
        """⚡ Pure GPU Blurring Pyramid (Zero System RAM overhead)."""
        H, W = img_tensor.shape[2:]
        low_res = F.interpolate(
            img_tensor,
            size=(H // 16, W // 16),
            mode="bilinear",
            align_corners=False,
        )
        return F.interpolate(
            low_res, size=(H, W), mode="bilinear", align_corners=False
        )

    def dynamic_range_normalization(self, z, z_neutral, z_orig):
        """Unified Min-Max normalization based on true empirical bounds."""
        denominator = z_orig - z_neutral
        if abs(denominator) < 1e-6:
            return 1.0
        normalized = (z - z_neutral) / denominator
        return float(np.clip(normalized, 0.0, 1.0))

    def compute_del_ins_auc(
        self, model, explainer, img_tensor, class_idx, is_native_heatmap=False
    ):
        model.eval()
        H, W = img_tensor.shape[2:]

        with torch.no_grad():
            out_orig = model(img_tensor)
            raw_pred = (
                out_orig[0][0, class_idx].item()
                if isinstance(out_orig, (tuple, list))
                else out_orig[0, class_idx].item()
            )
        z_orig = (
            raw_pred
            if is_native_heatmap
            else torch.sigmoid(torch.tensor(raw_pred)).item()
        )

        neutral = self.get_neutral_canvas(img_tensor)
        with torch.no_grad():
            out_neutral = model(neutral)
            raw_neutral = (
                out_neutral[0][0, class_idx].item()
                if isinstance(out_neutral, (tuple, list))
                else out_neutral[0, class_idx].item()
            )
        z_neutral = (
            raw_neutral
            if is_native_heatmap
            else torch.sigmoid(torch.tensor(raw_neutral)).item()
        )

        if is_native_heatmap:
            with torch.no_grad():
                heatmap = model(img_tensor)[1][0, class_idx]
        else:
            targets = [ClassifierOutputTarget(class_idx)]
            with suppress_stdout_stderr(), torch.enable_grad():
                local_img = img_tensor.detach().clone().requires_grad_(True)

                for module in model.modules():
                    if hasattr(module, "inplace"):
                        module.inplace = False

                for param in model.parameters():
                    param.requires_grad = True

                model.zero_grad()
                heatmap_np = explainer(
                    input_tensor=local_img, targets=targets
                )[0]

            heatmap = torch.from_numpy(heatmap_np).to(img_tensor.device)

        heatmap_res = F.interpolate(
            heatmap.view(1, 1, *heatmap.shape).float(),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

        sorted_indices = torch.argsort(
            heatmap_res.view(-1), descending=True
        )
        step_size = len(sorted_indices) // self.steps

        del_img = img_tensor.clone()
        ins_img = neutral.clone()

        del_scores = []
        ins_scores = []

        with torch.no_grad():
            for step in range(self.steps + 1):
                out_del = model(del_img)
                out_ins = model(ins_img)

                raw_del = (
                    out_del[0][0, class_idx].item()
                    if isinstance(out_del, (tuple, list))
                    else out_del[0, class_idx].item()
                )
                raw_ins = (
                    out_ins[0][0, class_idx].item()
                    if isinstance(out_ins, (tuple, list))
                    else out_ins[0, class_idx].item()
                )

                z_del = (
                    raw_del
                    if is_native_heatmap
                    else torch.sigmoid(torch.tensor(raw_del)).item()
                )
                z_ins = (
                    raw_ins
                    if is_native_heatmap
                    else torch.sigmoid(torch.tensor(raw_ins)).item()
                )

                del_scores.append(
                    self.dynamic_range_normalization(
                        z_del, z_neutral, z_orig
                    )
                )
                ins_scores.append(
                    self.dynamic_range_normalization(
                        z_ins, z_neutral, z_orig
                    )
                )

                if step < self.steps:
                    idx_alter = sorted_indices[
                        step * step_size : (step + 1) * step_size
                    ]
                    rows, cols = idx_alter // W, idx_alter % W
                    del_img[0, :, rows, cols] = neutral[0, :, rows, cols]
                    ins_img[0, :, rows, cols] = img_tensor[0, :, rows, cols]

        x_axis = np.linspace(0, 1, len(del_scores))
        del_auc = simpson(y=del_scores, x=x_axis)
        ins_auc = simpson(y=ins_scores, x=x_axis)

        del heatmap, heatmap_res, sorted_indices, neutral, del_img, ins_img
        return float(del_auc), float(ins_auc), del_scores, ins_scores

    def compute_robustness(
        self, model, explainer, img_tensor, class_idx, is_native_heatmap=False
    ):
        noise_std = 0.05
        noisy_img = torch.clamp(
            img_tensor + (torch.randn_like(img_tensor) * noise_std), -2.5, 2.5
        )

        if is_native_heatmap:
            with torch.no_grad():
                map_orig = model(img_tensor)[1][0, class_idx].view(-1)
                map_noisy = model(noisy_img)[1][0, class_idx].view(-1)
        else:
            targets = [ClassifierOutputTarget(class_idx)]
            with suppress_stdout_stderr(), torch.enable_grad():
                local_img = img_tensor.detach().clone().requires_grad_(True)
                local_noisy = noisy_img.detach().clone().requires_grad_(True)

                for module in model.modules():
                    if hasattr(module, "inplace"):
                        module.inplace = False

                for param in model.parameters():
                    param.requires_grad = True

                model.zero_grad()
                map_orig_np = explainer(
                    input_tensor=local_img, targets=targets
                )[0]

                model.zero_grad()
                map_noisy_np = explainer(
                    input_tensor=local_noisy, targets=targets
                )[0]

            map_orig = (
                torch.from_numpy(map_orig_np).to(img_tensor.device).view(-1)
            )
            map_noisy = (
                torch.from_numpy(map_noisy_np).to(img_tensor.device).view(-1)
            )

        mo_np = map_orig.cpu().numpy()
        mn_np = map_noisy.cpu().numpy()
        spearman_corr, _ = stats.spearmanr(mo_np, mn_np)

        del noisy_img, map_orig, map_noisy, mo_np, mn_np
        if np.isnan(spearman_corr):
            return 0.0
        return float(max(0.0, spearman_corr))


# =====================================================================
# 🚀 MAIN AUDIT EXECUTION
# =====================================================================
def main():
    print("=" * 85)
    print("🚀 INITIALIZING EXPLAINABILITY METRICS AUDIT ENGINE")
    print("=" * 85)

    voc_dir = os.path.join(DATA_ROOT, "VOCdevkit")
    should_download = not os.path.exists(voc_dir)
    if not should_download:
        print(f"📁 Local dataset found at '{voc_dir}'. Skipping download.")
    else:
        print(f"⚠️ Dataset not found at '{voc_dir}'. Downloading...")

    base_val_dataset = VOCSegmentation(
        root=DATA_ROOT, year="2012", image_set="val", download=should_download
    )
    val_dataset = AlbumentationsDatasetWrapper(
        base_dataset=base_val_dataset, transform=val_transform_pipeline
    )

    num_workers = 0 if os.name == "nt" else 2

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=seg_collate_fn_for_val,
    )

    # 2. Load Models
    print("\n📦 Loading Model Checkpoints...")
    models_dict = {}

    def load_model_ckpt(path, model_class=None, **model_kwargs):
        if not os.path.exists(path):
            print(f"⚠️ Warning: Checkpoint '{path}' not found! Skipping.")
            return None

        # Pass weights_only=False to silence PyTorch warning
        ckpt = torch.load(path, map_location=DEVICE, weights_only=False)

        # Case A: Entire model object was saved
        if isinstance(ckpt, nn.Module):
            model = ckpt

        # Case B: Weight dictionary (state_dict / OrderedDict) was saved
        else:
            if model_class is None:
                raise ValueError(
                    f"Checkpoint at '{path}' is a state_dict. "
                    "You must provide a 'model_class' to load these weights into."
                )

            # Unpack state_dict if nested
            if isinstance(ckpt, dict):
                if "state_dict" in ckpt:
                    state_dict = ckpt["state_dict"]
                elif "model" in ckpt and isinstance(ckpt["model"], dict):
                    state_dict = ckpt["model"]
                else:
                    state_dict = ckpt
            else:
                state_dict = ckpt

            # Instantiate and load
            model = model_class(**model_kwargs)
            model.load_state_dict(state_dict)

        model.eval().to(DEVICE)
        return model

    # Load Baseline Model
    baseline_needed = any(
        ENABLED_METHODS[k]
        for k in ["Grad-CAM", "Grad-CAM++", "Eigen-CAM", "Score-CAM"]
    )
    if baseline_needed:
        models_dict["baseline"] = load_model_ckpt(
            CKPT_PATHS["baseline"], 
            model_class=StandardBaseline
        )

    # Load ConvX Model Variants
    if ENABLED_METHODS["ConvX"]:
        models_dict["convx"] = load_model_ckpt(
            CKPT_PATHS["convx"], 
            model_class=ConvX
        )

    if ENABLED_METHODS["ConvX+ (S)"]:
        models_dict["convx_plus_scribble"] = load_model_ckpt(
            CKPT_PATHS["convx_plus_scribble"], 
            model_class=ConvX
        )

    if ENABLED_METHODS["ConvX+ (M)"]:
        models_dict["convx_plus_mask"] = load_model_ckpt(
            CKPT_PATHS["convx_plus_mask"], 
            model_class=ConvX
        )

    explainers_dict = {}
    if baseline_needed and models_dict.get("baseline") is not None:
        wrapped_baseline = ModelWrapper(models_dict["baseline"])

        if hasattr(wrapped_baseline.model, "backbone"):
            target_layers = [wrapped_baseline.model.backbone[-1]]
        else:
            target_layers = [list(wrapped_baseline.model.children())[-1]]

        if ENABLED_METHODS["Grad-CAM"]:
            explainers_dict["Grad-CAM"] = GradCAM(
                model=wrapped_baseline, target_layers=target_layers
            )
        if ENABLED_METHODS["Grad-CAM++"]:
            explainers_dict["Grad-CAM++"] = GradCAMPlusPlus(
                model=wrapped_baseline, target_layers=target_layers
            )
        if ENABLED_METHODS["Eigen-CAM"]:
            explainers_dict["Eigen-CAM"] = EigenCAM(
                model=wrapped_baseline, target_layers=target_layers
            )
        if ENABLED_METHODS["Score-CAM"]:
            explainers_dict["Score-CAM"] = ScoreCAM(
                model=wrapped_baseline, target_layers=target_layers
            )

    configurations = []

    if ENABLED_METHODS["Grad-CAM"] and "Grad-CAM" in explainers_dict:
        configurations.append(
            ("Grad-CAM", models_dict["baseline"], explainers_dict["Grad-CAM"], False)
        )
    if ENABLED_METHODS["Grad-CAM++"] and "Grad-CAM++" in explainers_dict:
        configurations.append(
            ("Grad-CAM++", models_dict["baseline"], explainers_dict["Grad-CAM++"], False)
        )
    if ENABLED_METHODS["Eigen-CAM"] and "Eigen-CAM" in explainers_dict:
        configurations.append(
            ("Eigen-CAM", models_dict["baseline"], explainers_dict["Eigen-CAM"], False)
        )
    if ENABLED_METHODS["Score-CAM"] and "Score-CAM" in explainers_dict:
        configurations.append(
            ("Score-CAM", models_dict["baseline"], explainers_dict["Score-CAM"], False)
        )

    if ENABLED_METHODS["ConvX"] and models_dict.get("convx") is not None:
        configurations.append(("ConvX", models_dict["convx"], None, True))
    if ENABLED_METHODS["ConvX+ (S)"] and models_dict.get("convx_plus_scribble") is not None:
        configurations.append(("ConvX+ (S)", models_dict["convx_plus_scribble"], None, True))
    if ENABLED_METHODS["ConvX+ (M)"] and models_dict.get("convx_plus_mask") is not None:
        configurations.append(("ConvX+ (M)", models_dict["convx_plus_mask"], None, True))

    active_methods = [cfg[0] for cfg in configurations]
    print(f"\n✅ Active Evaluation Pipeline ({len(active_methods)} Methods): {active_methods}")

    metrics_engine = XAIMetricsEngine(steps=EVAL_STEPS)

    del_aucs = {m: [] for m in active_methods}
    ins_aucs = {m: [] for m in active_methods}
    robust_scores = {m: [] for m in active_methods}
    del_curves_tracked = {m: [] for m in active_methods}
    ins_curves_tracked = {m: [] for m in active_methods}

    count = 0
    pbar = tqdm(total=NUM_SAMPLES, desc="Computing XAI Metrics")

    for img_tensor, mask_tensor in val_loader:
        if count >= NUM_SAMPLES:
            break

        img_tensor = img_tensor.to(DEVICE)
        gt_classes = [c for c in torch.unique(mask_tensor) if 0 < c <= 20]
        if not gt_classes:
            continue

        target_cls = int(gt_classes[0]) - 1

        for m, net, expl, is_native in configurations:
            try:
                outputs = metrics_engine.compute_del_ins_auc(
                    net, expl, img_tensor, target_cls, is_native
                )
                del_auc, ins_auc, d_scores, i_scores = outputs

                rob_score = metrics_engine.compute_robustness(
                    net, expl, img_tensor, target_cls, is_native
                )

                del_aucs[m].append(del_auc)
                ins_aucs[m].append(ins_auc)
                del_curves_tracked[m].append(d_scores)
                ins_curves_tracked[m].append(i_scores)
                robust_scores[m].append(rob_score)
            except Exception as e:
                print(f"\n⚠️ Failure on method {m} for sample {count}: {e}")

            if expl is not None and hasattr(expl, "activations_and_gradients"):
                expl.activations_and_gradients.activations = []
                expl.activations_and_gradients.gradients = []

        count += 1
        pbar.update(1)

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    pbar.close()

    # 6. Tabular Summary Report & Excel Export
    print("\n" + "=" * 85)
    print(
        f"{'Method Configuration':<20} | {'Valid Imgs':<10} | {'Del AUC (↓)':<12} | {'Ins AUC (↑)':<12} | {'Robustness (↑)':<14}"
    )
    print("-" * 85)

    summary_rows = []

    for m in active_methods:
        valid_count = len(del_aucs[m])
        mean_del = np.mean(del_aucs[m]) if valid_count > 0 else 0.0
        mean_ins = np.mean(ins_aucs[m]) if valid_count > 0 else 0.0
        mean_rob = np.mean(robust_scores[m]) if valid_count > 0 else 0.0

        print(
            f"{m:<20} | {valid_count:<10} | {mean_del:<12.4f} | {mean_ins:<12.4f} | {mean_rob:<14.4f}"
        )

        summary_rows.append(
            {
                "Method Configuration": m,
                "Valid Images": valid_count,
                "Deletion AUC (↓)": round(mean_del, 4),
                "Insertion AUC (↑)": round(mean_ins, 4),
                "Robustness (↑)": round(mean_rob, 4),
            }
        )

    print("=" * 85)

    # 7. Generate Springer-Compliant PDF Figures
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6.5,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6))
    x_steps = np.linspace(0, 100, EVAL_STEPS + 1)

    colors = {
        "Grad-CAM": "#2CA02C",
        "Grad-CAM++": "#9467BD",
        "Eigen-CAM": "#8C564B",
        "Score-CAM": "#FF7F0E",
        "ConvX": "#D62728",
        "ConvX+ (S)": "#17BECF",
        "CConvX+ (M)": "#1F77B4",
    }

    for m in active_methods:
        if len(del_curves_tracked[m]) == 0:
            continue

        avg_del_curve = np.mean(del_curves_tracked[m], axis=0)
        avg_ins_curve = np.mean(ins_curves_tracked[m], axis=0)

        axes[0].plot(
            x_steps,
            avg_del_curve,
            label=f"{m} ({np.mean(del_aucs[m]):.2f})",
            color=colors.get(m, "#000000"),
            linewidth=1.5,
        )
        axes[1].plot(
            x_steps,
            avg_ins_curve,
            label=f"{m} ({np.mean(ins_aucs[m]):.2f})",
            color=colors.get(m, "#000000"),
            linewidth=1.5,
        )

    axes[0].set_title("Deletion Curve (↓)", fontweight="bold")
    axes[0].set_xlabel("% Pixels Removed")
    axes[0].set_ylabel("Norm. Confidence")
    axes[0].set_xlim(0, 100)
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].set_title("Insertion Curve (↑)", fontweight="bold")
    axes[1].set_xlabel("% Pixels Added")
    axes[1].set_ylabel("Norm. Confidence")
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].grid(True, linestyle="--", alpha=0.4, linewidth=0.5)
    axes[1].legend(loc="lower right", frameon=False)

    plt.tight_layout()
    output_pdf = os.path.join(PLOTS_DIR, "explainability_curves.pdf")
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight", pad_inches=0.01)
    print(f"📈 Saved academic vector plot to: {output_pdf}")
    plt.close()


if __name__ == "__main__":
    main()