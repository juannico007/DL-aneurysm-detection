"""
Multi-Model Comparison Script for Aneurysm Detection

This script compares multiple trained models by:
1. Running sliding inference on a test patient with an aneurysm
2. Visualizing segmentation outputs at the aneurysm slice
3. Comparing training metrics across models
4. Creating 3D overlay visualizations

Configuration:
- Edit MODEL_CONFIGS to specify models to compare
- Edit INFERENCE_CONFIG for inference parameters
"""

from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import csv
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet
from model.training_pipeline import make_heatmaps


# =====================================================================
# CONFIGURATION
# =====================================================================

# Models to compare - add/remove model directories as needed
MODEL_CONFIGS = [
    "cloud_models/RusnacAM/AMR_003",
    "cloud_models/MihaiB-dev/suppression-loss_threshold-on-gauss-5",
]

INFERENCE_CONFIG = {
    "test_h5_path": "test_dataset.h5",
    "test_csv": "test.csv",
    "test_localizers_csv": "test_localizers.csv",
    "patch_size": 64,
    "stride": 24,
    "threshold": 0.5,
    "blend_mode": "hann",  # Patch blending mode: 'hann' for smooth windowing, 'uniform' for simple averaging
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "use_mixed_precision": False,
    "gauss_sigma": 5.0,  # For GT generation
    "output_dir": "model_comparison_results",
    "patient_id": None,  # If None, randomly select; otherwise specify SeriesInstanceUID
}


# =====================================================================
# DATA CLASSES
# =====================================================================

@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    model: UNet
    hyperparams: Dict
    history: Dict
    has_segmentation: bool
    has_classification: bool
    has_regression: bool


# =====================================================================
# MODEL LOADING UTILITIES
# =====================================================================

def load_model_config(model_dir: Path) -> Dict:
    """Load hyperparameters.json from model directory."""
    hp_path = model_dir / "hyperparameters.json"
    if not hp_path.exists():
        raise FileNotFoundError(f"hyperparameters.json not found in {model_dir}")
    
    with hp_path.open() as f:
        return json.load(f)


def load_trained_model(model_dir: Path) -> ModelInfo:
    """
    Load a trained model from directory containing:
    - hyperparameters.json
    - *.pt checkpoint file
    - history.pickle
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    # Load hyperparameters
    hyperparams = load_model_config(model_dir)
    unet_kwargs = hyperparams.get("unet", {})
    
    # Determine model capabilities
    has_segmentation = unet_kwargs.get("out_channels", 1) >= 1
    has_classification = unet_kwargs.get("class_output", 0) >= 1
    # Regression is typically indicated by class_output > 1 or a separate parameter
    has_regression = unet_kwargs.get("class_output", 0) > 1
    
    # Create model
    model = UNet(**unet_kwargs)
    
    # Find checkpoint file
    pt_files = list(model_dir.glob("*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt checkpoint found in {model_dir}")
    checkpoint_path = pt_files[0]
    
    # Load weights
    print(f"[Info] Loading {model_dir.name} from {checkpoint_path.name}")
    state = torch.load(checkpoint_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [Warning] Missing keys: {missing[:3]}..." if len(missing) > 3 else f"  [Warning] Missing keys: {missing}")
    if unexpected:
        print(f"  [Warning] Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"  [Warning] Unexpected keys: {unexpected}")
    
    # Load history
    history_path = model_dir / "history.pickle"
    history = {}
    if history_path.exists():
        with history_path.open("rb") as f:
            history = pickle.load(f)
    else:
        print(f"  [Warning] No history.pickle found for {model_dir.name}")
    
    return ModelInfo(
        name=model_dir.name,
        model=model,
        hyperparams=hyperparams,
        history=history,
        has_segmentation=has_segmentation,
        has_classification=has_classification,
        has_regression=has_regression,
    )


# =====================================================================
# PATIENT SELECTION
# =====================================================================

def select_test_patient(test_csv: Path, patient_id: Optional[str] = None) -> str:
    """
    Select a test patient with an aneurysm.
    If patient_id is provided, validate it has an aneurysm.
    Otherwise, randomly select from positive cases.
    """
    df = pd.read_csv(test_csv)
    
    if patient_id is not None:
        # Validate provided patient
        patient_row = df[df["SeriesInstanceUID"] == patient_id]
        if patient_row.empty:
            raise ValueError(f"Patient {patient_id} not found in {test_csv}")
        if patient_row["Aneurysm Present"].values[0] != 1:
            raise ValueError(f"Patient {patient_id} does not have an aneurysm")
        print(f"[Info] Using specified patient: {patient_id}")
        return patient_id
    
    # Select random positive case
    positive_cases = df[df["Aneurysm Present"] == 1]
    if positive_cases.empty:
        raise ValueError(f"No positive cases found in {test_csv}")
    
    selected = positive_cases.sample(n=1).iloc[0]
    patient_id = selected["SeriesInstanceUID"]
    print(f"[Info] Randomly selected patient: {patient_id}")
    return patient_id


# =====================================================================
# GROUND TRUTH GENERATION
# =====================================================================

def load_localizer_points(series_id: str, localizers_csv: Path) -> List[Tuple[float, float, float]]:
    """Load aneurysm localizer points for a specific patient."""
    points = []
    
    if not localizers_csv.exists():
        print(f"[Warning] Localizers CSV not found: {localizers_csv}")
        return points
    
    with localizers_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SeriesInstanceUID") != series_id:
                continue
            try:
                z = float(row["z_new"])
                y = float(row["y_new"])
                x = float(row["x_new"])
                points.append((z, y, x))
            except (KeyError, ValueError) as e:
                print(f"[Warning] Failed to parse localizer row: {e}")
                continue
    
    return points


def build_gaussian_gt(shape: Tuple[int, int, int], points: List[Tuple[float, float, float]], 
                     sigma: float = 5.0) -> np.ndarray:
    """Build Gaussian ground truth heatmap from localizer points."""
    if not points:
        return np.zeros(shape, dtype=np.float32)
    
    heatmap = np.zeros(shape, dtype=np.float32)
    for z, y, x in points:
        heatmap += make_heatmaps(shape, (z, y, x), [sigma])[0]
    
    return np.clip(heatmap, 0.0, 1.0)


def find_aneurysm_slice(points: List[Tuple[float, float, float]]) -> int:
    """Find the central z-slice where aneurysm is most prominent."""
    if not points:
        return 0
    
    z_coords = [p[0] for p in points]
    return int(np.mean(z_coords))


# =====================================================================
# SLIDING INFERENCE
# =====================================================================

def sliding_window(volume: np.ndarray, patch_size: int, stride: int):
    """Generate sliding window patches with their positions."""
    D, H, W = volume.shape
    z_starts = list(range(0, max(D - patch_size, 0) + 1, stride))
    y_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
    x_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
    
    # Ensure we cover edges
    if z_starts[-1] != D - patch_size:
        z_starts.append(D - patch_size)
    if y_starts[-1] != H - patch_size:
        y_starts.append(H - patch_size)
    if x_starts[-1] != W - patch_size:
        x_starts.append(W - patch_size)
    
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = volume[z:z + patch_size, y:y + patch_size, x:x + patch_size]
                yield patch, (z, y, x)


def build_weight_mask(patch_size: int, mode: str = "hann") -> np.ndarray:
    """
    Build a weight mask for blending patches.
    
    Args:
        patch_size: Size of the cubic patch
        mode: Blending mode ('hann' for Hann windowing, 'uniform' for equal weights)
    
    Returns:
        3D weight mask array
    """
    if mode == "hann":
        # Hann window provides smooth transitions at patch boundaries
        h = np.hanning(patch_size)
        return (h[:, None, None] * h[None, :, None] * h[None, None, :]).astype(np.float32)
    return np.ones((patch_size, patch_size, patch_size), dtype=np.float32)


def run_inference_for_model(
    model_info: ModelInfo,
    volume: np.ndarray,
    config: Dict
) -> Dict[str, np.ndarray]:
    """
    Run sliding window inference for a single model with Hann window blending.
    Returns dict with 'prob', 'mask', and optionally 'classification', 'regression'.
    """
    model = model_info.model
    device = torch.device(config["device"])
    model.to(device)
    model.eval()
    
    patch_size = config["patch_size"]
    stride = config["stride"]
    threshold = config["threshold"]
    blend_mode = config.get("blend_mode", "hann")  # Default to Hann windowing
    use_amp = config.get("use_mixed_precision", False) and device.type == "cuda"
    
    # Initialize output arrays
    prob_map = np.zeros_like(volume, dtype=np.float32)
    weight_map = np.zeros_like(volume, dtype=np.float32)
    
    # Build weight patch for blending
    weight_patch = build_weight_mask(patch_size, mode=blend_mode)
    
    classification_outputs = []
    regression_outputs = []
    
    print(f"  Running sliding inference for {model_info.name} (blend={blend_mode})...")
    
    with torch.no_grad():
        for patch_np, (z, y, x) in tqdm(
            sliding_window(volume, patch_size, stride),
            desc=f"  {model_info.name}",
            leave=False
        ):
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(
                device=device, dtype=torch.float32
            )
            
            if use_amp:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(patch_t)
            else:
                outputs = model(patch_t)
            
            # Handle different output formats
            # Possible outputs: (seg_logits, class_output, attention) or just seg_logits
            if isinstance(outputs, tuple):
                seg_logits = outputs[0]
                if len(outputs) > 1 and outputs[1] is not None:
                    class_output = outputs[1]
                    classification_outputs.append(torch.sigmoid(class_output).cpu().numpy())
            else:
                seg_logits = outputs
            
            # Process segmentation output with weighted blending
            if seg_logits is not None:
                seg_probs = torch.sigmoid(seg_logits.float()).cpu().numpy()[0, 0]
                prob_map[z:z + patch_size, y:y + patch_size, x:x + patch_size] += seg_probs * weight_patch
                weight_map[z:z + patch_size, y:y + patch_size, x:x + patch_size] += weight_patch
    
    # Normalize by weights
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    mask = (prob_map >= threshold).astype(np.uint8)
    
    result = {
        "prob": prob_map,
        "mask": mask,
    }
    
    if classification_outputs:
        result["classification"] = np.mean(classification_outputs)
    
    return result


# =====================================================================
# VISUALIZATION FUNCTIONS
# =====================================================================

def plot_segmentation_comparison(
    volume: np.ndarray,
    slice_idx: int,
    ground_truth: np.ndarray,
    model_results: Dict[str, Dict],
    output_path: Path,
    aneurysm_points: List[Tuple[float, float, float]]
):
    """
    Create comprehensive comparison plot showing:
    - Original CT slice
    - Ground truth
    - Each model's probability map
    - Each model's binary mask
    """
    n_models = len(model_results)
    n_cols = min(4, n_models + 2)  # CT, GT, + models
    n_rows = int(np.ceil((n_models + 2) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).flatten()
    
    plot_idx = 0
    
    # Original CT slice
    ax = axes[plot_idx]
    ax.imshow(volume[slice_idx, :, :], cmap='gray')
    ax.set_title('Original CT Slice', fontsize=12, fontweight='bold')
    ax.axis('off')
    
    # Mark aneurysm locations
    for z, y, x in aneurysm_points:
        if abs(z - slice_idx) < 3:  # Within 3 slices
            ax.plot(x, y, 'r*', markersize=15, markeredgewidth=2, markeredgecolor='yellow')
    plot_idx += 1
    
    # Ground truth
    ax = axes[plot_idx]
    ax.imshow(volume[slice_idx, :, :], cmap='gray')
    gt_overlay = ax.imshow(ground_truth[slice_idx, :, :], cmap='hot', alpha=0.5, vmin=0, vmax=1)
    ax.set_title('Ground Truth (Gaussian)', fontsize=12, fontweight='bold')
    ax.axis('off')
    plt.colorbar(gt_overlay, ax=ax, fraction=0.046)
    plot_idx += 1
    
    # Each model's outputs
    for model_name, results in model_results.items():
        # Probability map
        ax = axes[plot_idx]
        ax.imshow(volume[slice_idx, :, :], cmap='gray')
        prob_overlay = ax.imshow(
            results["prob"][slice_idx, :, :], 
            cmap='hot', 
            alpha=0.5, 
            vmin=0, 
            vmax=1
        )
        ax.set_title(f'{model_name}\nProbability', fontsize=10)
        ax.axis('off')
        plt.colorbar(prob_overlay, ax=ax, fraction=0.046)
        plot_idx += 1
        
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved segmentation comparison to {output_path}")


def create_3d_overlay(
    volume: np.ndarray,
    prob_map: np.ndarray,
    output_path: Path,
    model_name: str,
    slice_axis: int = 0,
    slice_idx: Optional[int] = None
):
    """
    Create a visualization with brain volume overlapped with segmentation probability map.
    
    Parameters:
        volume: 3D CT volume
        prob_map: 3D probability map
        output_path: Where to save the visualization
        model_name: Name of the model for title
        slice_axis: Which axis to slice (0=axial, 1=coronal, 2=sagittal)
        slice_idx: Which slice to show (if None, use middle)
    """
    if slice_idx is None:
        slice_idx = volume.shape[slice_axis] // 2
    
    # Select the slice
    if slice_axis == 0:
        vol_slice = volume[slice_idx, :, :]
        prob_slice = prob_map[slice_idx, :, :]
        axis_name = "Axial"
    elif slice_axis == 1:
        vol_slice = volume[:, slice_idx, :]
        prob_slice = prob_map[:, slice_idx, :]
        axis_name = "Coronal"
    else:
        vol_slice = volume[:, :, slice_idx]
        prob_slice = prob_map[:, :, slice_idx]
        axis_name = "Sagittal"
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Show CT
    ax.imshow(vol_slice, cmap='gray')
    
    # Overlay probability map with transparency
    overlay = ax.imshow(prob_slice, cmap='hot', alpha=0.6, vmin=0, vmax=1)
    
    ax.set_title(f'{model_name} - {axis_name} Slice {slice_idx}\n3D Overlay', 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.colorbar(overlay, ax=ax, label='Segmentation Probability', fraction=0.046)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Info] Saved 3D overlay to {output_path}")


def plot_metric_comparison(
    histories: Dict[str, Dict],
    metric: str,
    output_path: Path
):
    """Plot a single metric comparison across all models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    for idx, (model_name, history) in enumerate(histories.items()):
        train_key = f"train_{metric}"
        val_key = f"val_{metric}"
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], 
                   label=f"{model_name} (train)", 
                   color=colors[idx], 
                   linestyle='-', 
                   linewidth=2)
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], 
                   label=f"{model_name} (val)", 
                   color=colors[idx], 
                   linestyle='--', 
                   linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_all_metrics_comparison(
    histories: Dict[str, Dict],
    output_dir: Path
):
    """Generate comparison plots for all available metrics."""
    # Find all available metrics
    all_metrics = set()
    for history in histories.values():
        for key in history.keys():
            if key.startswith("train_"):
                metric = key[len("train_"):]
                all_metrics.add(metric)
    
    print(f"[Info] Generating comparison plots for {len(all_metrics)} metrics...")
    
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    for metric in sorted(all_metrics):
        output_path = metrics_dir / f"{metric}_comparison.png"
        plot_metric_comparison(histories, metric, output_path)
        print(f"  Generated {metric}_comparison.png")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    print("=" * 70)
    print("Multi-Model Comparison for Aneurysm Detection")
    print("=" * 70)
    
    # Load all models
    print("\n[Step 1] Loading models...")
    models = []
    for model_path in MODEL_CONFIGS:
        try:
            model_info = load_trained_model(Path(model_path))
            models.append(model_info)
            print(f"  ✓ Loaded {model_info.name}")
            print(f"    - Segmentation: {model_info.has_segmentation}")
            print(f"    - Classification: {model_info.has_classification}")
            print(f"    - Regression: {model_info.has_regression}")
        except Exception as e:
            print(f"  ✗ Failed to load {model_path}: {e}")
    
    if not models:
        print("[Error] No models loaded successfully!")
        return
    
    # Select test patient
    print("\n[Step 2] Selecting test patient...")
    patient_id = select_test_patient(
        Path(INFERENCE_CONFIG["test_csv"]),
        INFERENCE_CONFIG.get("patient_id")
    )
    
    # Load patient volume
    print("\n[Step 3] Loading patient volume...")
    h5_path = Path(INFERENCE_CONFIG["test_h5_path"])
    with h5py.File(h5_path, "r") as f:
        volume = f["series"][patient_id]["vol"][:]
    volume = np.nan_to_num(np.asarray(volume, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Volume shape: {volume.shape}")
    
    # Load ground truth
    print("\n[Step 4] Loading ground truth localizers...")
    localizers_path = Path(INFERENCE_CONFIG["test_localizers_csv"])
    aneurysm_points = load_localizer_points(patient_id, localizers_path)
    print(f"  Found {len(aneurysm_points)} aneurysm location(s)")
    
    gt_heatmap = build_gaussian_gt(
        volume.shape, 
        aneurysm_points, 
        sigma=INFERENCE_CONFIG["gauss_sigma"]
    )
    
    aneurysm_slice = find_aneurysm_slice(aneurysm_points)
    print(f"  Central aneurysm slice: {aneurysm_slice}")
    
    # Run inference for all models
    print("\n[Step 5] Running inference for all models...")
    model_results = {}
    for model_info in models:
        if not model_info.has_segmentation:
            print(f"  Skipping {model_info.name} (no segmentation head)")
            continue
        
        try:
            results = run_inference_for_model(model_info, volume, INFERENCE_CONFIG)
            model_results[model_info.name] = results
            print(f"  ✓ Completed {model_info.name}")
            
            # Save results
            output_dir = Path(INFERENCE_CONFIG["output_dir"])
            results_dir = output_dir / "inference_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(results_dir / f"{model_info.name}_prob.npy", results["prob"])
            np.save(results_dir / f"{model_info.name}_mask.npy", results["mask"])
            
        except Exception as e:
            print(f"  ✗ Failed inference for {model_info.name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate visualizations
    print("\n[Step 6] Generating visualizations...")
    output_dir = Path(INFERENCE_CONFIG["output_dir"])
    
    # Segmentation comparison at aneurysm slice
    if model_results:
        plot_segmentation_comparison(
            volume,
            aneurysm_slice,
            gt_heatmap,
            model_results,
            output_dir / f"segmentation_comparison_{patient_id[:20]}.png",
            aneurysm_points
        )
        
        # Create 3D overlays for each model
        overlay_dir = output_dir / "3d_overlays"
        for model_name, results in model_results.items():
            create_3d_overlay(
                volume,
                results["prob"],
                overlay_dir / f"{model_name}_overlay_axial.png",
                model_name,
                slice_axis=0,
                slice_idx=aneurysm_slice
            )
    
    # Metrics comparison
    print("\n[Step 7] Generating metrics comparison plots...")
    histories = {model.name: model.history for model in models if model.history}
    if histories:
        plot_all_metrics_comparison(histories, output_dir)
    else:
        print("  No histories available for comparison")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Patient ID: {patient_id}")
    print(f"Models compared: {len(models)}")
    print(f"Models with segmentation: {len(model_results)}")
    print(f"Aneurysm locations: {len(aneurysm_points)}")
    print(f"Output directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print(f"  - Segmentation comparison: segmentation_comparison_{patient_id[:20]}.png")
    print(f"  - 3D overlays: 3d_overlays/")
    print(f"  - Metrics plots: metrics/")
    print(f"  - Inference results: inference_results/")
    print("=" * 70)


if __name__ == "__main__":
    main()
