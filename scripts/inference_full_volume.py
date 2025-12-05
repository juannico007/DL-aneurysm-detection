"""
Run full-volume inference on a specific series to verify model calibration and localization.
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Iterable

import h5py
import numpy as np
import torch
import torch.nn.functional as F

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet

def sliding_window(volume: np.ndarray, patch_size: int, stride: int) -> Iterable[Tuple[np.ndarray, Tuple[int, int, int]]]:
    """Generate patches from volume with sliding window."""
    D, H, W = volume.shape
    z_starts = list(range(0, max(D - patch_size, 0) + 1, stride))
    y_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
    x_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
    
    # Ensure we cover the last part
    if z_starts[-1] != D - patch_size: z_starts.append(D - patch_size)
    if y_starts[-1] != H - patch_size: y_starts.append(H - patch_size)
    if x_starts[-1] != W - patch_size: x_starts.append(W - patch_size)
    
    for z in z_starts:
        for y in y_starts:
            for x in x_starts:
                patch = volume[z : z + patch_size, y : y + patch_size, x : x + patch_size]
                yield patch, (z, y, x)

def build_weight_mask(patch_size: int, mode: str = "hann") -> np.ndarray:
    """Build a weight mask for blending patches."""
    if mode == "hann":
        h = np.hanning(patch_size)
        return (h[:, None, None] * h[None, :, None] * h[None, None, :]).astype(np.float32)
    return np.ones((patch_size, patch_size, patch_size), dtype=np.float32)

def load_model(checkpoint_path: Path, device: str) -> UNet:
    """Load the model from checkpoint."""
    # Try to find hyperparameters
    hp_path = checkpoint_path.parent / "hyperparameters.json"
    unet_kwargs = {}
    if hp_path.exists():
        with hp_path.open() as f:
            hp_data = json.load(f)
            if "unet" in hp_data:
                unet_kwargs.update(hp_data["unet"])
    
    model = UNet(**unet_kwargs)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model

def run_inference(
    h5_path: Path,
    series_id: str,
    model: UNet,
    device: str,
    patch_size: int = 64,
    stride: int = 32,
    output_path: Optional[Path] = None
):
    print(f"[Info] Loading volume for {series_id}...")
    with h5py.File(h5_path, "r") as f:
        if series_id not in f["series"]:
            raise ValueError(f"Series {series_id} not found in {h5_path}")
        vol = f["series"][series_id]["vol"][:]
    
    # Normalize if needed (assuming pre-normalized in HDF5 as per training code)
    vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0)
    
    print(f"[Info] Volume shape: {vol.shape}")
    
    prob_map = np.zeros_like(vol, dtype=np.float32)
    weight_map = np.zeros_like(vol, dtype=np.float32)
    weight_patch = build_weight_mask(patch_size, mode="hann")
    
    print("[Info] Running sliding window inference...")
    with torch.no_grad():
        for patch_np, (z, y, x) in sliding_window(vol, patch_size, stride):
            patch_t = torch.from_numpy(patch_np).unsqueeze(0).unsqueeze(0).to(device)
            
            # Inference
            outputs = model(patch_t)
            seg_logits = outputs[0]
            seg_probs = torch.sigmoid(seg_logits).cpu().numpy()[0, 0]
            
            prob_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += seg_probs * weight_patch
            weight_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += weight_patch
            
    # Normalize by weights
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    
    # Analysis
    max_prob = prob_map.max()
    max_idx = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    mean_prob = prob_map.mean()
    
    print("\n" + "="*40)
    print(f"RESULTS for {series_id}")
    print("="*40)
    print(f"Global Max Probability: {max_prob:.6f}")
    print(f"Location of Max (z,y,x): {max_idx}")
    print(f"Mean Probability:       {mean_prob:.6f}")
    
    # Histogram analysis
    print("\nProbability Distribution:")
    hist, bins = np.histogram(prob_map, bins=[0, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 1.0])
    for i in range(len(hist)):
        print(f"  [{bins[i]:.2f}, {bins[i+1]:.2f}): {hist[i]} voxels")
        
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Save as NPZ
        np.savez_compressed(output_path, prob=prob_map, vol=vol)
        print(f"\n[Info] Saved results to {output_path}")
        
        # Try saving NIfTI
        try:
            import nibabel as nib
            affine = np.eye(4)
            nib.save(nib.Nifti1Image(prob_map, affine), str(output_path).replace(".npz", ".nii.gz"))
            print(f"[Info] Saved NIfTI to {str(output_path).replace('.npz', '.nii.gz')}")
        except ImportError:
            pass

def load_localizers(csv_path: Path, series_id: str) -> list:
    """Load ground truth coordinates for a series."""
    import csv
    coords = []
    if not csv_path.exists():
        print(f"[Warn] Localizer CSV not found: {csv_path}")
        return coords
        
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SeriesInstanceUID") == series_id:
                try:
                    # Note: CSV has x_new, y_new, z_new
                    x = float(row["x_new"])
                    y = float(row["y_new"])
                    z = float(row["z_new"])
                    coords.append((z, y, x))
                except (ValueError, KeyError):
                    continue
    return coords

CONFIG = {
    "h5_path": "train_dataset.h5",
    "series_id": "1.2.826.0.1.3680043.8.498.10030095840917973694487307992374923817",
    "model_path": "cloud_models/MihaiB-dev/new-loss-2_curriculum-learning-alpha-5/MihaiB-dev_new-loss-2_curriculum-learning-alpha-5.pt",
    "output": "cloud_models/MihaiB-dev/new-loss-2_curriculum-learning-alpha-5/inference_test.npz",
    "patch_size": 64,
    "stride": 32,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "localizers": "train_localizers.csv"
}

def main():
    # Use CONFIG dictionary
    args = CONFIG
    
    # Load GT
    gt_coords = load_localizers(Path(args["localizers"]), args["series_id"])
    
    model = load_model(Path(args["model_path"]), args["device"])
    run_inference(
        Path(args["h5_path"]),
        args["series_id"],
        model,
        args["device"],
        args["patch_size"],
        args["stride"],
        Path(args["output"])
    )
    
    if gt_coords:
        print("\nGround Truth Targets (z, y, x):")
        for i, (z, y, x) in enumerate(gt_coords):
            print(f"  Target {i+1}: ({z:.2f}, {y:.2f}, {x:.2f})")
    else:
        print("\n[Info] No ground truth found in localizers CSV.")

if __name__ == "__main__":
    main()
