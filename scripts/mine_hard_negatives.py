"""
Mine hard negatives from training data.
Runs inference on full volumes, finds false positives (high probability, far from GT),
and appends them to train_patches.csv as negative samples.
"""

import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model.unet import UNet

def load_model(checkpoint_path: Path, device: str) -> UNet:
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

def sliding_window_inference(
    vol: np.ndarray,
    model: UNet,
    device: str,
    patch_size: int = 64,
    stride: int = 32
) -> np.ndarray:
    """Run sliding window inference and return probability map."""
    D, H, W = vol.shape
    prob_map = np.zeros_like(vol, dtype=np.float32)
    weight_map = np.zeros_like(vol, dtype=np.float32)
    
    # Hann window
    h = np.hanning(patch_size)
    weight_patch = (h[:, None, None] * h[None, :, None] * h[None, None, :]).astype(np.float32)
    
    z_starts = list(range(0, max(D - patch_size, 0) + 1, stride))
    y_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
    x_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
    
    if z_starts[-1] != D - patch_size: z_starts.append(D - patch_size)
    if y_starts[-1] != H - patch_size: y_starts.append(H - patch_size)
    if x_starts[-1] != W - patch_size: x_starts.append(W - patch_size)
    
    with torch.no_grad():
        for z in z_starts:
            for y in y_starts:
                for x in x_starts:
                    patch = vol[z : z + patch_size, y : y + patch_size, x : x + patch_size]
                    patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
                    
                    outputs = model(patch_t)
                    seg_probs = torch.sigmoid(outputs[0]).cpu().numpy()[0, 0]
                    
                    prob_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += seg_probs * weight_patch
                    weight_map[z:z+patch_size, y:y+patch_size, x:x+patch_size] += weight_patch
                    
    weight_map[weight_map == 0] = 1.0
    prob_map /= weight_map
    return prob_map

def load_gt_coords(localizer_csv: Path, series_id: str) -> List[Tuple[float, float, float]]:
    coords = []
    if not localizer_csv.exists():
        return coords
    with localizer_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("SeriesInstanceUID") == series_id:
                try:
                    coords.append((
                        float(row["z_new"]),
                        float(row["y_new"]),
                        float(row["x_new"])
                    ))
                except ValueError:
                    continue
    return coords

def is_safe_from_gt(
    patch_center: Tuple[int, int, int],
    half_size: int,
    gt_coords: List[Tuple[float, float, float]],
    min_dist: float
) -> bool:
    """
    Check if the patch defined by center and half_size is at least min_dist 
    away from all GT points (box-to-point distance).
    """
    z, y, x = patch_center
    
    # Patch Box
    z_min, z_max = z - half_size, z + half_size
    y_min, y_max = y - half_size, y + half_size
    x_min, x_max = x - half_size, x + half_size
    
    for gz, gy, gx in gt_coords:
        # Compute squared distance from point to box
        # If point is inside box, dist is 0.
        dz = max(z_min - gz, 0, gz - z_max)
        dy = max(y_min - gy, 0, gy - y_max)
        dx = max(x_min - gx, 0, gx - x_max)
        
        sq_dist = dz*dz + dy*dy + dx*dx
        dist = np.sqrt(sq_dist)
        
        if dist < min_dist:
            return False
    return True

CONFIG = {
    "h5_path": "train_dataset.h5",
    "patches_csv": "train_patches.csv",
    "localizers_csv": "train_localizers.csv",
    "patients_csv": "train.csv",
    "model_path": "cloud_models/MihaiB-dev/new-loss-2_curriculum-learning-alpha-5/MihaiB-dev_new-loss-2_curriculum-learning-alpha-5.pt",
    "series_ids": ["all"], # "all" = all patients in patients_csv
    "threshold": 0.8,
    "patch_size": 64,
    "min_dist": 33.0, # Increased to > patch_size/2 to ensure aneurysm is outside
    "jitter": 10,     # Random offset for center
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def main():
    args = CONFIG
    patch_size = args["patch_size"]
    half_size = patch_size // 2
    
    # Load existing patches to avoid duplicates (approximate check)
    existing_centers = set()
    patches_path = Path(args["patches_csv"])
    if patches_path.exists():
        with patches_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    existing_centers.add((
                        row["series_id"],
                        int(float(row["center_z"])),
                        int(float(row["center_y"])),
                        int(float(row["center_x"]))
                    ))
                except ValueError:
                    continue
    
    model = load_model(Path(args["model_path"]), args["device"])
    
    series_ids = args["series_ids"]
    
    # Handle "all" mode (Unified Positive + Healthy)
    if not series_ids or series_ids == ["all"]:
        import pandas as pd
        if Path(args["patients_csv"]).exists():
            df = pd.read_csv(args["patients_csv"])
            series_ids = df["SeriesInstanceUID"].tolist()
            print(f"[Info] Mining ALL {len(series_ids)} patients from {args['patients_csv']}.")
        else:
            # Fallback to H5 keys if csv missing
            with h5py.File(args["h5_path"], "r") as f:
                series_ids = list(f["series"].keys())
            print(f"[Info] Mining ALL {len(series_ids)} series from H5 (CSV not found).")
            
    new_rows = []
    
    for sid in tqdm(series_ids, desc="Mining Series"):
        with h5py.File(args["h5_path"], "r") as f:
            if sid not in f["series"]:
                continue
            vol = f["series"][sid]["vol"][:]
            vol = np.nan_to_num(np.asarray(vol, dtype=np.float32), nan=0.0)
            
        gt_coords = load_gt_coords(Path(args["localizers_csv"]), sid)
        # gt_coords will be empty for healthy patients, which is handled correctly below.
        
        # Run inference
        prob_map = sliding_window_inference(vol, model, args["device"])
        
        # Find peaks > threshold
        max_prob = prob_map.max()
        if max_prob < args["threshold"]:
            continue
            
        peak_z, peak_y, peak_x = np.unravel_index(np.argmax(prob_map), prob_map.shape)
        
        # Apply Jitter to center with retries
        # We want the peak to be within the patch, but not necessarily at the center.
        # Center = Peak + Offset
        # Offset should be within [-jitter, +jitter]
        jitter = args["jitter"]
        max_attempts = 10
        
        valid_center = None
        
        for _ in range(max_attempts):
            off_z = np.random.randint(-jitter, jitter + 1)
            off_y = np.random.randint(-jitter, jitter + 1)
            off_x = np.random.randint(-jitter, jitter + 1)
            
            z = peak_z + off_z
            y = peak_y + off_y
            x = peak_x + off_x
            
            # Ensure patch is within bounds
            D, H, W = vol.shape
            z = max(half_size, min(D - half_size, z))
            y = max(half_size, min(H - half_size, y))
            x = max(half_size, min(W - half_size, x))
            
            # Check distance to GT using the helper function
            # This ensures the aneurysm is far enough from the patch BORDER (min_dist)
            if is_safe_from_gt((z, y, x), half_size, gt_coords, args["min_dist"]):
                valid_center = (z, y, x)
                break
        
        if valid_center:
            z, y, x = valid_center
            # Check if we already have a patch near here
            is_duplicate = False
            for esid, ez, ey, ex in existing_centers:
                if esid == sid and abs(ez - z) < 10 and abs(ey - y) < 10 and abs(ex - x) < 10:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                print(f"[Found FP] Series {sid}: Prob {max_prob:.4f} at Peak({peak_z}, {peak_y}, {peak_x}) -> Center({z}, {y}, {x})")
                row_data = [
                    sid,
                    f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", # center x,y,z
                    "", "", "", # aneurysm global
                    "", "", "", # aneurysm rel
                    "0", # label
                    "hard_negative" # location
                ]
                
                # Write immediately to file
                with patches_path.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row_data)
                    
                existing_centers.add((sid, z, y, x))

if __name__ == "__main__":
    main()
